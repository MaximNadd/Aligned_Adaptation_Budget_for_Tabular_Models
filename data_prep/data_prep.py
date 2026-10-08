"""Prepare ALL datasets once, then the whole pipeline runs offline.

Usage:
    python3 data_prep/data_prep.py --config configs/grid.yaml         # real prep (network)
    python3 data_prep/data_prep.py --synthetic --data-dir data/prepared_smoke
After success a READY marker is written; run stages refuse to start without it.
"""
import argparse
import hashlib
import json
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

DROPS = {}   # dataset_key -> reason string; written into registry.json


def _drop(reg, key, exc):
    reg[key] = None
    DROPS[key] = f"{type(exc).__name__}: {exc}"
    print(f"[prep] DROP {key}: {DROPS[key]}", flush=True)
    traceback.print_exc(limit=4)


def _patch_xport_pandas2():
    """xport 3.6.1 (needed: 3.2.1 crashes on underscore-leading SAS names like
    BRFSS '_STATE' via namedtuple) pins pandas<1.4 and passes the removed
    positional `fastpath` to Series.__init__ on pandas>=2. Re-implement
    Variable.__init__ faithfully without it. Verified by an underscore-name
    XPT roundtrip on pandas 2.x/3.x."""
    try:
        import functools
        import pandas as pd
        import xport
    except ImportError:
        return
    orig = xport.Variable.__init__

    @functools.wraps(orig)
    def _init(self, data=None, index=None, dtype=None, name=None, copy=False,
              fastpath=False, label=None, vtype=None, width=None, format=None,
              informat=None, **kwds):
        metadata = {"label": label, "vtype": vtype, "width": width,
                    "format": format, "informat": informat}
        pd.Series.__init__(self, data, index, dtype, name, copy, **kwds)
        for n, v in metadata.items():
            if v is not None:
                setattr(self, n, v)
        self.copy_metadata(data)
        for n, v in metadata.items():
            setattr(self, n, getattr(self, n, v))

    xport.Variable.__init__ = _init


# --------------------------------------------------------------------------- #
def _standardise(df, y):
    """DataFrame + label Series -> float32 X, int64 y. Ordinal-encode object /
    category columns, median-impute numerics, NaN category -> -1."""
    import pandas as pd
    X = df.copy()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "category":
            X[c] = pd.factorize(X[c], use_na_sentinel=True)[0]
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
            if X[c].isna().any():
                X[c] = X[c].fillna(X[c].median())
    y = pd.factorize(pd.Series(y))[0]
    return X.to_numpy(dtype=np.float32), y.astype(np.int64)


def _save(data_dir: Path, key: str, suite: str, source: str,
          Xtr, ytr, Xid, yid, Xval, yval,
          train_idx, val_idx, train_digest, val_digest,
          Xood=None, yood=None, note=""):
    arrays = {
        "X_train": Xtr, "y_train": ytr,
        "X_val": Xval, "y_val": yval,
        "train_idx": train_idx,     # global indices into the ORIGINAL dataset
        "val_idx": val_idx,         # global indices into the ORIGINAL dataset
        "X_id": Xid, "y_id": yid,
    }
    if Xood is not None:
        arrays.update({"X_ood": Xood, "y_ood": yood})
    
    np.savez_compressed(data_dir / f"{key}.npz", **arrays)
    
    return {
        "suite": suite, "source": source, "has_ood": Xood is not None,
        "n_train": int(len(ytr)), "n_val": int(len(yval)),
        "n_id": int(len(yid)),
        "n_ood": int(len(yood)) if yood is not None else 0,
        "n_features": int(Xtr.shape[1]),
        "n_classes": int(len(np.unique(ytr))),
        "train_digest": train_digest,
        "val_digest": val_digest,
        "note": note,
    }


def _split_train_val(X, y, val_ratio=0.2, seed=42):
    n = len(y)
    n_val = int(n * val_ratio)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    
    # Sort both for deterministic digests
    val_idx.sort()
    train_idx.sort()
    val_digest = hashlib.sha256(val_idx.tobytes()).hexdigest()
    train_digest = hashlib.sha256(train_idx.tobytes()).hexdigest()
    
    return (X[train_idx], y[train_idx],
            X[val_idx],   y[val_idx],
            train_idx, val_idx, train_digest, val_digest)

# --------------------------------------------------------------------------- #
def prep_synthetic(data_dir: Path):
    """Two tiny fake datasets (one with an OOD split) for the smoke chain."""
    rng = np.random.default_rng(0)
    reg = {}
    for key, suite, ood in [("synth_shift", "tableshift", True),
                            ("synth_iid", "openml_cc18", False)]:
        n, d = 600, 8
        X = rng.normal(size=(n, d)).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, .5, n) > 0).astype(np.int64)
        
        # Phase 1: Split off fixed validation
        Xtr, ytr, Xval, yval, train_idx, val_idx, train_digest, val_digest = _split_train_val(X, y)
        Xid, yid = X[500:550], y[500:550] # adjust sizes as needed
        
        Xood = yood = None
        if ood:
            Xood = X[550:] + rng.normal(1.0, 0.5, size=X[550:].shape).astype(np.float32)
            yood = y[550:]
            
        reg[key] = _save(data_dir, key, suite, "synthetic", 
                         Xtr, ytr, Xid, yid, Xval, yval,
                        train_idx, val_idx, train_digest, val_digest,
                         Xood, yood, note="smoke-only")
    return reg


# --------------------------------------------------------------------------- #

def _shim_ray():
    """tableshift imports ray.data at module import time for its distributed
    path; the plain get_dataset/get_pandas route does not need it. Real ray is
    undesirable on this box (heavy, and its compiled deps may not run on the
    no-AVX vCPU), so install a stub unless real ray is present. If tableshift
    actually calls into ray for some task, the stub raises and that task is
    dropped with the reason recorded in the registry."""
    import importlib.util, sys, types
    if importlib.util.find_spec("ray") is not None:
        return
    def _mod_getattr(attr):
        # Dunders MUST raise AttributeError: real torch iterates sys.modules via
        # inspect at import time and asks every module for __file__ etc.; a stub
        # that answers with junk breaks that walk.
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        raise RuntimeError(f"tableshift called ray.{attr} — this task needs real ray")
    ray = types.ModuleType("ray"); data = types.ModuleType("ray.data")
    ray.__getattr__ = _mod_getattr
    data.__getattr__ = _mod_getattr
    ray.data = data
    sys.modules["ray"], sys.modules["ray.data"] = ray, data


def prep_tableshift(data_dir: Path, names, cache_dir: Path):
    reg = {}
    _shim_ray()
    _patch_xport_pandas2()
    try:
        from tableshift import get_dataset
    except ImportError as e:
        print(f"[prep] tableshift unavailable ({e}); dropping suite", flush=True)
        return {n: None for n in names}
    
    for name in names:
        key = f"ts_{name}"
        try:
            dset = get_dataset(name, cache_dir=str(cache_dir))
            Xtr_full, ytr_full, _, _ = dset.get_pandas("train")
            Xid, yid, _, _ = dset.get_pandas("id_test")
            Xood, yood, _, _ = dset.get_pandas("ood_test")
            
            Xtr_full, ytr_full = _standardise(Xtr_full, ytr_full)
            Xid, yid = _standardise(Xid, yid)
            Xood, yood = _standardise(Xood, yood)
            
            # Phase 1: Split off fixed validation
            Xtr, ytr, Xval, yval, val_idx, val_digest = _split_train_val(Xtr_full, ytr_full)
            
            reg[key] = _save(data_dir, key, "tableshift", f"tableshift:{name}",
                             Xtr, ytr, Xid, yid, Xval, yval, val_idx, val_digest,
                             Xood, yood)
            print(f"[prep] {key} ok", flush=True)
        except Exception as e:
            _drop(reg, key, e)
    return reg


PUMS_URL = ("https://www2.census.gov/programs-surveys/acs/data/pums/"
            "{year}/{horizon}/csv_p{st}.zip")


def _ensure_pums(cache_dir: Path, year: str, horizon: str, states):
    """folktables (load_acs.py, verified 0.0.12) skips its downloader ONLY when
    the EXTRACTED psam_p{fips}.csv already exists; otherwise it re-downloads
    with requests.get and no user-agent — census.gov answers that with an HTML
    page, which is the whole ACS failure. So: fetch the zip with a browser
    user-agent, verify the zip magic, AND extract the per-state person CSV to
    exactly the path folktables checks."""
    import urllib.request
    import zipfile
    try:
        from folktables.load_acs import _STATE_CODES
    except ImportError:
        _STATE_CODES = {"CA": "06", "SD": "46", "PR": "72"}
    d = cache_dir / year / horizon
    d.mkdir(parents=True, exist_ok=True)
    for st in states:
        csv_name = f"psam_p{_STATE_CODES[st.upper()]}.csv"
        if (d / csv_name).exists():
            continue
        p = d / f"csv_p{st.lower()}.zip"
        if not (p.exists() and p.open("rb").read(2) == b"PK"):
            if p.exists():
                p.unlink()                   # corrupted leftover (HTML page)
            url = PUMS_URL.format(year=year, horizon=horizon, st=st.lower())
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
            print(f"[prep] fetching PUMS {st.upper()} from census.gov ...",
                  flush=True)
            with urllib.request.urlopen(req, timeout=600) as r, p.open("wb") as f:
                shutil.copyfileobj(r, f)
            if p.open("rb").read(2) != b"PK":
                p.unlink()
                raise RuntimeError(f"census.gov returned non-zip content for {st}"
                                   " — download PUMS manually, see RUN_ORDER")
        print(f"[prep] extracting {csv_name} for {st.upper()} ...", flush=True)
        with zipfile.ZipFile(p) as z:
            z.extract(csv_name, path=d)


ACS_TASKS = {  # WhyShift-style spatial shift reproduced via folktables (public ACS PUMS)
    "acs_income_ca2sd":  ("ACSIncome", "CA", "SD"),
    "acs_income_ca2pr":  ("ACSIncome", "CA", "PR"),
    "acs_pubcov_ca2sd":  ("ACSPublicCoverage", "CA", "SD"),
    "acs_mobility_ca2sd": ("ACSMobility", "CA", "SD"),
}


def prep_whyshift(data_dir: Path, keys, cache_dir: Path):
    reg = {}
    try:
        import folktables
        from folktables import ACSDataSource
    except ImportError as e:
        print(f"[prep] folktables unavailable ({e}); dropping suite", flush=True)
        return {k: None for k in keys}
    
    src = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person",
                        root_dir=str(cache_dir))
    needed_states = sorted({s for k in keys for s in ACS_TASKS[k][1:]})
    try:
        _ensure_pums(cache_dir, "2018", "1-Year", needed_states)
    except Exception as e:
        print(f"[prep] PUMS prefetch problem ({e}) — folktables may still recover per state", flush=True)
        
    for key in keys:
        task_name, st_src, st_tgt = ACS_TASKS[key]
        task = getattr(folktables, task_name)
        try:
            Xs, ys, _ = task.df_to_numpy(src.get_data(states=[st_src], download=True))
            Xt, yt, _ = task.df_to_numpy(src.get_data(states=[st_tgt], download=True))
            
            Xs, ys = _standardise(Xs, ys)
            Xt, yt = _standardise(Xt, yt)
            
            # Split source into train/val/id_test
            rng = np.random.default_rng(42)
            idx = rng.permutation(len(ys))
            n_id = max(1000, int(0.2 * len(ys)))
            tr_idx, te_idx = idx[n_id:], idx[:n_id]
            
            Xtr_full, ytr_full = Xs[tr_idx], ys[tr_idx]
            Xid, yid = Xs[te_idx], ys[te_idx]
            
            # Phase 1: Split off fixed validation
            Xtr, ytr, Xval, yval, val_idx, val_digest = _split_train_val(Xtr_full, ytr_full)
            
            reg[key] = _save(data_dir, key, "whyshift", f"folktables:{task_name}",
                             Xtr, ytr, Xid, yid, Xval, yval, val_idx, val_digest,
                             Xt, yt, note=f"{st_src}->{st_tgt}, ACS 2018 1-Year")
            print(f"[prep] {key} ok", flush=True)
        except Exception as e:
            _drop(reg, key, e)
    return reg


def prep_cc18(data_dir: Path, keys, cache_dir: Path):
    reg = {}
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    
    for key in keys:
        did = int(key.split("_")[1])
        try:
            bunch = fetch_openml(data_id=did, as_frame=True,
                                 data_home=str(cache_dir), parser="auto")
            X, y = _standardise(bunch.data, bunch.target)
            
            # First split: train+val vs id_test
            X_tv, Xid, y_tv, yid = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y)
            
            # Phase 1: Split off fixed validation from train+val
            Xtr, ytr, Xval, yval, val_idx, val_digest = _split_train_val(X_tv, y_tv)
            
            reg[key] = _save(data_dir, key, "openml_cc18", f"openml:{did}",
                             Xtr, ytr, Xid, yid, Xval, yval, val_idx, val_digest)
            print(f"[prep] {key} ok ({bunch.details.get('name', '')})", flush=True)
        except Exception as e:
            _drop(reg, key, e)
    return reg

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/grid.yaml")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--cache-dir", default="data/raw_cache")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(a.config).read_text())
    data_dir = Path(a.data_dir or cfg["runtime"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(a.cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)

    if a.synthetic:
        registry = prep_synthetic(data_dir)
    else:
        d = cfg["datasets"]
        registry = {}
        registry.update(prep_tableshift(data_dir, d["tableshift"], cache_dir))
        registry.update(prep_whyshift(data_dir, d["whyshift"], cache_dir))
        registry.update(prep_cc18(data_dir, d["openml_cc18"], cache_dir))

    realised = {k: v for k, v in registry.items() if v is not None}
    (data_dir / "registry.json").write_text(json.dumps(
        {"datasets": realised, "dropped": DROPS}, indent=2))
    print(f"[prep] realised={len(realised)} dropped={len(DROPS)}")
    for k, why in sorted(DROPS.items()):
        print(f"[prep]   dropped {k}: {why}")
    if not realised:
        sys.exit("[prep] nothing prepared — do not proceed")
    (data_dir / "READY").write_text("ok\n")
    print(f"[prep] READY written to {data_dir} — pipeline can now run offline")

if __name__ == "__main__":
    main()
