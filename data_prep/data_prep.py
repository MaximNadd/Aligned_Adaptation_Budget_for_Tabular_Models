"""Prepare ALL datasets once, then the whole pipeline runs offline.

Usage:
    python3 data_prep/data_prep.py --config configs/grid.yaml         # real prep (network)
    python3 data_prep/data_prep.py --synthetic --data-dir data/prepared_smoke
After success a READY marker is written; run stages refuse to start without it.
"""
import argparse
import hashlib
import json
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


def _save(data_dir: Path, key: str, suite: str, source: str, Xtr, ytr, Xval, yval, train_idx, val_idx, train_digest, val_digest, note=""):
    arrays = {
        "X_train": Xtr, "y_train": ytr,
        "X_val": Xval, "y_val": yval,
        "train_idx": train_idx,     # global indices into the ORIGINAL dataset
        "val_idx": val_idx,         # global indices into the ORIGINAL dataset
    }
    
    np.savez_compressed(data_dir / f"{key}.npz", **arrays)
    
    return {
        "suite": suite, "source": source,
        "n_train": int(len(ytr)), "n_val": int(len(yval)),
        "n_features": int(Xtr.shape[1]),
        "n_classes": int(len(np.unique(ytr))),
        "train_digest": train_digest,
        "val_digest": val_digest,
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
    
    return (X[train_idx], y[train_idx], X[val_idx], y[val_idx], train_idx, val_idx, train_digest, val_digest)

# --------------------------------------------------------------------------- #
def prep_synthetic(data_dir: Path, max_volume: int = 500, n_features: int = 8, headroom: float = 2.0):
    """Synthetic datasets sized to accommodate the real volume grid.

    `max_volume` is the largest integer volume in the config (e.g. 300000).
    We need n_train >= max_volume after the 20% val split, and we want
    "full" to be *meaningfully* larger than max_volume so the two are
    distinct experiments — hence `headroom`.
    """
    # 80% of the pool becomes X_train after the val split.
    # Multiply by `headroom` so "full" > max_volume by a visible margin.
    n_train_pool = int(max_volume / 0.8 * headroom)

    print(f"[prep] synthetic: n_total={n_train_pool:,} ", flush=True)

    rng = np.random.default_rng(0)
    reg = {}
    for key, suite in [("synth_iid", "openml_cc18")]:
        X = rng.normal(size=(n_train_pool, n_features)).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, .5, n_train_pool) > 0).astype(np.int64)

        # Phase 1: split the pool into fixed train / fixed val
        (Xtr, ytr, Xval, yval, train_idx, val_idx, train_digest, val_digest) = _split_train_val(X[:n_train_pool], y[:n_train_pool])


        reg[key] = _save(data_dir, key, suite, "synthetic", Xtr, ytr, Xval, yval, train_idx, val_idx, train_digest, val_digest)
    return reg


# --------------------------------------------------------------------------- #

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
                        
            # Phase 1: Split off fixed validation from train+val
            Xtr, ytr, Xval, yval, train_idx, val_idx, train_digest, val_digest = _split_train_val(X, y)
            
            reg[key] = _save(data_dir, key, "openml_cc18", f"openml:{did}", Xtr, ytr, Xval, yval,train_idx, val_idx, train_digest, val_digest)
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

    vols = cfg.get("synthetic_volume_levels", cfg.get("volume_levels", []))
    int_vols = [v for v in vols if isinstance(v, int)]
    max_volume = max(int_vols)

    if a.synthetic:
        registry = prep_synthetic(data_dir, max_volume=max_volume)
    else:
        d = cfg["datasets"]
        registry = {}
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
