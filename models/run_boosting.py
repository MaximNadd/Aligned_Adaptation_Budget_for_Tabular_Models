"""
Phase 3: Boosting Branch.

For each (dataset, volume, seed, model):
  1. Fit with DEFAULT hyperparameters. Record metrics + costs.
  2. Run random search for max(budgets) trials. Track best-so-far on val.
  3. At each checkpoint (5, 10, 15, 30), evaluate the current best-so-far. Emit one row per checkpoint.
  4. Append rows to a parquet file for resume.

Usage:
    python src/run_boosting.py --config configs/grid.yaml
    python src/run_boosting.py --config configs/grid.yaml --synthetic
    python src/run_boosting.py --config configs/grid.yaml --models xgboost
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
import os
import threading
import time
import statistics
from contextlib import nullcontext

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss


try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

AVAILABLE_MODELS = []
if HAS_XGB: AVAILABLE_MODELS.append("xgboost")
if HAS_LGB: AVAILABLE_MODELS.append("lightgbm")
if HAS_CAT: AVAILABLE_MODELS.append("catboost")


# ============================================================
# GPU telemetry
# ============================================================

GPU_METRIC_KEYS = [
    "gpu_available",
    "gpu_name",
    "gpu_mem_total_mb",
    "gpu_util_mean_pct",
    "gpu_util_max_pct",
    "gpu_mem_used_mean_mb",
    "gpu_mem_used_max_mb",
    "gpu_mem_used_peak_pct",
    "gpu_temp_max_c",
    "gpu_power_mean_w",
    "gpu_power_max_w",
    "gpu_n_samples",
    "model_device",          # "gpu" / "cpu"
    "gpu_used_effective",    # bool: on GPU AND util_mean_pct >= 5
]


def _first_visible_gpu() -> int:
    """Pick which physical GPU index to monitor, honoring CUDA_VISIBLE_DEVICES."""
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if v:
        parts = [p.strip() for p in v.split(",") if p.strip()]
        if parts and parts[0].isdigit():
            return int(parts[0])
    return 0


class GPUMonitor:
    """
    Background NVML sampler with pause/resume/snapshot.

    Usage:
        mon = GPUMonitor(interval=0.05)
        mon.__enter__()
        mon.pause()
        try:
            mon.resume()
            ...fit...
            mon.pause()
            metrics = mon.snapshot()   # per-fit summary, clears samples
        finally:
            mon.__exit__(None, None, None)
    """

    def __init__(self, interval: float = 0.05, device_index: int | None = None):
        self.interval = interval
        self.device_index = _first_visible_gpu() if device_index is None else device_index
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None
        self._available = False
        self._device_name: str | None = None
        self._total_mem_mb: float | None = None

    # -- pause/resume/snapshot --
    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def snapshot(self) -> dict:
        """Summarize samples collected since last snapshot, then clear."""
        summ = self.summary()
        self.samples.clear()
        return summ

    # -- lifecycle --
    def _init(self) -> None:
        if not _HAS_PYNVML:
            return
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            name = pynvml.nvmlDeviceGetName(self._handle)
            self._device_name = name.decode() if isinstance(name, bytes) else name
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._total_mem_mb = mem.total / 1024**2
            self._available = True
        except Exception:
            self._available = False

    def _sample_once(self) -> None:
        try:
            h = self._handle
            util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None
            try:
                power_w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                power_w = None
            self.samples.append({
                "util_pct": float(util),
                "mem_used_mb": mem.used / 1024**2,
                "mem_total_mb": mem.total / 1024**2,
                "temp_c": None if temp is None else float(temp),
                "power_w": power_w,
            })
        except Exception:
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._paused.is_set():
                self._sample_once()
            self._stop.wait(self.interval)

    def __enter__(self) -> "GPUMonitor":
        self._init()
        if self._available:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if _HAS_PYNVML and self._available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    # -- summary --
    def summary(self) -> dict:
        base = {k: None for k in GPU_METRIC_KEYS}
        base.update({
            "gpu_available": bool(self._available),
            "gpu_name": self._device_name,
            "gpu_mem_total_mb": self._total_mem_mb,
            "gpu_n_samples": 0,
            "model_device": None,
            "gpu_used_effective": False,
        })
        if not self._available or not self.samples:
            return base

        utils = [s["util_pct"] for s in self.samples]
        mems = [s["mem_used_mb"] for s in self.samples]
        total = self.samples[0]["mem_total_mb"] or self._total_mem_mb or 0.0
        temps = [s["temp_c"] for s in self.samples if s["temp_c"] is not None]
        powers = [s["power_w"] for s in self.samples if s["power_w"] is not None]

        base.update({
            "gpu_util_mean_pct": float(statistics.fmean(utils)),
            "gpu_util_max_pct": float(max(utils)),
            "gpu_mem_used_mean_mb": float(statistics.fmean(mems)),
            "gpu_mem_used_max_mb": float(max(mems)),
            "gpu_mem_used_peak_pct": float(max(mems) / total * 100.0) if total else None,
            "gpu_temp_max_c": float(max(temps)) if temps else None,
            "gpu_power_mean_w": float(statistics.fmean(powers)) if powers else None,
            "gpu_power_max_w": float(max(powers)) if powers else None,
            "gpu_n_samples": len(self.samples),
        })
        return base


# ============================================================
# GPU parameter routing
# ============================================================

def gpu_params_for(model_name: str) -> tuple[dict, str]:
    """
    Returns (extra_model_params, device_str) where device_str is 'gpu' or 'cpu'.
    Uses the modern API for XGBoost >= 2.0 and the legacy one otherwise.
    """
    name = model_name.lower()

    if "xgboost" in name or name == "xgb":
        try:
            import xgboost as _xgb
            major = int(_xgb.__version__.split(".")[0])
        except Exception:
            major = 0
        if major >= 2:
            return {"device": "cuda", "tree_method": "hist"}, "gpu"
        return {"tree_method": "gpu_hist", "predictor": "gpu_predictor"}, "gpu"

    if "lightgbm" in name or name == "lgbm":
        return {
            "device_type": "gpu",
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
        }, "gpu"

    if "catboost" in name:
        return {"task_type": "GPU", "devices": "0"}, "gpu"

    return {}, "cpu"


def _tag_device(gpu_row: dict, model_device: str) -> dict:
    """Copy a raw GPUMonitor.summary() dict and attach device/effectiveness flags."""
    out = dict(gpu_row)
    out["model_device"] = model_device
    out["gpu_used_effective"] = bool(
        out.get("gpu_available")
        and model_device == "gpu"
        and (out.get("gpu_util_mean_pct") or 0.0) >= 5.0
    )
    return out


def _empty_gpu_row() -> dict:
    d = {k: None for k in GPU_METRIC_KEYS}
    d["model_device"] = None
    d["gpu_used_effective"] = False
    return d






# ============================================================================
# Hyperparameter search spaces and defaults
# ============================================================================
# Spec format: (kind, low, high)
#   "int" → randint(low, high+1)
#   "uni" → uniform(low, high)
#   "log" → 10 ** uniform(log10(low), log10(high))

SEARCH_SPACES = {
    "xgboost": {
        "n_estimators":     ("int", 100, 2000),
        "max_depth":        ("int", 3, 12),
        "learning_rate":    ("log", 1e-2, 3e-1),
        "subsample":        ("uni", 0.5, 1.0),
        "colsample_bytree": ("uni", 0.5, 1.0),
        "reg_alpha":        ("log", 1e-8, 1.0),
        "reg_lambda":       ("log", 1e-8, 1.0),
        "min_child_weight": ("int", 1, 20),
    },
    "lightgbm": {
        "n_estimators":      ("int", 100, 2000),
        "num_leaves":        ("int", 15, 255),
        "learning_rate":     ("log", 1e-2, 3e-1),
        "feature_fraction":  ("uni", 0.5, 1.0),
        "bagging_fraction":  ("uni", 0.5, 1.0),
        "lambda_l1":         ("log", 1e-8, 1.0),
        "lambda_l2":         ("log", 1e-8, 1.0),
        "min_child_samples": ("int", 5, 100),
    },
    "catboost": {
        "iterations":          ("int", 100, 2000),
        "depth":               ("int", 3, 10),
        "learning_rate":       ("log", 1e-2, 3e-1),
        "l2_leaf_reg":         ("log", 1e-2, 30.0),
        "bagging_temperature": ("uni", 0.0, 1.0),
        "border_count":        ("int", 32, 255),
    },
}

DEFAULT_PARAMS = {
    "xgboost": {
        "n_estimators": 300, "max_depth": 6, "learning_rate": 0.1,
        "subsample": 1.0, "colsample_bytree": 1.0,
        "reg_alpha": 0.0, "reg_lambda": 1.0, "min_child_weight": 1,
    },
    "lightgbm": {
        "n_estimators": 300, "num_leaves": 31, "learning_rate": 0.1,
        "feature_fraction": 1.0, "bagging_fraction": 1.0,
        "lambda_l1": 0.0, "lambda_l2": 0.0, "min_child_samples": 20,
    },
    "catboost": {
        "iterations": 300, "depth": 6, "learning_rate": 0.1,
        "l2_leaf_reg": 3.0, "bagging_temperature": 0.0,
        "border_count": 128,
    },
}


# ============================================================================
# Model construction and hyperparameter sampling
# ============================================================================
def build_model(model_name, params, seed):
    """Construct a model instance with the given params + seed."""
    p = dict(params)
    if model_name == "xgboost":
        p.update(random_state=seed, n_jobs=-1, eval_metric="logloss", tree_method="hist", verbosity=0)
        return xgb.XGBClassifier(**p)
    elif model_name == "lightgbm":
        p.update(random_state=seed, n_jobs=-1, verbose=-1)
        return lgb.LGBMClassifier(**p)
    elif model_name == "catboost":
        p.update(random_seed=seed, verbose=0, thread_count=-1)
        return CatBoostClassifier(**p)
    raise ValueError(f"Unknown model: {model_name}")


def sample_params(rng, space):
    """Sample one hyperparameter configuration from a search space."""
    out = {}
    for k, spec in space.items():
        kind = spec[0]
        if kind == "int":
            out[k] = int(rng.integers(spec[1], spec[2] + 1))
        elif kind == "uni":
            out[k] = float(rng.uniform(spec[1], spec[2]))
        elif kind == "log":
            out[k] = float(10 ** rng.uniform(np.log10(spec[1]), np.log10(spec[2])))
        else:
            raise ValueError(f"Unknown distribution kind: {kind}")
    return out


def derive_seed(*parts):
    """Derive a deterministic sub-seed from arbitrary string parts."""
    s = "_".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


# ============================================================================
# Evaluation
# ============================================================================
def make_evaluator(y_train):
    """Return an evaluate(model, X, y) -> (auc, logloss) for binary or multiclass."""
    n_classes = len(np.unique(y_train))

    if n_classes == 2:
        def evaluate(model, X, y):
            proba = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, proba)
            ll = log_loss(y, np.clip(proba, 1e-7, 1 - 1e-7))
            return float(auc), float(ll)
    else:
        def evaluate(model, X, y):
            proba = model.predict_proba(X)
            auc = roc_auc_score(y, proba, multi_class="ovr", average="macro")
            ll = log_loss(y, np.clip(proba, 1e-7, 1 - 1e-7))
            return float(auc), float(ll)

    return evaluate


# ============================================================================
# The main experiment runner
# ============================================================================
def run_one_experiment(
    model_name, dataset_key, volume, seed,
    X_train_full, y_train_full, local_idx,
    X_val, y_val,
    budgets, max_budget,
    gpu_interval: float = 0.05,
):
    """
    Run one (model, dataset, volume, seed) experiment.

    Runs 1 baseline fit + `max_budget` random-search trials, and emits:
      - one `phase="default"`, `budget=0` row
      - one `phase="tuned"` row per configured checkpoint in `budgets`
        (best-so-far score at that trial number, plus that fit's GPU metrics)
    """
    evaluate = make_evaluator(y_train_full)

    X_sub = X_train_full[local_idx]
    y_sub = y_train_full[local_idx]

    common = dict(
        dataset_key=dataset_key, model=model_name,
        volume_level=int(volume), seed=int(seed),
        n_train_rows=int(len(y_sub)),
        n_features=int(X_train_full.shape[1]),
    )

    budgets_set = {int(b) for b in budgets}
    if max_budget not in budgets_set:
        raise ValueError(
            f"max_budget={max_budget} not present in budgets={sorted(budgets_set)}"
        )

    gpu_params, model_device = gpu_params_for(model_name)

    rows = []
    mon = GPUMonitor(interval=gpu_interval)
    mon.__enter__()
    mon.pause()
    try:
        # ---------- Baseline ----------
        d_seed = derive_seed(model_name, dataset_key, volume, seed, "default")
        base_cfg = {**DEFAULT_PARAMS[model_name], **gpu_params}

        mon.resume()
        t0 = time.time()
        try:
            model = build_model(model_name, base_cfg, d_seed)
            model.fit(X_sub, y_sub)
            fit_s = time.time() - t0
        finally:
            mon.pause()
        base_gpu = _tag_device(mon.snapshot(), model_device)

        val_auc, val_ll = evaluate(model, X_val, y_val)
        rows.append({
            **common, **base_gpu,
            "phase": "default", "budget": 0,
            "val_auc": val_auc, "val_logloss": val_ll,
            "fit_time_s": fit_s,
            "config_json": json.dumps(DEFAULT_PARAMS[model_name], sort_keys=True),
        })

        # ---------- Random search with checkpoint emission ----------
        space = SEARCH_SPACES[model_name]
        best_val_auc = -np.inf
        best_val_ll  = float("nan")
        best_config  = None
        best_gpu     = _empty_gpu_row()
        cumulative_time = 0.0

        for trial in range(1, max_budget + 1):
            t_seed = derive_seed(model_name, dataset_key, volume, seed, "trial", trial)
            rng = np.random.default_rng(t_seed)
            config = sample_params(rng, space)
            trial_cfg = {**config, **gpu_params}

            # ----- fit (monitored) -----
            mon.resume()
            t0 = time.time()
            fit_s = None
            try:
                model = build_model(model_name, trial_cfg, t_seed)
                model.fit(X_sub, y_sub)
                fit_s = time.time() - t0
            except Exception as e:
                print(f"    [trial {trial}] fit failed: {type(e).__name__}: {e}")
            finally:
                mon.pause()
                trial_gpu = _tag_device(mon.snapshot(), model_device)

            # ----- eval -----
            val_auc = None
            val_ll  = float("nan")
            if fit_s is not None:
                try:
                    val_auc, val_ll = evaluate(model, X_val, y_val)
                    cumulative_time += fit_s
                except Exception as e:
                    print(f"    [trial {trial}] eval failed: {type(e).__name__}: {e}")

                # ----- update best-so-far -----
                if val_auc is not None and val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_val_ll  = val_ll
                    best_config  = config
                    best_gpu     = trial_gpu

            # ----- emit a checkpoint row at every configured budget -----
            if trial in budgets_set:
                auc = float(best_val_auc) if np.isfinite(best_val_auc) else float("nan")
                rows.append({
                    **common, **best_gpu,
                    "phase": "tuned", "budget": int(trial),
                    "val_auc": auc,
                    "val_logloss": float(best_val_ll),
                    "fit_time_s": cumulative_time,
                    "config_json": (
                        json.dumps(best_config, sort_keys=True)
                        if best_config is not None else "{}"
                    ),
                })

        return rows
    finally:
        mon.__exit__(None, None, None)


# ============================================================================
# Main loop with resume
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grid.yaml")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--gpu-monitor-interval", type=float, default=0.05,
                    help="seconds between NVML samples (default 0.05)")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())

    data_dir = Path(args.data_dir or cfg["runtime"]["data_dir"])
    if args.synthetic:
        data_dir = Path("data/prepared_smoke")

    # Guard: Phase 1 and Phase 2 must be complete
    if not (data_dir / "registry.json").exists():
        raise SystemExit(f"[phase3] registry.json not found in {data_dir}")
    if not (data_dir / "subsample_registry.json").exists():
        raise SystemExit(f"[phase3] subsample_registry.json not found in {data_dir}")

    output_path = Path(args.output or data_dir / "results_boosting.parquet")

    with open(data_dir / "registry.json") as f:
        reg = json.load(f)["datasets"]
    with open(data_dir / "subsample_registry.json") as f:
        sub_reg = json.load(f)

    datasets = args.datasets or list(reg.keys())
    model_names = args.models or AVAILABLE_MODELS
    model_names = [m for m in model_names if m in AVAILABLE_MODELS]
    if not model_names:
        raise SystemExit("[phase3] No boosting libraries available.")

    budgets = cfg.get("boosting_budgets", [5, 10, 15, 30])
    budgets = sorted({int(b) for b in budgets})
    max_budget = max(budgets)
    if max_budget not in budgets:
        budgets.append(max_budget)
        budgets = sorted(set(budgets))

    print(f"[phase3] Models: {model_names}")
    print(f"[phase3] Budgets: {budgets} (max={max_budget})")
    print(f"[phase3] Datasets: {datasets}")
    print(f"[phase3] GPU monitor: interval={args.gpu_monitor_interval}s "
          f"device_index={_first_visible_gpu()} "
          f"nvml={'yes' if _HAS_PYNVML else 'no'}")

    # ---- Resume ----
    if output_path.exists():
        existing = pd.read_parquet(output_path)

        # Backfill new GPU columns on old rows so schema stays consistent
        for col in GPU_METRIC_KEYS:
            if col not in existing.columns:
                existing[col] = None

        done = set(
            (r.dataset_key, r.model, r.volume_level, r.seed)
            for r in existing.itertuples(index=False)
            if r.phase == "tuned" and r.budget == max_budget
        )
        print(f"[phase3] Resuming: {len(done)} experiments already complete")
    else:
        existing = pd.DataFrame()
        done = set()

    # ---- Main loop ----
    for ds_key in datasets:
        if ds_key not in reg:
            print(f"[phase3] Skipping {ds_key} (not in registry)")
            continue

        data = np.load(data_dir / f"{ds_key}.npz")
        X_train_full, y_train_full = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]

        print(f"\n[phase3] Dataset: {ds_key} "
              f"(train={len(y_train_full)}, val={len(y_val)})")

        ds_experiments = [v for v in sub_reg.values() if v["dataset_key"] == ds_key]

        for entry in ds_experiments:
            vol = entry["volume_level"]
            seed = entry["seed"]
            local_idx = np.array(entry["local_indices"], dtype=np.int64)

            for model_name in model_names:
                key = (ds_key, model_name, vol, seed)
                if key in done:
                    continue

                print(f"  → {model_name} vol={vol} seed={seed}", flush=True)
                try:
                    rows = run_one_experiment(
                        model_name, ds_key, vol, seed,
                        X_train_full, y_train_full, local_idx,
                        X_val, y_val, budgets, max_budget,
                        gpu_interval=args.gpu_monitor_interval,
                    )
                    existing = pd.concat(
                        [existing, pd.DataFrame(rows)], ignore_index=True,
                    )
                    existing.to_parquet(output_path, index=False)
                    done.add(key)
                except Exception as e:
                    print(f"    FAILED: {type(e).__name__}: {e}")
                    traceback.print_exc(limit=3)

            print("\n")

    print(f"\n[phase3] Done. {len(existing)} total rows in {output_path}")


if __name__ == "__main__":
    main()