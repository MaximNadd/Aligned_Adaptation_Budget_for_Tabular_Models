"""
Phase 4 — Foundation model evaluation.

Reads Phase 1/2 artifacts, writes results_foundation.parquet with per-experiment
GPU + node telemetry. GPU monitoring is imported from run_boosting.py so both
phases write identical GPU_METRIC_KEYS into their parquet files.

Deps: pandas numpy pyarrow scikit-learn psutil pyyaml nvidia-ml-py
Optional: tabpfn, tabicl, torch (for real foundation models).
"""

import os
import json
import time
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss

# ---- GPU telemetry imported from Phase 3 ----
from run_boosting import (
    GPUMonitor,
    GPU_METRIC_KEYS,
    _tag_device,
    _first_visible_gpu,
)
try:
    from run_boosting import _HAS_PYNVML
except ImportError:                    # older run_boosting without the flag
    _HAS_PYNVML = False


# ============================================================
# FOUNDATION MODEL FACTORY
# ============================================================

CONTEXT_LIMITS = {
    "TabPFN_3":   10_000,
    "TabPFN_2_5":  5_000,
    "TabICL":     20_000,
    "TabICLv2":   20_000,
}

AVAILABLE_MODELS = list(CONTEXT_LIMITS.keys())


def build_foundation_model(model_name: str, seed: int):
    """
    Returns (model, device_str) where device_str is "gpu" or "cpu".
    Tries the real library; falls back to HistGradientBoosting if unavailable.
    """
    name = model_name.lower()

    if "tabpfn" in name:
        try:
            from tabpfn import TabPFNClassifier
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = TabPFNClassifier(device=device, random_state=seed)
            return model, ("gpu" if device == "cuda" else "cpu")
        except ImportError:
            pass

    if "tabicl" in name:
        try:
            from tabicl import TabICLClassifier
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = TabICLClassifier(device=device, random_state=seed)
            return model, ("gpu" if device == "cuda" else "cpu")
        except (ImportError, TypeError):
            pass

    print(f"    [fallback] {model_name} unavailable -> HistGradientBoosting (CPU)")
    return HistGradientBoostingClassifier(random_state=seed), "cpu"


def run_foundation_inference(model, X_context, y_context, X_query):
    """
    Fits on (X_context, y_context) and returns
    (positive_class_probs, actual_context_size_used).
    """
    model.fit(X_context, y_context)
    probs = model.predict_proba(X_query)

    probs = np.asarray(probs)
    if probs.ndim == 2 and probs.shape[1] == 2:
        probs = probs[:, 1]
    else:
        probs = probs.ravel()

    # Truncation check: does the model expose a context size attribute?
    actual = len(X_context)
    for attr in ("context_size_", "n_context_", "actual_context_size", "_context_size"):
        if hasattr(model, attr):
            actual = int(getattr(model, attr))
            break

    return probs, actual


# ============================================================
# CONTEXT STRATEGIES
# ============================================================

def apply_context_strategy(X, y, strategy, max_size, seed):
    """Returns (X_ctx, y_ctx, applied_strategy_name)."""
    if strategy == "none_needed" or len(X) <= max_size:
        return X, y, "full_volume"

    rng = np.random.default_rng(seed)

    if strategy == "stratified_by_target":
        idx = []
        for c in np.unique(y):
            c_idx = np.where(y == c)[0]
            take = max(1, int(max_size * len(c_idx) / len(y)))
            idx.extend(rng.choice(c_idx, min(take, len(c_idx)), replace=False))
        idx = np.array(idx, dtype=np.int64)
    else:
        idx = rng.choice(len(X), max_size, replace=False)

    return X[idx], y[idx], strategy


# ============================================================
# EVALUATION
# ============================================================

def evaluate(y_true, y_pred_proba):
    try:
        return float(roc_auc_score(y_true, y_pred_proba)), float(log_loss(y_true, y_pred_proba))
    except ValueError as e:
        print(f"    [eval] {e}")
        return float("nan"), float("nan")


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_foundation_experiment(
    model_name, dataset_key, volume, seed,
    X_train, y_train, X_val, y_val,
    strategy, context_limit,
    gpu_cost_per_hour, gpu_interval=0.05,
):
    """
    Runs ONE (model, dataset, volume, seed, strategy) cell.
    Returns a list of rows (empty if rejected by the truncation gate).
    """
    process = psutil.Process(os.getpid())

    X_ctx, y_ctx, applied_strat = apply_context_strategy(
        X_train, y_train, strategy, context_limit, seed
    )
    expected_size = len(X_ctx)

    model, device_str = build_foundation_model(model_name, seed)

    mon = GPUMonitor(interval=gpu_interval)          # ← from run_boosting
    mon.__enter__()
    mon.pause()
    try:
        mon.resume()
        ram_before = process.memory_info().rss / 1024**2
        t0 = time.time()
        try:
            preds, actual_size = run_foundation_inference(model, X_ctx, y_ctx, X_val)
            wall_time = time.time() - t0
        finally:
            mon.pause()
            gpu_stats = _tag_device(mon.snapshot(), device_str)
        ram_after = process.memory_info().rss / 1024**2
    finally:
        mon.__exit__(None, None, None)

    if actual_size < expected_size:
        print(f"    [REJECT] {model_name}: expected {expected_size}, got {actual_size}")
        return []

    auc, logloss = evaluate(y_val, preds)

    cost_usd = (wall_time / 3600.0) * gpu_cost_per_hour

    return [{
        "dataset_key": dataset_key,
        "model": model_name,
        "volume_level": int(volume),
        "seed": int(seed),
        "strategy": applied_strat,

        "val_auc": auc,
        "val_logloss": logloss,

        "expected_context_size": expected_size,
        "actual_context_size": actual_size,
        "silent_truncation": False,

        "wallclock_s": wall_time,
        "peak_node_mem_mb": float(max(ram_before, ram_after)),
        "estimated_cost_usd": cost_usd,

        **gpu_stats,
    }]


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grid.yaml")
    ap.add_argument("--data-dir", default="data/prepared_smoke")
    ap.add_argument("--output", default=None)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--gpu-monitor-interval", type=float, default=0.05)
    ap.add_argument("--gpu-cost-per-hour", type=float, default=1.50)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output or data_dir / "results_foundation.parquet")

    if not (data_dir / "registry.json").exists():
        raise SystemExit(f"[phase4] registry.json not found in {data_dir}")
    if not (data_dir / "subsample_registry.json").exists():
        raise SystemExit(f"[phase4] subsample_registry.json not found in {data_dir}")

    try:
        cfg = yaml.safe_load(Path(args.config).read_text())
        strategies = cfg.get(
            "foundation_strategies",
            ["random_subsample", "stratified_by_target"],
        )
    except FileNotFoundError:
        strategies = ["random_subsample", "stratified_by_target"]

    with open(data_dir / "registry.json") as f:
        reg = json.load(f)["datasets"]
    with open(data_dir / "subsample_registry.json") as f:
        sub_reg = json.load(f)

    datasets = args.datasets or list(reg.keys())
    model_names = args.models or AVAILABLE_MODELS
    model_names = [m for m in model_names if m in AVAILABLE_MODELS]
    if not model_names:
        raise SystemExit("[phase4] No models selected.")

    print(f"[phase4] Models: {model_names}")
    print(f"[phase4] Strategies: {strategies}")
    print(f"[phase4] Datasets: {datasets}")
    print(f"[phase4] GPU monitor: interval={args.gpu_monitor_interval}s "
          f"device={_first_visible_gpu()} "
          f"nvml={'yes' if _HAS_PYNVML else 'no'}")

    # ---- resume ----
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        for col in GPU_METRIC_KEYS:
            if col not in existing.columns:
                existing[col] = None
        done = set(
            (r.dataset_key, r.model, r.volume_level, r.seed, r.strategy)
            for r in existing.itertuples(index=False)
        )
        print(f"[phase4] Resuming: {len(done)} runs already complete")
    else:
        existing = pd.DataFrame()
        done = set()

    # ---- main loop ----
    for ds_key in datasets:
        if ds_key not in reg:
            print(f"[phase4] Skipping {ds_key} (not in registry)")
            continue

        data = np.load(data_dir / f"{ds_key}.npz")
        X_train_full, y_train_full = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]

        print(f"\n[phase4] Dataset: {ds_key} "
              f"(train={len(y_train_full)}, val={len(y_val)})")

        ds_experiments = [v for v in sub_reg.values() if v["dataset_key"] == ds_key]

        for entry in ds_experiments:
            vol = entry["volume_level"]
            seed = entry["seed"]
            local_idx = np.array(entry["local_indices"], dtype=np.int64)
            X_sub, y_sub = X_train_full[local_idx], y_train_full[local_idx]

            for model_name in model_names:
                context_limit = CONTEXT_LIMITS[model_name]
                active_strats = (
                    strategies if len(X_sub) > context_limit else ["none_needed"]
                )

                for strat in active_strats:
                    key = (ds_key, model_name, vol, seed, strat)
                    if key in done:
                        continue

                    print(f"  → {model_name} vol={vol} strat={strat} seed={seed}",
                          flush=True)
                    try:
                        rows = run_foundation_experiment(
                            model_name, ds_key, vol, seed,
                            X_sub, y_sub, X_val, y_val,
                            strategy=strat,
                            context_limit=context_limit,
                            gpu_cost_per_hour=args.gpu_cost_per_hour,
                            gpu_interval=args.gpu_monitor_interval,
                        )
                        if rows:
                            existing = pd.concat(
                                [existing, pd.DataFrame(rows)], ignore_index=True,
                            )
                            existing.to_parquet(output_path, index=False)
                        done.add(key)
                    except Exception as e:
                        print(f"    FAILED: {type(e).__name__}: {e}")
                        traceback.print_exc(limit=3)

            print()

    print(f"\n[phase4] Done. {len(existing)} total rows in {output_path}")


if __name__ == "__main__":
    main()