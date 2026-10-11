"""
Phase 3: Boosting Branch.

For each (dataset, volume, seed, model):
  1. Fit with DEFAULT hyperparameters. Record metrics + costs.
  2. Run random search for max(budgets) trials. Track best-so-far on val.
  3. At each checkpoint (5, 10, 15, 30), evaluate the current best-so-far
     model on ID and OOD test sets. Emit one row per checkpoint.
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
        p.update(random_state=seed, n_jobs=-1, eval_metric="logloss",
                 tree_method="hist", verbosity=0)
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
    X_val, y_val, X_id, y_id, X_ood, y_ood,
    budgets, max_budget,
):
    """Run one (model, dataset, volume, seed) experiment. Returns list of rows."""
    evaluate = make_evaluator(y_train_full)

    # Slice the training data using the LOCAL indices from Phase 2
    X_sub = X_train_full[local_idx]
    y_sub = y_train_full[local_idx]

    common = dict(
        dataset_key=dataset_key, model=model_name,
        volume_level=int(volume), seed=int(seed),
        n_train_rows=int(len(y_sub)),
        n_features=int(X_train_full.shape[1]),
    )

    rows = []

    # ---- Baseline: default parameters ----
    d_seed = derive_seed(model_name, dataset_key, volume, seed, "default")
    t0 = time.time()
    model = build_model(model_name, DEFAULT_PARAMS[model_name], d_seed)
    model.fit(X_sub, y_sub)
    fit_s = time.time() - t0

    val_auc, val_ll = evaluate(model, X_val, y_val)
    id_auc,  id_ll  = evaluate(model, X_id,  y_id)
    if X_ood is not None:
        ood_auc, ood_ll = evaluate(model, X_ood, y_ood)
    else:
        ood_auc, ood_ll = float("nan"), float("nan")

    rows.append({
        **common,
        "phase": "default", "budget": 0,
        "val_auc": val_auc, "val_logloss": val_ll,
        "id_auc":  id_auc,  "id_logloss":  id_ll,
        "ood_auc": ood_auc, "ood_logloss": ood_ll,
        "fit_time_s": fit_s,
        "config_json": json.dumps(DEFAULT_PARAMS[model_name], sort_keys=True),
    })

    # ---- Random search ----
    space = SEARCH_SPACES[model_name]
    best_val_auc = -np.inf
    best_val_ll  = float("nan")
    best_config  = None
    best_model   = None
    cumulative_time = 0.0
    n_completed_trials = 0

    for trial in range(1, max_budget + 1):
        t_seed = derive_seed(model_name, dataset_key, volume, seed, "trial", trial)
        rng = np.random.default_rng(t_seed)
        config = sample_params(rng, space)

        t0 = time.time()
        try:
            model = build_model(model_name, config, t_seed)
            model.fit(X_sub, y_sub)
            fit_s = time.time() - t0
            val_auc, val_ll = evaluate(model, X_val, y_val)
            n_completed_trials += 1
            cumulative_time += fit_s
        except Exception as e:
            print(f"    [trial {trial}] skipped: {type(e).__name__}: {e}")
            continue

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_ll  = val_ll
            best_config  = config
            best_model   = model

        # If we hit a checkpoint, emit a row for the current best-so-far
        if trial in budgets:
            id_auc,  id_ll  = evaluate(best_model, X_id,  y_id)
            if X_ood is not None:
                ood_auc, ood_ll = evaluate(best_model, X_ood, y_ood)
            else:
                ood_auc, ood_ll = float("nan"), float("nan")

            rows.append({
                **common,
                "phase": "tuned", "budget": int(trial),
                "val_auc": best_val_auc, "val_logloss": best_val_ll,
                "id_auc":  id_auc,       "id_logloss":  id_ll,
                "ood_auc": ood_auc,      "ood_logloss": ood_ll,
                "fit_time_s": cumulative_time,
                "config_json": json.dumps(best_config, sort_keys=True),
            })

    return rows


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

    datasets  = args.datasets or list(reg.keys())
    model_names = args.models or AVAILABLE_MODELS
    model_names = [m for m in model_names if m in AVAILABLE_MODELS]
    if not model_names:
        raise SystemExit("[phase3] No boosting libraries available.")

    budgets = cfg.get("boosting_budgets", [5, 10, 15, 30])
    max_budget = max(budgets)
    print(f"[phase3] Models: {model_names}")
    print(f"[phase3] Budgets: {budgets} (max={max_budget})")
    print(f"[phase3] Datasets: {datasets}")

    # ---- Resume ----
    if output_path.exists():
        existing = pd.read_parquet(output_path)
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
        X_id,  y_id  = data["X_id"],  data["y_id"]
        X_ood = data["X_ood"] if "X_ood" in data.files else None
        y_ood = data["y_ood"] if "y_ood" in data.files else None

        print(f"\n[phase3] Dataset: {ds_key} "
              f"(train={len(y_train_full)}, val={len(y_val)}, "
              f"id={len(y_id)}, ood={0 if y_ood is None else len(y_ood)})")

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
                        X_val, y_val, X_id, y_id, X_ood, y_ood,
                        budgets, max_budget,
                    )
                    existing = pd.concat(
                        [existing, pd.DataFrame(rows)], ignore_index=True,
                    )
                    existing.to_parquet(output_path, index=False)
                    done.add(key)
                except Exception as e:
                    print(f"    FAILED: {type(e).__name__}: {e}")
                    traceback.print_exc(limit=3)

    print(f"\n[phase3] Done. {len(existing)} total rows in {output_path}")


if __name__ == "__main__":
    main()