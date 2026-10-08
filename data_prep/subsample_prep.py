"""
Phase 2: Subsample Generation.

Reads the Phase 1 prepared datasets (which contain X_train, val_idx, etc.)
and generates deterministic training subsamples for specified volume levels
and seeds.

Writes a subsample_registry.json to <data_dir> mapping:
    (dataset_key, volume_level, seed) -> {
        "train_indices": [...], 
        "indices_digest": "...",
        "n_samples": N
    }

These indices are used by Phase 3 (Boosting) and Phase 4 (Foundation) to
ensure they evaluate on the EXACT SAME training rows.

Usage:
    python src/subsample_prep.py --config configs/grid.yaml
    python src/subsample_prep.py --synthetic --data-dir data/prepared_smoke
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

def generate_subsamples(data_dir: Path, config: dict):
    """Phase 2: Generate deterministic subsamples for each dataset."""
    registry_path = data_dir / "registry.json"
    if not registry_path.exists():
        sys.exit(f"[subsample] Phase 1 registry not found at {registry_path}. Run data_prep.py first.")
    
    phase1_registry = json.loads(registry_path.read_text())
    datasets = phase1_registry.get("datasets", {})
    
    # Configuration for Phase 2
    volume_levels = config.get("volume_levels") # Example
    seeds = config.get("seeds") # Example
    
    subsample_registry = {}
    
    for key, meta in datasets.items():
        npz_path = data_dir / f"{key}.npz"
        if not npz_path.exists():
            print(f"[subsample] Missing {npz_path}, skipping.")
            continue
            
        data = np.load(npz_path)
        n_train = data["X_train"].shape[0]
        val_idx = set(data["val_idx"].tolist())
        
        # "Determine available volume levels (only rows ∈ remaining rows)"
        remaining_rows = np.arange(n_train, dtype=np.int64)
        max_volume = len(remaining_rows)
        
        # Resolve "full" to the actual max_volume, and filter out impossible sizes
        requested_volumes = []
        for v in volume_levels:
            if v == "full":
                requested_volumes.append(max_volume)
            elif isinstance(v, int) and v <= max_volume:
                requested_volumes.append(v)
                
        # Remove duplicates (in case "full" equals one of the integer levels) and sort
        available_volumes = sorted(list(set(requested_volumes)))
        
        if not available_volumes:
            available_volumes = [max_volume] # Fallback if all requested levels are too big
            
        print(f"[subsample] {key}: n_train={n_train}, n_val={len(val_idx)}, "
              f"remaining={max_volume}, available_volumes={available_volumes}")


        train_idx_global = data["train_idx"]
        
        for vol in available_volumes:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                local_idx = rng.choice(remaining_rows, size=vol, replace=False).astype(np.int64)
                local_idx.sort()
                digest = hashlib.sha256(local_idx.tobytes()).hexdigest()
                
                # Global indices (for audit / downstream direct slicing of the original data)
                global_idx = train_idx_global[local_idx]
                
                registry_key = f"{key}_vol{vol}_seed{seed}"
                subsample_registry[registry_key] = {
                    "dataset_key": key,
                    "volume_level": vol,
                    "seed": seed,
                    "local_indices": local_idx.tolist(),   # slice X_train
                    "global_indices": global_idx.tolist(), # slice X_original (audit)
                    "indices_digest": digest,              # hash of local indices
                    "n_samples": vol,
                }
                
        print(f"[subsample] Generated {len(available_volumes) * len(seeds)} "
              f"subsamples for {key}")

    # Save Phase 2 registry
    out_path = data_dir / "subsample_registry.json"
    out_path.write_text(json.dumps(subsample_registry, indent=2))
    print(f"[subsample] Wrote {len(subsample_registry)} entries to {out_path}")
    
    # Write READY marker for Phase 2
    (data_dir / "READY_PHASE2").write_text("ok\n")
    print(f"[subsample] READY_PHASE2 written. Phases 3 and 4 can now run.")

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/grid.yaml")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    import yaml
    if a.synthetic:
        data_dir = Path(a.data_dir or "data/prepared_smoke")
        # Hardcoded synthetic config for Phase 2
        cfg = {
            "volume_levels": [50, 100, 200],
            "seeds": [0, 1]
        }
    else:
        cfg = yaml.safe_load(Path(a.config).read_text())
        data_dir = Path(a.data_dir or cfg["runtime"]["data_dir"])
    
    if not data_dir.exists():
        sys.exit(f"[subsample] Data dir {data_dir} does not exist.")
        
    generate_subsamples(data_dir, cfg)

if __name__ == "__main__":
    main()