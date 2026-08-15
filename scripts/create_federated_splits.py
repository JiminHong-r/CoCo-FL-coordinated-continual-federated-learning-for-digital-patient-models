"""Build and persist fixed federated splits (Spec WP2 task 4).

For each α ∈ {0.1, 0.5, 1.0, 10.0}:
  1. Partition patients into N hospitals via Dirichlet (Carlos-style).
  2. Within each hospital, split patients into train / val / test (70/15/15).
  3. Verify no leakage:
       - no patient in two hospitals
       - no patient in two partitions of the same hospital
  4. Save the mapping to JSON:

       data/federated_splits/alpha_{α}.json
       {
         "alpha": 0.1,
         "num_hospitals": 3,
         "seed": 42,
         "hospitals": {
           "0": {"train": [pid, ...], "val": [...], "test": [...]},
           "1": {...}, "2": {...}
         }
       }

  5. Print per-hospital × per-partition stats.

Downstream code can load these JSON files instead of computing splits inline,
guaranteeing every experiment runs on identical patient sets.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from cocofl.federated.hospital_split import dirichlet_split_patients


def patient_train_val_test(pids: list[int], frac_train: float, frac_val: float, seed: int
                            ) -> tuple[list[int], list[int], list[int]]:
    """Disjoint train/val/test patient lists at the given fractions."""
    rng = random.Random(seed)
    shuffled = list(pids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * frac_train)
    n_val = int(n * frac_val)
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def verify_no_leakage(split_json: dict) -> None:
    """Assert no patient ID appears in two hospitals or two partitions."""
    seen: dict[int, str] = {}
    for hid, parts in split_json["hospitals"].items():
        for partition, pids in parts.items():
            for pid in pids:
                key = f"h{hid}/{partition}"
                if pid in seen:
                    raise AssertionError(
                        f"LEAKAGE: patient {pid} in both {seen[pid]} and {key}"
                    )
                seen[pid] = key


def summarise(split_json: dict) -> str:
    lines = [f"α={split_json['alpha']}, num_hospitals={split_json['num_hospitals']}, "
             f"seed={split_json['seed']}"]
    for hid, parts in split_json["hospitals"].items():
        total = sum(len(v) for v in parts.values())
        ratios = {k: f"{len(v)/total:.0%}" if total else "—" for k, v in parts.items()}
        sizes = {k: len(v) for k, v in parts.items()}
        lines.append(f"  hosp {hid}: total={total:,}  train={sizes['train']:,}({ratios['train']}) "
                     f"val={sizes['val']:,}({ratios['val']}) test={sizes['test']:,}({ratios['test']})")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenized-dir", required=True, type=Path,
                   help="dir with static_data.pickle (the source patient pool)")
    p.add_argument("--alphas", default="0.1,0.5,1.0,10.0",
                   help="comma-separated Dirichlet concentrations")
    p.add_argument("--num-hospitals", type=int, default=3)
    p.add_argument("--frac-train", type=float, default=0.7)
    p.add_argument("--frac-val", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.tokenized_dir / "static_data.pickle", "rb") as f:
        static_data = pickle.load(f)
    all_pids = list(static_data.keys())
    print(f"Source pool: {args.tokenized_dir}  ({len(all_pids):,} patients)\n")

    index = []
    for alpha in alphas:
        split = dirichlet_split_patients(all_pids, args.num_hospitals, alpha, seed=args.seed)
        hospitals = {}
        # Use a slightly different seed per hospital for the train/val/test shuffle
        # so the within-hospital order doesn't correlate across hospitals.
        for hid, pids in split.pids_per_hospital.items():
            train, val, test = patient_train_val_test(
                pids, args.frac_train, args.frac_val,
                seed=args.seed + 1000 + int(hid),
            )
            hospitals[str(hid)] = {"train": train, "val": val, "test": test}
        split_json = {
            "alpha": alpha,
            "num_hospitals": args.num_hospitals,
            "seed": args.seed,
            "source_dir": str(args.tokenized_dir),
            "frac_train": args.frac_train,
            "frac_val": args.frac_val,
            "frac_test": 1.0 - args.frac_train - args.frac_val,
            "total_patients": len(all_pids),
            "hospitals": hospitals,
        }
        verify_no_leakage(split_json)
        out_path = args.out_dir / f"alpha_{alpha}.json"
        out_path.write_text(json.dumps(split_json))
        print(summarise(split_json))
        print(f"  -> wrote {out_path}\n")
        index.append({"alpha": alpha, "file": out_path.name,
                       "num_hospitals": args.num_hospitals,
                       "patient_total": len(all_pids)})

    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"Index: {args.out_dir / 'index.json'}")


if __name__ == "__main__":
    main()
