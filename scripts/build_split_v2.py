"""WP2 splits aligned with the updated TD (2026-06-08).

Spec (Sec. 2.2 of the new TD):
  1. Extract a global held-out test set (10%) BEFORE any other split.
  2. Reserve 40% of patients EXCLUSIVELY for population pre-training;
     no centre has access to this portion.
  3. Apply Dirichlet partitioning to the remaining 50% to assign patients
     to three simulated centres (α ∈ {0.1, 0.5, 1.0, 10.0}).

The three shares are taken as fractions of the **total** cohort
(10% + 40% + 50% = 100%), so the spec's "remaining 50%" holds literally.
All three sets are carved from the same source pool and are mutually
disjoint at the patient level:

  - global_test  (10%) → honest population AUROC evaluation
  - population   (40%) → population pre-training only, no centre access
  - federated    (50%) → Dirichlet-split into ``num_hospitals`` centres

Pass one or more ``--tokenized-dir`` arguments; their ``static_data.pickle``
patient pools are merged into a single cohort before splitting (e.g. point
at both ``mimic_train`` and ``mimic_test`` to use the full cohort).

Outputs ``--out-dir``:
  - global_test.json            {patient_ids: [...], n: ...}
  - population_reserve.json     {patient_ids: [...], n: ...}
  - alpha_{0.1,0.5,1.0,10.0}.json  hospitals on the 50% federated pool
  - index.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

from cocofl.federated.hospital_split import dirichlet_split_patients


def load_pool(tokenized_dirs: list[Path]) -> list:
    """Merge static_data patient IDs from one or more tokenized dirs.

    Raises if the same patient ID appears in more than one source dir, so
    pooling cannot silently introduce duplicates / cross-pool leakage.
    """
    pool: dict = {}
    for d in tokenized_dirs:
        with open(d / "static_data.pickle", "rb") as f:
            static = pickle.load(f)
        overlap = pool.keys() & static.keys()
        if overlap:
            raise ValueError(
                f"{len(overlap)} patient IDs in {d} already seen in an earlier "
                f"source dir; sources must be disjoint to pool them safely."
            )
        pool.update(static)
    return sorted(pool.keys())


def build_age_class_labels(tokenized_dirs, target_pids):
    """Coarse age-tertile class label per patient for class-aware partitioning.

    age = (first_event_time - birth) / year, computed from the tokenized timeline
    (birth from static_data MEDS_BIRTH, first event time from the patient's first
    token). Tertile cutoffs over the target pool give 3 classes
    (age_young / age_mid / age_old). Patients with no birth record default to
    age_mid so they don't form a degenerate class. EVERY target pid gets a label
    (required — class-aware Dirichlet drops unlabelled patients).
    """
    import numpy as np
    from cocofl.data.base import TimelineDataset
    MICRO_PER_YR = 31_557_600_000_000
    target = set(target_pids)

    birth, first_time = {}, {}
    for d in tokenized_dirs:
        with open(d / "static_data.pickle", "rb") as f:
            static = pickle.load(f)
        for pid, data in static.items():
            b = data.get("MEDS_BIRTH", {}).get("time", [None])
            if b and b[0] is not None:
                birth[pid] = b[0]
        ds = TimelineDataset(d, n_positions=2048)
        for shard in ds._data.shards:
            stimes = shard["times"][:]
            spids = shard["patient_ids"].tolist()
            soffs = shard["patient_offsets"].tolist()
            for i, pid in enumerate(spids):
                if pid in target:
                    first_time[pid] = int(stimes[int(soffs[i])])

    ages = {pid: (first_time[pid] - birth[pid]) / MICRO_PER_YR
            for pid in target if pid in birth and pid in first_time}
    vals = np.array(list(ages.values()), dtype=float)
    q1, q2 = np.percentile(vals, [33.33, 66.67])
    labels = {}
    for pid in target_pids:
        a = ages.get(pid)
        if a is None or not (a == a):
            labels[pid] = "age_mid"
        elif a < q1:
            labels[pid] = "age_young"
        elif a < q2:
            labels[pid] = "age_mid"
        else:
            labels[pid] = "age_old"
    n_real = len(ages)
    print(f"  age class labels: {n_real}/{len(target_pids)} have real age "
          f"(tertiles {q1:.0f}/{q2:.0f}y); "
          f"counts={ {c: sum(1 for v in labels.values() if v == c) for c in set(labels.values())} }")
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenized-dir", required=True, type=Path, nargs="+",
                    help="one or more tokenized dirs with static_data.pickle; "
                         "their patient pools are merged into one cohort")
    ap.add_argument("--out-dir", required=True, type=Path)
    # Three-way split fractions of the TOTAL cohort (must sum to 1.0).
    ap.add_argument("--global-test-frac", type=float, default=0.10)
    ap.add_argument("--population-frac", type=float, default=0.40)
    ap.add_argument("--federated-frac", type=float, default=0.50)
    # Within-hospital train/val/test fractions (of each hospital's patients).
    ap.add_argument("--frac-train", type=float, default=0.70)
    ap.add_argument("--frac-val", type=float, default=0.15)
    ap.add_argument("--frac-test", type=float, default=0.15)
    ap.add_argument("--alphas", default="0.1,0.5,1.0,10.0")
    ap.add_argument("--num-hospitals", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class-aware", action="store_true",
                    help="partition the 50%% federated pool by per-patient age "
                         "tertile (class-aware Dirichlet) so centres differ in "
                         "patient distribution, not just size. Default off "
                         "(quantity-only).")
    args = ap.parse_args()

    three_way = args.global_test_frac + args.population_frac + args.federated_frac
    if abs(three_way - 1.0) > 1e-6:
        raise ValueError(
            f"global-test ({args.global_test_frac}) + population "
            f"({args.population_frac}) + federated ({args.federated_frac}) "
            f"= {three_way}, must sum to 1.0"
        )

    all_pids = load_pool(args.tokenized_dir)
    rng = random.Random(args.seed)
    rng.shuffle(all_pids)

    n_total = len(all_pids)
    n_test = int(round(n_total * args.global_test_frac))
    n_pop = int(round(n_total * args.population_frac))
    # federated takes the remainder so the three sets exactly tile the cohort
    global_test_pids = sorted(all_pids[:n_test])
    population_pids = sorted(all_pids[n_test:n_test + n_pop])
    federated_pool_pids = sorted(all_pids[n_test + n_pop:])

    sources = [str(d) for d in args.tokenized_dir]
    print(f"Source pool: {sources}  ({n_total} patients)")
    print(f"  global_test:    {len(global_test_pids):,} "
          f"({len(global_test_pids)/n_total*100:.1f}%)")
    print(f"  population:     {len(population_pids):,} "
          f"({len(population_pids)/n_total*100:.1f}%)")
    print(f"  federated_pool: {len(federated_pool_pids):,} "
          f"({len(federated_pool_pids)/n_total*100:.1f}%)")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Write global_test.json
    (args.out_dir / "global_test.json").write_text(json.dumps({
        "patient_ids": global_test_pids,
        "n": len(global_test_pids),
        "frac": args.global_test_frac,
        "sources": sources,
        "seed": args.seed,
    }, indent=2))

    # Write population_reserve.json
    (args.out_dir / "population_reserve.json").write_text(json.dumps({
        "patient_ids": population_pids,
        "n": len(population_pids),
        "frac": args.population_frac,
        "sources": sources,
        "seed": args.seed,
        "note": "exclusive population pre-training pool; no centre has access",
    }, indent=2))

    # Optional class-aware labels (age tertiles) for the federated pool.
    class_labels = None
    if args.class_aware:
        print("Class-aware partitioning: building per-patient age labels ...")
        class_labels = build_age_class_labels(args.tokenized_dir, federated_pool_pids)

    # Build Dirichlet splits on the federated (50%) pool
    alphas = [float(a) for a in args.alphas.split(",")]
    index_entries = []
    for alpha in alphas:
        split = dirichlet_split_patients(
            federated_pool_pids, args.num_hospitals, alpha, seed=args.seed,
            class_labels=class_labels,
        )
        hospitals = {}
        for hid, hosp_pids in split.pids_per_hospital.items():
            shuffled = list(hosp_pids)
            rng_h = random.Random(args.seed + hid)
            rng_h.shuffle(shuffled)
            n_h = len(shuffled)
            n_tr = int(round(n_h * args.frac_train))
            n_va = int(round(n_h * args.frac_val))
            hospitals[str(hid)] = {
                "train": sorted(shuffled[:n_tr]),
                "val": sorted(shuffled[n_tr:n_tr + n_va]),
                "test": sorted(shuffled[n_tr + n_va:]),
            }
        path = args.out_dir / f"alpha_{alpha}.json"
        path.write_text(json.dumps({
            "alpha": alpha,
            "num_hospitals": args.num_hospitals,
            "seed": args.seed,
            "sources": sources,
            "global_test_excluded": True,
            "population_excluded": True,
            "global_test_n": len(global_test_pids),
            "population_n": len(population_pids),
            "federated_pool_n": len(federated_pool_pids),
            "frac_train": args.frac_train,
            "frac_val": args.frac_val,
            "frac_test": args.frac_test,
            "total_patients": n_total,
            "hospitals": hospitals,
        }, indent=2))
        sizes = {hid: len(h["train"]) + len(h["val"]) + len(h["test"])
                 for hid, h in hospitals.items()}
        print(f"  α={alpha}: hospital sizes = {sizes}")
        index_entries.append({"alpha": alpha, "path": path.name, "sizes": sizes})

    # Index file
    (args.out_dir / "index.json").write_text(json.dumps({
        "global_test_path": "global_test.json",
        "population_reserve_path": "population_reserve.json",
        "alpha_splits": index_entries,
        "frac_global_test": args.global_test_frac,
        "frac_population": args.population_frac,
        "frac_federated": args.federated_frac,
        "sources": sources,
        "n_total": n_total,
        "n_global_test": len(global_test_pids),
        "n_population": len(population_pids),
        "n_federated_pool": len(federated_pool_pids),
        "spec_note": (
            "Per WP2 TD (2026-06-08): one cohort split 10/40/50 of total into "
            "global_test (extracted first), exclusive population reserve, and "
            "the federated pool (Dirichlet-split into hospitals). All three "
            "sets are mutually disjoint at the patient level."
        ),
    }, indent=2))

    # Sanity: cross-check disjointness of the three top-level sets and that no
    # hospital partition contains a global_test or population patient.
    test_set = set(global_test_pids)
    pop_set = set(population_pids)
    fed_set = set(federated_pool_pids)
    assert not (test_set & pop_set), "LEAK: global_test ∩ population"
    assert not (test_set & fed_set), "LEAK: global_test ∩ federated"
    assert not (pop_set & fed_set), "LEAK: population ∩ federated"
    assert len(test_set) + len(pop_set) + len(fed_set) == n_total, (
        "three-way split does not tile the cohort"
    )
    for alpha in alphas:
        d = json.loads((args.out_dir / f"alpha_{alpha}.json").read_text())
        seen: dict[int, str] = {}
        for hid, parts in d["hospitals"].items():
            in_split = set(parts["train"]) | set(parts["val"]) | set(parts["test"])
            assert not (in_split & test_set), (
                f"LEAK: α={alpha} hospital {hid} contains global_test patients"
            )
            assert not (in_split & pop_set), (
                f"LEAK: α={alpha} hospital {hid} contains population patients"
            )
            # no patient in two partitions / two hospitals of the same split
            for partition in ("train", "val", "test"):
                for pid in parts[partition]:
                    key = f"h{hid}/{partition}"
                    if pid in seen:
                        raise AssertionError(
                            f"LEAK: α={alpha} patient {pid} in both "
                            f"{seen[pid]} and {key}"
                        )
                    seen[pid] = key

    print("\n✅ global_test / population / federated mutually disjoint; "
          "no leakage into hospital partitions")
    print(f"Wrote {args.out_dir}/")


if __name__ == "__main__":
    main()
