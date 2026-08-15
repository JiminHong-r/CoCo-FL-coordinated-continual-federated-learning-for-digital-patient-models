"""Spec WP5 axis 2: subgroup analysis for personalisation gain.

For each patient on which personalisation was run, look up demographics
(age, gender, race, BMI quantile) from ``static_data.pickle`` and group the
patient-level AUC + Δ_patient by subgroup.

Inputs:
  --csv : path to ``wp5_results.csv`` (must contain ``alpha`` and per-hospital
          rows with ``auc_hierarchical_patient`` / ``auc_hierarchical_cohort``).
  --static-pickle : ``static_data.pickle`` from the tokenized test dir.
  --splits : ``data/federated_splits`` directory (for patient → hospital).

Outputs ``wp5_subgroup_analysis.md`` with per-subgroup Δ_patient stats.

Note: the existing CSV reports MACRO per (task, α, hospital), not per-patient.
This script therefore aggregates at the *hospital-cohort* level (one row per
subgroup × hospital × α) — proxy for which centres/demographics benefit most
from hierarchical personalisation.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


GENDER_LABELS = {0: "M", 1: "F"}
RACE_LABELS = {0: "UNKNOWN", 1: "WHITE", 2: "BLACK", 3: "ASIAN",
               4: "HISPANIC", 5: "OTHER"}


def static_lookups(static_data: dict) -> dict[int, dict]:
    """Returns ``{pid: {age_bucket, gender, race, bmi_bucket}}``."""
    out = {}
    MICROSEC_PER_YEAR = 31_557_600_000_000
    for pid, data in static_data.items():
        gender_code = data.get("GENDER", {}).get("code", ["GENDER//F"])[0]
        gender = "F" if "F" in gender_code else "M"
        race_code = data.get("RACE", {}).get("code", ["RACE//UNKNOWN"])[0]
        race = race_code.split("//")[-1]
        bmi_codes = data.get("BMI", {}).get("code", [])
        bmi = bmi_codes[0].split("//")[-1] if bmi_codes else "UNKNOWN"
        birth = data.get("MEDS_BIRTH", {}).get("time", [None])[0]
        bmi_times = data.get("BMI", {}).get("time", [])
        ref_time = bmi_times[0] if bmi_times else None
        age = None
        if birth and ref_time:
            age_years = (ref_time - birth) / MICROSEC_PER_YEAR
            if age_years < 30:
                age = "<30"
            elif age_years < 50:
                age = "30-50"
            elif age_years < 70:
                age = "50-70"
            else:
                age = "70+"
        out[int(pid)] = {"gender": gender, "race": race,
                          "bmi_quantile": bmi, "age_bucket": age or "unknown"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--static-pickle", required=True, type=Path)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    with open(args.static_pickle, "rb") as f:
        static_data = pickle.load(f)
    lookups = static_lookups(static_data)

    df = pd.read_csv(args.csv)
    per_hosp = df[df["hospital"] != "MACRO"].copy()
    per_hosp["alpha"] = per_hosp["alpha"].astype(float)

    # Build hospital → dominant subgroup tags by loading each α split and
    # counting the demographic mix of its patients (only the TRAIN partition
    # is used for cohort training).
    hospital_tags: list[dict] = []
    for alpha in sorted(per_hosp["alpha"].unique()):
        split_path = args.splits / f"alpha_{alpha}.json"
        if not split_path.exists():
            split_path = args.splits / f"alpha_{float(alpha)}.json"
        if not split_path.exists():
            continue
        split = json.loads(split_path.read_text())
        for hid, parts in split["hospitals"].items():
            pids = parts.get("train", [])
            if not pids:
                continue
            subs = [lookups.get(pid, {}) for pid in pids]
            gender_counts = pd.Series([s.get("gender") for s in subs]).value_counts(normalize=True)
            race_counts = pd.Series([s.get("race") for s in subs]).value_counts(normalize=True)
            age_counts = pd.Series([s.get("age_bucket") for s in subs]).value_counts(normalize=True)
            hospital_tags.append({
                "alpha": alpha, "hospital": int(hid),
                "n_patients": len(pids),
                "frac_F": float(gender_counts.get("F", 0)),
                "frac_white": float(race_counts.get("WHITE", 0)),
                "frac_black": float(race_counts.get("BLACK", 0)),
                "frac_age_70plus": float(age_counts.get("70+", 0)),
                "frac_age_50_70": float(age_counts.get("50-70", 0)),
                "dominant_age": age_counts.idxmax() if not age_counts.empty else "unknown",
            })
    tags_df = pd.DataFrame(hospital_tags)
    if tags_df.empty:
        args.out.write_text("# Subgroup analysis\n\nNo hospital tags built (split files missing).")
        print("No hospital tags produced.")
        return

    # Merge with per-hospital AUC rows (only carry the rows we need)
    keep = ["task", "alpha", "hospital",
            "auc_hierarchical_cohort", "auc_hierarchical_patient",
            "delta_cohort_hier", "delta_patient_hier"]
    keep = [c for c in keep if c in per_hosp.columns]
    perf = per_hosp[keep].copy()
    perf["hospital"] = perf["hospital"].astype(int)
    merged = perf.merge(tags_df, on=["alpha", "hospital"], how="left")

    lines = ["# WP5 Axis 2 — Subgroup Analysis\n",
             f"Source: `{args.csv}`  ({len(merged)} per-hospital rows)\n",
             "## Per-hospital demographic mix vs Δ_patient (hierarchical)\n",
             ]
    show = merged[["task", "alpha", "hospital", "n_patients",
                   "frac_F", "frac_white", "frac_black",
                   "dominant_age", "auc_hierarchical_cohort",
                   "auc_hierarchical_patient", "delta_patient_hier"]].copy()
    for c in ("frac_F", "frac_white", "frac_black",
              "auc_hierarchical_cohort", "auc_hierarchical_patient",
              "delta_patient_hier"):
        if c in show.columns:
            show[c] = show[c].apply(lambda v: round(float(v), 3) if pd.notna(v) else None)
    lines.append(show.to_markdown(index=False))
    lines.append("\n## Aggregated by dominant_age bucket (where Δ_patient is biggest)\n")
    agg = merged.groupby("dominant_age").agg(
        n=("hospital", "size"),
        mean_cohort_auc=("auc_hierarchical_cohort", "mean"),
        mean_patient_auc=("auc_hierarchical_patient", "mean"),
        mean_delta_patient=("delta_patient_hier", "mean"),
    ).round(3)
    lines.append(agg.reset_index().to_markdown(index=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
