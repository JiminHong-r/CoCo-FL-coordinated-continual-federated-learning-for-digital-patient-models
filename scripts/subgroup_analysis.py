"""WP5 axis 2: for which patients, and when, does personalisation help?

``patient_records.csv`` has one row per patient the patient layer ran on, with
``val_auroc_patient`` and ``val_auroc_cohort``. This joins each patient to their
demographics and groups them. It also groups by the conditions of the run (fitting
data, evaluation evidence, alpha, task), which answer the "when" part.

Age is the main grouping (the class-aware split is by age). Across the whole cohort
only ~69 % of patients have a usable birth record, but every patient the patient
layer ran on has one, so coverage is 100 % here. Gender and race are known for
everyone. Coverage is printed for each grouping.

::

    python scripts/subgroup_analysis.py \\
        --records "results_v11full_classaware_*/patient_records.csv" \\
        --static-pickle data/tokenized/mimic_train/static_data.pickle \\
        --out axis2_subgroups.md
"""
from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import pandas as pd

MICROSEC_PER_YEAR = 31_557_600_000_000


def demographics(static_data: dict) -> pd.DataFrame:
    """One row per patient: gender, race, and age bucket where derivable."""
    rows = []
    for pid, data in static_data.items():
        gender_code = data.get("GENDER", {}).get("code", ["GENDER//F"])[0]
        race_code = data.get("RACE", {}).get("code", ["RACE//UNKNOWN"])[0]

        birth_times = data.get("MEDS_BIRTH", {}).get("time") or [None]
        birth = birth_times[0]
        # Age from the earliest non-birth event, as
        # build_split_v2.build_age_class_labels does. Using a BMI time would drop
        # patients without BMI; using MEDS_BIRTH itself would make every age zero.
        times = [t for k, v in data.items()
                 if k != "MEDS_BIRTH" and isinstance(v, dict)
                 for t in (v.get("time") or []) if t is not None]
        age_bucket = "unknown"
        if birth is not None and times:
            yrs = (min(times) - birth) / MICROSEC_PER_YEAR
            age_bucket = ("<30" if yrs < 30 else "30-50" if yrs < 50
                          else "50-70" if yrs < 70 else "70+")
        rows.append({
            "patient_id": int(pid),
            "gender": "F" if "F" in gender_code else "M",
            "race": race_code.split("//")[-1],
            "age_bucket": age_bucket,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True,
                    help="glob for patient_records.csv files")
    ap.add_argument("--static-pickle", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    files = sorted(glob.glob(args.records))
    if not files:
        raise SystemExit(f"no patient_records.csv matched {args.records!r}")
    rec = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    with open(args.static_pickle, "rb") as f:
        demo = demographics(pickle.load(f))

    df = rec.merge(demo, on="patient_id", how="left")
    # The patient layer's own gain, on the slice acceptance is decided on.
    df["delta"] = df["val_auroc_patient"] - df["val_auroc_cohort"]
    ev = df[df["evaluable"].astype(bool)]

    lines = [
        "# WP5 Axis 2 — Subgroup analysis (per patient)\n",
        f"Source: {len(files)} cells, {len(df):,} patient records; "
        f"{len(ev):,} evaluable, {int(df['accepted'].sum()):,} accepted.\n",
        "`delta` is the patient adapter's validation AUROC minus the cohort "
        "model's on the same slice — the quantity acceptance is decided on. It is "
        "undefined where the slice is single-class, so only evaluable rows are "
        "grouped.\n",
    ]

    for var in ("gender", "race", "age_bucket"):
        known = df[var].notna() & (df[var] != "unknown") & (df[var] != "UNKNOWN")
        lines.append(f"\n## By {var}  (coverage {100.0 * known.mean():.0f}%)\n")
        if ev.empty:
            lines.append("_no evaluable patients_\n")
            continue
        agg = ev.groupby(var).agg(
            n=("patient_id", "size"),
            accepted=("accepted", "sum"),
            accept_rate=("accepted", "mean"),
            mean_delta=("delta", "mean"),
        ).round(3).sort_values("n", ascending=False)
        lines.append(agg.reset_index().to_markdown(index=False))

    # The "when" part: the data conditions under which the patient layer pays off.
    if not ev.empty:
        lines.append("\n## When: by amount of fitting data (`n_fit` quartile)\n")
        q = ev.copy()
        q["n_fit_q"] = pd.qcut(q["n_fit"], 4, duplicates="drop")
        agg = q.groupby("n_fit_q", observed=True).agg(
            n=("patient_id", "size"),
            accept_rate=("accepted", "mean"),
            mean_delta=("delta", "mean"),
        ).round(3)
        lines.append(agg.reset_index().astype({"n_fit_q": str}).to_markdown(index=False))

        lines.append("\n## When: by evidence in the evaluation slice (`n_eval_pos`)\n")
        q = ev.copy()
        q["pos_band"] = pd.cut(q["n_eval_pos"], [-0.1, 0.5, 2.5, 10.5, 1e9],
                               labels=["0", "1-2", "3-10", ">10"])
        agg = q.groupby("pos_band", observed=True).agg(
            n=("patient_id", "size"),
            accept_rate=("accepted", "mean"),
            mean_delta=("delta", "mean"),
        ).round(3)
        lines.append(agg.reset_index().astype({"pos_band": str}).to_markdown(index=False))

        lines.append("\n## When: by heterogeneity level (alpha)\n")
        agg = ev.groupby("alpha").agg(
            n=("patient_id", "size"),
            accept_rate=("accepted", "mean"),
            mean_delta=("delta", "mean"),
        ).round(3)
        lines.append(agg.reset_index().to_markdown(index=False))

        lines.append("\n## When: by task\n")
        agg = ev.groupby("task").agg(
            n=("patient_id", "size"),
            accept_rate=("accepted", "mean"),
            mean_delta=("delta", "mean"),
        ).round(3)
        lines.append(agg.reset_index().to_markdown(index=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"Wrote {args.out}  ({len(df):,} records, {len(ev):,} evaluable)")


if __name__ == "__main__":
    main()
