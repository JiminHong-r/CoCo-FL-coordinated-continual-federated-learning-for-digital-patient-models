"""Generate a Markdown summary of WP5 results from wp5_results.csv.

Sections:
  1. MACRO AUROC per (task × α) for all 5 methods
  2. MACRO AUPRC per (task × α)
  3. MACRO ECE per (task × α)
  4. Personalisation gain (Δ_cohort, Δ_patient) per task × α
  5. Catastrophic forgetting (BWT per task × α, MACRO)
  6. Communication + wall-clock summary
  7. Best method per (task × α) by macro AUROC

Usage::

    python scripts/summarize_wp5.py --csv results_wp5_v2/wp5_results.csv \
                                     --out results_wp5_v2/wp5_summary.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

METHODS = [
    "population_only", "local_finetune", "standard_fedavg",
    "top_down_only", "hierarchical_cohort",
]


def _pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot to (task × α) rows, method columns."""
    cols = [f"{value_col}_{m}" for m in METHODS]
    cols = [c for c in cols if c in df.columns]
    macro = df[df["hospital"] == "MACRO"][["task", "alpha"] + cols].copy()
    macro = macro.rename(columns={c: c.replace(f"{value_col}_", "") for c in cols})
    return macro


def _format_md(df: pd.DataFrame, title: str, fmt: str = "{:.3f}") -> str:
    if df.empty:
        return f"### {title}\n\n_(no data)_\n"
    lines = [f"### {title}\n"]
    out = df.copy()
    for col in out.columns:
        if col in ("task", "alpha"):
            continue
        out[col] = out[col].apply(lambda v: fmt.format(v) if pd.notna(v) else "—")
    lines.append(out.to_markdown(index=False))
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    lines: list[str] = [f"# WP5 Comprehensive Evaluation Summary\n",
                         f"Source: `{args.csv}`  ({len(df)} rows)\n"]

    lines.append("## 1. AUROC (macro across hospitals)\n")
    lines.append(_format_md(_pivot(df, "auc"), "MACRO AUROC"))

    if any(c.startswith("auprc_") for c in df.columns):
        lines.append("## 2. AUPRC (macro)\n")
        lines.append(_format_md(_pivot(df, "auprc"), "MACRO AUPRC"))

    if any(c.startswith("ece_") for c in df.columns):
        lines.append("## 3. ECE (lower = better calibration)\n")
        lines.append(_format_md(_pivot(df, "ece"), "MACRO ECE"))

    macro = df[df["hospital"] == "MACRO"]
    if "delta_cohort_hier" in df.columns:
        lines.append("## 4. Personalisation Gain (hierarchical method)\n")
        per_hosp_gain = df[df["hospital"] != "MACRO"][
            ["task", "alpha", "hospital", "delta_cohort_hier", "delta_patient_hier",
             "fwt_hier", "bwt_hier_cohort"]
        ].copy()
        # Average across hospitals
        gain_macro = per_hosp_gain.groupby(["task", "alpha"], as_index=False).mean(numeric_only=True)
        lines.append(_format_md(
            gain_macro.drop(columns=["hospital"]) if "hospital" in gain_macro.columns else gain_macro,
            "Δ_cohort, Δ_patient, FWT, BWT (averaged across hospitals)",
        ))

    if "comm_bytes_cohort_per_round" in df.columns:
        lines.append("## 5. Communication & Wall-clock (macro)\n")
        cw = macro[["task", "alpha", "comm_bytes_cohort_per_round",
                    "wall_clock_local_ft", "wall_clock_fedavg",
                    "wall_clock_top_down", "wall_clock_hierarchical"]].copy()
        cw["comm_bytes_cohort_per_round"] = cw["comm_bytes_cohort_per_round"].apply(
            lambda v: f"{v:,.0f}B" if pd.notna(v) else "—")
        for c in cw.columns:
            if c.startswith("wall_clock_"):
                cw[c] = cw[c].apply(lambda v: f"{v:.0f}s" if pd.notna(v) else "—")
        lines.append("### Per-round comm cost + wall-clock seconds per method\n")
        lines.append(cw.to_markdown(index=False))
        lines.append("")

    # Best method per (task, α) by AUROC
    pivot_auc = _pivot(df, "auc")
    if not pivot_auc.empty:
        lines.append("## 6. Best method per (task × α) by macro AUROC\n")
        best_rows = []
        method_cols = [c for c in pivot_auc.columns if c not in ("task", "alpha")]
        for _, row in pivot_auc.iterrows():
            method_aucs = {m: row[m] for m in method_cols if pd.notna(row[m])}
            if not method_aucs:
                continue
            best_method = max(method_aucs, key=method_aucs.get)
            best_rows.append({"task": row["task"], "alpha": row["alpha"],
                              "best_method": best_method,
                              "best_auc": f"{method_aucs[best_method]:.3f}"})
        lines.append(pd.DataFrame(best_rows).to_markdown(index=False))
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
