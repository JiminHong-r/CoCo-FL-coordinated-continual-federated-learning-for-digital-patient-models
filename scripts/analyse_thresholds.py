"""Spec WP5 axis 5: identify α thresholds beyond which methods degrade.

For each (task, method) in a WP5 results CSV, fit AUROC(α) and report:
  - best α
  - worst α
  - degradation_range = AUROC_max - AUROC_min
  - 'cliff' = largest single-step drop as α decreases (heterogeneity rises)

Usage::

    python scripts/analyse_thresholds.py --csv results_wp5_v2/wp5_results.csv \
                                          --out results_wp5_v2/threshold_analysis.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

METHODS = [
    "population_only", "local_finetune", "standard_fedavg",
    "top_down_only", "hierarchical_cohort",
]


def analyse_per_task(macro_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for task in sorted(macro_df["task"].unique()):
        sub = macro_df[macro_df["task"] == task].sort_values("alpha")
        for method in METHODS:
            col = f"auc_{method}"
            if col not in sub.columns:
                continue
            series = sub.set_index("alpha")[col].dropna()
            if series.empty:
                continue
            best_alpha = series.idxmax()
            worst_alpha = series.idxmin()
            best_auc = float(series.loc[best_alpha])
            worst_auc = float(series.loc[worst_alpha])
            # Largest drop as α DECREASES (i.e., heterogeneity grows). The
            # alphas are sorted ascending; we look at differences going
            # from higher α to lower α.
            sorted_alphas = sorted(series.index, reverse=True)
            cliff = 0.0
            cliff_at = None
            for i in range(len(sorted_alphas) - 1):
                a_hi, a_lo = sorted_alphas[i], sorted_alphas[i + 1]
                drop = float(series.loc[a_hi] - series.loc[a_lo])
                if drop > cliff:
                    cliff = drop
                    cliff_at = f"{a_hi}→{a_lo}"
            rows.append({
                "task": task, "method": method,
                "best_alpha": best_alpha, "best_auc": round(best_auc, 3),
                "worst_alpha": worst_alpha, "worst_auc": round(worst_auc, 3),
                "auc_range": round(best_auc - worst_auc, 3),
                "max_single_step_drop": round(cliff, 3),
                "drop_at_transition": cliff_at or "—",
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    macro = df[df["hospital"] == "MACRO"].copy()
    table = analyse_per_task(macro)

    lines = [f"# WP5 Axis 5 — Robustness to α heterogeneity\n",
             f"Source: `{args.csv}` ({len(macro)} macro rows)\n",
             "## Per-method α sensitivity\n",
             "Columns:\n",
             "- **auc_range** — max minus min AUROC across α values; larger = method is more α-sensitive.\n",
             "- **max_single_step_drop** — biggest AUROC drop in a single α transition "
             "(heterogeneity step). A 'cliff' near 0 means smooth degradation; large cliff = brittle.\n",
             "- **drop_at_transition** — the α pair where the drop happens.\n\n",
             table.to_markdown(index=False),
             "",
             "## Method ranking by α-robustness (smaller auc_range = more robust)\n",
             ]
    rank = table.groupby("method")["auc_range"].mean().sort_values()
    lines.append(rank.round(3).to_markdown(index=True, headers=["method", "mean_auc_range"]))
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
