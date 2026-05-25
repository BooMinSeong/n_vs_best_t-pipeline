"""Bar-chart versions of difficulty_gap_vs_oracle and difficulty_accuracy plots.

For each model with mathfull data at N=2048, generates:
- difficulty_gap_vs_oracle_mathfull_N2048_{model}_bar.png   (ΔAcc vs oracle-T)
- difficulty_accuracy_mathfull_N2048_{model}_bar.png         (raw accuracy)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_TABLE = Path("outputs/n_vs_best_t/best_t_table.csv")
DEFAULT_FIGS = Path("outputs/n_vs_best_t/figs")

STRATA = ["L1", "L2", "L3", "L4", "L5"]

BASELINES = [
    ("acc_t0p1_mean",            "T=0.1"),
    ("acc_t1p0_mean",            "T=1.0"),
    ("acc_random_t_mean",        "random_T"),
    ("acc_equal_mix_mean",       "Temperature Pool"),
    ("acc_consensus_vote_mean",  "Temperature Consensus"),
]

COLORS = {
    "T=0.1":           "#2ca02c",
    "T=1.0":           "#ff7f0e",
    "random_T":        "#1f77b4",
    "Temperature Pool":       "#9467bd",
    "Temperature Consensus":  "#d62728",
    "oracle-T":        "#333333",
}

FIG_DPI = 200


def plot_gap_bar(bt: pd.DataFrame, model: str, dataset: str, N: int,
                 out_path: Path) -> None:
    """Bar chart: ΔAcc vs oracle-T (pp) per difficulty level."""
    sub = bt[(bt.model == model) & (bt.dataset == dataset) & (bt.N == N)]
    levels = [s for s in STRATA if not sub[sub.stratum == s].empty]
    if not levels:
        return

    x = np.arange(len(levels))
    n_bars = len(BASELINES)
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, (col, label) in enumerate(BASELINES):
        gaps = []
        for lv in levels:
            row = sub[sub.stratum == lv]
            if row.empty or col not in row.columns:
                gaps.append(0)
            else:
                r = row.iloc[0]
                gaps.append((r[col] - r.best_t_mean) * 100)
        offset = (i - n_bars / 2 + 0.5) * width
        ax.bar(x + offset, gaps, width, label=label, color=COLORS[label],
               edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="black", lw=1.0, ls="--", label="oracle-T")
    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=11)
    ax.set_xlabel("Difficulty Level", fontsize=12)
    ax.set_ylabel("ΔAcc vs oracle-T (pp)", fontsize=12)
    ax.set_title(f"Accuracy by Difficulty", fontsize=13)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def plot_accuracy_bar(bt: pd.DataFrame, model: str, dataset: str, N: int,
                      out_path: Path, show_oracle: bool = True) -> None:
    """Bar chart: raw accuracy (%) per difficulty level."""
    sub = bt[(bt.model == model) & (bt.dataset == dataset) & (bt.N == N)]
    levels = [s for s in STRATA if not sub[sub.stratum == s].empty]
    if not levels:
        return

    if show_oracle:
        all_series = [("best_t_mean", "oracle-T")] + BASELINES
    else:
        all_series = list(BASELINES)
    x = np.arange(len(levels))
    n_bars = len(all_series)
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, (col, label) in enumerate(all_series):
        vals = []
        for lv in levels:
            row = sub[sub.stratum == lv]
            if row.empty or col not in row.columns:
                vals.append(0)
            else:
                vals.append(row.iloc[0][col] * 100)
        offset = (i - n_bars / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=label, color=COLORS[label],
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=11)
    ax.set_xlabel("Difficulty Level", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Accuracy by Difficulty", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-t-table", type=Path, default=DEFAULT_TABLE)
    ap.add_argument("--figs-dir", type=Path, default=DEFAULT_FIGS)
    ap.add_argument("--dataset", default="mathfull")
    ap.add_argument("--N", type=int, default=2048)
    args = ap.parse_args()

    bt = pd.read_csv(args.best_t_table)
    args.figs_dir.mkdir(parents=True, exist_ok=True)

    sub = bt[(bt.dataset == args.dataset) & (bt.N == args.N)]
    models = sorted(sub.model.unique())

    for model in models:
        safe_model = model.replace(".", "_").replace("-", "_")
        gap_path = args.figs_dir / f"difficulty_gap_vs_oracle_{args.dataset}_N{args.N}_{safe_model}_bar.png"
        acc_path = args.figs_dir / f"difficulty_accuracy_{args.dataset}_N{args.N}_{safe_model}_bar.png"
        acc_no_oracle_path = args.figs_dir / f"difficulty_accuracy_{args.dataset}_N{args.N}_{safe_model}_bar_no_oracle.png"
        plot_gap_bar(bt, model, args.dataset, args.N, gap_path)
        plot_accuracy_bar(bt, model, args.dataset, args.N, acc_path, show_oracle=True)
        plot_accuracy_bar(bt, model, args.dataset, args.N, acc_no_oracle_path, show_oracle=False)
        print(f"  {model} → {gap_path.name}, {acc_path.name}, {acc_no_oracle_path.name}")

    print(f"\nDone. {len(models)} models rendered.")


if __name__ == "__main__":
    main()
