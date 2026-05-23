"""Per-combo plots:
- A: T*(N) trajectory (6 lines: overall + L1..L5)
- B: 5-curve comparison (best-T, T=1.0, T=0.1, random_T, equal_mix)
- C: gap vs baseline (best - baseline for each of 4 baselines)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_TABLE = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/best_t_table.csv")
DEFAULT_LONG = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/long/long_all.csv")
DEFAULT_FIGS = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/figs")

STRATA = ["overall", "L1", "L2", "L3", "L4", "L5", "Lr"]
STRATUM_COLORS = {
    "overall": "black", "L1": "#1f77b4", "L2": "#2ca02c",
    "L3": "#bcbd22", "L4": "#ff7f0e", "L5": "#d62728",
    "Lr": "#9467bd",  # recoverable (purple)
}


def plot_t_star_trajectory(sub: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for stratum in STRATA:
        s = sub[sub.stratum == stratum].sort_values("N")
        if s.empty:
            continue
        # mark sparse (n_problems < 5) with dashed
        n_prob = int(s["level_n_problems"].iloc[0])
        lw = 2.0
        ls = "--" if n_prob < 5 else "-"
        alpha = 0.5 if n_prob < 5 else 1.0
        ax.plot(s["N"], s["t_star"], marker="o", ls=ls, lw=lw,
                color=STRATUM_COLORS[stratum], alpha=alpha,
                label=f"{stratum} (n={n_prob})")
        # CI band (optional — may not exist after sim-only re-merge)
        if "t_star_ci_low" in s.columns and not s["t_star_ci_low"].isna().all():
            ax.fill_between(s["N"], s["t_star_ci_low"], s["t_star_ci_high"],
                            color=STRATUM_COLORS[stratum], alpha=0.1)
    ax.set_xscale("log", base=2)
    n_ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
    ax.set_xticks(n_ticks)
    ax.set_xticklabels(n_ticks, fontsize=8)
    ax.set_xlabel("N (sample budget, log scale)")
    ax.set_ylabel("T*  (arg max acc over T)")
    ax.set_ylim(0.0, 1.3)
    ax.set_yticks(np.arange(0.1, 1.3, 0.1))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    model = sub["model"].iloc[0]; dataset = sub["dataset"].iloc[0]
    ax.set_title(f"T*(N) trajectory — {model} / {dataset}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_5_curves(sub_strat: pd.DataFrame, stratum: str, out_path: Path) -> None:
    """sub_strat must be already filtered to one stratum, one (model, dataset)."""
    s = sub_strat.sort_values("N")
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    series_defs = [
        ("best_t_mean",          "best-T",         "C0",  "best_t_seed_std"),
        ("acc_t1p0_mean",        "T=1.0",          "C3",  "acc_t1p0_std"),
        ("acc_t0p1_mean",        "T=0.1",          "C2",  "acc_t0p1_std"),
        ("acc_random_t_mean",    "random_T",       "C1",  "acc_random_t_std"),
        ("acc_equal_mix_mean",   "equal_mix",      "C4",  "acc_equal_mix_std"),
        ("acc_consensus_vote_mean","consensus_vote","C6", "acc_consensus_vote_std"),
    ]
    for col, label, color, std_col in series_defs:
        if col not in s.columns or s[col].isna().all():
            continue
        ax.plot(s["N"], s[col], marker="o", color=color, label=label, lw=1.8)
        if std_col in s.columns:
            std = s[std_col].fillna(0.0)
            ax.fill_between(s["N"], s[col] - std, s[col] + std, color=color, alpha=0.12)

    ax.set_xscale("log", base=2)
    n_ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
    ax.set_xticks(n_ticks)
    ax.set_xticklabels(n_ticks, fontsize=8)
    ax.set_xlabel("N")
    ax.set_ylabel("maj@N accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    model = s["model"].iloc[0]; dataset = s["dataset"].iloc[0]
    n_prob = int(s["level_n_problems"].iloc[0])
    ax.set_title(f"5-curve compare — {model}/{dataset} [{stratum} n={n_prob}]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_5_curves_zoom(sub_strat: pd.DataFrame, stratum: str, out_path: Path,
                        n_min: int = 64) -> None:
    """Zoomed Plot B — N>=n_min, y-axis tight around top competitive baselines.

    Same series as plot_5_curves, but x-range starts at n_min and y-range auto-fits to
    {best-T, equal_mix, consensus_vote} band ±1pp so close differences are visible.
    """
    s = sub_strat[sub_strat.N >= n_min].sort_values("N")
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    series_defs = [
        ("best_t_mean",          "best-T",         "C0",  "best_t_seed_std"),
        ("acc_t1p0_mean",        "T=1.0",          "C3",  "acc_t1p0_std"),
        ("acc_t0p1_mean",        "T=0.1",          "C2",  "acc_t0p1_std"),
        ("acc_random_t_mean",    "random_T",       "C1",  "acc_random_t_std"),
        ("acc_equal_mix_mean",   "equal_mix",      "C4",  "acc_equal_mix_std"),
        ("acc_consensus_vote_mean","consensus_vote","C6", "acc_consensus_vote_std"),
    ]
    # Determine zoom y-range from the TOP competitive baselines only
    zoom_cols = ["best_t_mean", "acc_equal_mix_mean", "acc_consensus_vote_mean"]
    vals = []
    for c in zoom_cols:
        if c in s.columns:
            vals.extend(s[c].dropna().tolist())
    if vals:
        y_lo = max(0.0, min(vals) - 0.01)
        y_hi = min(1.0, max(vals) + 0.005)
        # Add a small margin
        margin = max(0.005, (y_hi - y_lo) * 0.1)
        y_lo -= margin; y_hi += margin
    else:
        y_lo, y_hi = 0.0, 1.0

    for col, label, color, std_col in series_defs:
        if col not in s.columns or s[col].isna().all():
            continue
        ax.plot(s["N"], s[col], marker="o", color=color, label=label, lw=1.8)
        if std_col in s.columns:
            std = s[std_col].fillna(0.0)
            ax.fill_between(s["N"], s[col] - std, s[col] + std, color=color, alpha=0.12)

    ax.set_xscale("log", base=2)
    n_ticks = [n for n in [64, 128, 256, 512, 1024, 1536, 2048] if n >= n_min]
    ax.set_xticks(n_ticks)
    ax.set_xticklabels(n_ticks, fontsize=8)
    ax.set_xlabel(f"N (zoom: N≥{n_min})")
    ax.set_ylabel("maj@N accuracy")
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    model = s["model"].iloc[0]; dataset = s["dataset"].iloc[0]
    n_prob = int(s["level_n_problems"].iloc[0])
    ax.set_title(f"5-curve compare ZOOM N≥{n_min} — {model}/{dataset} [{stratum} n={n_prob}]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_gap_vs_baseline(sub_strat: pd.DataFrame, stratum: str, out_path: Path) -> None:
    s = sub_strat.sort_values("N")
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    gap_defs = [
        ("gap_vs_t1p0",           "best − T=1.0",           "C3"),
        ("gap_vs_t0p1",           "best − T=0.1",           "C2"),
        ("gap_vs_random_t",       "best − random_T",        "C1"),
        ("gap_vs_equal_mix",      "best − equal_mix",       "C4"),
        ("gap_vs_consensus_vote", "best − consensus_vote",  "C6"),
    ]
    for col, label, color in gap_defs:
        if col not in s.columns or s[col].isna().all():
            continue
        ax.plot(s["N"], s[col] * 100, marker="o", color=color, label=label, lw=1.8)
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xscale("log", base=2)
    n_ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
    ax.set_xticks(n_ticks)
    ax.set_xticklabels(n_ticks, fontsize=8)
    ax.set_xlabel("N")
    ax.set_ylabel("ΔAcc (pp)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    model = s["model"].iloc[0]; dataset = s["dataset"].iloc[0]
    n_prob = int(s["level_n_problems"].iloc[0])
    ax.set_title(f"best-T gap vs baselines — {model}/{dataset} [{stratum} n={n_prob}]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-t-table", type=Path, default=DEFAULT_TABLE)
    ap.add_argument("--figs-dir", type=Path, default=DEFAULT_FIGS)
    ap.add_argument("--strata-for-bc", default="overall,L1,L2,L3,L4,L5,Lr",
                    help="strata to render Plot B/C for (comma-separated)")
    args = ap.parse_args()

    bt = pd.read_csv(args.best_t_table)
    strata_bc = args.strata_for_bc.split(",")
    args.figs_dir.mkdir(parents=True, exist_ok=True)

    for (model, dataset), sub in bt.groupby(["model", "dataset"]):
        combo_dir = args.figs_dir / f"{model}__{dataset}"
        combo_dir.mkdir(exist_ok=True)
        # Plot A
        plot_t_star_trajectory(sub, combo_dir / "A_t_star_trajectory.png")
        # Plot B & C per stratum
        for stratum in strata_bc:
            sub_s = sub[sub.stratum == stratum]
            if sub_s.empty:
                continue
            plot_5_curves(sub_s, stratum, combo_dir / f"B_5curve_{stratum}.png")
            plot_5_curves_zoom(sub_s, stratum, combo_dir / f"B_5curve_{stratum}_zoom.png",
                                n_min=64)
            plot_5_curves_zoom(sub_s, stratum, combo_dir / f"B_5curve_{stratum}_zoom128.png",
                                n_min=128)
            plot_gap_vs_baseline(sub_s, stratum, combo_dir / f"C_gap_{stratum}.png")
        print(f"  rendered {model}/{dataset} → {combo_dir}")

    print(f"\nDone. {args.figs_dir}")


if __name__ == "__main__":
    main()
