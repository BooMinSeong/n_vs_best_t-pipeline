"""Per-combo plots:
- A: T*(N) trajectory (6 lines: overall + L1..L5)
- B: 5-curve comparison (best-T, T=1.0, T=0.1, random_T, equal_mix)
- C: gap vs baseline (best - baseline for each of 4 baselines)

Two versions: _with_fixed (includes best_fixed_t) and _no_fixed (without it).
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

FIG_DPI = 200


def plot_t_star_trajectory(sub: pd.DataFrame, out_path: Path, show_fixed: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    for stratum in STRATA:
        s = sub[sub.stratum == stratum].sort_values("N")
        if s.empty:
            continue
        n_prob = int(s["level_n_problems"].iloc[0])
        lw = 2.0
        ls = "--" if n_prob < 5 else "-"
        alpha = 0.5 if n_prob < 5 else 1.0
        ax.plot(s["N"], s["t_star"], marker="o", ls=ls, lw=lw,
                color=STRATUM_COLORS[stratum], alpha=alpha,
                label=stratum)
        # CI band
        if "t_star_ci_low" in s.columns and not s["t_star_ci_low"].isna().all():
            ax.fill_between(s["N"], s["t_star_ci_low"], s["t_star_ci_high"],
                            color=STRATUM_COLORS[stratum], alpha=0.1)
        # best_fixed_t horizontal line
        if show_fixed and "best_fixed_t" in s.columns and not s["best_fixed_t"].isna().all():
            ft = s["best_fixed_t"].iloc[0]
            ax.axhline(ft, color=STRATUM_COLORS[stratum], ls=":", lw=1.5, alpha=0.6)
    ax.set_xscale("log", base=2)
    n_ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    ax.set_xticks(n_ticks)
    ax.set_xticklabels(n_ticks, fontsize=8)
    ax.set_xlabel("N (sample budget, log scale)")
    ax.set_ylabel("T*  (arg max acc over T)")
    ax.set_ylim(0.0, 1.3)
    ax.set_yticks(np.arange(0.1, 1.3, 0.1))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    model = sub["model"].iloc[0]; dataset = sub["dataset"].iloc[0]
    ax.set_title("T*(N) trajectory")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def _5curve_series(include_fixed: bool):
    """Return series definitions for 5-curve plot."""
    defs = []
    if include_fixed:
        defs.append(("best_fixed_t_mean", "best-fixed-T", "C0", "best_fixed_t_std"))
    defs.extend([
        ("acc_t1p0_mean",        "T=1.0",          "C3",  "acc_t1p0_std"),
        ("acc_t0p1_mean",        "T=0.1",          "C2",  "acc_t0p1_std"),
        ("acc_random_t_mean",    "random_T",       "C1",  "acc_random_t_std"),
        ("acc_equal_mix_mean",   "Temperature Pool",      "C4",  "acc_equal_mix_std"),
        ("acc_consensus_vote_mean","Temperature Consensus","C6", "acc_consensus_vote_std"),
    ])
    return defs


def plot_5_curves(sub_strat: pd.DataFrame, stratum: str, out_path: Path,
                  include_fixed: bool = True, show_band: bool = True) -> None:
    s = sub_strat.sort_values("N")
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    for col, label, color, std_col in _5curve_series(include_fixed):
        if col not in s.columns or s[col].isna().all():
            continue
        ax.plot(s["N"], s[col], marker="o", color=color, label=label, lw=1.8)
        if show_band and std_col in s.columns:
            std = s[std_col].fillna(0.0)
            ax.fill_between(s["N"], s[col] - std, s[col] + std, color=color, alpha=0.12)

    ax.set_xscale("log", base=2)
    n_ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    ax.set_xticks(n_ticks)
    ax.set_xticklabels(n_ticks, fontsize=8)
    ax.set_xlabel("N")
    ax.set_ylabel("maj@N accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    model = s["model"].iloc[0]; dataset = s["dataset"].iloc[0]
    ax.set_title(f"maj@N — {stratum}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def plot_5_curves_zoom(sub_strat: pd.DataFrame, stratum: str, out_path: Path,
                        n_min: int = 64, include_fixed: bool = True,
                        show_band: bool = True) -> None:
    s = sub_strat[sub_strat.N >= n_min].sort_values("N")
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 7))

    # Determine zoom y-range from the TOP competitive baselines only
    zoom_cols = ["acc_equal_mix_mean", "acc_consensus_vote_mean"]
    if include_fixed:
        zoom_cols.insert(0, "best_fixed_t_mean")
    vals = []
    for c in zoom_cols:
        if c in s.columns:
            vals.extend(s[c].dropna().tolist())
    if vals:
        y_lo = max(0.0, min(vals) - 0.01)
        y_hi = min(1.0, max(vals) + 0.005)
        margin = max(0.005, (y_hi - y_lo) * 0.1)
        y_lo -= margin; y_hi += margin
    else:
        y_lo, y_hi = 0.0, 1.0

    for col, label, color, std_col in _5curve_series(include_fixed):
        if col not in s.columns or s[col].isna().all():
            continue
        ax.plot(s["N"], s[col], marker="o", color=color, label=label, lw=1.8)
        if show_band and std_col in s.columns:
            std = s[std_col].fillna(0.0)
            ax.fill_between(s["N"], s[col] - std, s[col] + std, color=color, alpha=0.12)

    ax.set_xscale("log", base=2)
    n_ticks = [n for n in [64, 128, 256, 512, 1024, 2048] if n >= n_min]
    ax.set_xticks(n_ticks)
    ax.set_xticklabels(n_ticks, fontsize=8)
    ax.set_xlabel(f"N (zoom: N≥{n_min})")
    ax.set_ylabel("maj@N accuracy")
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    model = s["model"].iloc[0]; dataset = s["dataset"].iloc[0]
    ax.set_title(f"ZOOM N≥{n_min} — {stratum}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def _gap_defs(use_fixed: bool):
    """Return gap definitions. If use_fixed, use gap_fixed_vs_* columns."""
    prefix = "gap_fixed_vs_" if use_fixed else "gap_vs_"
    label_src = "best-fixed-T" if use_fixed else "best"
    return [
        (f"{prefix}t1p0",           f"{label_src} − T=1.0",           "C3"),
        (f"{prefix}t0p1",           f"{label_src} − T=0.1",           "C2"),
        (f"{prefix}random_t",       f"{label_src} − random_T",        "C1"),
        (f"{prefix}equal_mix",      f"{label_src} − Temperature Pool",       "C4"),
        (f"{prefix}consensus_vote", f"{label_src} − Temperature Consensus",  "C6"),
    ]


def plot_gap_vs_baseline(sub_strat: pd.DataFrame, stratum: str, out_path: Path,
                         use_fixed: bool = True) -> None:
    s = sub_strat.sort_values("N")
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    for col, label, color in _gap_defs(use_fixed):
        if col not in s.columns or s[col].isna().all():
            continue
        ax.plot(s["N"], s[col] * 100, marker="o", color=color, label=label, lw=1.8)
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xscale("log", base=2)
    n_ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    ax.set_xticks(n_ticks)
    ax.set_xticklabels(n_ticks, fontsize=8)
    ax.set_xlabel("N")
    ax.set_ylabel("ΔAcc (pp)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    model = s["model"].iloc[0]; dataset = s["dataset"].iloc[0]
    title_src = "best-fixed-T" if use_fixed else "best-T"
    ax.set_title(f"{title_src} gap vs baselines — {stratum}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
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
        # Plot A — two versions
        plot_t_star_trajectory(sub, combo_dir / "A_t_star_trajectory_no_fixed.png",
                               show_fixed=False)
        plot_t_star_trajectory(sub, combo_dir / "A_t_star_trajectory_with_fixed.png",
                               show_fixed=True)
        # Plot B & C per stratum — two versions each
        for stratum in strata_bc:
            sub_s = sub[sub.stratum == stratum]
            if sub_s.empty:
                continue
            # B — both with-band (default) and no-band (_no_band suffix) variants
            for show_band, band_sfx in [(True, ""), (False, "_no_band")]:
                # B — no fixed
                plot_5_curves(sub_s, stratum,
                              combo_dir / f"B_5curve_{stratum}_no_fixed{band_sfx}.png",
                              include_fixed=False, show_band=show_band)
                plot_5_curves_zoom(sub_s, stratum,
                                    combo_dir / f"B_5curve_{stratum}_zoom_no_fixed{band_sfx}.png",
                                    n_min=64, include_fixed=False, show_band=show_band)
                plot_5_curves_zoom(sub_s, stratum,
                                    combo_dir / f"B_5curve_{stratum}_zoom128_no_fixed{band_sfx}.png",
                                    n_min=128, include_fixed=False, show_band=show_band)
                # B — with fixed
                plot_5_curves(sub_s, stratum,
                              combo_dir / f"B_5curve_{stratum}_with_fixed{band_sfx}.png",
                              include_fixed=True, show_band=show_band)
                plot_5_curves_zoom(sub_s, stratum,
                                    combo_dir / f"B_5curve_{stratum}_zoom_with_fixed{band_sfx}.png",
                                    n_min=64, include_fixed=True, show_band=show_band)
                plot_5_curves_zoom(sub_s, stratum,
                                    combo_dir / f"B_5curve_{stratum}_zoom128_with_fixed{band_sfx}.png",
                                    n_min=128, include_fixed=True, show_band=show_band)
            # C — no fixed (uses original gap_vs_*)
            plot_gap_vs_baseline(sub_s, stratum,
                                  combo_dir / f"C_gap_{stratum}_no_fixed.png",
                                  use_fixed=False)
            # C — with fixed (uses gap_fixed_vs_*)
            plot_gap_vs_baseline(sub_s, stratum,
                                  combo_dir / f"C_gap_{stratum}_with_fixed.png",
                                  use_fixed=True)
        print(f"  rendered {model}/{dataset} → {combo_dir}")

    print(f"\nDone. {args.figs_dir}")


if __name__ == "__main__":
    main()
