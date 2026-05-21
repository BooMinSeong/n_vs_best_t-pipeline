"""Plot allocation strategy comparison.

For each stratum, line plot of 5 strategies (4 equal_mix alloc + T=1.0) over N.
Also gap-vs-T=1.0 plot for emphasis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_CSV = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/alloc_compare/math500_Phi.csv")
DEFAULT_OUT = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/alloc_compare")

EM_DEFS = [
    ("em_lowfirst",  "equal_mix low-T first (original)",  "C4", "-"),
    ("em_highfirst", "equal_mix high-T first",            "C3", "-"),
    ("em_random",    "equal_mix random rotation",         "C2", "-"),
    ("em_floor",     "equal_mix floor (N//12 ×12)",       "C1", "-"),
]
CV_DEFS = [
    ("cv_lowfirst",  "consensus_vote low-T first",        "C4", ":"),
    ("cv_highfirst", "consensus_vote high-T first",       "C3", ":"),
    ("cv_random",    "consensus_vote random rotation",    "C2", ":"),
    ("cv_floor",     "consensus_vote floor",              "C1", ":"),
]
T1_DEF = ("T1.0", "T=1.0 baseline", "black", "--")
STRATA = ["overall", "L1", "L2", "L3", "L4", "L5", "Lr"]


def _plot_one(ax, sub, defs_list, include_t1: bool, stratum: str, n_prob: int,
              ylabel: str, title_suffix: str, mark_lines: bool = True) -> None:
    for strat, label, color, ls in defs_list:
        s = sub[sub.strategy == strat].sort_values("N")
        if s.empty:
            continue
        ax.plot(s.N, s["mean"], marker="o", color=color, label=label,
                lw=1.6, ls=ls, ms=4)
        std = s["std6"].fillna(0)
        ax.fill_between(s.N, s["mean"] - std, s["mean"] + std,
                        color=color, alpha=0.10)
    if include_t1:
        s = sub[sub.strategy == "T1.0"].sort_values("N")
        if not s.empty:
            ax.plot(s.N, s["mean"], marker="s", color="black",
                    label="T=1.0 baseline", lw=2.0, ls="--", ms=4)
            std = s["std6"].fillna(0)
            ax.fill_between(s.N, s["mean"] - std, s["mean"] + std,
                            color="black", alpha=0.08)
    if mark_lines:
        for n_mult in [12, 24, 48, 96, 192, 384, 768, 1536]:
            ax.axvline(n_mult, color="gray", alpha=0.18, lw=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    ax.set_title(f"{title_suffix} [{stratum} n={n_prob}]")


def plot_strata(df: pd.DataFrame, out_dir: Path, combo: str) -> None:
    for stratum in STRATA:
        sub = df[df.stratum == stratum]
        if sub.empty:
            continue
        n_prob = int(sub["n_pids"].iloc[0])
        if n_prob == 0:
            continue

        # 2×2: top em variants, bottom cv variants, left raw, right gap
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex="col")

        # Top-left: em variants raw
        _plot_one(axes[0, 0], sub, EM_DEFS, include_t1=True, stratum=stratum,
                  n_prob=n_prob, ylabel="maj@N accuracy",
                  title_suffix=f"equal_mix variants — {combo}")
        # Bottom-left: cv variants raw
        _plot_one(axes[1, 0], sub, CV_DEFS, include_t1=True, stratum=stratum,
                  n_prob=n_prob, ylabel="maj@N accuracy",
                  title_suffix=f"consensus_vote variants — {combo}")

        # Right: gap vs T=1.0 (top em, bottom cv)
        t1 = sub[sub.strategy == "T1.0"].set_index("N")["mean"]
        for ax_idx, defs in [(0, EM_DEFS), (1, CV_DEFS)]:
            ax2 = axes[ax_idx, 1]
            for strat, label, color, ls in defs:
                s = sub[sub.strategy == strat].sort_values("N").set_index("N")
                if s.empty: continue
                common = s.index.intersection(t1.index)
                gap = (s.loc[common, "mean"] - t1.loc[common]) * 100
                ax2.plot(common, gap, marker="o", color=color, label=label,
                          lw=1.6, ls=ls, ms=4)
            ax2.axhline(0, color="black", lw=0.7, ls="--", label="T=1.0 (=0)")
            for n_mult in [12, 24, 48, 96, 192, 384, 768, 1536]:
                ax2.axvline(n_mult, color="gray", alpha=0.18, lw=0.5)
            ax2.set_xscale("log", base=2)
            ax2.set_xlabel("N")
            ax2.set_ylabel("ΔAcc vs T=1.0 (pp)")
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=7, loc="best")
            prefix = "equal_mix" if ax_idx == 0 else "consensus_vote"
            ax2.set_title(f"{prefix} gap vs T=1.0 [{stratum}]")

        fig.suptitle(f"alloc compare 4 variants × 2 agg — {combo} [{stratum}]",
                     y=1.0, fontsize=12)
        fig.tight_layout()
        out_path = out_dir / f"alloc_compare_{stratum}.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    combo = df["combo"].iloc[0]
    print(f"plotting {combo} — {len(df)} rows")
    plot_strata(df, args.out_dir, combo)


if __name__ == "__main__":
    main()
