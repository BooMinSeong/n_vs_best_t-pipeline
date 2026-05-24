"""Cross-comparison plots:
- D: heatmap T*(N=256) rows=model, cols=dataset (one per stratum)
- E: faceted grid 4 model × 5 dataset family, best-T trajectory overlay (overall only)
- F: win-margin scatter — x=(best-equal_mix)@N=16, y=same@N=256

Two versions: _with_fixed (includes best_fixed_t) and _no_fixed (without it).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_TABLE = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/best_t_table.csv")
DEFAULT_OUT = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/figs/cross")

DATASETS = ["math1k", "mathfull", "math500", "gsm8kfull", "aime"]
MODELS_ORDER = ["Qwen2.5-3B", "Qwen3-4B-Instruct-2507", "Phi-4-mini-instruct", "Llama-3.2-3B"]

FIG_DPI = 200


def plot_heatmap(bt: pd.DataFrame, stratum: str, N: int, out_path: Path,
                 use_fixed: bool = False) -> None:
    sub = bt[(bt.stratum == stratum) & (bt.N == N)]
    if sub.empty:
        return
    t_col = "best_fixed_t" if (use_fixed and "best_fixed_t" in sub.columns) else "t_star"
    acc_col = "best_fixed_t_mean" if (use_fixed and "best_fixed_t_mean" in sub.columns) else "best_t_mean"
    pivot_t = sub.pivot(index="model", columns="dataset", values=t_col)
    pivot_acc = sub.pivot(index="model", columns="dataset", values=acc_col)
    pivot_t = pivot_t.reindex(index=MODELS_ORDER, columns=DATASETS)
    pivot_acc = pivot_acc.reindex(index=MODELS_ORDER, columns=DATASETS)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot_t.values, cmap="viridis", vmin=0.1, vmax=1.2, aspect="auto")
    for i in range(pivot_t.shape[0]):
        for j in range(pivot_t.shape[1]):
            v = pivot_t.iat[i, j]
            a = pivot_acc.iat[i, j]
            if pd.isna(v):
                ax.text(j, i, "—", ha="center", va="center", color="white", fontsize=9)
            else:
                ax.text(j, i, f"T={v:.1f}\n{a*100:.1f}%",
                        ha="center", va="center",
                        color="white" if v > 0.7 else "black", fontsize=8)
    ax.set_xticks(range(len(DATASETS))); ax.set_xticklabels(DATASETS, rotation=20, ha="right")
    ax.set_yticks(range(len(MODELS_ORDER))); ax.set_yticklabels(MODELS_ORDER)
    t_label = "best-fixed-T" if use_fixed else "T*"
    ax.set_title(f"{t_label}  at N={N}, stratum={stratum}")
    plt.colorbar(im, ax=ax, label=t_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def plot_grid_trajectories(bt: pd.DataFrame, stratum: str, out_path: Path,
                           include_fixed: bool = False) -> None:
    sub = bt[bt.stratum == stratum]
    fig, axes = plt.subplots(len(MODELS_ORDER), len(DATASETS),
                             figsize=(18, 12), sharex=True, sharey=True)
    for i, model in enumerate(MODELS_ORDER):
        for j, dataset in enumerate(DATASETS):
            ax = axes[i, j]
            if i == 0:
                ax.set_title(dataset, fontsize=9)
            if j == 0:
                ax.set_ylabel(model.replace("Instruct-2507", "Inst"), fontsize=8)
            s = sub[(sub.model == model) & (sub.dataset == dataset)].sort_values("N")
            if s.empty:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=14)
                ax.set_xticks([])
                continue
            series = [("acc_t1p0_mean", "C3"), ("acc_t0p1_mean", "C2"),
                      ("acc_random_t_mean", "C1"), ("acc_equal_mix_mean", "C4")]
            if include_fixed:
                series.insert(0, ("best_fixed_t_mean", "C0"))
            else:
                series.insert(0, ("best_t_mean", "C0"))
            for col, color in series:
                if col in s.columns and not s[col].isna().all():
                    ax.plot(s["N"], s[col], marker=".", color=color, lw=1.2, ms=4)
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 16, 256]); ax.set_xticklabels([1, 16, 256], fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="y", labelsize=7)

    first_label = "best-fixed-T" if include_fixed else "best-T"
    handles = [plt.Line2D([], [], color=c, marker=".", label=lab) for lab, c in
               [(first_label, "C0"), ("T=1.0", "C3"), ("T=0.1", "C2"),
                ("random_T", "C1"), ("equal_mix", "C4")]]
    fig.legend(handles=handles, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"5-curve compare grid — stratum={stratum}", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_win_margin_scatter(bt: pd.DataFrame, out_path: Path) -> None:
    sub16 = bt[bt.N == 16][["model", "dataset", "stratum", "gap_vs_equal_mix"]].rename(
        columns={"gap_vs_equal_mix": "gap16"})
    sub256 = bt[bt.N == 256][["model", "dataset", "stratum", "gap_vs_equal_mix"]].rename(
        columns={"gap_vs_equal_mix": "gap256"})
    merged = sub16.merge(sub256, on=["model", "dataset", "stratum"])
    merged = merged.dropna(subset=["gap16", "gap256"])
    if merged.empty:
        print("No data for win-margin scatter — equal_mix likely missing.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    overall = merged[merged.stratum == "overall"]
    others = merged[merged.stratum != "overall"]
    ax.scatter(others.gap16 * 100, others.gap256 * 100, c="gray", alpha=0.5,
               s=30, label="per-level")
    ax.scatter(overall.gap16 * 100, overall.gap256 * 100, c="C0",
               s=80, label="overall", edgecolors="k")
    for _, r in overall.iterrows():
        ax.annotate(f"{r.model[:8]}/{r.dataset}",
                    (r.gap16 * 100, r.gap256 * 100),
                    fontsize=7, alpha=0.8, xytext=(3, 3),
                    textcoords="offset points")
    lim = max(abs(merged.gap16.min()), abs(merged.gap16.max()),
              abs(merged.gap256.min()), abs(merged.gap256.max())) * 100
    lim = max(lim, 1) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "--", color="gray", lw=0.7)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("best − equal_mix @ N=16 (pp)")
    ax.set_ylabel("best − equal_mix @ N=256 (pp)")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_title("best-T win margin: N=16 vs N=256")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-t-table", type=Path, default=DEFAULT_TABLE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bt = pd.read_csv(args.best_t_table)

    for stratum in ["overall", "L5"]:
        for N in [64, 256]:
            # no_fixed
            plot_heatmap(bt, stratum, N,
                         args.out_dir / f"D_heatmap_{stratum}_N{N}_no_fixed.png",
                         use_fixed=False)
            # with_fixed
            plot_heatmap(bt, stratum, N,
                         args.out_dir / f"D_heatmap_{stratum}_N{N}_with_fixed.png",
                         use_fixed=True)
    # E — two versions
    for stratum in ["overall", "L5"]:
        plot_grid_trajectories(bt, stratum,
                               args.out_dir / f"E_grid_{stratum}_no_fixed.png",
                               include_fixed=False)
        plot_grid_trajectories(bt, stratum,
                               args.out_dir / f"E_grid_{stratum}_with_fixed.png",
                               include_fixed=True)
    # F — single version (scatter is about gap, not T overlay)
    plot_win_margin_scatter(bt, args.out_dir / "F_win_margin_scatter.png")
    print(f"Cross plots → {args.out_dir}")


if __name__ == "__main__":
    main()
