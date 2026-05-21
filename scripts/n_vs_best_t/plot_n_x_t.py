"""N × T plots — the central object of this analysis.

Per (model, dataset, stratum):
- G. acc(T, N) heatmap with T*(N) trajectory overlaid (white line + dots)
- H. regret heatmap: acc(T,N) − max_T acc(T,N) (shows how costly suboptimal T is at each N)

Cross-combo:
- I. 9-panel small-multiples grid: heatmap per combo with T*(N) overlay (overall stratum)

Source: long_all.csv (raw per-(T,N) accuracy) + best_t_table.csv (T*(N)).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LONG_DEFAULT = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/long/long_all.csv")
SIM_EXT_DEFAULT = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/sim_extended_all.csv")
TABLE_DEFAULT = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/best_t_table.csv")
FIGS_DEFAULT = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/figs")

N_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048]
T_GRID = [round(0.1 * i, 1) for i in range(1, 13)]

MODELS_ORDER = ["Qwen2.5-3B", "Qwen3-4B-Instruct-2507", "Phi-4-mini-instruct", "Llama-3.2-3B"]
DATASETS_ORDER = ["math1k", "mathfull", "math500", "gsm8kfull", "aime"]


def get_acc_grid(long: pd.DataFrame, model: str, dataset: str, stratum: str) -> np.ndarray | None:
    """Return (n_T, n_N) accuracy matrix; None if missing.

    `long` must have columns (model, dataset, stratum, N, T, mean_acc).
    If sim source was loaded, single-T baseline rows have been remapped to T column.
    """
    sub = long[(long.model == model) & (long.dataset == dataset) & (long.stratum == stratum)]
    if sub.empty:
        return None
    pv = sub.pivot_table(index="T", columns="N", values="mean_acc")
    pv = pv.reindex(index=T_GRID, columns=N_GRID)
    if pv.isna().all().all():
        return None
    return pv.values  # rows=T, cols=N


def load_acc_long(long_path: Path, sim_ext_path: Path | None) -> pd.DataFrame:
    """Prefer sim_extended (covers N up to 1536, all single-T). Fallback to markdown long_all."""
    if sim_ext_path is not None and sim_ext_path.exists():
        sim = pd.read_csv(sim_ext_path)
        # Filter to single-T baselines only
        is_single = sim["baseline"].str.match(r"^T\d+\.\d+$")
        sim_t = sim[is_single].copy()
        sim_t["T"] = sim_t["baseline"].str[1:].astype(float)
        sim_t = sim_t.rename(columns={"mean": "mean_acc"})
        sim_t["source"] = "sim"
        return sim_t[["model", "dataset", "stratum", "N", "T", "mean_acc", "source"]]
    # Fallback: markdown
    return pd.read_csv(long_path)


def plot_heatmap_with_tstar(acc: np.ndarray, t_star_by_n: dict[int, float],
                             title: str, out_path: Path, regret: bool = False) -> None:
    """acc shape (n_T, n_N). If regret=True, plot acc − max over T per column (≤ 0)."""
    if regret:
        col_max = np.nanmax(acc, axis=0, keepdims=True)
        z = (acc - col_max) * 100  # pp loss, ≤ 0
        cmap = "Reds_r"
        vmin, vmax = -20, 0
        cbar_label = "ΔAcc vs best-T (pp, ≤0)"
    else:
        z = acc * 100
        cmap = "viridis"
        vmin = np.nanmin(z); vmax = np.nanmax(z)
        cbar_label = "maj@N accuracy (%)"

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(z, aspect="auto", origin="lower", cmap=cmap,
                   vmin=vmin, vmax=vmax,
                   extent=[-0.5, len(N_GRID) - 0.5, -0.5, len(T_GRID) - 0.5])
    # ticks
    ax.set_xticks(range(len(N_GRID))); ax.set_xticklabels(N_GRID)
    ax.set_yticks(range(len(T_GRID))); ax.set_yticklabels([f"{t:.1f}" for t in T_GRID])
    ax.set_xlabel("N (sample budget)")
    ax.set_ylabel("T (temperature)")

    # T*(N) overlay
    xs = []; ys = []
    for i, n in enumerate(N_GRID):
        t = t_star_by_n.get(n)
        if t is None:
            continue
        xs.append(i)
        # map t to y-index
        y_idx = int(round((t - 0.1) * 10))
        ys.append(y_idx)
    ax.plot(xs, ys, "o-", color="white", lw=2.0, ms=7,
            markeredgecolor="black", markerfacecolor="white", label="T*(N)")
    # annotate values
    for x, y in zip(xs, ys):
        v = z[y, x]
        txt = f"{v:.1f}" if regret else f"{v:.0f}"
        ax.text(x, y - 0.35, txt, ha="center", va="top",
                color="white", fontsize=7, fontweight="bold")

    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def grid_of_heatmaps(long: pd.DataFrame, bt: pd.DataFrame, stratum: str,
                     out_path: Path, regret: bool = False) -> None:
    """4×5 grid (model × dataset) of N×T heatmaps with T*(N) overlay, single stratum."""
    fig, axes = plt.subplots(len(MODELS_ORDER), len(DATASETS_ORDER),
                             figsize=(16, 11), sharex=True, sharey=True)
    cmap = "Reds_r" if regret else "viridis"
    vmin = -15 if regret else 0
    vmax = 0 if regret else 100

    last_im = None
    for i, model in enumerate(MODELS_ORDER):
        for j, dataset in enumerate(DATASETS_ORDER):
            ax = axes[i, j]
            if i == 0: ax.set_title(dataset, fontsize=10)
            if j == 0: ax.set_ylabel(model.replace("Instruct-2507", "Inst"), fontsize=9)
            acc = get_acc_grid(long, model, dataset, stratum)
            if acc is None:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=20)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            if regret:
                col_max = np.nanmax(acc, axis=0, keepdims=True)
                z = (acc - col_max) * 100
            else:
                z = acc * 100
            im = ax.imshow(z, aspect="auto", origin="lower", cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           extent=[-0.5, len(N_GRID) - 0.5,
                                   -0.5, len(T_GRID) - 0.5])
            last_im = im
            # T*(N) overlay
            sub_bt = bt[(bt.model == model) & (bt.dataset == dataset) & (bt.stratum == stratum)]
            xs = []; ys = []
            for k, n in enumerate(N_GRID):
                row = sub_bt[sub_bt.N == n]
                if row.empty:
                    continue
                t = float(row.iloc[0].t_star)
                xs.append(k); ys.append(int(round((t - 0.1) * 10)))
            ax.plot(xs, ys, "o-", color="white", lw=1.2, ms=4,
                    markeredgecolor="black", markerfacecolor="white")
            ax.set_xticks([0, 4, 8]); ax.set_xticklabels(["1", "16", "256"], fontsize=7)
            ax.set_yticks([0, 5, 11]); ax.set_yticklabels(["0.1", "0.6", "1.2"], fontsize=7)

    cbar_label = "ΔAcc vs best-T (pp)" if regret else "acc (%)"
    title = f"Regret heatmap (acc − max_T acc) grid — stratum={stratum}" if regret \
        else f"acc(T, N) heatmap grid — stratum={stratum}"
    fig.suptitle(title, y=1.0, fontsize=12)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), label=cbar_label,
                     shrink=0.6, pad=0.02)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_combo_by_difficulty(long: pd.DataFrame, bt: pd.DataFrame,
                              model: str, dataset: str, out_path: Path,
                              regret: bool = False) -> None:
    """For ONE (model, dataset), show 6 N×T heatmaps (overall, L1..L5) side-by-side.

    Each panel shows acc(T,N) (or regret) with T*(N) overlay.
    Reveals how the optimal-T landscape changes with difficulty.
    """
    strata = ["overall", "L1", "L2", "L3", "L4", "L5", "Lr"]
    fig, axes = plt.subplots(1, 7, figsize=(21, 4.5), sharex=True, sharey=True)
    cmap = "Reds_r" if regret else "viridis"
    vmin = -15 if regret else 0
    vmax = 0 if regret else 100

    last_im = None
    for ax, stratum in zip(axes, strata):
        acc = get_acc_grid(long, model, dataset, stratum)
        sub_bt = bt[(bt.model == model) & (bt.dataset == dataset) & (bt.stratum == stratum)]
        n_prob = int(sub_bt.iloc[0].level_n_problems) if len(sub_bt) else 0
        ax.set_title(f"{stratum} (n={n_prob})", fontsize=10)
        if acc is None or sub_bt.empty:
            ax.text(0.5, 0.5, "—", ha="center", va="center",
                    transform=ax.transAxes, color="gray", fontsize=20)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        if regret:
            col_max = np.nanmax(acc, axis=0, keepdims=True)
            z = (acc - col_max) * 100
        else:
            z = acc * 100
        im = ax.imshow(z, aspect="auto", origin="lower", cmap=cmap,
                       vmin=vmin, vmax=vmax,
                       extent=[-0.5, len(N_GRID) - 0.5, -0.5, len(T_GRID) - 0.5])
        last_im = im
        # T*(N) overlay
        xs = []; ys = []
        for k, n in enumerate(N_GRID):
            row = sub_bt[sub_bt.N == n]
            if row.empty:
                continue
            t = float(row.iloc[0].t_star)
            xs.append(k); ys.append(int(round((t - 0.1) * 10)))
        ax.plot(xs, ys, "o-", color="white", lw=1.5, ms=5,
                markeredgecolor="black", markerfacecolor="white")
        ax.set_xticks([0, 4, 8]); ax.set_xticklabels(["1", "16", "256"], fontsize=8)
        ax.set_yticks([0, 5, 11]); ax.set_yticklabels(["0.1", "0.6", "1.2"], fontsize=8)
        ax.set_xlabel("N", fontsize=9)

    axes[0].set_ylabel("T", fontsize=9)
    cbar_label = "ΔAcc vs best-T (pp)" if regret else "acc (%)"
    suffix = " — regret" if regret else ""
    fig.suptitle(f"{model} / {dataset}: N×T landscape by difficulty{suffix}",
                 fontsize=12, y=1.02)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.tolist(), label=cbar_label, shrink=0.8, pad=0.01)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_t_star_all_combo_by_stratum(bt: pd.DataFrame, out_path: Path) -> None:
    """6 subplots (overall + L1..L5), each shows T*(N) line per combo.

    Reveals whether all combos converge on similar T*(N) when stratified by difficulty.
    """
    strata = ["overall", "L1", "L2", "L3", "L4", "L5", "Lr"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    # hide unused 8th axis
    if len(strata) < len(axes):
        for ax in axes[len(strata):]:
            ax.axis("off")
    axes = axes[:len(strata)]
    combos = sorted(bt.groupby(["model", "dataset"]).groups.keys())
    cmap = plt.cm.tab10

    for ax, stratum in zip(axes, strata):
        for idx, (model, dataset) in enumerate(combos):
            s = bt[(bt.model == model) & (bt.dataset == dataset) & (bt.stratum == stratum)]
            if s.empty:
                continue
            s = s.sort_values("N")
            n_prob = int(s["level_n_problems"].iloc[0])
            lw = 1.8 if n_prob >= 5 else 0.8
            ls = "-" if n_prob >= 5 else ":"
            label = f"{model[:8]}/{dataset}" + (f" (n={n_prob})" if n_prob < 5 else "")
            ax.plot(s["N"], s["t_star"], marker="o", lw=lw, ls=ls,
                    color=cmap(idx % 10), label=label, alpha=0.85)
        ax.set_xscale("log", base=2)
        ax.set_xticks(N_GRID); ax.set_xticklabels(N_GRID, fontsize=7)
        ax.set_yticks(np.arange(0.1, 1.3, 0.1))
        ax.set_ylim(0.05, 1.25)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"stratum = {stratum}", fontsize=10)
        ax.set_xlabel("N"); ax.set_ylabel("T*")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("T*(N) trajectory — all combos × all strata", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long-csv", type=Path, default=LONG_DEFAULT)
    ap.add_argument("--sim-ext-csv", type=Path, default=SIM_EXT_DEFAULT)
    ap.add_argument("--best-t-table", type=Path, default=TABLE_DEFAULT)
    ap.add_argument("--figs-dir", type=Path, default=FIGS_DEFAULT)
    ap.add_argument("--strata", default="overall,L1,L2,L3,L4,L5",
                    help="strata to render per-combo heatmaps for")
    args = ap.parse_args()

    long = load_acc_long(args.long_csv, args.sim_ext_csv)
    print(f"Loaded acc grid source: "
          f"{'sim_extended' if 'source' in long.columns and (long['source']=='sim').any() else 'markdown'}")
    bt = pd.read_csv(args.best_t_table)
    args.figs_dir.mkdir(parents=True, exist_ok=True)

    strata = args.strata.split(",")

    # Per-combo G/H plots
    for (model, dataset), _ in bt.groupby(["model", "dataset"]):
        combo_dir = args.figs_dir / f"{model}__{dataset}"
        combo_dir.mkdir(exist_ok=True)
        for stratum in strata:
            acc = get_acc_grid(long, model, dataset, stratum)
            if acc is None:
                continue
            sub_bt = bt[(bt.model == model) & (bt.dataset == dataset) & (bt.stratum == stratum)]
            t_star_by_n = {int(r.N): float(r.t_star) for _, r in sub_bt.iterrows()}
            n_prob_row = sub_bt.iloc[0] if len(sub_bt) else None
            n_prob = int(n_prob_row.level_n_problems) if n_prob_row is not None else 0
            title_g = f"acc(T,N) — {model}/{dataset} [{stratum} n={n_prob}]"
            title_h = f"regret(T,N) — {model}/{dataset} [{stratum} n={n_prob}]"
            plot_heatmap_with_tstar(
                acc, t_star_by_n, title_g,
                combo_dir / f"G_acc_heatmap_{stratum}.png", regret=False)
            plot_heatmap_with_tstar(
                acc, t_star_by_n, title_h,
                combo_dir / f"H_regret_heatmap_{stratum}.png", regret=True)
        print(f"  {model}/{dataset} G/H plots done")

    # Cross-combo grids — all 6 strata
    cross_dir = args.figs_dir / "cross"
    cross_dir.mkdir(exist_ok=True)
    for stratum in ["overall", "L1", "L2", "L3", "L4", "L5", "Lr"]:
        grid_of_heatmaps(long, bt, stratum, cross_dir / f"I_grid_acc_{stratum}.png", regret=False)
        grid_of_heatmaps(long, bt, stratum, cross_dir / f"I_grid_regret_{stratum}.png", regret=True)
        print(f"  cross grid {stratum} done")

    # NEW: per-combo "by-difficulty grid" — 6 strata side-by-side for ONE combo
    # Helps see how N×T landscape morphs across difficulty for a single (model, dataset).
    for (model, dataset), _ in bt.groupby(["model", "dataset"]):
        combo_dir = args.figs_dir / f"{model}__{dataset}"
        plot_combo_by_difficulty(long, bt, model, dataset, combo_dir / "J_by_difficulty_grid.png")
        plot_combo_by_difficulty(long, bt, model, dataset,
                                  combo_dir / "J_by_difficulty_grid_regret.png", regret=True)
        print(f"  {model}/{dataset} J plots done")

    # NEW: T*(N) overlay across all combos AND strata in one figure
    plot_t_star_all_combo_by_stratum(bt, cross_dir / "K_t_star_all_combo_by_stratum.png")
    print("  K plot done")


if __name__ == "__main__":
    main()
