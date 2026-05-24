"""N × T plots — the central object of this analysis.

Per (model, dataset, stratum):
- G. acc(T, N) heatmap with T*(N) trajectory overlaid (white line + dots)
- H. regret heatmap: acc(T,N) − max_T acc(T,N) (shows how costly suboptimal T is at each N)

Cross-combo:
- I. 9-panel small-multiples grid: heatmap per combo with T*(N) overlay (overall stratum)

Source: long_all.csv (raw per-(T,N) accuracy) + best_t_table.csv (T*(N)).

Two versions: _with_fixed (includes best_fixed_t overlay) and _no_fixed (without it).
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

FIG_DPI = 200


def get_acc_grid(long: pd.DataFrame, model: str, dataset: str, stratum: str) -> np.ndarray | None:
    sub = long[(long.model == model) & (long.dataset == dataset) & (long.stratum == stratum)]
    if sub.empty:
        return None
    pv = sub.pivot_table(index="T", columns="N", values="mean_acc")
    pv = pv.reindex(index=T_GRID, columns=N_GRID)
    if pv.isna().all().all():
        return None
    return pv.values  # rows=T, cols=N


def load_acc_long(long_path: Path, sim_ext_path: Path | None) -> pd.DataFrame:
    if sim_ext_path is not None and sim_ext_path.exists():
        sim = pd.read_csv(sim_ext_path)
        is_single = sim["baseline"].str.match(r"^T\d+\.\d+$")
        sim_t = sim[is_single].copy()
        sim_t["T"] = sim_t["baseline"].str[1:].astype(float)
        sim_t = sim_t.rename(columns={"mean": "mean_acc"})
        sim_t["source"] = "sim"
        return sim_t[["model", "dataset", "stratum", "N", "T", "mean_acc", "source"]]
    return pd.read_csv(long_path)


def plot_heatmap_with_tstar(acc: np.ndarray, t_star_by_n: dict[int, float],
                             title: str, out_path: Path, regret: bool = False,
                             show_fixed: bool = False,
                             best_fixed_t: float | None = None) -> None:
    if regret:
        col_max = np.nanmax(acc, axis=0, keepdims=True)
        z = (acc - col_max) * 100
        cmap = "Reds_r"
        vmin, vmax = -20, 0
        cbar_label = "ΔAcc vs best-T (pp, ≤0)"
    else:
        z = acc * 100
        cmap = "viridis"
        vmin = np.nanmin(z); vmax = np.nanmax(z)
        cbar_label = "maj@N accuracy (%)"

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(z, aspect="auto", origin="lower", cmap=cmap,
                   vmin=vmin, vmax=vmax,
                   extent=[-0.5, len(N_GRID) - 0.5, -0.5, len(T_GRID) - 0.5])
    ax.set_xticks(range(len(N_GRID))); ax.set_xticklabels(N_GRID)
    ax.set_yticks(range(len(T_GRID))); ax.set_yticklabels([f"{t:.1f}" for t in T_GRID])
    ax.set_xlabel("N (sample budget)")
    ax.set_ylabel("T (temperature)")

    if show_fixed and best_fixed_t is not None:
        # Horizontal line at best_fixed_t
        y_idx = (best_fixed_t - 0.1) * 10
        ax.axhline(y_idx, color="white", lw=2.5, ls="--",
                   label=f"best-fixed-T={best_fixed_t:.1f}")
        # Annotate accuracy values along the fixed T line
        t_row_idx = int(round((best_fixed_t - 0.1) * 10))
        if 0 <= t_row_idx < len(T_GRID):
            for x_idx in range(len(N_GRID)):
                v = z[t_row_idx, x_idx]
                if not np.isnan(v):
                    txt = f"{v:.1f}" if regret else f"{v:.0f}"
                    ax.text(x_idx, t_row_idx - 0.35, txt, ha="center", va="top",
                            color="white", fontsize=7, fontweight="bold")
    else:
        # Original T*(N) overlay
        xs = []; ys = []
        for i, n in enumerate(N_GRID):
            t = t_star_by_n.get(n)
            if t is None:
                continue
            xs.append(i)
            y_idx = int(round((t - 0.1) * 10))
            ys.append(y_idx)
        ax.plot(xs, ys, "o-", color="white", lw=2.0, ms=7,
                markeredgecolor="black", markerfacecolor="white", label="T*(N)")
        for x, y in zip(xs, ys):
            v = z[y, x]
            txt = f"{v:.1f}" if regret else f"{v:.0f}"
            ax.text(x, y - 0.35, txt, ha="center", va="top",
                    color="white", fontsize=7, fontweight="bold")

    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def grid_of_heatmaps(long: pd.DataFrame, bt: pd.DataFrame, stratum: str,
                     out_path: Path, regret: bool = False,
                     show_fixed: bool = False) -> None:
    fig, axes = plt.subplots(len(MODELS_ORDER), len(DATASETS_ORDER),
                             figsize=(20, 14), sharex=True, sharey=True)
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

            sub_bt = bt[(bt.model == model) & (bt.dataset == dataset) & (bt.stratum == stratum)]
            if show_fixed and "best_fixed_t" in sub_bt.columns and not sub_bt["best_fixed_t"].isna().all():
                ft = sub_bt["best_fixed_t"].iloc[0]
                y_idx = (ft - 0.1) * 10
                ax.axhline(y_idx, color="white", lw=1.5, ls="--")
            else:
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
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_combo_by_difficulty(long: pd.DataFrame, bt: pd.DataFrame,
                              model: str, dataset: str, out_path: Path,
                              regret: bool = False,
                              show_fixed: bool = False) -> None:
    strata = ["overall", "L1", "L2", "L3", "L4", "L5", "Lr"]
    fig, axes = plt.subplots(1, 7, figsize=(28, 6), sharex=True, sharey=True)
    cmap = "Reds_r" if regret else "viridis"
    vmin = -15 if regret else 0
    vmax = 0 if regret else 100

    last_im = None
    for ax, stratum in zip(axes, strata):
        acc = get_acc_grid(long, model, dataset, stratum)
        sub_bt = bt[(bt.model == model) & (bt.dataset == dataset) & (bt.stratum == stratum)]
        ax.set_title(stratum, fontsize=10)
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

        if show_fixed and "best_fixed_t" in sub_bt.columns and not sub_bt["best_fixed_t"].isna().all():
            ft = sub_bt["best_fixed_t"].iloc[0]
            y_idx = (ft - 0.1) * 10
            ax.axhline(y_idx, color="white", lw=1.5, ls="--")
        else:
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
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_t_star_all_combo_by_stratum(bt: pd.DataFrame, out_path: Path,
                                      show_fixed: bool = False) -> None:
    strata = ["overall", "L1", "L2", "L3", "L4", "L5", "Lr"]
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharex=True, sharey=True)
    axes = axes.flatten()
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
            label = f"{model[:8]}/{dataset}"
            ax.plot(s["N"], s["t_star"], marker="o", lw=lw, ls=ls,
                    color=cmap(idx % 10), label=label, alpha=0.85)
            if show_fixed and "best_fixed_t" in s.columns and not s["best_fixed_t"].isna().all():
                ft = s["best_fixed_t"].iloc[0]
                ax.axhline(ft, color=cmap(idx % 10), lw=1.0, ls=":", alpha=0.5)
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
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
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

    # Per-combo G/H plots — two versions each
    for (model, dataset), _ in bt.groupby(["model", "dataset"]):
        combo_dir = args.figs_dir / f"{model}__{dataset}"
        combo_dir.mkdir(exist_ok=True)
        for stratum in strata:
            acc = get_acc_grid(long, model, dataset, stratum)
            if acc is None:
                continue
            sub_bt = bt[(bt.model == model) & (bt.dataset == dataset) & (bt.stratum == stratum)]
            t_star_by_n = {int(r.N): float(r.t_star) for _, r in sub_bt.iterrows()}
            ft = None
            if "best_fixed_t" in sub_bt.columns and not sub_bt["best_fixed_t"].isna().all():
                ft = float(sub_bt["best_fixed_t"].iloc[0])
            title_g = f"acc(T,N) — {model}/{dataset} [{stratum}]"
            title_h = f"regret(T,N) — {model}/{dataset} [{stratum}]"
            # no_fixed versions
            plot_heatmap_with_tstar(
                acc, t_star_by_n, title_g,
                combo_dir / f"G_acc_heatmap_{stratum}_no_fixed.png",
                regret=False, show_fixed=False)
            plot_heatmap_with_tstar(
                acc, t_star_by_n, title_h,
                combo_dir / f"H_regret_heatmap_{stratum}_no_fixed.png",
                regret=True, show_fixed=False)
            # with_fixed versions
            plot_heatmap_with_tstar(
                acc, t_star_by_n, title_g,
                combo_dir / f"G_acc_heatmap_{stratum}_with_fixed.png",
                regret=False, show_fixed=True, best_fixed_t=ft)
            plot_heatmap_with_tstar(
                acc, t_star_by_n, title_h,
                combo_dir / f"H_regret_heatmap_{stratum}_with_fixed.png",
                regret=True, show_fixed=True, best_fixed_t=ft)
        print(f"  {model}/{dataset} G/H plots done")

    # Cross-combo grids — two versions
    cross_dir = args.figs_dir / "cross"
    cross_dir.mkdir(exist_ok=True)
    for stratum in ["overall", "L1", "L2", "L3", "L4", "L5", "Lr"]:
        for sf, suffix in [(False, "_no_fixed"), (True, "_with_fixed")]:
            grid_of_heatmaps(long, bt, stratum,
                             cross_dir / f"I_grid_acc_{stratum}{suffix}.png",
                             regret=False, show_fixed=sf)
            grid_of_heatmaps(long, bt, stratum,
                             cross_dir / f"I_grid_regret_{stratum}{suffix}.png",
                             regret=True, show_fixed=sf)
        print(f"  cross grid {stratum} done")

    # Per-combo "by-difficulty grid" — two versions
    for (model, dataset), _ in bt.groupby(["model", "dataset"]):
        combo_dir = args.figs_dir / f"{model}__{dataset}"
        for sf, suffix in [(False, "_no_fixed"), (True, "_with_fixed")]:
            plot_combo_by_difficulty(long, bt, model, dataset,
                                     combo_dir / f"J_by_difficulty_grid{suffix}.png",
                                     show_fixed=sf)
            plot_combo_by_difficulty(long, bt, model, dataset,
                                     combo_dir / f"J_by_difficulty_grid_regret{suffix}.png",
                                     regret=True, show_fixed=sf)
        print(f"  {model}/{dataset} J plots done")

    # K plot — two versions
    plot_t_star_all_combo_by_stratum(bt,
                                      cross_dir / "K_t_star_all_combo_by_stratum_no_fixed.png",
                                      show_fixed=False)
    plot_t_star_all_combo_by_stratum(bt,
                                      cross_dir / "K_t_star_all_combo_by_stratum_with_fixed.png",
                                      show_fixed=True)
    print("  K plot done")


if __name__ == "__main__":
    main()
