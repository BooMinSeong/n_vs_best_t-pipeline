"""Stage 3 — paper-ready figures (spec section 3.3).

Fig 1  Regret CDF (R1 + R2 panels)        per (model,dataset,N)
Fig 2  Pairwise per-problem regret scatter per (model,dataset,N), representative pairs
Fig 3  T*(p) histogram                     per (model,dataset,N)
Fig 4  T*-conditional mean regret line     per (model,dataset,N)
Fig 5  Oracle gap matrix (model x dataset) one figure per N
Fig 6  N-dependence of regret p95          per (model,dataset)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from per_problem_regret import config as C

# ----- style -----------------------------------------------------------------
PALETTE = sns.color_palette("colorblind")
FIG_STRATEGIES = ["T0.1", "T1.0", "best_fixed_T", "random_T",
                  "equal_mix", "consensus_vote"]
STRAT_COLOR = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(FIG_STRATEGIES)}


def _setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def _save(fig, path_noext: Path):
    path_noext.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_noext.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_noext.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def lbl(s: str) -> str:
    return C.DISPLAY_NAME.get(s, s if s != "best_fixed_T" else "Best fixed T")


# ----- Fig 1 -----------------------------------------------------------------
def fig_regret_cdf(regret: pd.DataFrame, figdir: Path):
    for (model, dataset, N), g in regret.groupby(["model", "dataset", "N"]):
        fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
        for ax, (rcol, rname) in zip(axes, [("regret_R1", "R1: vs per-problem oracle"),
                                            ("regret_R2", "R2: vs dataset oracle")]):
            for s in FIG_STRATEGIES:
                x = np.sort(g[g.strategy == s][rcol].to_numpy())
                if x.size == 0:
                    continue
                y = np.arange(1, x.size + 1) / x.size
                ax.plot(np.concatenate([[0], x]), np.concatenate([[0], y]),
                        drawstyle="steps-post",
                        label=lbl(s), color=STRAT_COLOR[s], lw=1.4)
            ax.set_xlabel("regret"); ax.set_title(rname)
            ax.set_xlim(left=0)
        axes[0].set_ylabel("cumulative fraction of problems")
        axes[1].legend(loc="lower right", framealpha=0.9)
        fig.suptitle(f"{model} / {dataset} / N={N} — regret CDF", fontsize=9)
        _save(fig, figdir / f"{model}__{dataset}" / f"fig1_regret_cdf_N{N}")


# ----- Fig 2 -----------------------------------------------------------------
def fig_pairwise_scatter(regret: pd.DataFrame, figdir: Path):
    pairs = [("T1.0", "equal_mix"), ("T1.0", "consensus_vote"),
             ("equal_mix", "consensus_vote")]
    for (model, dataset, N), g in regret.groupby(["model", "dataset", "N"]):
        wide = g.pivot_table(index="problem_id", columns="strategy", values="regret_R2")
        valid = [(a, b) for a, b in pairs if a in wide and b in wide]
        if not valid:
            continue
        fig, axes = plt.subplots(1, len(valid), figsize=(3 * len(valid), 3))
        if len(valid) == 1:
            axes = [axes]
        for ax, (a, b) in zip(axes, valid):
            ax.scatter(wide[a], wide[b], s=14, alpha=0.6, color=PALETTE[0])
            hi = float(np.nanmax([wide[a].max(), wide[b].max(), 0.01]))
            ax.plot([0, hi], [0, hi], ls="--", color="grey", lw=1)
            ax.set_xlabel(f"{lbl(a)} regret"); ax.set_ylabel(f"{lbl(b)} regret")
            ax.set_aspect("equal", "box")
        fig.suptitle(f"{model} / {dataset} / N={N} — per-problem regret (R2)", fontsize=9)
        _save(fig, figdir / f"{model}__{dataset}" / f"fig2_pairwise_scatter_N{N}")


# ----- Fig 3 -----------------------------------------------------------------
def fig_t_star_hist(tstar_dist: pd.DataFrame, figdir: Path):
    for (model, dataset, N), g in tstar_dist.groupby(["model", "dataset", "N"]):
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        g = g.sort_values("T_star")
        ax.bar(g.T_star, g.n_problems, width=0.07, color=PALETTE[2])
        ax.set_xlabel(r"$T^*(p)$"); ax.set_ylabel("# problems")
        ax.set_xticks(C.T_VALUES); ax.tick_params(axis="x", labelrotation=90)
        ax.set_title(f"{model} / {dataset} / N={N}")
        _save(fig, figdir / f"{model}__{dataset}" / f"fig3_t_star_hist_N{N}")


# ----- Fig 4 -----------------------------------------------------------------
def fig_t_star_conditional(cond: pd.DataFrame, figdir: Path):
    c = cond[cond.regret_kind == "R2"]
    for (model, dataset, N), g in c.groupby(["model", "dataset", "N"]):
        fig, ax = plt.subplots(figsize=(4, 2.8))
        for s in FIG_STRATEGIES:
            gs = g[g.strategy == s].sort_values("T_star")
            if gs.empty:
                continue
            alphas = np.where(gs["count"] >= 5, 1.0, 0.3)
            ax.plot(gs.T_star, gs["mean"], color=STRAT_COLOR[s], lw=1.3,
                    marker="o", ms=3, label=lbl(s))
            ax.scatter(gs.T_star, gs["mean"], s=18 * alphas + 2,
                       color=STRAT_COLOR[s], alpha=0.9)
        ax.set_xlabel(r"$T^*(p)$ (native 12 values)")
        ax.set_ylabel("mean regret (R2)")
        ax.set_xticks(C.T_VALUES); ax.tick_params(axis="x", labelrotation=90)
        ax.legend(fontsize=6, ncol=2)
        ax.set_title(f"{model} / {dataset} / N={N}")
        _save(fig, figdir / f"{model}__{dataset}" / f"fig4_t_star_conditional_N{N}")


# ----- Fig 5 -----------------------------------------------------------------
def fig_oracle_gap_matrix(gap: pd.DataFrame, figdir: Path):
    for N, g in gap.groupby("N"):
        piv = g.pivot_table(index="model", columns="dataset", values="gap_pp")
        fig, ax = plt.subplots(figsize=(1.2 + 0.7 * piv.shape[1],
                                        1.0 + 0.5 * piv.shape[0]))
        sns.heatmap(piv, annot=True, fmt=".2f", cmap="viridis", ax=ax,
                    cbar_kws={"label": "oracle gap (pp)"})
        ax.set_title(f"Per-problem - dataset oracle gap (pp), N={N}")
        _save(fig, figdir / f"fig5_oracle_gap_matrix_N{N}")


# ----- Fig 6 -----------------------------------------------------------------
def fig_n_dependence(stats: pd.DataFrame, figdir: Path):
    s = stats[stats.regret_kind == "R2"]
    for (model, dataset), g in s.groupby(["model", "dataset"]):
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        for strat in FIG_STRATEGIES:
            gs = g[g.strategy == strat].sort_values("N")
            if gs.empty:
                continue
            ax.plot(gs.N, gs.regret_p95, marker="o", ms=3, lw=1.3,
                    label=lbl(strat), color=STRAT_COLOR[strat])
        ax.set_xscale("log", base=2)
        ax.set_xlabel("N"); ax.set_ylabel("regret p95 (R2)")
        ax.legend(fontsize=6, ncol=2)
        ax.set_title(f"{model} / {dataset}")
        _save(fig, figdir / f"{model}__{dataset}" / "fig6_n_dependence_p95")


def make_all(art: dict, figdir: Path):
    _setup()
    fig_regret_cdf(art["regret"], figdir)
    fig_pairwise_scatter(art["regret"], figdir)
    fig_t_star_hist(art["summary_t_star_distribution"], figdir)
    fig_t_star_conditional(art["summary_t_star_conditional_regret"], figdir)
    fig_oracle_gap_matrix(art["summary_oracle_gap"], figdir)
    fig_n_dependence(art["summary_distribution_stats"], figdir)
    print(f"figures written under {figdir}")


def main() -> None:
    from per_problem_regret import analyze
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=C.OUT_DIR)
    ap.add_argument("--combos", default=None)
    args = ap.parse_args()
    combos = args.combos.split(",") if args.combos else None
    art = analyze.run(args.out_dir, combos, C.Config())
    make_all(art, args.out_dir / "figs")


if __name__ == "__main__":
    main()
