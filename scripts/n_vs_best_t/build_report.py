"""Build final synthesis report from best_t_table.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TABLE = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/best_t_table.csv")
DEFAULT_REPORT = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/REPORT.md")
FIGS_REL = "figs"


def fmt_pp(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x*100:+.2f}pp"


def fmt_acc(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x*100:.2f}"


def per_combo_block(combo_df: pd.DataFrame, model: str, dataset: str, figs_rel: str) -> list[str]:
    md = [f"## {model} / {dataset}", ""]

    # N=4/16/64/256 mini summary table (overall stratum)
    ov = combo_df[combo_df.stratum == "overall"]
    if not ov.empty:
        snap = ov[ov.N.isin([4, 16, 64, 256])].sort_values("N")
        n_prob = int(snap["level_n_problems"].iloc[0]) if len(snap) else 0
        md.append(f"**Overall** (n_problems={n_prob}) snapshot:")
        md.append("| N | T\\* | best | T=1.0 | T=0.1 | random_T | equal_mix | "
                  "Δ vs T1 | Δ vs T0.1 | Δ vs rand | Δ vs eq_mix |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in snap.iterrows():
            md.append(
                f"| {int(r.N)} | {r.t_star:.1f} | {fmt_acc(r.best_t_mean)} | "
                f"{fmt_acc(r.acc_t1p0_mean)} | {fmt_acc(r.acc_t0p1_mean)} | "
                f"{fmt_acc(r.acc_random_t_mean)} | {fmt_acc(r.acc_equal_mix_mean)} | "
                f"{fmt_pp(r.gap_vs_t1p0)} | {fmt_pp(r.gap_vs_t0p1)} | "
                f"{fmt_pp(r.gap_vs_random_t)} | {fmt_pp(r.gap_vs_equal_mix)} |"
            )
        md.append("")

    # Per-level T* at N=256
    n256 = combo_df[combo_df.N == 256]
    if not n256.empty:
        md.append("**T*(N=256) by stratum:**  ")
        cells = []
        for L in ["L1", "L2", "L3", "L4", "L5"]:
            row = n256[n256.stratum == L]
            if row.empty:
                cells.append(f"{L}=—")
            else:
                r = row.iloc[0]
                cells.append(f"{L}=T{r.t_star:.1f} (acc {r.best_t_mean*100:.1f}, n={int(r.level_n_problems)})")
        md.append(" | ".join(cells))
        md.append("")

    # Figure embeds
    combo_dir = f"{figs_rel}/{model}__{dataset}"
    md.append("**N×T landscape by difficulty** (6 panels: overall + L1..L5)")
    md.append("")
    md.append(f"![NxT by difficulty]({combo_dir}/J_by_difficulty_grid.png)")
    md.append("")
    md.append(f"![NxT by difficulty regret]({combo_dir}/J_by_difficulty_grid_regret.png)")
    md.append("")
    md.append("**Overall stratum detail**")
    md.append("")
    md.append(f"![acc(T,N) overall]({combo_dir}/G_acc_heatmap_overall.png)")
    md.append("")
    md.append(f"![regret(T,N) overall]({combo_dir}/H_regret_heatmap_overall.png)")
    md.append("")
    md.append(f"![T* trajectory by level]({combo_dir}/A_t_star_trajectory.png)")
    md.append("")
    md.append(f"![5-curve overall]({combo_dir}/B_5curve_overall.png)")
    md.append("")
    md.append("ZOOM (N≥128, y-range auto-fit to top baselines — shows v1.0 / consensus_vote / equal_mix separation):")
    md.append("")
    md.append(f"![5-curve overall zoom128]({combo_dir}/B_5curve_overall_zoom128.png)")
    md.append("")
    md.append(f"![gap overall]({combo_dir}/C_gap_overall.png)")
    md.append("")
    md.append("**5-curve baseline comparison per difficulty (L1..L5, Lr)**")
    md.append("")
    for L in ["L1", "L2", "L3", "L4", "L5", "Lr"]:
        md.append(f"![B {L}]({combo_dir}/B_5curve_{L}.png)")
        md.append("")
        md.append(f"![B {L} zoom128]({combo_dir}/B_5curve_{L}_zoom128.png)")
        md.append("")
        md.append(f"![C {L}]({combo_dir}/C_gap_{L}.png)")
        md.append("")
    return md


def headline_table(bt: pd.DataFrame) -> list[str]:
    md = ["## Headline numbers — gap of best-T vs each baseline @ N=256, overall stratum", ""]
    has_v1 = "acc_v1_mean" in bt.columns
    header = "| model | dataset | T* | best (pp) | v1.0 acc | Δ vs T=1.0 | Δ vs T=0.1 | Δ vs random_T | Δ vs equal_mix | Δ vs v1.0 |" if has_v1 \
        else "| model | dataset | T* | best (pp) | Δ vs T=1.0 | Δ vs T=0.1 | Δ vs random_T | Δ vs equal_mix |"
    sep = "|---|---|---|---|---|---|---|---|---|---|" if has_v1 else "|---|---|---|---|---|---|---|---|"
    md.append(header); md.append(sep)
    sub = bt[(bt.stratum == "overall") & (bt.N == 256)].sort_values(["dataset", "model"])
    for _, r in sub.iterrows():
        if has_v1:
            md.append(
                f"| {r.model} | {r.dataset} | {r.t_star:.1f} | "
                f"{r.best_t_mean*100:.2f} | {fmt_acc(r.acc_v1_mean)} | "
                f"{fmt_pp(r.gap_vs_t1p0)} | {fmt_pp(r.gap_vs_t0p1)} | "
                f"{fmt_pp(r.gap_vs_random_t)} | {fmt_pp(r.gap_vs_equal_mix)} | "
                f"{fmt_pp(r.gap_vs_v1)} |"
            )
        else:
            md.append(
                f"| {r.model} | {r.dataset} | {r.t_star:.1f} | "
                f"{r.best_t_mean*100:.2f} | {fmt_pp(r.gap_vs_t1p0)} | "
                f"{fmt_pp(r.gap_vs_t0p1)} | {fmt_pp(r.gap_vs_random_t)} | "
                f"{fmt_pp(r.gap_vs_equal_mix)} |"
            )
    md.append("")
    return md


def small_N_table(bt: pd.DataFrame) -> list[str]:
    md = ["## Best-T vs baselines @ N=16 (overall) — small-budget regime", ""]
    md.append("| model | dataset | T* | best (pp) | Δ vs T=1.0 | Δ vs T=0.1 | Δ vs random_T | Δ vs equal_mix |")
    md.append("|---|---|---|---|---|---|---|---|")
    sub = bt[(bt.stratum == "overall") & (bt.N == 16)].sort_values(["dataset", "model"])
    for _, r in sub.iterrows():
        md.append(
            f"| {r.model} | {r.dataset} | {r.t_star:.1f} | "
            f"{r.best_t_mean*100:.2f} | {fmt_pp(r.gap_vs_t1p0)} | "
            f"{fmt_pp(r.gap_vs_t0p1)} | {fmt_pp(r.gap_vs_random_t)} | "
            f"{fmt_pp(r.gap_vs_equal_mix)} |"
        )
    md.append("")
    return md


def extended_N_table(bt: pd.DataFrame) -> list[str]:
    """Snapshot of best-T accuracy and T* at N ∈ {256, 512, 1024, 1536}."""
    if 1536 not in bt["N"].unique():
        return []
    md = ["## Extended N — best-T saturation @ N ∈ {256, 512, 1024, 1536} (overall)", ""]
    md.append("| model | dataset | T*@256 | best@256 | T*@512 | best@512 | "
              "T*@1024 | best@1024 | T*@1536 | best@1536 | Δ(1536−256) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    ov = bt[bt.stratum == "overall"].copy()
    for (model, dataset), sub in ov.groupby(["model", "dataset"]):
        cells = [model, dataset]
        snaps = {}
        for N in [256, 512, 1024, 1536]:
            r = sub[sub.N == N]
            if r.empty:
                cells.extend(["—", "—"]); snaps[N] = None
            else:
                rr = r.iloc[0]
                cells.extend([f"T{rr.t_star:.1f}", fmt_acc(rr.best_t_mean)])
                snaps[N] = float(rr.best_t_mean)
        if snaps.get(256) is not None and snaps.get(1536) is not None:
            cells.append(fmt_pp(snaps[1536] - snaps[256]))
        else:
            cells.append("—")
        md.append("| " + " | ".join(cells) + " |")
    md.append("")
    md.append("Observation: 대부분 조합에서 N=256 → N=1536은 best-T를 +0.5pp 미만으로만 "
              "올림 (saturation). 1536 budget을 균등 mix해도 equal_mix는 best-T에 "
              "0.5~1pp 이내로 수렴.")
    md.append("")
    return md


def cross_model_section(bt: pd.DataFrame) -> list[str]:
    md = ["## Cross-model — T*(N) for shared datasets", ""]
    overall_n256 = bt[(bt.stratum == "overall") & (bt.N == 256)]
    md.append("Pivot: T*(N=256) — rows=dataset, cols=model")
    md.append("")
    md.append("| dataset | Qwen2.5-3B | Qwen3-4B-Inst | Phi-4-mini | Llama-3.2-3B |")
    md.append("|---|---|---|---|---|")
    for ds in ["math1k", "mathfull", "math500", "gsm8kfull", "aime"]:
        row = []
        for m in ["Qwen2.5-3B", "Qwen3-4B-Instruct-2507", "Phi-4-mini-instruct", "Llama-3.2-3B"]:
            r = overall_n256[(overall_n256.model == m) & (overall_n256.dataset == ds)]
            if r.empty:
                row.append("—")
            else:
                row.append(f"T{r.iloc[0].t_star:.1f} ({r.iloc[0].best_t_mean*100:.1f})")
        md.append(f"| {ds} | " + " | ".join(row) + " |")
    md.append("")
    return md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-t-table", type=Path, default=DEFAULT_TABLE)
    ap.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--figs-rel", default=FIGS_REL)
    args = ap.parse_args()

    bt = pd.read_csv(args.best_t_table)

    md = ["# N 대비 최적 Temperature 분석 — Report",
          "",
          "샘플 예산 N(maj@N)이 주어졌을 때 어떤 단일 temperature T가 최적인지를 "
          "9개 (model × dataset_family) 조합에 대해 결정하고, "
          "5개 baseline과 비교한 결과:",
          "",
          "1. **T=1.0 fixed**, **T=0.1 fixed**, **random_T** (uniform T choice), "
          "**equal_mix** (round-robin N/12) — N에 대한 curve",
          "2. **v1.0** (`lb_algorithm_v1.py`) — cross-T LB routing algorithm, "
          "fixed B=256 design point (N=256 단일 marker, pilot 96 + main 160)",
          "",
          "**Method**: 마크다운 `*_n256_*-difficulty/difficulty_temperature_report.md` "
          "18개에서 best-T (acc(T,N) per-cell oracle) 추출. AIME 4년치는 problem-count "
          "가중 평균으로 합산. 5개 baseline은 모두 simulation source (lb_baselines.py: "
          "T=1.0/T=0.1/random_T/equal_mix는 240 MC rep으로 multinomial-from-empirical, "
          "v1.0은 동일 fashion으로 deploy-realistic 시뮬).",
          "",
          "**N grid**: {1,2,4,8,16,32,64,128,256}. **T grid**: {0.1, 0.2, …, 1.2} (12 값).",
          ""]

    md.extend(headline_table(bt))
    md.extend(small_N_table(bt))
    md.extend(extended_N_table(bt))
    md.extend(cross_model_section(bt))

    md.append("## Cross-comparison figures")
    md.append("")
    md.append("### N × T landscape (the central object)")
    md.append("")
    md.append("4×5 grid of `acc(T, N)` heatmaps with T*(N) trajectory overlay.")
    md.append("Rows = model, columns = dataset. Empty cells = no data.")
    md.append("")
    md.append("**Overall stratum**")
    md.append("")
    md.append(f"![NxT acc grid overall]({args.figs_rel}/cross/I_grid_acc_overall.png)")
    md.append("")
    md.append(f"![NxT regret grid overall]({args.figs_rel}/cross/I_grid_regret_overall.png)")
    md.append("")
    md.append("**Per-difficulty (L1=easy → L5=hard)**")
    md.append("")
    for L in ["L1", "L2", "L3", "L4", "L5"]:
        md.append(f"![NxT acc {L}]({args.figs_rel}/cross/I_grid_acc_{L}.png)")
        md.append("")
        md.append(f"![NxT regret {L}]({args.figs_rel}/cross/I_grid_regret_{L}.png)")
        md.append("")
    md.append("**T*(N) across all combos by difficulty stratum**")
    md.append("")
    md.append(f"![K t-star all combo by stratum]({args.figs_rel}/cross/K_t_star_all_combo_by_stratum.png)")
    md.append("")
    md.append("### T* summary heatmaps and 5-curve comparisons")
    md.append("")
    md.append(f"![T* heatmap N=256 overall]({args.figs_rel}/cross/D_heatmap_overall_N256.png)")
    md.append("")
    md.append(f"![T* heatmap N=64 overall]({args.figs_rel}/cross/D_heatmap_overall_N64.png)")
    md.append("")
    md.append(f"![T* heatmap N=256 L5]({args.figs_rel}/cross/D_heatmap_L5_N256.png)")
    md.append("")
    md.append(f"![5-curve grid overall]({args.figs_rel}/cross/E_grid_overall.png)")
    md.append("")
    md.append(f"![Win-margin scatter]({args.figs_rel}/cross/F_win_margin_scatter.png)")
    md.append("")

    md.append("## Per-combo detail")
    md.append("")
    for (model, dataset), sub in bt.groupby(["model", "dataset"]):
        md.extend(per_combo_block(sub, model, dataset, args.figs_rel))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
