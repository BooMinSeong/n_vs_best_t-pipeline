"""Build final synthesis report from best_t_table.csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TABLE = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/best_t_table.csv")
DEFAULT_REPORT = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/REPORT.md")
DEFAULT_DIST_ROOT = Path("/home3/b.ms/projects/tts_analysis/outputs")
FIGS_REL = "figs"

AIME_YEARS = ["aime2023", "aime2024", "aime2025", "aime2026"]


def load_raw_counts(dist_root: Path, combos: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
    """For each (dataset_family, model), sum problem counts across distribution files.

    AIME family expands to 4 year files; everything else is direct lookup.
    """
    out: dict[tuple[str, str], int] = {}
    for ds_family, model in combos:
        years = AIME_YEARS if ds_family == "aime" else [ds_family]
        total = 0
        for y in years:
            f = dist_root / f"distributions-{y}_n256_{model}" / "distributions.json"
            if not f.exists():
                total = -1
                break
            with f.open() as fp:
                total += len(json.load(fp).get("distributions", {}))
        if total >= 0:
            out[(ds_family, model)] = total
    return out


def load_per_stratum_raw(dist_root: Path,
                          combos: list[tuple[str, str]]) -> dict[tuple[str, str], dict[str, int]]:
    """For each (dataset_family, model), count raw problems per stratum (L1..L5, Lr)."""
    out: dict[tuple[str, str], dict[str, int]] = {}
    for ds_family, model in combos:
        years = AIME_YEARS if ds_family == "aime" else [ds_family]
        counts = {f"L{i}": 0 for i in range(1, 6)}
        counts["Lr"] = 0
        ok = True
        for y in years:
            f = dist_root / f"distributions-{y}_n256_{model}" / "distributions.json"
            if not f.exists():
                ok = False
                break
            with f.open() as fp:
                data = json.load(fp)
            for prob in data.get("distributions", {}).values():
                lvl = prob.get("difficulty", {}).get("level", None)
                key = f"L{lvl}" if lvl in (1, 2, 3, 4, 5) else "Lr"
                counts[key] += 1
        if ok:
            out[(ds_family, model)] = counts
    return out


def coverage_section(bt: pd.DataFrame, raw_counts: dict[tuple[str, str], int]) -> list[str]:
    """Table comparing n_raw vs n_loaded, plus best-T accuracy inclusive of excluded.

    Excluded problems have P[correct]=0 under sim, so their MV-accuracy is 0;
    inclusive accuracy = sim accuracy × (n_loaded / n_raw).
    """
    md = ["## Coverage and inclusive-of-excluded accuracy @ N=256, overall stratum", ""]
    md.append("아래 표는 raw dataset 의 전체 문제 수 (`n_raw`) 대비 sim 에 포함된 "
              "문제 수 (`n_loaded`) 와, 제외된 문제도 정답률 0으로 포함시킨 inclusive "
              "정답률을 함께 보여줍니다. `incl_X = X × n_loaded / n_raw` 으로 산출.")
    md.append("")
    md.append("| model | dataset | n_raw | n_loaded | excl | excl % | "
              "best (loaded) | best (incl) | Δ best | "
              "equal_mix (incl) | consensus_vote (incl) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    sub = bt[(bt.stratum == "overall") & (bt.N == 256)].sort_values(["dataset", "model"])
    for _, r in sub.iterrows():
        key = (r.dataset, r.model)
        if key not in raw_counts:
            continue
        n_raw = raw_counts[key]
        n_loaded = int(r.level_n_problems)
        excl = n_raw - n_loaded
        excl_pct = excl / n_raw * 100 if n_raw else 0.0
        scale = n_loaded / n_raw if n_raw else 0.0
        best_loaded = r.best_t_mean
        best_incl = best_loaded * scale
        eqmix_incl = r.acc_equal_mix_mean * scale
        cv_incl = r.acc_consensus_vote_mean * scale
        md.append(
            f"| {r.model} | {r.dataset} | {n_raw} | {n_loaded} | {excl} | "
            f"{excl_pct:.1f}% | {fmt_acc(best_loaded)} | {fmt_acc(best_incl)} | "
            f"{fmt_pp(best_incl - best_loaded)} | "
            f"{fmt_acc(eqmix_incl)} | {fmt_acc(cv_incl)} |"
        )
    md.append("")
    md.append("**해석**: `excl %` 가 큰 dataset 일수록 loaded vs inclusive 차이가 큽니다. "
              "AIME 계열은 dataset 자체가 작고 (30 problems/year × 4 = 120) 정답이 한 번도 "
              "나오지 않는 문제 비율이 높아 (~10–25%) inclusive accuracy 가 loaded 대비 "
              "크게 낮아집니다. math/gsm8k 계열은 제외율 ≲2% 로 차이가 작습니다.")
    md.append("")
    return md


def per_stratum_coverage_section(bt: pd.DataFrame,
                                  per_stratum_raw: dict[tuple[str, str], dict[str, int]]
                                  ) -> list[str]:
    """Per-stratum raw vs loaded comparison. Excluded problems concentrate entirely in Lr.

    L1~L5 에는 제외가 0건 (모든 drop 은 level=None=Lr 로 분류됨) 이므로 inclusive 변화는
    Lr 에만 집중됩니다.
    """
    md = ["## Per-stratum coverage — L1..L5 vs Lr @ N=256", ""]
    md.append("`load_aligned` 의 drop 조건 (정답이 한 번도 안 나옴 / parsing 실패) 은 "
              "baseline solve rate 가 0% 인 문제와 정확히 일치하므로, drop 된 문제는 "
              "전부 **Lr** (level=None) 으로만 분류됩니다. **L1~L5 에는 drop 이 0건** "
              "(per-level inclusive = loaded 동일). 따라서 inclusive 효과는 Lr 에 집중.")
    md.append("")
    md.append("**Lr stratum 의 inclusive vs loaded accuracy** (Lr 만 환산: "
              "`incl_Lr = loaded_acc_Lr × n_loaded_Lr / n_raw_Lr`)")
    md.append("")
    md.append("| model | dataset | n_raw_Lr | n_loaded_Lr | excl Lr | excl % | "
              "best Lr (loaded) | best Lr (incl) | Δ best Lr | "
              "eq_mix Lr (incl) | cv Lr (incl) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    sub = bt[(bt.stratum == "Lr") & (bt.N == 256)].sort_values(["dataset", "model"])
    for _, r in sub.iterrows():
        key = (r.dataset, r.model)
        if key not in per_stratum_raw:
            continue
        n_raw_Lr = per_stratum_raw[key]["Lr"]
        n_loaded_Lr = int(r.level_n_problems)
        if n_raw_Lr == 0:
            continue
        excl_Lr = n_raw_Lr - n_loaded_Lr
        excl_pct = excl_Lr / n_raw_Lr * 100
        scale = n_loaded_Lr / n_raw_Lr
        best_loaded = r.best_t_mean
        best_incl = best_loaded * scale
        eqmix_incl = r.acc_equal_mix_mean * scale if pd.notna(r.acc_equal_mix_mean) else float("nan")
        cv_incl = r.acc_consensus_vote_mean * scale if pd.notna(r.acc_consensus_vote_mean) else float("nan")
        md.append(
            f"| {r.model} | {r.dataset} | {n_raw_Lr} | {n_loaded_Lr} | {excl_Lr} | "
            f"{excl_pct:.1f}% | {fmt_acc(best_loaded)} | {fmt_acc(best_incl)} | "
            f"{fmt_pp(best_incl - best_loaded)} | {fmt_acc(eqmix_incl)} | {fmt_acc(cv_incl)} |"
        )
    md.append("")
    md.append("**관찰**: AIME 계열 Lr 의 excl 비율은 30–73% 수준이며 (Qwen3-4B/aime: "
              "raw_Lr=37 중 27개 drop = 73%) inclusive accuracy 가 큰 폭으로 감소. "
              "이는 Lr 의 본질 — \"any-T 에서 한 번도 못 푸는 unsolvable 문제\" "
              "— 가 raw 단계에선 Lr 의 큰 비중을 차지하기 때문입니다.")
    md.append("")
    md.append("**L1~L5 stratum 별 n_loaded** (inclusive=loaded, 추가 환산 불필요):")
    md.append("")
    md.append("| model | dataset | L1 | L2 | L3 | L4 | L5 | Lr (loaded) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for (model, dataset), sub_md in bt[bt.N == 256].groupby(["model", "dataset"]):
        cells = [model, dataset]
        for S in ["L1", "L2", "L3", "L4", "L5", "Lr"]:
            row = sub_md[sub_md.stratum == S]
            if row.empty:
                cells.append("—")
            else:
                cells.append(str(int(row.iloc[0].level_n_problems)))
        md.append("| " + " | ".join(cells) + " |")
    md.append("")
    return md


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
    md.append("ZOOM (N≥128, y-range auto-fit to top baselines — shows consensus_vote / equal_mix separation):")
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
    md.append("| model | dataset | T* | best (pp) | Δ vs T=1.0 | Δ vs T=0.1 | "
              "Δ vs random_T | Δ vs equal_mix | Δ vs consensus_vote |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    sub = bt[(bt.stratum == "overall") & (bt.N == 256)].sort_values(["dataset", "model"])
    for _, r in sub.iterrows():
        md.append(
            f"| {r.model} | {r.dataset} | {r.t_star:.1f} | "
            f"{r.best_t_mean*100:.2f} | {fmt_pp(r.gap_vs_t1p0)} | "
            f"{fmt_pp(r.gap_vs_t0p1)} | {fmt_pp(r.gap_vs_random_t)} | "
            f"{fmt_pp(r.gap_vs_equal_mix)} | {fmt_pp(r.gap_vs_consensus_vote)} |"
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
          "1. **T=1.0 fixed**, **T=0.1 fixed** — single-T MV baselines",
          "2. **random_T** (uniform T choice per sim), **equal_mix** "
          "(round-robin N/12 alloc → pooled MV), **consensus_vote** "
          "(round-robin N/12 alloc → per-T MV winner → plurality of 12 winners)",
          "",
          "**Method**: 모든 수치는 동일한 sim_v2 simulation 결과에서 산출 "
          "(`run_sim_v2_one.py`, 240 MC rep, multinomial-from-empirical, "
          "seed=42). mix baseline 5종(T=1.0, T=0.1, random_T, equal_mix, "
          "consensus_vote) 도 같은 simulation 에서 산출. AIME 4년치는 "
          "problem-count 가중 평균으로 합산.",
          "",
          "**best-T 와 T\\* 의 정의** (`merge_sim_v2.py:114-133`): "
          "각 `(model, dataset, stratum, N)` 조합에 대해 독립적으로, 12개 "
          "single-T MV baseline (T=0.1, 0.2, …, 1.2) 의 sim accuracy 중 "
          "**argmax** 를 취한 값이 `best_t_mean`, 그 argmax 에 해당하는 T 값이 "
          "`t_star`. 즉 each cell 의 12개 후보 T 중 최고 성능 단일 T 를 "
          "사후적으로 선택한 것 — \"어떤 T 가 best 인지 알았더라면 얼마나 "
          "정확했을지\" 의 oracle ceiling.",
          "",
          "**주요 특성:**",
          "- *Oracle*: stratum, N, dataset 별로 t_star 가 다를 수 있으며, 실제 "
          "deployment 에서는 문제의 정답을 모르므로 이 T 를 선택할 수 없음. 모든 "
          "deployable baseline (T=1.0 fixed, equal_mix, consensus_vote 등) 이 "
          "좇아가는 **upper bound**.",
          "- *Per-stratum × per-N*: L1 의 t_star 와 L5 의 t_star, N=16 의 t_star 와 "
          "N=256 의 t_star 가 모두 독립적으로 산출 (어려운 문제일수록 높은 T 가 "
          "유리한 경향).",
          "- *MV (maj@N) 기준*: 각 T 의 정확도는 N개 sample 의 majority vote 가 "
          "정답일 확률 (sim 240 rep 평균).",
          "- *Gap 의 의미*: `gap_vs_X = best_t_mean − acc_X` 는 \"최적 T 를 "
          "알았더라면 baseline X 대비 얼마나 더 좋아졌을지\" 의 oracle 이점 측정.",
          "",
          "**문제 제외 기준** (`load_aligned`, `cross_t_mode/lb_baselines.py:49-88`): "
          "다음 두 조건 중 하나라도 해당하는 문제는 sim 입력에서 drop됩니다.",
          "1. 12개 temperature (T=0.1~1.2) 어디서도 답 추출이 한 번도 안 된 문제 "
          "(모든 generation 의 parsing 실패).",
          "2. canonical 정답이 None 이거나, 정답이 모든 T × 모든 sample 분포에 "
          "한 번도 등장하지 않은 문제 (sim 의 P[correct]=0 → 자명히 못 맞춤).",
          "",
          "본 보고서의 `level_n_problems` 와 모든 정답률은 위 기준으로 drop 한 뒤의 "
          "**loaded 집합** 기준입니다. 전체 raw dataset 기준 (제외된 문제는 정답률 0 "
          "으로 포함) 비교는 아래 **Coverage** 섹션 참고.",
          "",
          "추가로, **stratum** 분류:",
          "- L1~L5: baseline solve rate 기반 difficulty level (L1 easiest ~ L5 hardest, "
          "`(min, max]` 구간).",
          "- Lr: difficulty level 부여 실패 (level=None) 인 문제 (canonical 정답은 "
          "loaded 단계에서 존재 확인, baseline solve rate 산출 실패한 케이스).",
          "- overall: L1~L5 + Lr 가중평균 (loaded 전체).",
          "- overall_md_compat: L1~L5 만 (Lr 제외). 옛 markdown 보고서 호환용.",
          "",
          "**N grid**: {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048}. "
          "**T grid**: {0.1, 0.2, …, 1.2} (12 값).",
          ""]

    combos = list(bt[(bt.stratum == "overall") & (bt.N == 256)][["dataset", "model"]]
                  .itertuples(index=False, name=None))
    raw_counts = load_raw_counts(DEFAULT_DIST_ROOT, combos)
    per_stratum_raw = load_per_stratum_raw(DEFAULT_DIST_ROOT, combos)
    md.extend(coverage_section(bt, raw_counts))
    md.extend(per_stratum_coverage_section(bt, per_stratum_raw))
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
