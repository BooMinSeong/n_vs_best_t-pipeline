"""Build Dataset-level Performance Comparison markdown report.

Reads sim_v2/*.csv (per-combo long-format, includes per-year AIME) and emits
DATASET_COMPARISON.md with one section per dataset and one sub-section per N.

Datasets cover aime2023..aime2026 (per year, not merged), gsm8kfull, math1k,
math500, mathfull. N grid is {32, 64, 128, 256, 1024, 2048}, stratum=overall.
Best-T column uses argmax over 12 single-T baselines with the same seed=42
jitter as merge_sim_v2.py for tie-breaking consistency.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SIM_V2_DIR = Path("outputs/n_vs_best_t/sim_v2")
DEFAULT_OUT = Path("outputs/n_vs_best_t/DATASET_COMPARISON.md")

COMBO_RE = re.compile(r"^(?P<dataset>aime\d{4}|math1k|mathfull|math500|gsm8kfull)"
                      r"_(?P<model>.+)$")

DATASET_ORDER = ["aime2023", "aime2024", "aime2025", "aime2026", "aime",
                 "gsm8kfull", "math1k", "math500", "mathfull"]
AIME_YEARS = ["aime2023", "aime2024", "aime2025", "aime2026"]
N_VALUES = [32, 64, 128, 256, 1024, 2048]
STRATA = ["L1", "L2", "L3", "L4", "L5", "Lr"]
SINGLE_T = [round(0.1 * (i + 1), 1) for i in range(12)]
SINGLE_T_NAMES = [f"T{t:.1f}" for t in SINGLE_T]

MODEL_DISPLAY = {
    "Llama-3.2-3B": "Llama-3.2-3B-Instruct",
    "Phi-4-mini-instruct": "Phi-4-mini-instruct",
    "Qwen2.5-3B": "Qwen2.5-3B-Instruct",
    "Qwen3-4B-Instruct-2507": "Qwen3-4B-Instruct-2507",
}

HEADER = (
    "# Dataset-level Performance Comparison\n\n"
    f"N={{{', '.join(str(n) for n in N_VALUES)}}} 기준, stratum=overall, "
    "각 전략별 accuracy 비교.\n\n"
    "전략 설명:\n"
    "- **T=0.1 / T=1.0**: 고정 temperature로 N개 샘플 생성 후 majority vote\n"
    "- **Random T**: 12개 T에서 균등 샘플링 후 majority vote\n"
    "- **\\MethodEqualMix{}**: 12개 T에서 동일 비율 샘플링 후 majority vote\n"
    "- **\\MethodCV{} (Consensus Vote)**: temperature별 majority → 최종 majority\n"
    "- **Best T (oracle)**: 최적 T를 사전에 알고 있는 upper bound\n"
)

TABLE_HEADER = (
    "| Model | T=0.1 | T=1.0 | Random T | \\MethodEqualMix{} | "
    "\\MethodCV{} | Best T (oracle) |\n"
    "|-------|-------|-------|----------|-----------|------|-----------------|"
)

DIFF_TABLE_HEADER = (
    "| Model | Stratum | n | T=0.1 | T=1.0 | Random T | \\MethodEqualMix{} | "
    "\\MethodCV{} | Best T (oracle) |\n"
    "|-------|---------|---|-------|-------|----------|-----------|------|-----------------|"
)


def load_sim_v2(sim_v2_dir: Path) -> pd.DataFrame:
    """Load every per-combo CSV and attach (dataset, model) columns from filename."""
    frames = []
    for csv in sorted(sim_v2_dir.glob("*.csv")):
        m = COMBO_RE.match(csv.stem)
        if not m:
            continue
        df = pd.read_csv(csv)
        df["dataset"] = m.group("dataset")
        df["model"] = m.group("model")
        frames.append(df)
    if not frames:
        raise SystemExit(f"No CSVs in {sim_v2_dir}")
    return pd.concat(frames, ignore_index=True)


def merge_aime(df: pd.DataFrame) -> pd.DataFrame:
    """Add problem-count weighted rows under dataset='aime' across AIME years.

    Matches the weighting in merge_sim_v2.py: mean weighted by n_pids per
    (model, baseline, stratum, N) group.
    """
    aime = df[df.dataset.isin(AIME_YEARS)]
    if aime.empty:
        return df
    rows = []
    for keys, sub in aime.groupby(["model", "baseline", "stratum", "N"]):
        w = sub["n_pids"].to_numpy(dtype=float)
        m = sub["mean"].to_numpy(dtype=float)
        s = sub["std6"].to_numpy(dtype=float)
        valid = ~np.isnan(m) & (w > 0)
        if not valid.any():
            continue
        w = w[valid]; m = m[valid]; s = s[valid]
        tw = w.sum()
        rows.append({
            "model": keys[0], "baseline": keys[1], "stratum": keys[2],
            "N": keys[3], "dataset": "aime",
            "mean": float((w * m).sum() / tw),
            "std6": float(np.sqrt((w * s**2).sum() / tw)),
            "n_pids": int(tw),
            "combo": f"aime_{keys[0]}",
        })
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def pick_best_t(sub: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float]:
    """Argmax mean over 12 single-T baselines with tiny jitter for tie-breaking."""
    st = sub[sub.baseline.isin(SINGLE_T_NAMES)].copy()
    st["T"] = st["baseline"].str[1:].astype(float)
    st = st.sort_values("T")
    means = st["mean"].to_numpy()
    temps = st["T"].to_numpy()
    valid = ~np.isnan(means)
    if not valid.any():
        return float("nan"), float("nan")
    m = means[valid]
    t = temps[valid]
    jitter = rng.uniform(0, 1e-9, size=m.shape)
    idx = int(np.argmax(m + jitter))
    return float(m[idx]), float(t[idx])


def fmt_acc(x: float) -> str:
    return "—" if np.isnan(x) else f"{x:.4f}"


def fmt_best(mean: float, t_star: float) -> str:
    if np.isnan(mean):
        return "—"
    return f"{mean:.4f} (T={t_star:.1f})"


def get_mean(sub: pd.DataFrame, baseline: str) -> float:
    row = sub[sub.baseline == baseline]
    if row.empty:
        return float("nan")
    return float(row["mean"].iloc[0])


def build_difficulty_section(df: pd.DataFrame, dataset: str) -> list[str]:
    lines = [f"## {dataset} — by difficulty", ""]
    ds = df[df.dataset == dataset]
    if ds.empty:
        lines.extend(["_(no data)_", "", "---", ""])
        return lines
    models = sorted(ds["model"].unique(),
                    key=lambda m: MODEL_DISPLAY.get(m, m).lower())
    any_table = False
    for N in N_VALUES:
        sub_N = ds[(ds.N == N) & ds.stratum.isin(STRATA)]
        if sub_N.empty:
            continue
        body = []
        rng = np.random.default_rng(42)
        for model in models:
            for stratum in STRATA:
                sub_ms = sub_N[(sub_N.model == model) & (sub_N.stratum == stratum)]
                if sub_ms.empty:
                    continue
                n_pids = int(sub_ms["n_pids"].iloc[0])
                if n_pids == 0:
                    continue
                t0p1 = get_mean(sub_ms, "T0.1")
                t1p0 = get_mean(sub_ms, "T1.0")
                rand_t = get_mean(sub_ms, "random_T")
                eqmix = get_mean(sub_ms, "equal_mix")
                cv = get_mean(sub_ms, "consensus_vote")
                best_mean, t_star = pick_best_t(sub_ms, rng)
                display = MODEL_DISPLAY.get(model, model)
                body.append(
                    f"| {display} | {stratum} | {n_pids} | {fmt_acc(t0p1)} | "
                    f"{fmt_acc(t1p0)} | {fmt_acc(rand_t)} | {fmt_acc(eqmix)} | "
                    f"{fmt_acc(cv)} | {fmt_best(best_mean, t_star)} |"
                )
        if not body:
            continue
        any_table = True
        lines.append(f"### N={N}")
        lines.append("")
        lines.append(DIFF_TABLE_HEADER)
        lines.extend(body)
        lines.append("")
    if not any_table:
        lines.append("_(no per-stratum data)_")
        lines.append("")
    lines.append("---")
    lines.append("")
    return lines


def build_section(df: pd.DataFrame, dataset: str) -> list[str]:
    lines = [f"## {dataset}", ""]
    ds = df[df.dataset == dataset]
    if ds.empty:
        lines.append("_(no data)_")
        lines.append("")
        return lines
    models = sorted(ds["model"].unique(),
                    key=lambda m: MODEL_DISPLAY.get(m, m).lower())
    for N in N_VALUES:
        sub_N = ds[(ds.N == N) & (ds.stratum == "overall")]
        if sub_N.empty:
            continue
        lines.append(f"### N={N}")
        lines.append("")
        lines.append(TABLE_HEADER)
        rng = np.random.default_rng(42)
        for model in models:
            sub_m = sub_N[sub_N.model == model]
            if sub_m.empty:
                continue
            t0p1 = get_mean(sub_m, "T0.1")
            t1p0 = get_mean(sub_m, "T1.0")
            rand_t = get_mean(sub_m, "random_T")
            eqmix = get_mean(sub_m, "equal_mix")
            cv = get_mean(sub_m, "consensus_vote")
            best_mean, t_star = pick_best_t(sub_m, rng)
            display = MODEL_DISPLAY.get(model, model)
            lines.append(
                f"| {display} | {fmt_acc(t0p1)} | {fmt_acc(t1p0)} | "
                f"{fmt_acc(rand_t)} | {fmt_acc(eqmix)} | {fmt_acc(cv)} | "
                f"{fmt_best(best_mean, t_star)} |"
            )
        lines.append("")
    lines.append("---")
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-v2-dir", type=Path, default=DEFAULT_SIM_V2_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    df = load_sim_v2(args.sim_v2_dir)
    df = merge_aime(df)
    print(f"Loaded {len(df)} rows from {df['dataset'].nunique()} datasets, "
          f"{df['model'].nunique()} models")

    out = [HEADER, "---", ""]
    for ds in DATASET_ORDER:
        out.extend(build_section(df, ds))
    out.append("# Per-difficulty Breakdown")
    out.append("")
    out.append("각 데이터셋에서 difficulty stratum (L1=easy → L5=hard, Lr=정답 한 번도 안나온 unsolvable bucket) 별 동일 비교.")
    out.append("")
    for ds in DATASET_ORDER:
        out.extend(build_difficulty_section(df, ds))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
