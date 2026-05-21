"""Merge per-combo equal_mix CSVs into best_t_table.csv.

AIME 4 years are merged via problem-count weighted mean per (model, stratum, N),
matching the markdown-based AIME merge in parse_difficulty_reports.py.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

EQUAL_MIX_DIR = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/equal_mix")
BEST_T_TABLE = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/best_t_table.csv")

COMBO_RE = re.compile(r"^(?P<dataset>aime\d{4}|math1k|mathfull|math500|gsm8kfull)"
                      r"_(?P<model>.+)$")


def load_all_combos(equal_mix_dir: Path) -> pd.DataFrame:
    rows = []
    for csv in sorted(equal_mix_dir.glob("*.csv")):
        df = pd.read_csv(csv)
        combo = csv.stem
        m = COMBO_RE.match(combo)
        if not m:
            print(f"SKIP {combo}: cannot parse")
            continue
        df["dataset_year"] = m.group("dataset")
        df["model"] = m.group("model")
        rows.append(df)
    if not rows:
        raise SystemExit("No equal_mix CSVs found.")
    return pd.concat(rows, ignore_index=True)


def merge_aime(df: pd.DataFrame) -> pd.DataFrame:
    is_aime = df["dataset_year"].str.startswith("aime")
    non_aime = df[~is_aime].copy()
    non_aime["dataset"] = non_aime["dataset_year"]
    aime = df[is_aime].copy()
    if aime.empty:
        return non_aime[["model", "dataset", "stratum", "N",
                         "equal_mix_mean", "equal_mix_std6", "n_pids"]]

    rows = []
    for keys, sub in aime.groupby(["model", "stratum", "N"]):
        # problem-count weighted mean across 4 years
        weights = sub["n_pids"].to_numpy(dtype=float)
        means = sub["equal_mix_mean"].to_numpy(dtype=float)
        stds = sub["equal_mix_std6"].to_numpy(dtype=float)
        valid = ~np.isnan(means) & (weights > 0)
        if not valid.any():
            rows.append({"model": keys[0], "dataset": "aime",
                         "stratum": keys[1], "N": keys[2],
                         "equal_mix_mean": float("nan"),
                         "equal_mix_std6": float("nan"), "n_pids": 0})
            continue
        w = weights[valid]; m = means[valid]; s = stds[valid]
        total_w = w.sum()
        mean_a = float((w * m).sum() / total_w)
        std_a = float(np.sqrt((w * s**2).sum() / total_w))
        rows.append({"model": keys[0], "dataset": "aime",
                     "stratum": keys[1], "N": keys[2],
                     "equal_mix_mean": mean_a, "equal_mix_std6": std_a,
                     "n_pids": int(total_w)})
    aime_merged = pd.DataFrame(rows)
    non_aime_out = non_aime[["model", "dataset", "stratum", "N",
                             "equal_mix_mean", "equal_mix_std6", "n_pids"]]
    return pd.concat([non_aime_out, aime_merged], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--equal-mix-dir", type=Path, default=EQUAL_MIX_DIR)
    ap.add_argument("--best-t-table", type=Path, default=BEST_T_TABLE)
    args = ap.parse_args()

    raw = load_all_combos(args.equal_mix_dir)
    print(f"Loaded {len(raw)} rows from {raw['dataset_year'].nunique()} combos")
    merged = merge_aime(raw)
    print(f"After AIME merge: {len(merged)} rows, {merged.groupby(['model','dataset']).ngroups} (model,dataset) combos")

    # Save standalone
    em_out = args.equal_mix_dir.parent / "equal_mix_all.csv"
    merged.sort_values(["model", "dataset", "stratum", "N"]).to_csv(em_out, index=False)
    print(f"Wrote {em_out}")

    # Merge into best_t_table
    best = pd.read_csv(args.best_t_table)
    em_keyed = merged.rename(columns={
        "equal_mix_mean": "_em_mean", "equal_mix_std6": "_em_std",
        "n_pids": "_em_n",
    })
    # drop previously-NaN equal_mix columns from best
    best = best.drop(columns=[c for c in ("acc_equal_mix_mean", "acc_equal_mix_std",
                                          "gap_vs_equal_mix") if c in best.columns])
    best_merged = best.merge(em_keyed, on=["model", "dataset", "stratum", "N"], how="left")
    best_merged["acc_equal_mix_mean"] = best_merged["_em_mean"]
    best_merged["acc_equal_mix_std"] = best_merged["_em_std"]
    best_merged["gap_vs_equal_mix"] = best_merged["best_t_mean"] - best_merged["_em_mean"]
    best_merged = best_merged.drop(columns=["_em_mean", "_em_std", "_em_n"])

    best_merged.to_csv(args.best_t_table, index=False)
    print(f"Updated {args.best_t_table}")

    # Quick summary
    print("\n=== gap vs equal_mix at N=256 (overall) ===")
    head = best_merged[(best_merged.stratum == "overall") & (best_merged.N == 256)][
        ["model", "dataset", "best_t_mean", "acc_equal_mix_mean", "gap_vs_equal_mix"]
    ]
    print(head.to_string(index=False))


if __name__ == "__main__":
    main()
