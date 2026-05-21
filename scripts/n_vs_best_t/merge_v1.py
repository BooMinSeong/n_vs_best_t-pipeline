"""Merge per-combo v1 CSVs into best_t_table.csv.

Adds acc_v1_mean, acc_v1_std columns at N=256 (other N rows: NaN since v1.0 is
designed for fixed B=256). AIME 4 years merged via problem-count weighted mean.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

V1_DIR = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/v1")
BEST_T_TABLE = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/best_t_table.csv")

COMBO_RE = re.compile(r"^(?P<dataset>aime\d{4}|math1k|mathfull|math500|gsm8kfull)"
                      r"_(?P<model>.+)$")


def load_all(v1_dir: Path) -> pd.DataFrame:
    rows = []
    for csv in sorted(v1_dir.glob("*.csv")):
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
        raise SystemExit("No v1 CSVs found.")
    return pd.concat(rows, ignore_index=True)


def merge_aime(df: pd.DataFrame) -> pd.DataFrame:
    is_aime = df["dataset_year"].str.startswith("aime")
    non_aime = df[~is_aime].copy(); non_aime["dataset"] = non_aime["dataset_year"]
    aime = df[is_aime].copy()
    if aime.empty:
        return non_aime[["model", "dataset", "stratum", "N", "mean", "std6", "n_pids"]]

    rows = []
    for keys, sub in aime.groupby(["model", "stratum", "N"]):
        weights = sub["n_pids"].to_numpy(dtype=float)
        means = sub["mean"].to_numpy(dtype=float)
        stds = sub["std6"].to_numpy(dtype=float)
        valid = ~np.isnan(means) & (weights > 0)
        if not valid.any():
            rows.append({"model": keys[0], "dataset": "aime",
                         "stratum": keys[1], "N": keys[2],
                         "mean": float("nan"), "std6": float("nan"), "n_pids": 0})
            continue
        w = weights[valid]; m = means[valid]; s = stds[valid]
        total_w = w.sum()
        rows.append({"model": keys[0], "dataset": "aime",
                     "stratum": keys[1], "N": keys[2],
                     "mean": float((w * m).sum() / total_w),
                     "std6": float(np.sqrt((w * s**2).sum() / total_w)),
                     "n_pids": int(total_w)})
    merged = pd.DataFrame(rows)
    non_aime_out = non_aime[["model", "dataset", "stratum", "N", "mean", "std6", "n_pids"]]
    return pd.concat([non_aime_out, merged], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-dir", type=Path, default=V1_DIR)
    ap.add_argument("--best-t-table", type=Path, default=BEST_T_TABLE)
    args = ap.parse_args()

    raw = load_all(args.v1_dir)
    print(f"Loaded {len(raw)} rows from {raw['dataset_year'].nunique()} dataset-years")
    merged = merge_aime(raw)
    print(f"After AIME merge: {len(merged)} rows, {merged.groupby(['model','dataset']).ngroups} combos")

    # Save standalone
    v1_all = args.v1_dir.parent / "v1_all.csv"
    merged.sort_values(["model", "dataset", "stratum", "N"]).to_csv(v1_all, index=False)
    print(f"Wrote {v1_all}")

    best = pd.read_csv(args.best_t_table)
    # Drop any previous v1 columns
    best = best.drop(columns=[c for c in ("acc_v1_mean", "acc_v1_std", "gap_vs_v1")
                              if c in best.columns])
    merged_keyed = merged.rename(columns={"mean": "acc_v1_mean", "std6": "acc_v1_std"})[
        ["model", "dataset", "stratum", "N", "acc_v1_mean", "acc_v1_std"]
    ]
    best_merged = best.merge(merged_keyed, on=["model", "dataset", "stratum", "N"], how="left")
    best_merged["gap_vs_v1"] = best_merged["best_t_mean"] - best_merged["acc_v1_mean"]

    best_merged.to_csv(args.best_t_table, index=False)
    print(f"Updated {args.best_t_table}")

    print("\n=== v1.0 @ N=256, overall stratum ===")
    head = best_merged[(best_merged.stratum == "overall") & (best_merged.N == 256)][
        ["model", "dataset", "best_t_mean", "acc_v1_mean", "gap_vs_v1",
         "acc_equal_mix_mean"]
    ]
    head["v1_vs_equal_mix"] = head["acc_v1_mean"] - head["acc_equal_mix_mean"]
    print(head.to_string(index=False))


if __name__ == "__main__":
    main()
