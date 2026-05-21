"""Merge per-combo sim baseline CSVs (4 baselines) into best_t_table.csv.

REPLACES the markdown-derived T=1.0/T=0.1/random_T values with simulation
values from lb_baselines per_problem_sims, alongside equal_mix.
Markdown-derived values are preserved in a side CSV for sanity check.

AIME 4 years merged via problem-count weighted mean per (model, baseline, stratum, N).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

SIM_DIR = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/sim_baselines")
BEST_T_TABLE = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/best_t_table.csv")

COMBO_RE = re.compile(r"^(?P<dataset>aime\d{4}|math1k|mathfull|math500|gsm8kfull)"
                      r"_(?P<model>.+)$")

# Map sim baseline name → best_t_table column prefix
COL_MAP = {
    "T1.0": "acc_t1p0",
    "T0.1": "acc_t0p1",
    "random_T": "acc_random_t",
    "equal_mix": "acc_equal_mix",
}


def load_all(sim_dir: Path) -> pd.DataFrame:
    rows = []
    for csv in sorted(sim_dir.glob("*.csv")):
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
        raise SystemExit("No sim CSVs found.")
    return pd.concat(rows, ignore_index=True)


def merge_aime(df: pd.DataFrame) -> pd.DataFrame:
    is_aime = df["dataset_year"].str.startswith("aime")
    non_aime = df[~is_aime].copy(); non_aime["dataset"] = non_aime["dataset_year"]
    aime = df[is_aime].copy()
    if aime.empty:
        return non_aime[["model", "dataset", "baseline", "stratum", "N",
                         "mean", "std6", "n_pids"]]

    rows = []
    for keys, sub in aime.groupby(["model", "baseline", "stratum", "N"]):
        weights = sub["n_pids"].to_numpy(dtype=float)
        means = sub["mean"].to_numpy(dtype=float)
        stds = sub["std6"].to_numpy(dtype=float)
        valid = ~np.isnan(means) & (weights > 0)
        if not valid.any():
            rows.append({"model": keys[0], "dataset": "aime",
                         "baseline": keys[1], "stratum": keys[2], "N": keys[3],
                         "mean": float("nan"), "std6": float("nan"), "n_pids": 0})
            continue
        w = weights[valid]; m = means[valid]; s = stds[valid]
        total_w = w.sum()
        mean_a = float((w * m).sum() / total_w)
        std_a = float(np.sqrt((w * s**2).sum() / total_w))
        rows.append({"model": keys[0], "dataset": "aime",
                     "baseline": keys[1], "stratum": keys[2], "N": keys[3],
                     "mean": mean_a, "std6": std_a, "n_pids": int(total_w)})
    aime_merged = pd.DataFrame(rows)
    non_aime_out = non_aime[["model", "dataset", "baseline", "stratum", "N",
                             "mean", "std6", "n_pids"]]
    return pd.concat([non_aime_out, aime_merged], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dir", type=Path, default=SIM_DIR)
    ap.add_argument("--best-t-table", type=Path, default=BEST_T_TABLE)
    args = ap.parse_args()

    raw = load_all(args.sim_dir)
    print(f"Loaded {len(raw)} rows, {raw['dataset_year'].nunique()} dataset-years, "
          f"{raw['baseline'].nunique()} baselines")
    merged = merge_aime(raw)
    print(f"After AIME merge: {len(merged)} rows, "
          f"{merged.groupby(['model','dataset']).ngroups} (model,dataset) combos")

    # Save standalone long-format
    sim_all = args.sim_dir.parent / "sim_baselines_all.csv"
    merged.sort_values(["model", "dataset", "baseline", "stratum", "N"]).to_csv(sim_all, index=False)
    print(f"Wrote {sim_all}")

    # Load best_t_table and preserve OLD (markdown-derived) baseline columns as side CSV
    best = pd.read_csv(args.best_t_table)
    mdcols = ["acc_t1p0_mean", "acc_t1p0_std",
              "acc_t0p1_mean", "acc_t0p1_std",
              "acc_random_t_mean", "acc_random_t_across_t_std", "acc_random_t_seed_std_avg"]
    md_side = best[["model", "dataset", "stratum", "N"] + [c for c in mdcols if c in best.columns]].copy()
    md_side_path = args.sim_dir.parent / "best_t_table_markdown_baselines.csv"
    md_side.to_csv(md_side_path, index=False)
    print(f"Preserved markdown-derived baselines → {md_side_path}")

    # Pivot sim values to wide for merging
    wide = merged.pivot_table(index=["model", "dataset", "stratum", "N"],
                              columns="baseline",
                              values=["mean", "std6"]).reset_index()
    # Flatten MultiIndex columns
    wide.columns = [
        f"{a}_{b}" if b else a for a, b in wide.columns
    ]
    # Rename to best_t_table convention
    rename = {}
    for sim_name, prefix in COL_MAP.items():
        rename[f"mean_{sim_name}"] = f"{prefix}_mean"
        rename[f"std6_{sim_name}"] = f"{prefix}_std"
    wide = wide.rename(columns=rename)

    # Drop existing baseline columns from best and merge in sim ones
    drop_cols = [c for c in best.columns if c.startswith(("acc_t1p0", "acc_t0p1",
                                                            "acc_random_t", "acc_equal_mix",
                                                            "gap_vs_"))]
    best = best.drop(columns=drop_cols)
    keep_sim_cols = ["model", "dataset", "stratum", "N"] + \
        [f"{p}_mean" for p in COL_MAP.values()] + \
        [f"{p}_std" for p in COL_MAP.values()]
    wide = wide[keep_sim_cols]
    best_merged = best.merge(wide, on=["model", "dataset", "stratum", "N"], how="left")

    # Recompute gaps
    best_merged["gap_vs_t1p0"] = best_merged["best_t_mean"] - best_merged["acc_t1p0_mean"]
    best_merged["gap_vs_t0p1"] = best_merged["best_t_mean"] - best_merged["acc_t0p1_mean"]
    best_merged["gap_vs_random_t"] = best_merged["best_t_mean"] - best_merged["acc_random_t_mean"]
    best_merged["gap_vs_equal_mix"] = best_merged["best_t_mean"] - best_merged["acc_equal_mix_mean"]

    best_merged.to_csv(args.best_t_table, index=False)
    print(f"Updated {args.best_t_table} (all 4 baselines from sim)")

    # Sanity check: report new headline numbers
    print("\n=== Sim-based baselines @ N=256, overall ===")
    head = best_merged[(best_merged.stratum == "overall") & (best_merged.N == 256)][
        ["model", "dataset", "t_star", "best_t_mean",
         "acc_t1p0_mean", "acc_t0p1_mean",
         "acc_random_t_mean", "acc_equal_mix_mean"]
    ]
    print(head.to_string(index=False))


if __name__ == "__main__":
    main()
