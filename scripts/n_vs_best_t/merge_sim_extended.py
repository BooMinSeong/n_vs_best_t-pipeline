"""Merge extended sim CSVs (12 single-T + 2 mix baselines, N up to 1536).

For each (model, dataset, stratum, N):
- Computes sim-derived best-T from 12 single-T accuracies (replaces markdown best-T)
- Updates 4 baseline columns (T1.0, T0.1, random_T, equal_mix)
- Adds N>256 rows (markdown didn't have these)

Preserves markdown-derived best-T as best_t_mean_md / t_star_md for comparison.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

SIM_DIR = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/sim_extended")
BEST_T_TABLE = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/best_t_table.csv")

COMBO_RE = re.compile(r"^(?P<dataset>aime\d{4}|math1k|mathfull|math500|gsm8kfull)"
                      r"_(?P<model>.+)$")

SINGLE_T = [round(0.1*(i+1), 1) for i in range(12)]
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
            print(f"SKIP {combo}: cannot parse"); continue
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
        rows.append({"model": keys[0], "dataset": "aime",
                     "baseline": keys[1], "stratum": keys[2], "N": keys[3],
                     "mean": float((w * m).sum() / total_w),
                     "std6": float(np.sqrt((w * s**2).sum() / total_w)),
                     "n_pids": int(total_w)})
    merged = pd.DataFrame(rows)
    non_aime_out = non_aime[["model", "dataset", "baseline", "stratum", "N",
                             "mean", "std6", "n_pids"]]
    return pd.concat([non_aime_out, merged], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dir", type=Path, default=SIM_DIR)
    ap.add_argument("--best-t-table", type=Path, default=BEST_T_TABLE)
    args = ap.parse_args()

    raw = load_all(args.sim_dir)
    print(f"Loaded {len(raw)} rows, baselines={raw['baseline'].nunique()}")
    merged = merge_aime(raw)

    # Save full sim long-format
    sim_all = args.sim_dir.parent / "sim_extended_all.csv"
    merged.sort_values(["model", "dataset", "stratum", "baseline", "N"]).to_csv(sim_all, index=False)
    print(f"Wrote {sim_all} ({len(merged)} rows)")

    # === Compute sim-derived best-T per (model, dataset, stratum, N) ===
    # Filter to single-T baselines only
    single_t_names = [f"T{t:.1f}" for t in SINGLE_T]
    single_t = merged[merged["baseline"].isin(single_t_names)].copy()
    single_t["T"] = single_t["baseline"].str[1:].astype(float)

    best_rows = []
    rng = np.random.default_rng(42)
    for keys, sub in single_t.groupby(["model", "dataset", "stratum", "N"]):
        sub = sub.sort_values("T")
        if len(sub) == 0:
            continue
        temps = sub["T"].to_numpy()
        means = sub["mean"].to_numpy()
        stds = sub["std6"].to_numpy()
        n_prob = int(sub["n_pids"].iloc[0])
        # jitter for tie-break
        jittered = means + rng.uniform(0, 1e-9, size=means.shape)
        valid = ~np.isnan(means)
        if not valid.any():
            continue
        means_v = means[valid]; temps_v = temps[valid]; stds_v = stds[valid]
        jittered_v = jittered[valid]
        idx = int(np.argmax(jittered_v))
        best_rows.append({
            "model": keys[0], "dataset": keys[1], "stratum": keys[2], "N": keys[3],
            "best_t_mean_sim": float(means_v[idx]),
            "best_t_std_sim": float(stds_v[idx]),
            "t_star_sim": float(temps_v[idx]),
            "level_n_problems_sim": n_prob,
        })
    best_df = pd.DataFrame(best_rows)

    # === Load existing best_t_table, preserve markdown best-T, replace with sim ===
    existing = pd.read_csv(args.best_t_table)
    # Rename current best-T columns to _md (markdown-derived) for preservation
    rename_md = {
        "best_t_mean": "best_t_mean_md",
        "best_t_seed_std": "best_t_seed_std_md",
        "t_star": "t_star_md",
        "t_star_ci_low": "t_star_ci_low_md",
        "t_star_ci_high": "t_star_ci_high_md",
    }
    for k, v in rename_md.items():
        if k in existing.columns and v not in existing.columns:
            existing = existing.rename(columns={k: v})

    # === Build new wide sim baselines table for the 4 mix/headline series ===
    mix = merged[merged["baseline"].isin(list(COL_MAP.keys()))].copy()
    wide = mix.pivot_table(index=["model", "dataset", "stratum", "N"],
                           columns="baseline",
                           values=["mean", "std6"]).reset_index()
    wide.columns = [f"{a}_{b}" if b else a for a, b in wide.columns]
    rename = {}
    for sim_name, prefix in COL_MAP.items():
        rename[f"mean_{sim_name}"] = f"{prefix}_mean"
        rename[f"std6_{sim_name}"] = f"{prefix}_std"
    wide = wide.rename(columns=rename)
    keep_cols = ["model", "dataset", "stratum", "N"] + \
        [f"{p}_mean" for p in COL_MAP.values()] + \
        [f"{p}_std" for p in COL_MAP.values()]
    wide = wide[keep_cols]

    # === Build NEW best_t_table by outer-joining sim best-T and sim baselines ===
    # Start from sim_best as the row skeleton (covers all N including >256)
    new_table = best_df.merge(wide, on=["model", "dataset", "stratum", "N"], how="outer")
    # Drop sim_best’s level_n_problems column → use the one from merged single_t already in best_df
    new_table = new_table.rename(columns={
        "best_t_mean_sim": "best_t_mean",
        "best_t_std_sim": "best_t_seed_std",
        "t_star_sim": "t_star",
        "level_n_problems_sim": "level_n_problems",
    })
    new_table["dataset_family"] = new_table["dataset"]

    # Bring back markdown best-T (and v1.0) by merging on (model, dataset, stratum, N)
    md_cols = [c for c in existing.columns if c.endswith("_md") or c in
               ("acc_v1_mean", "acc_v1_std", "gap_vs_v1")]
    keep_existing = ["model", "dataset", "stratum", "N"] + md_cols
    keep_existing = list(dict.fromkeys(keep_existing))  # dedupe
    keep_existing = [c for c in keep_existing if c in existing.columns]
    if md_cols:
        new_table = new_table.merge(existing[keep_existing],
                                    on=["model", "dataset", "stratum", "N"], how="left")

    # Recompute gaps using NEW (sim) best-T
    new_table["gap_vs_t1p0"] = new_table["best_t_mean"] - new_table["acc_t1p0_mean"]
    new_table["gap_vs_t0p1"] = new_table["best_t_mean"] - new_table["acc_t0p1_mean"]
    new_table["gap_vs_random_t"] = new_table["best_t_mean"] - new_table["acc_random_t_mean"]
    new_table["gap_vs_equal_mix"] = new_table["best_t_mean"] - new_table["acc_equal_mix_mean"]
    if "acc_v1_mean" in new_table.columns:
        new_table["gap_vs_v1"] = new_table["best_t_mean"] - new_table["acc_v1_mean"]

    new_table = new_table.sort_values(["model", "dataset", "stratum", "N"]).reset_index(drop=True)
    new_table.to_csv(args.best_t_table, index=False)
    print(f"Updated {args.best_t_table} ({len(new_table)} rows)")

    print("\n=== Sim-derived best-T (extended N grid) — overall stratum ===")
    head = new_table[(new_table.stratum == "overall")].pivot_table(
        index=["model", "dataset"], columns="N", values="t_star")
    print(head.to_string(float_format=lambda x: f"{x:.1f}"))

    print("\n=== best-T accuracy @ N=512, 1024, 1536 (overall) ===")
    snap = new_table[(new_table.stratum == "overall") &
                     (new_table.N.isin([256, 512, 1024, 1536]))][
        ["model", "dataset", "N", "t_star", "best_t_mean",
         "acc_equal_mix_mean", "acc_t1p0_mean"]]
    print(snap.to_string(index=False))


if __name__ == "__main__":
    main()
