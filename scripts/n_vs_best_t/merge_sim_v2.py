"""Merge sim_v2 CSVs (Lr stratum + consensus_vote, N to 2048) + v1_multiB CSVs.

Replaces best_t_table.csv content:
- 7 strata (overall, L1..L5, Lr, overall_md_compat)
- N grid {1..2048}
- Sim-derived best-T from 12 single-T baselines
- 4 mix baselines: T=1.0, T=0.1, random_T, equal_mix
- NEW: consensus_vote
- v1.0 across B={128..2048}

AIME merge: problem-count weighted mean per (model, baseline, stratum, N|B).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

SIM_V2_DIR = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/sim_v2")
V1_MB_DIR = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/v1_multiB")
BEST_T_TABLE = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/best_t_table.csv")

COMBO_RE = re.compile(r"^(?P<dataset>aime\d{4}|math1k|mathfull|math500|gsm8kfull)"
                      r"_(?P<model>.+)$")

SINGLE_T = [round(0.1*(i+1), 1) for i in range(12)]
COL_MAP = {
    "T1.0": "acc_t1p0",
    "T0.1": "acc_t0p1",
    "random_T": "acc_random_t",
    "equal_mix": "acc_equal_mix",
    "consensus_vote": "acc_consensus_vote",
}


def load_long_csvs(dir_path: Path, key_name: str = "N") -> pd.DataFrame:
    rows = []
    for csv in sorted(dir_path.glob("*.csv")):
        df = pd.read_csv(csv)
        combo = csv.stem
        m = COMBO_RE.match(combo)
        if not m:
            print(f"SKIP {combo}"); continue
        df["dataset_year"] = m.group("dataset")
        df["model"] = m.group("model")
        rows.append(df)
    if not rows:
        raise SystemExit(f"No CSVs in {dir_path}")
    return pd.concat(rows, ignore_index=True)


def merge_aime(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    """Weighted-mean AIME 4 years per (model, *key_cols)."""
    is_aime = df["dataset_year"].str.startswith("aime")
    non_aime = df[~is_aime].copy(); non_aime["dataset"] = non_aime["dataset_year"]
    aime = df[is_aime].copy()
    out_cols = ["model", "dataset"] + key_cols + ["mean", "std6", "n_pids"]
    if aime.empty:
        return non_aime[out_cols]
    rows = []
    grp_cols = ["model"] + key_cols
    for keys, sub in aime.groupby(grp_cols):
        w = sub["n_pids"].to_numpy(dtype=float)
        m = sub["mean"].to_numpy(dtype=float)
        s = sub["std6"].to_numpy(dtype=float)
        valid = ~np.isnan(m) & (w > 0)
        if not valid.any():
            row = dict(zip(grp_cols, keys))
            row.update(dict(dataset="aime", mean=float("nan"),
                            std6=float("nan"), n_pids=0))
            rows.append(row); continue
        w = w[valid]; m = m[valid]; s = s[valid]
        tw = w.sum()
        row = dict(zip(grp_cols, keys))
        row.update(dict(dataset="aime",
                        mean=float((w * m).sum() / tw),
                        std6=float(np.sqrt((w * s**2).sum() / tw)),
                        n_pids=int(tw)))
        rows.append(row)
    aime_merged = pd.DataFrame(rows)
    return pd.concat([non_aime[out_cols], aime_merged[out_cols]], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-v2-dir", type=Path, default=SIM_V2_DIR)
    ap.add_argument("--v1-mb-dir", type=Path, default=V1_MB_DIR)
    ap.add_argument("--best-t-table", type=Path, default=BEST_T_TABLE)
    args = ap.parse_args()

    # ============ sim_v2 ============
    sim_raw = load_long_csvs(args.sim_v2_dir, key_name="N")
    print(f"sim_v2: {len(sim_raw)} rows, baselines={sim_raw['baseline'].nunique()}")
    sim_merged = merge_aime(sim_raw, key_cols=["baseline", "stratum", "N"])
    sim_path = args.sim_v2_dir.parent / "sim_v2_all.csv"
    sim_merged.sort_values(["model", "dataset", "stratum", "baseline", "N"]).to_csv(sim_path, index=False)
    print(f"Wrote {sim_path} ({len(sim_merged)} rows)")

    # ============ Sim-derived best-T per (model, dataset, stratum, N) ============
    single_t_names = [f"T{t:.1f}" for t in SINGLE_T]
    st = sim_merged[sim_merged["baseline"].isin(single_t_names)].copy()
    st["T"] = st["baseline"].str[1:].astype(float)

    best_rows = []
    rng = np.random.default_rng(42)
    for keys, sub in st.groupby(["model", "dataset", "stratum", "N"]):
        sub = sub.sort_values("T")
        temps = sub["T"].to_numpy()
        means = sub["mean"].to_numpy()
        stds = sub["std6"].to_numpy()
        n_prob = int(sub["n_pids"].iloc[0])
        valid = ~np.isnan(means)
        if not valid.any():
            continue
        means_v = means[valid]; temps_v = temps[valid]; stds_v = stds[valid]
        jittered = means_v + rng.uniform(0, 1e-9, size=means_v.shape)
        idx = int(np.argmax(jittered))
        best_rows.append({
            "model": keys[0], "dataset": keys[1], "stratum": keys[2], "N": keys[3],
            "best_t_mean": float(means_v[idx]),
            "best_t_seed_std": float(stds_v[idx]),
            "t_star": float(temps_v[idx]),
            "level_n_problems": n_prob,
        })
    best_df = pd.DataFrame(best_rows)
    best_df["dataset_family"] = best_df["dataset"]

    # ============ Pivot mix baselines to wide ============
    mix = sim_merged[sim_merged["baseline"].isin(list(COL_MAP.keys()))].copy()
    wide = mix.pivot_table(index=["model", "dataset", "stratum", "N"],
                           columns="baseline", values=["mean", "std6"]).reset_index()
    wide.columns = [f"{a}_{b}" if b else a for a, b in wide.columns]
    rename = {}
    for sim_name, prefix in COL_MAP.items():
        rename[f"mean_{sim_name}"] = f"{prefix}_mean"
        rename[f"std6_{sim_name}"] = f"{prefix}_std"
    wide = wide.rename(columns=rename)
    keep = ["model", "dataset", "stratum", "N"] + \
        [f"{p}_mean" for p in COL_MAP.values()] + \
        [f"{p}_std" for p in COL_MAP.values()]
    wide = wide[keep]

    new_table = best_df.merge(wide, on=["model", "dataset", "stratum", "N"], how="outer")

    # ============ v1 multi-B ============
    if args.v1_mb_dir.exists() and any(args.v1_mb_dir.glob("*.csv")):
        v1_raw = load_long_csvs(args.v1_mb_dir, key_name="B")
        print(f"v1 multi-B: {len(v1_raw)} rows")
        v1_merged = merge_aime(v1_raw, key_cols=["B", "stratum"])
        v1_merged = v1_merged.rename(columns={"B": "N", "mean": "acc_v1_mean",
                                              "std6": "acc_v1_std"})
        v1_path = args.v1_mb_dir.parent / "v1_multiB_all.csv"
        v1_merged.to_csv(v1_path, index=False)
        v1_keyed = v1_merged[["model", "dataset", "stratum", "N",
                              "acc_v1_mean", "acc_v1_std"]]
        new_table = new_table.merge(v1_keyed, on=["model", "dataset", "stratum", "N"], how="left")

    # ============ Best fixed T (use T* at N=2048 for each combo/stratum) ============
    max_n = 2048
    fixed_t_map = {}  # (model, dataset, stratum) → best_fixed_t
    for (model, dataset, stratum), grp in best_df.groupby(["model", "dataset", "stratum"]):
        row_max = grp[grp.N == max_n]
        if row_max.empty:
            # fallback: use largest available N
            row_max = grp.loc[grp.N.idxmax()]
            fixed_t_map[(model, dataset, stratum)] = float(row_max["t_star"] if isinstance(row_max, pd.Series) else row_max.iloc[0]["t_star"])
        else:
            fixed_t_map[(model, dataset, stratum)] = float(row_max.iloc[0]["t_star"])

    # For each row, look up fixed T and get the accuracy at that fixed T from single-T data
    fixed_t_col = []
    fixed_t_mean_col = []
    fixed_t_std_col = []
    for _, row in new_table.iterrows():
        key = (row["model"], row["dataset"], row["stratum"])
        ft = fixed_t_map.get(key)
        if ft is None:
            fixed_t_col.append(np.nan)
            fixed_t_mean_col.append(np.nan)
            fixed_t_std_col.append(np.nan)
            continue
        fixed_t_col.append(ft)
        # Look up accuracy at this fixed T for this (model, dataset, stratum, N)
        t_name = f"T{ft:.1f}"
        lookup = st[(st.model == row["model"]) & (st.dataset == row["dataset"])
                     & (st.stratum == row["stratum"]) & (st.N == row["N"])
                     & (st.baseline == t_name)]
        if not lookup.empty:
            fixed_t_mean_col.append(float(lookup.iloc[0]["mean"]))
            fixed_t_std_col.append(float(lookup.iloc[0]["std6"]))
        else:
            fixed_t_mean_col.append(np.nan)
            fixed_t_std_col.append(np.nan)

    new_table["best_fixed_t"] = fixed_t_col
    new_table["best_fixed_t_mean"] = fixed_t_mean_col
    new_table["best_fixed_t_std"] = fixed_t_std_col

    # ============ Gaps ============
    new_table["gap_vs_t1p0"] = new_table["best_t_mean"] - new_table["acc_t1p0_mean"]
    new_table["gap_vs_t0p1"] = new_table["best_t_mean"] - new_table["acc_t0p1_mean"]
    new_table["gap_vs_random_t"] = new_table["best_t_mean"] - new_table["acc_random_t_mean"]
    new_table["gap_vs_equal_mix"] = new_table["best_t_mean"] - new_table["acc_equal_mix_mean"]
    new_table["gap_vs_consensus_vote"] = new_table["best_t_mean"] - new_table["acc_consensus_vote_mean"]
    if "acc_v1_mean" in new_table.columns:
        new_table["gap_vs_v1"] = new_table["best_t_mean"] - new_table["acc_v1_mean"]

    # Gaps for best_fixed_t
    new_table["gap_fixed_vs_t1p0"] = new_table["best_fixed_t_mean"] - new_table["acc_t1p0_mean"]
    new_table["gap_fixed_vs_t0p1"] = new_table["best_fixed_t_mean"] - new_table["acc_t0p1_mean"]
    new_table["gap_fixed_vs_random_t"] = new_table["best_fixed_t_mean"] - new_table["acc_random_t_mean"]
    new_table["gap_fixed_vs_equal_mix"] = new_table["best_fixed_t_mean"] - new_table["acc_equal_mix_mean"]
    new_table["gap_fixed_vs_consensus_vote"] = new_table["best_fixed_t_mean"] - new_table["acc_consensus_vote_mean"]

    new_table = new_table.sort_values(["model", "dataset", "stratum", "N"]).reset_index(drop=True)
    new_table.to_csv(args.best_t_table, index=False)
    print(f"Updated {args.best_t_table} ({len(new_table)} rows)")

    # ============ Summary at N=256, overall (incl Lr) ============
    print("\n=== Summary @ N=256, overall (incl Lr) ===")
    ov = new_table[(new_table.stratum == "overall") & (new_table.N == 256)][
        ["model", "dataset", "level_n_problems", "t_star", "best_t_mean",
         "acc_t1p0_mean", "acc_consensus_vote_mean", "acc_equal_mix_mean",
         "acc_v1_mean"]]
    print(ov.to_string(index=False))

    print("\n=== overall (incl Lr) vs overall_md_compat at N=256 ===")
    for stratum in ["overall", "overall_md_compat"]:
        sub = new_table[(new_table.stratum == stratum) & (new_table.N == 256)][
            ["model", "dataset", "level_n_problems", "best_t_mean"]
        ].rename(columns={"best_t_mean": f"best_{stratum}",
                          "level_n_problems": f"n_{stratum}"})
        print(stratum, sub.to_string(index=False))


if __name__ == "__main__":
    main()
