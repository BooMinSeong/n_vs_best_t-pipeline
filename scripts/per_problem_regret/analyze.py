"""Stage 2 — oracle, regret, and distribution analysis (spec sections 2.2-2.9).

Reads the per-combo a_hat parquets produced by compute_a_hat.py and emits:

    per_problem_oracle.parquet            (model,dataset,problem_id,N,T_star,a_star)
    per_problem_regret.parquet            (...,strategy,regret_R1,regret_R2)
    summary_distribution_stats.csv        per (model,dataset,N,strategy) x {R1,R2}
    summary_oracle_gap.csv                dataset vs per-problem oracle headroom
    summary_t_star_distribution.csv       T*(p) histogram
    summary_t_star_conditional_regret.csv regret conditioned on native T*
    summary_paired_comparison.csv         Wilcoxon + BH-FDR + win/tie/loss
    summary_stochastic_dominance.csv      first-order dominance of regret CDFs

R1 = per-problem oracle regret   = a_star(p,N) - a_hat(p,s,N)
R2 = dataset-level oracle regret = a_hat(p, T*_dataset, N) - a_hat(p,s,N)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from per_problem_regret import config as C

SINGLE_T = set(C.SINGLE_T_NAMES)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_a_hat(out_dir: Path, combos: list[str] | None = None) -> pd.DataFrame:
    files = sorted(out_dir.glob("a_hat_per_problem__*.parquet"))
    if combos:
        files = [f for f in files
                 if f.stem.replace("a_hat_per_problem__", "") in combos]
    if not files:
        raise FileNotFoundError(f"no a_hat parquets in {out_dir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


# --------------------------------------------------------------------------- #
# Oracle, dataset-best-T, regret  (per model/dataset/N)
# --------------------------------------------------------------------------- #
def per_problem_oracle(g: pd.DataFrame) -> pd.DataFrame:
    """g: rows for one (model,dataset,N). Return per-problem T_star/a_star over single-T."""
    st = g[g["strategy"].isin(SINGLE_T)]
    idx = st.groupby("problem_id")["a_hat"].idxmax()
    out = st.loc[idx, ["problem_id", "strategy", "a_hat"]].copy()
    out = out.rename(columns={"strategy": "T_star_name", "a_hat": "a_star"})
    out["T_star"] = out["T_star_name"].str.removeprefix("T").astype(float)
    return out.drop(columns="T_star_name")


def dataset_best_t(g: pd.DataFrame) -> str:
    """Strategy name of the dataset-level oracle (argmax_T mean_p a_hat)."""
    st = g[g["strategy"].isin(SINGLE_T)]
    means = st.groupby("strategy")["a_hat"].mean()
    return means.idxmax()


def build_regret(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (oracle_table, regret_table, dataset_best_t_map). Fully vectorised
    (no per-row iteration) so it scales to mathfull-size combos."""
    oracle_parts, regret_parts = [], []
    best_t_map: dict[tuple, str] = {}
    for (model, dataset, N), g in df.groupby(["model", "dataset", "N"], sort=False):
        orc = per_problem_oracle(g)
        best_t = dataset_best_t(g)
        best_t_map[(model, dataset, N)] = best_t
        a_star_s = orc.set_index("problem_id")["a_star"]
        # per-problem accuracy at the dataset-optimal fixed T (= best_fixed_T strategy)
        bft_s = g.loc[g.strategy == best_t].set_index("problem_id")["a_hat"]

        o = orc.copy()
        o["model"], o["dataset"], o["N"] = model, dataset, N
        oracle_parts.append(o[["model", "dataset", "N", "problem_id", "T_star", "a_star"]])

        bft_rows = g[g.strategy == best_t].copy()
        bft_rows["strategy"] = "best_fixed_T"
        gg = pd.concat([g, bft_rows], ignore_index=True)[
            ["model", "dataset", "N", "problem_id", "strategy", "a_hat"]].copy()
        gg["regret_R1"] = gg.problem_id.map(a_star_s).to_numpy() - gg.a_hat
        gg["regret_R2"] = gg.problem_id.map(bft_s).to_numpy() - gg.a_hat
        regret_parts.append(gg)

    oracle = pd.concat(oracle_parts, ignore_index=True)
    regret = pd.concat(regret_parts, ignore_index=True)
    return oracle, regret, best_t_map


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def _dist_stats(x: np.ndarray) -> dict:
    return {
        "n_problems": len(x),
        "regret_mean": float(np.mean(x)),
        "regret_median": float(np.median(x)),
        "regret_std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "regret_p25": float(np.percentile(x, 25)),
        "regret_p75": float(np.percentile(x, 75)),
        "regret_p90": float(np.percentile(x, 90)),
        "regret_p95": float(np.percentile(x, 95)),
        "regret_p99": float(np.percentile(x, 99)),
        "regret_max": float(np.max(x)),
        "regret_gt_10pp": float(np.mean(x > 0.1)),
        "regret_gt_20pp": float(np.mean(x > 0.2)),
    }


def distribution_stats(regret: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, dataset, N, strat), g in regret.groupby(
            ["model", "dataset", "N", "strategy"]):
        for rcol, rname in [("regret_R1", "R1"), ("regret_R2", "R2")]:
            row = {"model": model, "dataset": dataset, "N": N,
                   "strategy": strat, "regret_kind": rname}
            row.update(_dist_stats(g[rcol].to_numpy()))
            rows.append(row)
    return pd.DataFrame(rows)


def oracle_gap(df: pd.DataFrame, oracle: pd.DataFrame, best_t_map: dict) -> pd.DataFrame:
    rows = []
    for (model, dataset, N), g in df.groupby(["model", "dataset", "N"]):
        best_t = best_t_map[(model, dataset, N)]
        ds_oracle = g[g["strategy"] == best_t]["a_hat"].mean()
        pp = oracle.query("model==@model and dataset==@dataset and N==@N")
        pp_oracle = pp["a_star"].mean()
        rows.append({
            "model": model, "dataset": dataset, "N": N,
            "dataset_best_t": best_t,
            "dataset_oracle_acc": float(ds_oracle),
            "per_problem_oracle_acc": float(pp_oracle),
            "gap_pp": float((pp_oracle - ds_oracle) * 100),
        })
    return pd.DataFrame(rows)


def t_star_distribution(oracle: pd.DataFrame) -> pd.DataFrame:
    return (oracle.groupby(["model", "dataset", "N", "T_star"])
            .size().reset_index(name="n_problems"))


def t_star_conditional_regret(regret: pd.DataFrame, oracle: pd.DataFrame) -> pd.DataFrame:
    merged = regret.merge(
        oracle[["model", "dataset", "N", "problem_id", "T_star"]],
        on=["model", "dataset", "N", "problem_id"])
    rows = []
    for (model, dataset, N, strat, tstar), g in merged.groupby(
            ["model", "dataset", "N", "strategy", "T_star"]):
        for rcol, rname in [("regret_R1", "R1"), ("regret_R2", "R2")]:
            x = g[rcol].to_numpy()
            rows.append({
                "model": model, "dataset": dataset, "N": N, "strategy": strat,
                "T_star": tstar, "regret_kind": rname,
                "mean": float(np.mean(x)), "median": float(np.median(x)),
                "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
                "count": len(x),
            })
    return pd.DataFrame(rows)


def paired_comparison(df: pd.DataFrame, regret: pd.DataFrame,
                      best_t_map: dict, cfg: C.Config) -> pd.DataFrame:
    """Per (model,dataset,N): Delta(p)=a_hat(s1)-a_hat(s2) for representative pairs."""
    # a_hat per (model,dataset,N,strategy,problem_id), incl. best_fixed_T
    ah = regret[["model", "dataset", "N", "problem_id", "strategy", "a_hat"]]
    rows = []
    for (model, dataset, N), g in ah.groupby(["model", "dataset", "N"]):
        wide = g.pivot_table(index="problem_id", columns="strategy", values="a_hat")
        block = []
        for s1, s2 in C.PAIRED_COMPARISONS:
            if s1 not in wide or s2 not in wide:
                continue
            d = (wide[s1] - wide[s2]).dropna().to_numpy()
            if len(d) == 0:
                continue
            try:
                w_p = stats.wilcoxon(d, alternative="two-sided",
                                     zero_method="wilcox").pvalue
            except ValueError:  # all-zero differences
                w_p = 1.0
            if not np.isfinite(w_p):  # degenerate (e.g. all ties)
                w_p = 1.0
            thr = cfg.win_tie_threshold
            block.append({
                "model": model, "dataset": dataset, "N": N,
                "strategy_a": s1, "strategy_b": s2,
                "delta_mean": float(d.mean()),
                "delta_std": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                "delta_p25": float(np.percentile(d, 25)),
                "delta_median": float(np.median(d)),
                "delta_p75": float(np.percentile(d, 75)),
                "n_problems": len(d),
                "wins_a": int(np.sum(d > thr)),
                "ties": int(np.sum(np.abs(d) <= thr)),
                "wins_b": int(np.sum(d < -thr)),
                "wilcoxon_p": float(w_p),
            })
        # Benjamini-Hochberg FDR within this (model,dataset,N)
        if block:
            pvals = np.array([b["wilcoxon_p"] for b in block])
            order = np.argsort(pvals)
            m = len(pvals)
            adj = np.empty(m)
            prev = 1.0
            for rank, i in enumerate(order[::-1]):
                k = m - rank
                prev = min(prev, pvals[i] * m / k)
                adj[i] = prev
            for b, a in zip(block, adj):
                b["adj_p_value"] = float(a)
            rows.extend(block)
    return pd.DataFrame(rows)


def stochastic_dominance(regret: pd.DataFrame, cfg: C.Config) -> pd.DataFrame:
    """First-order stochastic dominance of regret CDFs (lower regret = better).

    A dominates B iff F_A(x) >= F_B(x) for all x (A more mass at low regret).
    """
    strat_set = ["T1.0", "equal_mix", "consensus_vote", "random_T", "best_fixed_T"]
    rows = []
    for (model, dataset, N), g in regret.groupby(["model", "dataset", "N"]):
        for rcol, rname in [("regret_R1", "R1"), ("regret_R2", "R2")]:
            by = {s: g[g.strategy == s][rcol].to_numpy()
                  for s in strat_set if (g.strategy == s).any()}
            present = list(by)
            for i, a in enumerate(present):
                for b in present:
                    if a == b:
                        continue
                    ra, rb = by[a], by[b]
                    hi = max(ra.max(), rb.max(), 1e-9)
                    grid = np.linspace(min(ra.min(), rb.min(), 0.0), hi,
                                       cfg.dominance_n_grid)
                    cdf_a = (ra[:, None] <= grid[None, :]).mean(axis=0)
                    cdf_b = (rb[:, None] <= grid[None, :]).mean(axis=0)
                    violations = cdf_b - cdf_a  # >0 => B better at that point
                    rows.append({
                        "model": model, "dataset": dataset, "N": N,
                        "regret_kind": rname,
                        "strategy_a": a, "strategy_b": b,
                        "a_dominates_b": bool(violations.max() <= 1e-9),
                        "max_violation": float(violations.max()),
                    })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(out_dir: Path, combos: list[str] | None, cfg: C.Config) -> dict:
    df = load_a_hat(out_dir, combos)
    oracle, regret, best_t_map = build_regret(df)

    oracle.to_parquet(out_dir / "per_problem_oracle.parquet", index=False)
    regret.drop(columns="a_hat").to_parquet(
        out_dir / "per_problem_regret.parquet", index=False)

    tables = {
        "summary_distribution_stats": distribution_stats(regret),
        "summary_oracle_gap": oracle_gap(df, oracle, best_t_map),
        "summary_t_star_distribution": t_star_distribution(oracle),
        "summary_t_star_conditional_regret": t_star_conditional_regret(regret, oracle),
        "summary_paired_comparison": paired_comparison(df, regret, best_t_map, cfg),
        "summary_stochastic_dominance": stochastic_dominance(regret, cfg),
    }
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"wrote {name}.csv ({len(t)} rows)")
    return {"a_hat": df, "oracle": oracle, "regret": regret,
            "best_t_map": best_t_map, **tables}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=C.OUT_DIR)
    ap.add_argument("--combos", default=None, help="comma list; default = all present")
    args = ap.parse_args()
    combos = args.combos.split(",") if args.combos else None
    run(args.out_dir, combos, C.Config())


if __name__ == "__main__":
    main()
