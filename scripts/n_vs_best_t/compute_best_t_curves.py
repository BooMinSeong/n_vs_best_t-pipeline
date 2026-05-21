"""Compute best-T curve + 3 marginal baselines (T=1.0, T=0.1, random_T) per (model, dataset, stratum, N).

Reads outputs/n_vs_best_t/long/long_all.csv. Writes outputs/n_vs_best_t/best_t_table.csv
with equal_mix columns left as NaN (filled by Phase 2b).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_LONG = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/long/long_all.csv")
DEFAULT_OUT = Path("/home3/b.ms/projects/tts_analysis/outputs/n_vs_best_t/best_t_table.csv")

EXPECTED_TEMPS = [round(0.1 * i, 1) for i in range(1, 13)]


def best_t_with_jitter(temps: np.ndarray, means: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Argmax with tiny jitter to break ties uniformly across T (feedback_argsort_tiebreak_bias)."""
    jittered = means + rng.uniform(0, 1e-9, size=means.shape)
    idx = int(np.argmax(jittered))
    return float(temps[idx]), float(means[idx])


def bootstrap_t_star(temps: np.ndarray, means: np.ndarray, stds: np.ndarray,
                      n_reps: int = 500, rng_seed: int = 42) -> tuple[float, float]:
    """Bootstrap T* by resampling each T's accuracy ~ N(mean, std). Return (low, high) 80% CI."""
    rng = np.random.default_rng(rng_seed)
    t_stars = []
    for _ in range(n_reps):
        noisy = rng.normal(means, np.maximum(stds, 1e-6))
        # jitter for tie-break
        noisy_j = noisy + rng.uniform(0, 1e-9, size=noisy.shape)
        t_stars.append(temps[int(np.argmax(noisy_j))])
    arr = np.array(t_stars)
    return float(np.quantile(arr, 0.1)), float(np.quantile(arr, 0.9))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long-csv", type=Path, default=DEFAULT_LONG)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--bootstrap-reps", type=int, default=500)
    args = ap.parse_args()

    df = pd.read_csv(args.long_csv)
    print(f"Loaded {len(df)} rows from {args.long_csv}")

    rng = np.random.default_rng(42)
    rows = []
    for keys, sub in df.groupby(["model", "dataset", "dataset_family", "stratum", "N"]):
        model, dataset, family, stratum, N = keys
        sub = sub.sort_values("T")
        # require full 12-T grid for marginal baselines
        if len(sub) != 12 or list(sub["T"].round(1).tolist()) != EXPECTED_TEMPS:
            # might happen for some (level, N) — skip
            continue

        temps = sub["T"].to_numpy()
        means = sub["mean_acc"].to_numpy()
        stds = sub["std_acc"].to_numpy()
        n_prob = int(sub["level_n_problems"].iloc[0])

        # best-T
        t_star, best_mean = best_t_with_jitter(temps, means, rng)
        # winner curse: also report best mean's seed-std
        best_std = float(stds[np.argmin(np.abs(temps - t_star))])
        t_low, t_high = bootstrap_t_star(temps, means, stds, n_reps=args.bootstrap_reps, rng_seed=42)

        # T=1.0
        idx_t1 = int(np.argmin(np.abs(temps - 1.0)))
        acc_t1, std_t1 = float(means[idx_t1]), float(stds[idx_t1])
        # T=0.1
        idx_t0p1 = int(np.argmin(np.abs(temps - 0.1)))
        acc_t0p1, std_t0p1 = float(means[idx_t0p1]), float(stds[idx_t0p1])

        # random_T expectation
        acc_random = float(np.mean(means))
        # variance: across-T spread + average seed noise, reported separately
        std_random_across_T = float(np.std(means, ddof=0))
        std_random_seed = float(np.mean(stds))

        rows.append({
            "model": model,
            "dataset": dataset,
            "dataset_family": family,
            "stratum": stratum,
            "level_n_problems": n_prob,
            "N": int(N),
            "best_t_mean": best_mean,
            "best_t_seed_std": best_std,
            "t_star": t_star,
            "t_star_ci_low": t_low,
            "t_star_ci_high": t_high,
            "acc_t1p0_mean": acc_t1,
            "acc_t1p0_std": std_t1,
            "acc_t0p1_mean": acc_t0p1,
            "acc_t0p1_std": std_t0p1,
            "acc_random_t_mean": acc_random,
            "acc_random_t_across_t_std": std_random_across_T,
            "acc_random_t_seed_std_avg": std_random_seed,
            "acc_equal_mix_mean": np.nan,
            "acc_equal_mix_std": np.nan,
            "gap_vs_t1p0": best_mean - acc_t1,
            "gap_vs_t0p1": best_mean - acc_t0p1,
            "gap_vs_random_t": best_mean - acc_random,
            "gap_vs_equal_mix": np.nan,
        })

    out_df = pd.DataFrame(rows).sort_values(
        ["model", "dataset", "stratum", "N"]
    ).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(out_df)} rows)")

    # Quick summary at N=256 overall
    print("\n=== best-T at N=256 (overall stratum) ===")
    head = out_df[(out_df.stratum == "overall") & (out_df.N == 256)][
        ["model", "dataset", "t_star", "best_t_mean",
         "acc_t1p0_mean", "acc_t0p1_mean", "acc_random_t_mean",
         "gap_vs_t1p0", "gap_vs_t0p1", "gap_vs_random_t"]
    ]
    print(head.to_string(index=False))


if __name__ == "__main__":
    main()
