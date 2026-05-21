"""Run simulation for ONE combo capturing all 4 baselines from the same MC pass.

Reuses lb_baselines.per_problem_sims (computes T1.0, every single-T, random_T,
equal_mix in one shot from raw distributions.json).

Outputs long-format CSV: combo, N, stratum, baseline, mean, std6, n_pids
where baseline ∈ {T1.0, T0.1, random_T, equal_mix}.
overall stratum = problem-count weighted mean over L1..L5 (matches markdown overall).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_t_mode.lb_baselines import load_aligned, per_problem_sims  # type: ignore


DEFAULT_N_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
DEFAULT_N_SIMS = 240
DEFAULT_SEED = 42
# All 12 single-T + 2 mix baselines. (T1.0 == T1.0 also in single-T list; we
# keep one canonical name per series.)
BASELINES = [f"T{0.1*(i+1):.1f}" for i in range(12)] + ["random_T", "equal_mix"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist-path", type=Path, required=True)
    ap.add_argument("--combo-name", required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--n-grid", default=",".join(str(n) for n in DEFAULT_N_GRID))
    ap.add_argument("--n-sims", type=int, default=DEFAULT_N_SIMS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    n_grid = sorted(int(x) for x in args.n_grid.split(","))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[{args.combo_name}] loading {args.dist_path}", flush=True)
    problems = list(load_aligned(args.dist_path))
    print(f"[{args.combo_name}] {len(problems)} problems loaded ({time.time()-t0:.1f}s)",
          flush=True)

    # pid_results[pid][N][baseline] = (n_sims,)
    pid_results: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    levels: dict[str, int | None] = {}
    t0 = time.time()
    for i, (pid, P, ci, other_idx, level, _temps) in enumerate(problems):
        sub_rng = np.random.default_rng(args.seed + (i * 31337))
        all_bnames = per_problem_sims(P, ci, other_idx, sub_rng, n_grid, args.n_sims)
        # keep only the 4 baselines we care about
        pid_results[pid] = {
            N: {b: all_bnames[N][b] for b in BASELINES} for N in n_grid
        }
        levels[pid] = level
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(problems)} ({time.time()-t0:.1f}s)", flush=True)
    print(f"[{args.combo_name}] sims done ({time.time()-t0:.1f}s)", flush=True)

    pids = list(pid_results.keys())
    n_blocks = 6
    block_size = args.n_sims // n_blocks
    if block_size * n_blocks != args.n_sims:
        print(f"WARNING: n_sims={args.n_sims} not divisible by {n_blocks}", flush=True)

    rows = []
    for N in n_grid:
        for baseline in BASELINES:
            mat = np.stack([pid_results[p][N][baseline] for p in pids], axis=0)
            blocks = np.zeros((mat.shape[0], n_blocks), dtype=np.float64)
            for b in range(n_blocks):
                blocks[:, b] = mat[:, b*block_size:(b+1)*block_size].mean(axis=1)

            # per-level
            per_L_mean: dict[int, float] = {}
            per_L_n: dict[int, int] = {}
            per_L_std: dict[int, float] = {}
            for L in range(1, 6):
                mask = np.array([levels[p] == L for p in pids])
                if not mask.any():
                    per_L_mean[L] = float("nan"); per_L_std[L] = float("nan"); per_L_n[L] = 0
                    rows.append({"combo": args.combo_name, "baseline": baseline,
                                 "N": N, "stratum": f"L{L}",
                                 "mean": float("nan"), "std6": float("nan"), "n_pids": 0})
                    continue
                L_blocks = blocks[mask].mean(axis=0)
                per_L_mean[L] = float(L_blocks.mean())
                per_L_std[L] = float(L_blocks.std(ddof=1))
                per_L_n[L] = int(mask.sum())
                rows.append({"combo": args.combo_name, "baseline": baseline,
                             "N": N, "stratum": f"L{L}",
                             "mean": per_L_mean[L], "std6": per_L_std[L],
                             "n_pids": per_L_n[L]})

            # overall: problem-count weighted mean over L1..L5 (matches markdown definition)
            valid_L = [L for L in range(1, 6) if per_L_n[L] > 0]
            if not valid_L:
                overall_mean = float("nan"); overall_std = float("nan"); overall_n = 0
            else:
                total_w = sum(per_L_n[L] for L in valid_L)
                overall_mean = sum(per_L_n[L] * per_L_mean[L] for L in valid_L) / total_w
                overall_std = float(np.sqrt(
                    sum(per_L_n[L] * per_L_std[L]**2 for L in valid_L) / total_w))
                overall_n = total_w
            rows.append({"combo": args.combo_name, "baseline": baseline,
                         "N": N, "stratum": "overall",
                         "mean": overall_mean, "std6": overall_std, "n_pids": overall_n})

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"[{args.combo_name}] wrote {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
