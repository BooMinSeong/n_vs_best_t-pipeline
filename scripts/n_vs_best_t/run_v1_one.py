"""Run lb_algorithm_v1.py at B=256 on ONE combo, aggregate to per-stratum.

Outputs CSV: combo, N, stratum, mean, std6, n_pids (matches sim_baselines schema).
Only N=256 is filled.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_t_mode.lb_algorithm_v1 import (  # type: ignore
    algorithm_v1, load_problem_distributions,
)

DEFAULT_B = 256
DEFAULT_N_SIMS = 240
DEFAULT_SEED = 12345


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist-path", type=Path, required=True)
    ap.add_argument("--combo-name", required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--B", type=int, default=DEFAULT_B)
    ap.add_argument("--n-sims", type=int, default=DEFAULT_N_SIMS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[{args.combo_name}] loading {args.dist_path}", flush=True)
    problems = list(load_problem_distributions(args.dist_path))
    print(f"[{args.combo_name}] {len(problems)} problems loaded ({time.time()-t0:.1f}s)",
          flush=True)

    # per-PID: (n_sims,) acc + level + path counts
    per_pid_blocks: dict[str, np.ndarray] = {}
    levels: dict[str, int | None] = {}
    path_counter: dict[str, int] = {}
    t0 = time.time()
    for i, (pid, P, ci, other_idx, level) in enumerate(problems):
        sub_rng = np.random.default_rng(args.seed + i * 31337)
        n_real = other_idx
        accs = np.zeros(args.n_sims, dtype=np.float32)
        for s in range(args.n_sims):
            acc, path = algorithm_v1(P, ci, n_real, args.B, sub_rng)
            accs[s] = acc
            path_counter[path] = path_counter.get(path, 0) + 1
        per_pid_blocks[pid] = accs
        levels[pid] = level
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(problems)} ({time.time()-t0:.1f}s)", flush=True)
    print(f"[{args.combo_name}] v1 done ({time.time()-t0:.1f}s)", flush=True)

    pids = list(per_pid_blocks.keys())
    n_blocks = 6
    block_size = args.n_sims // n_blocks
    if block_size * n_blocks != args.n_sims:
        print(f"WARNING: n_sims={args.n_sims} not divisible by {n_blocks}", flush=True)

    # (n_pid, n_blocks)
    mat = np.stack([per_pid_blocks[p] for p in pids], axis=0)
    blocks = np.zeros((mat.shape[0], n_blocks), dtype=np.float64)
    for b in range(n_blocks):
        blocks[:, b] = mat[:, b*block_size:(b+1)*block_size].mean(axis=1)

    rows = []
    per_L_mean: dict[int, float] = {}
    per_L_n: dict[int, int] = {}
    per_L_std: dict[int, float] = {}
    for L in range(1, 6):
        mask = np.array([levels[p] == L for p in pids])
        if not mask.any():
            per_L_mean[L] = float("nan"); per_L_std[L] = float("nan"); per_L_n[L] = 0
            rows.append({"combo": args.combo_name, "N": args.B, "stratum": f"L{L}",
                         "mean": float("nan"), "std6": float("nan"), "n_pids": 0})
            continue
        L_blocks = blocks[mask].mean(axis=0)
        per_L_mean[L] = float(L_blocks.mean())
        per_L_std[L] = float(L_blocks.std(ddof=1))
        per_L_n[L] = int(mask.sum())
        rows.append({"combo": args.combo_name, "N": args.B, "stratum": f"L{L}",
                     "mean": per_L_mean[L], "std6": per_L_std[L], "n_pids": per_L_n[L]})

    valid_L = [L for L in range(1, 6) if per_L_n[L] > 0]
    if not valid_L:
        overall_mean = float("nan"); overall_std = float("nan"); overall_n = 0
    else:
        total_w = sum(per_L_n[L] for L in valid_L)
        overall_mean = sum(per_L_n[L] * per_L_mean[L] for L in valid_L) / total_w
        overall_std = float(np.sqrt(
            sum(per_L_n[L] * per_L_std[L]**2 for L in valid_L) / total_w))
        overall_n = total_w
    rows.append({"combo": args.combo_name, "N": args.B, "stratum": "overall",
                 "mean": overall_mean, "std6": overall_std, "n_pids": overall_n})

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"[{args.combo_name}] wrote {args.out_csv}", flush=True)
    print(f"[{args.combo_name}] path distribution:", flush=True)
    total = sum(path_counter.values())
    for k, v in sorted(path_counter.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k}: {v/total*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
