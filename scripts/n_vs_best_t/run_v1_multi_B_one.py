"""Run lb_algorithm_v1 at MULTIPLE B values, aggregate per-stratum (L1..L5 + Lr).

Outputs long CSV: combo, B, stratum, mean, std6, n_pids.
v1.0 needs B ≥ ~96 (pilot fixed at 96 for mid path). Default grid: 128..2048.
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

DEFAULT_B_GRID = [128, 256, 512, 1024, 1536, 2048]
DEFAULT_N_SIMS = 240
DEFAULT_SEED = 12345


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist-path", type=Path, required=True)
    ap.add_argument("--combo-name", required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--b-grid", default=",".join(str(b) for b in DEFAULT_B_GRID))
    ap.add_argument("--n-sims", type=int, default=DEFAULT_N_SIMS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    b_grid = sorted(int(x) for x in args.b_grid.split(","))

    t0 = time.time()
    print(f"[{args.combo_name}] loading {args.dist_path}", flush=True)
    problems = list(load_problem_distributions(args.dist_path))
    print(f"[{args.combo_name}] {len(problems)} problems loaded ({time.time()-t0:.1f}s)",
          flush=True)

    # per_pid[pid][B] = (n_sims,) acc, level
    pid_accs: dict[str, dict[int, np.ndarray]] = {}
    levels: dict[str, int | None] = {}
    t0 = time.time()
    for i, (pid, P, ci, other_idx, level) in enumerate(problems):
        n_real = other_idx
        pid_accs[pid] = {}
        for B in b_grid:
            sub_rng = np.random.default_rng(args.seed + i * 31337 + B)
            accs = np.zeros(args.n_sims, dtype=np.float32)
            for s in range(args.n_sims):
                acc, _path = algorithm_v1(P, ci, n_real, B, sub_rng)
                accs[s] = acc
            pid_accs[pid][B] = accs
        levels[pid] = level
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(problems)} ({time.time()-t0:.1f}s)", flush=True)
    print(f"[{args.combo_name}] v1 multi-B done ({time.time()-t0:.1f}s)", flush=True)

    pids = list(pid_accs.keys())
    n_blocks = 6
    block_size = args.n_sims // n_blocks

    # Stratum map: L1..L5 + Lr
    stratum_keys = ["L1", "L2", "L3", "L4", "L5", "Lr"]
    pid_to_stratum: dict[str, str] = {}
    for p in pids:
        lvl = levels[p]
        pid_to_stratum[p] = f"L{lvl}" if lvl in (1, 2, 3, 4, 5) else "Lr"

    rows = []
    for B in b_grid:
        mat = np.stack([pid_accs[p][B] for p in pids], axis=0)
        blocks = np.zeros((mat.shape[0], n_blocks), dtype=np.float64)
        for b in range(n_blocks):
            blocks[:, b] = mat[:, b*block_size:(b+1)*block_size].mean(axis=1)

        per_S_mean, per_S_n, per_S_std = {}, {}, {}
        for S in stratum_keys:
            mask = np.array([pid_to_stratum[p] == S for p in pids])
            if not mask.any():
                per_S_mean[S] = float("nan"); per_S_std[S] = float("nan"); per_S_n[S] = 0
                rows.append({"combo": args.combo_name, "B": B, "stratum": S,
                             "mean": float("nan"), "std6": float("nan"), "n_pids": 0})
                continue
            sb = blocks[mask].mean(axis=0)
            per_S_mean[S] = float(sb.mean())
            per_S_std[S] = float(sb.std(ddof=1))
            per_S_n[S] = int(mask.sum())
            rows.append({"combo": args.combo_name, "B": B, "stratum": S,
                         "mean": per_S_mean[S], "std6": per_S_std[S], "n_pids": per_S_n[S]})

        valid_S = [S for S in stratum_keys if per_S_n[S] > 0]
        if valid_S:
            tw = sum(per_S_n[S] for S in valid_S)
            om = sum(per_S_n[S] * per_S_mean[S] for S in valid_S) / tw
            os = float(np.sqrt(sum(per_S_n[S] * per_S_std[S]**2 for S in valid_S) / tw))
            rows.append({"combo": args.combo_name, "B": B, "stratum": "overall",
                         "mean": om, "std6": os, "n_pids": tw})

        valid_L = [S for S in ["L1","L2","L3","L4","L5"] if per_S_n[S] > 0]
        if valid_L:
            tw = sum(per_S_n[S] for S in valid_L)
            om = sum(per_S_n[S] * per_S_mean[S] for S in valid_L) / tw
            os = float(np.sqrt(sum(per_S_n[S] * per_S_std[S]**2 for S in valid_L) / tw))
            rows.append({"combo": args.combo_name, "B": B, "stratum": "overall_md_compat",
                         "mean": om, "std6": os, "n_pids": tw})

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"[{args.combo_name}] wrote {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
