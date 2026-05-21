"""Compare 4 equal_mix allocation strategies + T=1.0 baseline on ONE dataset.

Strategies (M = 12 temperatures):
- em_lowfirst   : original lb_baselines, alloc[:rem] += 1 (low-T gets extra)
- em_highfirst  : reverse, alloc[-rem:] += 1 (high-T gets extra)
- em_random     : per-sim random rotation, no systematic bias
- em_floor      : use only N_used = (N // M) * M samples, perfect equal split
                  (wastes N % M samples; N < M yields no samples → NaN)

Aggregation = plurality MV with tie-share (same as lb_baselines).
Baseline = T=1.0 fixed (all N samples at T=1.0, single multinomial draw).

Outputs CSV with columns combo, strategy, N, stratum, mean, std6, n_pids.
Stratum = overall (incl Lr), L1..L5, Lr.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_t_mode.lb_baselines import load_aligned, _share_with_ties  # type: ignore


DEFAULT_DIST = Path("/home3/b.ms/projects/tts_analysis/outputs/"
                    "distributions-math500_n256_Phi-4-mini-instruct/distributions.json")
DEFAULT_OUT = Path("/home3/b.ms/projects/tts_analysis/outputs/"
                   "n_vs_best_t/alloc_compare/math500_Phi.csv")
DEFAULT_N_GRID = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256,
                  384, 512, 768, 1024, 1536, 2048]  # extra resolution
DEFAULT_N_SIMS = 240
DEFAULT_SEED = 42

ALLOCS = ["lowfirst", "highfirst", "random", "floor"]
# Each alloc spawns 2 strategies: em_<alloc> (pool MV) and cv_<alloc> (per-T MV plurality)
STRATEGIES = [f"em_{a}" for a in ALLOCS] + [f"cv_{a}" for a in ALLOCS] + ["T1.0"]


def alloc_lowfirst(N: int, M: int) -> np.ndarray:
    base = N // M; rem = N - base * M
    out = np.full(M, base, dtype=int); out[:rem] += 1
    return out


def alloc_highfirst(N: int, M: int) -> np.ndarray:
    base = N // M; rem = N - base * M
    out = np.full(M, base, dtype=int); out[-rem:] += 1 if rem > 0 else 0
    if rem > 0:
        out[-rem:] += 1; out[-rem:] -= 1  # idempotent guard — already done
    # Above is awkward; simpler:
    out = np.full(M, base, dtype=int)
    if rem > 0:
        out[-rem:] += 1
    return out


def alloc_floor(N: int, M: int) -> np.ndarray:
    base = N // M
    return np.full(M, base, dtype=int)


def alloc_random(N: int, M: int, rng: np.random.Generator) -> np.ndarray:
    base = N // M; rem = N - base * M
    out = np.full(M, base, dtype=int)
    if rem > 0:
        idxs = rng.choice(M, size=rem, replace=False)
        out[idxs] += 1
    return out


def run_alloc_variant(P: np.ndarray, correct_idx: int, other_idx: int,
                       rng: np.random.Generator, alloc_method: str, N: int,
                       n_sims: int) -> tuple[np.ndarray, np.ndarray]:
    """Run one alloc variant, return (em_acc, cv_acc) both shape (n_sims,).

    em_acc = equal_mix (pool counts → MV)
    cv_acc = consensus_vote (per-T MV winner → plurality)
    Both NaN if alloc.sum()==0 (em_floor at N<12).
    """
    M, K = P.shape
    real_slice = slice(0, other_idx)
    pool = np.zeros((n_sims, K), dtype=np.int64)
    consensus_counts = np.zeros((n_sims, K), dtype=np.int32)

    if alloc_method == "random":
        # Per-sim random allocation: each sim has its own alloc array
        allocs = np.zeros((n_sims, M), dtype=np.int32)
        for s in range(n_sims):
            allocs[s] = alloc_random(N, M, rng)
        if allocs.sum() == 0:
            return (np.full(n_sims, np.nan, dtype=np.float32),
                    np.full(n_sims, np.nan, dtype=np.float32))
        for ti in range(M):
            counts_per_sim = np.zeros((n_sims, K), dtype=np.int64)
            winners = np.full(n_sims, -1, dtype=np.int32)
            unique_vals, inverse = np.unique(allocs[:, ti], return_inverse=True)
            for uv_idx, uv in enumerate(unique_vals):
                mask = inverse == uv_idx
                if uv > 0 and mask.any():
                    counts_uv = rng.multinomial(int(uv), P[ti], size=int(mask.sum()))
                    counts_per_sim[mask] = counts_uv
                    real_counts_uv = counts_uv[:, real_slice]
                    if real_counts_uv.shape[1] > 0:
                        winners[mask] = real_counts_uv.argmax(axis=1)
            pool += counts_per_sim
            valid_mask = winners >= 0
            if valid_mask.any():
                np.add.at(consensus_counts,
                          (np.arange(n_sims)[valid_mask], winners[valid_mask]), 1)
    else:
        if alloc_method == "lowfirst":
            alloc = alloc_lowfirst(N, M)
        elif alloc_method == "highfirst":
            alloc = alloc_highfirst(N, M)
        elif alloc_method == "floor":
            alloc = alloc_floor(N, M)
        else:
            raise ValueError(alloc_method)
        if alloc.sum() == 0:
            return (np.full(n_sims, np.nan, dtype=np.float32),
                    np.full(n_sims, np.nan, dtype=np.float32))
        for ti in range(M):
            if alloc[ti] > 0:
                counts = rng.multinomial(int(alloc[ti]), P[ti], size=n_sims)
                pool += counts
                real_counts = counts[:, real_slice]
                if real_counts.shape[1] > 0:
                    winners = real_counts.argmax(axis=1)
                    np.add.at(consensus_counts, (np.arange(n_sims), winners), 1)

    em_acc = _share_with_ties(pool[:, real_slice], correct_idx)
    cv_acc = _share_with_ties(consensus_counts[:, real_slice], correct_idx)
    return em_acc, cv_acc


def run_strategy(P: np.ndarray, correct_idx: int, other_idx: int,
                  rng: np.random.Generator, strategy: str, N: int,
                  n_sims: int) -> np.ndarray:
    """Compatibility wrapper for old API used by main()."""
    M, _ = P.shape
    real_slice = slice(0, other_idx)
    if strategy == "T1.0":
        counts = rng.multinomial(N, P[9], size=n_sims)[:, real_slice]
        return _share_with_ties(counts, correct_idx)
    if strategy.startswith("em_") or strategy.startswith("cv_"):
        prefix, alloc_method = strategy.split("_", 1)
        em, cv = run_alloc_variant(P, correct_idx, other_idx, rng, alloc_method, N, n_sims)
        return em if prefix == "em" else cv
    raise ValueError(strategy)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist-path", type=Path, default=DEFAULT_DIST)
    ap.add_argument("--combo-name", default="math500_Phi-4-mini-instruct")
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-grid", default=",".join(str(n) for n in DEFAULT_N_GRID))
    ap.add_argument("--n-sims", type=int, default=DEFAULT_N_SIMS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    n_grid = sorted(int(x) for x in args.n_grid.split(","))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"loading {args.dist_path}", flush=True)
    problems = list(load_aligned(args.dist_path))
    print(f"  {len(problems)} problems loaded ({time.time()-t0:.1f}s)", flush=True)

    # pid_results[pid][N][strategy] = (n_sims,) acc
    pid_results: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    levels: dict[str, int | None] = {}
    t0 = time.time()
    for i, (pid, P, ci, other_idx, level, _temps) in enumerate(problems):
        pid_results[pid] = {N: {} for N in n_grid}
        levels[pid] = level
        for N in n_grid:
            # T=1.0 baseline (independent seed)
            t1_rng = np.random.default_rng(args.seed + i * 31337 + N * 7919)
            pid_results[pid][N]["T1.0"] = run_strategy(
                P, ci, other_idx, t1_rng, "T1.0", N, args.n_sims)
            # 4 alloc variants × {em, cv} from paired computation
            for alloc_method in ALLOCS:
                am_rng = np.random.default_rng(
                    args.seed + i * 31337 + N * 7919 + hash(alloc_method) % 65536)
                em_acc, cv_acc = run_alloc_variant(
                    P, ci, other_idx, am_rng, alloc_method, N, args.n_sims)
                pid_results[pid][N][f"em_{alloc_method}"] = em_acc
                pid_results[pid][N][f"cv_{alloc_method}"] = cv_acc
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(problems)} ({time.time()-t0:.1f}s)", flush=True)
    print(f"  sims done ({time.time()-t0:.1f}s)", flush=True)

    pids = list(pid_results.keys())
    n_blocks = 6
    block_size = args.n_sims // n_blocks

    # Stratum: L1..L5 + Lr
    stratum_keys = ["L1", "L2", "L3", "L4", "L5", "Lr"]
    pid_to_stratum = {p: (f"L{levels[p]}" if levels[p] in (1,2,3,4,5) else "Lr")
                       for p in pids}

    rows = []
    for N in n_grid:
        for strategy in STRATEGIES:
            mat = np.stack([pid_results[p][N][strategy] for p in pids], axis=0)
            blocks = np.zeros((mat.shape[0], n_blocks), dtype=np.float64)
            for b in range(n_blocks):
                blocks[:, b] = mat[:, b*block_size:(b+1)*block_size].mean(axis=1)

            per_S_mean, per_S_n, per_S_std = {}, {}, {}
            for S in stratum_keys:
                mask = np.array([pid_to_stratum[p] == S for p in pids])
                if not mask.any():
                    per_S_mean[S] = float("nan"); per_S_std[S] = float("nan"); per_S_n[S] = 0
                    rows.append({"combo": args.combo_name, "strategy": strategy,
                                 "N": N, "stratum": S,
                                 "mean": float("nan"), "std6": float("nan"), "n_pids": 0})
                    continue
                sb = blocks[mask].mean(axis=0)
                per_S_mean[S] = float(np.nanmean(sb))
                per_S_std[S] = float(np.nanstd(sb, ddof=1))
                per_S_n[S] = int(mask.sum())
                rows.append({"combo": args.combo_name, "strategy": strategy,
                             "N": N, "stratum": S,
                             "mean": per_S_mean[S], "std6": per_S_std[S],
                             "n_pids": per_S_n[S]})

            valid_S = [S for S in stratum_keys if per_S_n[S] > 0
                       and not np.isnan(per_S_mean[S])]
            if valid_S:
                tw = sum(per_S_n[S] for S in valid_S)
                om = sum(per_S_n[S] * per_S_mean[S] for S in valid_S) / tw
                os = float(np.sqrt(sum(per_S_n[S] * per_S_std[S]**2 for S in valid_S) / tw))
                rows.append({"combo": args.combo_name, "strategy": strategy,
                             "N": N, "stratum": "overall",
                             "mean": om, "std6": os, "n_pids": tw})

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
