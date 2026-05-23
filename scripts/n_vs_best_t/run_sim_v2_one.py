"""Sim baselines v2 — adds consensus_vote, extends N to 2048.

Per (problem, N):
- 12 single-T MV (T0.1..T1.2)
- random_T: pick 1 T uniformly per sim, use that T's MV result
- equal_mix: round-robin N/12 alloc → pool counts → plurality MV
- consensus_vote (NEW): round-robin N/12 alloc → per-T MV winner → plurality of 12 winners

Output long-format CSV: combo, baseline, N, stratum, mean, std6, n_pids.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_t_mode.lb_baselines import load_aligned, equal_alloc, _share_with_ties  # type: ignore


DEFAULT_N_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048]
DEFAULT_N_SIMS = 240
DEFAULT_SEED = 42
SINGLE_T_NAMES = [f"T{0.1*(i+1):.1f}" for i in range(12)]
MIX_BASELINES = ["random_T", "equal_mix", "consensus_vote"]


def per_problem_sims_v2(P: np.ndarray, correct_idx: int, other_idx: int,
                         rng: np.random.Generator, N_grid: list[int],
                         n_sims: int) -> dict[int, dict[str, np.ndarray]]:
    """Return dict[N] → dict[baseline_name] of (n_sims,) acc array."""
    M, K = P.shape
    real_slice = slice(0, other_idx)
    result: dict[int, dict[str, np.ndarray]] = {N: {} for N in N_grid}

    for N in N_grid:
        # ============ Single-T sampling ============
        per_T_acc = np.zeros((M, n_sims), dtype=np.float32)
        per_T_winner_idx = np.zeros((M, n_sims), dtype=np.int32)  # for consensus_vote
        for ti in range(M):
            counts = rng.multinomial(N, P[ti], size=n_sims)[:, real_slice]
            per_T_acc[ti] = _share_with_ties(counts, correct_idx)
            # MV winner per (T, sim): argmax with ties → just argmax (first)
            per_T_winner_idx[ti] = counts.argmax(axis=1)

        # Single-T baselines
        for ti, name in enumerate(SINGLE_T_NAMES):
            result[N][name] = per_T_acc[ti]

        # random_T: pick 1 T uniformly per sim, use that T's acc
        t_choice = rng.integers(0, M, size=n_sims)
        result[N]["random_T"] = per_T_acc[t_choice, np.arange(n_sims)]

        # ============ Equal-mix allocation: per-T draws with N/12 per T ============
        alloc = equal_alloc(N, M)
        pool = np.zeros((n_sims, K), dtype=np.int64)
        per_T_em_winner = np.full((M, n_sims), -1, dtype=np.int32)  # -1 = no real-answer winner
        for ti in range(M):
            if alloc[ti] > 0:
                em_counts = rng.multinomial(alloc[ti], P[ti], size=n_sims)
                pool += em_counts
                # winner over real answers only; mark -1 when all samples fell into "other"
                em_real = em_counts[:, real_slice]
                em_winner = em_real.argmax(axis=1)
                em_winner[em_real.max(axis=1) == 0] = -1
                per_T_em_winner[ti] = em_winner

        # equal_mix: pool counts → plurality MV
        result[N]["equal_mix"] = _share_with_ties(pool[:, real_slice], correct_idx)

        # consensus_vote: per-T winner → plurality vote among valid (non-(-1)) winners.
        # other-only sims for a T contribute 0 votes for that T (matches equal_mix's
        # behaviour of ignoring "other" counts when picking the pool winner).
        consensus_counts = np.zeros((n_sims, K), dtype=np.int32)
        for ti in range(M):
            if alloc[ti] > 0:
                winners = per_T_em_winner[ti]
                valid_sims = np.where(winners != -1)[0]
                if valid_sims.size:
                    np.add.at(consensus_counts,
                              (valid_sims, winners[valid_sims]), 1)
        cv_acc = _share_with_ties(consensus_counts[:, real_slice], correct_idx)
        result[N]["consensus_vote"] = cv_acc

    return result


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

    all_baselines = SINGLE_T_NAMES + MIX_BASELINES
    pid_results: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    levels: dict[str, int | None] = {}
    t0 = time.time()
    for i, (pid, P, ci, other_idx, level, _temps) in enumerate(problems):
        sub_rng = np.random.default_rng(args.seed + (i * 31337))
        all_bnames = per_problem_sims_v2(P, ci, other_idx, sub_rng, n_grid, args.n_sims)
        pid_results[pid] = all_bnames
        levels[pid] = level
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(problems)} ({time.time()-t0:.1f}s)", flush=True)
    print(f"[{args.combo_name}] sims done ({time.time()-t0:.1f}s)", flush=True)

    pids = list(pid_results.keys())
    n_blocks = 6
    block_size = args.n_sims // n_blocks

    # Build stratum map: L1..L5 from level field; Lr (recoverable) = level==None
    # (problems with no canonical at T=0.1 were already dropped by load_aligned).
    stratum_keys = ["L1", "L2", "L3", "L4", "L5", "Lr"]
    pid_to_stratum: dict[str, str] = {}
    for p in pids:
        lvl = levels[p]
        if lvl in (1, 2, 3, 4, 5):
            pid_to_stratum[p] = f"L{lvl}"
        else:  # None or other unexpected → Lr (recoverable, since canonical exists)
            pid_to_stratum[p] = "Lr"

    rows = []
    for N in n_grid:
        for baseline in all_baselines:
            mat = np.stack([pid_results[p][N][baseline] for p in pids], axis=0)
            blocks = np.zeros((mat.shape[0], n_blocks), dtype=np.float64)
            for b in range(n_blocks):
                blocks[:, b] = mat[:, b*block_size:(b+1)*block_size].mean(axis=1)

            per_S_mean: dict[str, float] = {}
            per_S_n: dict[str, int] = {}
            per_S_std: dict[str, float] = {}
            for S in stratum_keys:
                mask = np.array([pid_to_stratum[p] == S for p in pids])
                if not mask.any():
                    per_S_mean[S] = float("nan"); per_S_std[S] = float("nan"); per_S_n[S] = 0
                    rows.append({"combo": args.combo_name, "baseline": baseline,
                                 "N": N, "stratum": S,
                                 "mean": float("nan"), "std6": float("nan"), "n_pids": 0})
                    continue
                S_blocks = blocks[mask].mean(axis=0)
                per_S_mean[S] = float(S_blocks.mean())
                per_S_std[S] = float(S_blocks.std(ddof=1))
                per_S_n[S] = int(mask.sum())
                rows.append({"combo": args.combo_name, "baseline": baseline,
                             "N": N, "stratum": S,
                             "mean": per_S_mean[S], "std6": per_S_std[S],
                             "n_pids": per_S_n[S]})

            # overall: weighted over L1..L5 + Lr (includes recoverable)
            valid_S = [S for S in stratum_keys if per_S_n[S] > 0]
            if not valid_S:
                overall_mean = float("nan"); overall_std = float("nan"); overall_n = 0
            else:
                total_w = sum(per_S_n[S] for S in valid_S)
                overall_mean = sum(per_S_n[S] * per_S_mean[S] for S in valid_S) / total_w
                overall_std = float(np.sqrt(
                    sum(per_S_n[S] * per_S_std[S]**2 for S in valid_S) / total_w))
                overall_n = total_w
            rows.append({"combo": args.combo_name, "baseline": baseline,
                         "N": N, "stratum": "overall",
                         "mean": overall_mean, "std6": overall_std, "n_pids": overall_n})

            # overall_md_compat: weighted over L1..L5 only (matches markdown convention)
            valid_L = [S for S in ["L1","L2","L3","L4","L5"] if per_S_n[S] > 0]
            if not valid_L:
                md_mean = float("nan"); md_std = float("nan"); md_n = 0
            else:
                tw = sum(per_S_n[S] for S in valid_L)
                md_mean = sum(per_S_n[S] * per_S_mean[S] for S in valid_L) / tw
                md_std = float(np.sqrt(sum(per_S_n[S] * per_S_std[S]**2 for S in valid_L) / tw))
                md_n = tw
            rows.append({"combo": args.combo_name, "baseline": baseline,
                         "N": N, "stratum": "overall_md_compat",
                         "mean": md_mean, "std6": md_std, "n_pids": md_n})

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"[{args.combo_name}] wrote {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
