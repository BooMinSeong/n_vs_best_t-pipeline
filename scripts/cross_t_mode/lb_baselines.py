"""LB-Step1: Baseline scaling curves at finite N (the bar we must dominate).

For math1k_n256_Qwen2.5-3B (12T × 256 × 6 seeds, pooled empirical distribution):

  Four oracle-free baselines:
    random_T      per-query 1 T uniform, all N samples from that T, MV
    T=1.0         all N samples at T=1.0, MV
    equal_mix     N split evenly across 12 T (round-robin), MV
    best_fixed_T  best single T (T=0.5 from step3h: 75.40%), MV

  Also report each single-T curve for context.

  Grid: N ∈ {16, 32, 64, 128, 256}.
  Replicates: 200 MC sims per (problem, N) for stable estimates.
  Aggregation: per-problem accuracy → mean across problems (overall, per-L1..L5).

  Three difficulty mixtures (re-weighting L1..L5 averages):
    uniform:    (0.2, 0.2, 0.2, 0.2, 0.2)
    easy-heavy: (0.40, 0.30, 0.15, 0.10, 0.05)
    hard-heavy: (0.05, 0.10, 0.15, 0.30, 0.40)

Output: CSV with one row per (baseline, N, mixture) + per-L columns.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


DIST_PATH = Path("/home3/b.ms/projects/tts_analysis/outputs/"
                 "distributions-math1k_n256_Qwen2.5-3B/distributions_v2.json")
OUT_DIR = Path("/home3/b.ms/tmp/cross_t_lb/step1")
N_GRID = [16, 32, 64, 128, 256]
N_SIMS = 200
RNG_SEED = 42

MIXTURES = {
    "uniform":    (0.20, 0.20, 0.20, 0.20, 0.20),
    "easy-heavy": (0.40, 0.30, 0.15, 0.10, 0.05),
    "hard-heavy": (0.05, 0.10, 0.15, 0.30, 0.40),
}


def load_aligned(path: Path):
    """Yield (pid, P, correct_idx, level) where P is (n_T, K_kept+other)."""
    with path.open() as f:
        data = json.load(f)
    temps = sorted(float(t) for t in data["metadata"]["temperatures"])
    for pid, prob in data["distributions"].items():
        by_t = prob["by_temp"]
        # union of answers
        ans_to_idx = {}
        for t in temps:
            for ans in by_t[str(t)]["answer_probs"].keys():
                if ans not in ans_to_idx:
                    ans_to_idx[ans] = len(ans_to_idx)
        if not ans_to_idx:
            continue
        n_a = len(ans_to_idx)
        n_T = len(temps)
        P_full = np.zeros((n_T, n_a), dtype=np.float64)
        for ti, t in enumerate(temps):
            for ans, p in by_t[str(t)]["answer_probs"].items():
                P_full[ti, ans_to_idx[ans]] = p
        correct = by_t[str(temps[0])]["correct_canonical"]
        if correct is None or correct not in ans_to_idx:
            continue
        correct_idx = ans_to_idx[correct]
        # build "other" bucket: cap top-K answers + correct, residual to "other"
        max_K = 60
        max_across_T = P_full.max(axis=0)
        top_idx = np.argsort(-max_across_T)[:max_K].tolist()
        keep = sorted(set(top_idx + [correct_idx]))
        idx_remap = {old: new for new, old in enumerate(keep)}
        P_kept = P_full[:, keep]
        other = (1.0 - P_kept.sum(axis=1, keepdims=True)).clip(min=0.0)
        # renormalize to make rows sum to 1 (for multinomial sampling)
        P = np.concatenate([P_kept, other], axis=1)
        P = P / P.sum(axis=1, keepdims=True).clip(min=1e-12)
        new_correct = idx_remap[correct_idx]
        other_idx = P.shape[1] - 1
        level = prob.get("difficulty", {}).get("level", None)
        yield pid, P, new_correct, other_idx, level, temps


def _share_with_ties(counts: np.ndarray, correct_idx: int) -> np.ndarray:
    """Per-sim plurality MV accuracy, splitting ties at correct uniformly.

    counts: (..., K_real) integer counts. correct_idx: int.
    Returns: (...,) float in [0, 1].
    """
    max_v = counts.max(axis=-1, keepdims=True)
    correct_at_max = (counts[..., correct_idx:correct_idx + 1] == max_v).squeeze(-1)
    has_any = max_v.squeeze(-1) > 0
    num_tied = (counts == max_v).sum(axis=-1)
    return np.where(correct_at_max & has_any,
                    1.0 / np.maximum(num_tied, 1), 0.0)


def equal_alloc(N: int, M: int) -> np.ndarray:
    """N samples split as evenly as possible across M T (round-robin)."""
    base = N // M
    rem = N - base * M
    out = np.full(M, base, dtype=int)
    out[:rem] += 1
    return out


def per_problem_sims(P, correct_idx, other_idx, rng, N_grid, n_sims):
    """Return dict[N] → dict[baseline_name] of (n_sims,) acc array."""
    M, K = P.shape
    real_slice = slice(0, other_idx)
    result = {N: {} for N in N_grid}

    for N in N_grid:
        # Per-T draws: (M, n_sims, K) counts
        per_T_acc = np.zeros((M, n_sims), dtype=np.float32)
        for ti in range(M):
            counts = rng.multinomial(N, P[ti], size=n_sims)[:, real_slice]
            per_T_acc[ti] = _share_with_ties(counts, correct_idx)

        # random_T: pick 1 T uniformly per sim, use that T's acc.
        # Standard interpretation: expected acc = mean over T of acc(T, N).
        # We use the actual per-sim variance by sampling T per sim.
        t_choice = rng.integers(0, M, size=n_sims)
        random_T_acc = per_T_acc[t_choice, np.arange(n_sims)]
        result[N]["random_T"] = random_T_acc

        # T=1.0: T index 9 (0.1, 0.2, ..., 1.0, 1.1, 1.2)
        result[N]["T1.0"] = per_T_acc[9]

        # Save every single-T for downstream best_fixed_T pick
        for ti in range(M):
            result[N][f"T{0.1 * (ti + 1):.1f}"] = per_T_acc[ti]

        # equal_mix: pool counts across T via round-robin alloc
        alloc = equal_alloc(N, M)
        pool = np.zeros((n_sims, K), dtype=np.int64)
        for ti in range(M):
            if alloc[ti] > 0:
                pool += rng.multinomial(alloc[ti], P[ti], size=n_sims)
        result[N]["equal_mix"] = _share_with_ties(pool[:, real_slice], correct_idx)

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist-path", type=Path, default=DIST_PATH)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--n-grid", default=",".join(str(n) for n in N_GRID))
    ap.add_argument("--n-sims", type=int, default=N_SIMS)
    ap.add_argument("--rng-seed", type=int, default=RNG_SEED)
    ap.add_argument("--max-pids", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_sims = args.n_sims
    n_grid = sorted(int(x) for x in args.n_grid.split(","))

    t0 = time.time()
    print(f"[lb_baselines] loading {args.dist_path}")
    problems = list(load_aligned(args.dist_path))
    if args.max_pids > 0:
        problems = problems[:args.max_pids]
    print(f"[lb_baselines] {len(problems)} problems loaded ({time.time()-t0:.1f}s)")

    # Collect: per_pid_results[pid] = {N: {bname: (n_sims,) acc}}
    pid_results = {}
    levels = {}
    rng = np.random.default_rng(args.rng_seed)
    t0 = time.time()
    for i, (pid, P, ci, other_idx, level, temps) in enumerate(problems):
        sub_rng = np.random.default_rng(args.rng_seed + (i * 31337))
        pid_results[pid] = per_problem_sims(P, ci, other_idx, sub_rng, n_grid, n_sims)
        levels[pid] = level
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(problems)} ({time.time()-t0:.1f}s)")
    print(f"[lb_baselines] simulations done ({time.time()-t0:.1f}s)")

    # Aggregate: per-baseline, per-N, per-level mean and seed-block std (6 blocks)
    pids = list(pid_results.keys())
    bname_keys = list(pid_results[pids[0]][n_grid[0]].keys())
    n_blocks = 6
    block_size = n_sims // n_blocks  # e.g., 200/6 ≈ 33 sims per block

    summary_rows = []
    for bname in bname_keys:
        for N in n_grid:
            # Stack: (n_pids, n_sims)
            mat = np.stack([pid_results[p][N][bname] for p in pids], axis=0)
            # block-level means: (n_pids, n_blocks)
            blocks = np.zeros((mat.shape[0], n_blocks), dtype=np.float64)
            for b in range(n_blocks):
                blocks[:, b] = mat[:, b * block_size:(b + 1) * block_size].mean(axis=1)
            # per-block overall acc (mean across pids), std across blocks
            per_block_overall = blocks.mean(axis=0)  # (n_blocks,)
            overall_mean = float(per_block_overall.mean())
            overall_std = float(per_block_overall.std(ddof=1))

            row = {"baseline": bname, "N": N,
                   "overall_mean": overall_mean,
                   "overall_std6": overall_std,
                   "n_pids": int(mat.shape[0])}

            # per-level (1..5) breakdown
            per_L_mean = {}
            for L in range(1, 6):
                mask = np.array([levels[p] == L for p in pids])
                if not mask.any():
                    row[f"L{L}_mean"] = float("nan")
                    row[f"L{L}_std6"] = float("nan")
                    row[f"L{L}_n"] = 0
                    per_L_mean[L] = float("nan")
                    continue
                L_blocks = blocks[mask].mean(axis=0)  # (n_blocks,)
                row[f"L{L}_mean"] = float(L_blocks.mean())
                row[f"L{L}_std6"] = float(L_blocks.std(ddof=1))
                row[f"L{L}_n"] = int(mask.sum())
                per_L_mean[L] = row[f"L{L}_mean"]

            # mixtures (re-weight per-L means)
            for mix_name, weights in MIXTURES.items():
                # only count levels with valid data
                valid = [(per_L_mean[L], w) for L, w in zip(range(1, 6), weights)
                         if not np.isnan(per_L_mean[L])]
                if not valid:
                    row[f"mix_{mix_name}"] = float("nan")
                    continue
                w_sum = sum(w for _, w in valid)
                row[f"mix_{mix_name}"] = sum(m * w for m, w in valid) / w_sum

            summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    out_csv = args.out_dir / "lb_baselines_math1k.csv"
    df.to_csv(out_csv, index=False)
    print(f"[lb_baselines] wrote {out_csv}")

    # markdown summary: focus on the four headline baselines
    headline = ["random_T", "T1.0", "equal_mix", "T0.5"]  # T0.5 = best_fixed_T
    md = ["# LB-Step1: Baseline scaling curves (math1k, Qwen2.5-3B)", "",
          f"- pids = {len(pids)} | n_sims = {n_sims} | n_blocks (proxy seed) = {n_blocks}",
          "- four headline baselines: random_T, T=1.0, equal_mix, best_fixed_T (T=0.5)",
          "- Per-L breakdown and 3 mixtures (uniform / easy-heavy / hard-heavy) at each N", ""]

    md.append("## Overall accuracy (mean ± seed-block std)")
    md.append("| baseline | " + " | ".join(f"N={n}" for n in n_grid) + " |")
    md.append("|---" + "|---" * len(n_grid) + "|")
    for bname in headline:
        cells = []
        for n in n_grid:
            r = df[(df.baseline == bname) & (df.N == n)].iloc[0]
            cells.append(f"{r.overall_mean*100:.2f}±{r.overall_std6*100:.2f}")
        md.append(f"| {bname} | " + " | ".join(cells) + " |")
    md.append("")

    for mix_name in MIXTURES:
        md.append(f"## Mixture: {mix_name} weights = {MIXTURES[mix_name]}")
        md.append("| baseline | " + " | ".join(f"N={n}" for n in n_grid) + " |")
        md.append("|---" + "|---" * len(n_grid) + "|")
        for bname in headline:
            cells = []
            for n in n_grid:
                r = df[(df.baseline == bname) & (df.N == n)].iloc[0]
                v = r[f"mix_{mix_name}"]
                cells.append(f"{v*100:.2f}")
            md.append(f"| {bname} | " + " | ".join(cells) + " |")
        md.append("")

    md.append("## Per-level breakdown @ N=256 (the deploy ceiling)")
    md.append("| baseline | L1 | L2 | L3 | L4 | L5 |")
    md.append("|---|---|---|---|---|---|")
    for bname in headline:
        r = df[(df.baseline == bname) & (df.N == 256)].iloc[0]
        cells = [f"{r[f'L{L}_mean']*100:.2f}" for L in range(1, 6)]
        md.append(f"| {bname} | " + " | ".join(cells) + " |")
    md.append("")

    md.append("## All single-T fixed @ N=256 (for context)")
    md.append("| T | acc |")
    md.append("|---|---|")
    for ti in range(12):
        t = 0.1 * (ti + 1)
        bname = f"T{t:.1f}"
        r = df[(df.baseline == bname) & (df.N == 256)].iloc[0]
        md.append(f"| {bname} | {r.overall_mean*100:.2f}±{r.overall_std6*100:.2f} |")
    md.append("")

    (args.out_dir / "lb_baselines_math1k.md").write_text("\n".join(md))
    print(f"[lb_baselines] wrote {args.out_dir / 'lb_baselines_math1k.md'}")


if __name__ == "__main__":
    main()
