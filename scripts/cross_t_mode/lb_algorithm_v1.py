"""LB Algorithm v1.0 — Cross-T Lower-Bound Routing.

Trajectory-feature staged routing algorithm. Deployable (no oracle, no
difficulty classifier). Settles the v0.5 → v0.10a_AND development trajectory.

Flow:
  Stage 0a — light pilot, 4 samples/T  (48 total)
  Stage 0a-gate (hardness):
     pilot_entropy_min  < TAU_EASY=0.5    → easy: skip Stage 0b
     pilot_entropy_min  > TAU_HARD=1.0    → hard: equal_mix fallback (return)
     else                                  → mid: continue to Stage 0b
  Stage 0b — extended pilot, +4/T → 96 total

  Mode-pair extraction (top-2 widest top-1 regions across 12 T)

  Gate A — no_top1 detection:
     high_sharp_at_T≥1.0 < 0.26  OR  entropy[T=0.1] > 2.04
        → equal_mix fallback (catastrophic-loss protection)

  Gate B — mode routing:
     low_mode if:
        sbT_idx > 8
        OR low_width > 5.5
        OR (high_width < 4  AND  high_T_mean > 0.95)   [v1.0 strict cut]
     else high_mode

  Stage 4 — main allocation: remaining budget → chosen mode region T's
            (round-robin even split)

  Stage 5 — final MV: plurality vote on pilot + main pool

Hyperparameters tuned on math1k_Qwen2.5-3B. Generalises cleanly to:
  mathfull_Qwen2.5-3B, gsm8k_Qwen2.5-3B, aime2025/26_Qwen2.5-3B,
  mathfull_Llama-3.2-3B.
Weaker generalisation on Qwen3-4B-Instruct-2507 (likely needs model-aware
threshold re-tune).

See V1_REPORT.md for full provenance.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# v1.0 hyperparameters (FROZEN)
# ============================================================================
B = 256                                   # total per-query budget
N_PA = 4                                  # Stage 0a per-T pilot size
N_PB = 4                                  # Stage 0b additional per-T size
TAU_EASY = 0.5                            # easy/mid hardness threshold
TAU_HARD = 1.0                            # mid/hard hardness threshold
GA_HIGH_SHARP = 0.26                      # Gate A: high-T top1 freq cutoff
GA_LOW_ENT = 2.04                         # Gate A: entropy[T=0.1] cutoff
GB_SBT_IDX = 8                            # Gate B: swap-boundary index cutoff
GB_LOW_WIDTH = 5                          # Gate B: low_mode width cutoff
GB_HIGH_WIDTH_MAX = 4                     # Gate B: AND-cut high_width upper
GB_HIGH_T_MEAN_MIN = 0.95                 # Gate B: AND-cut high_T_mean lower

N_SIMS_DEFAULT = 600
RNG_SEED_DEFAULT = 12345


# ============================================================================
# Helper functions
# ============================================================================

def equal_alloc(N: int, M: int) -> np.ndarray:
    """Round-robin allocation of N items into M bins."""
    base = N // M; rem = N - base * M
    out = np.full(M, base, dtype=int)
    out[:rem] += 1
    return out


def share_with_ties(counts: np.ndarray, correct_idx: int) -> float:
    """Plurality MV accuracy with uniform tie-share for correct answer."""
    max_v = counts.max()
    if max_v == 0:
        return 0.0
    tied = (counts == max_v)
    if tied[correct_idx]:
        return 1.0 / int(tied.sum())
    return 0.0


def load_problem_distributions(path: Path):
    """Load aligned probability matrices from a distributions JSON file.

    Supports both schemas (v2 with 'difficulty' block, and legacy).
    Yields (pid, P, correct_idx, other_idx, level).
    """
    with path.open() as f:
        data = json.load(f)
    items = data["distributions"] if "distributions" in data else data
    temps = sorted(float(t) for t in data["metadata"]["temperatures"]) \
        if "metadata" in data else None

    for pid, prob in items.items():
        if isinstance(prob, dict) and "by_temp" in prob:
            by_t = prob["by_temp"]
            level = prob.get("difficulty", {}).get("level")
        else:
            by_t = prob
            level = None
        if temps is None:
            temps = sorted(float(t) for t in by_t.keys())

        ans_to_idx = {}
        for t in temps:
            for ans in by_t[str(t)].get("answer_probs", {}):
                if ans not in ans_to_idx:
                    ans_to_idx[ans] = len(ans_to_idx)
        if not ans_to_idx:
            continue
        n_a, n_T = len(ans_to_idx), len(temps)
        P_full = np.zeros((n_T, n_a))
        for ti, t in enumerate(temps):
            for ans, p in by_t[str(t)]["answer_probs"].items():
                P_full[ti, ans_to_idx[ans]] = p
        correct = by_t[str(temps[0])]["correct_canonical"]
        if correct is None or correct not in ans_to_idx:
            continue
        correct_idx = ans_to_idx[correct]
        max_K = 60
        max_across_T = P_full.max(axis=0)
        top_idx = np.argsort(-max_across_T)[:max_K].tolist()
        keep = sorted(set(top_idx + [correct_idx]))
        idx_remap = {old: new for new, old in enumerate(keep)}
        P_kept = P_full[:, keep]
        other = (1.0 - P_kept.sum(axis=1, keepdims=True)).clip(min=0.0)
        P = np.concatenate([P_kept, other], axis=1)
        P = P / P.sum(axis=1, keepdims=True).clip(min=1e-12)
        new_correct = idx_remap[correct_idx]
        other_idx = P.shape[1] - 1
        yield pid, P, new_correct, other_idx, level


# ============================================================================
# Algorithm v1.0
# ============================================================================

def algorithm_v1(P, correct_idx: int, n_real: int, B: int, rng) -> tuple[float, str]:
    """v1.0 single-query routing.

    Returns (accuracy_share, path_label).
    """
    M, K = P.shape

    # --- Stage 0a: light pilot ---
    pilot = np.zeros((M, K), dtype=np.int32)
    for ti in range(M):
        pilot[ti] = rng.multinomial(N_PA, P[ti])
    cur_n = N_PA
    n_pilot_used = M * N_PA

    freq = pilot.astype(np.float64) / cur_n
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.sum(np.where(freq > 0, freq * np.log(freq), 0.0), axis=1)
    pilot_em = float(ent.min())

    # --- Stage 0a gate ---
    if pilot_em < TAU_EASY:
        path_pilot = "easy"
    elif pilot_em > TAU_HARD:
        # Hard fallback: equal_mix with remaining budget
        return _equal_mix_finish(P, B, pilot, n_pilot_used, rng, n_real,
                                  correct_idx), "hard_em"
    else:
        # Mid path: Stage 0b extended pilot
        for ti in range(M):
            pilot[ti] += rng.multinomial(N_PB, P[ti])
        cur_n = N_PA + N_PB
        n_pilot_used = M * cur_n
        path_pilot = "mid"

    # --- Gate A signals ---
    pilot_real = pilot[:, :n_real]
    top1_seq = pilot_real.argmax(axis=1)
    freq_full = pilot.astype(np.float64) / cur_n
    top1_freq_per_T = pilot_real.max(axis=1) / cur_n
    n_high = min(3, M)  # last 3 T positions (T≥1.0 for 12-T grids)
    high_sharp = float(top1_freq_per_T[-n_high:].mean())
    with np.errstate(divide="ignore", invalid="ignore"):
        ent_t0 = float(-np.sum(np.where(freq_full[0] > 0,
                                          freq_full[0] * np.log(freq_full[0]),
                                          0.0)))

    if high_sharp < GA_HIGH_SHARP or ent_t0 > GA_LOW_ENT:
        return _equal_mix_finish(P, B, pilot, n_pilot_used, rng, n_real,
                                  correct_idx), f"gateA_em_{path_pilot}"

    # --- Mode-pair extraction ---
    if len(set(top1_seq)) < 2:
        chosen_T = list(range(M))
        path_route = "no_swap"
    else:
        regions = []
        cur_ans = top1_seq[0]; start = 0
        for i in range(1, M):
            if top1_seq[i] != cur_ans:
                regions.append((cur_ans, list(range(start, i))))
                cur_ans = top1_seq[i]; start = i
        regions.append((cur_ans, list(range(start, M))))
        regions.sort(key=lambda r: -len(r[1]))
        r1, r2 = regions[0], regions[1]
        if np.mean(r1[1]) < np.mean(r2[1]):
            low_mode, high_mode = r1, r2
        else:
            low_mode, high_mode = r2, r1
        sbT_idx = M - 1
        for i in range(1, M):
            if top1_seq[i] != top1_seq[0]:
                sbT_idx = i; break
        low_width = len(low_mode[1])
        high_width = len(high_mode[1])
        high_T_mean = float(np.mean([(ti + 1) * 0.1 for ti in high_mode[1]]))

        # Gate B: strict AND-cut helps catch L4/L5 low_only
        strong_low_only = (high_width < GB_HIGH_WIDTH_MAX
                            and high_T_mean > GB_HIGH_T_MEAN_MIN)
        if (sbT_idx > GB_SBT_IDX or low_width > GB_LOW_WIDTH or strong_low_only):
            chosen_T = low_mode[1]
            path_route = "low_mode"
        else:
            chosen_T = high_mode[1]
            path_route = "high_mode"

    # --- Stage 4: main allocation on chosen region ---
    return _region_finish(P, B, pilot, n_pilot_used, rng, n_real, correct_idx,
                           chosen_T), f"route_{path_route}_{path_pilot}"


def _equal_mix_finish(P, B, current_pilot, current_total, rng, n_real,
                       correct_idx):
    M, K = P.shape
    remaining = max(B - current_total, 0)
    if remaining > 0:
        alloc = equal_alloc(remaining, M)
        main = np.zeros((M, K), dtype=np.int32)
        for ti in range(M):
            if alloc[ti] > 0:
                main[ti] = rng.multinomial(alloc[ti], P[ti])
        pool = current_pilot + main
    else:
        pool = current_pilot
    return share_with_ties(pool[:, :n_real].sum(axis=0), correct_idx)


def _region_finish(P, B, current_pilot, current_total, rng, n_real, correct_idx,
                    T_indices):
    M, K = P.shape
    remaining = max(B - current_total, 0)
    main = np.zeros((M, K), dtype=np.int32)
    if remaining > 0 and len(T_indices) > 0:
        per_T = remaining // len(T_indices)
        rem = remaining - per_T * len(T_indices)
        for j, ti in enumerate(T_indices):
            n = per_T + (1 if j < rem else 0)
            if n > 0:
                main[ti] = rng.multinomial(n, P[ti])
    pool = current_pilot + main
    return share_with_ties(pool[:, :n_real].sum(axis=0), correct_idx)


# ============================================================================
# CLI: simulate on a distributions file
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist-path", type=Path, required=True,
                    help="distributions.json or distributions_v2.json")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--B", type=int, default=B)
    ap.add_argument("--n-sims", type=int, default=N_SIMS_DEFAULT)
    ap.add_argument("--rng-seed", type=int, default=RNG_SEED_DEFAULT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    problems = list(load_problem_distributions(args.dist_path))
    print(f"loaded {len(problems)} problems")

    pid_results = []
    path_counter = {}
    for i, (pid, P, ci, other_idx, level) in enumerate(problems):
        sub_rng = np.random.default_rng(args.rng_seed + i * 31337)
        n_real = other_idx
        accs = np.zeros(args.n_sims, dtype=np.float32)
        for s in range(args.n_sims):
            acc, path = algorithm_v1(P, ci, n_real, args.B, sub_rng)
            accs[s] = acc
            path_counter[path] = path_counter.get(path, 0) + 1
        pid_results.append({"pid": pid, "level": level, "acc_mean": float(accs.mean())})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(problems)} ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(pid_results)
    df.to_csv(args.out_dir / "v1_per_pid.csv", index=False)
    overall = df["acc_mean"].mean()
    print(f"\noverall acc = {overall*100:.2f}%")
    print(f"path distribution:")
    total = sum(path_counter.values())
    for k, v in sorted(path_counter.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v/total*100:.2f}%")
    pd.Series(path_counter).to_csv(args.out_dir / "v1_path_counts.csv")


if __name__ == "__main__":
    main()
