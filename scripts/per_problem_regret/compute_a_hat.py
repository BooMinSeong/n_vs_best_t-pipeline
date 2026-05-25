"""Stage 1 — per-problem maj@N accuracy table (spec section 2.1 / 2.3).

For one combo, Monte-Carlo estimate a_hat(problem, strategy, N) by sampling from the
empirical multinomial distribution P (reusing the existing v2 sim engine so the
per-problem mean reconciles with the dataset-level tables).  Output one parquet:

    a_hat_per_problem__<combo>.parquet
        columns: model, dataset, problem_id, level, stratum, strategy, N, a_hat
        plus reproducibility metadata: git_commit, timestamp, config_hash

A side-car CSV maps the integer problem_id back to the original problem text.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_t_mode.lb_baselines import load_aligned  # type: ignore  # noqa: E402
from n_vs_best_t.run_sim_v2_one import per_problem_sims_v2  # type: ignore  # noqa: E402

from per_problem_regret import config as C  # noqa: E402


def level_to_stratum(level) -> str:
    return f"L{level}" if level in (1, 2, 3, 4, 5) else "Lr"


def _one_problem(idx: int, P, ci, other_idx, level, seed, n_grid, n_sims):
    """Return list of row dicts for one problem (collapse n_sims -> mean a_hat)."""
    sub_rng = np.random.default_rng(seed + idx * 31337)
    sims = per_problem_sims_v2(P, ci, other_idx, sub_rng, n_grid, n_sims)
    stratum = level_to_stratum(level)
    rows = []
    for N in n_grid:
        for strat, arr in sims[N].items():
            rows.append({
                "problem_id": idx,
                "level": level if level is not None else -1,
                "stratum": stratum,
                "strategy": strat,
                "N": N,
                "a_hat": float(arr.mean()),
            })
    return rows


def compute_combo(combo: str, dist_path: Path, cfg: C.Config,
                  n_jobs: int = -1) -> pd.DataFrame:
    dataset, model = C.parse_combo(combo)
    t0 = time.time()
    print(f"[{combo}] loading {dist_path}", flush=True)
    problems = list(load_aligned(dist_path))
    print(f"[{combo}] {len(problems)} problems loaded ({time.time()-t0:.1f}s)", flush=True)

    pid_text = {i: pid for i, (pid, *_rest) in enumerate(problems)}

    jobs = (
        delayed(_one_problem)(i, P, ci, other_idx, level, cfg.seed,
                              cfg.n_grid, cfg.n_sims)
        for i, (pid, P, ci, other_idx, level, _temps) in enumerate(problems)
    )
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, batch_size="auto")(
        tqdm(jobs, total=len(problems), desc=f"{combo}")
    )
    print(f"[{combo}] sims done ({time.time()-t0:.1f}s)", flush=True)

    rows = [r for sub in results for r in sub]
    df = pd.DataFrame(rows)
    df["model"] = model
    df["dataset"] = dataset
    df["git_commit"] = C.git_commit()
    df["timestamp"] = C.now_iso()
    df["config_hash"] = cfg.hash()
    cols = ["model", "dataset", "problem_id", "level", "stratum",
            "strategy", "N", "a_hat", "git_commit", "timestamp", "config_hash"]
    return df[cols], pid_text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--combo", required=True, help="e.g. aime2025_Qwen3-4B-Instruct-2507")
    ap.add_argument("--dist-path", type=Path, default=None,
                    help="override; otherwise resolved from combos.txt")
    ap.add_argument("--out-dir", type=Path, default=C.OUT_DIR)
    ap.add_argument("--n-sims", type=int, default=C.Config.n_sims)
    ap.add_argument("--seed", type=int, default=C.Config.seed)
    ap.add_argument("--n-grid", default=None, help="comma list; default = full grid")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    cfg = C.Config(n_sims=args.n_sims, seed=args.seed)
    if args.n_grid:
        cfg.n_grid = sorted(int(x) for x in args.n_grid.split(","))

    dist_path = args.dist_path or C.load_combos()[args.combo]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df, pid_text = compute_combo(args.combo, dist_path, cfg, n_jobs=args.n_jobs)

    out_pq = out_dir / f"a_hat_per_problem__{args.combo}.parquet"
    df.to_parquet(out_pq, index=False)
    pid_csv = out_dir / f"pid_index__{args.combo}.csv"
    pd.DataFrame(
        [{"problem_id": k, "problem_text": v} for k, v in pid_text.items()]
    ).to_csv(pid_csv, index=False)
    print(f"[{args.combo}] wrote {out_pq} ({len(df)} rows) + {pid_csv.name}", flush=True)


if __name__ == "__main__":
    main()
