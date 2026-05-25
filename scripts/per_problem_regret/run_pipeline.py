"""End-to-end driver: compute a_hat -> analyze -> figures -> REPORT.md.

    python scripts/per_problem_regret/run_pipeline.py \
        --combos aime2025_Qwen3-4B-Instruct-2507 --n-grid 128

Omit --combos to use every combo in combos.txt; omit --n-grid for the full grid.
Use --skip-compute to re-run only analysis/figures/report from existing parquets.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from per_problem_regret import analyze, build_report, compute_a_hat, plots
from per_problem_regret import config as C


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--combos", default=None, help="comma list; default = all in combos.txt")
    ap.add_argument("--out-dir", type=Path, default=C.OUT_DIR)
    ap.add_argument("--n-sims", type=int, default=C.Config.n_sims)
    ap.add_argument("--seed", type=int, default=C.Config.seed)
    ap.add_argument("--n-grid", default=None)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--skip-compute", action="store_true")
    args = ap.parse_args()

    cfg = C.Config(n_sims=args.n_sims, seed=args.seed)
    if args.n_grid:
        cfg.n_grid = sorted(int(x) for x in args.n_grid.split(","))
    combo_paths = C.load_combos()
    combos = args.combos.split(",") if args.combos else list(combo_paths)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_compute:
        for combo in combos:
            df, pid_text = compute_a_hat.compute_combo(
                combo, combo_paths[combo], cfg, n_jobs=args.n_jobs)
            df.to_parquet(args.out_dir / f"a_hat_per_problem__{combo}.parquet", index=False)
            import pandas as pd
            pd.DataFrame([{"problem_id": k, "problem_text": v}
                          for k, v in pid_text.items()]).to_csv(
                args.out_dir / f"pid_index__{combo}.csv", index=False)
            print(f"[{combo}] a_hat written", flush=True)

    art = analyze.run(args.out_dir, combos, cfg)
    plots.make_all(art, args.out_dir / "figs")
    text = build_report.build(args.out_dir)
    (args.out_dir / "REPORT.md").write_text(text)
    print(f"wrote {args.out_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
