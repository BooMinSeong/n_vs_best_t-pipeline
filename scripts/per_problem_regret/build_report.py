"""Stage 4 — assemble a *split* report from the summary tables.

Instead of one 12k-line flat file (247 sections), emit:

    REPORT.md                       top-level index: methodology, oracle-gap matrix,
                                     cross-combo overview, links to per-combo files
    reports/<model>__<dataset>.md   one file per combo, organised as compact
                                     cross-N tables (not 13 repeated per-N blocks)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from per_problem_regret import config as C

# strategies shown in the report (R2 regret); best_fixed_T omitted from regret
# tables since it is 0 by construction.
REGRET_STRATS = ["T0.1", "T1.0", "random_T", "equal_mix", "consensus_vote"]
SNAPSHOT_N = 256  # point-in-time N for T* / paired / dominance tables


def _fmt(x, nd=4):
    return "—" if pd.isna(x) else f"{x:.{nd}f}"


def disp(s: str) -> str:
    return C.DISPLAY_NAME.get(s, "Best fixed T" if s == "best_fixed_T" else s)


def slug(model: str, dataset: str) -> str:
    return f"{model}__{dataset}"


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    out.append("")
    return out


def _pick_n(available, target=SNAPSHOT_N) -> int:
    av = sorted(int(x) for x in available)
    return target if target in av else av[-1]


# --------------------------------------------------------------------------- #
# Per-combo file
# --------------------------------------------------------------------------- #
def build_combo(model, dataset, gap, dist, tstar, paired, dom) -> str:
    mq = "model==@model and dataset==@dataset"
    g = gap.query(mq).sort_values("N")
    snap = _pick_n(g.N.unique())
    md = [f"# {model} / {dataset}", "",
          "[← index](../REPORT.md) · "
          f"figures: `figs/{slug(model, dataset)}/`", "",
          "Regret R2 = (accuracy of the best *single fixed* temperature, chosen on the "
          "whole dataset) − (accuracy of the strategy), per problem. R1 (vs the "
          "per-problem oracle) lives in the parquet/CSV outputs.", ""]

    # Oracle gap vs N
    md += ["## Oracle gap vs N", "",
           "Per-problem oracle headroom over the best single fixed T (caveat: "
           "per-problem oracle ~1pp upward biased).", ""]
    rows = [[str(int(r.N)), r.dataset_best_t, _fmt(r.dataset_oracle_acc),
             _fmt(r.per_problem_oracle_acc), f"+{r.gap_pp:.2f}"]
            for _, r in g.iterrows()]
    md += _md_table(["N", "best fixed T", "dataset oracle acc",
                     "per-problem oracle acc", "gap (pp)"], rows)

    # Regret cross-N tables (mean + p95)
    d2 = dist.query(mq + " and regret_kind=='R2'")
    for metric, label in [("regret_mean", "mean"), ("regret_p95", "p95 (tail)")]:
        piv = d2.pivot_table(index="N", columns="strategy", values=metric)
        cols = [c for c in REGRET_STRATS if c in piv.columns]
        md += [f"## R2 regret — {label} by strategy × N", ""]
        rows = [[str(int(n))] + [_fmt(piv.loc[n, c]) for c in cols]
                for n in sorted(piv.index)]
        md += _md_table(["N"] + [disp(c) for c in cols], rows)

    # T* distribution at snapshot N
    t = tstar.query(mq + " and N==@snap").sort_values("T_star")
    if len(t):
        tot = t.n_problems.sum()
        md += [f"## T\\* distribution @ N={snap} ({int(tot)} problems)", ""]
        rows = [[f"{r.T_star:.1f}", str(int(r.n_problems)),
                 f"{r.n_problems / tot * 100:.1f}%"] for _, r in t.iterrows()]
        md += _md_table(["T*", "n_problems", "share"], rows)

    # Paired comparison at snapshot N
    p = paired.query(mq + " and N==@snap")
    if len(p):
        md += [f"## Paired comparison @ N={snap} (Δ = A − B per-problem accuracy)", ""]
        rows = [[disp(r.strategy_a), disp(r.strategy_b), _fmt(r.delta_mean),
                 f"{int(r.wins_a)}/{int(r.ties)}/{int(r.wins_b)}",
                 f"{r.wilcoxon_p:.2g}", f"{r.adj_p_value:.2g}"]
                for _, r in p.iterrows()]
        md += _md_table(["A", "B", "Δ mean", "win/tie/loss", "Wilcoxon p", "BH adj-p"], rows)

    # Stochastic dominance at snapshot N
    dd = dom.query(mq + " and N==@snap and regret_kind=='R2' and a_dominates_b")
    md += [f"## Stochastic dominance @ N={snap} (R2)", ""]
    if len(dd):
        md += [f"- **{disp(r.strategy_a)}** dominates {disp(r.strategy_b)}"
               for _, r in dd.iterrows()] + [""]
    else:
        md += ["_No strategy first-order dominates another (regret CDFs cross)._", ""]
    return "\n".join(md)


# --------------------------------------------------------------------------- #
# Top-level index
# --------------------------------------------------------------------------- #
def build_index(gap, dist) -> str:
    snap = _pick_n(gap.N.unique())
    md = ["# Per-Problem Regret Analysis", "",
          "> **Methodology**: per-problem maj@N accuracy is Monte-Carlo estimated by "
          "sampling from the empirical per-temperature answer distribution "
          "(`distributions.json`, 6×256=1536 generations/problem-T), not by bootstrapping "
          "raw generations — same currency as the dataset-level `sim_v2` tables.", "",
          "> **Caveat**: the per-problem oracle is max-over-T of noisy estimates, ~1pp "
          "**upward** biased. Absolute oracle gaps inherit this; strategy-vs-strategy "
          "comparisons do not (common shift cancels).", "",
          f"Snapshot tables use N={snap}. Per-combo detail (all N) in `reports/`.", ""]

    # Oracle gap matrix (model x dataset) at snapshot N
    gsnap = gap.query("N==@snap")
    piv = gsnap.pivot_table(index="model", columns="dataset", values="gap_pp")
    md += [f"## Oracle gap matrix (pp) @ N={snap}", "",
           "Per-problem-oracle minus best-single-fixed-T accuracy. Larger = more to "
           "gain from per-problem temperature selection.", ""]
    cols = list(piv.columns)
    rows = [[m] + [("—" if pd.isna(piv.loc[m, c]) else f"{piv.loc[m, c]:.2f}")
                   for c in cols] for m in piv.index]
    md += _md_table(["model \\ dataset"] + cols, rows)

    # Cross-combo overview: deployable-strategy mean R2 regret at snapshot N
    d2 = dist.query("N==@snap and regret_kind=='R2'")
    md += [f"## Deployable-strategy mean R2 regret @ N={snap}", "",
           "Lower = closer to the best single fixed T.", ""]
    over = d2.pivot_table(index=["dataset", "model"], columns="strategy",
                          values="regret_mean")
    cols = [c for c in ["T1.0", "random_T", "equal_mix", "consensus_vote"]
            if c in over.columns]
    rows = [[f"{ds} / {mo}"] + [_fmt(over.loc[(ds, mo), c]) for c in cols]
            for ds, mo in over.index]
    md += _md_table(["dataset / model"] + [disp(c) for c in cols], rows)

    # TOC with per-combo gap at snapshot N
    md += ["## Per-combo reports", ""]
    for (model, dataset), grp in gsnap.groupby(["model", "dataset"]):
        gp = grp.iloc[0].gap_pp
        md.append(f"- [{model} / {dataset}](reports/{slug(model, dataset)}.md) "
                  f"— oracle gap +{gp:.2f} pp")
    md.append("")
    return "\n".join(md)


def build(out_dir: Path) -> str:
    gap = pd.read_csv(out_dir / "summary_oracle_gap.csv")
    dist = pd.read_csv(out_dir / "summary_distribution_stats.csv")
    tstar = pd.read_csv(out_dir / "summary_t_star_distribution.csv")
    paired = pd.read_csv(out_dir / "summary_paired_comparison.csv")
    dom = pd.read_csv(out_dir / "summary_stochastic_dominance.csv")

    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for (model, dataset) in gap[["model", "dataset"]].drop_duplicates().itertuples(index=False):
        text = build_combo(model, dataset, gap, dist, tstar, paired, dom)
        (reports_dir / f"{slug(model, dataset)}.md").write_text(text)

    index = build_index(gap, dist)
    (out_dir / "REPORT.md").write_text(index)
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=C.OUT_DIR)
    args = ap.parse_args()
    build(args.out_dir)
    n = len(list((args.out_dir / "reports").glob("*.md")))
    print(f"wrote {args.out_dir / 'REPORT.md'} + {n} per-combo files under reports/")


if __name__ == "__main__":
    main()
