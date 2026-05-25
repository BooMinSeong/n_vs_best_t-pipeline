# Phi-4-mini-instruct / aime2024

[← index](../REPORT.md) · figures: `figs/Phi-4-mini-instruct__aime2024/`

Regret R2 = (accuracy of the best *single fixed* temperature, chosen on the whole dataset) − (accuracy of the strategy), per problem. R1 (vs the per-problem oracle) lives in the parquet/CSV outputs.

## Oracle gap vs N

Per-problem oracle headroom over the best single fixed T (caveat: per-problem oracle ~1pp upward biased).

| N | best fixed T | dataset oracle acc | per-problem oracle acc | gap (pp) |
|---|---|---|---|---|
| 1 | T0.1 | 0.0356 | 0.0427 | +0.71 |
| 2 | T0.1 | 0.0382 | 0.0474 | +0.92 |
| 4 | T0.1 | 0.0428 | 0.0519 | +0.91 |
| 8 | T0.4 | 0.0442 | 0.0510 | +0.68 |
| 16 | T0.5 | 0.0431 | 0.0464 | +0.33 |
| 32 | T0.1 | 0.0425 | 0.0434 | +0.09 |
| 64 | T0.1 | 0.0426 | 0.0427 | +0.01 |
| 128 | T0.1 | 0.0420 | 0.0420 | +0.00 |
| 256 | T0.1 | 0.0407 | 0.0407 | +0.00 |
| 512 | T0.1 | 0.0397 | 0.0397 | +0.00 |
| 1024 | T0.1 | 0.0387 | 0.0387 | +0.00 |
| 1536 | T0.1 | 0.0385 | 0.0385 | +0.00 |
| 2048 | T0.1 | 0.0385 | 0.0385 | +0.00 |

## R2 regret — mean by strategy × N

| N | T=0.1 | T=1.0 | RandomT | Temperature Pool | Temperature Consensus |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0184 | 0.0104 | 0.0000 | 0.0000 |
| 2 | 0.0000 | 0.0160 | 0.0094 | 0.0005 | 0.0005 |
| 4 | 0.0000 | 0.0179 | 0.0109 | 0.0018 | 0.0018 |
| 8 | 0.0004 | 0.0164 | 0.0091 | 0.0027 | 0.0027 |
| 16 | 0.0001 | 0.0123 | 0.0083 | 0.0035 | 0.0042 |
| 32 | 0.0000 | 0.0093 | 0.0080 | 0.0037 | 0.0032 |
| 64 | 0.0000 | 0.0058 | 0.0080 | 0.0041 | 0.0032 |
| 128 | 0.0000 | 0.0038 | 0.0071 | 0.0036 | 0.0034 |
| 256 | 0.0000 | 0.0023 | 0.0056 | 0.0022 | 0.0022 |
| 512 | 0.0000 | 0.0012 | 0.0046 | 0.0012 | 0.0012 |
| 1024 | 0.0000 | 0.0003 | 0.0034 | 0.0003 | 0.0003 |
| 1536 | 0.0000 | 0.0001 | 0.0032 | 0.0001 | 0.0001 |
| 2048 | 0.0000 | 0.0000 | 0.0033 | 0.0000 | 0.0000 |

## R2 regret — p95 (tail) by strategy × N

| N | T=0.1 | T=1.0 | RandomT | Temperature Pool | Temperature Consensus |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0494 | 0.0457 | 0.0031 | 0.0031 |
| 2 | 0.0000 | 0.0517 | 0.0464 | 0.0092 | 0.0092 |
| 4 | 0.0000 | 0.0351 | 0.0309 | 0.0216 | 0.0216 |
| 8 | 0.0042 | 0.0473 | 0.0413 | 0.0114 | 0.0114 |
| 16 | 0.0038 | 0.0351 | 0.0294 | 0.0165 | 0.0249 |
| 32 | 0.0000 | 0.0727 | 0.0564 | 0.0020 | 0.0085 |
| 64 | 0.0000 | 0.0324 | 0.0673 | 0.0002 | 0.0007 |
| 128 | 0.0000 | 0.0053 | 0.0613 | 0.0000 | 0.0000 |
| 256 | 0.0000 | 0.0001 | 0.0405 | 0.0000 | 0.0000 |
| 512 | 0.0000 | 0.0000 | 0.0225 | 0.0000 | 0.0000 |
| 1024 | 0.0000 | 0.0000 | 0.0046 | 0.0000 | 0.0000 |
| 1536 | 0.0000 | 0.0000 | 0.0016 | 0.0000 | 0.0000 |
| 2048 | 0.0000 | 0.0000 | 0.0006 | 0.0000 | 0.0000 |

## T\* distribution @ N=256 (26 problems)

| T* | n_problems | share |
|---|---|---|
| 0.1 | 26 | 100.0% |

## Paired comparison @ N=256 (Δ = A − B per-problem accuracy)

| A | B | Δ mean | win/tie/loss | Wilcoxon p | BH adj-p |
|---|---|---|---|---|---|
| T=1.0 | Temperature Pool | -0.0000 | 0/26/0 | 0.32 | 0.32 |
| T=1.0 | Temperature Consensus | -0.0000 | 0/26/0 | 0.18 | 0.32 |
| Temperature Pool | Temperature Consensus | -0.0000 | 0/26/0 | 0.32 | 0.32 |
| T=1.0 | Best fixed T | -0.0023 | 0/25/1 | 0.18 | 0.32 |
| Temperature Pool | Best fixed T | -0.0022 | 0/25/1 | 0.32 | 0.32 |

## Stochastic dominance @ N=256 (R2)

- **T=1.0** dominates RandomT
- **Temperature Pool** dominates T=1.0
- **Temperature Pool** dominates RandomT
- **Temperature Consensus** dominates T=1.0
- **Temperature Consensus** dominates Temperature Pool
- **Temperature Consensus** dominates RandomT
- **Best fixed T** dominates T=1.0
- **Best fixed T** dominates Temperature Pool
- **Best fixed T** dominates Temperature Consensus
- **Best fixed T** dominates RandomT
