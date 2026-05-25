# Phi-4-mini-instruct / aime2026

[← index](../REPORT.md) · figures: `figs/Phi-4-mini-instruct__aime2026/`

Regret R2 = (accuracy of the best *single fixed* temperature, chosen on the whole dataset) − (accuracy of the strategy), per problem. R1 (vs the per-problem oracle) lives in the parquet/CSV outputs.

## Oracle gap vs N

Per-problem oracle headroom over the best single fixed T (caveat: per-problem oracle ~1pp upward biased).

| N | best fixed T | dataset oracle acc | per-problem oracle acc | gap (pp) |
|---|---|---|---|---|
| 1 | T0.5 | 0.0289 | 0.0314 | +0.24 |
| 2 | T0.5 | 0.0317 | 0.0345 | +0.28 |
| 4 | T0.5 | 0.0365 | 0.0397 | +0.32 |
| 8 | T0.5 | 0.0454 | 0.0488 | +0.34 |
| 16 | T0.5 | 0.0507 | 0.0558 | +0.51 |
| 32 | T0.5 | 0.0547 | 0.0599 | +0.52 |
| 64 | T0.4 | 0.0559 | 0.0639 | +0.80 |
| 128 | T0.4 | 0.0561 | 0.0629 | +0.68 |
| 256 | T0.4 | 0.0559 | 0.0593 | +0.34 |
| 512 | T0.4 | 0.0545 | 0.0553 | +0.09 |
| 1024 | T0.4 | 0.0521 | 0.0521 | +0.00 |
| 1536 | T0.4 | 0.0497 | 0.0497 | +0.00 |
| 2048 | T0.4 | 0.0481 | 0.0481 | +0.00 |

## R2 regret — mean by strategy × N

| N | T=0.1 | T=1.0 | RandomT | Temperature Pool | Temperature Consensus |
|---|---|---|---|---|---|
| 1 | 0.0095 | 0.0099 | 0.0067 | 0.0094 | 0.0094 |
| 2 | 0.0117 | 0.0094 | 0.0071 | 0.0081 | 0.0081 |
| 4 | 0.0146 | 0.0135 | 0.0100 | 0.0070 | 0.0070 |
| 8 | 0.0238 | 0.0214 | 0.0149 | 0.0081 | 0.0081 |
| 16 | 0.0317 | 0.0286 | 0.0190 | 0.0216 | 0.0300 |
| 32 | 0.0413 | 0.0389 | 0.0241 | 0.0275 | 0.0456 |
| 64 | 0.0476 | 0.0439 | 0.0271 | 0.0323 | 0.0453 |
| 128 | 0.0526 | 0.0460 | 0.0277 | 0.0301 | 0.0394 |
| 256 | 0.0549 | 0.0473 | 0.0289 | 0.0272 | 0.0326 |
| 512 | 0.0543 | 0.0479 | 0.0289 | 0.0210 | 0.0224 |
| 1024 | 0.0521 | 0.0477 | 0.0278 | 0.0155 | 0.0135 |
| 1536 | 0.0497 | 0.0468 | 0.0252 | 0.0114 | 0.0092 |
| 2048 | 0.0481 | 0.0462 | 0.0238 | 0.0084 | 0.0069 |

## R2 regret — p95 (tail) by strategy × N

| N | T=0.1 | T=1.0 | RandomT | Temperature Pool | Temperature Consensus |
|---|---|---|---|---|---|
| 1 | 0.0684 | 0.0800 | 0.0539 | 0.0772 | 0.0772 |
| 2 | 0.0930 | 0.0700 | 0.0458 | 0.0656 | 0.0656 |
| 4 | 0.1121 | 0.1071 | 0.0769 | 0.0435 | 0.0435 |
| 8 | 0.2072 | 0.1586 | 0.1052 | 0.0604 | 0.0604 |
| 16 | 0.2401 | 0.1992 | 0.1370 | 0.1616 | 0.2101 |
| 32 | 0.2594 | 0.2325 | 0.1485 | 0.2076 | 0.2677 |
| 64 | 0.3792 | 0.3710 | 0.2613 | 0.2376 | 0.3656 |
| 128 | 0.3514 | 0.3512 | 0.2458 | 0.2383 | 0.3406 |
| 256 | 0.3064 | 0.3065 | 0.2349 | 0.2422 | 0.3014 |
| 512 | 0.2640 | 0.2640 | 0.2153 | 0.1632 | 0.1928 |
| 1024 | 0.2123 | 0.2123 | 0.1833 | 0.1044 | 0.0635 |
| 1536 | 0.1629 | 0.1629 | 0.1365 | 0.0687 | 0.0252 |
| 2048 | 0.1318 | 0.1318 | 0.1149 | 0.0388 | 0.0095 |

## T\* distribution @ N=256 (24 problems)

| T* | n_problems | share |
|---|---|---|
| 0.1 | 21 | 87.5% |
| 0.4 | 1 | 4.2% |
| 0.5 | 2 | 8.3% |

## Paired comparison @ N=256 (Δ = A − B per-problem accuracy)

| A | B | Δ mean | win/tie/loss | Wilcoxon p | BH adj-p |
|---|---|---|---|---|---|
| T=1.0 | Temperature Pool | -0.0201 | 0/23/1 | 0.32 | 0.4 |
| T=1.0 | Temperature Consensus | -0.0148 | 0/23/1 | 0.11 | 0.18 |
| Temperature Pool | Temperature Consensus | 0.0053 | 1/23/0 | 1 | 1 |
| T=1.0 | Best fixed T | -0.0473 | 0/21/3 | 0.11 | 0.18 |
| Temperature Pool | Best fixed T | -0.0272 | 0/21/3 | 0.11 | 0.18 |

## Stochastic dominance @ N=256 (R2)

- **Temperature Pool** dominates T=1.0
- **Temperature Consensus** dominates T=1.0
- **RandomT** dominates T=1.0
- **RandomT** dominates Temperature Consensus
- **Best fixed T** dominates T=1.0
- **Best fixed T** dominates Temperature Pool
- **Best fixed T** dominates Temperature Consensus
- **Best fixed T** dominates RandomT
