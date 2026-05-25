# Qwen3-4B-Instruct-2507 / aime2024

[← index](../REPORT.md) · figures: `figs/Qwen3-4B-Instruct-2507__aime2024/`

Regret R2 = (accuracy of the best *single fixed* temperature, chosen on the whole dataset) − (accuracy of the strategy), per problem. R1 (vs the per-problem oracle) lives in the parquet/CSV outputs.

## Oracle gap vs N

Per-problem oracle headroom over the best single fixed T (caveat: per-problem oracle ~1pp upward biased).

| N | best fixed T | dataset oracle acc | per-problem oracle acc | gap (pp) |
|---|---|---|---|---|
| 1 | T0.7 | 0.3464 | 0.3734 | +2.69 |
| 2 | T1.0 | 0.3575 | 0.3835 | +2.60 |
| 4 | T1.0 | 0.4048 | 0.4348 | +3.00 |
| 8 | T1.0 | 0.4377 | 0.4675 | +2.99 |
| 16 | T0.9 | 0.4556 | 0.4838 | +2.81 |
| 32 | T0.9 | 0.4665 | 0.4900 | +2.35 |
| 64 | T0.7 | 0.4735 | 0.4893 | +1.59 |
| 128 | T0.7 | 0.4765 | 0.4866 | +1.01 |
| 256 | T0.7 | 0.4759 | 0.4824 | +0.64 |
| 512 | T0.7 | 0.4741 | 0.4774 | +0.32 |
| 1024 | T0.7 | 0.4713 | 0.4722 | +0.08 |
| 1536 | T0.7 | 0.4697 | 0.4699 | +0.02 |
| 2048 | T0.7 | 0.4682 | 0.4683 | +0.01 |

## R2 regret — mean by strategy × N

| N | T=0.1 | T=1.0 | RandomT | Temperature Pool | Temperature Consensus |
|---|---|---|---|---|---|
| 1 | 0.0144 | 0.0029 | 0.0071 | 0.0155 | 0.0155 |
| 2 | 0.0132 | 0.0000 | 0.0055 | 0.0124 | 0.0124 |
| 4 | 0.0171 | 0.0000 | 0.0086 | 0.0136 | 0.0136 |
| 8 | 0.0221 | 0.0000 | 0.0109 | 0.0131 | 0.0131 |
| 16 | 0.0297 | 0.0002 | 0.0127 | 0.0131 | 0.0067 |
| 32 | 0.0387 | 0.0008 | 0.0157 | 0.0135 | 0.0204 |
| 64 | 0.0469 | 0.0032 | 0.0189 | 0.0140 | 0.0181 |
| 128 | 0.0522 | 0.0067 | 0.0221 | 0.0156 | 0.0189 |
| 256 | 0.0527 | 0.0091 | 0.0240 | 0.0145 | 0.0171 |
| 512 | 0.0511 | 0.0102 | 0.0242 | 0.0127 | 0.0158 |
| 1024 | 0.0482 | 0.0092 | 0.0237 | 0.0098 | 0.0126 |
| 1536 | 0.0466 | 0.0081 | 0.0230 | 0.0082 | 0.0109 |
| 2048 | 0.0451 | 0.0066 | 0.0209 | 0.0067 | 0.0091 |

## R2 regret — p95 (tail) by strategy × N

| N | T=0.1 | T=1.0 | RandomT | Temperature Pool | Temperature Consensus |
|---|---|---|---|---|---|
| 1 | 0.1146 | 0.0348 | 0.0319 | 0.1237 | 0.1237 |
| 2 | 0.1420 | 0.0000 | 0.0729 | 0.1326 | 0.1326 |
| 4 | 0.1392 | 0.0000 | 0.0859 | 0.1422 | 0.1422 |
| 8 | 0.1236 | 0.0000 | 0.0875 | 0.1182 | 0.1182 |
| 16 | 0.1408 | 0.0452 | 0.0620 | 0.0511 | 0.0977 |
| 32 | 0.1696 | 0.0404 | 0.0497 | 0.0434 | 0.1689 |
| 64 | 0.2981 | 0.0282 | 0.1452 | 0.1461 | 0.0993 |
| 128 | 0.3065 | 0.0052 | 0.1770 | 0.0981 | 0.1142 |
| 256 | 0.2825 | 0.0001 | 0.1870 | 0.0462 | 0.0926 |
| 512 | 0.2458 | 0.0000 | 0.1933 | 0.0105 | 0.0807 |
| 1024 | 0.1907 | 0.0000 | 0.1629 | 0.0004 | 0.0603 |
| 1536 | 0.1592 | 0.0000 | 0.1398 | 0.0000 | 0.0551 |
| 2048 | 0.1299 | 0.0000 | 0.1147 | 0.0000 | 0.0484 |

## T\* distribution @ N=256 (26 problems)

| T* | n_problems | share |
|---|---|---|
| 0.1 | 23 | 88.5% |
| 0.3 | 1 | 3.8% |
| 0.7 | 1 | 3.8% |
| 0.9 | 1 | 3.8% |

## Paired comparison @ N=256 (Δ = A − B per-problem accuracy)

| A | B | Δ mean | win/tie/loss | Wilcoxon p | BH adj-p |
|---|---|---|---|---|---|
| T=1.0 | Temperature Pool | 0.0054 | 2/24/0 | 0.066 | 0.11 |
| T=1.0 | Temperature Consensus | 0.0080 | 2/24/0 | 0.091 | 0.11 |
| Temperature Pool | Temperature Consensus | 0.0026 | 1/25/0 | 0.09 | 0.11 |
| T=1.0 | Best fixed T | -0.0091 | 0/25/1 | 0.59 | 0.59 |
| Temperature Pool | Best fixed T | -0.0145 | 0/24/2 | 0.042 | 0.11 |

## Stochastic dominance @ N=256 (R2)

- **T=1.0** dominates Temperature Pool
- **T=1.0** dominates Temperature Consensus
- **Best fixed T** dominates Temperature Pool
