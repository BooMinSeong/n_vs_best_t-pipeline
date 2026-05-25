# Qwen3-4B-Instruct-2507 / aime2025

[← index](../REPORT.md) · figures: `figs/Qwen3-4B-Instruct-2507__aime2025/`

Regret R2 = (accuracy of the best *single fixed* temperature, chosen on the whole dataset) − (accuracy of the strategy), per problem. R1 (vs the per-problem oracle) lives in the parquet/CSV outputs.

## Oracle gap vs N

Per-problem oracle headroom over the best single fixed T (caveat: per-problem oracle ~1pp upward biased).

| N | best fixed T | dataset oracle acc | per-problem oracle acc | gap (pp) |
|---|---|---|---|---|
| 1 | T0.2 | 0.3762 | 0.3884 | +1.22 |
| 2 | T0.2 | 0.3828 | 0.3965 | +1.37 |
| 4 | T0.2 | 0.4150 | 0.4259 | +1.09 |
| 8 | T0.3 | 0.4403 | 0.4472 | +0.70 |
| 16 | T0.3 | 0.4513 | 0.4579 | +0.66 |
| 32 | T0.3 | 0.4604 | 0.4661 | +0.58 |
| 64 | T0.3 | 0.4668 | 0.4745 | +0.77 |
| 128 | T0.3 | 0.4727 | 0.4815 | +0.88 |
| 256 | T0.3 | 0.4781 | 0.4879 | +0.98 |
| 512 | T0.3 | 0.4865 | 0.4933 | +0.68 |
| 1024 | T0.3 | 0.4938 | 0.4970 | +0.31 |
| 1536 | T0.3 | 0.4972 | 0.4986 | +0.14 |
| 2048 | T0.3 | 0.4985 | 0.4993 | +0.08 |

## R2 regret — mean by strategy × N

| N | T=0.1 | T=1.0 | RandomT | Temperature Pool | Temperature Consensus |
|---|---|---|---|---|---|
| 1 | 0.0058 | 0.0259 | 0.0173 | 0.0047 | 0.0047 |
| 2 | 0.0043 | 0.0238 | 0.0150 | 0.0018 | 0.0018 |
| 4 | 0.0061 | 0.0263 | 0.0158 | 0.0027 | 0.0027 |
| 8 | 0.0111 | 0.0336 | 0.0206 | 0.0115 | 0.0115 |
| 16 | 0.0150 | 0.0392 | 0.0243 | 0.0179 | 0.0264 |
| 32 | 0.0218 | 0.0486 | 0.0300 | 0.0269 | 0.0456 |
| 64 | 0.0295 | 0.0555 | 0.0362 | 0.0372 | 0.0513 |
| 128 | 0.0359 | 0.0633 | 0.0426 | 0.0492 | 0.0560 |
| 256 | 0.0401 | 0.0689 | 0.0482 | 0.0630 | 0.0632 |
| 512 | 0.0471 | 0.0773 | 0.0583 | 0.0756 | 0.0740 |
| 1024 | 0.0511 | 0.0847 | 0.0653 | 0.0846 | 0.0834 |
| 1536 | 0.0526 | 0.0881 | 0.0683 | 0.0881 | 0.0874 |
| 2048 | 0.0526 | 0.0894 | 0.0703 | 0.0894 | 0.0890 |

## R2 regret — p95 (tail) by strategy × N

| N | T=0.1 | T=1.0 | RandomT | Temperature Pool | Temperature Consensus |
|---|---|---|---|---|---|
| 1 | 0.0435 | 0.1134 | 0.0849 | 0.0396 | 0.0396 |
| 2 | 0.0433 | 0.1132 | 0.0667 | 0.0143 | 0.0143 |
| 4 | 0.0424 | 0.1702 | 0.0935 | 0.0146 | 0.0146 |
| 8 | 0.0626 | 0.1959 | 0.1268 | 0.0612 | 0.0612 |
| 16 | 0.0779 | 0.2658 | 0.1890 | 0.1324 | 0.2146 |
| 32 | 0.1065 | 0.3583 | 0.2676 | 0.2448 | 0.4342 |
| 64 | 0.0879 | 0.4854 | 0.3196 | 0.3551 | 0.4931 |
| 128 | 0.1126 | 0.6131 | 0.3987 | 0.5087 | 0.5691 |
| 256 | 0.1172 | 0.7120 | 0.4473 | 0.6571 | 0.6509 |
| 512 | 0.1726 | 0.8042 | 0.5319 | 0.7868 | 0.7647 |
| 1024 | 0.1815 | 0.8843 | 0.5981 | 0.8823 | 0.8678 |
| 1536 | 0.1801 | 0.9198 | 0.6181 | 0.9198 | 0.9119 |
| 2048 | 0.1643 | 0.9326 | 0.6342 | 0.9326 | 0.9272 |

## T\* distribution @ N=256 (22 problems)

| T* | n_problems | share |
|---|---|---|
| 0.1 | 20 | 90.9% |
| 0.2 | 1 | 4.5% |
| 0.3 | 1 | 4.5% |

## Paired comparison @ N=256 (Δ = A − B per-problem accuracy)

| A | B | Δ mean | win/tie/loss | Wilcoxon p | BH adj-p |
|---|---|---|---|---|---|
| T=1.0 | Temperature Pool | -0.0059 | 0/20/2 | 0.11 | 0.14 |
| T=1.0 | Temperature Consensus | -0.0057 | 0/20/2 | 0.11 | 0.14 |
| Temperature Pool | Temperature Consensus | 0.0002 | 1/21/0 | 0.59 | 0.59 |
| T=1.0 | Best fixed T | -0.0689 | 0/20/2 | 0.11 | 0.14 |
| Temperature Pool | Best fixed T | -0.0630 | 0/20/2 | 0.11 | 0.14 |

## Stochastic dominance @ N=256 (R2)

- **Temperature Pool** dominates T=1.0
- **Temperature Consensus** dominates T=1.0
- **RandomT** dominates T=1.0
- **Best fixed T** dominates T=1.0
- **Best fixed T** dominates Temperature Pool
- **Best fixed T** dominates Temperature Consensus
- **Best fixed T** dominates RandomT
