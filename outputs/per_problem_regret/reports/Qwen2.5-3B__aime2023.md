# Qwen2.5-3B / aime2023

[← index](../REPORT.md) · figures: `figs/Qwen2.5-3B__aime2023/`

Regret R2 = (accuracy of the best *single fixed* temperature, chosen on the whole dataset) − (accuracy of the strategy), per problem. R1 (vs the per-problem oracle) lives in the parquet/CSV outputs.

## Oracle gap vs N

Per-problem oracle headroom over the best single fixed T (caveat: per-problem oracle ~1pp upward biased).

| N | best fixed T | dataset oracle acc | per-problem oracle acc | gap (pp) |
|---|---|---|---|---|
| 1 | T0.1 | 0.0761 | 0.0836 | +0.76 |
| 2 | T0.6 | 0.0773 | 0.0863 | +0.90 |
| 4 | T0.8 | 0.0833 | 0.0883 | +0.50 |
| 8 | T0.9 | 0.0849 | 0.0880 | +0.31 |
| 16 | T0.9 | 0.0847 | 0.0877 | +0.30 |
| 32 | T0.9 | 0.0834 | 0.0868 | +0.34 |
| 64 | T0.9 | 0.0815 | 0.0842 | +0.27 |
| 128 | T0.9 | 0.0789 | 0.0809 | +0.20 |
| 256 | T0.9 | 0.0765 | 0.0774 | +0.09 |
| 512 | T0.9 | 0.0746 | 0.0747 | +0.01 |
| 1024 | T0.9 | 0.0741 | 0.0741 | +0.00 |
| 1536 | T0.1 | 0.0741 | 0.0741 | +0.00 |
| 2048 | T0.1 | 0.0741 | 0.0741 | +0.00 |

## R2 regret — mean by strategy × N

| N | T=0.1 | T=1.0 | RandomT | Temperature Pool | Temperature Consensus |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0083 | 0.0052 | 0.0006 | 0.0006 |
| 2 | 0.0015 | 0.0061 | 0.0037 | 0.0010 | 0.0010 |
| 4 | 0.0059 | 0.0007 | 0.0027 | 0.0036 | 0.0036 |
| 8 | 0.0091 | 0.0001 | 0.0036 | 0.0048 | 0.0048 |
| 16 | 0.0100 | 0.0001 | 0.0044 | 0.0061 | 0.0067 |
| 32 | 0.0092 | 0.0005 | 0.0045 | 0.0064 | 0.0080 |
| 64 | 0.0074 | 0.0012 | 0.0040 | 0.0061 | 0.0058 |
| 128 | 0.0048 | 0.0016 | 0.0031 | 0.0045 | 0.0032 |
| 256 | 0.0024 | 0.0014 | 0.0018 | 0.0024 | 0.0020 |
| 512 | 0.0005 | 0.0004 | 0.0005 | 0.0005 | 0.0004 |
| 1024 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1536 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2048 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## R2 regret — p95 (tail) by strategy × N

| N | T=0.1 | T=1.0 | RandomT | Temperature Pool | Temperature Consensus |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.0866 | 0.0515 | 0.0052 | 0.0052 |
| 2 | 0.0180 | 0.0520 | 0.0229 | 0.0138 | 0.0138 |
| 4 | 0.0137 | 0.0138 | 0.0108 | 0.0072 | 0.0072 |
| 8 | 0.0171 | 0.0035 | 0.0142 | 0.0139 | 0.0139 |
| 16 | 0.0176 | 0.0000 | 0.0125 | 0.0113 | 0.0093 |
| 32 | 0.0226 | 0.0067 | 0.0163 | 0.0175 | 0.0111 |
| 64 | 0.0169 | 0.0054 | 0.0124 | 0.0146 | 0.0015 |
| 128 | 0.0095 | 0.0050 | 0.0083 | 0.0093 | 0.0000 |
| 256 | 0.0053 | 0.0039 | 0.0050 | 0.0053 | 0.0014 |
| 512 | 0.0013 | 0.0013 | 0.0013 | 0.0013 | 0.0008 |
| 1024 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 |
| 1536 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2048 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## T\* distribution @ N=256 (27 problems)

| T* | n_problems | share |
|---|---|---|
| 0.1 | 23 | 85.2% |
| 0.9 | 2 | 7.4% |
| 1.1 | 2 | 7.4% |

## Paired comparison @ N=256 (Δ = A − B per-problem accuracy)

| A | B | Δ mean | win/tie/loss | Wilcoxon p | BH adj-p |
|---|---|---|---|---|---|
| T=1.0 | Temperature Pool | 0.0010 | 1/26/0 | 0.18 | 0.22 |
| T=1.0 | Temperature Consensus | 0.0006 | 1/26/0 | 1 | 1 |
| Temperature Pool | Temperature Consensus | -0.0004 | 0/27/0 | 0.11 | 0.22 |
| T=1.0 | Best fixed T | -0.0014 | 0/26/1 | 0.18 | 0.22 |
| Temperature Pool | Best fixed T | -0.0024 | 0/26/1 | 0.18 | 0.22 |

## Stochastic dominance @ N=256 (R2)

- **T=1.0** dominates Temperature Pool
- **Temperature Consensus** dominates Temperature Pool
- **RandomT** dominates Temperature Pool
- **Best fixed T** dominates T=1.0
- **Best fixed T** dominates Temperature Pool
