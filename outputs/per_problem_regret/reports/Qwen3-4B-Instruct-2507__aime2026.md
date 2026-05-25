# Qwen3-4B-Instruct-2507 / aime2026

[← index](../REPORT.md) · figures: `figs/Qwen3-4B-Instruct-2507__aime2026/`

Regret R2 = (accuracy of the best *single fixed* temperature, chosen on the whole dataset) − (accuracy of the strategy), per problem. R1 (vs the per-problem oracle) lives in the parquet/CSV outputs.

## Oracle gap vs N

Per-problem oracle headroom over the best single fixed T (caveat: per-problem oracle ~1pp upward biased).

| N | best fixed T | dataset oracle acc | per-problem oracle acc | gap (pp) |
|---|---|---|---|---|
| 1 | T0.6 | 0.3775 | 0.4003 | +2.28 |
| 2 | T0.8 | 0.3917 | 0.4121 | +2.03 |
| 4 | T0.8 | 0.4516 | 0.4726 | +2.10 |
| 8 | T0.8 | 0.4941 | 0.5095 | +1.54 |
| 16 | T0.8 | 0.5137 | 0.5275 | +1.38 |
| 32 | T0.8 | 0.5244 | 0.5358 | +1.14 |
| 64 | T1.1 | 0.5319 | 0.5399 | +0.80 |
| 128 | T1.1 | 0.5373 | 0.5412 | +0.40 |
| 256 | T1.1 | 0.5406 | 0.5416 | +0.11 |
| 512 | T1.1 | 0.5416 | 0.5417 | +0.01 |
| 1024 | T1.1 | 0.5417 | 0.5417 | +0.00 |
| 1536 | T1.0 | 0.5417 | 0.5417 | +0.00 |
| 2048 | T1.0 | 0.5417 | 0.5417 | +0.00 |

## R2 regret — mean by strategy × N

| N | T=0.1 | T=1.0 | RandomT | Temperature Pool | Temperature Consensus |
|---|---|---|---|---|---|
| 1 | 0.0155 | 0.0042 | 0.0078 | 0.0150 | 0.0150 |
| 2 | 0.0189 | 0.0053 | 0.0096 | 0.0193 | 0.0193 |
| 4 | 0.0276 | 0.0033 | 0.0099 | 0.0216 | 0.0216 |
| 8 | 0.0328 | 0.0038 | 0.0123 | 0.0102 | 0.0102 |
| 16 | 0.0327 | 0.0026 | 0.0119 | 0.0095 | 0.0198 |
| 32 | 0.0311 | 0.0016 | 0.0116 | 0.0098 | 0.0388 |
| 64 | 0.0315 | 0.0021 | 0.0126 | 0.0125 | 0.0348 |
| 128 | 0.0352 | 0.0012 | 0.0152 | 0.0146 | 0.0269 |
| 256 | 0.0388 | 0.0008 | 0.0174 | 0.0153 | 0.0223 |
| 512 | 0.0413 | 0.0002 | 0.0175 | 0.0144 | 0.0191 |
| 1024 | 0.0417 | 0.0000 | 0.0164 | 0.0121 | 0.0157 |
| 1536 | 0.0417 | 0.0000 | 0.0161 | 0.0098 | 0.0135 |
| 2048 | 0.0417 | 0.0000 | 0.0160 | 0.0085 | 0.0120 |

## R2 regret — p95 (tail) by strategy × N

| N | T=0.1 | T=1.0 | RandomT | Temperature Pool | Temperature Consensus |
|---|---|---|---|---|---|
| 1 | 0.1513 | 0.0510 | 0.0416 | 0.1414 | 0.1414 |
| 2 | 0.1720 | 0.0443 | 0.0401 | 0.1553 | 0.1553 |
| 4 | 0.1696 | 0.0210 | 0.0529 | 0.1246 | 0.1246 |
| 8 | 0.1884 | 0.0147 | 0.0624 | 0.0537 | 0.0537 |
| 16 | 0.2212 | 0.0226 | 0.0346 | 0.0372 | 0.0717 |
| 32 | 0.1548 | 0.0277 | 0.0279 | 0.0340 | 0.2372 |
| 64 | 0.0705 | 0.0049 | 0.0088 | 0.0068 | 0.1163 |
| 128 | 0.0253 | 0.0002 | 0.0017 | 0.0003 | 0.0338 |
| 256 | 0.0036 | 0.0000 | 0.0002 | 0.0000 | 0.0009 |
| 512 | 0.0000 | 0.0000 | 0.0033 | 0.0000 | 0.0003 |
| 1024 | 0.0000 | 0.0000 | 0.0012 | 0.0000 | 0.0000 |
| 1536 | 0.0000 | 0.0000 | 0.0005 | 0.0000 | 0.0000 |
| 2048 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## T\* distribution @ N=256 (24 problems)

| T* | n_problems | share |
|---|---|---|
| 0.1 | 20 | 83.3% |
| 0.2 | 1 | 4.2% |
| 0.4 | 1 | 4.2% |
| 0.6 | 1 | 4.2% |
| 1.1 | 1 | 4.2% |

## Paired comparison @ N=256 (Δ = A − B per-problem accuracy)

| A | B | Δ mean | win/tie/loss | Wilcoxon p | BH adj-p |
|---|---|---|---|---|---|
| T=1.0 | Temperature Pool | 0.0144 | 1/23/0 | 0.18 | 0.3 |
| T=1.0 | Temperature Consensus | 0.0215 | 2/22/0 | 0.11 | 0.27 |
| Temperature Pool | Temperature Consensus | 0.0070 | 2/22/0 | 0.11 | 0.27 |
| T=1.0 | Best fixed T | -0.0008 | 1/22/1 | 0.65 | 0.65 |
| Temperature Pool | Best fixed T | -0.0153 | 1/22/1 | 0.65 | 0.65 |

## Stochastic dominance @ N=256 (R2)

- **T=1.0** dominates Temperature Pool
- **T=1.0** dominates Temperature Consensus
- **T=1.0** dominates RandomT
- **Temperature Pool** dominates Temperature Consensus
- **Temperature Pool** dominates RandomT
- **RandomT** dominates Temperature Consensus
- **Best fixed T** dominates Temperature Consensus
