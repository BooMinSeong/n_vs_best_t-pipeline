# Qwen3-4B-Instruct-2507 / aime2023

[← index](../REPORT.md) · figures: `figs/Qwen3-4B-Instruct-2507__aime2023/`

Regret R2 = (accuracy of the best *single fixed* temperature, chosen on the whole dataset) − (accuracy of the strategy), per problem. R1 (vs the per-problem oracle) lives in the parquet/CSV outputs.

## Oracle gap vs N

Per-problem oracle headroom over the best single fixed T (caveat: per-problem oracle ~1pp upward biased).

| N | best fixed T | dataset oracle acc | per-problem oracle acc | gap (pp) |
|---|---|---|---|---|
| 1 | T0.9 | 0.3584 | 0.3916 | +3.32 |
| 2 | T0.9 | 0.3746 | 0.4057 | +3.11 |
| 4 | T0.9 | 0.4417 | 0.4669 | +2.52 |
| 8 | T0.9 | 0.4871 | 0.5055 | +1.84 |
| 16 | T0.7 | 0.5072 | 0.5167 | +0.95 |
| 32 | T0.7 | 0.5153 | 0.5197 | +0.44 |
| 64 | T0.5 | 0.5206 | 0.5216 | +0.09 |
| 128 | T0.5 | 0.5232 | 0.5232 | +0.00 |
| 256 | T0.5 | 0.5238 | 0.5238 | +0.00 |
| 512 | T0.4 | 0.5238 | 0.5238 | +0.00 |
| 1024 | T0.2 | 0.5238 | 0.5238 | +0.00 |
| 1536 | T0.2 | 0.5238 | 0.5238 | +0.00 |
| 2048 | T0.2 | 0.5238 | 0.5238 | +0.00 |

## R2 regret — mean by strategy × N

| N | T=0.1 | T=1.0 | RandomT | Temperature Pool | Temperature Consensus |
|---|---|---|---|---|---|
| 1 | 0.0203 | 0.0070 | 0.0110 | 0.0176 | 0.0176 |
| 2 | 0.0194 | 0.0062 | 0.0099 | 0.0177 | 0.0177 |
| 4 | 0.0320 | 0.0078 | 0.0145 | 0.0270 | 0.0270 |
| 8 | 0.0393 | 0.0081 | 0.0155 | 0.0161 | 0.0161 |
| 16 | 0.0386 | 0.0086 | 0.0143 | 0.0119 | 0.0089 |
| 32 | 0.0365 | 0.0102 | 0.0112 | 0.0070 | 0.0113 |
| 64 | 0.0351 | 0.0121 | 0.0110 | 0.0076 | 0.0125 |
| 128 | 0.0332 | 0.0112 | 0.0094 | 0.0056 | 0.0056 |
| 256 | 0.0306 | 0.0083 | 0.0080 | 0.0028 | 0.0027 |
| 512 | 0.0310 | 0.0042 | 0.0069 | 0.0006 | 0.0011 |
| 1024 | 0.0338 | 0.0015 | 0.0070 | 0.0001 | 0.0002 |
| 1536 | 0.0357 | 0.0005 | 0.0075 | 0.0000 | 0.0001 |
| 2048 | 0.0366 | 0.0002 | 0.0071 | 0.0000 | 0.0000 |

## R2 regret — p95 (tail) by strategy × N

| N | T=0.1 | T=1.0 | RandomT | Temperature Pool | Temperature Consensus |
|---|---|---|---|---|---|
| 1 | 0.1300 | 0.0530 | 0.0754 | 0.1304 | 0.1304 |
| 2 | 0.1110 | 0.0404 | 0.0595 | 0.1142 | 0.1142 |
| 4 | 0.1535 | 0.0513 | 0.0657 | 0.1478 | 0.1478 |
| 8 | 0.2277 | 0.0492 | 0.0680 | 0.1071 | 0.1071 |
| 16 | 0.2853 | 0.0350 | 0.0588 | 0.0415 | 0.0650 |
| 32 | 0.2441 | 0.0164 | 0.0402 | 0.0137 | 0.1053 |
| 64 | 0.1754 | 0.0099 | 0.0211 | 0.0097 | 0.0195 |
| 128 | 0.0900 | 0.0003 | 0.0094 | 0.0001 | 0.0004 |
| 256 | 0.0250 | 0.0000 | 0.0026 | 0.0000 | 0.0000 |
| 512 | 0.0030 | 0.0000 | 0.0004 | 0.0000 | 0.0000 |
| 1024 | 0.0002 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1536 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2048 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## T\* distribution @ N=256 (21 problems)

| T* | n_problems | share |
|---|---|---|
| 0.1 | 19 | 90.5% |
| 0.3 | 1 | 4.8% |
| 0.5 | 1 | 4.8% |

## Paired comparison @ N=256 (Δ = A − B per-problem accuracy)

| A | B | Δ mean | win/tie/loss | Wilcoxon p | BH adj-p |
|---|---|---|---|---|---|
| T=1.0 | Temperature Pool | -0.0056 | 0/20/1 | 0.32 | 0.32 |
| T=1.0 | Temperature Consensus | -0.0056 | 0/20/1 | 0.32 | 0.32 |
| Temperature Pool | Temperature Consensus | -0.0001 | 0/21/0 | 0.32 | 0.32 |
| T=1.0 | Best fixed T | -0.0083 | 0/20/1 | 0.32 | 0.32 |
| Temperature Pool | Best fixed T | -0.0028 | 0/20/1 | 0.32 | 0.32 |

## Stochastic dominance @ N=256 (R2)

- **Temperature Pool** dominates T=1.0
- **Temperature Pool** dominates RandomT
- **Temperature Consensus** dominates T=1.0
- **Temperature Consensus** dominates Temperature Pool
- **Temperature Consensus** dominates RandomT
- **Best fixed T** dominates T=1.0
- **Best fixed T** dominates Temperature Pool
- **Best fixed T** dominates Temperature Consensus
- **Best fixed T** dominates RandomT
