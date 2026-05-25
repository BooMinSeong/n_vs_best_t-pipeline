# Phi-4-mini-instruct / aime2025

[← index](../REPORT.md) · figures: `figs/Phi-4-mini-instruct__aime2025/`

Regret R2 = (accuracy of the best *single fixed* temperature, chosen on the whole dataset) − (accuracy of the strategy), per problem. R1 (vs the per-problem oracle) lives in the parquet/CSV outputs.

## Oracle gap vs N

Per-problem oracle headroom over the best single fixed T (caveat: per-problem oracle ~1pp upward biased).

| N | best fixed T | dataset oracle acc | per-problem oracle acc | gap (pp) |
|---|---|---|---|---|
| 1 | T0.2 | 0.0213 | 0.0256 | +0.43 |
| 2 | T0.3 | 0.0241 | 0.0291 | +0.50 |
| 4 | T0.2 | 0.0222 | 0.0276 | +0.54 |
| 8 | T0.3 | 0.0201 | 0.0245 | +0.44 |
| 16 | T0.2 | 0.0164 | 0.0195 | +0.30 |
| 32 | T0.2 | 0.0129 | 0.0157 | +0.28 |
| 64 | T0.3 | 0.0095 | 0.0115 | +0.20 |
| 128 | T0.3 | 0.0077 | 0.0087 | +0.09 |
| 256 | T0.3 | 0.0055 | 0.0058 | +0.03 |
| 512 | T0.3 | 0.0027 | 0.0027 | +0.00 |
| 1024 | T0.3 | 0.0008 | 0.0008 | +0.00 |
| 1536 | T0.3 | 0.0002 | 0.0002 | +0.00 |
| 2048 | T0.3 | 0.0001 | 0.0001 | +0.00 |

## R2 regret — mean by strategy × N

| N | T=0.1 | T=1.0 | RandomT | Temperature Pool | Temperature Consensus |
|---|---|---|---|---|---|
| 1 | 0.0032 | 0.0106 | 0.0066 | 0.0024 | 0.0024 |
| 2 | 0.0031 | 0.0113 | 0.0071 | 0.0017 | 0.0017 |
| 4 | 0.0037 | 0.0117 | 0.0076 | 0.0012 | 0.0012 |
| 8 | 0.0041 | 0.0136 | 0.0084 | 0.0046 | 0.0046 |
| 16 | 0.0043 | 0.0142 | 0.0078 | 0.0100 | 0.0120 |
| 32 | 0.0045 | 0.0125 | 0.0071 | 0.0106 | 0.0122 |
| 64 | 0.0044 | 0.0095 | 0.0059 | 0.0094 | 0.0094 |
| 128 | 0.0056 | 0.0077 | 0.0056 | 0.0077 | 0.0077 |
| 256 | 0.0051 | 0.0055 | 0.0044 | 0.0055 | 0.0055 |
| 512 | 0.0027 | 0.0027 | 0.0022 | 0.0027 | 0.0027 |
| 1024 | 0.0008 | 0.0008 | 0.0007 | 0.0008 | 0.0008 |
| 1536 | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0002 |
| 2048 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 |

## R2 regret — p95 (tail) by strategy × N

| N | T=0.1 | T=1.0 | RandomT | Temperature Pool | Temperature Consensus |
|---|---|---|---|---|---|
| 1 | 0.0273 | 0.0698 | 0.0393 | 0.0247 | 0.0247 |
| 2 | 0.0312 | 0.0789 | 0.0437 | 0.0169 | 0.0169 |
| 4 | 0.0349 | 0.0903 | 0.0508 | 0.0100 | 0.0100 |
| 8 | 0.0475 | 0.1129 | 0.0689 | 0.0413 | 0.0413 |
| 16 | 0.0391 | 0.0815 | 0.0484 | 0.0592 | 0.0720 |
| 32 | 0.0197 | 0.0404 | 0.0165 | 0.0280 | 0.0361 |
| 64 | 0.0030 | 0.0167 | 0.0031 | 0.0153 | 0.0142 |
| 128 | 0.0005 | 0.0036 | 0.0005 | 0.0036 | 0.0029 |
| 256 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 512 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1024 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1536 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2048 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## T\* distribution @ N=256 (26 problems)

| T* | n_problems | share |
|---|---|---|
| 0.1 | 25 | 96.2% |
| 0.3 | 1 | 3.8% |

## Paired comparison @ N=256 (Δ = A − B per-problem accuracy)

| A | B | Δ mean | win/tie/loss | Wilcoxon p | BH adj-p |
|---|---|---|---|---|---|
| T=1.0 | Temperature Pool | 0.0000 | 0/26/0 | 1 | 1 |
| T=1.0 | Temperature Consensus | -0.0000 | 0/26/0 | 0.32 | 0.4 |
| Temperature Pool | Temperature Consensus | -0.0000 | 0/26/0 | 0.32 | 0.4 |
| T=1.0 | Best fixed T | -0.0055 | 0/25/1 | 0.32 | 0.4 |
| Temperature Pool | Best fixed T | -0.0055 | 0/25/1 | 0.32 | 0.4 |

## Stochastic dominance @ N=256 (R2)

- **T=1.0** dominates Temperature Pool
- **T=1.0** dominates Temperature Consensus
- **Temperature Pool** dominates T=1.0
- **Temperature Pool** dominates Temperature Consensus
- **Temperature Consensus** dominates T=1.0
- **Temperature Consensus** dominates Temperature Pool
- **RandomT** dominates T=1.0
- **RandomT** dominates Temperature Pool
- **RandomT** dominates Temperature Consensus
- **Best fixed T** dominates T=1.0
- **Best fixed T** dominates Temperature Pool
- **Best fixed T** dominates Temperature Consensus
