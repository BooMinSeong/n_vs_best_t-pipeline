# Per-Problem Regret Analysis

> **Methodology**: per-problem maj@N accuracy is Monte-Carlo estimated by sampling from the empirical per-temperature answer distribution (`distributions.json`, 6×256=1536 generations/problem-T), not by bootstrapping raw generations — same currency as the dataset-level `sim_v2` tables.

> **Caveat**: the per-problem oracle is max-over-T of noisy estimates, ~1pp **upward** biased. Absolute oracle gaps inherit this; strategy-vs-strategy comparisons do not (common shift cancels).

Snapshot tables use N=256. Per-combo detail (all N) in `reports/`.

## Oracle gap matrix (pp) @ N=256

Per-problem-oracle minus best-single-fixed-T accuracy. Larger = more to gain from per-problem temperature selection.

| model \ dataset | aime2023 | aime2024 | aime2025 | aime2026 | gsm8kfull | math500 | mathfull |
|---|---|---|---|---|---|---|---|
| Llama-3.2-3B | — | — | — | — | 2.52 | — | 5.17 |
| Phi-4-mini-instruct | 0.41 | 0.00 | 0.03 | 0.34 | 2.14 | 5.77 | — |
| Qwen2.5-3B | 0.09 | 2.45 | 0.17 | 0.08 | 2.51 | — | 4.42 |
| Qwen3-4B-Instruct-2507 | 0.00 | 0.64 | 0.98 | 0.11 | — | — | — |

## Deployable-strategy mean R2 regret @ N=256

Lower = closer to the best single fixed T.

| dataset / model | T=1.0 | RandomT | Temperature Pool | Temperature Consensus |
|---|---|---|---|---|
| aime2023 / Phi-4-mini-instruct | 0.0494 | 0.0305 | 0.0275 | 0.0257 |
| aime2023 / Qwen2.5-3B | 0.0014 | 0.0018 | 0.0024 | 0.0020 |
| aime2023 / Qwen3-4B-Instruct-2507 | 0.0083 | 0.0080 | 0.0028 | 0.0027 |
| aime2024 / Phi-4-mini-instruct | 0.0023 | 0.0056 | 0.0022 | 0.0022 |
| aime2024 / Qwen2.5-3B | 0.0472 | 0.0313 | 0.0277 | 0.0223 |
| aime2024 / Qwen3-4B-Instruct-2507 | 0.0091 | 0.0240 | 0.0145 | 0.0171 |
| aime2025 / Phi-4-mini-instruct | 0.0055 | 0.0044 | 0.0055 | 0.0055 |
| aime2025 / Qwen2.5-3B | 0.0521 | 0.0195 | 0.0017 | 0.0022 |
| aime2025 / Qwen3-4B-Instruct-2507 | 0.0689 | 0.0482 | 0.0630 | 0.0632 |
| aime2026 / Phi-4-mini-instruct | 0.0473 | 0.0289 | 0.0272 | 0.0326 |
| aime2026 / Qwen2.5-3B | 0.0645 | 0.0354 | 0.0284 | 0.0240 |
| aime2026 / Qwen3-4B-Instruct-2507 | 0.0008 | 0.0174 | 0.0153 | 0.0223 |
| gsm8kfull / Llama-3.2-3B | 0.0030 | 0.0350 | 0.0214 | 0.0169 |
| gsm8kfull / Phi-4-mini-instruct | 0.0001 | 0.0121 | 0.0063 | 0.0045 |
| gsm8kfull / Qwen2.5-3B | 0.0000 | 0.0088 | 0.0058 | 0.0050 |
| math500 / Phi-4-mini-instruct | 0.0367 | 0.0350 | 0.0119 | 0.0075 |
| mathfull / Llama-3.2-3B | 0.0206 | 0.0461 | 0.0218 | 0.0113 |
| mathfull / Qwen2.5-3B | 0.0086 | 0.0144 | 0.0054 | 0.0051 |

## Per-combo reports

- [Llama-3.2-3B / gsm8kfull](reports/Llama-3.2-3B__gsm8kfull.md) — oracle gap +2.52 pp
- [Llama-3.2-3B / mathfull](reports/Llama-3.2-3B__mathfull.md) — oracle gap +5.17 pp
- [Phi-4-mini-instruct / aime2023](reports/Phi-4-mini-instruct__aime2023.md) — oracle gap +0.41 pp
- [Phi-4-mini-instruct / aime2024](reports/Phi-4-mini-instruct__aime2024.md) — oracle gap +0.00 pp
- [Phi-4-mini-instruct / aime2025](reports/Phi-4-mini-instruct__aime2025.md) — oracle gap +0.03 pp
- [Phi-4-mini-instruct / aime2026](reports/Phi-4-mini-instruct__aime2026.md) — oracle gap +0.34 pp
- [Phi-4-mini-instruct / gsm8kfull](reports/Phi-4-mini-instruct__gsm8kfull.md) — oracle gap +2.14 pp
- [Phi-4-mini-instruct / math500](reports/Phi-4-mini-instruct__math500.md) — oracle gap +5.77 pp
- [Qwen2.5-3B / aime2023](reports/Qwen2.5-3B__aime2023.md) — oracle gap +0.09 pp
- [Qwen2.5-3B / aime2024](reports/Qwen2.5-3B__aime2024.md) — oracle gap +2.45 pp
- [Qwen2.5-3B / aime2025](reports/Qwen2.5-3B__aime2025.md) — oracle gap +0.17 pp
- [Qwen2.5-3B / aime2026](reports/Qwen2.5-3B__aime2026.md) — oracle gap +0.08 pp
- [Qwen2.5-3B / gsm8kfull](reports/Qwen2.5-3B__gsm8kfull.md) — oracle gap +2.51 pp
- [Qwen2.5-3B / mathfull](reports/Qwen2.5-3B__mathfull.md) — oracle gap +4.42 pp
- [Qwen3-4B-Instruct-2507 / aime2023](reports/Qwen3-4B-Instruct-2507__aime2023.md) — oracle gap +0.00 pp
- [Qwen3-4B-Instruct-2507 / aime2024](reports/Qwen3-4B-Instruct-2507__aime2024.md) — oracle gap +0.64 pp
- [Qwen3-4B-Instruct-2507 / aime2025](reports/Qwen3-4B-Instruct-2507__aime2025.md) — oracle gap +0.98 pp
- [Qwen3-4B-Instruct-2507 / aime2026](reports/Qwen3-4B-Instruct-2507__aime2026.md) — oracle gap +0.11 pp
