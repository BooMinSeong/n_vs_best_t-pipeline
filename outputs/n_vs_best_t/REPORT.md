# N 대비 최적 Temperature 분석 — Report

샘플 예산 N(maj@N)이 주어졌을 때 어떤 단일 temperature T가 최적인지를 9개 (model × dataset_family) 조합에 대해 결정하고, 5개 baseline과 비교한 결과:

1. **T=1.0 fixed**, **T=0.1 fixed** — single-T MV baselines
2. **random_T** (uniform T choice per sim), **equal_mix** (round-robin N/12 alloc → pooled MV), **consensus_vote** (round-robin N/12 alloc → per-T MV winner → plurality of 12 winners)

**Method**: 모든 수치는 동일한 sim_v2 simulation 결과에서 산출 (`run_sim_v2_one.py`, 240 MC rep, multinomial-from-empirical, seed=42). mix baseline 5종(T=1.0, T=0.1, random_T, equal_mix, consensus_vote) 도 같은 simulation 에서 산출. AIME 4년치는 problem-count 가중 평균으로 합산.

**best-T 와 T\* 의 정의** (`merge_sim_v2.py:114-133`): 각 `(model, dataset, stratum, N)` 조합에 대해 독립적으로, 12개 single-T MV baseline (T=0.1, 0.2, …, 1.2) 의 sim accuracy 중 **argmax** 를 취한 값이 `best_t_mean`, 그 argmax 에 해당하는 T 값이 `t_star`. 즉 each cell 의 12개 후보 T 중 최고 성능 단일 T 를 사후적으로 선택한 것 — "어떤 T 가 best 인지 알았더라면 얼마나 정확했을지" 의 oracle ceiling.

**주요 특성:**
- *Oracle*: stratum, N, dataset 별로 t_star 가 다를 수 있으며, 실제 deployment 에서는 문제의 정답을 모르므로 이 T 를 선택할 수 없음. 모든 deployable baseline (T=1.0 fixed, equal_mix, consensus_vote 등) 이 좇아가는 **upper bound**.
- *Per-stratum × per-N*: L1 의 t_star 와 L5 의 t_star, N=16 의 t_star 와 N=256 의 t_star 가 모두 독립적으로 산출 (어려운 문제일수록 높은 T 가 유리한 경향).
- *MV (maj@N) 기준*: 각 T 의 정확도는 N개 sample 의 majority vote 가 정답일 확률 (sim 240 rep 평균).
- *Gap 의 의미*: `gap_vs_X = best_t_mean − acc_X` 는 "최적 T 를 알았더라면 baseline X 대비 얼마나 더 좋아졌을지" 의 oracle 이점 측정.

**문제 제외 기준** (`load_aligned`, `cross_t_mode/lb_baselines.py:49-88`): 다음 두 조건 중 하나라도 해당하는 문제는 sim 입력에서 drop됩니다.
1. 12개 temperature (T=0.1~1.2) 어디서도 답 추출이 한 번도 안 된 문제 (모든 generation 의 parsing 실패).
2. canonical 정답이 None 이거나, 정답이 모든 T × 모든 sample 분포에 한 번도 등장하지 않은 문제 (sim 의 P[correct]=0 → 자명히 못 맞춤).

본 보고서의 `level_n_problems` 와 모든 정답률은 위 기준으로 drop 한 뒤의 **loaded 집합** 기준입니다. 전체 raw dataset 기준 (제외된 문제는 정답률 0 으로 포함) 비교는 아래 **Coverage** 섹션 참고.

추가로, **stratum** 분류:
- L1~L5: baseline solve rate 기반 difficulty level (L1 easiest ~ L5 hardest, `(min, max]` 구간).
- Lr: difficulty level 부여 실패 (level=None) 인 문제 (canonical 정답은 loaded 단계에서 존재 확인, baseline solve rate 산출 실패한 케이스).
- overall: L1~L5 + Lr 가중평균 (loaded 전체).
- overall_md_compat: L1~L5 만 (Lr 제외). 옛 markdown 보고서 호환용.

**N grid**: {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048}. **T grid**: {0.1, 0.2, …, 1.2} (12 값).

## Coverage and inclusive-of-excluded accuracy @ N=256, overall stratum

아래 표는 raw dataset 의 전체 문제 수 (`n_raw`) 대비 sim 에 포함된 문제 수 (`n_loaded`) 와, 제외된 문제도 정답률 0으로 포함시킨 inclusive 정답률을 함께 보여줍니다. `incl_X = X × n_loaded / n_raw` 으로 산출.

| model | dataset | n_raw | n_loaded | excl | excl % | best (loaded) | best (incl) | Δ best | equal_mix (incl) | consensus_vote (incl) |
|---|---|---|---|---|---|---|---|---|---|---|
| Phi-4-mini-instruct | aime | 120 | 102 | 18 | 15.0% | 4.73 | 4.02 | -0.71pp | 3.11 | 3.08 |
| Qwen2.5-3B | aime | 120 | 105 | 15 | 12.5% | 9.78 | 8.56 | -1.22pp | 7.77 | 8.03 |
| Qwen3-4B-Instruct-2507 | aime | 120 | 93 | 27 | 22.5% | 48.72 | 37.76 | -10.96pp | 37.33 | 37.10 |
| Llama-3.2-3B | gsm8kfull | 1319 | 1318 | 1 | 0.1% | 85.14 | 85.08 | -0.06pp | 82.94 | 83.39 |
| Phi-4-mini-instruct | gsm8kfull | 1319 | 1319 | 0 | 0.0% | 91.93 | 91.93 | +0.00pp | 91.32 | 91.51 |
| Qwen2.5-3B | gsm8kfull | 1319 | 1318 | 1 | 0.1% | 90.07 | 90.00 | -0.07pp | 89.38 | 89.47 |
| Qwen2.5-3B | math1k | 1000 | 1000 | 0 | 0.0% | 75.15 | 75.15 | +0.00pp | 74.54 | 74.35 |
| Phi-4-mini-instruct | math500 | 500 | 494 | 6 | 1.2% | 67.69 | 66.88 | -0.81pp | 65.71 | 66.15 |
| Llama-3.2-3B | mathfull | 5000 | 4915 | 85 | 1.7% | 61.24 | 60.19 | -1.04pp | 58.08 | 59.13 |
| Qwen2.5-3B | mathfull | 5000 | 4965 | 35 | 0.7% | 76.07 | 75.54 | -0.53pp | 75.00 | 75.02 |
| Qwen3-4B-Instruct-2507 | mathfull | 5000 | 4862 | 138 | 2.8% | 88.99 | 86.53 | -2.46pp | 86.27 | 86.26 |

**해석**: `excl %` 가 큰 dataset 일수록 loaded vs inclusive 차이가 큽니다. AIME 계열은 dataset 자체가 작고 (30 problems/year × 4 = 120) 정답이 한 번도 나오지 않는 문제 비율이 높아 (~10–25%) inclusive accuracy 가 loaded 대비 크게 낮아집니다. math/gsm8k 계열은 제외율 ≲2% 로 차이가 작습니다.

## Per-stratum coverage — L1..L5 vs Lr @ N=256

`load_aligned` 의 drop 조건 (정답이 한 번도 안 나옴 / parsing 실패) 은 baseline solve rate 가 0% 인 문제와 정확히 일치하므로, drop 된 문제는 전부 **Lr** (level=None) 으로만 분류됩니다. **L1~L5 에는 drop 이 0건** (per-level inclusive = loaded 동일). 따라서 inclusive 효과는 Lr 에 집중.

**Lr stratum 의 inclusive vs loaded accuracy** (Lr 만 환산: `incl_Lr = loaded_acc_Lr × n_loaded_Lr / n_raw_Lr`)

| model | dataset | n_raw_Lr | n_loaded_Lr | excl Lr | excl % | best Lr (loaded) | best Lr (incl) | Δ best Lr | eq_mix Lr (incl) | cv Lr (incl) |
|---|---|---|---|---|---|---|---|---|---|---|
| Phi-4-mini-instruct | aime | 62 | 44 | 18 | 29.0% | 0.00 | 0.00 | +0.00pp | 0.00 | 0.00 |
| Qwen2.5-3B | aime | 57 | 42 | 15 | 26.3% | 0.00 | 0.00 | +0.00pp | 0.00 | 0.00 |
| Qwen3-4B-Instruct-2507 | aime | 37 | 10 | 27 | 73.0% | 0.00 | 0.00 | +0.00pp | 0.00 | 0.00 |
| Llama-3.2-3B | gsm8kfull | 61 | 60 | 1 | 1.6% | 6.56 | 6.46 | -0.11pp | 0.30 | 1.19 |
| Phi-4-mini-instruct | gsm8kfull | 31 | 31 | 0 | 0.0% | 10.09 | 10.09 | +0.00pp | 0.09 | 1.45 |
| Qwen2.5-3B | gsm8kfull | 29 | 28 | 1 | 3.4% | 7.35 | 7.10 | -0.25pp | 0.07 | 2.20 |
| Phi-4-mini-instruct | math500 | 23 | 17 | 6 | 26.1% | 5.88 | 4.35 | -1.53pp | 4.35 | 4.35 |
| Llama-3.2-3B | mathfull | 396 | 311 | 85 | 21.5% | 2.37 | 1.86 | -0.51pp | 0.12 | 0.32 |
| Qwen2.5-3B | mathfull | 229 | 194 | 35 | 15.3% | 4.05 | 3.43 | -0.62pp | 0.03 | 0.22 |
| Qwen3-4B-Instruct-2507 | mathfull | 234 | 96 | 138 | 59.0% | 3.86 | 1.58 | -2.28pp | 0.43 | 0.48 |

**관찰**: AIME 계열 Lr 의 excl 비율은 30–73% 수준이며 (Qwen3-4B/aime: raw_Lr=37 중 27개 drop = 73%) inclusive accuracy 가 큰 폭으로 감소. 이는 Lr 의 본질 — "any-T 에서 한 번도 못 푸는 unsolvable 문제" — 가 raw 단계에선 Lr 의 큰 비중을 차지하기 때문입니다.

**L1~L5 stratum 별 n_loaded** (inclusive=loaded, 추가 환산 불필요):

| model | dataset | L1 | L2 | L3 | L4 | L5 | Lr (loaded) |
|---|---|---|---|---|---|---|---|
| Llama-3.2-3B | gsm8kfull | 901 | 62 | 48 | 65 | 182 | 60 |
| Llama-3.2-3B | mathfull | 1689 | 375 | 315 | 476 | 1749 | 311 |
| Phi-4-mini-instruct | aime | 1 | 2 | — | — | 55 | 44 |
| Phi-4-mini-instruct | gsm8kfull | 1046 | 73 | 47 | 43 | 79 | 31 |
| Phi-4-mini-instruct | math500 | 215 | 43 | 30 | 51 | 138 | 17 |
| Qwen2.5-3B | aime | 3 | — | 3 | 2 | 55 | 42 |
| Qwen2.5-3B | gsm8kfull | 1018 | 79 | 41 | 47 | 105 | 28 |
| Qwen2.5-3B | math1k | 199 | 199 | 199 | 203 | 200 | — |
| Qwen2.5-3B | mathfull | 2715 | 384 | 293 | 361 | 1018 | 194 |
| Qwen3-4B-Instruct-2507 | aime | 18 | 9 | 11 | 9 | 36 | 10 |
| Qwen3-4B-Instruct-2507 | mathfull | 3844 | 208 | 153 | 161 | 400 | 96 |

## Headline numbers — gap of best-T vs each baseline @ N=256, overall stratum

| model | dataset | T* | best (pp) | Δ vs T=1.0 | Δ vs T=0.1 | Δ vs random_T | Δ vs equal_mix | Δ vs consensus_vote |
|---|---|---|---|---|---|---|---|---|
| Phi-4-mini-instruct | aime | 0.5 | 4.73 | +2.06pp | +1.66pp | +1.32pp | +1.07pp | +1.11pp |
| Qwen2.5-3B | aime | 0.2 | 9.78 | +3.49pp | +0.98pp | +1.47pp | +0.90pp | +0.60pp |
| Qwen3-4B-Instruct-2507 | aime | 0.7 | 48.72 | +0.46pp | +2.45pp | +0.85pp | +0.55pp | +0.85pp |
| Llama-3.2-3B | gsm8kfull | 0.9 | 85.14 | +0.29pp | +7.72pp | +3.49pp | +2.14pp | +1.69pp |
| Phi-4-mini-instruct | gsm8kfull | 0.8 | 91.93 | +0.01pp | +4.05pp | +1.21pp | +0.61pp | +0.42pp |
| Qwen2.5-3B | gsm8kfull | 1.0 | 90.07 | +0.00pp | +3.82pp | +0.90pp | +0.62pp | +0.53pp |
| Qwen2.5-3B | math1k | 0.6 | 75.15 | +3.03pp | +7.90pp | +3.23pp | +0.62pp | +0.80pp |
| Phi-4-mini-instruct | math500 | 0.5 | 67.69 | +3.71pp | +4.93pp | +3.54pp | +1.19pp | +0.74pp |
| Llama-3.2-3B | mathfull | 0.7 | 61.24 | +2.03pp | +6.25pp | +4.57pp | +2.15pp | +1.08pp |
| Qwen2.5-3B | mathfull | 0.7 | 76.07 | +0.85pp | +4.26pp | +1.45pp | +0.54pp | +0.52pp |
| Qwen3-4B-Instruct-2507 | mathfull | 0.9 | 88.99 | +0.26pp | +1.26pp | +0.41pp | +0.27pp | +0.28pp |

## Best-T vs baselines @ N=16 (overall) — small-budget regime

| model | dataset | T* | best (pp) | Δ vs T=1.0 | Δ vs T=0.1 | Δ vs random_T | Δ vs equal_mix |
|---|---|---|---|---|---|---|---|
| Phi-4-mini-instruct | aime | 0.4 | 5.00 | +2.10pp | +1.13pp | +1.20pp | +0.98pp |
| Qwen2.5-3B | aime | 0.3 | 8.99 | +2.82pp | +0.85pp | +1.55pp | +0.71pp |
| Qwen3-4B-Instruct-2507 | aime | 0.8 | 47.31 | +0.42pp | +2.24pp | +0.67pp | +0.50pp |
| Llama-3.2-3B | gsm8kfull | 0.9 | 83.17 | +0.27pp | +5.90pp | +2.72pp | +1.87pp |
| Phi-4-mini-instruct | gsm8kfull | 0.8 | 91.12 | +0.00pp | +3.41pp | +1.10pp | +0.49pp |
| Qwen2.5-3B | gsm8kfull | 0.9 | 88.98 | +0.11pp | +2.92pp | +0.68pp | +0.63pp |
| Qwen2.5-3B | math1k | 0.6 | 70.50 | +4.63pp | +5.77pp | +3.32pp | +0.93pp |
| Phi-4-mini-instruct | math500 | 0.6 | 65.66 | +3.76pp | +3.73pp | +3.48pp | +1.20pp |
| Llama-3.2-3B | mathfull | 0.7 | 57.98 | +2.54pp | +4.60pp | +3.91pp | +1.32pp |
| Qwen2.5-3B | mathfull | 0.6 | 74.16 | +1.85pp | +3.19pp | +1.63pp | +0.63pp |
| Qwen3-4B-Instruct-2507 | mathfull | 0.8 | 88.70 | +0.18pp | +1.19pp | +0.35pp | +0.26pp |

## Extended N — best-T saturation @ N ∈ {256, 512, 1024, 1536} (overall)

| model | dataset | T*@256 | best@256 | T*@512 | best@512 | T*@1024 | best@1024 | T*@1536 | best@1536 | Δ(1536−256) |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-3B | gsm8kfull | T0.9 | 85.14 | T0.9 | 85.26 | T0.9 | 85.37 | T0.9 | 85.39 | +0.25pp |
| Llama-3.2-3B | mathfull | T0.7 | 61.24 | T0.7 | 61.43 | T0.7 | 61.51 | T0.7 | 61.54 | +0.30pp |
| Phi-4-mini-instruct | aime | T0.5 | 4.73 | T0.4 | 4.66 | T0.5 | 4.53 | T0.5 | 4.56 | -0.17pp |
| Phi-4-mini-instruct | gsm8kfull | T0.8 | 91.93 | T0.8 | 91.98 | T0.8 | 91.99 | T0.8 | 92.01 | +0.08pp |
| Phi-4-mini-instruct | math500 | T0.5 | 67.69 | T0.4 | 67.82 | T0.5 | 68.03 | T0.5 | 68.09 | +0.40pp |
| Qwen2.5-3B | aime | T0.2 | 9.78 | T0.2 | 9.78 | T0.2 | 9.72 | T0.2 | 9.64 | -0.14pp |
| Qwen2.5-3B | gsm8kfull | T1.0 | 90.07 | T1.0 | 90.10 | T0.9 | 90.12 | T0.9 | 90.14 | +0.07pp |
| Qwen2.5-3B | math1k | T0.6 | 75.15 | T0.6 | 75.27 | T0.6 | 75.30 | T0.6 | 75.35 | +0.19pp |
| Qwen2.5-3B | mathfull | T0.7 | 76.07 | T0.6 | 76.13 | T0.6 | 76.17 | T0.6 | 76.18 | +0.11pp |
| Qwen3-4B-Instruct-2507 | aime | T0.7 | 48.72 | T0.7 | 48.60 | T0.7 | 48.61 | T0.7 | 48.45 | -0.26pp |
| Qwen3-4B-Instruct-2507 | mathfull | T0.9 | 88.99 | T0.9 | 89.00 | T0.9 | 89.03 | T0.9 | 89.04 | +0.05pp |

Observation: 대부분 조합에서 N=256 → N=1536은 best-T를 +0.5pp 미만으로만 올림 (saturation). 1536 budget을 균등 mix해도 equal_mix는 best-T에 0.5~1pp 이내로 수렴.

## Cross-model — T*(N) for shared datasets

Pivot: T*(N=256) — rows=dataset, cols=model

| dataset | Qwen2.5-3B | Qwen3-4B-Inst | Phi-4-mini | Llama-3.2-3B |
|---|---|---|---|---|
| math1k | T0.6 (75.2) | — | — | — |
| mathfull | T0.7 (76.1) | T0.9 (89.0) | — | T0.7 (61.2) |
| math500 | — | — | T0.5 (67.7) | — |
| gsm8kfull | T1.0 (90.1) | — | T0.8 (91.9) | T0.9 (85.1) |
| aime | T0.2 (9.8) | T0.7 (48.7) | T0.5 (4.7) | — |

## Cross-comparison figures

### N × T landscape (the central object)

4×5 grid of `acc(T, N)` heatmaps with T*(N) trajectory overlay.
Rows = model, columns = dataset. Empty cells = no data.

**Overall stratum**

![NxT acc grid overall](figs/cross/I_grid_acc_overall.png)

![NxT regret grid overall](figs/cross/I_grid_regret_overall.png)

**Per-difficulty (L1=easy → L5=hard)**

![NxT acc L1](figs/cross/I_grid_acc_L1.png)

![NxT regret L1](figs/cross/I_grid_regret_L1.png)

![NxT acc L2](figs/cross/I_grid_acc_L2.png)

![NxT regret L2](figs/cross/I_grid_regret_L2.png)

![NxT acc L3](figs/cross/I_grid_acc_L3.png)

![NxT regret L3](figs/cross/I_grid_regret_L3.png)

![NxT acc L4](figs/cross/I_grid_acc_L4.png)

![NxT regret L4](figs/cross/I_grid_regret_L4.png)

![NxT acc L5](figs/cross/I_grid_acc_L5.png)

![NxT regret L5](figs/cross/I_grid_regret_L5.png)

**T*(N) across all combos by difficulty stratum**

![K t-star all combo by stratum](figs/cross/K_t_star_all_combo_by_stratum.png)

### T* summary heatmaps and 5-curve comparisons

![T* heatmap N=256 overall](figs/cross/D_heatmap_overall_N256.png)

![T* heatmap N=64 overall](figs/cross/D_heatmap_overall_N64.png)

![T* heatmap N=256 L5](figs/cross/D_heatmap_L5_N256.png)

![5-curve grid overall](figs/cross/E_grid_overall.png)

![Win-margin scatter](figs/cross/F_win_margin_scatter.png)

## Per-combo detail

## Llama-3.2-3B / gsm8kfull

**Overall** (n_problems=1318) snapshot:
| N | T\* | best | T=1.0 | T=0.1 | random_T | equal_mix | Δ vs T1 | Δ vs T0.1 | Δ vs rand | Δ vs eq_mix |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.6 | 78.70 | 76.99 | 76.37 | 76.55 | 77.65 | +1.71pp | +2.34pp | +2.15pp | +1.05pp |
| 16 | 0.9 | 83.17 | 82.90 | 77.27 | 80.46 | 81.31 | +0.27pp | +5.90pp | +2.72pp | +1.87pp |
| 64 | 0.9 | 84.63 | 84.41 | 77.37 | 81.40 | 82.61 | +0.21pp | +7.26pp | +3.23pp | +2.02pp |
| 256 | 0.9 | 85.14 | 84.85 | 77.43 | 81.66 | 83.01 | +0.29pp | +7.72pp | +3.49pp | +2.14pp |

**T*(N=256) by stratum:**  
L1=T0.1 (acc 100.0, n=901) | L2=T0.1 (acc 100.0, n=62) | L3=T0.6 (acc 88.8, n=48) | L4=T0.9 (acc 85.3, n=65) | L5=T0.9 (acc 38.6, n=182)

**N×T landscape by difficulty** (6 panels: overall + L1..L5)

![NxT by difficulty](figs/Llama-3.2-3B__gsm8kfull/J_by_difficulty_grid.png)

![NxT by difficulty regret](figs/Llama-3.2-3B__gsm8kfull/J_by_difficulty_grid_regret.png)

**Overall stratum detail**

![acc(T,N) overall](figs/Llama-3.2-3B__gsm8kfull/G_acc_heatmap_overall.png)

![regret(T,N) overall](figs/Llama-3.2-3B__gsm8kfull/H_regret_heatmap_overall.png)

![T* trajectory by level](figs/Llama-3.2-3B__gsm8kfull/A_t_star_trajectory.png)

![5-curve overall](figs/Llama-3.2-3B__gsm8kfull/B_5curve_overall.png)

ZOOM (N≥128, y-range auto-fit to top baselines — shows consensus_vote / equal_mix separation):

![5-curve overall zoom128](figs/Llama-3.2-3B__gsm8kfull/B_5curve_overall_zoom128.png)

![gap overall](figs/Llama-3.2-3B__gsm8kfull/C_gap_overall.png)

**5-curve baseline comparison per difficulty (L1..L5, Lr)**

![B L1](figs/Llama-3.2-3B__gsm8kfull/B_5curve_L1.png)

![B L1 zoom128](figs/Llama-3.2-3B__gsm8kfull/B_5curve_L1_zoom128.png)

![C L1](figs/Llama-3.2-3B__gsm8kfull/C_gap_L1.png)

![B L2](figs/Llama-3.2-3B__gsm8kfull/B_5curve_L2.png)

![B L2 zoom128](figs/Llama-3.2-3B__gsm8kfull/B_5curve_L2_zoom128.png)

![C L2](figs/Llama-3.2-3B__gsm8kfull/C_gap_L2.png)

![B L3](figs/Llama-3.2-3B__gsm8kfull/B_5curve_L3.png)

![B L3 zoom128](figs/Llama-3.2-3B__gsm8kfull/B_5curve_L3_zoom128.png)

![C L3](figs/Llama-3.2-3B__gsm8kfull/C_gap_L3.png)

![B L4](figs/Llama-3.2-3B__gsm8kfull/B_5curve_L4.png)

![B L4 zoom128](figs/Llama-3.2-3B__gsm8kfull/B_5curve_L4_zoom128.png)

![C L4](figs/Llama-3.2-3B__gsm8kfull/C_gap_L4.png)

![B L5](figs/Llama-3.2-3B__gsm8kfull/B_5curve_L5.png)

![B L5 zoom128](figs/Llama-3.2-3B__gsm8kfull/B_5curve_L5_zoom128.png)

![C L5](figs/Llama-3.2-3B__gsm8kfull/C_gap_L5.png)

![B Lr](figs/Llama-3.2-3B__gsm8kfull/B_5curve_Lr.png)

![B Lr zoom128](figs/Llama-3.2-3B__gsm8kfull/B_5curve_Lr_zoom128.png)

![C Lr](figs/Llama-3.2-3B__gsm8kfull/C_gap_Lr.png)

## Llama-3.2-3B / mathfull

**Overall** (n_problems=4915) snapshot:
| N | T\* | best | T=1.0 | T=0.1 | random_T | equal_mix | Δ vs T1 | Δ vs T0.1 | Δ vs rand | Δ vs eq_mix |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.4 | 51.71 | 47.36 | 50.00 | 48.33 | 51.30 | +4.36pp | +1.71pp | +3.38pp | +0.41pp |
| 16 | 0.7 | 57.98 | 55.44 | 53.39 | 54.07 | 56.66 | +2.54pp | +4.60pp | +3.91pp | +1.32pp |
| 64 | 0.7 | 60.41 | 58.39 | 54.57 | 56.05 | 58.55 | +2.02pp | +5.84pp | +4.36pp | +1.86pp |
| 256 | 0.7 | 61.24 | 59.21 | 54.99 | 56.66 | 59.09 | +2.03pp | +6.25pp | +4.57pp | +2.15pp |

**T*(N=256) by stratum:**  
L1=T0.1 (acc 100.0, n=1689) | L2=T0.1 (acc 100.0, n=375) | L3=T0.1 (acc 89.7, n=315) | L4=T0.6 (acc 75.2, n=476) | L5=T0.8 (acc 19.4, n=1749)

**N×T landscape by difficulty** (6 panels: overall + L1..L5)

![NxT by difficulty](figs/Llama-3.2-3B__mathfull/J_by_difficulty_grid.png)

![NxT by difficulty regret](figs/Llama-3.2-3B__mathfull/J_by_difficulty_grid_regret.png)

**Overall stratum detail**

![acc(T,N) overall](figs/Llama-3.2-3B__mathfull/G_acc_heatmap_overall.png)

![regret(T,N) overall](figs/Llama-3.2-3B__mathfull/H_regret_heatmap_overall.png)

![T* trajectory by level](figs/Llama-3.2-3B__mathfull/A_t_star_trajectory.png)

![5-curve overall](figs/Llama-3.2-3B__mathfull/B_5curve_overall.png)

ZOOM (N≥128, y-range auto-fit to top baselines — shows consensus_vote / equal_mix separation):

![5-curve overall zoom128](figs/Llama-3.2-3B__mathfull/B_5curve_overall_zoom128.png)

![gap overall](figs/Llama-3.2-3B__mathfull/C_gap_overall.png)

**5-curve baseline comparison per difficulty (L1..L5, Lr)**

![B L1](figs/Llama-3.2-3B__mathfull/B_5curve_L1.png)

![B L1 zoom128](figs/Llama-3.2-3B__mathfull/B_5curve_L1_zoom128.png)

![C L1](figs/Llama-3.2-3B__mathfull/C_gap_L1.png)

![B L2](figs/Llama-3.2-3B__mathfull/B_5curve_L2.png)

![B L2 zoom128](figs/Llama-3.2-3B__mathfull/B_5curve_L2_zoom128.png)

![C L2](figs/Llama-3.2-3B__mathfull/C_gap_L2.png)

![B L3](figs/Llama-3.2-3B__mathfull/B_5curve_L3.png)

![B L3 zoom128](figs/Llama-3.2-3B__mathfull/B_5curve_L3_zoom128.png)

![C L3](figs/Llama-3.2-3B__mathfull/C_gap_L3.png)

![B L4](figs/Llama-3.2-3B__mathfull/B_5curve_L4.png)

![B L4 zoom128](figs/Llama-3.2-3B__mathfull/B_5curve_L4_zoom128.png)

![C L4](figs/Llama-3.2-3B__mathfull/C_gap_L4.png)

![B L5](figs/Llama-3.2-3B__mathfull/B_5curve_L5.png)

![B L5 zoom128](figs/Llama-3.2-3B__mathfull/B_5curve_L5_zoom128.png)

![C L5](figs/Llama-3.2-3B__mathfull/C_gap_L5.png)

![B Lr](figs/Llama-3.2-3B__mathfull/B_5curve_Lr.png)

![B Lr zoom128](figs/Llama-3.2-3B__mathfull/B_5curve_Lr_zoom128.png)

![C Lr](figs/Llama-3.2-3B__mathfull/C_gap_Lr.png)

## Phi-4-mini-instruct / aime

**Overall** (n_problems=102) snapshot:
| N | T\* | best | T=1.0 | T=0.1 | random_T | equal_mix | Δ vs T1 | Δ vs T0.1 | Δ vs rand | Δ vs eq_mix |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.4 | 4.66 | 2.97 | 4.08 | 3.56 | 4.43 | +1.69pp | +0.58pp | +1.10pp | +0.23pp |
| 16 | 0.4 | 5.00 | 2.90 | 3.86 | 3.80 | 4.02 | +2.10pp | +1.13pp | +1.20pp | +0.98pp |
| 64 | 0.4 | 4.89 | 2.75 | 3.41 | 3.62 | 3.67 | +2.14pp | +1.48pp | +1.27pp | +1.22pp |
| 256 | 0.5 | 4.73 | 2.67 | 3.07 | 3.41 | 3.66 | +2.06pp | +1.66pp | +1.32pp | +1.07pp |

**T*(N=256) by stratum:**  
L1=T0.8 (acc 100.0, n=1) | L2=T0.5 (acc 100.0, n=2) | L3=— | L4=— | L5=T0.5 (acc 3.3, n=55)

**N×T landscape by difficulty** (6 panels: overall + L1..L5)

![NxT by difficulty](figs/Phi-4-mini-instruct__aime/J_by_difficulty_grid.png)

![NxT by difficulty regret](figs/Phi-4-mini-instruct__aime/J_by_difficulty_grid_regret.png)

**Overall stratum detail**

![acc(T,N) overall](figs/Phi-4-mini-instruct__aime/G_acc_heatmap_overall.png)

![regret(T,N) overall](figs/Phi-4-mini-instruct__aime/H_regret_heatmap_overall.png)

![T* trajectory by level](figs/Phi-4-mini-instruct__aime/A_t_star_trajectory.png)

![5-curve overall](figs/Phi-4-mini-instruct__aime/B_5curve_overall.png)

ZOOM (N≥128, y-range auto-fit to top baselines — shows consensus_vote / equal_mix separation):

![5-curve overall zoom128](figs/Phi-4-mini-instruct__aime/B_5curve_overall_zoom128.png)

![gap overall](figs/Phi-4-mini-instruct__aime/C_gap_overall.png)

**5-curve baseline comparison per difficulty (L1..L5, Lr)**

![B L1](figs/Phi-4-mini-instruct__aime/B_5curve_L1.png)

![B L1 zoom128](figs/Phi-4-mini-instruct__aime/B_5curve_L1_zoom128.png)

![C L1](figs/Phi-4-mini-instruct__aime/C_gap_L1.png)

![B L2](figs/Phi-4-mini-instruct__aime/B_5curve_L2.png)

![B L2 zoom128](figs/Phi-4-mini-instruct__aime/B_5curve_L2_zoom128.png)

![C L2](figs/Phi-4-mini-instruct__aime/C_gap_L2.png)

![B L3](figs/Phi-4-mini-instruct__aime/B_5curve_L3.png)

![B L3 zoom128](figs/Phi-4-mini-instruct__aime/B_5curve_L3_zoom128.png)

![C L3](figs/Phi-4-mini-instruct__aime/C_gap_L3.png)

![B L4](figs/Phi-4-mini-instruct__aime/B_5curve_L4.png)

![B L4 zoom128](figs/Phi-4-mini-instruct__aime/B_5curve_L4_zoom128.png)

![C L4](figs/Phi-4-mini-instruct__aime/C_gap_L4.png)

![B L5](figs/Phi-4-mini-instruct__aime/B_5curve_L5.png)

![B L5 zoom128](figs/Phi-4-mini-instruct__aime/B_5curve_L5_zoom128.png)

![C L5](figs/Phi-4-mini-instruct__aime/C_gap_L5.png)

![B Lr](figs/Phi-4-mini-instruct__aime/B_5curve_Lr.png)

![B Lr zoom128](figs/Phi-4-mini-instruct__aime/B_5curve_Lr_zoom128.png)

![C Lr](figs/Phi-4-mini-instruct__aime/C_gap_Lr.png)

## Phi-4-mini-instruct / gsm8kfull

**Overall** (n_problems=1319) snapshot:
| N | T\* | best | T=1.0 | T=0.1 | random_T | equal_mix | Δ vs T1 | Δ vs T0.1 | Δ vs rand | Δ vs eq_mix |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.6 | 88.27 | 87.33 | 86.66 | 87.07 | 87.81 | +0.94pp | +1.60pp | +1.20pp | +0.46pp |
| 16 | 0.8 | 91.12 | 91.12 | 87.71 | 90.02 | 90.63 | +0.00pp | +3.41pp | +1.10pp | +0.49pp |
| 64 | 1.0 | 91.75 | 91.75 | 87.89 | 90.56 | 91.15 | +0.00pp | +3.86pp | +1.19pp | +0.60pp |
| 256 | 0.8 | 91.93 | 91.92 | 87.88 | 90.72 | 91.32 | +0.01pp | +4.05pp | +1.21pp | +0.61pp |

**T*(N=256) by stratum:**  
L1=T0.1 (acc 100.0, n=1046) | L2=T0.1 (acc 100.0, n=73) | L3=T0.8 (acc 89.1, n=47) | L4=T0.9 (acc 92.3, n=43) | L5=T1.1 (acc 34.7, n=79)

**N×T landscape by difficulty** (6 panels: overall + L1..L5)

![NxT by difficulty](figs/Phi-4-mini-instruct__gsm8kfull/J_by_difficulty_grid.png)

![NxT by difficulty regret](figs/Phi-4-mini-instruct__gsm8kfull/J_by_difficulty_grid_regret.png)

**Overall stratum detail**

![acc(T,N) overall](figs/Phi-4-mini-instruct__gsm8kfull/G_acc_heatmap_overall.png)

![regret(T,N) overall](figs/Phi-4-mini-instruct__gsm8kfull/H_regret_heatmap_overall.png)

![T* trajectory by level](figs/Phi-4-mini-instruct__gsm8kfull/A_t_star_trajectory.png)

![5-curve overall](figs/Phi-4-mini-instruct__gsm8kfull/B_5curve_overall.png)

ZOOM (N≥128, y-range auto-fit to top baselines — shows consensus_vote / equal_mix separation):

![5-curve overall zoom128](figs/Phi-4-mini-instruct__gsm8kfull/B_5curve_overall_zoom128.png)

![gap overall](figs/Phi-4-mini-instruct__gsm8kfull/C_gap_overall.png)

**5-curve baseline comparison per difficulty (L1..L5, Lr)**

![B L1](figs/Phi-4-mini-instruct__gsm8kfull/B_5curve_L1.png)

![B L1 zoom128](figs/Phi-4-mini-instruct__gsm8kfull/B_5curve_L1_zoom128.png)

![C L1](figs/Phi-4-mini-instruct__gsm8kfull/C_gap_L1.png)

![B L2](figs/Phi-4-mini-instruct__gsm8kfull/B_5curve_L2.png)

![B L2 zoom128](figs/Phi-4-mini-instruct__gsm8kfull/B_5curve_L2_zoom128.png)

![C L2](figs/Phi-4-mini-instruct__gsm8kfull/C_gap_L2.png)

![B L3](figs/Phi-4-mini-instruct__gsm8kfull/B_5curve_L3.png)

![B L3 zoom128](figs/Phi-4-mini-instruct__gsm8kfull/B_5curve_L3_zoom128.png)

![C L3](figs/Phi-4-mini-instruct__gsm8kfull/C_gap_L3.png)

![B L4](figs/Phi-4-mini-instruct__gsm8kfull/B_5curve_L4.png)

![B L4 zoom128](figs/Phi-4-mini-instruct__gsm8kfull/B_5curve_L4_zoom128.png)

![C L4](figs/Phi-4-mini-instruct__gsm8kfull/C_gap_L4.png)

![B L5](figs/Phi-4-mini-instruct__gsm8kfull/B_5curve_L5.png)

![B L5 zoom128](figs/Phi-4-mini-instruct__gsm8kfull/B_5curve_L5_zoom128.png)

![C L5](figs/Phi-4-mini-instruct__gsm8kfull/C_gap_L5.png)

![B Lr](figs/Phi-4-mini-instruct__gsm8kfull/B_5curve_Lr.png)

![B Lr zoom128](figs/Phi-4-mini-instruct__gsm8kfull/B_5curve_Lr_zoom128.png)

![C Lr](figs/Phi-4-mini-instruct__gsm8kfull/C_gap_Lr.png)

## Phi-4-mini-instruct / math500

**Overall** (n_problems=494) snapshot:
| N | T\* | best | T=1.0 | T=0.1 | random_T | equal_mix | Δ vs T1 | Δ vs T0.1 | Δ vs rand | Δ vs eq_mix |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.4 | 60.65 | 54.46 | 59.30 | 56.83 | 60.03 | +6.19pp | +1.35pp | +3.82pp | +0.61pp |
| 16 | 0.6 | 65.66 | 61.90 | 61.94 | 62.18 | 64.46 | +3.76pp | +3.73pp | +3.48pp | +1.20pp |
| 64 | 0.6 | 67.25 | 63.82 | 62.80 | 63.77 | 65.96 | +3.43pp | +4.45pp | +3.48pp | +1.29pp |
| 256 | 0.5 | 67.69 | 63.98 | 62.76 | 64.16 | 66.50 | +3.71pp | +4.93pp | +3.54pp | +1.19pp |

**T*(N=256) by stratum:**  
L1=T0.1 (acc 100.0, n=215) | L2=T0.1 (acc 100.0, n=43) | L3=T0.4 (acc 98.0, n=30) | L4=T0.8 (acc 59.3, n=51) | L5=T0.9 (acc 17.2, n=138)

**N×T landscape by difficulty** (6 panels: overall + L1..L5)

![NxT by difficulty](figs/Phi-4-mini-instruct__math500/J_by_difficulty_grid.png)

![NxT by difficulty regret](figs/Phi-4-mini-instruct__math500/J_by_difficulty_grid_regret.png)

**Overall stratum detail**

![acc(T,N) overall](figs/Phi-4-mini-instruct__math500/G_acc_heatmap_overall.png)

![regret(T,N) overall](figs/Phi-4-mini-instruct__math500/H_regret_heatmap_overall.png)

![T* trajectory by level](figs/Phi-4-mini-instruct__math500/A_t_star_trajectory.png)

![5-curve overall](figs/Phi-4-mini-instruct__math500/B_5curve_overall.png)

ZOOM (N≥128, y-range auto-fit to top baselines — shows consensus_vote / equal_mix separation):

![5-curve overall zoom128](figs/Phi-4-mini-instruct__math500/B_5curve_overall_zoom128.png)

![gap overall](figs/Phi-4-mini-instruct__math500/C_gap_overall.png)

**5-curve baseline comparison per difficulty (L1..L5, Lr)**

![B L1](figs/Phi-4-mini-instruct__math500/B_5curve_L1.png)

![B L1 zoom128](figs/Phi-4-mini-instruct__math500/B_5curve_L1_zoom128.png)

![C L1](figs/Phi-4-mini-instruct__math500/C_gap_L1.png)

![B L2](figs/Phi-4-mini-instruct__math500/B_5curve_L2.png)

![B L2 zoom128](figs/Phi-4-mini-instruct__math500/B_5curve_L2_zoom128.png)

![C L2](figs/Phi-4-mini-instruct__math500/C_gap_L2.png)

![B L3](figs/Phi-4-mini-instruct__math500/B_5curve_L3.png)

![B L3 zoom128](figs/Phi-4-mini-instruct__math500/B_5curve_L3_zoom128.png)

![C L3](figs/Phi-4-mini-instruct__math500/C_gap_L3.png)

![B L4](figs/Phi-4-mini-instruct__math500/B_5curve_L4.png)

![B L4 zoom128](figs/Phi-4-mini-instruct__math500/B_5curve_L4_zoom128.png)

![C L4](figs/Phi-4-mini-instruct__math500/C_gap_L4.png)

![B L5](figs/Phi-4-mini-instruct__math500/B_5curve_L5.png)

![B L5 zoom128](figs/Phi-4-mini-instruct__math500/B_5curve_L5_zoom128.png)

![C L5](figs/Phi-4-mini-instruct__math500/C_gap_L5.png)

![B Lr](figs/Phi-4-mini-instruct__math500/B_5curve_Lr.png)

![B Lr zoom128](figs/Phi-4-mini-instruct__math500/B_5curve_Lr_zoom128.png)

![C Lr](figs/Phi-4-mini-instruct__math500/C_gap_Lr.png)

## Qwen2.5-3B / aime

**Overall** (n_problems=105) snapshot:
| N | T\* | best | T=1.0 | T=0.1 | random_T | equal_mix | Δ vs T1 | Δ vs T0.1 | Δ vs rand | Δ vs eq_mix |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.2 | 7.19 | 5.04 | 6.76 | 5.94 | 7.08 | +2.15pp | +0.42pp | +1.24pp | +0.11pp |
| 16 | 0.3 | 8.99 | 6.17 | 8.14 | 7.45 | 8.29 | +2.82pp | +0.85pp | +1.55pp | +0.71pp |
| 64 | 0.3 | 9.86 | 6.45 | 8.59 | 8.16 | 9.11 | +3.41pp | +1.26pp | +1.70pp | +0.74pp |
| 256 | 0.2 | 9.78 | 6.30 | 8.80 | 8.32 | 8.88 | +3.49pp | +0.98pp | +1.47pp | +0.90pp |

**T*(N=256) by stratum:**  
L1=T1.1 (acc 100.0, n=3) | L2=— | L3=T0.3 (acc 100.0, n=3) | L4=T0.6 (acc 100.0, n=2) | L5=T0.2 (acc 4.1, n=55)

**N×T landscape by difficulty** (6 panels: overall + L1..L5)

![NxT by difficulty](figs/Qwen2.5-3B__aime/J_by_difficulty_grid.png)

![NxT by difficulty regret](figs/Qwen2.5-3B__aime/J_by_difficulty_grid_regret.png)

**Overall stratum detail**

![acc(T,N) overall](figs/Qwen2.5-3B__aime/G_acc_heatmap_overall.png)

![regret(T,N) overall](figs/Qwen2.5-3B__aime/H_regret_heatmap_overall.png)

![T* trajectory by level](figs/Qwen2.5-3B__aime/A_t_star_trajectory.png)

![5-curve overall](figs/Qwen2.5-3B__aime/B_5curve_overall.png)

ZOOM (N≥128, y-range auto-fit to top baselines — shows consensus_vote / equal_mix separation):

![5-curve overall zoom128](figs/Qwen2.5-3B__aime/B_5curve_overall_zoom128.png)

![gap overall](figs/Qwen2.5-3B__aime/C_gap_overall.png)

**5-curve baseline comparison per difficulty (L1..L5, Lr)**

![B L1](figs/Qwen2.5-3B__aime/B_5curve_L1.png)

![B L1 zoom128](figs/Qwen2.5-3B__aime/B_5curve_L1_zoom128.png)

![C L1](figs/Qwen2.5-3B__aime/C_gap_L1.png)

![B L2](figs/Qwen2.5-3B__aime/B_5curve_L2.png)

![B L2 zoom128](figs/Qwen2.5-3B__aime/B_5curve_L2_zoom128.png)

![C L2](figs/Qwen2.5-3B__aime/C_gap_L2.png)

![B L3](figs/Qwen2.5-3B__aime/B_5curve_L3.png)

![B L3 zoom128](figs/Qwen2.5-3B__aime/B_5curve_L3_zoom128.png)

![C L3](figs/Qwen2.5-3B__aime/C_gap_L3.png)

![B L4](figs/Qwen2.5-3B__aime/B_5curve_L4.png)

![B L4 zoom128](figs/Qwen2.5-3B__aime/B_5curve_L4_zoom128.png)

![C L4](figs/Qwen2.5-3B__aime/C_gap_L4.png)

![B L5](figs/Qwen2.5-3B__aime/B_5curve_L5.png)

![B L5 zoom128](figs/Qwen2.5-3B__aime/B_5curve_L5_zoom128.png)

![C L5](figs/Qwen2.5-3B__aime/C_gap_L5.png)

![B Lr](figs/Qwen2.5-3B__aime/B_5curve_Lr.png)

![B Lr zoom128](figs/Qwen2.5-3B__aime/B_5curve_Lr_zoom128.png)

![C Lr](figs/Qwen2.5-3B__aime/C_gap_Lr.png)

## Qwen2.5-3B / gsm8kfull

**Overall** (n_problems=1318) snapshot:
| N | T\* | best | T=1.0 | T=0.1 | random_T | equal_mix | Δ vs T1 | Δ vs T0.1 | Δ vs rand | Δ vs eq_mix |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.5 | 86.08 | 84.68 | 85.11 | 85.15 | 85.76 | +1.40pp | +0.97pp | +0.93pp | +0.32pp |
| 16 | 0.9 | 88.98 | 88.87 | 86.06 | 88.30 | 88.35 | +0.11pp | +2.92pp | +0.68pp | +0.63pp |
| 64 | 1.0 | 89.84 | 89.84 | 86.27 | 88.99 | 89.23 | +0.00pp | +3.57pp | +0.84pp | +0.61pp |
| 256 | 1.0 | 90.07 | 90.07 | 86.24 | 89.17 | 89.45 | +0.00pp | +3.82pp | +0.90pp | +0.62pp |

**T*(N=256) by stratum:**  
L1=T0.1 (acc 100.0, n=1018) | L2=T0.1 (acc 100.0, n=79) | L3=T0.6 (acc 92.8, n=41) | L4=T0.9 (acc 73.5, n=47) | L5=T1.2 (acc 34.6, n=105)

**N×T landscape by difficulty** (6 panels: overall + L1..L5)

![NxT by difficulty](figs/Qwen2.5-3B__gsm8kfull/J_by_difficulty_grid.png)

![NxT by difficulty regret](figs/Qwen2.5-3B__gsm8kfull/J_by_difficulty_grid_regret.png)

**Overall stratum detail**

![acc(T,N) overall](figs/Qwen2.5-3B__gsm8kfull/G_acc_heatmap_overall.png)

![regret(T,N) overall](figs/Qwen2.5-3B__gsm8kfull/H_regret_heatmap_overall.png)

![T* trajectory by level](figs/Qwen2.5-3B__gsm8kfull/A_t_star_trajectory.png)

![5-curve overall](figs/Qwen2.5-3B__gsm8kfull/B_5curve_overall.png)

ZOOM (N≥128, y-range auto-fit to top baselines — shows consensus_vote / equal_mix separation):

![5-curve overall zoom128](figs/Qwen2.5-3B__gsm8kfull/B_5curve_overall_zoom128.png)

![gap overall](figs/Qwen2.5-3B__gsm8kfull/C_gap_overall.png)

**5-curve baseline comparison per difficulty (L1..L5, Lr)**

![B L1](figs/Qwen2.5-3B__gsm8kfull/B_5curve_L1.png)

![B L1 zoom128](figs/Qwen2.5-3B__gsm8kfull/B_5curve_L1_zoom128.png)

![C L1](figs/Qwen2.5-3B__gsm8kfull/C_gap_L1.png)

![B L2](figs/Qwen2.5-3B__gsm8kfull/B_5curve_L2.png)

![B L2 zoom128](figs/Qwen2.5-3B__gsm8kfull/B_5curve_L2_zoom128.png)

![C L2](figs/Qwen2.5-3B__gsm8kfull/C_gap_L2.png)

![B L3](figs/Qwen2.5-3B__gsm8kfull/B_5curve_L3.png)

![B L3 zoom128](figs/Qwen2.5-3B__gsm8kfull/B_5curve_L3_zoom128.png)

![C L3](figs/Qwen2.5-3B__gsm8kfull/C_gap_L3.png)

![B L4](figs/Qwen2.5-3B__gsm8kfull/B_5curve_L4.png)

![B L4 zoom128](figs/Qwen2.5-3B__gsm8kfull/B_5curve_L4_zoom128.png)

![C L4](figs/Qwen2.5-3B__gsm8kfull/C_gap_L4.png)

![B L5](figs/Qwen2.5-3B__gsm8kfull/B_5curve_L5.png)

![B L5 zoom128](figs/Qwen2.5-3B__gsm8kfull/B_5curve_L5_zoom128.png)

![C L5](figs/Qwen2.5-3B__gsm8kfull/C_gap_L5.png)

![B Lr](figs/Qwen2.5-3B__gsm8kfull/B_5curve_Lr.png)

![B Lr zoom128](figs/Qwen2.5-3B__gsm8kfull/B_5curve_Lr_zoom128.png)

![C Lr](figs/Qwen2.5-3B__gsm8kfull/C_gap_Lr.png)

## Qwen2.5-3B / math1k

**Overall** (n_problems=1000) snapshot:
| N | T\* | best | T=1.0 | T=0.1 | random_T | equal_mix | Δ vs T1 | Δ vs T0.1 | Δ vs rand | Δ vs eq_mix |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.4 | 60.39 | 53.79 | 57.43 | 56.59 | 60.00 | +6.60pp | +2.97pp | +3.80pp | +0.40pp |
| 16 | 0.6 | 70.50 | 65.87 | 64.73 | 67.18 | 69.58 | +4.63pp | +5.77pp | +3.32pp | +0.93pp |
| 64 | 0.6 | 74.18 | 70.50 | 66.87 | 70.95 | 73.47 | +3.68pp | +7.32pp | +3.23pp | +0.72pp |
| 256 | 0.6 | 75.15 | 72.12 | 67.25 | 71.92 | 74.54 | +3.03pp | +7.90pp | +3.23pp | +0.62pp |

**T*(N=256) by stratum:**  
L1=T0.2 (acc 100.0, n=199) | L2=T0.1 (acc 100.0, n=199) | L3=T0.2 (acc 93.0, n=199) | L4=T0.6 (acc 72.9, n=203) | L5=T1.0 (acc 19.6, n=200)

**N×T landscape by difficulty** (6 panels: overall + L1..L5)

![NxT by difficulty](figs/Qwen2.5-3B__math1k/J_by_difficulty_grid.png)

![NxT by difficulty regret](figs/Qwen2.5-3B__math1k/J_by_difficulty_grid_regret.png)

**Overall stratum detail**

![acc(T,N) overall](figs/Qwen2.5-3B__math1k/G_acc_heatmap_overall.png)

![regret(T,N) overall](figs/Qwen2.5-3B__math1k/H_regret_heatmap_overall.png)

![T* trajectory by level](figs/Qwen2.5-3B__math1k/A_t_star_trajectory.png)

![5-curve overall](figs/Qwen2.5-3B__math1k/B_5curve_overall.png)

ZOOM (N≥128, y-range auto-fit to top baselines — shows consensus_vote / equal_mix separation):

![5-curve overall zoom128](figs/Qwen2.5-3B__math1k/B_5curve_overall_zoom128.png)

![gap overall](figs/Qwen2.5-3B__math1k/C_gap_overall.png)

**5-curve baseline comparison per difficulty (L1..L5, Lr)**

![B L1](figs/Qwen2.5-3B__math1k/B_5curve_L1.png)

![B L1 zoom128](figs/Qwen2.5-3B__math1k/B_5curve_L1_zoom128.png)

![C L1](figs/Qwen2.5-3B__math1k/C_gap_L1.png)

![B L2](figs/Qwen2.5-3B__math1k/B_5curve_L2.png)

![B L2 zoom128](figs/Qwen2.5-3B__math1k/B_5curve_L2_zoom128.png)

![C L2](figs/Qwen2.5-3B__math1k/C_gap_L2.png)

![B L3](figs/Qwen2.5-3B__math1k/B_5curve_L3.png)

![B L3 zoom128](figs/Qwen2.5-3B__math1k/B_5curve_L3_zoom128.png)

![C L3](figs/Qwen2.5-3B__math1k/C_gap_L3.png)

![B L4](figs/Qwen2.5-3B__math1k/B_5curve_L4.png)

![B L4 zoom128](figs/Qwen2.5-3B__math1k/B_5curve_L4_zoom128.png)

![C L4](figs/Qwen2.5-3B__math1k/C_gap_L4.png)

![B L5](figs/Qwen2.5-3B__math1k/B_5curve_L5.png)

![B L5 zoom128](figs/Qwen2.5-3B__math1k/B_5curve_L5_zoom128.png)

![C L5](figs/Qwen2.5-3B__math1k/C_gap_L5.png)

![B Lr](figs/Qwen2.5-3B__math1k/B_5curve_Lr.png)

![B Lr zoom128](figs/Qwen2.5-3B__math1k/B_5curve_Lr_zoom128.png)

![C Lr](figs/Qwen2.5-3B__math1k/C_gap_Lr.png)

## Qwen2.5-3B / mathfull

**Overall** (n_problems=4965) snapshot:
| N | T\* | best | T=1.0 | T=0.1 | random_T | equal_mix | Δ vs T1 | Δ vs T0.1 | Δ vs rand | Δ vs eq_mix |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.4 | 69.67 | 65.62 | 68.35 | 67.34 | 69.42 | +4.05pp | +1.32pp | +2.33pp | +0.25pp |
| 16 | 0.6 | 74.16 | 72.31 | 70.98 | 72.53 | 73.53 | +1.85pp | +3.19pp | +1.63pp | +0.63pp |
| 64 | 0.7 | 75.67 | 74.48 | 71.67 | 74.16 | 75.10 | +1.19pp | +4.00pp | +1.52pp | +0.57pp |
| 256 | 0.7 | 76.07 | 75.22 | 71.81 | 74.62 | 75.53 | +0.85pp | +4.26pp | +1.45pp | +0.54pp |

**T*(N=256) by stratum:**  
L1=T0.1 (acc 100.0, n=2715) | L2=T0.1 (acc 100.0, n=384) | L3=T0.2 (acc 92.6, n=293) | L4=T0.6 (acc 73.6, n=361) | L5=T0.9 (acc 18.2, n=1018)

**N×T landscape by difficulty** (6 panels: overall + L1..L5)

![NxT by difficulty](figs/Qwen2.5-3B__mathfull/J_by_difficulty_grid.png)

![NxT by difficulty regret](figs/Qwen2.5-3B__mathfull/J_by_difficulty_grid_regret.png)

**Overall stratum detail**

![acc(T,N) overall](figs/Qwen2.5-3B__mathfull/G_acc_heatmap_overall.png)

![regret(T,N) overall](figs/Qwen2.5-3B__mathfull/H_regret_heatmap_overall.png)

![T* trajectory by level](figs/Qwen2.5-3B__mathfull/A_t_star_trajectory.png)

![5-curve overall](figs/Qwen2.5-3B__mathfull/B_5curve_overall.png)

ZOOM (N≥128, y-range auto-fit to top baselines — shows consensus_vote / equal_mix separation):

![5-curve overall zoom128](figs/Qwen2.5-3B__mathfull/B_5curve_overall_zoom128.png)

![gap overall](figs/Qwen2.5-3B__mathfull/C_gap_overall.png)

**5-curve baseline comparison per difficulty (L1..L5, Lr)**

![B L1](figs/Qwen2.5-3B__mathfull/B_5curve_L1.png)

![B L1 zoom128](figs/Qwen2.5-3B__mathfull/B_5curve_L1_zoom128.png)

![C L1](figs/Qwen2.5-3B__mathfull/C_gap_L1.png)

![B L2](figs/Qwen2.5-3B__mathfull/B_5curve_L2.png)

![B L2 zoom128](figs/Qwen2.5-3B__mathfull/B_5curve_L2_zoom128.png)

![C L2](figs/Qwen2.5-3B__mathfull/C_gap_L2.png)

![B L3](figs/Qwen2.5-3B__mathfull/B_5curve_L3.png)

![B L3 zoom128](figs/Qwen2.5-3B__mathfull/B_5curve_L3_zoom128.png)

![C L3](figs/Qwen2.5-3B__mathfull/C_gap_L3.png)

![B L4](figs/Qwen2.5-3B__mathfull/B_5curve_L4.png)

![B L4 zoom128](figs/Qwen2.5-3B__mathfull/B_5curve_L4_zoom128.png)

![C L4](figs/Qwen2.5-3B__mathfull/C_gap_L4.png)

![B L5](figs/Qwen2.5-3B__mathfull/B_5curve_L5.png)

![B L5 zoom128](figs/Qwen2.5-3B__mathfull/B_5curve_L5_zoom128.png)

![C L5](figs/Qwen2.5-3B__mathfull/C_gap_L5.png)

![B Lr](figs/Qwen2.5-3B__mathfull/B_5curve_Lr.png)

![B Lr zoom128](figs/Qwen2.5-3B__mathfull/B_5curve_Lr_zoom128.png)

![C Lr](figs/Qwen2.5-3B__mathfull/C_gap_Lr.png)

## Qwen3-4B-Instruct-2507 / aime

**Overall** (n_problems=93) snapshot:
| N | T\* | best | T=1.0 | T=0.1 | random_T | equal_mix | Δ vs T1 | Δ vs T0.1 | Δ vs rand | Δ vs eq_mix |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.7 | 42.39 | 41.98 | 41.14 | 41.79 | 41.27 | +0.41pp | +1.24pp | +0.60pp | +1.12pp |
| 16 | 0.8 | 47.31 | 46.89 | 45.07 | 46.64 | 46.80 | +0.42pp | +2.24pp | +0.67pp | +0.50pp |
| 64 | 0.7 | 48.54 | 48.06 | 46.14 | 47.80 | 48.00 | +0.48pp | +2.40pp | +0.74pp | +0.54pp |
| 256 | 0.7 | 48.72 | 48.26 | 46.27 | 47.87 | 48.17 | +0.46pp | +2.45pp | +0.85pp | +0.55pp |

**T*(N=256) by stratum:**  
L1=T0.3 (acc 100.0, n=18) | L2=T1.0 (acc 100.0, n=9) | L3=T1.2 (acc 100.0, n=11) | L4=T0.7 (acc 67.8, n=9) | L5=T0.3 (acc 5.2, n=36)

**N×T landscape by difficulty** (6 panels: overall + L1..L5)

![NxT by difficulty](figs/Qwen3-4B-Instruct-2507__aime/J_by_difficulty_grid.png)

![NxT by difficulty regret](figs/Qwen3-4B-Instruct-2507__aime/J_by_difficulty_grid_regret.png)

**Overall stratum detail**

![acc(T,N) overall](figs/Qwen3-4B-Instruct-2507__aime/G_acc_heatmap_overall.png)

![regret(T,N) overall](figs/Qwen3-4B-Instruct-2507__aime/H_regret_heatmap_overall.png)

![T* trajectory by level](figs/Qwen3-4B-Instruct-2507__aime/A_t_star_trajectory.png)

![5-curve overall](figs/Qwen3-4B-Instruct-2507__aime/B_5curve_overall.png)

ZOOM (N≥128, y-range auto-fit to top baselines — shows consensus_vote / equal_mix separation):

![5-curve overall zoom128](figs/Qwen3-4B-Instruct-2507__aime/B_5curve_overall_zoom128.png)

![gap overall](figs/Qwen3-4B-Instruct-2507__aime/C_gap_overall.png)

**5-curve baseline comparison per difficulty (L1..L5, Lr)**

![B L1](figs/Qwen3-4B-Instruct-2507__aime/B_5curve_L1.png)

![B L1 zoom128](figs/Qwen3-4B-Instruct-2507__aime/B_5curve_L1_zoom128.png)

![C L1](figs/Qwen3-4B-Instruct-2507__aime/C_gap_L1.png)

![B L2](figs/Qwen3-4B-Instruct-2507__aime/B_5curve_L2.png)

![B L2 zoom128](figs/Qwen3-4B-Instruct-2507__aime/B_5curve_L2_zoom128.png)

![C L2](figs/Qwen3-4B-Instruct-2507__aime/C_gap_L2.png)

![B L3](figs/Qwen3-4B-Instruct-2507__aime/B_5curve_L3.png)

![B L3 zoom128](figs/Qwen3-4B-Instruct-2507__aime/B_5curve_L3_zoom128.png)

![C L3](figs/Qwen3-4B-Instruct-2507__aime/C_gap_L3.png)

![B L4](figs/Qwen3-4B-Instruct-2507__aime/B_5curve_L4.png)

![B L4 zoom128](figs/Qwen3-4B-Instruct-2507__aime/B_5curve_L4_zoom128.png)

![C L4](figs/Qwen3-4B-Instruct-2507__aime/C_gap_L4.png)

![B L5](figs/Qwen3-4B-Instruct-2507__aime/B_5curve_L5.png)

![B L5 zoom128](figs/Qwen3-4B-Instruct-2507__aime/B_5curve_L5_zoom128.png)

![C L5](figs/Qwen3-4B-Instruct-2507__aime/C_gap_L5.png)

![B Lr](figs/Qwen3-4B-Instruct-2507__aime/B_5curve_Lr.png)

![B Lr zoom128](figs/Qwen3-4B-Instruct-2507__aime/B_5curve_Lr_zoom128.png)

![C Lr](figs/Qwen3-4B-Instruct-2507__aime/C_gap_Lr.png)

## Qwen3-4B-Instruct-2507 / mathfull

**Overall** (n_problems=4862) snapshot:
| N | T\* | best | T=1.0 | T=0.1 | random_T | equal_mix | Δ vs T1 | Δ vs T0.1 | Δ vs rand | Δ vs eq_mix |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.8 | 86.98 | 86.84 | 86.23 | 86.77 | 86.64 | +0.14pp | +0.74pp | +0.21pp | +0.34pp |
| 16 | 0.8 | 88.70 | 88.53 | 87.51 | 88.35 | 88.44 | +0.18pp | +1.19pp | +0.35pp | +0.26pp |
| 64 | 0.8 | 88.91 | 88.70 | 87.70 | 88.53 | 88.67 | +0.22pp | +1.21pp | +0.38pp | +0.24pp |
| 256 | 0.9 | 88.99 | 88.73 | 87.72 | 88.58 | 88.72 | +0.26pp | +1.26pp | +0.41pp | +0.27pp |

**T*(N=256) by stratum:**  
L1=T0.1 (acc 100.0, n=3844) | L2=T0.1 (acc 100.0, n=208) | L3=T0.3 (acc 94.4, n=153) | L4=T0.7 (acc 58.1, n=161) | L5=T1.1 (acc 12.1, n=400)

**N×T landscape by difficulty** (6 panels: overall + L1..L5)

![NxT by difficulty](figs/Qwen3-4B-Instruct-2507__mathfull/J_by_difficulty_grid.png)

![NxT by difficulty regret](figs/Qwen3-4B-Instruct-2507__mathfull/J_by_difficulty_grid_regret.png)

**Overall stratum detail**

![acc(T,N) overall](figs/Qwen3-4B-Instruct-2507__mathfull/G_acc_heatmap_overall.png)

![regret(T,N) overall](figs/Qwen3-4B-Instruct-2507__mathfull/H_regret_heatmap_overall.png)

![T* trajectory by level](figs/Qwen3-4B-Instruct-2507__mathfull/A_t_star_trajectory.png)

![5-curve overall](figs/Qwen3-4B-Instruct-2507__mathfull/B_5curve_overall.png)

ZOOM (N≥128, y-range auto-fit to top baselines — shows consensus_vote / equal_mix separation):

![5-curve overall zoom128](figs/Qwen3-4B-Instruct-2507__mathfull/B_5curve_overall_zoom128.png)

![gap overall](figs/Qwen3-4B-Instruct-2507__mathfull/C_gap_overall.png)

**5-curve baseline comparison per difficulty (L1..L5, Lr)**

![B L1](figs/Qwen3-4B-Instruct-2507__mathfull/B_5curve_L1.png)

![B L1 zoom128](figs/Qwen3-4B-Instruct-2507__mathfull/B_5curve_L1_zoom128.png)

![C L1](figs/Qwen3-4B-Instruct-2507__mathfull/C_gap_L1.png)

![B L2](figs/Qwen3-4B-Instruct-2507__mathfull/B_5curve_L2.png)

![B L2 zoom128](figs/Qwen3-4B-Instruct-2507__mathfull/B_5curve_L2_zoom128.png)

![C L2](figs/Qwen3-4B-Instruct-2507__mathfull/C_gap_L2.png)

![B L3](figs/Qwen3-4B-Instruct-2507__mathfull/B_5curve_L3.png)

![B L3 zoom128](figs/Qwen3-4B-Instruct-2507__mathfull/B_5curve_L3_zoom128.png)

![C L3](figs/Qwen3-4B-Instruct-2507__mathfull/C_gap_L3.png)

![B L4](figs/Qwen3-4B-Instruct-2507__mathfull/B_5curve_L4.png)

![B L4 zoom128](figs/Qwen3-4B-Instruct-2507__mathfull/B_5curve_L4_zoom128.png)

![C L4](figs/Qwen3-4B-Instruct-2507__mathfull/C_gap_L4.png)

![B L5](figs/Qwen3-4B-Instruct-2507__mathfull/B_5curve_L5.png)

![B L5 zoom128](figs/Qwen3-4B-Instruct-2507__mathfull/B_5curve_L5_zoom128.png)

![C L5](figs/Qwen3-4B-Instruct-2507__mathfull/C_gap_L5.png)

![B Lr](figs/Qwen3-4B-Instruct-2507__mathfull/B_5curve_Lr.png)

![B Lr zoom128](figs/Qwen3-4B-Instruct-2507__mathfull/B_5curve_Lr_zoom128.png)

![C Lr](figs/Qwen3-4B-Instruct-2507__mathfull/C_gap_Lr.png)
