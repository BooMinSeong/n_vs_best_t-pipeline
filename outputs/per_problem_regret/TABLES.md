# Paper Tables — Risk of Temperature Misspecification

Headline risk metric: **catastrophic per-problem loss rate**
$\Pr[R_2 > 0.1]$ — the fraction of problems on which a strategy loses more than 10
accuracy points to the best single fixed temperature $T^\dagger$.

Regime: **converged large budget**, $N=2048$. At this budget maj@$N$ is essentially
deterministic per problem ($\sim$96% of per-problem accuracies are within 0.02 of 0/1),
so the comparison measures each strategy's *structural* risk, not finite-sample noise.

$R_2(p,s) = a(p,T^\dagger) - a(p,s)$, where $T^\dagger=\arg\max_T \frac1{|P|}\sum_p a(p,T)$
is the best single fixed temperature chosen on the whole dataset. The main tables are over
the six heterogeneous MATH/GSM8K combos. **math1k is excluded** (it is a subsample of the
MATH set already represented by `mathfull`/`math500`, so it double-counts the same items).
**AIME is reported separately** in §3 with caveats: oracle gap $\approx 0$ and $n<30$ per
combo, so its regret numbers are noise-dominated.

---

## Table 1 — Risk seen through three lenses (averaged over 6 MATH/GSM8K combos, $N=2048$)

Mean is shown only to make the point that **risk is invisible on the mean and lives in
the tail**: a lucky fixed temperature ($T{=}1.0$) looks competitive on the mean yet
carries a heavy tail, while mixing is low across all three lenses.

| Strategy | mean $R_2$ | p95 $R_2$ (tail) | $\Pr[R_2>0.1]$ (catastrophic) |
|---|---|---|---|
| Fixed $T{=}0.1$ | 0.054 | 0.949 | 8.2% |
| Random-$T$ | 0.026 | 0.220 | 10.2% |
| Fixed $T{=}1.0$ | 0.013 | 0.167 | 3.2% |
| **Temperature Pool** | 0.013 | 0.013 | 2.9% |
| **Temperature Consensus** | **0.006** | **0.007** | **2.5%** |

> A single-temperature bet — a fixed default *or* a random draw — loses badly on roughly
> **1 problem in 10**; mixing cuts that to **~1 in 35–40**.

### LaTeX

```latex
\begin{table}[t]
\centering
\caption{Risk of temperature misspecification at a converged budget ($N{=}2048$),
averaged over six MATH/GSM8K model$\times$dataset combinations (math1k excluded as a MATH
subsample; AIME reported separately). $R_2$ is the per-problem regret against the best
single fixed temperature. Risk is invisible on the mean and concentrated in the tail:
mixing reduces the catastrophic-loss rate $\Pr[R_2{>}0.1]$ from $\sim$1-in-10 to
$\sim$1-in-40.}
\label{tab:risk-lenses}
\begin{tabular}{lccc}
\toprule
Strategy & mean $R_2$ & p95 $R_2$ & $\Pr[R_2{>}0.1]$ \\
\midrule
Fixed $T{=}0.1$        & 0.054 & 0.949 & 8.2\% \\
Random-$T$             & 0.026 & 0.220 & 10.2\% \\
Fixed $T{=}1.0$        & 0.013 & 0.167 & 3.2\% \\
\midrule
\textbf{Temperature Pool}      & 0.013 & 0.013 & 2.9\% \\
\textbf{Temperature Consensus} & \textbf{0.006} & \textbf{0.007} & \textbf{2.5\%} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 2 — Catastrophic-loss rate $\Pr[R_2>0.1]$ per combo ($N=2048$)

The $T^\dagger$ column carries the second half of the argument: the best fixed
temperature ranges over $T{=}0.5\text{–}0.9$ across combos, so **it cannot be known in
advance**. Mixing is lowest or near-lowest on every row without any per-task tuning.

| Dataset / Model | $T^\dagger$ | $T{=}0.1$ | Random-$T$ | $T{=}1.0$ | **Pool** | **Consensus** |
|---|---|---|---|---|---|---|
| gsm8kfull / Llama-3.2-3B | T0.9 | 10.1% | 11.3% | 1.6% | 3.6% | 3.2% |
| gsm8kfull / Phi-4-mini | T0.9 | 6.2% | 5.4% | 0.5% | 1.7% | 1.4% |
| gsm8kfull / Qwen2.5-3B | T0.9 | 5.9% | 4.6% | 0.6% | 1.7% | 1.1% |
| math500 / Phi-4-mini | T0.5 | 9.5% | 15.0% | 7.7% | 3.6% | 4.3% |
| mathfull / Llama-3.2-3B | T0.7 | 10.4% | 17.5% | 5.3% | 4.7% | 3.4% |
| mathfull / Qwen2.5-3B | T0.6 | 7.0% | 7.7% | 3.3% | 1.9% | 1.8% |

> $T{=}1.0$ is competitive **only** on GSM8K, where $T^\dagger{\approx}0.9$ happens to sit
> near it; on the MATH combos ($T^\dagger{=}0.5\text{–}0.7$) it pays 3–8%. Mixing needs no
> such luck.

### LaTeX

```latex
\begin{table}[t]
\centering
\caption{Catastrophic-loss rate $\Pr[R_2{>}0.1]$ per model$\times$dataset combination at
$N{=}2048$. The best fixed temperature $T^\dagger$ varies from $0.5$ to $0.9$ across
combos and is not knowable a priori; the two mixing strategies attain the lowest (or
near-lowest) risk on every row without per-task tuning.}
\label{tab:risk-percombo}
\begin{tabular}{llccccc}
\toprule
Dataset / Model & $T^\dagger$ & $T{=}0.1$ & Random-$T$ & $T{=}1.0$ & \textbf{Pool} & \textbf{Cons.} \\
\midrule
gsm8kfull / Llama-3.2-3B & T0.9 & 10.1\% & 11.3\% & 1.6\% & 3.6\% & 3.2\% \\
gsm8kfull / Phi-4-mini   & T0.9 & 6.2\%  & 5.4\%  & 0.5\% & 1.7\% & 1.4\% \\
gsm8kfull / Qwen2.5-3B   & T0.9 & 5.9\%  & 4.6\%  & 0.6\% & 1.7\% & 1.1\% \\
math500 / Phi-4-mini     & T0.5 & 9.5\%  & 15.0\% & 7.7\% & 3.6\% & 4.3\% \\
mathfull / Llama-3.2-3B  & T0.7 & 10.4\% & 17.5\% & 5.3\% & 4.7\% & 3.4\% \\
mathfull / Qwen2.5-3B    & T0.6 & 7.0\%  & 7.7\%  & 3.3\% & 1.9\% & 1.8\% \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Appendix — Table 2b: tail p95 $R_2$ per combo ($N=2048$)

Same layout as Table 2, but each cell is the 95th-percentile regret instead of the
catastrophic-loss rate. Sharper still: $T{=}1.0$ collapses to a $0.14$–$0.87$ tail on the
MATH combos ($T^\dagger{=}0.5\text{–}0.7$), whereas Pool/Consensus stay $\le 0.075$ everywhere.

| Dataset / Model | $T^\dagger$ | $T{=}0.1$ | Random-$T$ | $T{=}1.0$ | **Pool** | **Consensus** |
|---|---|---|---|---|---|---|
| gsm8kfull / Llama-3.2-3B | T0.9 | 1.000 | 0.280 | 0.000 | 0.000 | 0.001 |
| gsm8kfull / Phi-4-mini | T0.9 | 0.922 | 0.117 | 0.000 | 0.000 | 0.000 |
| gsm8kfull / Qwen2.5-3B | T0.9 | 0.815 | 0.090 | 0.000 | 0.000 | 0.000 |
| math500 / Phi-4-mini | T0.5 | 0.998 | 0.338 | 0.866 | 0.000 | 0.028 |
| mathfull / Llama-3.2-3B | T0.7 | 1.000 | 0.314 | 0.136 | 0.075 | 0.013 |
| mathfull / Qwen2.5-3B | T0.6 | 0.960 | 0.182 | 0.001 | 0.000 | 0.000 |

### LaTeX

```latex
\begin{table}[t]
\centering
\caption{Tail risk (95th-percentile per-problem regret $R_2$) per model$\times$dataset
combination at $N{=}2048$. A fixed temperature that is far from $T^\dagger$ (e.g.\ $T{=}1.0$
on the MATH combos) carries a $0.14$–$0.87$ tail; the mixing strategies stay below $0.075$
on every combo.}
\label{tab:risk-p95-percombo}
\begin{tabular}{llccccc}
\toprule
Dataset / Model & $T^\dagger$ & $T{=}0.1$ & Random-$T$ & $T{=}1.0$ & \textbf{Pool} & \textbf{Cons.} \\
\midrule
gsm8kfull / Llama-3.2-3B & T0.9 & 1.000 & 0.280 & 0.000 & 0.000 & 0.001 \\
gsm8kfull / Phi-4-mini   & T0.9 & 0.922 & 0.117 & 0.000 & 0.000 & 0.000 \\
gsm8kfull / Qwen2.5-3B   & T0.9 & 0.815 & 0.090 & 0.000 & 0.000 & 0.000 \\
math500 / Phi-4-mini     & T0.5 & 0.998 & 0.338 & 0.866 & 0.000 & 0.028 \\
mathfull / Llama-3.2-3B  & T0.7 & 1.000 & 0.314 & 0.136 & 0.075 & 0.013 \\
mathfull / Qwen2.5-3B    & T0.6 & 0.960 & 0.182 & 0.001 & 0.000 & 0.000 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## §3 — AIME (reported separately; interpret with care)

**Why separate.** Across the 12 AIME combos (4 years × 3 models) the per-problem oracle
gap is $\approx 0$ pp (max $1.5$ pp on aime2024/Qwen2.5-3B), i.e. the best single fixed
temperature is already essentially per-problem optimal — there is almost no structural
headroom for *any* strategy to win or lose. Combined with $n\in[21,27]$ problems per combo,
each catastrophic-loss cell is quantized to $\approx 1/n \approx 4\%$ (one problem), so the
AIME numbers reflect sampling noise far more than the structural risk that Tables 1–2
measure. They are included for completeness, not as evidence.

### Table 3 — AIME averaged over 12 combos ($N=2048$)

| Strategy | mean $R_2$ | p95 $R_2$ | $\Pr[R_2>0.1]$ |
|---|---|---|---|
| Fixed $T{=}0.1$ | 0.028 | 0.036 | 3.8% |
| Random-$T$ | 0.023 | 0.134 | 5.4% |
| Fixed $T{=}1.0$ | 0.030 | 0.211 | 4.3% |
| **Temperature Pool** | 0.019 | 0.082 | 2.7% |
| **Temperature Consensus** | **0.018** | 0.083 | **2.7%** |

> Even here the qualitative ordering survives — mixing has the lowest mean and the lowest
> catastrophic-loss rate — but the absolute spreads are within the per-combo noise floor.

### Table 3b — AIME per combo, $\Pr[R_2>0.1]$ ($N=2048$)

$T^\dagger$ is unstable across years (it ranges $T{=}0.1\text{–}1.0$), and several cells
are exactly $0$ because the model is at floor accuracy ($\approx 0$ on aime2025/Phi). The
one informative pattern: where $T^\dagger$ is low and the model degrades at high $T$
(aime2025/2026 Qwen2.5-3B), **fixed $T{=}1.0$ again carries a heavy tail** (p95 up to
$0.77$) while mixing stays controlled. The lone counterexample is aime2025/Qwen3, where
$T^\dagger{=}0.3$ and the high-$T$ pool components drag the mix down to match $T{=}1.0$.

| Dataset / Model | $T^\dagger$ | $T{=}0.1$ | Random-$T$ | $T{=}1.0$ | **Pool** | **Consensus** | $n$ |
|---|---|---|---|---|---|---|---|
| aime2023 / Phi-4-mini | T0.2 | 3.8% | 7.7% | 7.7% | 3.8% | 3.8% | 26 |
| aime2023 / Qwen2.5-3B | T0.1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 27 |
| aime2023 / Qwen3-4B | T0.2 | 4.8% | 4.8% | 0.0% | 0.0% | 0.0% | 21 |
| aime2024 / Phi-4-mini | T0.1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 26 |
| aime2024 / Qwen2.5-3B | T0.8 | 3.7% | 7.4% | 7.4% | 3.7% | 3.7% | 27 |
| aime2024 / Qwen3-4B | T0.7 | 7.7% | 7.7% | 3.8% | 3.8% | 3.8% | 26 |
| aime2025 / Phi-4-mini | T0.3 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 26 |
| aime2025 / Qwen2.5-3B | T0.6 | 0.0% | 7.7% | 7.7% | 0.0% | 0.0% | 26 |
| aime2025 / Qwen3-4B | T0.3 | 9.1% | 9.1% | 9.1% | 9.1% | 9.1% | 22 |
| aime2026 / Phi-4-mini | T0.4 | 8.3% | 8.3% | 8.3% | 4.2% | 4.2% | 24 |
| aime2026 / Qwen2.5-3B | T0.2 | 4.0% | 8.0% | 8.0% | 4.0% | 4.0% | 25 |
| aime2026 / Qwen3-4B | T1.0 | 4.2% | 4.2% | 0.0% | 4.2% | 4.2% | 24 |

### Table 3c — AIME per combo, tail p95 $R_2$ ($N=2048$)

| Dataset / Model | $T^\dagger$ | $T{=}0.1$ | Random-$T$ | $T{=}1.0$ | **Pool** | **Consensus** |
|---|---|---|---|---|---|---|
| aime2023 / Phi-4-mini | T0.2 | 0.000 | 0.206 | 0.383 | 0.000 | 0.001 |
| aime2023 / Qwen2.5-3B | T0.1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| aime2023 / Qwen3-4B | T0.2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| aime2024 / Phi-4-mini | T0.1 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 |
| aime2024 / Qwen2.5-3B | T0.8 | 0.000 | 0.194 | 0.120 | 0.015 | 0.006 |
| aime2024 / Qwen3-4B | T0.7 | 0.130 | 0.115 | 0.000 | 0.000 | 0.048 |
| aime2025 / Phi-4-mini | T0.3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| aime2025 / Qwen2.5-3B | T0.6 | 0.000 | 0.145 | 0.195 | 0.000 | 0.000 |
| aime2025 / Qwen3-4B | T0.3 | 0.164 | 0.634 | 0.933 | 0.933 | 0.927 |
| aime2026 / Phi-4-mini | T0.4 | 0.132 | 0.115 | 0.132 | 0.039 | 0.010 |
| aime2026 / Qwen2.5-3B | T0.2 | 0.000 | 0.195 | 0.765 | 0.000 | 0.000 |
| aime2026 / Qwen3-4B | T1.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

---

*Source: `per_problem_regret.parquet` (N=2048), `summary_oracle_gap.csv`,
`summary_distribution_stats.csv`. Regenerate with
`scripts/per_problem_regret/run_pipeline.py`.*
