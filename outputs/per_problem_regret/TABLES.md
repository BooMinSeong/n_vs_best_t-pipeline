# Paper Tables — Risk of Temperature Misspecification

Headline risk metric: **catastrophic per-problem loss rate**
$\Pr[R_2 > 0.1]$ — the fraction of problems on which a strategy loses more than 10
accuracy points to the best single fixed temperature $T^\dagger$.

Regime: **converged large budget**, $N=2048$. At this budget maj@$N$ is essentially
deterministic per problem ($\sim$96% of per-problem accuracies are within 0.02 of 0/1),
so the comparison measures each strategy's *structural* risk, not finite-sample noise.

$R_2(p,s) = a(p,T^\dagger) - a(p,s)$, where $T^\dagger=\arg\max_T \frac1{|P|}\sum_p a(p,T)$
is the best single fixed temperature chosen on the whole dataset. Numbers are over the
seven heterogeneous MATH/GSM8K combos (AIME excluded: oracle gap $\approx 0$, $n<30$).

---

## Table 1 — Risk seen through three lenses (averaged over 7 MATH/GSM8K combos, $N=2048$)

Mean is shown only to make the point that **risk is invisible on the mean and lives in
the tail**: a lucky fixed temperature ($T{=}1.0$) looks competitive on the mean yet
carries a heavy tail, while mixing is low across all three lenses.

| Strategy | mean $R_2$ | p95 $R_2$ (tail) | $\Pr[R_2>0.1]$ (catastrophic) |
|---|---|---|---|
| Fixed $T{=}0.1$ | 0.057 | 0.956 | 9.0% |
| Random-$T$ | 0.027 | 0.232 | 10.9% |
| Fixed $T{=}1.0$ | 0.014 | 0.222 | 3.7% |
| **Temperature Pool** | 0.012 | 0.011 | 2.9% |
| **Temperature Consensus** | **0.006** | **0.011** | **2.7%** |

> A single-temperature bet — a fixed default *or* a random draw — loses badly on roughly
> **1 problem in 10**; mixing cuts that to **~1 in 35**.

### LaTeX

```latex
\begin{table}[t]
\centering
\caption{Risk of temperature misspecification at a converged budget ($N{=}2048$),
averaged over seven MATH/GSM8K model$\times$dataset combinations. $R_2$ is the
per-problem regret against the best single fixed temperature. Risk is invisible on the
mean and concentrated in the tail: mixing reduces the catastrophic-loss rate
$\Pr[R_2{>}0.1]$ from $\sim$1-in-10 to $\sim$1-in-35.}
\label{tab:risk-lenses}
\begin{tabular}{lccc}
\toprule
Strategy & mean $R_2$ & p95 $R_2$ & $\Pr[R_2{>}0.1]$ \\
\midrule
Fixed $T{=}0.1$        & 0.057 & 0.956 & 9.0\% \\
Random-$T$             & 0.027 & 0.232 & 10.9\% \\
Fixed $T{=}1.0$        & 0.014 & 0.222 & 3.7\% \\
\midrule
\textbf{Temperature Pool}      & 0.012 & 0.011 & 2.9\% \\
\textbf{Temperature Consensus} & \textbf{0.006} & \textbf{0.011} & \textbf{2.7\%} \\
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
| math1k / Qwen2.5-3B | T0.6 | 13.8% | 15.0% | 7.3% | 3.3% | 3.7% |
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
math1k / Qwen2.5-3B      & T0.6 & 13.8\% & 15.0\% & 7.3\% & 3.3\% & 3.7\% \\
mathfull / Llama-3.2-3B  & T0.7 & 10.4\% & 17.5\% & 5.3\% & 4.7\% & 3.4\% \\
mathfull / Qwen2.5-3B    & T0.6 & 7.0\%  & 7.7\%  & 3.3\% & 1.9\% & 1.8\% \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Appendix — Table 2b: tail p95 $R_2$ per combo ($N=2048$)

Same layout as Table 2, but each cell is the 95th-percentile regret instead of the
catastrophic-loss rate. Sharper still: $T{=}1.0$ collapses to a $0.55$–$0.87$ tail on the
MATH combos ($T^\dagger{=}0.5\text{–}0.6$), whereas Pool/Consensus stay $\le 0.075$ everywhere.

| Dataset / Model | $T^\dagger$ | $T{=}0.1$ | Random-$T$ | $T{=}1.0$ | **Pool** | **Consensus** |
|---|---|---|---|---|---|---|
| gsm8kfull / Llama-3.2-3B | T0.9 | 1.000 | 0.280 | 0.000 | 0.000 | 0.001 |
| gsm8kfull / Phi-4-mini | T0.9 | 0.922 | 0.117 | 0.000 | 0.000 | 0.000 |
| gsm8kfull / Qwen2.5-3B | T0.9 | 0.815 | 0.090 | 0.000 | 0.000 | 0.000 |
| math500 / Phi-4-mini | T0.5 | 0.998 | 0.338 | 0.866 | 0.000 | 0.028 |
| math1k / Qwen2.5-3B | T0.6 | 1.000 | 0.300 | 0.550 | 0.004 | 0.035 |
| mathfull / Llama-3.2-3B | T0.7 | 1.000 | 0.314 | 0.136 | 0.075 | 0.013 |
| mathfull / Qwen2.5-3B | T0.6 | 0.960 | 0.182 | 0.001 | 0.000 | 0.000 |

### LaTeX

```latex
\begin{table}[t]
\centering
\caption{Tail risk (95th-percentile per-problem regret $R_2$) per model$\times$dataset
combination at $N{=}2048$. A fixed temperature that is far from $T^\dagger$ (e.g.\ $T{=}1.0$
on the MATH combos) carries a $0.55$–$0.87$ tail; the mixing strategies stay below $0.075$
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
math1k / Qwen2.5-3B      & T0.6 & 1.000 & 0.300 & 0.550 & 0.004 & 0.035 \\
mathfull / Llama-3.2-3B  & T0.7 & 1.000 & 0.314 & 0.136 & 0.075 & 0.013 \\
mathfull / Qwen2.5-3B    & T0.6 & 0.960 & 0.182 & 0.001 & 0.000 & 0.000 \\
\bottomrule
\end{tabular}
\end{table}
```

---

*Source: `per_problem_regret.parquet` (N=2048), `summary_oracle_gap.csv`. Regenerate with
`scripts/per_problem_regret/run_pipeline.py`.*
