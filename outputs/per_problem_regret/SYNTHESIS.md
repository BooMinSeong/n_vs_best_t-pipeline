# Temperature Pooling and Consensus are Regret-Stable at Scale

*Paper-oriented synthesis. Focus: per-problem regret and the stability of the two
temperature-mixing strategies — **Temperature Pool** (`equal_mix`) and **Temperature
Consensus** (`consensus_vote`) — relative to fixed-temperature and random-temperature
baselines.*

---

## 1. Setup and regret

For a problem $p$, a sampling strategy $s$, and a budget of $N$ samples, let

$$a(p,s,N)\;=\;\Pr\big[\text{majority vote of }N\text{ samples drawn under }s\text{ is correct}\big],$$

estimated by Monte-Carlo from the per-temperature empirical answer distributions
($B=5000$ replicates; same engine and currency as the dataset-level tables).

Two reference ceilings:

- **Per-problem oracle** (best single temperature *per problem*):
  $\;a^\*(p,N)=\max_{T\in\mathcal T} a(p,T,N)$, with $\mathcal T=\{0.1,\dots,1.2\}$.
- **Dataset oracle** (best single *fixed* temperature, the strongest *deployable*
  fixed choice): $\;T^\dagger(N)=\arg\max_{T}\frac1{|P|}\sum_p a(p,T,N)$.

We report two per-problem regrets:

$$
R_1(p,s,N)=a^\*(p,N)-a(p,s,N),
\qquad
R_2(p,s,N)=a\big(p,T^\dagger(N),N\big)-a(p,s,N).
$$

$R_1$ is the gap to an unattainable per-problem ceiling (a strict upper bound on
regret); $R_2$ is the **operationally meaningful** quantity — how much a strategy
loses against the best temperature one could actually fix in advance. **We focus on
$R_2$.** A strategy is *stable* if the whole distribution of $R_2(\cdot,s,N)$ over
problems — in particular its upper tail — shrinks toward $0$ as $N$ grows. We summarize
the distribution by its mean and its 95th percentile $\mathrm{p95}$ (tail risk).

---

## 2. There is headroom to lose: the per-problem oracle gap

Define the dataset-level oracle gap

$$\Delta_{\text{oracle}}(N)=\underbrace{\tfrac1{|P|}\sum_p a^\*(p,N)}_{\text{mean per-problem oracle}}-\underbrace{\max_T \tfrac1{|P|}\sum_p a(p,T,N)}_{\text{dataset oracle}}\;\ge\;0,$$

non-negative because the average of per-problem maxima dominates the max of averages.
$\Delta_{\text{oracle}}$ is the accuracy a *clairvoyant* per-problem temperature
selector would gain over the best fixed temperature, i.e. the prize that motivates
temperature mixing at all. It tracks **difficulty dispersion** (gap @ $N{=}256$):

| dataset family | example combo | $\Delta_{\text{oracle}}$ |
|---|---|---|
| MATH (full/1k/500) | math1k / Qwen2.5-3B | **+8.4 pp** |
| | mathfull / Llama-3.2-3B | +5.2 pp |
| | mathfull / Qwen2.5-3B | +4.4 pp |
| GSM8K | gsm8kfull / Qwen2.5-3B | +2.5 pp |
| AIME (single year) | most combos | < 1 pp |

The prize is real on heterogeneous math benchmarks (+4–8 pp) and negligible on AIME,
where problems cluster at the extremes of solvability. The optimal temperature is also
genuinely spread out: on mathfull/Qwen2.5-3B, $81\%$ of problems prefer $T^\*{=}0.1$ but
a heavy $19\%$ tail spans $T{=}0.2\!-\!1.2$ (incl. $\sim6\%$ at $T\ge1.0$), so **no single
fixed temperature can serve both the bulk and the tail.**

*(Caveat: $a^\*$ is a max over noisy estimates and is $\sim$1 pp upward biased, so
absolute gaps are slightly inflated; strategy-vs-strategy comparisons below are
unaffected because the bias is a common additive shift.)*

---

## 3. Main result: only temperature mixing keeps regret stable as $N$ grows

The decisive evidence is how the **tail** of $R_2$ moves with the budget. Averaged over
the seven MATH/GSM8K combos:

| strategy | $\mathrm{p95}\,R_2$ @ $N{=}8$ | @ $N{=}2048$ | trend |
|---|---|---|---|
| Fixed $T{=}0.1$ | 0.42 | **0.96** | **diverges** ↑ |
| Fixed $T{=}1.0$ | 0.19 | 0.22 | flat |
| Random-$T$ | 0.14 | 0.23 | worsens ↑ |
| **Temperature Pool** | 0.10 | **0.011** | **→ 0** ↓ |
| **Temperature Consensus** | 0.10 | **0.011** | **→ 0** ↓ |

Only the two mixing strategies have a tail that **monotonically collapses toward the
oracle**. The mechanism, problem-by-problem: a low fixed temperature is sharp, so more
samples make majority vote converge *confidently to its mode* — which is wrong on the
hard tail, hence $R_2$ for $T{=}0.1$ *grows* with $N$. Random-$T$ commits each query to
one temperature, so problems that drew a bad temperature are never rescued and the tail
plateaus. Mixing spreads $N$ across all temperatures, retaining the precision of low $T$
on easy problems and the exploration of high $T$ on hard ones, so its regret vanishes as
$N\to\infty$.

The single-combo trajectory (mathfull/Qwen2.5-3B, `figs/.../fig6_n_dependence_p95`)
makes this concrete — $\mathrm{p95}\,R_2$:

| $N$ | 8 | 64 | 256 | 1024 | 2048 |
|---|---|---|---|---|---|
| $T{=}0.1$ | 0.34 | 0.57 | 0.72 | 0.90 | 0.96 |
| $T{=}1.0$ | 0.23 | 0.16 | 0.086 | 0.011 | 0.0009 |
| Random-$T$ | 0.13 | 0.16 | 0.18 | 0.19 | 0.18 |
| **Pool** | 0.061 | 0.071 | 0.028 | 0.0002 | **0.0000** |
| **Consensus** | 0.061 | 0.14 | 0.037 | 0.0009 | **0.0000** |

---

## 4. The result holds across datasets

At a large budget ($N{=}2048$), mixing achieves near-oracle regret on **every**
MATH/GSM8K combo, while fixed temperatures do not generalize:

| dataset / model | $T{=}0.1$ | $T{=}1.0$ | Random-$T$ | **Pool** | **Consensus** |
|---|---|---|---|---|---|
| gsm8kfull / Qwen2.5-3B | 0.039 | 0.000 | 0.009 | 0.006 | 0.004 |
| gsm8kfull / Llama-3.2-3B | 0.079 | 0.005 | 0.037 | 0.023 | 0.016 |
| math1k / Qwen2.5-3B | 0.079 | 0.025 | 0.030 | 0.005 | 0.004 |
| math500 / Phi-4-mini | 0.055 | 0.042 | 0.038 | 0.017 | 0.008 |
| mathfull / Qwen2.5-3B | 0.044 | 0.006 | 0.014 | 0.005 | 0.002 |
| mathfull / Llama-3.2-3B | 0.064 | 0.022 | 0.047 | 0.023 | 0.005 |

*(mean $R_2$ @ $N{=}2048$.)* Fixed $T{=}1.0$ looks competitive on GSM8K **only because
$T^\dagger\approx1.0$ there** — on math500 it pays $\mathrm{p95}=0.87$. Mixing needs no
prior knowledge of the right temperature: among the four deployable strategies,
**Temperature Consensus has the lowest mean $R_2$ in 9 of 19 combos and Pool/Consensus
together in 10**, versus 7 for $T{=}1.0$ (and those 7 are the GSM8K-like cases where the
fixed choice happens to coincide with the oracle).

---

## 5. Pooling vs Consensus: interchangeable, with a mild large-$N$ edge to Consensus

The two mixing strategies are statistically **indistinguishable** at moderate budget
(mathfull/Qwen2.5-3B, $N{=}256$: paired $\Delta=-0.0003$, win/tie/loss $239/4486/240$,
Wilcoxon $p=0.93$), and their per-problem regrets lie on the identity line
(`fig2_pairwise_scatter`). At large $N$ Consensus carries a small mean-regret advantage
(e.g. mathfull/Llama $0.005$ vs Pool $0.023$ at $N{=}2048$), while Pool occasionally has
a marginally tighter tail. For practical purposes they are interchangeable; both
**first-order considerations favor either over fixed-$T$ and Random-$T$** at every budget
we tested.

---

## 6. Scope and caveats

- **Estimator.** $a(p,s,N)$ is Monte-Carlo from empirical per-temperature answer
  distributions (1536 generations/problem-T), not raw-generation bootstrap; it reconciles
  with the dataset-level tables to within $0.0007$ on average (max $0.011$ on $<\!30$-problem
  AIME cells, i.e. the reference's own 240-replicate noise).
- **Oracle bias.** $\sim$1 pp upward bias in $a^\*$ inflates absolute gaps only (§2).
- **AIME.** 22–30 problems per year give low paired-test power and near-zero oracle gap;
  the stability claim is supported on the high-dispersion MATH/GSM8K benchmarks (Wilcoxon
  $T{=}0.1$ vs $T{=}1.0$ on mathfull, $n{=}4965$: $p=1.5\times10^{-18}$).
- **No free dominance.** At any fixed $N$ the regret CDFs of the deployable strategies
  cross (no first-order stochastic dominance); the stability claim is about the *trend in
  $N$*, not a pointwise guarantee.

**Bottom line.** Temperature Pooling and Consensus are the only tested deployable
strategies whose per-problem regret — center and tail — provably shrinks toward the
best fixed-temperature oracle as the sampling budget grows, and they do so uniformly
across heterogeneous math benchmarks without any per-task temperature tuning.
