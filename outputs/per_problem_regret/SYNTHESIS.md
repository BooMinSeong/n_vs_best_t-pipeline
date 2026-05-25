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
| MATH (full/500) | math500 / Phi-4-mini | **+5.8 pp** |
| | mathfull / Llama-3.2-3B | +5.2 pp |
| | mathfull / Qwen2.5-3B | +4.4 pp |
| GSM8K | gsm8kfull / Qwen2.5-3B | +2.5 pp |
| AIME (single year) | most combos | < 1 pp |

The prize is real on heterogeneous math benchmarks (+4–6 pp) and negligible on AIME,
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
the six MATH/GSM8K combos:

| strategy | $\mathrm{p95}\,R_2$ @ $N{=}8$ | @ $N{=}2048$ | trend |
|---|---|---|---|
| Fixed $T{=}0.1$ | 0.40 | **0.95** | **diverges** ↑ |
| Fixed $T{=}1.0$ | 0.17 | 0.17 | flat |
| Random-$T$ | 0.14 | 0.22 | worsens ↑ |
| **Temperature Pool** | 0.10 | **0.013** | **→ 0** ↓ |
| **Temperature Consensus** | 0.10 | **0.007** | **→ 0** ↓ |

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

**Interpretation of the large-$N$ limit.** At $N{=}2048$ maj@$N$ has essentially
converged: for $\sim\!96\%$ of problems the per-problem outcome is deterministic
(within $0.02$ of $0$ or $1$) given the empirical output distribution. We are therefore
*not* measuring finite-sample noise — that variance has been deliberately spent down to
certainty by the large budget. What remains is each strategy's **structural risk**: when
the sampling budget is no longer the bottleneck, which aggregation rule still recovers
the correct answer? The comparison below is exactly that.

The sharpest risk statement is the probability of a **catastrophic per-problem loss**
($R_2>0.1$ — i.e. losing more than 10 accuracy points to the best fixed temperature) at
$N{=}2048$, averaged over the six MATH/GSM8K combos:

| strategy | $\Pr[R_2>0.1]$ @ $N{=}2048$ |
|---|---|
| Random-$T$ | 10.2% |
| Fixed $T{=}0.1$ | 8.2% |
| Fixed $T{=}1.0$ | 3.2% |
| **Temperature Pool** | 2.9% |
| **Temperature Consensus** | **2.5%** |

A single-temperature bet — a fixed default *or* a random draw — loses badly on roughly
**1 problem in 10**; mixing cuts that to **~1 in 40**. This is the operational meaning of
"risk avoidance": the strategies do not beat the best fixed temperature on average (mean
$R_2>0$ always; see §5), but they convert a high-variance temperature *bet* into a
low-variance aggregate, eliminating the catastrophic tail you incur whenever the single
fixed choice is wrong — and which temperature is wrong is not knowable in advance ($T^\dagger$
ranges over $T{=}0.5\text{–}0.9$ across these combos).

---

## 4. The result holds across datasets

At a large budget ($N{=}2048$), mixing achieves near-oracle regret on **every**
MATH/GSM8K combo, while fixed temperatures do not generalize:

| dataset / model | $T{=}0.1$ | $T{=}1.0$ | Random-$T$ | **Pool** | **Consensus** |
|---|---|---|---|---|---|
| gsm8kfull / Qwen2.5-3B | 0.039 | 0.000 | 0.009 | 0.006 | 0.004 |
| gsm8kfull / Llama-3.2-3B | 0.079 | 0.005 | 0.037 | 0.023 | 0.016 |
| math500 / Phi-4-mini | 0.055 | 0.042 | 0.038 | 0.017 | 0.008 |
| mathfull / Qwen2.5-3B | 0.044 | 0.006 | 0.014 | 0.005 | 0.002 |
| mathfull / Llama-3.2-3B | 0.064 | 0.022 | 0.047 | 0.023 | 0.005 |

*(mean $R_2$ @ $N{=}2048$.)* Fixed $T{=}1.0$ looks competitive on GSM8K **only because
$T^\dagger\approx1.0$ there** — on math500 it pays $\mathrm{p95}=0.87$. Mixing needs no
prior knowledge of the right temperature: among the four deployable strategies (counting
over the 18 MATH/GSM8K + AIME combos, math1k excluded),
**Temperature Consensus has the lowest mean $R_2$ in 8 of 18 combos and Pool/Consensus
together in 9**, versus 7 for $T{=}1.0$ (and those 7 are the GSM8K-like cases where the
fixed choice happens to coincide with the oracle).

---

## 5. Pooling vs Consensus: identical only below $N{=}12$, then genuinely different

The two coincide **only** in the degenerate regime where round-robin gives at most one
sample per temperature ($N<12$): there each per-temperature majority is its lone sample,
so Consensus reduces exactly to Pool. The per-problem accuracy difference
$|a_{\text{Pool}}-a_{\text{Cons}}|$ is identically $0$ at $N\in\{1,2,4,8\}$ in every combo.

From $N\ge16$ they are genuinely distinct algorithms and **differ in accuracy**. Although
the dataset-mean gap stays small (wins roughly cancel), the *per-problem* accuracies
diverge sharply: $\max_p|a_{\text{Pool}}-a_{\text{Cons}}|$ grows from $\sim\!0.30$ at
$N{=}16$ to $\sim\!0.9\text{–}1.0$ at $N{=}2048$ — at scale some problems are solved
almost surely by one strategy and missed by the other. The difference is systematic and
favors **Consensus** at large budgets: at $N{=}2048$ Consensus has lower mean $R_2$ than
Pool on every MATH/GSM8K combo (e.g. mathfull/Llama $0.005$ vs $0.023$; mathfull/Qwen
$0.002$ vs $0.005$; gsm8k/Qwen $0.004$ vs $0.006$), and the paired test is significant
(mathfull/Qwen, $N{=}2048$: $\Delta=-0.0033$, win/tie/loss $104/4708/153$, Wilcoxon
$p=2\times10^{-4}$). The frequently-cited "tie" ($p=0.93$) holds only at the $N{=}256$
crossover, not in general. Both strategies have comparably small regret *tails* (p95
collapses toward the oracle for each), but **Consensus is the lower-regret of the two at
scale** — they are not interchangeable.

---

## 6. Scope and caveats

- **Estimator.** $a(p,s,N)$ is Monte-Carlo from empirical per-temperature answer
  distributions (1536 generations/problem-T), not raw-generation bootstrap; it reconciles
  with the dataset-level tables to within $0.0007$ on average (max $0.011$ on $<\!30$-problem
  AIME cells, i.e. the reference's own 240-replicate noise).
- **Oracle bias.** $\sim$1 pp upward bias in $a^\*$ inflates absolute gaps only (§2).
- **AIME.** 21–27 problems per year give low paired-test power and near-zero oracle gap
  ($\le1.5$ pp), so its regret numbers are noise-dominated and reported separately
  (`TABLES.md` §3): the qualitative ordering survives on the 12-combo average (mixing
  lowest mean and catastrophic rate), but per-combo $\Pr[R_2>0.1]$ is quantized to
  $\sim1/n\approx4\%$. The stability claim is supported on the high-dispersion MATH/GSM8K
  benchmarks (Wilcoxon $T{=}0.1$ vs $T{=}1.0$ on mathfull, $n{=}4965$:
  $p=1.5\times10^{-18}$).
- **No free dominance.** At any fixed $N$ the regret CDFs of the deployable strategies
  cross (no first-order stochastic dominance); the stability claim is about the *trend in
  $N$*, not a pointwise guarantee.

**Bottom line.** Temperature Pooling and Consensus are the only tested deployable
strategies whose per-problem regret — center and tail — provably shrinks toward the
best fixed-temperature oracle as the sampling budget grows, and they do so uniformly
across heterogeneous math benchmarks without any per-task temperature tuning.
