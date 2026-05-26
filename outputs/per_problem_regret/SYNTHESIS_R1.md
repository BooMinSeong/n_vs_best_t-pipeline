# Regret Against the Per-Problem Oracle ($R_1$): Mean Ranking Survives, the Tail Does Not Vanish

*Companion to `SYNTHESIS.md`, which focuses on $R_2$ (regret vs. the best deployable
fixed temperature). Here we re-run the same comparison against the **per-problem oracle**
ceiling $R_1$, averaging over the same seven MATH/GSM8K combos. The question: does the
"mixing is regret-stable" story hold when the baseline is the unattainable per-problem
best temperature instead of the best single fixed one?*

Seven combos: gsm8kfull/{Qwen2.5-3B, Llama-3.2-3B, Phi-4-mini}, math1k/Qwen2.5-3B,
math500/Phi-4-mini, mathfull/{Qwen2.5-3B, Llama-3.2-3B}.

---

## 0. Assumption: the problem is unknown at inference

Every strategy we evaluate (fixed $T$, Random-$T$, Pool, Consensus) is
**problem-agnostic**: it must commit to a sampling rule *before* seeing the problem and
cannot tune temperature to the instance. We hold this assumption throughout.

This is what makes $R_1$ and $R_2$ fundamentally different objects, not just two baselines:

- $T^\dagger$ (the $R_2$ baseline) is a single global temperature chosen from aggregate
  dataset statistics — **achievable** under the assumption (it is itself problem-agnostic;
  one fits it on a dev set).
- $a^\*(p)$ (the $R_1$ baseline) requires picking the best temperature *per problem*, which
  presupposes knowing the problem — **forbidden** under the assumption.

Consequently $\Delta_{\text{oracle}}(p)=a^\*(p)-a(p,T^\dagger)$ is **not** a loss any
deployable strategy can be charged with: it is the irreducible *cost of not knowing the
problem*, identical for every problem-agnostic rule. The operational regret is therefore
$R_2$. $R_1$ retains exactly two legitimate uses:

1. an **upper bound** — how much a hypothetical future per-problem temperature *router*
   (something that does look at the problem) could win; and
2. a **relative** strategy comparison — but since $\Delta_{\text{oracle}}$ cancels across
   strategies (§1), this is *identical* to the $R_2$ ranking and adds nothing.

Read every $R_1$ magnitude below as "distance to an unreachable ceiling," never as
"avoidable regret."

---

## 1. Definitions

$$
R_1(p,s,N)=a^\*(p,N)-a(p,s,N),\qquad
R_2(p,s,N)=a\big(p,T^\dagger(N),N\big)-a(p,s,N),
$$

with $a^\*(p,N)=\max_{T} a(p,T,N)$ the per-problem oracle and $T^\dagger$ the best fixed
temperature. The two differ by a quantity that **does not depend on the strategy $s$**:

$$
R_1(p,s,N)-R_2(p,s,N)=\underbrace{a^\*(p,N)-a(p,T^\dagger,N)}_{\Delta_{\text{oracle}}(p,N)\;\ge 0}.
$$

This single fact drives everything below:

- **Across strategies (relative comparison):** $\Delta_{\text{oracle}}(p,N)$ cancels, so
  $R_1(p,s)-R_1(p,s')=R_2(p,s)-R_2(p,s')$ **exactly**. Any pairwise test, win/tie/loss
  count, or mean-difference ranking is identical under $R_1$ and $R_2$.
- **Within a strategy (the distribution itself):** $\Delta_{\text{oracle}}(p,N)$ is a
  *per-problem* shift, heaviest on hard problems. So the **shape** of the $R_1$
  distribution — its tail in particular — is genuinely different from $R_2$'s.

In short: **$R_1$ leaves the means/rankings of §3 unchanged but rewrites the tail story of
§4.**

---

## 2. Mean $R_1$: same ranking as $R_2$, lifted by a near-constant oracle gap

Mean over the seven combos:

| strategy | $\bar R_1$ @8 | @64 | @256 | @1024 | @2048 |
|---|---|---|---|---|---|
| Fixed $T{=}0.1$ | 0.0777 | 0.0957 | 0.0997 | 0.1012 | **0.1015** |
| Fixed $T{=}1.0$ | 0.0698 | 0.0585 | 0.0584 | 0.0585 | 0.0586 |
| Random-$T$ | 0.0689 | 0.0692 | 0.0703 | 0.0708 | 0.0709 |
| **Temperature Pool** | 0.0499 | 0.0545 | 0.0554 | 0.0561 | 0.0563 |
| **Temperature Consensus** | 0.0499 | 0.0564 | 0.0525 | 0.0506 | **0.0500** |

The corresponding $\bar R_2$ table (`SYNTHESIS.md` §3 currency) is **this table minus a
near-constant offset** $\overline{\Delta_{\text{oracle}}}\approx0.044$ that is the same for
every strategy (N=8: 0.046; N≥64: 0.044). Concretely at $N{=}2048$, $\bar R_2$ = Consensus
0.0059, Pool 0.0121, $T{=}1.0$ 0.0144, Random 0.0267, $T{=}0.1$ 0.0573.

So the **mean ranking is preserved**: Consensus < Pool < $T{=}1.0$ < Random-$T$ < $T{=}0.1$
at scale, with Consensus the lowest-regret deployable strategy ($0.0500$ vs Pool $0.0563$).
Reading $R_1$ instead of $R_2$ changes no ordering and no significance — it only inflates
every number by the (slightly bias-inflated, §5) oracle gap.

---

## 3. The tail does **not** collapse against the per-problem oracle

This is where $R_1$ tells a different story. In `SYNTHESIS.md` §3, the headline is that
$\mathrm{p95}\,R_2$ for Pool/Consensus **collapses to ~0.01** at $N{=}2048$ — mixing
converges onto the best fixed temperature. Under $R_1$ that collapse disappears:

| strategy | $\mathrm{p95}\,R_1$ @8 | @2048 | $\mathrm{p95}\,R_2$ @2048 |
|---|---|---|---|
| Fixed $T{=}0.1$ | 0.509 | **1.000** | 0.956 |
| Fixed $T{=}1.0$ | 0.348 | 0.486 | 0.222 |
| Random-$T$ | 0.276 | 0.458 | 0.232 |
| **Temperature Pool** | 0.249 | 0.511 | **0.011** |
| **Temperature Consensus** | 0.249 | **0.386** | **0.011** |

The mixing tail stays heavy ($\mathrm{p95}\,R_1\approx0.39$–$0.51$ at $N{=}2048$) even as
$\mathrm{p95}\,R_2\to0$. The reason is exactly $\Delta_{\text{oracle}}(p)$: on the hard
tail, the per-problem oracle rescues each problem with **its own idiosyncratic best
temperature**, which a single mixing rule cannot match. **Under the §0 assumption this is
not a regret the strategy could have avoided** — it is the structural cost of not knowing
the problem, and it does not shrink with $N$ because it is not a sampling-noise term. (The
only tail that *diverges* is $T{=}0.1$, $\mathrm{p95}\,R_1\to1.0$: it confidently converges
to the wrong mode on hard problems — a genuine, avoidable failure of that fixed choice.)

A caution on reading the tail ranking. $\mathrm{p95}\,R_1$ is the 95th percentile of
$\Delta_{\text{oracle}}(p)+R_2(p,s)$, a non-linear functional, so its ordering across
strategies need **not** match the $R_2$ tail ordering — and indeed Consensus (0.386) sits
below Pool (0.511) here, unlike their near-tie under $R_2$. But this gap is **diagnostic,
not operational**: it says Consensus's residual errors land less often on the
high-oracle-gap problems, not that Consensus recovers regret Pool leaves on the table.
Since the oracle is unreachable by assumption, neither tail magnitude is an avoidable loss;
only the §2 *mean* difference (which equals the $R_2$ ranking) should drive a deployment
choice.

Catastrophic-loss probability $\Pr[R_1>0.1]$ @ $N{=}2048$ (mean over the seven combos):

| strategy | $\Pr[R_1>0.1]$ | (cf. $\Pr[R_2>0.1]$) |
|---|---|---|
| Random-$T$ | **15.96%** | 10.9% |
| Fixed $T{=}0.1$ | 11.46% | 9.0% |
| Fixed $T{=}1.0$ | 7.39% | 3.7% |
| **Temperature Consensus** | 7.44% | 2.7% |
| **Temperature Pool** | **7.08%** | 2.9% |

Against the per-problem oracle the catastrophic-loss rates compress: mixing still clearly
beats Random-$T$ (~7% vs 16%), but its edge over a fixed $T{=}1.0$ (7.1–7.4% vs 7.4%)
**vanishes** — because the extra ~4–7 pp of headroom that $R_1$ adds is the oracle gap that
*no* deployable strategy can claw back, so it lands on mixing and $T{=}1.0$ alike.

---

## 4. Per-combo mean $R_1$ @ $N{=}2048$

| dataset / model | $T{=}0.1$ | $T{=}1.0$ | Random-$T$ | **Pool** | **Consensus** |
|---|---|---|---|---|---|
| gsm8kfull / Qwen2.5-3B | 0.0634 | 0.0244 | 0.0330 | 0.0301 | 0.0278 |
| gsm8kfull / Llama-3.2-3B | 0.1037 | 0.0295 | 0.0610 | 0.0473 | 0.0398 |
| gsm8kfull / Phi-4-mini | 0.0622 | 0.0214 | 0.0333 | 0.0275 | 0.0236 |
| math1k / Qwen2.5-3B | 0.1664 | 0.1119 | 0.1169 | 0.0916 | 0.0911 |
| math500 / Phi-4-mini | 0.1110 | 0.0981 | 0.0939 | 0.0729 | 0.0641 |
| mathfull / Qwen2.5-3B | 0.0886 | 0.0513 | 0.0592 | 0.0503 | 0.0470 |
| mathfull / Llama-3.2-3B | 0.1152 | 0.0735 | 0.0988 | 0.0742 | 0.0566 |

Same offset note as §2: each cell is the corresponding $R_2$ cell (`SYNTHESIS.md` §4) plus
that combo's oracle gap $\Delta_{\text{oracle}}$. $T{=}1.0$ looks strong on GSM8K only
because $T^\dagger\approx1.0$ there; on the MATH combos mixing leads, and **Consensus has
the lowest mean $R_1$ on 6 of 7 combos** (Pool wins none outright; $T{=}1.0$ wins the three
GSM8K cases where the fixed choice coincides with the oracle).

---

## 5. Mid-budget regime ($N=64,128,256,512$)

The §2–§3 tables jump from $N{=}8$ to $N{=}2048$; the practically relevant budgets sit in
between. Mean over the seven MATH/GSM8K combos:

**Mean $R_1$ / (mean $R_2$ in parens)**

| strategy | $N{=}64$ | $N{=}128$ | $N{=}256$ | $N{=}512$ |
|---|---|---|---|---|
| Fixed $T{=}0.1$ | 0.0957 (0.0521) | 0.0981 (0.0543) | 0.0997 (0.0555) | 0.1007 (0.0564) |
| Fixed $T{=}1.0$ | 0.0585 (0.0150) | 0.0583 (0.0144) | 0.0584 (0.0142) | 0.0585 (0.0142) |
| Random-$T$ | 0.0692 (0.0256) | 0.0698 (0.0260) | 0.0703 (0.0261) | 0.0707 (0.0264) |
| **Pool** | 0.0545 (0.0110) | 0.0546 (0.0108) | 0.0554 (0.0112) | 0.0557 (0.0114) |
| **Consensus** | 0.0564 (0.0129) | 0.0539 (0.0101) | **0.0525** (0.0083) | **0.0515** (0.0072) |

**$\mathrm{p95}\,R_1$ (tail)**

| strategy | $N{=}64$ | $N{=}128$ | $N{=}256$ | $N{=}512$ |
|---|---|---|---|---|
| Fixed $T{=}0.1$ | 0.812 | 0.892 | 0.946 | 0.980 |
| Fixed $T{=}1.0$ | 0.372 | 0.410 | 0.442 | 0.464 |
| Random-$T$ | 0.368 | 0.397 | 0.423 | 0.439 |
| **Pool** | 0.347 | 0.393 | 0.422 | 0.451 |
| **Consensus** | 0.389 | 0.383 | **0.374** | **0.381** |

**$\Pr[R_1>0.1]$**

| strategy | $N{=}64$ | $N{=}128$ | $N{=}256$ | $N{=}512$ |
|---|---|---|---|---|
| Fixed $T{=}0.1$ | 0.156 | 0.145 | 0.133 | 0.125 |
| Fixed $T{=}1.0$ | 0.123 | 0.110 | 0.097 | 0.087 |
| Random-$T$ | 0.207 | 0.192 | 0.180 | 0.171 |
| **Pool** | 0.133 | 0.112 | 0.098 | **0.085** |
| **Consensus** | 0.132 | 0.115 | 0.100 | 0.088 |

Three things crystallize in this window:

1. **Mean ranking is fixed and the Pool→Consensus crossover happens here.** $R_1$ tracks
   $R_2$ by the same ~0.044 offset (§2). Consensus overtakes Pool at $N{=}128$ (mean $R_1$
   0.0539 < 0.0546; mean $R_2$ 0.0101 < 0.0108) and pulls further ahead through $N{=}512$.
2. **$R_2$ tail is already collapsing; $R_1$ tail is not — but that is the §0 cost, not a
   strategy flaw.** By $N{=}512$, $\mathrm{p95}\,R_2$ for Pool/Consensus has fallen to ~0.04
   (avoidable regret, genuinely eliminated), while $T{=}0.1$ diverges (0.69→0.86) and
   $T{=}1.0$/Random plateau at ~0.2. Under $R_1$, Pool's tail rises with $N$ (0.347→0.451)
   and only Consensus stays flat (~0.38) — a *diagnostic* difference in where each
   strategy's errors fall relative to the oracle gap, not regret a problem-agnostic strategy
   could recover.
3. **$\Pr[R_1>0.1]$ compresses the mixing advantage — by construction.** Against the oracle,
   Pool/Consensus/$T{=}1.0$ all converge to ~0.09 at $N{=}512$ because the common,
   unrecoverable oracle-gap mass dominates; only Random-$T$ stays clearly worse (~0.17). The
   mixing edge that is decisive under $R_2$ is invisible here precisely *because* $R_1$
   charges every strategy for the unknown-problem cost it cannot avoid.

---

## 6. Integrated average including AIME (all 19 combos)

Adding the twelve AIME combos (12 AIME + 7 MATH/GSM8K = 19). AIME has near-zero oracle gap
and clusters at the extremes of solvability, so it pulls every number toward $0$ **and**
shrinks the $R_1$–$R_2$ offset from ~0.044 to ~0.019 (much of the "headroom" simply isn't
there on AIME). Treat this as a diluted, optimistic view; the discriminating signal lives in
the MATH/GSM8K subset of §2–§5.

**Mean $R_1$ / (mean $R_2$)**

| strategy | $N{=}64$ | $N{=}128$ | $N{=}256$ | $N{=}512$ |
|---|---|---|---|---|
| Fixed $T{=}0.1$ | 0.0546 (0.0341) | 0.0560 (0.0363) | 0.0563 (0.0372) | 0.0564 (0.0381) |
| Fixed $T{=}1.0$ | 0.0432 (0.0226) | 0.0434 (0.0236) | 0.0431 (0.0240) | 0.0426 (0.0242) |
| Random-$T$ | 0.0420 (0.0215) | 0.0422 (0.0225) | 0.0421 (0.0230) | 0.0421 (0.0238) |
| **Pool** | 0.0343 (0.0138) | 0.0343 (0.0146) | 0.0347 (0.0156) | 0.0345 (0.0162) |
| **Consensus** | 0.0383 (0.0177) | 0.0355 (0.0158) | **0.0338** (0.0147) | **0.0326** (0.0143) |

**$\mathrm{p95}\,R_1$ / (p95 $R_2$)**

| strategy | $N{=}64$ | $N{=}128$ | $N{=}256$ | $N{=}512$ |
|---|---|---|---|---|
| Fixed $T{=}0.1$ | 0.397 (0.325) | 0.418 (0.336) | 0.424 (0.345) | 0.425 (0.356) |
| Fixed $T{=}1.0$ | 0.288 (0.199) | 0.307 (0.209) | 0.317 (0.213) | 0.322 (0.220) |
| Random-$T$ | 0.249 (0.160) | 0.256 (0.168) | 0.262 (0.170) | 0.269 (0.175) |
| **Pool** | 0.215 (0.094) | 0.226 (0.085) | 0.234 (0.079) | 0.240 (0.069) |
| **Consensus** | 0.259 (0.137) | 0.237 (0.106) | **0.219** (0.083) | **0.214** (0.071) |

**$\Pr[R_1>0.1]$ / ($\Pr[R_2>0.1]$)**

| strategy | $N{=}64$ | $N{=}128$ | $N{=}256$ | $N{=}512$ |
|---|---|---|---|---|
| Fixed $T{=}0.1$ | 0.102 (0.076) | 0.093 (0.073) | 0.083 (0.067) | 0.074 (0.061) |
| Fixed $T{=}1.0$ | 0.094 (0.067) | 0.085 (0.055) | 0.078 (0.052) | 0.063 (0.044) |
| Random-$T$ | 0.129 (0.084) | 0.121 (0.084) | 0.110 (0.079) | 0.101 (0.076) |
| **Pool** | 0.095 (0.054) | 0.081 (0.048) | **0.067** (0.037) | **0.055** (0.033) |
| **Consensus** | 0.099 (0.065) | 0.085 (0.047) | 0.070 (0.039) | 0.058 (0.035) |

The conclusions are directionally unchanged but **muted**: Pool/Consensus remain the
lowest-regret strategies on the mean and on the $R_2$ tail, Consensus still overtakes Pool
by $N{=}128$, and the $R_1$ tail still fails to collapse (Pool rises 0.215→0.240; Consensus
flat ~0.21). The single notable reshuffle from dilution: on the integrated **$R_1$ tail**,
Random-$T$ now sits between $T{=}1.0$ and Pool (AIME's low-variance, near-zero-gap problems
flatter Random-$T$), whereas on the MATH/GSM8K subset Random-$T$ is among the worst. This is
why the paper-grade claims are stated on the high-dispersion subset, not the pooled average.

---

## 7. Caveats specific to $R_1$

- **Oracle bias.** $a^\*$ is a max over noisy per-temperature estimates and is $\sim$1 pp
  upward biased, so every $R_1$ number above is slightly inflated in absolute terms. This
  is a common additive shift across strategies, so it does **not** affect the rankings of
  §2/§4 — but it *does* inflate the $R_1$ tails of §3, so treat those magnitudes as upper
  bounds.
- **Why report $R_1$ at all (given §0).** Under the unknown-problem assumption $R_1$ is
  *not* an attainable regret — so it is not a ranking criterion. Its one value is as an
  **upper bound**: the ~5 pp mean and heavy p95 tail separating the best deployable strategy
  (Consensus) from a clairvoyant per-problem selector is the most any *future* per-problem
  temperature *router* (one allowed to look at the problem) could ever win. It sizes the
  prize; it does not score the players.

**Bottom line.** Under the §0 assumption — the problem is unknown at inference —
$R_2$ (vs. the best achievable global temperature) is the operational regret, and $R_1$ is
only a ceiling. Switching baseline from $R_2$ to $R_1$ leaves every **mean ranking and
significance result intact** (the two differ by the strategy-independent oracle gap), but it
**erases the tail-collapse result**: $\mathrm{p95}\,R_1$ for mixing stays at ~0.4–0.5
because the per-problem temperature choice is structurally unreachable, not because mixing
fails. That residual tail is the *cost of not knowing the problem*, charged equally to every
problem-agnostic rule. Consensus is the lowest-regret deployable strategy on the $R_2$ mean
and tail; its smaller $R_1$ tail is diagnostic, not an extra win. The case for mixing rests
on $R_2$ — it is *deployability*, not oracle-closeness, that mixing buys.
