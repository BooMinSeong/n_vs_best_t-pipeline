# Regret Against the Per-Problem Oracle ($R_1$): Mean Ranking Survives, the Tail Does Not Vanish

*Companion to `SYNTHESIS.md`, which focuses on $R_2$ (regret vs. the best deployable
fixed temperature). Here we re-run the same comparison against the **per-problem oracle**
ceiling $R_1$, aggregated over the same dataset families. The question: does the
"mixing is regret-stable" story hold when the baseline is the unattainable per-problem
best temperature instead of the best single fixed one?*

Aggregation unit (same family scheme as `PER_MODEL_REGRET.md`): the **six gsm8k+math
model-family units** — gsm8k = `gsm8kfull` for {Qwen2.5-3B, Llama-3.2-3B, Phi-4-mini};
math = `mathfull` (Qwen2.5-3B, Llama-3.2-3B) or `math500` (Phi-4-mini), treated as one MATH
family with `math1k` excluded. §6 adds the aime family. Equal weight per unit.

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

In short: **$R_1$ leaves the means/rankings of §2 unchanged but rewrites the tail story of
§3.**

---

## 2. Mean $R_1$: same ranking as $R_2$, lifted by a near-constant oracle gap

Mean over the six gsm8k+math model-family units (gsm8k = `gsm8kfull`; math =
`mathfull`/`math500`, `math1k` excluded; equal weight):

| strategy | $\bar R_1$ @8 | @64 | @256 | @1024 | @2048 |
|---|---|---|---|---|---|
| Fixed $T{=}0.1$ | 0.0699 | 0.0858 | 0.0892 | 0.0904 | **0.0907** |
| Fixed $T{=}1.0$ | 0.0588 | 0.0487 | 0.0490 | 0.0495 | 0.0497 |
| Random-$T$ | 0.0612 | 0.0617 | 0.0628 | 0.0631 | 0.0632 |
| **Temperature Pool** | 0.0446 | 0.0489 | 0.0497 | 0.0502 | 0.0504 |
| **Temperature Consensus** | 0.0446 | 0.0501 | 0.0459 | 0.0439 | **0.0432** |

The corresponding $\bar R_2$ table (`SYNTHESIS.md` §3 currency) is **this table minus a
near-constant offset** $\overline{\Delta_{\text{oracle}}}\approx0.037$ that is the same for
every strategy (N=8: 0.040; N≥64: 0.037). Concretely at $N{=}2048$, $\bar R_2$ = Consensus
0.0062, $T{=}1.0$ 0.0127, Pool 0.0134, Random 0.0262, $T{=}0.1$ 0.0537.

So the **mean ranking is preserved between $R_1$ and $R_2$** (they differ only by the common
offset): at scale **Consensus is the lowest-regret deployable strategy** ($\bar R_1\,0.0432$),
while $T{=}1.0$ and Pool are essentially tied for second ($0.0497$ vs $0.0504$ — a swap from
the earlier 7-combo basis, where `math1k` had penalized $T{=}1.0$). Reading $R_1$ instead of
$R_2$ changes no ordering and no significance — it only inflates every number by the (slightly
bias-inflated, §8) oracle gap. Note the mean alone does *not* separate mixing from a good
fixed temperature; that separation is a tail property (§3, §7).

---

## 3. The tail does **not** collapse against the per-problem oracle

This is where $R_1$ tells a different story. In `SYNTHESIS.md` §3, the headline is that
$\mathrm{p95}\,R_2$ for Pool/Consensus **collapses to ~0.01** at $N{=}2048$ — mixing
converges onto the best fixed temperature. Under $R_1$ that collapse disappears:

| strategy | $\mathrm{p95}\,R_1$ @8 | @2048 | $\mathrm{p95}\,R_2$ @2048 |
|---|---|---|---|
| Fixed $T{=}0.1$ | 0.496 | **1.000** | 0.949 |
| Fixed $T{=}1.0$ | 0.313 | 0.400 | 0.167 |
| Random-$T$ | 0.265 | 0.422 | 0.220 |
| **Temperature Pool** | 0.239 | 0.434 | **0.013** |
| **Temperature Consensus** | 0.239 | **0.299** | **0.007** |

The mixing tail stays heavy ($\mathrm{p95}\,R_1\approx0.30$–$0.43$ at $N{=}2048$) even as
$\mathrm{p95}\,R_2\to0$. The reason is exactly $\Delta_{\text{oracle}}(p)$: on the hard
tail, the per-problem oracle rescues each problem with **its own idiosyncratic best
temperature**, which a single mixing rule cannot match. **Under the §0 assumption this is
not a regret the strategy could have avoided** — it is the structural cost of not knowing
the problem, and it does not shrink with $N$ because it is not a sampling-noise term. (The
only tail that *diverges* is $T{=}0.1$, $\mathrm{p95}\,R_1\to1.0$: it confidently converges
to the wrong mode on hard problems — a genuine, avoidable failure of that fixed choice.)

A caution on reading the tail ranking. $\mathrm{p95}\,R_1$ is the 95th percentile of
$\Delta_{\text{oracle}}(p)+R_2(p,s)$, a non-linear functional, so its ordering across
strategies need **not** match the $R_2$ tail ordering — and indeed Consensus (0.299) sits
below Pool (0.434) here, unlike their near-tie under $R_2$. But this gap is **diagnostic,
not operational**: it says Consensus's residual errors land less often on the
high-oracle-gap problems, not that Consensus recovers regret Pool leaves on the table.
Since the oracle is unreachable by assumption, neither tail magnitude is an avoidable loss;
only the §2 *mean* difference (which equals the $R_2$ ranking) should drive a deployment
choice.

Catastrophic-loss probability $\Pr[R_1>0.1]$ @ $N{=}2048$ (mean over the six gsm8k+math units):

| strategy | $\Pr[R_1>0.1]$ | (cf. $\Pr[R_2>0.1]$) |
|---|---|---|
| Random-$T$ | **14.53%** | 10.25% |
| Fixed $T{=}0.1$ | 10.27% | 8.19% |
| **Temperature Consensus** | 6.52% | 2.53% |
| **Temperature Pool** | 6.36% | 2.88% |
| Fixed $T{=}1.0$ | **6.31%** | 3.16% |

Against the per-problem oracle the catastrophic-loss rates compress: mixing still clearly
beats Random-$T$ (~6.4% vs 14.5%), but its edge over a fixed $T{=}1.0$ (6.3–6.5% vs 6.3%)
**vanishes** — because the extra ~3–4 pp of headroom that $R_1$ adds is the oracle gap that
*no* deployable strategy can claw back, so it lands on mixing and $T{=}1.0$ alike. (Under
$R_2$, by contrast, mixing's ~2.5–2.9% clearly beats $T{=}1.0$'s 3.2%.)

---

## 4. Per-unit mean $R_1$ @ $N{=}2048$ (six gsm8k+math units)

| family / model | $T{=}0.1$ | $T{=}1.0$ | Random-$T$ | **Pool** | **Consensus** |
|---|---|---|---|---|---|
| gsm8k / Qwen2.5-3B | 0.0634 | **0.0244** | 0.0330 | 0.0301 | 0.0278 |
| gsm8k / Llama-3.2-3B | 0.1037 | **0.0295** | 0.0610 | 0.0473 | 0.0398 |
| gsm8k / Phi-4-mini | 0.0622 | **0.0214** | 0.0333 | 0.0275 | 0.0236 |
| math / Qwen2.5-3B (mathfull) | 0.0886 | 0.0513 | 0.0592 | 0.0503 | **0.0470** |
| math / Phi-4-mini (math500) | 0.1110 | 0.0981 | 0.0939 | 0.0729 | **0.0641** |
| math / Llama-3.2-3B (mathfull) | 0.1152 | 0.0735 | 0.0988 | 0.0742 | **0.0566** |

Same offset note as §2: each cell is the corresponding $R_2$ cell (`SYNTHESIS.md` §4) plus
that unit's oracle gap $\Delta_{\text{oracle}}$. The split is clean: **$T{=}1.0$ wins all
three gsm8k units** (where $T^\dagger\approx1.0$ coincides with the oracle, so a single fixed
temperature is already near-optimal), while **Consensus wins all three of the harder math
units**. Pool wins none outright but trails Consensus closely. Mixing leads exactly where the
per-problem optimal temperature is dispersed (the math family), which is where a fixed choice
cannot serve both the easy bulk and the hard tail.

---

## 5. Mid-budget regime ($N=64,128,256,512$)

The §2–§3 tables jump from $N{=}8$ to $N{=}2048$; the practically relevant budgets sit in
between. Aggregated over the **six gsm8k+math model-family units** (gsm8k = `gsm8kfull`;
math = `mathfull`/`math500`, `math1k` excluded; equal weight per unit — same family scheme as
`PER_MODEL_REGRET.md`):

**Mean $R_1$ / (mean $R_2$ in parens)**

| strategy | $N{=}64$ | $N{=}128$ | $N{=}256$ | $N{=}512$ |
|---|---|---|---|---|
| Fixed $T{=}0.1$ | 0.0858 (0.0486) | 0.0879 (0.0505) | 0.0892 (0.0516) | 0.0900 (0.0525) |
| Fixed $T{=}1.0$ | 0.0487 (0.0115) | 0.0488 (0.0114) | 0.0490 (0.0115) | 0.0493 (0.0118) |
| Random-$T$ | 0.0617 (0.0244) | 0.0624 (0.0250) | 0.0628 (0.0252) | 0.0631 (0.0256) |
| **Pool** | 0.0489 (0.0116) | 0.0489 (0.0115) | 0.0497 (0.0121) | 0.0499 (0.0124) |
| **Consensus** | 0.0501 (0.0129) | 0.0476 (0.0102) | **0.0459** (0.0084) | **0.0447** (0.0073) |

**$\mathrm{p95}\,R_1$ (tail)**

| strategy | $N{=}64$ | $N{=}128$ | $N{=}256$ | $N{=}512$ |
|---|---|---|---|---|
| Fixed $T{=}0.1$ | 0.791 | 0.876 | 0.937 | 0.976 |
| Fixed $T{=}1.0$ | 0.316 | 0.345 | 0.366 | 0.379 |
| Random-$T$ | 0.349 | 0.370 | 0.393 | 0.405 |
| **Pool** | 0.318 | 0.358 | 0.373 | 0.389 |
| **Consensus** | 0.358 | 0.343 | **0.322** | **0.314** |

**$\Pr[R_1>0.1]$**

| strategy | $N{=}64$ | $N{=}128$ | $N{=}256$ | $N{=}512$ |
|---|---|---|---|---|
| Fixed $T{=}0.1$ | 0.139 | 0.128 | 0.119 | 0.111 |
| Fixed $T{=}1.0$ | 0.102 | 0.091 | 0.081 | 0.073 |
| Random-$T$ | 0.183 | 0.171 | 0.161 | 0.155 |
| **Pool** | 0.117 | 0.099 | 0.086 | **0.075** |
| **Consensus** | 0.117 | 0.101 | 0.089 | 0.077 |

Three things crystallize in this window:

1. **Mean ranking is fixed and the Pool→Consensus crossover happens here.** $R_1$ tracks
   $R_2$ by the same near-constant offset (~0.037 under this grouping, §2). Consensus overtakes
   Pool at $N{=}128$ (mean $R_1$ 0.0476 < 0.0489; mean $R_2$ 0.0102 < 0.0115) and pulls
   further ahead through $N{=}512$.
2. **$R_2$ tail is already collapsing; $R_1$ tail is not — but that is the §0 cost, not a
   strategy flaw.** By $N{=}512$, $\mathrm{p95}\,R_2$ for Pool/Consensus has fallen to ~0.03–0.04
   (avoidable regret, genuinely eliminated), while $T{=}0.1$ diverges (0.66→0.84) and
   $T{=}1.0$/Random plateau at ~0.16–0.21. Under $R_1$, Pool's tail rises with $N$ (0.318→0.389)
   and only Consensus stays flat/falling (~0.36→0.31) — a *diagnostic* difference in where each
   strategy's errors fall relative to the oracle gap, not regret a problem-agnostic strategy
   could recover.
3. **$\Pr[R_1>0.1]$ compresses the mixing advantage — by construction.** Against the oracle,
   Pool/Consensus/$T{=}1.0$ all converge to ~0.075 at $N{=}512$ because the common,
   unrecoverable oracle-gap mass dominates; only Random-$T$ stays clearly worse (~0.155). The
   mixing edge that is decisive under $R_2$ is invisible here precisely *because* $R_1$
   charges every strategy for the unknown-problem cost it cannot avoid.

---

## 6. Integrated average including AIME (nine model-family units)

Adding the **aime** family (one unit per model, the `aime2023–2026` mean) to §5's six units
gives nine equal-weight model-family units (six gsm8k+math + three aime: Phi-4, Qwen2.5,
Qwen3-4B). AIME has near-zero oracle gap and clusters at the extremes of solvability, so it
pulls every number toward $0$ **and** shrinks the $R_1$–$R_2$ offset from ~0.037 to ~0.027.
Because the family scheme caps AIME at **3 of 9 units (33%)** — rather than the 12-of-19
(63%) it would get under a raw per-combo average — this dilution is **milder** than a naive
pooled mean. Still treat the high-dispersion gsm8k+math subset of §2–§5 as the discriminating
signal.

**Mean $R_1$ / (mean $R_2$)**

| strategy | $N{=}64$ | $N{=}128$ | $N{=}256$ | $N{=}512$ |
|---|---|---|---|---|
| Fixed $T{=}0.1$ | 0.0674 (0.0402) | 0.0690 (0.0422) | 0.0698 (0.0433) | 0.0702 (0.0442) |
| Fixed $T{=}1.0$ | 0.0439 (0.0167) | 0.0441 (0.0173) | 0.0441 (0.0176) | 0.0439 (0.0179) |
| Random-$T$ | 0.0498 (0.0226) | 0.0503 (0.0235) | 0.0504 (0.0239) | 0.0505 (0.0245) |
| **Pool** | 0.0401 (0.0129) | 0.0401 (0.0133) | 0.0406 (0.0141) | 0.0407 (0.0146) |
| **Consensus** | 0.0426 (0.0154) | 0.0399 (0.0131) | **0.0383** (0.0118) | **0.0370** (0.0110) |

**$\mathrm{p95}\,R_1$ / (p95 $R_2$)**

| strategy | $N{=}64$ | $N{=}128$ | $N{=}256$ | $N{=}512$ |
|---|---|---|---|---|
| Fixed $T{=}0.1$ | 0.579 (0.477) | 0.631 (0.512) | 0.665 (0.546) | 0.684 (0.578) |
| Fixed $T{=}1.0$ | 0.290 (0.162) | 0.312 (0.171) | 0.325 (0.171) | 0.333 (0.181) |
| Random-$T$ | 0.293 (0.169) | 0.305 (0.179) | 0.318 (0.184) | 0.327 (0.189) |
| **Pool** | 0.258 (0.098) | 0.282 (0.083) | 0.290 (0.070) | 0.298 (0.053) |
| **Consensus** | 0.300 (0.141) | 0.279 (0.098) | **0.257** (0.069) | **0.248** (0.051) |

**$\Pr[R_1>0.1]$ / ($\Pr[R_2>0.1]$)**

| strategy | $N{=}64$ | $N{=}128$ | $N{=}256$ | $N{=}512$ |
|---|---|---|---|---|
| Fixed $T{=}0.1$ | 0.116 (0.090) | 0.106 (0.085) | 0.097 (0.078) | 0.089 (0.072) |
| Fixed $T{=}1.0$ | 0.093 (0.057) | 0.084 (0.047) | 0.076 (0.044) | 0.065 (0.039) |
| Random-$T$ | 0.150 (0.094) | 0.141 (0.093) | 0.130 (0.090) | 0.123 (0.088) |
| **Pool** | 0.102 (0.054) | 0.087 (0.047) | **0.074** (0.038) | **0.063** (0.034) |
| **Consensus** | 0.105 (0.063) | 0.090 (0.046) | 0.077 (0.039) | 0.065 (0.034) |

The conclusions are directionally unchanged but **muted**: Pool/Consensus remain the
lowest-regret strategies on the mean and on the $R_2$ tail, Consensus still overtakes Pool
by $N{=}128$, and the $R_1$ tail still fails to collapse (Pool rises 0.258→0.298; Consensus
flat/falling ~0.30→0.25). The notable reshuffle from dilution: on the integrated **$R_1$ tail**,
Random-$T$ rises to sit alongside $T{=}1.0$ (~0.33 at $N{=}512$), whereas on the gsm8k+math
subset (§5) Random-$T$ has the *worst* $R_1$ tail (0.405). This is why the paper-grade claims
are stated on the high-dispersion subset, not the pooled average.

---

## 7. Reporting guidance: report the mean *and* a tail measure

Regret is a per-problem **distribution**, and its mean and its tail answer different
questions — reporting only one misrepresents the result.

- **Mean regret** is the first moment: "on average, how much accuracy do I give up?" It
  averages the many easy problems (regret $\approx0$) together with the few catastrophic
  ones, so it **dilutes tail risk**. Two strategies with equal means can have very different
  worst-case behavior.
- **Tail measures** — $\mathrm{p95}$ (threshold-free) or $\Pr[R>\tau]$ (exceedance) — answer
  "how often, and how badly, do I get burned?" This is the risk a per-problem deployment
  actually faces, since queries are solved one at a time.

Why this matters here specifically: the value of temperature mixing is **risk reduction, not
higher average accuracy**. On the mean, mixing does *not* win — mean $R_2$ for Pool/Consensus
is small but positive (they slightly trail the best fixed temperature). The benefit appears
only in the tail: at $N{=}2048$, $\Pr[R_2>0.1]$ falls to ~2.7–2.9% for mixing versus ~9–11%
for a single fixed/random temperature — a catastrophic-loss rate of ~1-in-35 vs ~1-in-10.
Reporting the mean alone would make mixing look like a slight loss and miss its entire point.

Recommended default: **report mean (expected cost) alongside $\mathrm{p95}$ or the regret
CDF (risk).** Use $\Pr[R>0.1]$ as a secondary, interpretable statistic ("how often a problem
loses >10 pp"), but never as the sole headline — its $0.1$ cutoff is an arbitrary reporting
choice (sweeping $\tau\in\{0.05,0.1,0.15,0.2\}$ leaves the strategy *ranking* unchanged, but
the absolute rates — especially Random-$T$'s — shift with $\tau$). For $R_1$ in particular (§0), even the tail is partly the unavoidable
oracle gap, so a $R_1$ tail number must be read as a ceiling, not an avoidable risk.

---

## 8. Caveats specific to $R_1$

- **Oracle bias.** $a^\*$ is a max over noisy per-temperature estimates and is $\sim$1 pp
  upward biased, so every $R_1$ number above is slightly inflated in absolute terms. This
  is a common additive shift across strategies, so it does **not** affect the rankings of
  §2/§4 — but it *does* inflate the $R_1$ tails of §3, so treat those magnitudes as upper
  bounds.
- **Why report $R_1$ at all (given §0).** Under the unknown-problem assumption $R_1$ is
  *not* an attainable regret — so it is not a ranking criterion. Its one value is as an
  **upper bound**: the ~4 pp mean and heavy p95 tail separating the best deployable strategy
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
