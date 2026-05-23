# Arbitration Model Improvement Analysis Results

This document summarizes the findings from Phase 0 of the improvement plan
(`docs/arb/IMPROVEMENT_PLAN.md`) and determines which action items are
supported by the data. **Do not proceed to implementation without reviewing
this document first.**

---

## Analysis 1: MAPE Baseline

**What we measured:** Current 90th-percentile MAPE (the gap to the ±20% at
90% coverage target), per player type and tier.

**Results:**

| Model | 90th-pct MAPE | % within ±20% |
|-------|--------------|----------------|
| Pitcher (all tiers) | **63.8%** | 47.0% |
| Pitcher Year 1 | 58.5% | 48.6% |
| Pitcher Year 2 | 97.2% | 38.9% |
| Pitcher Year 3 | 53.9% | 53.4% |
| Batter (all tiers) | **65.5%** | 51.0% |
| Batter Year 1 | 65.9% | 58.6% |
| Batter Year 2 | 54.8% | 47.8% |
| Batter Year 3 | 70.9% | 46.7% |

**Interpretation:** The 90th-percentile MAPE is 64–66% vs. the 20% target. To hit the target we
would need to cut the 90th-percentile error by roughly **3×**. Two models achieving MAE < $1M
on a salary range of $600K–$13M will not produce 90th-percentile MAPE of 20% — the inherent
variance in arbitration negotiations (agent leverage, team budget, comparable contract timing)
makes this mathematically implausible with performance stats alone.

**Action item: Recalibrate accuracy targets.** The ±20% at 90% coverage target must be
revised before implementation. See the Recommendations section for proposed replacements.

---

## Analysis 2: Residual Analysis

**What we measured:** Whether proposed new features (WAR momentum, ST×WAR interaction) have
signal in the current model's residuals; whether errors scale with predicted salary
(heteroskedasticity).

**Results:**

| Signal | Pitcher | Batter |
|--------|---------|--------|
| Residual ~ `contract_year` | r=−0.092, p=0.096 (not sig.) | r=−0.220, p=0.0005 **sig.** |
| Residual ~ WAR momentum | r=+0.021, p=0.746 (not sig.) | r=+0.046, p=0.499 (not sig.) |
| Residual ~ ST×WAR interaction | r=−0.025, p=0.698 (not sig.) | r=−0.151, p=0.027 (marginal) |
| \|Residual\| ~ predicted value | r=+0.315, p≈0 **sig.** | r=+0.597, p≈0 **sig.** |

**Key findings:**

1. **Heteroskedasticity is real and strong** — both models produce errors that grow
   significantly with the predicted salary. This is the most important finding from
   Analysis 2. For batters, the correlation is r=+0.597, which is very strong. The model is
   systematically worst on the high-value players that matter most for the web app.
   This directly validates using a percentage-based error metric (MAPE) rather than a
   fixed-dollar tolerance — it's not a modeling choice but an empirical description of how
   this model's errors are distributed.

2. **WAR momentum adds no signal** — r=+0.02 to +0.05, both insignificant. This proposed
   feature from the improvement plan should be **dropped**.

3. **ST×WAR interaction adds marginal signal for batters only** — r=−0.151 at p=0.027.
   This is borderline significant and the effect is small. The improvement plan proposed
   adding this feature; the evidence is weak and it should be **low priority**.

4. **Contract year is a significant residual predictor for batters** (r=−0.220,
   p=0.0005) — the batter model's errors are correlated with time, suggesting it is not
   capturing market evolution. This supports the market anchor feature for batters.
   For pitchers, the relationship is not significant.

---

## Analysis 3: Within-Tier Variance

**What we measured:** Whether splitting models by arb year would meaningfully reduce the
prediction problem's difficulty, tested by comparing the coefficient of variation (CV = std/mean)
full-dataset vs. within each tier.

**Results:**

| Player Type | Full-dataset CV | Avg within-tier CV | Avg CV reduction |
|-------------|-----------------|-------------------|-----------------|
| Pitcher | 0.860 | 0.684 | **20.4%** |
| Batter | 0.928 | 0.740 | **20.2%** |

**Decision rule from the plan:** Proceed with tier-specific models if CV drop > 30%.

**Interpretation:** The ~20% CV reduction falls short of the 30% threshold. The reason is
that salary variance is high *within* every tier — not just across tiers. Within Year 1
pitchers (n=855), salaries still range from the league minimum to $7M+. The model already
uses `service_time` as a feature to capture within-tier position, so splitting models would
add architectural complexity without proportionate benefit.

**Action item: Deprioritize tier-specific models.** The improvement plan called this
the "highest impact" change; the data says otherwise. The ~20% CV reduction means
tier-specific models would reduce the inherent scatter only modestly. Do not implement
this architecture change before the higher-confidence improvements below.

---

## Analysis 4: Market Anchor Correlation

**What we measured:** Whether prior-year tier salary statistics (mean, p75, p90, max) add
predictive signal *beyond what WAR already captures*, using partial correlation.

**Results (partial correlation controlling for WAR):**

| Anchor Feature | Pitcher partial r | Batter partial r |
|----------------|-------------------|-----------------|
| `market_prior_mean` | **+0.555** | **+0.499** |
| `market_prior_p75` | **+0.537** | **+0.491** |
| `market_prior_p90` | **+0.553** | **+0.506** |
| `market_prior_max` | **+0.481** | **+0.471** |

All p-values are on the order of 10⁻⁶⁰ to 10⁻⁹⁴ — overwhelming significance.

**Interpretation:** This is the **strongest finding in the entire analysis.** After
controlling for WAR (the dominant predictor), prior-year market anchors explain an additional
~0.5 in partial correlation for both models. This is not a marginal effect — it's nearly as
strong as WAR itself. The market state in the prior year is a powerful predictor of what
players earn in the current year, independent of performance.

Why does this happen? Arbitration salaries are negotiated with reference to comparable
contracts from recent years. When the market for a position or tier shifts (e.g., a new
TV deal, post-COVID recovery, or a landmark player contract), all subsequent negotiations
in that tier are anchored to the new level. `contract_year` alone can't capture this because
it models only linear time trend; the market anchors capture the actual realized salary level.

**Action item: Implement market anchor features.** This is the highest-confidence, most
impactful change available. All four anchor variants are significant; start with
`market_prior_mean` and `market_prior_p90` as the most informative pair.

---

## Analysis 5: SP vs. RP Salary Distributions

**What we measured:** Whether starters and relievers have structurally different salary
distributions within the pitcher model, using mean comparison and the
Kolmogorov-Smirnov test.

**Results:**

| Tier | SP mean | RP mean | Difference | KS test |
|------|---------|---------|------------|---------|
| Year 1 | $2.60M | $1.67M | **+$0.92M** | p≈0 — significantly different |
| Year 2 | $4.61M | $2.96M | **+$1.65M** | p≈0 — significantly different |
| Year 3 | $7.08M | $4.63M | **+$2.45M** | p≈0 — significantly different |

**Interpretation:** Starters earn roughly $1–2.5M more than relievers in every tier. The
distributions are statistically distinct (KS p≈0 in all three tiers). Critically, the
current pitcher model has **no position feature at all** — it treats an SP and an RP with
identical stats identically. This is a structural gap. The $1.65M Year 2 mean difference
alone is larger than the model's entire MAE threshold.

**Action item: Add role feature to pitcher model.** This is a clear, high-confidence
improvement. The simplest implementation is a binary `is_starter` feature
(`position == "SP"`). A three-way encoding (SP / RP / CL) is also worth considering
given that closers may have different salary dynamics than setup relievers, though the CL
sample sizes are small.

---

## Analysis 6: Temporal Generalization

**What we measured:** Whether the model degrades on held-out future years
(train 2011–2022, test 2023–2025) vs. the random 80/20 split.

**Results:**

| Model | Temporal MAE | Random MAE | Temporal P90 MAPE | Random P90 MAPE |
|-------|-------------|------------|-------------------|-----------------|
| Pitcher | $0.891M | $0.893M | 72.2% | 63.8% |
| Batter | $0.937M | $0.978M | 48.1% | 65.5% |

**Interpretation:** MAE is essentially identical between temporal and random splits for both
models — no meaningful degradation. The pitcher P90 MAPE worsens by +8.4pp temporally
(72% vs 64%), but the batter P90 MAPE actually *improves* by 17pp in the temporal test,
suggesting the random-split error for batters was inflated by high-variance early years in
the test set.

**Action item: No urgent changes required for temporal generalization.** The model is not
systematically failing to generalize across time on the MAE metric. The market anchor
features are still recommended (Analysis 4 shows strong signal), but temporal degradation
is not the primary motivation for them — the main motivation is the market-reset mechanism.

---

## Recommendations

### Action items to pursue (in priority order)

**1. Add market anchor features** — the data overwhelmingly supports this
(partial r ≈ 0.5 for all variants, p < 10⁻⁶⁰). Implement the time-aware join described
in the improvement plan. Start with `market_prior_mean` and `market_prior_p90`. This is
the single highest-expected-impact change.

**2. Add role feature to pitcher model** — SP and RP salary distributions are
significantly different (KS p≈0, mean gap $0.9–2.5M per tier), and the pitcher model
currently has no position feature. Add `is_starter` (binary) or a three-category
encoding (SP / RP / CL) to `PITCHER_PERSONAL_FEATURES` in config.

**3. Recalibrate accuracy targets** — the current 90th-percentile MAPE is 64–66%.
The improvement plan's target of 90% within ±20% (i.e., 90th-percentile MAPE ≤ 20%)
is not achievable with performance stats alone given the inherent negotiation variance.
After implementing improvements 1 and 2, the realistic targets are:

| Metric | Current | Realistic Post-Improvement |
|--------|---------|---------------------------|
| MAE | ~$900K | < $700K |
| % within ±20% (per tier) | 39–59% | 55–65% |
| % within ±30% (per tier) | ~60–70% | 70–80% |
| 90th-pct MAPE (per tier) | 54–97% | 40–60% |

The "90% within ±20%" target from the improvement plan should be formally retired and
replaced with **75% within ±25%** as the primary coverage target, with **MAE < $700K**
as the overall threshold.

### Action items to deprioritize or drop

**Tier-specific models** — CV reduction is ~20%, below the 30% threshold. The within-tier
salary variance is still very high, meaning six separate models would add significant
complexity for modest accuracy gain. Deprioritize. Revisit only if improvements 1 and 2
fall short.

**WAR momentum feature** — residual correlation r=+0.02 to +0.05, not significant.
Drop from the plan.

**ST×WAR interaction feature** — marginal signal for batters (r=−0.15, p=0.027), none
for pitchers. Low value-to-complexity ratio. Drop from the plan.

**Full SP/RP model split** — a binary `is_starter` feature is sufficient given the modest
CL sample sizes (especially in Year 3). The KS test result supports the signal, but a
full model split is unnecessary when a single feature addresses it.

---

## Summary Table

| Improvement Plan Item | Evidence | Decision |
|-----------------------|----------|----------|
| Market anchor features | Very strong (partial r≈0.5, p<10⁻⁶⁰) | **Implement** |
| SP/RP role feature for pitchers | Strong (KS p≈0, $1–2.5M gap) | **Implement** |
| Recalibrate accuracy targets | 64–66% current vs 20% target — gap is structural | **Recalibrate** |
| Tier-specific models | Weak (20% CV drop vs 30% threshold) | **Deprioritize** |
| WAR momentum feature | No signal (r≈0.02–0.05) | **Drop** |
| ST×WAR interaction | Marginal batters only (r=−0.15) | **Drop** |
| Full SP/RP model split | Unnecessary given `is_starter` flag | **Drop** |
| Temporal generalization fix | Not degrading (MAE flat across time) | **Not needed** |
