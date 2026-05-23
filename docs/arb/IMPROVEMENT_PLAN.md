# Arbitration Model Improvement Plan

## The Core Problem

The models pass the overall MAE threshold (pitcher $893K, batter $978K, both < $1M) but fail
every per-tier precision target by a wide margin — hitting ~28–40% within tolerance vs. the 95%
goal. That target is also poorly designed: a fixed ±$230K tolerance treats a $2M player and a
$10M player identically, which is the wrong way to measure salary prediction accuracy.

This plan has four phases: pre-implementation analysis to validate assumptions, then architectural
changes, feature engineering, and target recalibration.

---

## Evaluation Metrics

The existing per-tier targets (95% within ±10% of tier average) are replaced by two metrics that
measure different things and catch different failure modes.

**MAE < $1M** is an average across all predictions: `mean(|actual − predicted|)`. It measures
the typical error size. It's sensitive to a few very large errors but says nothing about how
errors distribute across the salary range. A $1M error on a $2M player and a $1M error on a
$10M player contribute equally to MAE even though the first is a 50% miss and the second is 10%.

**Coverage: 90% of predictions within ±20% of actual salary** measures the 90th percentile of
`|actual − predicted| / actual`. This is the metric that maps directly to the user-facing claim:
"if the model projects $10M, there is a 90% chance the actual salary is between $8M and $12M."
It scales with salary magnitude, so a $2M player and a $10M player are held to the same
proportional standard.

You need both because they catch different failure modes:
- A model with every prediction exactly $1M off passes MAE but fails coverage for all
  players earning under $5M (±20% of $2M is only ±$400K).
- A model where 90% of predictions are within $100K but 10% are off by $9M barely fails
  MAE but passes coverage.

**Targets:**

| Metric | Threshold |
|--------|-----------|
| MAE | < $1.0M (keep — already passing) |
| Coverage (primary) | 90% of predictions within ±20% of actual, per tier |
| Coverage (stretch) | 90% of predictions within ±15% of actual, per tier |

---

## Phase 0: Pre-Implementation Analysis

Run these analyses before writing any model code. Each one directly tests an assumption behind
a proposed change. Results may reprioritize or eliminate steps in Phases 1–3.

All analyses go in `analysis/scripts/arb_improvement.py` with output graphs to
`analysis/graphs/arb_improvement/`.

### Analysis 1: MAPE Distribution (run first)

Compute `|actual − predicted| / actual` for every test-set prediction from the current models.
Report the 50th, 75th, 90th percentile of this distribution per tier.

**What this answers:** Where are we today on the ±20% at 90% target? How far do we have to
go? This sets the baseline and tells us whether the target is realistic with incremental
improvements or requires a fundamental rethink.

**Hypothesis to test:** Current 90th-percentile MAPE is probably 35–45% — we need to cut it
roughly in half to reach the 20% target.

### Analysis 2: Residual Analysis of the Current Model

Train the current pitcher + batter models, compute residuals `(predicted − actual)` for every
training example, then test:

- **Residuals vs. `contract_year`**: Strong correlation means the model isn't tracking market
  movement over time → market anchor features will help.
- **Residuals vs. WAR momentum** `(war_1y − war_3y / 3)`: Strong correlation means trajectory
  is an unmodeled signal → add momentum feature.
- **Residuals by position group (SP vs. RP)**: Systematically larger or biased RP residuals
  confirm the closer premium is unmodeled → `is_starter` flag or SP/RP split needed.
- **Residuals vs. predicted value (heteroskedasticity check)**: If larger predictions have
  larger errors, this directly validates the percentage-based coverage target and confirms
  the current model underperforms on high-salary players specifically.

**What this answers:** Which proposed features are likely to matter, and how severe is the
high-salary underperformance problem.

### Analysis 3: Within-Tier Variance (tests tier-specific model hypothesis)

Compute the coefficient of variation `(std / mean)` for the full dataset vs. within each tier,
separately for pitchers and batters. Also plot salary distributions within each tier as
histograms.

**What this answers:** If within-tier CV is substantially lower than cross-tier CV, separating
models by tier will concentrate each model on a narrower prediction range and reduce error. If
within-tier distributions are still heavily bimodal or fat-tailed, the gain from splitting is
limited.

**Decision rule:** If within-tier CV drops by more than 30% relative to the full-dataset CV,
proceed with tier-specific models as a high-priority change. If the drop is under 15%, it's a
lower priority and other changes should come first.

### Analysis 4: Market Anchor Correlation (tests the market reset hypothesis)

For every contract year Y, compute the prior-year tier statistics (mean, p75, p90, max) from
year Y-1. Join these back to year-Y contracts. Then compute the partial correlation between
these anchor values and actual salary, *after controlling for WAR* (i.e., the correlation
between anchor features and the residuals of a WAR-only model).

**What this answers:** Does prior-year market state add predictive signal beyond what's already
captured by performance stats? If `market_prior_p90` correlates with current salary at +0.20
or better after controlling for WAR, the feature is worth adding. If the partial correlation
is near zero, the market anchor is redundant with `contract_year`.

Also plot prior-year tier mean and p90 over time. If there are step-function jumps (e.g., 2016
TV deal, 2020 COVID dip) that `contract_year` treats as smooth, that visually confirms the
anchor feature is capturing something the linear time trend misses.

### Analysis 5: SP vs. RP Salary Distributions (tests structural split hypothesis)

Within each tier, plot salary distributions for SP, RP, and CL separately. Compute mean and
standard deviation per group per tier.

**What this answers:** If RP/CL salary within a tier is bimodal (many average closers at $1–2M,
a cluster of elite closers at $6–8M), the pitcher model is bridging a structural gap that the
`is_starter` binary feature can help address. If the SP and RP distributions overlap cleanly,
the binary flag is sufficient and a full SP/RP split is not needed.

### Analysis 6: Temporal Generalization

Train on 2011–2022, test on 2023–2025 (instead of random 80/20). Report MAE and MAPE
percentiles on the held-out years.

**What this answers:** Whether the model degrades on out-of-sample years. In production the
model will always be predicting a future year, so this is the realistic evaluation scenario.
If temporal generalization is significantly worse than random-split performance, market anchor
features become higher priority because they're specifically designed to track market evolution.

---

## Phase 1: Market Anchor Features

*Precondition: Analysis 4 shows partial correlation ≥ 0.20 after controlling for WAR.*

**The problem with `contract_year`:** The analysis found `contract_year` correlates only +0.065
with salary because arb salary growth has been essentially flat (+0.1% CAGR). But flat aggregate
growth masks real market dynamics: when a star sets a precedent, comparable players in subsequent
years cite that contract in arbitration hearings.

**`contract_year` models a linear trend. It cannot model step-function market movements.**

**New features — prior-year market anchors:**

For each contract in year Y, compute from contracts of the same player type and arb tier in Y-1:

| Feature | Description |
|---------|-------------|
| `market_prior_mean` | Mean salary, same tier + player type, prior year |
| `market_prior_p75` | 75th percentile salary, same tier + player type, prior year |
| `market_prior_p90` | 90th percentile salary, same tier + player type, prior year |
| `market_prior_max` | Maximum salary, same tier + player type, prior year |

**Implementation — time-aware join (no data leakage):**

```python
def add_market_anchors(df):
    df = df.copy()
    df['arb_year'] = df['service_time'].apply(get_tier_from_service_time)

    year_tier_stats = (
        df.groupby(['contract_year', 'arb_year', 'player_type'])['value']
        .agg(
            market_prior_mean='mean',
            market_prior_p75=lambda x: x.quantile(0.75),
            market_prior_p90=lambda x: x.quantile(0.90),
            market_prior_max='max',
        )
        .reset_index()
    )
    # Shift by one year: stats from year Y become the anchor for year Y+1
    year_tier_stats['contract_year'] += 1

    return df.merge(year_tier_stats, on=['contract_year', 'arb_year', 'player_type'], how='left')
```

Contracts from the first year in the dataset (2011) will have NaN for anchor features — handle
with median imputation in the existing preprocessor pipeline.

---

## Phase 2: Tier-Specific Models

*Precondition: Analysis 3 shows within-tier CV drops by ≥ 30% relative to full-dataset CV.*

**Current architecture:** One pitcher model + one batter model, each trained across all 3 arb
years with `service_time` as a feature.

**The problem:** A single model simultaneously solves two different tasks:
- Task A: Which tier does this player belong to? (largely answered by service_time already)
- Task B: Within that tier, where in the salary distribution do they fall?

Task A consumes model capacity that should go to Task B. The tier boundaries are known at
prediction time — there is no reason to make the model learn them.

**Proposed architecture:** Six models total — one per (player_type × arb_year) combination:

| Model | Est. training samples | Note |
|-------|-----------------------|------|
| Pitcher Year 1 | ~690 | Solid |
| Pitcher Year 2 | ~374 | Workable |
| Pitcher Year 3 | ~237 | Thin — consider Gradient Boosting |
| Batter Year 1 | ~519 | Solid |
| Batter Year 2 | ~281 | Workable |
| Batter Year 3 | ~178 | Thin — consider Gradient Boosting |

`service_time` remains a feature within each model (within Year 1, the difference between
ST 2.5 and ST 3.9 still matters). Feature correlations change per tier — `bat_home_runs_5y`
goes from 0.68 in Year 1 to 0.81 in Year 3, so tier-specific models will naturally weight
these differently.

For the two thin Year 3 models, evaluate Gradient Boosting alongside Random Forest — GBM
tends to generalize better on small datasets because it's less prone to overfitting than
deep random forests.

**Implementation:** A `TierArbModel` class that wraps six individual `ArbModel` instances,
routes inputs to the correct sub-model at prediction time, and serializes/deserializes all
six pipelines together.

---

## Phase 3: Feature Engineering

### Explicit service_time × WAR interaction

*Precondition: Analysis 2 shows this residual correlation is strong.*

The analysis found this interaction has +0.60–0.65 correlation with salary. The current model
leaves Random Forest to discover it implicitly; adding it explicitly reduces the learning burden:

```python
'st_war_interaction_pit' = service_time_normalized × pit_war_3y
'st_war_interaction_bat' = service_time_normalized × bat_war_3y
```

A player with 3 WAR in Year 3 earns more than the same player would in Year 1 — they've
proven durability and have maximum leverage entering their final arb year.

### WAR trajectory / momentum

*Precondition: Analysis 2 shows momentum correlates with residuals.*

```python
'war_momentum_pit' = pit_war_1y - (pit_war_3y / 3)
'war_momentum_bat' = bat_war_1y - (bat_war_3y / 3)
```

Positive = trending up (gets a premium), negative = declining (gets a discount). Impute
with 0 if the 3-year window is missing.

### Starter vs. reliever flag

*Precondition: Analysis 5 shows RP/CL salary distribution is structurally different from SP.*

```python
'is_starter' = (position == 'SP').astype(int)
```

Elite closers have a different WAR-to-salary ratio from starters. This binary feature gives
the model a lever to handle the closer premium without requiring a full SP/RP model split
(which would further fragment the already-thin Year 3 pitcher dataset).

---

## Implementation Sequence

1. **Phase 0** — Run all six analyses in `analysis/scripts/arb_improvement.py`. Review results
   before writing any model code.

2. **Phase 1** (if Analysis 4 confirms) — Add market anchor features to existing pitcher +
   batter models. Retrain and measure improvement in MAPE percentiles. This is the lowest-risk
   change: additive features, no structural change, clear rollback path.

3. **Phase 2** (if Analysis 3 confirms) — Implement `TierArbModel`. Retrain all six tier
   models with market anchor features included.

4. **Phase 3** — Add interaction terms, momentum, and `is_starter` flag. Ablate each one
   individually to confirm it improves the coverage metric before keeping it.

5. **Target validation** — After Phases 1–3, check whether 90% within ±20% is achievable.
   If not, Analysis 1's baseline will tell us the realistic ceiling and we can set the final
   target accordingly.

---

## Open Questions

1. **Calibrated prediction intervals for the web app:** The coverage metric tells us what
   fraction of predictions are within 20% of actual. Displaying a confidence interval to users
   (e.g., "$10M ± $2M") requires the model to *know* when it's uncertain. Random Forest point
   estimates don't do this natively — we'd need Quantile Regression Forests or conformal
   prediction applied post-hoc. This is a separate workstream from improving point estimate
   accuracy, and should be scoped separately once the model accuracy targets are met.

2. **Batter and Pitcher Year 3 thin data:** ~178 and ~237 training samples respectively.
   If Gradient Boosting outperforms Random Forest on these tiers (Analysis 0 can test both),
   we may want a hybrid architecture: RF for Year 1/2, GBM for Year 3.

3. **2011 market anchor gap:** No prior-year market data exists for 2011 contracts (~200 rows).
   Options: drop 2011 from training, or impute anchors with 2011 actuals (bounded circular
   dependency but unlikely to cause meaningful leakage at the tail of history).

4. **Should RP be split from SP entirely?** Full split doubles the models (12 total) and
   creates very small RP-Year-3 training sets (~50 samples). The `is_starter` binary in Phase 3
   is the softer approach — evaluate it before committing to a full split.
