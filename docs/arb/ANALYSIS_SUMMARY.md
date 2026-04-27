# Arbitration Model Analysis Summary

## Dataset Overview

- **Total arbitration contracts:** 2,850 (single-year only)
- **Pitchers:** 1,627 (57%)
- **Position players:** 1,223 (43%)
- **Service time range:** 2.16 - 5.99 years

### Super Two Players

**339 contracts** have service time < 3.0 years. These are "Super Two" players - those with 2+ years of service time who rank in the top ~22% of their service time class and qualify for an extra year of arbitration.

Super Two players enter arbitration a year earlier than typical players, giving them 4 arb years instead of 3. In our dataset, Super Two contracts show up with service times from ~2.16 to 3.0.

### By Arbitration Year:

| Arb Year | Service Time | Count | Avg Salary | Std Dev | Notes |
|----------|--------------|-------|------------|---------|-------|
| Year 1   | ST < 4.0     | 1,511 | $2.25M     | $1.67M  | Includes Super Two (ST 2-3) + regular first-timers (ST 3-4) |
| Year 2   | 4.0 ≤ ST < 5 | 819   | $4.00M     | $2.90M  | |
| Year 3   | 5.0 ≤ ST < 6 | 520   | $6.38M     | $4.68M  | Final arb year before free agency |

*Note: No contracts with ST ≥ 6 in dataset (would be free agents).*

### Detailed Service Time Breakdown:

| ST Bucket | Count | Mean Salary | Notes |
|-----------|-------|-------------|-------|
| 2.5-3.0   | 394   | $2.00M      | Super Two (first arb year) |
| 3.0-3.5   | 570   | $1.94M      | Regular first arb year |
| 3.5-4.0   | 589   | $2.87M      | Late first arb year |
| 4.0-4.5   | 378   | $3.37M      | Second arb year |
| 4.5-5.0   | 415   | $4.67M      | Late second arb year |
| 5.0-5.5   | 255   | $5.41M      | Third arb year |
| 5.5-6.0   | 248   | $7.30M      | Late third arb year |

Salaries increase consistently with service time, showing ~$1-2M jumps between arb years.

---

## Key Feature Correlations

### Position Players Only (n=1,223):

Correlations computed separately for each arb year, then averaged:

| Feature              | Year 1 | Year 2 | Year 3 | Notes                    |
|----------------------|--------|--------|--------|--------------------------|
| `bat_rbis_3y`        | +0.743 | +0.749 | +0.778 | Consistently strong      |
| `bat_war_3y`         | +0.741 | +0.746 | +0.760 | Key WAR metric           |
| `bat_home_runs_3y`   | +0.730 | +0.748 | +0.793 | **Strongest in Year 3!** |
| `bat_runs_3y`        | +0.741 | +0.763 | +0.776 |                          |
| `bat_war_5y`         | +0.680 | +0.770 | +0.775 | Longer window helps      |
| `bat_home_runs_5y`   | +0.680 | +0.769 | +0.807 | **Strongest overall!**   |

### Pitchers Only (n=1,627):

| Feature              | Year 1 | Year 2 | Year 3 | Notes                    |
|----------------------|--------|--------|--------|--------------------------|
| `pit_war_3y`         | +0.769 | +0.841 | +0.867 | **Strongest predictor**  |
| `pit_war_5y`         | +0.818 | +0.839 | +0.841 | Very strong              |
| `pit_war_1y`         | +0.686 | +0.744 | +0.697 | Recent performance       |
| `pit_strikeouts_3y`  | +0.724 | +0.760 | +0.769 |                          |
| `pit_strikeouts_5y`  | +0.719 | +0.766 | +0.752 |                          |
| `pit_wins_3y`        | +0.640 | +0.683 | +0.708 |                          |

### Key Insights:
1. **Longer windows (3y, 5y) strongly outperform 1-year** for both groups
2. **pit_war_3y is the strongest single predictor** (+0.87 in Year 3)
3. **Home runs and RBIs are top predictors** for position players
4. **WAR is consistently strong** across both player types
5. **Correlations increase with arb year** - later years show stronger stat-salary relationships

### Rate Stats vs Counting Stats

An important finding: **counting stats correlate significantly better with salary than rate stats**.

| Stat Type | Best Correlation | Example |
|-----------|------------------|---------|
| **Counting** | +0.82 | bat_rbis_5y, bat_home_runs_5y |
| **Rate** | +0.58 | bat_wrc_plus_3y |

Rate stats examined (all showed weaker correlations):
- `bat_wrc_plus_3y`: +0.58
- `bat_slugging_pct_5y`: +0.52
- `bat_ops_3y`: +0.50
- `bat_batting_avg_3y`: +0.31

**Why counting stats predict salary better:**

1. **Counting stats capture playing time** - Players who stay healthy and play 150+ games accumulate more HR, RBI, WAR
2. **Durability is valuable** - Teams pay for players who contribute consistently over a full season
3. **Star players both play more AND perform better** - Counting stats capture both performance and availability
4. **Rate stats normalize away playing time** - A .300 hitter in 100 games isn't necessarily more valuable than a .280 hitter in 162 games

This is a known phenomenon in baseball analytics - counting stats often predict salary better because they reflect both performance AND availability. For this reason, the model uses counting stats (WAR, HR, RBI, strikeouts) rather than rate stats (AVG, OPS, wRC+).

---

## Salary Inflation Analysis

Examining how arbitration salaries have evolved from 2011-2026:

### Compound Annual Growth Rate (CAGR) by Tier:

| Arb Year | Period | Start Avg | End Avg | CAGR |
|----------|--------|-----------|---------|------|
| Year 1   | 2011-2026 | $2.33M | $2.36M | **+0.1%/year** |
| Year 2   | 2011-2026 | $4.88M | $4.52M | **-0.5%/year** |
| Year 3   | 2012-2026 | $6.83M | $6.03M | **-0.9%/year** |

### Key Findings:

1. **Essentially flat growth** - Arb salaries have not kept pace with general inflation
2. **COVID impact visible** - 2020-2021 shows a notable dip (-15% overall)
3. **Pre-COVID growth** - 2017-2019 showed stronger growth (+7-10%/year)
4. **Post-COVID stagnation** - Recovery has been slow (~1-3%/year)

### Correlation with Contract Year:

| Tier | Correlation |
|------|-------------|
| Year 1 | +0.022 |
| Year 2 | +0.068 |
| Year 3 | +0.069 |
| **Overall** | **+0.065** |

The weak correlation (+0.065) suggests that `contract_year` is **not a strong predictor** of salary. Year-over-year variation in arb salaries is driven more by player performance than by market inflation.

**Implication for model:** While `contract_year` is included as a feature, its predictive power is limited. The model should not rely heavily on this feature.

---

## Personal Feature Analysis

### Age:
- **Correlation with salary:** -0.28 to -0.33 (negative across all tiers)
- Interpretation: Older arb players get paid less (performance tends to decline)

### Service Time (within tier):
- **Correlation with salary:** +0.19 to +0.25 (weak positive)
- Days of service within an arb year don't significantly impact salary

### Service Time × WAR Interaction:
- **Correlation with salary:** +0.60 to +0.65 (strong)
- Combining service time with performance is highly predictive
- Recommendation: Consider adding interaction term to model

---

## Final Feature Selection

Based on analysis findings:

### Personal Features:
- `age` - negatively correlated, captures performance decline
- `service_time` - determines arb tier
- `contract_year` - captures inflation/market trends
- `position` - one-hot encoded (different markets by position)

### Batting Stats (for position players):
- `bat_war_3y` (+0.74-0.76) - key WAR metric
- `bat_war_1y` - recent performance
- `bat_home_runs_3y` (+0.73-0.79)
- `bat_home_runs_5y` (+0.68-0.81) - strongest in later years
- `bat_rbis_3y` (+0.74-0.78)

### Pitching Stats (for pitchers):
- `pit_war_3y` (+0.77-0.87) - strongest predictor
- `pit_war_5y` (+0.82-0.84) - very strong
- `pit_war_1y` (+0.69-0.74) - recent performance
- `pit_strikeouts_3y` (+0.72-0.77)
- `pit_strikeouts_1y` (+0.68-0.69)

**Note:** ERA and FIP excluded due to weak predictive power. Longer windows (3y, 5y) are prioritized over 1y based on correlation analysis.

---

## Model Architecture Recommendations

1. **Single unified model** - Position can be a feature (one-hot encoded) rather than building separate pitcher/position player models

2. **RandomForest** preferred - Handles mixed feature types well, provides feature importances

3. **Tiered evaluation** - Different tolerance thresholds per arb year:
   - Year 1: ±$0.23M (10% of $2.25M avg)
   - Year 2: ±$0.40M (10% of $4.00M avg)
   - Year 3: ±$0.64M (10% of $6.38M avg)

4. **Consider interaction terms** if base model underperforms - service_time × WAR shows +0.60-0.65 correlation

---

## Graphs Generated

- `analysis/graphs/arb_feature_correlations_batting.png` - Batting feature correlations by arb year (position players only)
- `analysis/graphs/arb_feature_correlations_pitching.png` - Pitching feature correlations by arb year (pitchers only)
- `analysis/graphs/arb_age_salary_interaction.png` - Age effects on salary
- `analysis/graphs/arb_salary_by_position.png` - Position breakdown analysis
- `analysis/graphs/arb_service_time_analysis.png` - Service time deep dive
- `analysis/graphs/arb_salary_inflation.png` - Year-over-year salary inflation trends
