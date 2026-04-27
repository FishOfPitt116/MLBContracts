# Arbitration Salary Prediction Model

## Overview

This model predicts annual salaries for MLB players in their arbitration years (service time 2-6 years). Unlike pre-arb salaries which cluster tightly around the league minimum, arbitration salaries vary widely based on player performance, making this a more challenging prediction problem.

The model uses **separate pitcher and batter models** because:
1. Pitchers and position players have completely different stat profiles (pitching vs batting stats)
2. A unified model would have half its features as NaN for each prediction
3. Analysis showed pitching stats correlate more strongly with salary (+0.77-0.87) than batting stats (+0.73-0.81)

## Model Performance

### Pitcher Model

| Metric | Value |
|--------|-------|
| Mean Absolute Error (MAE) | $893K |
| Root Mean Squared Error (RMSE) | $1.44M |
| R² Score | 0.68 |
| Cross-validation MAE | $925K ± $90K |
| Training samples | 1,301 |
| Test samples | 326 |

#### Per-Tier Performance (Pitcher)

| Arb Year | Test N | Avg Salary | MAE | % Within ±10% |
|----------|--------|------------|-----|---------------|
| Year 1 | 173 | $2.15M | $601K | 39.9% |
| Year 2 | 95 | $3.60M | $1.07M | 28.4% |
| Year 3 | 58 | $5.85M | $1.47M | 39.7% |

### Batter Model

| Metric | Value |
|--------|-------|
| Mean Absolute Error (MAE) | $978K |
| Root Mean Squared Error (RMSE) | $1.73M |
| R² Score | 0.84 |
| Cross-validation MAE | $858K ± $47K |
| Training samples | 978 |
| Test samples | 245 |

#### Per-Tier Performance (Batter)

| Arb Year | Test N | Avg Salary | MAE | % Within ±10% |
|----------|--------|------------|-----|---------------|
| Year 1 | 133 | $2.41M | $498K | 39.8% |
| Year 2 | 67 | $4.72M | $991K | 35.8% |
| Year 3 | 45 | $8.62M | $2.38M | 33.3% |

### Error Distribution (Pitcher Model)

| Percentile | Error |
|------------|-------|
| 50th | $480K |
| 75th | $1.11M |
| 90th | $2.15M |
| 95th | $3.34M |
| 99th | $5.29M |

## Features

### Pitcher Model (8 features)

| Feature | Correlation | Description |
|---------|-------------|-------------|
| `pit_war_5y` | +0.82-0.84 | **Most important** - 5-year cumulative WAR |
| `pit_war_3y` | +0.77-0.87 | 3-year cumulative WAR |
| `pit_war_1y` | +0.69-0.74 | Recent single-year WAR |
| `pit_strikeouts_3y` | +0.72-0.77 | 3-year strikeout total |
| `pit_strikeouts_1y` | +0.68-0.69 | Recent strikeouts |
| `service_time` | +0.19-0.25 | Determines arb tier |
| `age` | -0.28-0.33 | Older players paid less |
| `contract_year` | +0.07 | Weak inflation signal |

### Batter Model (9 features)

| Feature | Correlation | Description |
|---------|-------------|-------------|
| `bat_home_runs_5y` | +0.68-0.81 | **Strongest in Year 3** |
| `bat_home_runs_3y` | +0.73-0.79 | 3-year home run total |
| `bat_rbis_3y` | +0.74-0.78 | 3-year RBI total |
| `bat_war_3y` | +0.74-0.76 | 3-year cumulative WAR |
| `bat_war_1y` | +0.53 | Recent single-year WAR |
| `position` | varies | One-hot encoded position |
| `service_time` | +0.19-0.25 | Determines arb tier |
| `age` | -0.28-0.33 | Older players paid less |
| `contract_year` | +0.07 | Weak inflation signal |

### Why Counting Stats Over Rate Stats?

Analysis showed counting stats correlate significantly better with salary:

| Stat Type | Best Correlation | Example |
|-----------|------------------|---------|
| **Counting** | +0.82 | bat_rbis_5y, bat_home_runs_5y |
| **Rate** | +0.58 | bat_wrc_plus_3y |

Counting stats capture both performance AND playing time. A player who hits .280 over 162 games is often more valuable than one who hits .300 over 100 games. Durability matters in arbitration negotiations.

### Feature Importance (Pitcher Model)

| Feature | Importance |
|---------|------------|
| `pit_war_5y` | 47.4% |
| `service_time` | 22.0% |
| `pit_war_3y` | 15.3% |
| `pit_strikeouts_1y` | 3.4% |
| `contract_year` | 3.4% |
| `pit_strikeouts_3y` | 3.4% |
| `pit_war_1y` | 2.7% |
| `age` | 2.4% |

## Algorithm: Random Forest

Random Forest was chosen over linear models for arbitration prediction because:

1. **Non-linear relationships** - Salary doesn't scale linearly with WAR; star players command premium multipliers
2. **Feature interactions** - Service time × performance interactions are automatically captured
3. **Handles mixed features** - Numeric stats + categorical position work naturally together
4. **No extrapolation needed** - Unlike pre-arb (which must predict future CBA minimums), arb salaries are driven by individual performance within known ranges

### Why Not Gradient Boosting?

Both models were evaluated. Random Forest showed comparable or better performance with more stable cross-validation results.

## Training Data

### Filters Applied

```python
contract_type == "arb"     # Arbitration contracts only
duration == 1              # Single-year contracts only
service_time.notna()       # Must have valid service time
position in PITCHER_POSITIONS  # For pitcher model: SP, RP, CL, P
position not in PITCHER_POSITIONS  # For batter model: all others
```

### Dataset Statistics

| Player Type | Total | Training | Test |
|-------------|-------|----------|------|
| Pitchers | 1,627 | 1,301 | 326 |
| Batters | 1,223 | 978 | 245 |
| **Combined** | **2,850** | **2,279** | **571** |

### By Arbitration Year

| Arb Year | Service Time | Count | Avg Salary | Std Dev |
|----------|--------------|-------|------------|---------|
| Year 1 | ST < 4.0 | 1,511 | $2.25M | $1.67M |
| Year 2 | 4.0 ≤ ST < 5 | 819 | $4.00M | $2.90M |
| Year 3 | 5.0 ≤ ST < 6 | 520 | $6.38M | $4.68M |

### Super Two Players

339 contracts (12%) have service time < 3.0 years. These are "Super Two" players who rank in the top ~22% of their service class and qualify for an extra arbitration year. They're included in Year 1 predictions.

## Usage

### Training

```bash
# Train both pitcher and batter models (recommended)
make train-arb

# Train individual models
make train-arb-pitcher
make train-arb-batter

# Or via Python directly
python -m models.arb.train --player-type all --save
python -m models.arb.train --player-type pitcher --save
python -m models.arb.train --player-type batter --save

# Compare Random Forest vs Gradient Boosting
python -m models.arb.train --player-type pitcher --compare
```

### Inspection

```bash
# View model performance and sample predictions
python -m models.arb.inspect --player-type pitcher
python -m models.arb.inspect --player-type batter
python -m models.arb.inspect --player-type all
```

### Programmatic Usage

```python
from models.arb.model import ArbModel
from models.arb.inspect import predict_for_pitcher, predict_for_batter

# Load trained models
pitcher_model = ArbModel.load(player_type="pitcher")
batter_model = ArbModel.load(player_type="batter")

# Predict for a pitcher
salary, arb_year = predict_for_pitcher(
    pitcher_model,
    age=27,
    service_time=4.000,  # Arb Year 2
    contract_year=2026,
    pit_war_1y=3.0,
    pit_war_3y=7.0,
    pit_war_5y=10.0,
    pit_strikeouts_1y=180,
    pit_strikeouts_3y=500,
)
print(f"Predicted: ${salary:.2f}M (Year {arb_year})")

# Predict for a batter
salary, arb_year = predict_for_batter(
    batter_model,
    age=26,
    service_time=4.050,  # Arb Year 2
    position="SS",
    contract_year=2026,
    bat_war_1y=3.5,
    bat_war_3y=8.0,
    bat_home_runs_3y=55,
    bat_home_runs_5y=85,
    bat_rbis_3y=200,
)
print(f"Predicted: ${salary:.2f}M (Year {arb_year})")
```

## Artifacts

Trained model artifacts are saved to `models/artifacts/`:

| File | Description |
|------|-------------|
| `arb_pitcher_model.pkl` | Serialized sklearn pipeline for pitchers |
| `arb_pitcher_metrics.json` | Pitcher model evaluation metrics |
| `arb_batter_model.pkl` | Serialized sklearn pipeline for batters |
| `arb_batter_metrics.json` | Batter model evaluation metrics |

## Accuracy Analysis

### Current Status

The models achieve reasonable overall MAE (~$900K) but fall short of the ambitious ±10% per-tier tolerance target:

| Target | Threshold | Pitcher | Batter | Status |
|--------|-----------|---------|--------|--------|
| Overall MAE | < $1.0M | $893K | $978K | **PASS** |
| Year 1 within ±$230K | 95% | 39.9% | 39.8% | FAIL |
| Year 2 within ±$400K | 95% | 28.4% | 35.8% | FAIL |
| Year 3 within ±$640K | 95% | 39.7% | 33.3% | FAIL |

### Why the Tier Targets Are Difficult

The ±10% tolerance targets (e.g., ±$230K for Year 1) are extremely tight:

1. **High salary variance** - Year 1 salaries range from $600K to $7.9M despite similar service time
2. **Unpredictable outliers** - Star closers (Chapman, Hader) and injury-bounce-back pitchers create large errors
3. **Negotiation dynamics** - Final salary depends on team budget, agent skill, and comparable contracts
4. **Data limitations** - We lack features like team payroll, arbitration hearing history, and contract leverage

### Worst Predictions Analysis

The largest errors come from predictable categories:

| Category | Example | Actual | Predicted | Error |
|----------|---------|--------|-----------|-------|
| Bounce-back year | Paxton 2020 | $12.5M | $5.4M | $7.1M |
| Elite closer premium | Chapman 2014 | $7.9M | $2.6M | $5.3M |
| Injury discount | Wood 2018 | $1.5M | $5.6M | $4.1M |
| Elite starter extension | Fried 2023 | $13.5M | $10.1M | $3.4M |

These outliers represent team-specific valuations and circumstances not captured in aggregate statistics.

### Realistic Accuracy Expectations

Given the inherent variance in arbitration negotiations:

| Metric | Current | Realistic Target |
|--------|---------|------------------|
| MAE | $0.9M | $0.7-0.8M |
| % within ±$500K | 50% | 60-70% |
| % within ±$1.0M | 72% | 80-85% |

## Potential Improvements

Per the design document, if targets aren't met:

1. **Add interaction terms** - `service_time × WAR` showed +0.60-0.65 correlation
2. **Try Gradient Boosting** - May capture non-linearities better
3. **Add more features** - `pit_wins_3y`, `bat_runs_3y` from analysis
4. **Adjust targets** - Consider ±15-20% tolerance or 80% threshold

## Limitations

1. **Outlier contracts** - Players with unique circumstances (injury history, elite closers, trade leverage) will be mispredicted

2. **Multi-year deals excluded** - Only single-year arbitration contracts are used; players who sign extensions are not in the training data

3. **Super Two complexity** - Super Two eligibility depends on service time ranking within a year, which we don't model

4. **No team context** - Team payroll, market size, and organizational philosophy affect negotiations but aren't included

5. **Historical bias** - Model trained on 2011-2025 data may not reflect future market changes

## Salary Inflation

Analysis found minimal salary inflation in arbitration:

| Arb Year | CAGR (2011-2026) |
|----------|------------------|
| Year 1 | +0.1%/year |
| Year 2 | -0.5%/year |
| Year 3 | -0.9%/year |

Unlike free agency, arbitration salaries have remained largely flat, with COVID causing a notable dip in 2020-2021. The `contract_year` feature has only +0.065 correlation with salary.

## Related Documentation

- [Analysis Summary](ANALYSIS_SUMMARY.md) - Detailed feature correlation analysis
- [Design Document](DESIGN.md) - Model architecture decisions
- [Pre-Arb Model](../pre_arb/README.md) - Companion model for pre-arbitration salaries
