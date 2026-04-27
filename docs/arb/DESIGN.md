# Arbitration Salary Prediction Model - Design Document

## Goal
Build a model to predict single-year arbitration contract values with 95% of predictions within 10% of the average salary for each arb year.

## Accuracy Targets (95% within tolerance per tier)

| Arb Year | Avg Salary | Tolerance (10%) | Count |
|----------|------------|-----------------|-------|
| Year 2   | $1.91M     | ±$0.19M         | 339   |
| Year 3   | $2.35M     | ±$0.24M         | 1,172 |
| Year 4   | $4.00M     | ±$0.40M         | 819   |
| Year 5   | $6.38M     | ±$0.64M         | 520   |

**Secondary metrics:** MAE, RMSE, R², MAPE (for overall model health)

---

## Phase 1: Comprehensive Data Analysis (BEFORE model implementation)

### Purpose
Determine the most important features for each arb year tier, understand how age and service time interact with performance stats, and guide final feature selection.

### Analysis Tasks

#### 1.1 Feature Correlations by Arb Year
**File:** `analysis/scripts/arb.py` (extend existing)

For each arb year (2, 3, 4, 5), compute:
- Top 10 features correlated with salary
- Compare: do different stats matter more in early vs late arb years?
- Identify features that are consistently predictive across all tiers

**Output:** `analysis/graphs/arb_feature_correlations_by_year.png`

#### 1.2 Age Interaction Effects
**File:** `analysis/scripts/arb.py`

Analyze:
- Salary vs age distribution per arb year
- Does age modify which stats predict salary?
- Young players (under 27) vs older players (27+): different feature importance?
- Age buckets: correlation heatmaps

**Output:** `analysis/graphs/arb_age_salary_interaction.png`

#### 1.3 Position Breakdowns
**File:** `analysis/scripts/arb.py`

Analyze:
- Salary distribution by position (box plots)
- Top predictive features by position group (SP, RP, C, IF, OF)
- Should position be a feature or should we build position-specific models?

**Output:** `analysis/graphs/arb_salary_by_position.png`

#### 1.4 Service Time Deep Dive
**File:** `analysis/scripts/arb.py`

Analyze:
- Within each arb year, does days of service time matter?
- Service time vs salary scatter with regression lines per tier
- Interaction: service_time × WAR → salary

**Output:** `analysis/graphs/arb_service_time_analysis.png`

#### 1.5 Summary Report
Generate a summary of findings to inform final feature selection:
- Which features to include in the model
- Whether to use separate models by player type or position
- Recommended feature engineering (interactions, transformations)

**Output:** `docs/arb/ANALYSIS_SUMMARY.md`

---

## Phase 2: Model Implementation

### Preliminary Feature Selection (to be refined by Phase 1)

**Personal (4):**
- `age`, `service_time`, `contract_year`, `position`

**Batting stats (5):**
- `bat_war_1y`, `bat_home_runs_1y`, `bat_rbis_1y`
- `bat_war_3y`, `bat_home_runs_3y`

**Pitching stats (5):**
- `pit_war_1y`, `pit_strikeouts_1y`, `pit_innings_pitched_1y`
- `pit_war_3y`, `pit_strikeouts_3y`

*Note: Final feature list will be determined by Phase 1 analysis findings.*

### Model Architecture

**Approach:** Single unified model (unless Phase 1 analysis suggests otherwise)
- RandomForest handles mixed features well
- Can revisit if analysis shows pitchers/batters need separate treatment

### File Structure

```
docs/arb/
├── DESIGN.md            # This design document
└── ANALYSIS_SUMMARY.md  # Summary of Phase 1 findings

analysis/scripts/
└── arb.py               # Extended with comprehensive analysis functions

analysis/graphs/
├── arb_feature_correlations_by_year.png
├── arb_age_salary_interaction.png
├── arb_salary_by_position.png
└── arb_service_time_analysis.png

models/arb/
├── __init__.py          # Export ArbModel
├── config.py            # Features, hyperparameters, tiered thresholds
├── features.py          # Data loading, filtering, preprocessing
├── model.py             # ArbModel class (train/evaluate/predict/save/load)
├── train.py             # CLI entrypoint
└── inspect.py           # Diagnostics and sample predictions

models/artifacts/
├── arb_model.pkl        # Serialized pipeline
└── arb_metrics.json     # Evaluation results with per-tier breakdown
```

---

## Shared Infrastructure to Reuse

| File | Functions |
|------|-----------|
| `models/preprocessing.py` | `load_contracts()`, `normalize_service_time()`, `build_preprocessor()`, `prepare_features()` |
| `models/evaluation.py` | `calculate_all_metrics()`, `format_metrics_report()` |
| `analysis/contract_analysis.py` | `CONTRACT_DATA`, `GRAPH_DIR`, `normalize_service_time()` |

---

## Testing & Validation

**Validation approach:**
1. 80/20 train/test split (stratified by arb year)
2. 5-fold cross-validation for stability
3. Per-tier accuracy check against 95% threshold

**Sanity checks:**
- Feature importances should align with Phase 1 correlation analysis
- Predictions should increase with arb year
- Outliers should correspond to known star players

**Run commands:**
```bash
# Run analysis (Phase 1)
make analyze

# Train and evaluate (Phase 2)
python -m models.arb.train --save

# Inspect model diagnostics
python -m models.arb.inspect
```

---

## Iteration Strategy

If 95% target not met for a tier:
1. Revisit Phase 1 analysis for that specific tier
2. Add tier-specific features identified in analysis
3. Try GradientBoosting instead of RandomForest
4. Consider separate models for pitchers vs batters (if analysis supports)
5. Feature engineering (interaction terms suggested by analysis)
