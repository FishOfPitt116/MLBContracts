# Pre-Arb Contract Value Prediction Model - Design Document

## Overview
Build a model to predict pre-arbitration player annual salary based on player attributes and prior-season performance statistics. This is the first of three models (pre-arb, arb, FA) that will combine to predict extension values.

## Design Decisions

### 1. Accuracy Metrics ✅
- **Primary**: MAE ≤ $0.15M (average error ≤ $150K)
- **95% accuracy requirement**: 95% of predictions within ±$0.25M of actual value
- **Rationale**: Dollar-based metrics avoid percentage distortion; pre-arb contracts cluster tightly around minimum salary

### 2. Scope ✅
- **Train on**: Typical single-year pre-arb contracts only (exclude extensions)
- **Mega-extensions**: Handled via composite approach - decompose by service time and sum predictions from pre-arb + arb + FA models
- **Rationale**: Cleaner training data; extensions validated as sum of regime-specific predictions

### 3. Target Variable ✅
- **Predict**: Annual salary for a player with given service time (< 3 years)
- **For 1-year contracts**: target = value
- **For extensions**: Will be decomposed into year-by-year predictions (future work)

---

## Dataset

**Source**: `dataset/contracts_with_stats.csv`

**Training Data Filter**:
```python
df[(df['contract_type'] == 'pre-arb') &
   (df['duration'] == 1) &  # Single-year contracts only
   (df['value'] < 5)]       # Exclude outlier extensions
```

**Expected records**: ~9,500 typical pre-arb contracts

**Features Available**:
- Personal: `age`, `service_time`, `position`
- Batting (1-year): `bat_war_1y`, `bat_ops_1y`, `bat_wrc_plus_1y`, `bat_home_runs_1y`, etc.
- Pitching (1-year): `pit_war_1y`, `pit_era_1y`, `pit_fip_1y`, `pit_whip_1y`, etc.

---

## Model Architecture

### Feature Selection
```python
PERSONAL_FEATURES = ['age', 'service_time', 'position']  # position one-hot encoded

BATTER_FEATURES = [
    'bat_war_1y', 'bat_ops_1y', 'bat_wrc_plus_1y',
    'bat_home_runs_1y', 'bat_rbis_1y', 'bat_stolen_bases_1y'
]

PITCHER_FEATURES = [
    'pit_war_1y', 'pit_era_1y', 'pit_fip_1y',
    'pit_whip_1y', 'pit_strikeouts_1y', 'pit_innings_pitched_1y'
]
```

### Model Pipeline
```
1. Load contracts_with_stats.csv
2. Filter to single-year pre-arb contracts < $5M
3. Handle missing stats (impute or exclude)
4. Encode categorical features (position)
5. Train/test split (80/20, stratified by year)
6. Train model (start with Ridge Regression, then Random Forest)
7. Evaluate on test set
8. Save model artifacts
```

### Models to Evaluate
1. **Ridge Regression** - Baseline, interpretable
2. **Random Forest** - Handles non-linear relationships
3. **Gradient Boosting (XGBoost)** - If RF underperforms

---

## File Structure

```
models/
├── __init__.py
├── pre_arb/
│   ├── __init__.py
│   ├── model.py          # PreArbModel class with train/predict
│   ├── features.py       # Feature selection and engineering
│   └── config.py         # Hyperparameters, feature lists
├── evaluation.py         # Shared metrics (MAE, % within tolerance)
└── train_pre_arb.py      # CLI entrypoint: python -m models.train_pre_arb
```

**Output artifacts** (saved to `models/artifacts/`):
- `pre_arb_model.pkl` - Trained model
- `pre_arb_scaler.pkl` - Feature scaler
- `pre_arb_metrics.json` - Evaluation results

---

## Validation Strategy

### Test Suite (`models/tests/test_pre_arb.py`)
```python
def test_mae_threshold():
    """MAE must be ≤ $0.15M"""
    assert metrics['mae'] <= 0.15

def test_95_percent_tolerance():
    """95% of predictions must be within ±$0.25M"""
    assert metrics['pct_within_tolerance'] >= 0.95

def test_cross_validation_stability():
    """CV scores should have low variance"""
    assert np.std(cv_scores) < 0.05
```

### Evaluation Metrics
- MAE (Mean Absolute Error)
- % within ±$0.25M tolerance
- R² (for reference, not primary)
- 5-fold cross-validation scores

---

## Implementation Steps

### Step 1: Create model infrastructure
- Create `models/` directory structure
- Implement `evaluation.py` with metric functions
- Implement `features.py` with feature selection

### Step 2: Implement PreArbModel
- Data loading and filtering
- Feature preprocessing pipeline
- Model training with sklearn
- Prediction interface

### Step 3: Create training script
- CLI entrypoint with arguments
- Train/test split logic
- Model persistence (joblib)
- Metrics output

### Step 4: Implement tests
- Unit tests for metrics
- Integration tests against dataset
- Validation that accuracy thresholds are met

### Step 5: Run and validate
- Execute training
- Review metrics
- Iterate if thresholds not met

---

## Files to Reference (Existing Patterns)

- `analysis/contract_analysis.py` - Data loading, AAV calculation
- `analysis/scripts/arb.py` - Filtering by contract type, visualization
- `data_generation/records.py` - Data structure definitions
- `archive/v1/src/models.py` - Previous model persistence patterns

---

## Future Work (Out of Scope)

1. **Arb model**: Predict arbitration-year salaries (service time 3-5)
2. **FA model**: Predict free-agent salaries (service time 6+)
3. **Extension validator**: Decompose extensions, sum predictions, compare to actual
4. **Dataset augmentation**: Create year-by-year records for multi-year contracts