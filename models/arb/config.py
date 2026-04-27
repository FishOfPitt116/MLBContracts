# Configuration for arbitration salary prediction model

# Data source
CONTRACTS_DATASET = "dataset/contracts_with_stats.csv"

# Training data filters
CONTRACT_TYPE = "arb"
MAX_DURATION = 1  # Single-year contracts only

# Player type classification
PITCHER_POSITIONS = ["SP", "RP", "CL", "P"]

# Tiered accuracy targets (95% of predictions within tolerance)
# Based on 10% of average salary per arb year
# Service time ranges (normalized):
#   Arb Year 1: ST < 4 (includes Super Two with ST 2-3 and regular first-timers ST 3-4)
#   Arb Year 2: 4 <= ST < 5
#   Arb Year 3: 5 <= ST < 6 (final arb year before free agency)
# Note: No Year 4 - players with ST >= 6 are free agents
TIER_THRESHOLDS = {
    1: {"avg_salary": 2.25, "tolerance": 0.23, "expected_count": 1511},
    2: {"avg_salary": 4.00, "tolerance": 0.40, "expected_count": 819},
    3: {"avg_salary": 6.38, "tolerance": 0.64, "expected_count": 520},
}

# Overall tolerance for aggregate metrics (used in format_metrics_report)
TOLERANCE = 0.50  # $500K for overall pct_within_tolerance

# Feature configuration
# Common personal features (used by both pitcher and batter models)
PERSONAL_FEATURES = ["age", "service_time", "contract_year"]
POSITION_FEATURE = "position"  # Will be one-hot encoded (batter model only)

# Pitcher model features (no position encoding needed - all are pitchers)
PITCHER_PERSONAL_FEATURES = ["age", "service_time", "contract_year"]

# Batter model features (includes position one-hot encoding)
BATTER_PERSONAL_FEATURES = ["age", "service_time", "contract_year"]

# Batting stats - based on Phase 1 analysis correlations (position players only)
# Correlations by year: Year 1 -> Year 2 -> Year 3
# bat_home_runs_5y: +0.68 -> +0.77 -> +0.81 (strongest in Year 3)
# bat_war_3y: +0.74 -> +0.75 -> +0.76
# bat_rbis_3y: +0.74 -> +0.75 -> +0.78
# bat_home_runs_3y: +0.73 -> +0.75 -> +0.79
BATTER_FEATURES = [
    "bat_war_3y",        # +0.74-0.76 - key WAR metric
    "bat_war_1y",        # recent performance
    "bat_home_runs_3y",  # +0.73-0.79
    "bat_home_runs_5y",  # +0.68-0.81 - strongest in later years
    "bat_rbis_3y",       # +0.74-0.78
]

# Pitching stats - based on Phase 1 analysis correlations (pitchers only)
# Correlations by year: Year 1 -> Year 2 -> Year 3
# pit_war_3y: +0.77 -> +0.84 -> +0.87 (strongest overall)
# pit_war_5y: +0.82 -> +0.84 -> +0.84
# pit_strikeouts_3y: +0.72 -> +0.76 -> +0.77
# Note: ERA/FIP have weak negative correlation, not included
PITCHER_FEATURES = [
    "pit_war_3y",         # +0.77-0.87 - strongest predictor
    "pit_war_5y",         # +0.82-0.84 - very strong
    "pit_war_1y",         # +0.69-0.74 - recent performance
    "pit_strikeouts_3y",  # +0.72-0.77
    "pit_strikeouts_1y",  # +0.68-0.69
]

# Target variable
TARGET = "value"

# Model hyperparameters
RANDOM_FOREST_N_ESTIMATORS = 200
RANDOM_FOREST_MAX_DEPTH = 15
RANDOM_FOREST_MIN_SAMPLES_SPLIT = 5
RANDOM_FOREST_MIN_SAMPLES_LEAF = 2
RANDOM_FOREST_RANDOM_STATE = 42

GRADIENT_BOOSTING_N_ESTIMATORS = 200
GRADIENT_BOOSTING_MAX_DEPTH = 5
GRADIENT_BOOSTING_LEARNING_RATE = 0.1
GRADIENT_BOOSTING_RANDOM_STATE = 42

# Training configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Accuracy thresholds
PCT_WITHIN_TIER_TOLERANCE_THRESHOLD = 0.95  # 95% per tier
MAE_THRESHOLD = 1.0  # $1M overall MAE (secondary metric)

# Artifact paths
ARTIFACTS_DIR = "models/artifacts"
MODEL_FILENAME = "arb_model.pkl"
METRICS_FILENAME = "arb_metrics.json"

# Player-type specific artifact filenames
PITCHER_MODEL_FILENAME = "arb_pitcher_model.pkl"
PITCHER_METRICS_FILENAME = "arb_pitcher_metrics.json"
BATTER_MODEL_FILENAME = "arb_batter_model.pkl"
BATTER_METRICS_FILENAME = "arb_batter_metrics.json"


def get_model_filename(player_type):
    """Get model filename for a specific player type."""
    if player_type == "pitcher":
        return PITCHER_MODEL_FILENAME
    elif player_type == "batter":
        return BATTER_MODEL_FILENAME
    else:
        return MODEL_FILENAME


def get_metrics_filename(player_type):
    """Get metrics filename for a specific player type."""
    if player_type == "pitcher":
        return PITCHER_METRICS_FILENAME
    elif player_type == "batter":
        return BATTER_METRICS_FILENAME
    else:
        return METRICS_FILENAME


def is_pitcher_position(position):
    """Check if a position is a pitcher position."""
    return position in PITCHER_POSITIONS


def get_tier_from_service_time(service_time):
    """
    Convert normalized service time to arbitration year (1-3).

    Args:
        service_time: Normalized service time (linear scale)

    Returns:
        int: Arb year (1, 2, or 3), or None if service_time is NaN

    Service time mapping:
        Year 1: ST < 4 (includes Super Two with ST 2-3 and regular first-timers ST 3-4)
        Year 2: 4 <= ST < 5
        Year 3: 5 <= ST < 6 (final arb year before free agency)
    """
    import pandas as pd

    if pd.isna(service_time):
        return None
    if service_time < 4:
        return 1
    elif service_time < 5:
        return 2
    else:
        return 3
