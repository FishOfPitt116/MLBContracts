# Configuration for pre-arb model

# Data source
CONTRACTS_DATASET = "dataset/contracts_with_stats.csv"

# Training data filters
CONTRACT_TYPE = "pre-arb"
MAX_DURATION = 1  # Single-year contracts only
MAX_VALUE = 5.0   # Exclude outlier extensions (in millions)

# Feature configuration
PERSONAL_FEATURES = ["age", "service_time", "contract_year"]
POSITION_FEATURE = "position"  # Will be one-hot encoded

# Player stats excluded - pre-arb salaries are determined almost entirely by year
BATTER_FEATURES = []
PITCHER_FEATURES = []

# Target variable
TARGET = "value"

# Model hyperparameters
RIDGE_ALPHA = 1.0
RANDOM_FOREST_N_ESTIMATORS = 100
RANDOM_FOREST_MAX_DEPTH = 10
RANDOM_FOREST_RANDOM_STATE = 42

# Training configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Accuracy thresholds
MAE_THRESHOLD = 0.15  # $0.15M = $150K
TOLERANCE = 0.25      # ±$0.25M for 95% accuracy
PCT_WITHIN_TOLERANCE_THRESHOLD = 0.95

# Artifact paths
ARTIFACTS_DIR = "models/artifacts"
MODEL_FILENAME = "pre_arb_model.pkl"
METRICS_FILENAME = "pre_arb_metrics.json"
