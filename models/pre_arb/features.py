"""Feature selection and engineering for pre-arb model."""

from models.pre_arb import config
from models.preprocessing import load_contracts, build_preprocessor, prepare_features


def get_feature_columns():
    """Return list of all feature columns used by the model."""
    return (
        config.PERSONAL_FEATURES
        + [config.POSITION_FEATURE]
        + config.BATTER_FEATURES
        + config.PITCHER_FEATURES
    )


def load_and_filter_data(filepath=None):
    """
    Load contracts with stats and filter to single-year pre-arb contracts.

    Returns:
        pd.DataFrame: Filtered dataset ready for feature engineering
    """
    df = load_contracts(filepath)

    # Filter to single-year pre-arb contracts under threshold value
    mask = (
        (df["contract_type"] == config.CONTRACT_TYPE)
        & (df["duration"] == config.MAX_DURATION)
        & (df["value"] < config.MAX_VALUE)
    )

    return df[mask].copy()


def get_preprocessor():
    """Build preprocessor for pre-arb model features."""
    numeric_features = config.PERSONAL_FEATURES + config.BATTER_FEATURES + config.PITCHER_FEATURES
    categorical_features = [config.POSITION_FEATURE]
    return build_preprocessor(numeric_features, categorical_features)


def get_features_and_target(df):
    """Extract features and target for pre-arb model."""
    return prepare_features(df, get_feature_columns(), config.TARGET)
