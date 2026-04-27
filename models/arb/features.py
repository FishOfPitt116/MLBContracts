"""Feature selection and engineering for arbitration model."""

from models.arb import config
from models.preprocessing import load_contracts, build_preprocessor, prepare_features


def get_feature_columns(player_type=None):
    """
    Return list of all feature columns used by the model.

    Args:
        player_type: "pitcher", "batter", or None (unified model)

    Returns:
        list: Feature column names
    """
    if player_type == "pitcher":
        return config.PITCHER_PERSONAL_FEATURES + config.PITCHER_FEATURES
    elif player_type == "batter":
        return (
            config.BATTER_PERSONAL_FEATURES
            + [config.POSITION_FEATURE]
            + config.BATTER_FEATURES
        )
    else:
        # Unified model (legacy) - includes all features
        return (
            config.PERSONAL_FEATURES
            + [config.POSITION_FEATURE]
            + config.BATTER_FEATURES
            + config.PITCHER_FEATURES
        )


def load_and_filter_data(filepath=None, player_type=None):
    """
    Load contracts with stats and filter to single-year arbitration contracts.

    Args:
        filepath: Path to CSV file (default from config)
        player_type: "pitcher", "batter", or None (all players)

    Returns:
        pd.DataFrame: Filtered dataset ready for feature engineering
    """
    df = load_contracts(filepath)

    # Filter to single-year arbitration contracts with valid service time
    mask = (
        (df["contract_type"] == config.CONTRACT_TYPE)
        & (df["duration"] == config.MAX_DURATION)
        & (df["service_time"].notna())  # Exclude contracts missing service time
    )

    filtered_df = df[mask].copy()

    # Apply player type filter if specified
    if player_type == "pitcher":
        filtered_df = filtered_df[
            filtered_df["position"].isin(config.PITCHER_POSITIONS)
        ]
    elif player_type == "batter":
        filtered_df = filtered_df[
            ~filtered_df["position"].isin(config.PITCHER_POSITIONS)
        ]

    return filtered_df


def load_pitcher_data(filepath=None):
    """Load contracts filtered to pitchers only."""
    return load_and_filter_data(filepath, player_type="pitcher")


def load_batter_data(filepath=None):
    """Load contracts filtered to batters (position players) only."""
    return load_and_filter_data(filepath, player_type="batter")


def get_preprocessor(player_type=None):
    """
    Build preprocessor for arbitration model features.

    Args:
        player_type: "pitcher", "batter", or None (unified model)

    Returns:
        ColumnTransformer: Configured preprocessor
    """
    if player_type == "pitcher":
        return get_pitcher_preprocessor()
    elif player_type == "batter":
        return get_batter_preprocessor()
    else:
        # Unified model (legacy)
        numeric_features = (
            config.PERSONAL_FEATURES
            + config.BATTER_FEATURES
            + config.PITCHER_FEATURES
        )
        categorical_features = [config.POSITION_FEATURE]
        return build_preprocessor(numeric_features, categorical_features)


def get_pitcher_preprocessor():
    """Build preprocessor for pitcher-only model (no position encoding needed)."""
    numeric_features = config.PITCHER_PERSONAL_FEATURES + config.PITCHER_FEATURES
    # No categorical features - all pitchers
    return build_preprocessor(numeric_features, categorical_features=None)


def get_batter_preprocessor():
    """Build preprocessor for batter-only model (includes position encoding)."""
    numeric_features = config.BATTER_PERSONAL_FEATURES + config.BATTER_FEATURES
    categorical_features = [config.POSITION_FEATURE]
    return build_preprocessor(numeric_features, categorical_features)


def get_pitcher_features():
    """Return feature columns for pitcher model."""
    return config.PITCHER_PERSONAL_FEATURES + config.PITCHER_FEATURES


def get_batter_features():
    """Return feature columns for batter model (includes position)."""
    return (
        config.BATTER_PERSONAL_FEATURES
        + [config.POSITION_FEATURE]
        + config.BATTER_FEATURES
    )


def get_features_and_target(df, player_type=None):
    """
    Extract features and target for arbitration model.

    Args:
        df: DataFrame with contracts and stats
        player_type: "pitcher", "batter", or None (unified model)

    Returns:
        tuple: (X dataframe, y series)
    """
    return prepare_features(df, get_feature_columns(player_type), config.TARGET)


def add_arb_year_column(df):
    """
    Add arb_year column based on normalized service time.

    Args:
        df: DataFrame with normalized service_time column

    Returns:
        DataFrame with arb_year column added
    """
    df = df.copy()
    df["arb_year"] = df["service_time"].apply(config.get_tier_from_service_time)
    return df
