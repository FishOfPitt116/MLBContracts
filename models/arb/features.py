"""Feature selection and engineering for arbitration model."""

from models.arb import config
from models.preprocessing import (
    load_contracts,
    build_preprocessor,
    prepare_features,
    normalize_service_time,
)


def _use_market_anchors(model_type):
    """Return True if the model profile calls for market anchor features."""
    profile = config.MODEL_PROFILES.get(model_type or "random_forest", {})
    return profile.get("use_market_anchors", False)


def add_market_anchors(df):
    """
    Add prior-year market anchor features (no data leakage).

    For each contract in year Y, attaches the mean and 90th-percentile salary
    for the same player_type + arb tier from year Y-1. Contracts in the first
    dataset year (2011) receive NaN, handled by median imputation downstream.

    Args:
        df: Full filtered arb DataFrame (all player types, before any split)

    Returns:
        DataFrame with market_prior_mean and market_prior_p90 columns added
    """
    df = df.copy()

    # Derive grouping keys without touching service_time (normalization is downstream)
    df["_st_norm"] = df["service_time"].apply(normalize_service_time)
    df["_arb_year"] = df["_st_norm"].apply(config.get_tier_from_service_time)
    df["_player_type"] = df["position"].apply(
        lambda p: "pitcher" if p in config.PITCHER_POSITIONS else "batter"
    )

    year_tier_stats = (
        df.groupby(["contract_year", "_arb_year", "_player_type"])["value"]
        .agg(
            market_prior_mean="mean",
            market_prior_p90=lambda x: x.quantile(0.90),
        )
        .reset_index()
    )
    # Shift: year Y stats become the anchor for year Y+1
    year_tier_stats["contract_year"] += 1

    df = df.merge(
        year_tier_stats,
        on=["contract_year", "_arb_year", "_player_type"],
        how="left",
    )
    return df.drop(columns=["_st_norm", "_arb_year", "_player_type"])


def get_feature_columns(player_type=None, model_type=None):
    """
    Return the list of feature columns for the given player type and model.

    Linear models (ridge, lasso) include market anchor features; tree models do not.
    """
    anchors = config.MARKET_ANCHOR_FEATURES if _use_market_anchors(model_type) else []

    if player_type == "pitcher":
        return config.PITCHER_PERSONAL_FEATURES + anchors + config.PITCHER_FEATURES
    elif player_type == "batter":
        return (
            config.BATTER_PERSONAL_FEATURES
            + anchors
            + [config.POSITION_FEATURE]
            + config.BATTER_FEATURES
        )
    else:
        # Unified model (legacy)
        return (
            config.PERSONAL_FEATURES
            + anchors
            + [config.POSITION_FEATURE]
            + config.BATTER_FEATURES
            + config.PITCHER_FEATURES
        )


def load_and_filter_data(filepath=None, player_type=None, model_type=None):
    """
    Load contracts with stats and filter to single-year arbitration contracts.

    Args:
        filepath:    Path to CSV file (default from config)
        player_type: "pitcher", "batter", or None (all players)
        model_type:  Algorithm name — determines whether anchor features are added

    Returns:
        pd.DataFrame: Filtered dataset ready for feature engineering
    """
    df = load_contracts(filepath)

    mask = (
        (df["contract_type"] == config.CONTRACT_TYPE)
        & (df["duration"] == config.MAX_DURATION)
        & (df["service_time"].notna())
    )
    filtered_df = df[mask].copy()

    # Add market anchors before player_type split so each tier's population is complete
    if _use_market_anchors(model_type):
        filtered_df = add_market_anchors(filtered_df)

    if player_type == "pitcher":
        filtered_df = filtered_df[filtered_df["position"].isin(config.PITCHER_POSITIONS)]
    elif player_type == "batter":
        filtered_df = filtered_df[~filtered_df["position"].isin(config.PITCHER_POSITIONS)]

    return filtered_df


def load_pitcher_data(filepath=None, model_type=None):
    """Load contracts filtered to pitchers only."""
    return load_and_filter_data(filepath, player_type="pitcher", model_type=model_type)


def load_batter_data(filepath=None, model_type=None):
    """Load contracts filtered to batters (position players) only."""
    return load_and_filter_data(filepath, player_type="batter", model_type=model_type)


def get_preprocessor(player_type=None, model_type=None):
    """
    Build sklearn preprocessor for the given player type and model.

    Args:
        player_type: "pitcher", "batter", or None (unified model)
        model_type:  Algorithm name — determines feature set

    Returns:
        ColumnTransformer: Configured preprocessor
    """
    anchors = config.MARKET_ANCHOR_FEATURES if _use_market_anchors(model_type) else []

    if player_type == "pitcher":
        numeric_features = config.PITCHER_PERSONAL_FEATURES + anchors + config.PITCHER_FEATURES
        return build_preprocessor(numeric_features, categorical_features=None)

    elif player_type == "batter":
        numeric_features = config.BATTER_PERSONAL_FEATURES + anchors + config.BATTER_FEATURES
        return build_preprocessor(numeric_features, categorical_features=[config.POSITION_FEATURE])

    else:
        # Unified model (legacy)
        numeric_features = config.PERSONAL_FEATURES + anchors + config.BATTER_FEATURES + config.PITCHER_FEATURES
        return build_preprocessor(numeric_features, categorical_features=[config.POSITION_FEATURE])


def get_pitcher_features(model_type=None):
    """Return feature columns for pitcher model."""
    anchors = config.MARKET_ANCHOR_FEATURES if _use_market_anchors(model_type) else []
    return config.PITCHER_PERSONAL_FEATURES + anchors + config.PITCHER_FEATURES


def get_batter_features(model_type=None):
    """Return feature columns for batter model (includes position)."""
    anchors = config.MARKET_ANCHOR_FEATURES if _use_market_anchors(model_type) else []
    return config.BATTER_PERSONAL_FEATURES + anchors + [config.POSITION_FEATURE] + config.BATTER_FEATURES


def get_features_and_target(df, player_type=None, model_type=None):
    """
    Extract features and target for arbitration model.

    Args:
        df:          DataFrame with contracts and stats
        player_type: "pitcher", "batter", or None (unified model)
        model_type:  Algorithm name — determines feature set

    Returns:
        tuple: (X dataframe, y series)
    """
    return prepare_features(df, get_feature_columns(player_type, model_type), config.TARGET)


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
