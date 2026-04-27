"""Shared preprocessing utilities for contract prediction models."""

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Days needed for one year of MLB service time
DAYS_PER_SERVICE_YEAR = 172

# Default dataset path
CONTRACTS_DATASET = "dataset/contracts_with_stats.csv"


def normalize_service_time(service_time):
    """
    Normalize service time from years.days format to linear scale.

    MLB service time is encoded as years.days where days is 0-172.
    For example, 2.028 means 2 years and 28 days.

    This converts to a linear scale where the fractional part represents
    the proportion of a service year (0-0.99 instead of 0-0.172).

    Examples:
        2.028 -> 2 + (28/172) = 2.163
        2.100 -> 2 + (100/172) = 2.581
        2.170 -> 2 + (170/172) = 2.988

    Args:
        service_time: Service time in years.days format

    Returns:
        Normalized service time on linear scale
    """
    if pd.isna(service_time):
        return service_time

    years = int(service_time)
    # The decimal part encodes days (e.g., 0.028 = 28 days, 0.100 = 100 days)
    days = round((service_time - years) * 1000)
    normalized = years + (days / DAYS_PER_SERVICE_YEAR)
    return normalized


def load_contracts(filepath=None):
    """
    Load contracts with stats dataset.

    Args:
        filepath: Path to CSV file (default: dataset/contracts_with_stats.csv)

    Returns:
        pd.DataFrame: Full contracts dataset
    """
    if filepath is None:
        filepath = CONTRACTS_DATASET
    return pd.read_csv(filepath)


def build_preprocessor(numeric_features, categorical_features=None):
    """
    Build sklearn preprocessor for feature transformation.

    Args:
        numeric_features: List of numeric column names
        categorical_features: List of categorical column names (default: None)

    Returns:
        ColumnTransformer: Configured preprocessor
    """
    transformers = []

    if numeric_features:
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", numeric_transformer, numeric_features))

    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", categorical_transformer, categorical_features))

    return ColumnTransformer(transformers=transformers)


def prepare_features(df, feature_cols, target_col):
    """
    Extract feature matrix and target from dataframe.

    Args:
        df: DataFrame with contracts and stats
        feature_cols: List of feature column names
        target_col: Name of target column

    Returns:
        tuple: (X dataframe with feature columns, y series with target)
    """
    # Ensure all required columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    X = df[feature_cols].copy()

    # Normalize service_time from years.days format to linear scale
    if "service_time" in X.columns:
        X["service_time"] = X["service_time"].apply(normalize_service_time)

    y = df[target_col].copy()

    return X, y
