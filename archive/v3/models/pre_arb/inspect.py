"""Inspect pre-arb model results and make sample predictions."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from models.pre_arb.model import PreArbModel
from models.pre_arb.features import load_and_filter_data, get_features_and_target
from models.pre_arb import config


def show_feature_importances(model):
    """Display feature importances for tree-based models."""
    importances = model.get_feature_importances()
    if importances is None:
        print("Feature importances not available (only for tree-based models)")
        return

    print("\n" + "=" * 50)
    print("Feature Importances")
    print("=" * 50)

    # Sort by importance
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    for name, importance in sorted_features:
        bar = "█" * int(importance * 50)
        print(f"{name:<30} {importance:.4f} {bar}")


def show_predictions_vs_actual(model, X_test, y_test, df_test, n=15):
    """Show sample predictions compared to actual values."""
    predictions = model.predict(X_test)
    errors = predictions - y_test.values

    print("\n" + "=" * 80)
    print("Sample Predictions vs Actual")
    print("=" * 80)
    print(f"{'Player':<30} {'Actual':>10} {'Predicted':>10} {'Error':>10}")
    print("-" * 80)

    # Get player names from contract_id
    for i in range(min(n, len(predictions))):
        idx = y_test.index[i]
        contract_id = df_test.loc[idx, 'contract_id']
        player_name = contract_id.rsplit('_', 2)[0].replace('_', ' ')
        actual = y_test.iloc[i]
        pred = predictions[i]
        err = errors[i]
        print(f"{player_name:<30} ${actual:>8.3f}M ${pred:>8.3f}M {err:>+9.3f}M")


def show_error_distribution(model, X_test, y_test):
    """Show distribution of prediction errors."""
    predictions = model.predict(X_test)
    errors = np.abs(predictions - y_test.values)

    print("\n" + "=" * 50)
    print("Error Distribution")
    print("=" * 50)
    print(f"Mean Absolute Error: ${np.mean(errors):.4f}M (${np.mean(errors)*1000:.1f}K)")
    print(f"Median Absolute Error: ${np.median(errors):.4f}M (${np.median(errors)*1000:.1f}K)")
    print(f"Max Absolute Error: ${np.max(errors):.4f}M")
    print(f"Min Absolute Error: ${np.min(errors):.4f}M")

    # Percentiles
    print("\nError Percentiles:")
    for p in [50, 75, 90, 95, 99]:
        val = np.percentile(errors, p)
        print(f"  {p}th percentile: ${val:.4f}M")

    # Counts within thresholds
    print("\nPredictions within threshold:")
    for thresh in [0.05, 0.10, 0.15, 0.25, 0.50]:
        pct = np.mean(errors <= thresh) * 100
        print(f"  Within ±${thresh}M: {pct:.1f}%")


def show_worst_predictions(model, X_test, y_test, df_test, n=10):
    """Show the worst predictions (largest errors)."""
    predictions = model.predict(X_test)
    errors = np.abs(predictions - y_test.values)

    # Get indices of worst predictions
    worst_indices = np.argsort(errors)[-n:][::-1]

    print("\n" + "=" * 80)
    print(f"Top {n} Worst Predictions (Largest Errors)")
    print("=" * 80)
    print(f"{'Player':<25} {'Year':>6} {'Actual':>10} {'Predicted':>10} {'Error':>10}")
    print("-" * 80)

    for i in worst_indices:
        idx = y_test.index[i]
        contract_id = df_test.loc[idx, 'contract_id']
        year = df_test.loc[idx, 'contract_year']
        player_name = contract_id.rsplit('_', 2)[0].replace('_', ' ')[:25]
        actual = y_test.iloc[i]
        pred = predictions[i]
        err = errors[i]
        print(f"{player_name:<25} {year:>6} ${actual:>8.3f}M ${pred:>8.3f}M ${err:>8.3f}M")


def show_salary_distribution(y_test):
    """Show actual salary distribution in test set."""
    print("\n" + "=" * 50)
    print("Actual Salary Distribution (Test Set)")
    print("=" * 50)
    print(f"Count: {len(y_test)}")
    print(f"Mean: ${y_test.mean():.4f}M")
    print(f"Median: ${y_test.median():.4f}M")
    print(f"Std Dev: ${y_test.std():.4f}M")
    print(f"Min: ${y_test.min():.4f}M")
    print(f"Max: ${y_test.max():.4f}M")


def predict_for_player(model, age, service_time, position, contract_year):
    """Make a prediction for a hypothetical player."""
    # Create a single-row dataframe with the features
    data = {
        'age': [age],
        'service_time': [service_time],
        'contract_year': [contract_year],
        'position': [position],
    }
    X = pd.DataFrame(data)
    prediction = model.predict(X)[0]
    return prediction


def main():
    print("Loading saved model...")
    model = PreArbModel.load()
    print(f"Model type: {model.model_type}")

    print("\nLoading test data...")
    df = load_and_filter_data()
    X, y = get_features_and_target(df)

    # Use same split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    df_test = df.loc[X_test.index]

    # Show various results
    show_salary_distribution(y_test)
    show_feature_importances(model)
    show_error_distribution(model, X_test, y_test)
    show_predictions_vs_actual(model, X_test, y_test, df_test)
    show_worst_predictions(model, X_test, y_test, df_test)

    # Example custom predictions
    print("\n" + "=" * 50)
    print("Example Custom Predictions (2026)")
    print("=" * 50)

    # Different positions/service times
    pred = predict_for_player(model, age=25, service_time=1.5, position="SS", contract_year=2026)
    print(f"SS (25yo, 1.5 service): ${pred:.3f}M")

    pred = predict_for_player(model, age=24, service_time=2.0, position="CF", contract_year=2026)
    print(f"CF (24yo, 2.0 service): ${pred:.3f}M")

    pred = predict_for_player(model, age=23, service_time=0.5, position="SP", contract_year=2026)
    print(f"SP (23yo, 0.5 service): ${pred:.3f}M")

    # Future year predictions
    print("\n" + "=" * 50)
    print("Future Year Predictions (SS, 25yo, 1.5 service)")
    print("=" * 50)
    for year in [2026, 2027, 2028, 2029, 2030]:
        pred = predict_for_player(model, age=25, service_time=1.5, position="SS", contract_year=year)
        print(f"{year}: ${pred:.3f}M")


if __name__ == "__main__":
    main()
