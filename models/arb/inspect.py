"""Inspect arbitration model results and make sample predictions."""

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from models.arb import config
from models.arb.features import (
    add_arb_year_column,
    get_feature_columns,
    get_features_and_target,
    load_and_filter_data,
)
from models.arb.model import ArbModel, format_tier_report
from models.preprocessing import normalize_service_time


def show_feature_importances(model, top_n=20):
    """Display feature importances for tree-based models."""
    importances = model.get_feature_importances()
    if importances is None:
        print("Feature importances not available (only for tree-based models)")
        return

    print("\n" + "=" * 60)
    print(f"Top {top_n} Feature Importances")
    print("=" * 60)

    # Sort by importance
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[
        :top_n
    ]
    for name, importance in sorted_features:
        bar = "█" * int(importance * 100)
        print(f"{name:<35} {importance:.4f} {bar}")


def show_predictions_vs_actual(model, X_test, y_test, df_test, n=20):
    """Show sample predictions compared to actual values."""
    predictions = model.predict(X_test)
    errors = predictions - y_test.values

    # Add arb year info
    X_with_tier = add_arb_year_column(X_test)

    print("\n" + "=" * 90)
    print("Sample Predictions vs Actual")
    print("=" * 90)
    print(
        f"{'Player':<25} {'Tier':>5} {'Actual':>10} {'Predicted':>10} {'Error':>10} {'Status':>8}"
    )
    print("-" * 90)

    # Get player names from contract_id
    for i in range(min(n, len(predictions))):
        idx = y_test.index[i]
        contract_id = df_test.loc[idx, "contract_id"]
        player_name = contract_id.rsplit("_", 2)[0].replace("_", " ")[:25]
        actual = y_test.iloc[i]
        pred = predictions[i]
        err = errors[i]
        tier = X_with_tier.iloc[i]["arb_year"]
        tier_tolerance = config.TIER_THRESHOLDS.get(int(tier), {}).get("tolerance", 0.5)
        status = "OK" if abs(err) <= tier_tolerance else "MISS"

        print(
            f"{player_name:<25} {int(tier):>5} ${actual:>8.3f}M ${pred:>8.3f}M {err:>+9.3f}M {status:>8}"
        )


def show_error_distribution(model, X_test, y_test):
    """Show distribution of prediction errors."""
    predictions = model.predict(X_test)
    errors = np.abs(predictions - y_test.values)

    print("\n" + "=" * 60)
    print("Error Distribution")
    print("=" * 60)
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
    for thresh in [0.20, 0.40, 0.60, 1.00, 2.00]:
        pct = np.mean(errors <= thresh) * 100
        print(f"  Within ±${thresh}M: {pct:.1f}%")


def show_worst_predictions(model, X_test, y_test, df_test, n=15):
    """Show the worst predictions (largest errors)."""
    predictions = model.predict(X_test)
    errors = np.abs(predictions - y_test.values)

    # Add arb year info
    X_with_tier = add_arb_year_column(X_test)

    # Get indices of worst predictions
    worst_indices = np.argsort(errors)[-n:][::-1]

    print("\n" + "=" * 90)
    print(f"Top {n} Worst Predictions (Largest Errors)")
    print("=" * 90)
    print(
        f"{'Player':<25} {'Year':>6} {'Tier':>5} {'Actual':>10} {'Predicted':>10} {'Error':>10}"
    )
    print("-" * 90)

    for i in worst_indices:
        idx = y_test.index[i]
        contract_id = df_test.loc[idx, "contract_id"]
        year = df_test.loc[idx, "contract_year"]
        player_name = contract_id.rsplit("_", 2)[0].replace("_", " ")[:25]
        actual = y_test.iloc[i]
        pred = predictions[i]
        err = errors[i]
        tier = X_with_tier.loc[idx, "arb_year"]

        print(
            f"{player_name:<25} {year:>6} {int(tier):>5} ${actual:>8.3f}M ${pred:>8.3f}M ${err:>8.3f}M"
        )


def show_salary_distribution(y_test, X_test):
    """Show actual salary distribution in test set by tier."""
    X_with_tier = add_arb_year_column(X_test)

    print("\n" + "=" * 60)
    print("Actual Salary Distribution (Test Set)")
    print("=" * 60)

    print(f"\nOverall (n={len(y_test)}):")
    print(f"  Mean: ${y_test.mean():.4f}M")
    print(f"  Median: ${y_test.median():.4f}M")
    print(f"  Std Dev: ${y_test.std():.4f}M")
    print(f"  Min: ${y_test.min():.4f}M")
    print(f"  Max: ${y_test.max():.4f}M")

    print("\nBy Arb Year:")
    for tier in [1, 2, 3]:
        tier_mask = X_with_tier["arb_year"] == tier
        tier_y = y_test[tier_mask]
        if len(tier_y) > 0:
            print(f"\n  Year {tier} (n={len(tier_y)}):")
            print(f"    Mean: ${tier_y.mean():.3f}M")
            print(f"    Median: ${tier_y.median():.3f}M")
            print(f"    Range: ${tier_y.min():.3f}M - ${tier_y.max():.3f}M")


def predict_for_pitcher(
    model,
    age,
    service_time,
    contract_year,
    pit_war_1y=0,
    pit_war_3y=0,
    pit_war_5y=0,
    pit_strikeouts_1y=0,
    pit_strikeouts_3y=0,
):
    """Make a prediction for a hypothetical pitcher."""
    normalized_st = normalize_service_time(service_time)

    data = {
        "age": [age],
        "service_time": [normalized_st],
        "contract_year": [contract_year],
        "pit_war_3y": [pit_war_3y],
        "pit_war_5y": [pit_war_5y],
        "pit_war_1y": [pit_war_1y],
        "pit_strikeouts_3y": [pit_strikeouts_3y],
        "pit_strikeouts_1y": [pit_strikeouts_1y],
    }
    X = pd.DataFrame(data)
    prediction = model.predict(X)[0]

    tier = config.get_tier_from_service_time(normalized_st)
    return prediction, tier


def predict_for_batter(
    model,
    age,
    service_time,
    position,
    contract_year,
    bat_war_1y=0,
    bat_war_3y=0,
    bat_home_runs_3y=0,
    bat_home_runs_5y=0,
    bat_rbis_3y=0,
):
    """Make a prediction for a hypothetical batter."""
    normalized_st = normalize_service_time(service_time)

    data = {
        "age": [age],
        "service_time": [normalized_st],
        "contract_year": [contract_year],
        "position": [position],
        "bat_war_3y": [bat_war_3y],
        "bat_war_1y": [bat_war_1y],
        "bat_home_runs_3y": [bat_home_runs_3y],
        "bat_home_runs_5y": [bat_home_runs_5y],
        "bat_rbis_3y": [bat_rbis_3y],
    }
    X = pd.DataFrame(data)
    prediction = model.predict(X)[0]

    tier = config.get_tier_from_service_time(normalized_st)
    return prediction, tier


def predict_for_player(
    model,
    age,
    service_time,
    position,
    contract_year,
    bat_war_1y=0,
    bat_war_3y=0,
    bat_home_runs_3y=0,
    bat_home_runs_5y=0,
    bat_rbis_3y=0,
    pit_war_1y=0,
    pit_war_3y=0,
    pit_war_5y=0,
    pit_strikeouts_1y=0,
    pit_strikeouts_3y=0,
):
    """Make a prediction for a hypothetical player (unified model only)."""
    # Normalize service time
    normalized_st = normalize_service_time(service_time)

    data = {
        "age": [age],
        "service_time": [normalized_st],
        "contract_year": [contract_year],
        "position": [position],
        "bat_war_3y": [bat_war_3y],
        "bat_war_1y": [bat_war_1y],
        "bat_home_runs_3y": [bat_home_runs_3y],
        "bat_home_runs_5y": [bat_home_runs_5y],
        "bat_rbis_3y": [bat_rbis_3y],
        "pit_war_3y": [pit_war_3y],
        "pit_war_5y": [pit_war_5y],
        "pit_war_1y": [pit_war_1y],
        "pit_strikeouts_3y": [pit_strikeouts_3y],
        "pit_strikeouts_1y": [pit_strikeouts_1y],
    }
    X = pd.DataFrame(data)
    prediction = model.predict(X)[0]

    tier = config.get_tier_from_service_time(normalized_st)
    return prediction, tier


def inspect_model(player_type=None):
    """Inspect a model of the given player type."""
    player_label = player_type or "unified"

    print(f"Loading saved {player_label} model...")
    model = ArbModel.load(player_type=player_type)
    print(f"Model type: {model.model_type}")
    print(f"Player type: {model.player_type or 'unified'}")

    print("\nLoading test data...")
    df = load_and_filter_data(player_type=player_type)
    X, y = get_features_and_target(df, player_type=player_type)

    print(f"Dataset size: {len(df):,} contracts")
    print(f"Features: {len(get_feature_columns(player_type))} columns")

    # Create stratification bins for consistent split
    # service_time is already normalized by prepare_features
    X_with_bins = X.copy()
    X_with_bins["strat_bin"] = pd.cut(
        X_with_bins["service_time"],
        bins=[0, 4, 5, 10],
        labels=["1", "2", "3"],
    )

    # Use same split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=X_with_bins["strat_bin"],
    )
    df_test = df.loc[X_test.index]

    # Show various results
    show_salary_distribution(y_test, X_test)
    show_feature_importances(model)
    show_error_distribution(model, X_test, y_test)

    # Evaluate by tier
    tier_metrics = model.evaluate_by_tier(X_test, y_test)
    print("\n" + format_tier_report(tier_metrics))

    show_predictions_vs_actual(model, X_test, y_test, df_test)
    show_worst_predictions(model, X_test, y_test, df_test)

    return model


def show_example_predictions(model, player_type):
    """Show example predictions for the given player type."""
    print("\n" + "=" * 60)
    print("Example Custom Predictions (2026)")
    print("=" * 60)

    if player_type == "pitcher":
        print("\nPitchers:")
        pred, tier = predict_for_pitcher(
            model,
            age=27,
            service_time=4.000,  # Arb Year 2
            contract_year=2026,
            pit_war_1y=3.0,
            pit_war_3y=7.0,
            pit_war_5y=10.0,
            pit_strikeouts_1y=180,
            pit_strikeouts_3y=500,
        )
        print(f"SP (27yo, 4.000 ST, 3.0 WAR): ${pred:.3f}M (Year {tier})")

        pred, tier = predict_for_pitcher(
            model,
            age=29,
            service_time=5.150,  # Arb Year 3
            contract_year=2026,
            pit_war_1y=1.5,
            pit_war_3y=4.0,
            pit_war_5y=6.0,
            pit_strikeouts_1y=80,
            pit_strikeouts_3y=220,
        )
        print(f"RP (29yo, 5.150 ST, 1.5 WAR): ${pred:.3f}M (Year {tier})")

        pred, tier = predict_for_pitcher(
            model,
            age=25,
            service_time=3.050,  # Arb Year 1
            contract_year=2026,
            pit_war_1y=2.5,
            pit_war_3y=5.5,
            pit_war_5y=5.5,
            pit_strikeouts_1y=150,
            pit_strikeouts_3y=380,
        )
        print(f"SP (25yo, 3.050 ST, 2.5 WAR): ${pred:.3f}M (Year {tier})")

    elif player_type == "batter":
        print("\nPosition Players:")
        pred, tier = predict_for_batter(
            model,
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
        print(f"SS (26yo, 4.050 ST, 3.5 WAR): ${pred:.3f}M (Year {tier})")

        pred, tier = predict_for_batter(
            model,
            age=28,
            service_time=5.100,  # Arb Year 3
            position="1B",
            contract_year=2026,
            bat_war_1y=2.0,
            bat_war_3y=5.0,
            bat_home_runs_3y=80,
            bat_home_runs_5y=130,
            bat_rbis_3y=250,
        )
        print(f"1B (28yo, 5.100 ST, 2.0 WAR): ${pred:.3f}M (Year {tier})")

        pred, tier = predict_for_batter(
            model,
            age=24,
            service_time=3.100,  # Arb Year 1
            position="CF",
            contract_year=2026,
            bat_war_1y=4.0,
            bat_war_3y=9.0,
            bat_home_runs_3y=60,
            bat_home_runs_5y=60,
            bat_rbis_3y=180,
        )
        print(f"CF (24yo, 3.100 ST, 4.0 WAR): ${pred:.3f}M (Year {tier})")

    else:
        # Unified model - show both
        print("\nPosition Players:")
        pred, tier = predict_for_player(
            model,
            age=26,
            service_time=4.050,
            position="SS",
            contract_year=2026,
            bat_war_1y=3.5,
            bat_war_3y=8.0,
            bat_home_runs_3y=55,
            bat_home_runs_5y=85,
            bat_rbis_3y=200,
        )
        print(f"SS (26yo, 4.050 ST, 3.5 WAR): ${pred:.3f}M (Year {tier})")

        pred, tier = predict_for_player(
            model,
            age=28,
            service_time=5.100,
            position="1B",
            contract_year=2026,
            bat_war_1y=2.0,
            bat_war_3y=5.0,
            bat_home_runs_3y=80,
            bat_home_runs_5y=130,
            bat_rbis_3y=250,
        )
        print(f"1B (28yo, 5.100 ST, 2.0 WAR): ${pred:.3f}M (Year {tier})")

        print("\nPitchers:")
        pred, tier = predict_for_player(
            model,
            age=27,
            service_time=4.000,
            position="SP",
            contract_year=2026,
            pit_war_1y=3.0,
            pit_war_3y=7.0,
            pit_war_5y=10.0,
            pit_strikeouts_1y=180,
            pit_strikeouts_3y=500,
        )
        print(f"SP (27yo, 4.000 ST, 3.0 WAR): ${pred:.3f}M (Year {tier})")

        pred, tier = predict_for_player(
            model,
            age=29,
            service_time=5.150,
            position="RP",
            contract_year=2026,
            pit_war_1y=1.5,
            pit_war_3y=4.0,
            pit_war_5y=6.0,
            pit_strikeouts_1y=80,
            pit_strikeouts_3y=220,
        )
        print(f"RP (29yo, 5.150 ST, 1.5 WAR): ${pred:.3f}M (Year {tier})")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect arbitration model results and make sample predictions"
    )
    parser.add_argument(
        "--player-type",
        choices=["pitcher", "batter", "all"],
        default="all",
        help="Player type model to inspect: pitcher, batter, or all (default: all)",
    )

    args = parser.parse_args()

    if args.player_type == "all":
        # Inspect both models
        for player_type in ["pitcher", "batter"]:
            print("\n" + "#" * 70)
            print(f"# {player_type.upper()} MODEL")
            print("#" * 70)

            model = inspect_model(player_type=player_type)
            show_example_predictions(model, player_type)
    else:
        model = inspect_model(player_type=args.player_type)
        show_example_predictions(model, args.player_type)


if __name__ == "__main__":
    main()
