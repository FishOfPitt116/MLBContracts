"""CLI entrypoint for training the arbitration model."""

import argparse
import sys

from models.arb import config
from models.arb.model import ArbModel, format_tier_report, train_and_evaluate


def train_single_model(model_type, player_type, save, artifacts_dir, verbose):
    """Train a single model (pitcher, batter, or unified)."""
    model, metrics, tier_metrics = train_and_evaluate(
        model_type=model_type, player_type=player_type, verbose=verbose
    )

    if save:
        model.save(artifacts_dir)
        player_label = player_type or "unified"
        if verbose:
            print(f"\n{player_label.capitalize()} model artifacts saved to {artifacts_dir}/")

    return model, metrics, tier_metrics


def compare_model_types(player_type, save, artifacts_dir, verbose):
    """Compare random_forest vs gradient_boosting for a given player type."""
    player_label = player_type or "unified"

    if verbose:
        print("=" * 70)
        print(f"Comparing Model Types for {player_label.upper()} Model")
        print("=" * 70)

    results = {}

    for model_type in ["random_forest", "gradient_boosting"]:
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Training {model_type.upper().replace('_', ' ')}")
            print("=" * 70)

        model, metrics, tier_metrics = train_and_evaluate(
            model_type=model_type, player_type=player_type, verbose=verbose
        )
        results[model_type] = {
            "model": model,
            "metrics": metrics,
            "tier_metrics": tier_metrics,
        }

    # Print comparison summary
    if verbose:
        print_model_comparison(results)

    # Determine best model
    rf = results["random_forest"]
    gb = results["gradient_boosting"]

    rf_passes = sum(
        1 for t in [1, 2, 3] if rf["tier_metrics"].get(t, {}).get("pass")
    )
    gb_passes = sum(
        1 for t in [1, 2, 3] if gb["tier_metrics"].get(t, {}).get("pass")
    )

    if rf_passes > gb_passes:
        best = "random_forest"
        reason = "more tier thresholds passed"
    elif gb_passes > rf_passes:
        best = "gradient_boosting"
        reason = "more tier thresholds passed"
    elif rf["metrics"]["mae"] <= gb["metrics"]["mae"]:
        best = "random_forest"
        reason = "lower MAE"
    else:
        best = "gradient_boosting"
        reason = "lower MAE"

    if verbose:
        print(f"\nRecommended model: {best.replace('_', ' ')} ({reason})")

    if save:
        best_model = results[best]["model"]
        best_model.save(artifacts_dir)
        if verbose:
            print(f"\nSaved {best.replace('_', ' ')} model to {artifacts_dir}/")

    return results, best


def print_model_comparison(results):
    """Print comparison summary for two model types."""
    print("\n" + "=" * 70)
    print("Model Comparison Summary")
    print("=" * 70)

    rf = results["random_forest"]
    gb = results["gradient_boosting"]

    print(f"\n{'Metric':<30} {'Random Forest':<20} {'Gradient Boosting':<20}")
    print("-" * 70)

    rf_m = rf["metrics"]
    gb_m = gb["metrics"]

    print(f"{'MAE':<30} ${rf_m['mae']:.4f}M{'':<12} ${gb_m['mae']:.4f}M")
    print(f"{'RMSE':<30} ${rf_m['rmse']:.4f}M{'':<12} ${gb_m['rmse']:.4f}M")
    print(f"{'R²':<30} {rf_m['r2']:.4f}{'':<15} {gb_m['r2']:.4f}")
    print(
        f"{'% within tolerance':<30} {rf_m['pct_within_tolerance']*100:.2f}%{'':<14} "
        f"{gb_m['pct_within_tolerance']*100:.2f}%"
    )
    print(
        f"{'CV MAE (mean ± std)':<30} "
        f"{rf_m['cv_mae_mean']:.3f}±{rf_m['cv_mae_std']:.3f}{'':<8} "
        f"{gb_m['cv_mae_mean']:.3f}±{gb_m['cv_mae_std']:.3f}"
    )

    # Per-tier comparison
    print("\nPer-Tier Accuracy (% within tier tolerance):")
    print(f"{'Tier':<10} {'Random Forest':<20} {'Gradient Boosting':<20}")
    print("-" * 50)

    for tier in [1, 2, 3]:
        rf_tier = rf["tier_metrics"].get(tier, {})
        gb_tier = gb["tier_metrics"].get(tier, {})

        rf_pct = rf_tier.get("pct_within_tolerance", 0) * 100
        gb_pct = gb_tier.get("pct_within_tolerance", 0) * 100

        rf_pass = "PASS" if rf_tier.get("pass") else "FAIL"
        gb_pass = "PASS" if gb_tier.get("pass") else "FAIL"

        print(f"Year {tier:<5} {rf_pct:.1f}% ({rf_pass}){'':<8} {gb_pct:.1f}% ({gb_pass})")


def train_all_player_types(model_type, save, artifacts_dir, verbose):
    """Train both pitcher and batter models."""
    all_results = {}

    for player_type in ["pitcher", "batter"]:
        if verbose:
            print("\n" + "=" * 70)
            print(f"Training {player_type.upper()} Model")
            print("=" * 70)

        model, metrics, tier_metrics = train_and_evaluate(
            model_type=model_type, player_type=player_type, verbose=verbose
        )
        all_results[player_type] = {
            "model": model,
            "metrics": metrics,
            "tier_metrics": tier_metrics,
        }

        if save:
            model.save(artifacts_dir)
            if verbose:
                print(f"\n{player_type.capitalize()} model saved to {artifacts_dir}/")

    # Print comparison summary
    if verbose:
        print_player_type_comparison(all_results)

    return all_results


def print_player_type_comparison(results):
    """Print comparison summary for pitcher vs batter models."""
    print("\n" + "=" * 70)
    print("Pitcher vs Batter Model Comparison")
    print("=" * 70)

    pitcher = results["pitcher"]
    batter = results["batter"]

    print(f"\n{'Metric':<30} {'Pitcher Model':<20} {'Batter Model':<20}")
    print("-" * 70)

    p_m = pitcher["metrics"]
    b_m = batter["metrics"]

    print(f"{'MAE':<30} ${p_m['mae']:.4f}M{'':<12} ${b_m['mae']:.4f}M")
    print(f"{'RMSE':<30} ${p_m['rmse']:.4f}M{'':<12} ${b_m['rmse']:.4f}M")
    print(f"{'R²':<30} {p_m['r2']:.4f}{'':<15} {b_m['r2']:.4f}")
    print(
        f"{'% within tolerance':<30} {p_m['pct_within_tolerance']*100:.2f}%{'':<14} "
        f"{b_m['pct_within_tolerance']*100:.2f}%"
    )
    print(
        f"{'CV MAE (mean ± std)':<30} "
        f"{p_m['cv_mae_mean']:.3f}±{p_m['cv_mae_std']:.3f}{'':<8} "
        f"{b_m['cv_mae_mean']:.3f}±{b_m['cv_mae_std']:.3f}"
    )

    # Per-tier comparison
    print("\nPer-Tier Accuracy (% within tier tolerance):")
    print(f"{'Tier':<10} {'Pitcher Model':<20} {'Batter Model':<20}")
    print("-" * 50)

    for tier in [1, 2, 3]:
        p_tier = pitcher["tier_metrics"].get(tier, {})
        b_tier = batter["tier_metrics"].get(tier, {})

        p_pct = p_tier.get("pct_within_tolerance", 0) * 100
        b_pct = b_tier.get("pct_within_tolerance", 0) * 100

        p_pass = "PASS" if p_tier.get("pass") else "FAIL"
        b_pass = "PASS" if b_tier.get("pass") else "FAIL"

        print(f"Year {tier:<5} {p_pct:.1f}% ({p_pass}){'':<8} {b_pct:.1f}% ({b_pass})")

    # Overall pass/fail
    p_all_pass = pitcher["tier_metrics"].get("all_pass", False)
    b_all_pass = batter["tier_metrics"].get("all_pass", False)

    print(f"\n{'Overall':<10} {'ALL PASS' if p_all_pass else 'FAIL':<20} {'ALL PASS' if b_all_pass else 'FAIL':<20}")


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the arbitration salary prediction model"
    )
    parser.add_argument(
        "--model-type",
        choices=["random_forest", "gradient_boosting"],
        default="random_forest",
        help="Type of model to train (default: random_forest)",
    )
    parser.add_argument(
        "--player-type",
        choices=["pitcher", "batter", "all"],
        default="all",
        help="Player type to train model for: pitcher, batter, or all (default: all)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save model artifacts after training",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=config.ARTIFACTS_DIR,
        help=f"Directory to save artifacts (default: {config.ARTIFACTS_DIR})",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Train and compare both random_forest and gradient_boosting models",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    if args.compare:
        # Compare model types for specified player type(s)
        if args.player_type == "all":
            # Compare for both pitcher and batter
            all_results = {}
            for player_type in ["pitcher", "batter"]:
                if verbose:
                    print("\n" + "#" * 70)
                    print(f"# {player_type.upper()} MODEL COMPARISON")
                    print("#" * 70)

                results, best = compare_model_types(
                    player_type, args.save, args.artifacts_dir, verbose
                )
                all_results[player_type] = {"results": results, "best": best}
        else:
            compare_model_types(
                args.player_type if args.player_type != "all" else None,
                args.save,
                args.artifacts_dir,
                verbose,
            )
        return 0

    # Train models
    if args.player_type == "all":
        # Train both pitcher and batter models
        all_results = train_all_player_types(
            args.model_type, args.save, args.artifacts_dir, verbose
        )

        # Check if all models pass
        all_pass = True
        for player_type, result in all_results.items():
            tier_pass = result["tier_metrics"].get("all_pass", False)
            mae_pass = result["metrics"]["mae"] <= config.MAE_THRESHOLD
            if not tier_pass or not mae_pass:
                all_pass = False
                if verbose and not tier_pass:
                    print(f"\nWarning: {player_type} model did not pass all tier thresholds")
                if verbose and not mae_pass:
                    print(
                        f"\nWarning: {player_type} model MAE (${result['metrics']['mae']:.3f}M) "
                        f"exceeds threshold (${config.MAE_THRESHOLD}M)"
                    )

        return 0 if all_pass else 1

    else:
        # Train single player type model
        player_type = args.player_type if args.player_type != "all" else None
        model, metrics, tier_metrics = train_single_model(
            args.model_type, player_type, args.save, args.artifacts_dir, verbose
        )

        # Return exit code based on tier thresholds
        all_pass = tier_metrics.get("all_pass", False)
        mae_pass = metrics["mae"] <= config.MAE_THRESHOLD

        if not all_pass:
            if verbose:
                print("\nWarning: Not all tier accuracy thresholds were met")
            return 1

        if not mae_pass:
            if verbose:
                print(f"\nWarning: Overall MAE (${metrics['mae']:.3f}M) exceeds threshold (${config.MAE_THRESHOLD}M)")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
