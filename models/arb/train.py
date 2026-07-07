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


ALL_MODEL_TYPES = list(config.MODEL_PROFILES.keys())


def compare_model_types(player_type, model_types, save, artifacts_dir, verbose):
    """Train and compare a set of model types for a given player type."""
    player_label = player_type or "unified"

    if verbose:
        print("=" * 70)
        print(f"Comparing model types for {player_label.upper()}: {', '.join(model_types)}")
        print("=" * 70)

    results = {}

    for model_type in model_types:
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

    if verbose:
        print_model_comparison(results, model_types)

    # Pick best by tier passes, then MAE
    best = min(
        model_types,
        key=lambda mt: (
            -sum(1 for t in [1, 2, 3] if results[mt]["tier_metrics"].get(t, {}).get("pass")),
            results[mt]["metrics"]["mae"],
        ),
    )

    if verbose:
        print(f"\nRecommended model: {best.replace('_', ' ')} (lowest MAE among most tier passes)")

    if save:
        results[best]["model"].save(artifacts_dir)
        if verbose:
            print(f"Saved {best} model to {artifacts_dir}/")

    return results, best


def print_model_comparison(results, model_types):
    """Print a comparison table across any number of model types."""
    col_w = 20
    header = f"{'Metric':<30}" + "".join(f"{mt.replace('_',' ').title():<{col_w}}" for mt in model_types)
    print("\n" + "=" * (30 + col_w * len(model_types)))
    print("Model Comparison Summary")
    print("=" * (30 + col_w * len(model_types)))
    print(f"\n{header}")
    print("-" * (30 + col_w * len(model_types)))

    def row(label, values):
        return f"{label:<30}" + "".join(f"{v:<{col_w}}" for v in values)

    print(row("MAE", [f"${results[mt]['metrics']['mae']:.4f}M" for mt in model_types]))
    print(row("RMSE", [f"${results[mt]['metrics']['rmse']:.4f}M" for mt in model_types]))
    print(row("R²", [f"{results[mt]['metrics']['r2']:.4f}" for mt in model_types]))
    print(row("% within tolerance",
              [f"{results[mt]['metrics']['pct_within_tolerance']*100:.2f}%" for mt in model_types]))
    print(row("CV MAE (mean ± std)",
              [f"{results[mt]['metrics']['cv_mae_mean']:.3f}±{results[mt]['metrics']['cv_mae_std']:.3f}"
               for mt in model_types]))

    print("\nPer-Tier Accuracy (% within tier tolerance):")
    tier_header = f"{'Tier':<10}" + "".join(f"{mt.replace('_',' ').title():<{col_w}}" for mt in model_types)
    print(tier_header)
    print("-" * (10 + col_w * len(model_types)))

    for tier in [1, 2, 3]:
        vals = []
        for mt in model_types:
            t = results[mt]["tier_metrics"].get(tier, {})
            pct = t.get("pct_within_tolerance", 0) * 100
            status = "PASS" if t.get("pass") else "FAIL"
            vals.append(f"{pct:.1f}% ({status})")
        print(f"Year {tier:<5}" + "".join(f"{v:<{col_w}}" for v in vals))


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
        choices=ALL_MODEL_TYPES,
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
        model_types = ALL_MODEL_TYPES

        if args.player_type == "all":
            all_results = {}
            for player_type in ["pitcher", "batter"]:
                if verbose:
                    print("\n" + "#" * 70)
                    print(f"# {player_type.upper()} MODEL COMPARISON")
                    print("#" * 70)

                results, best = compare_model_types(
                    player_type, model_types, args.save, args.artifacts_dir, verbose
                )
                all_results[player_type] = {"results": results, "best": best}
        else:
            compare_model_types(
                args.player_type if args.player_type != "all" else None,
                model_types,
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
