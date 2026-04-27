"""CLI entrypoint for training the pre-arb model."""

import argparse
import sys

from models.pre_arb.model import PreArbModel, train_and_evaluate
from models.pre_arb import config


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the pre-arbitration salary prediction model"
    )
    parser.add_argument(
        "--model-type",
        choices=["ridge", "random_forest"],
        default="ridge",
        help="Type of model to train (default: ridge)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save model artifacts after training"
    )
    parser.add_argument(
        "--artifacts-dir",
        default=config.ARTIFACTS_DIR,
        help=f"Directory to save artifacts (default: {config.ARTIFACTS_DIR})"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Train and compare both ridge and random forest models"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()
    verbose = not args.quiet

    if args.compare:
        if verbose:
            print("=" * 60)
            print("Comparing Models")
            print("=" * 60)

        results = {}

        for model_type in ["ridge", "random_forest"]:
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Training {model_type.upper()} model")
                print("=" * 60)

            model, metrics = train_and_evaluate(model_type=model_type, verbose=verbose)
            results[model_type] = {"model": model, "metrics": metrics}

        # Print comparison summary
        if verbose:
            print("\n" + "=" * 60)
            print("Model Comparison Summary")
            print("=" * 60)
            print(f"{'Metric':<30} {'Ridge':<15} {'Random Forest':<15}")
            print("-" * 60)

            ridge_m = results["ridge"]["metrics"]
            rf_m = results["random_forest"]["metrics"]

            print(f"{'MAE':<30} ${ridge_m['mae']:.4f}M{'':<7} ${rf_m['mae']:.4f}M")
            print(f"{'RMSE':<30} ${ridge_m['rmse']:.4f}M{'':<7} ${rf_m['rmse']:.4f}M")
            print(f"{'R²':<30} {ridge_m['r2']:.4f}{'':<10} {rf_m['r2']:.4f}")
            print(f"{'% within tolerance':<30} {ridge_m['pct_within_tolerance']*100:.2f}%{'':<9} {rf_m['pct_within_tolerance']*100:.2f}%")
            print(f"{'CV MAE (mean ± std)':<30} {ridge_m['cv_mae_mean']:.4f}±{ridge_m['cv_mae_std']:.3f}{'':<3} {rf_m['cv_mae_mean']:.4f}±{rf_m['cv_mae_std']:.3f}")

            # Recommend best model
            if ridge_m["mae"] <= rf_m["mae"]:
                best = "ridge"
            else:
                best = "random_forest"

            print(f"\nRecommended model: {best} (lower MAE)")

        if args.save:
            # Save the better model
            best_model = results[best]["model"]
            best_model.save(args.artifacts_dir)
            if verbose:
                print(f"\nSaved {best} model to {args.artifacts_dir}/")

    else:
        model, metrics = train_and_evaluate(model_type=args.model_type, verbose=verbose)

        if args.save:
            model.save(args.artifacts_dir)
            if verbose:
                print(f"\nModel artifacts saved to {args.artifacts_dir}/")

        # Return exit code based on thresholds
        mae_pass = metrics["mae"] <= config.MAE_THRESHOLD
        pct_pass = metrics["pct_within_tolerance"] >= config.PCT_WITHIN_TOLERANCE_THRESHOLD

        if not (mae_pass and pct_pass):
            if verbose:
                print("\nWarning: Model did not meet all accuracy thresholds")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
