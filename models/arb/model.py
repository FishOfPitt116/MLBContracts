"""Arbitration contract prediction model."""

import json
import os

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from models.arb import config
from models.arb.features import (
    add_arb_year_column,
    get_feature_columns,
    get_features_and_target,
    get_preprocessor,
    load_and_filter_data,
)
from models.evaluation import calculate_all_metrics, format_metrics_report
from models.preprocessing import normalize_service_time


class ArbModel:
    """Model to predict arbitration player salaries."""

    def __init__(self, model_type="random_forest", player_type=None):
        """
        Initialize the arbitration model.

        Args:
            model_type: "random_forest" or "gradient_boosting"
            player_type: "pitcher", "batter", or None (unified model)
        """
        self.model_type = model_type
        self.player_type = player_type
        self.pipeline = None
        self.metrics = None
        self.tier_metrics = None
        self.cv_scores = None

    def _create_model(self):
        """Create the underlying sklearn model based on model_type."""
        if self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=config.RANDOM_FOREST_N_ESTIMATORS,
                max_depth=config.RANDOM_FOREST_MAX_DEPTH,
                min_samples_split=config.RANDOM_FOREST_MIN_SAMPLES_SPLIT,
                min_samples_leaf=config.RANDOM_FOREST_MIN_SAMPLES_LEAF,
                random_state=config.RANDOM_FOREST_RANDOM_STATE,
                n_jobs=-1,
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=config.GRADIENT_BOOSTING_N_ESTIMATORS,
                max_depth=config.GRADIENT_BOOSTING_MAX_DEPTH,
                learning_rate=config.GRADIENT_BOOSTING_LEARNING_RATE,
                random_state=config.GRADIENT_BOOSTING_RANDOM_STATE,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X_train, y_train):
        """
        Train the model.

        Args:
            X_train: Training features DataFrame
            y_train: Training target Series
        """
        preprocessor = get_preprocessor(self.player_type)
        model = self._create_model()

        self.pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", model)]
        )

        self.pipeline.fit(X_train, y_train)

    def predict(self, X):
        """
        Make predictions.

        Args:
            X: Features DataFrame

        Returns:
            np.ndarray: Predicted values
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet")
        return self.pipeline.predict(X)

    def evaluate(self, X_test, y_test, tolerance=None):
        """
        Evaluate model on test data.

        Args:
            X_test: Test features DataFrame
            y_test: Test target Series
            tolerance: Tolerance for pct_within_tolerance (default from config)

        Returns:
            dict: Evaluation metrics
        """
        if tolerance is None:
            tolerance = config.TOLERANCE

        y_pred = self.predict(X_test)
        self.metrics = calculate_all_metrics(y_test, y_pred, tolerance)
        return self.metrics

    def evaluate_by_tier(self, X_test, y_test):
        """
        Evaluate model performance by arb year tier.

        Args:
            X_test: Test features DataFrame (must have service_time column)
            y_test: Test target Series

        Returns:
            dict: Per-tier metrics with pass/fail status
        """
        y_pred = self.predict(X_test)

        # Add arb_year based on service_time
        X_with_tier = add_arb_year_column(X_test)

        tier_results = {}
        all_tiers_pass = True

        for tier in [1, 2, 3]:
            tier_mask = X_with_tier["arb_year"] == tier
            tier_count = tier_mask.sum()

            if tier_count == 0:
                tier_results[tier] = {
                    "n_samples": 0,
                    "pass": None,
                    "message": "No samples in test set",
                }
                continue

            tier_y_true = y_test[tier_mask].values
            tier_y_pred = y_pred[tier_mask]

            tier_tolerance = config.TIER_THRESHOLDS[tier]["tolerance"]
            errors = np.abs(tier_y_true - tier_y_pred)
            within_tolerance = errors <= tier_tolerance
            pct_within = np.mean(within_tolerance)

            tier_pass = pct_within >= config.PCT_WITHIN_TIER_TOLERANCE_THRESHOLD
            if not tier_pass:
                all_tiers_pass = False

            tier_results[tier] = {
                "n_samples": int(tier_count),
                "avg_actual": float(np.mean(tier_y_true)),
                "avg_predicted": float(np.mean(tier_y_pred)),
                "tolerance": tier_tolerance,
                "mae": float(np.mean(errors)),
                "pct_within_tolerance": float(pct_within),
                "pass": tier_pass,
            }

        tier_results["all_pass"] = all_tiers_pass
        self.tier_metrics = tier_results
        return tier_results

    def cross_validate(self, X, y, cv=None):
        """
        Perform cross-validation.

        Args:
            X: Features DataFrame
            y: Target Series
            cv: Number of folds (default from config)

        Returns:
            dict: Cross-validation results
        """
        if cv is None:
            cv = config.CV_FOLDS

        if self.pipeline is None:
            preprocessor = get_preprocessor(self.player_type)
            model = self._create_model()
            self.pipeline = Pipeline(
                steps=[("preprocessor", preprocessor), ("model", model)]
            )

        # Use negative MAE for cross_val_score (sklearn convention)
        scores = cross_val_score(
            self.pipeline, X, y, cv=cv, scoring="neg_mean_absolute_error"
        )

        # Convert back to positive MAE
        self.cv_scores = -scores

        return {
            "cv_mae_mean": np.mean(self.cv_scores),
            "cv_mae_std": np.std(self.cv_scores),
            "cv_mae_scores": self.cv_scores.tolist(),
        }

    def save(self, artifacts_dir=None):
        """
        Save model artifacts to disk.

        Args:
            artifacts_dir: Directory to save to (default from config)
        """
        if artifacts_dir is None:
            artifacts_dir = config.ARTIFACTS_DIR

        os.makedirs(artifacts_dir, exist_ok=True)

        # Get appropriate filenames based on player type
        model_filename = config.get_model_filename(self.player_type)
        metrics_filename = config.get_metrics_filename(self.player_type)

        # Save pipeline (includes preprocessor and model)
        model_path = os.path.join(artifacts_dir, model_filename)
        dump(self.pipeline, model_path)

        # Save metrics if available
        metrics_to_save = {
            "model_type": self.model_type,
            "player_type": self.player_type,
        }

        if self.metrics is not None:
            metrics_to_save.update(self.metrics)

        if self.tier_metrics is not None:
            # Convert tier keys to strings and numpy types for JSON
            tier_metrics_json = {}
            for k, v in self.tier_metrics.items():
                if k == "all_pass":
                    continue
                # Convert numpy bool to Python bool in nested dict
                tier_dict = {}
                for inner_k, inner_v in v.items():
                    if isinstance(inner_v, (np.bool_, np.integer)):
                        tier_dict[inner_k] = bool(inner_v) if isinstance(inner_v, np.bool_) else int(inner_v)
                    else:
                        tier_dict[inner_k] = inner_v
                tier_metrics_json[str(k)] = tier_dict
            tier_metrics_json["all_pass"] = bool(self.tier_metrics.get("all_pass", False))
            metrics_to_save["tier_metrics"] = tier_metrics_json

        if self.cv_scores is not None:
            metrics_to_save["cv_mae_mean"] = float(np.mean(self.cv_scores))
            metrics_to_save["cv_mae_std"] = float(np.std(self.cv_scores))
            metrics_to_save["cv_mae_scores"] = [float(s) for s in self.cv_scores]

        metrics_path = os.path.join(artifacts_dir, metrics_filename)
        with open(metrics_path, "w") as f:
            json.dump(metrics_to_save, f, indent=2)

    @classmethod
    def load(cls, artifacts_dir=None, player_type=None):
        """
        Load model from disk.

        Args:
            artifacts_dir: Directory to load from (default from config)
            player_type: "pitcher", "batter", or None (unified model)

        Returns:
            ArbModel: Loaded model instance
        """
        if artifacts_dir is None:
            artifacts_dir = config.ARTIFACTS_DIR

        # Get appropriate filenames based on player type
        model_filename = config.get_model_filename(player_type)
        metrics_filename = config.get_metrics_filename(player_type)

        model_path = os.path.join(artifacts_dir, model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        instance = cls(player_type=player_type)
        instance.pipeline = load(model_path)

        # Try to load metrics
        metrics_path = os.path.join(artifacts_dir, metrics_filename)
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                saved_metrics = json.load(f)
                instance.model_type = saved_metrics.get("model_type", "unknown")
                instance.player_type = saved_metrics.get("player_type", player_type)
                instance.metrics = {
                    k: v
                    for k, v in saved_metrics.items()
                    if k not in ["model_type", "player_type", "tier_metrics", "cv_mae_mean", "cv_mae_std", "cv_mae_scores"]
                }
                instance.tier_metrics = saved_metrics.get("tier_metrics")

        return instance

    def get_feature_importances(self):
        """
        Get feature importances (for tree-based models).

        Returns:
            dict: Feature name to importance mapping, or None if not available
        """
        if self.pipeline is None:
            return None

        model = self.pipeline.named_steps["model"]
        if not hasattr(model, "feature_importances_"):
            return None

        preprocessor = self.pipeline.named_steps["preprocessor"]

        # Get feature names based on player type
        feature_names = []

        if self.player_type == "pitcher":
            numeric_features = (
                config.PITCHER_PERSONAL_FEATURES + config.PITCHER_FEATURES
            )
            feature_names.extend(numeric_features)
            # No categorical features for pitcher model
        elif self.player_type == "batter":
            numeric_features = (
                config.BATTER_PERSONAL_FEATURES + config.BATTER_FEATURES
            )
            feature_names.extend(numeric_features)
            # Get one-hot encoded position names
            cat_transformer = preprocessor.named_transformers_["cat"]
            encoder = cat_transformer.named_steps["encoder"]
            cat_features = encoder.get_feature_names_out([config.POSITION_FEATURE])
            feature_names.extend(cat_features)
        else:
            # Unified model (legacy)
            numeric_features = (
                config.PERSONAL_FEATURES
                + config.BATTER_FEATURES
                + config.PITCHER_FEATURES
            )
            feature_names.extend(numeric_features)
            # Get one-hot encoded position names
            cat_transformer = preprocessor.named_transformers_["cat"]
            encoder = cat_transformer.named_steps["encoder"]
            cat_features = encoder.get_feature_names_out([config.POSITION_FEATURE])
            feature_names.extend(cat_features)

        importances = model.feature_importances_
        return dict(zip(feature_names, importances))


def format_tier_report(tier_metrics):
    """
    Format tier metrics as human-readable report.

    Args:
        tier_metrics: Dictionary of per-tier metrics

    Returns:
        str: Formatted report string
    """
    lines = [
        "=" * 70,
        "Per-Tier Evaluation Results",
        "=" * 70,
        f"{'Tier':<8} {'N':<8} {'Avg Actual':<12} {'Tolerance':<12} {'MAE':<10} {'% Within':<12} {'Status':<8}",
        "-" * 70,
    ]

    for tier in [1, 2, 3]:
        if tier not in tier_metrics:
            continue

        t = tier_metrics[tier]
        if t.get("n_samples", 0) == 0:
            lines.append(f"Year {tier:<3} {'--':<8} {'--':<12} {'--':<12} {'--':<10} {'--':<12} {'N/A':<8}")
            continue

        status = "PASS" if t.get("pass") else "FAIL"
        status_icon = "PASS" if t.get("pass") else "FAIL ✗"

        lines.append(
            f"Year {tier:<3} {t['n_samples']:<8} "
            f"${t['avg_actual']:.2f}M{'':<5} "
            f"±${t['tolerance']:.2f}M{'':<5} "
            f"${t['mae']:.3f}M{'':<3} "
            f"{t['pct_within_tolerance']*100:.1f}%{'':<7} "
            f"{status_icon}"
        )

    lines.append("-" * 70)

    all_pass = tier_metrics.get("all_pass", False)
    overall_status = "ALL TIERS PASS" if all_pass else "SOME TIERS FAILED"
    lines.append(f"Overall: {overall_status}")
    lines.append("=" * 70)

    return "\n".join(lines)


def train_and_evaluate(model_type="random_forest", player_type=None, verbose=True):
    """
    Full training and evaluation pipeline.

    Args:
        model_type: "random_forest" or "gradient_boosting"
        player_type: "pitcher", "batter", or None (unified model)
        verbose: Print progress and results

    Returns:
        tuple: (model, overall_metrics, tier_metrics)
    """
    player_type_label = player_type or "unified"

    if verbose:
        print(f"Loading and filtering data for {player_type_label} model...")

    df = load_and_filter_data(player_type=player_type)
    X, y = get_features_and_target(df, player_type=player_type)

    if verbose:
        print(f"Dataset size: {len(df):,} contracts")
        print(f"Features: {len(get_feature_columns(player_type))} columns")

    # Train/test split (stratified by normalized service time buckets)
    # Create stratification bins based on service time (already normalized in prepare_features)
    X_with_bins = X.copy()
    X_with_bins["strat_bin"] = pd.cut(
        X_with_bins["service_time"],
        bins=[0, 4, 5, 10],
        labels=["1", "2", "3"],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=X_with_bins["strat_bin"],
    )

    if verbose:
        print(f"Training set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")

    # Train model
    if verbose:
        print(f"\nTraining {model_type} model...")

    model = ArbModel(model_type=model_type, player_type=player_type)
    model.train(X_train, y_train)

    # Evaluate overall
    if verbose:
        print("Evaluating on test set...")

    metrics = model.evaluate(X_test, y_test)

    # Evaluate by tier
    if verbose:
        print("Evaluating by tier...")

    tier_metrics = model.evaluate_by_tier(X_test, y_test)

    # Cross-validation
    if verbose:
        print(f"Running {config.CV_FOLDS}-fold cross-validation...")

    cv_results = model.cross_validate(X_train, y_train)
    metrics.update(cv_results)

    if verbose:
        print("\n" + format_metrics_report(metrics))
        print(f"\nCross-validation MAE: {cv_results['cv_mae_mean']:.4f} ± {cv_results['cv_mae_std']:.4f}")
        print("\n" + format_tier_report(tier_metrics))

    return model, metrics, tier_metrics
