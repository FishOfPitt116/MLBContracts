"""Pre-arbitration contract prediction model."""

import json
import os

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from models.pre_arb import config
from models.pre_arb.features import (
    get_preprocessor,
    get_feature_columns,
    load_and_filter_data,
    get_features_and_target,
)
from models.evaluation import calculate_all_metrics, format_metrics_report


class PreArbModel:
    """Model to predict pre-arbitration player salaries."""

    def __init__(self, model_type="ridge"):
        """
        Initialize the pre-arb model.

        Args:
            model_type: "ridge" or "random_forest"
        """
        self.model_type = model_type
        self.pipeline = None
        self.metrics = None
        self.cv_scores = None

    def _create_model(self):
        """Create the underlying sklearn model based on model_type."""
        if self.model_type == "ridge":
            return Ridge(alpha=config.RIDGE_ALPHA)
        elif self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=config.RANDOM_FOREST_N_ESTIMATORS,
                max_depth=config.RANDOM_FOREST_MAX_DEPTH,
                random_state=config.RANDOM_FOREST_RANDOM_STATE,
                n_jobs=-1,
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
        preprocessor = get_preprocessor()
        model = self._create_model()

        self.pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

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
            preprocessor = get_preprocessor()
            model = self._create_model()
            self.pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ])

        # Use negative MAE for cross_val_score (sklearn convention)
        scores = cross_val_score(
            self.pipeline, X, y,
            cv=cv,
            scoring="neg_mean_absolute_error"
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

        # Save pipeline (includes preprocessor and model)
        model_path = os.path.join(artifacts_dir, config.MODEL_FILENAME)
        dump(self.pipeline, model_path)

        # Save metrics if available
        if self.metrics is not None:
            metrics_path = os.path.join(artifacts_dir, config.METRICS_FILENAME)
            metrics_to_save = self.metrics.copy()

            # Add CV scores if available
            if self.cv_scores is not None:
                metrics_to_save["cv_mae_mean"] = float(np.mean(self.cv_scores))
                metrics_to_save["cv_mae_std"] = float(np.std(self.cv_scores))
                metrics_to_save["cv_mae_scores"] = [float(s) for s in self.cv_scores]

            metrics_to_save["model_type"] = self.model_type

            with open(metrics_path, "w") as f:
                json.dump(metrics_to_save, f, indent=2)

    @classmethod
    def load(cls, artifacts_dir=None):
        """
        Load model from disk.

        Args:
            artifacts_dir: Directory to load from (default from config)

        Returns:
            PreArbModel: Loaded model instance
        """
        if artifacts_dir is None:
            artifacts_dir = config.ARTIFACTS_DIR

        model_path = os.path.join(artifacts_dir, config.MODEL_FILENAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        instance = cls()
        instance.pipeline = load(model_path)

        # Try to load metrics
        metrics_path = os.path.join(artifacts_dir, config.METRICS_FILENAME)
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                instance.metrics = json.load(f)
                instance.model_type = instance.metrics.get("model_type", "unknown")

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

        # Get feature names
        feature_names = []
        numeric_features = config.PERSONAL_FEATURES + config.BATTER_FEATURES + config.PITCHER_FEATURES
        feature_names.extend(numeric_features)

        # Get one-hot encoded position names
        cat_transformer = preprocessor.named_transformers_["cat"]
        encoder = cat_transformer.named_steps["encoder"]
        cat_features = encoder.get_feature_names_out([config.POSITION_FEATURE])
        feature_names.extend(cat_features)

        importances = model.feature_importances_
        return dict(zip(feature_names, importances))


def train_and_evaluate(model_type="ridge", verbose=True):
    """
    Full training and evaluation pipeline.

    Args:
        model_type: "ridge" or "random_forest"
        verbose: Print progress and results

    Returns:
        tuple: (model, metrics dict)
    """
    if verbose:
        print(f"Loading and filtering data...")

    df = load_and_filter_data()
    X, y = get_features_and_target(df)

    if verbose:
        print(f"Dataset size: {len(df):,} contracts")
        print(f"Features: {len(get_feature_columns())} columns")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )

    if verbose:
        print(f"Training set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")

    # Train model
    if verbose:
        print(f"\nTraining {model_type} model...")

    model = PreArbModel(model_type=model_type)
    model.train(X_train, y_train)

    # Evaluate
    if verbose:
        print("Evaluating on test set...")

    metrics = model.evaluate(X_test, y_test)

    # Cross-validation
    if verbose:
        print(f"Running {config.CV_FOLDS}-fold cross-validation...")

    cv_results = model.cross_validate(X_train, y_train)
    metrics.update(cv_results)

    if verbose:
        print("\n" + format_metrics_report(metrics))
        print(f"\nCross-validation MAE: {cv_results['cv_mae_mean']:.4f} ± {cv_results['cv_mae_std']:.4f}")

        # Check thresholds
        mae_pass = metrics["mae"] <= config.MAE_THRESHOLD
        pct_pass = metrics["pct_within_tolerance"] >= config.PCT_WITHIN_TOLERANCE_THRESHOLD

        print("\nThreshold checks:")
        print(f"  MAE ≤ ${config.MAE_THRESHOLD}M: {'PASS ✓' if mae_pass else 'FAIL ✗'}")
        print(f"  95% within ±${config.TOLERANCE}M: {'PASS ✓' if pct_pass else 'FAIL ✗'}")

    return model, metrics
