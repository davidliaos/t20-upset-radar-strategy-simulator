"""Model training and evaluation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    DEFAULT_MODEL_ARTIFACT_NAME,
    MODELS_DIR,
    RANDOM_SEED,
    TRAIN_END_YEAR,
    VALID_END_YEAR,
)
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


@dataclass
class ModelArtifacts:
    """Container for fitted estimator and metadata."""

    model: Any
    feature_names: list[str]
    metadata: Dict[str, Any]


def build_preprocessor() -> ColumnTransformer:
    """Build a preprocessing pipeline for mixed feature types."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def train_logistic_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Train a regularized logistic regression baseline."""
    baseline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "clf",
                LogisticRegression(
                    random_state=RANDOM_SEED,
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    baseline.fit(X_train, y_train)
    return baseline


def calibrate_classifier(
    model: Pipeline, X_valid: pd.DataFrame, y_valid: pd.Series
) -> CalibratedClassifierCV:
    """Calibrate probabilities using sigmoid scaling on validation data.

    Supports both new and old scikit-learn calibration APIs.
    """
    try:
        # scikit-learn >=1.6 path.
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(model), method="sigmoid", cv=None
        )
    except ImportError:
        # older scikit-learn path.
        calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")

    calibrated.fit(X_valid, y_valid)
    return calibrated


def evaluate_binary_model(model, X_eval: pd.DataFrame, y_eval: pd.Series) -> Dict[str, float]:
    """Return a compact metric set for probability classifiers."""
    probs = model.predict_proba(X_eval)[:, 1]
    try:
        roc_auc = float(roc_auc_score(y_eval, probs))
    except ValueError:
        roc_auc = float("nan")

    return {
        "roc_auc": roc_auc,
        "log_loss": float(log_loss(y_eval, probs, labels=[0, 1])),
        "brier": float(brier_score_loss(y_eval, probs)),
        "positive_rate_pred": float(np.mean(probs >= 0.5)),
    }


def train_xgboost_baseline(X_train, y_train):
    """Train a basic XGBoost model if xgboost is installed."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is not installed. Install requirements first.") from exc

    xgb = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "clf",
                XGBClassifier(
                    random_state=RANDOM_SEED,
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                ),
            ),
        ]
    )
    xgb.fit(X_train, y_train)
    return xgb


def build_default_metadata(**overrides: Any) -> Dict[str, Any]:
    """Return metadata describing the default training setup."""
    metadata = {
        "model_family": "logistic_regression",
        "feature_columns": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "train_end_year": TRAIN_END_YEAR,
        "valid_end_year": VALID_END_YEAR,
        "is_calibrated": True,
        "trained_at_utc": datetime.now(UTC).isoformat(),
    }
    metadata.update(overrides)
    return metadata


def save_model_artifacts(
    model: Any,
    model_name: str = DEFAULT_MODEL_ARTIFACT_NAME,
    metadata: Dict[str, Any] | None = None,
    models_dir: Path = MODELS_DIR,
) -> Tuple[Path, Path]:
    """Persist model and metadata to disk."""
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{model_name}.joblib"
    metadata_path = models_dir / f"{model_name}_metadata.json"

    joblib.dump(model, model_path)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata or build_default_metadata(), f, indent=2)

    return model_path, metadata_path


def load_model_artifacts(
    model_name: str = DEFAULT_MODEL_ARTIFACT_NAME,
    models_dir: Path = MODELS_DIR,
) -> ModelArtifacts:
    """Load model and metadata from disk."""
    model_path = models_dir / f"{model_name}.joblib"
    metadata_path = models_dir / f"{model_name}_metadata.json"

    model = joblib.load(model_path)
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    feature_names = metadata.get("feature_columns", NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    return ModelArtifacts(model=model, feature_names=feature_names, metadata=metadata)

