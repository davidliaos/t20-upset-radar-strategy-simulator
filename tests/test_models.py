from pathlib import Path

from src.data_prep import (
    assign_favorite_underdog_from_elo,
    build_team1_win_target,
    time_based_split,
)
from src.features import build_pre_match_feature_frame
from src.models import (
    build_default_metadata,
    calibrate_classifier,
    evaluate_binary_model,
    load_model_artifacts,
    save_model_artifacts,
    train_logistic_baseline,
)


def _build_train_valid_sets(synthetic_matches_df):
    df = synthetic_matches_df.copy()
    df = build_team1_win_target(df)
    df = assign_favorite_underdog_from_elo(df)
    train_df, valid_df, _ = time_based_split(df)
    X_train, y_train = build_pre_match_feature_frame(train_df)
    X_valid, y_valid = build_pre_match_feature_frame(valid_df)
    return X_train, y_train, X_valid, y_valid


def test_logistic_and_calibrated_metrics_shape(synthetic_matches_df):
    X_train, y_train, X_valid, y_valid = _build_train_valid_sets(synthetic_matches_df)
    model = train_logistic_baseline(X_train, y_train)
    calibrated = calibrate_classifier(model, X_valid, y_valid)
    metrics = evaluate_binary_model(calibrated, X_valid, y_valid)

    assert {"roc_auc", "log_loss", "brier", "positive_rate_pred"} == set(metrics.keys())


def test_model_artifact_save_load_roundtrip(tmp_path: Path, synthetic_matches_df):
    X_train, y_train, X_valid, y_valid = _build_train_valid_sets(synthetic_matches_df)
    model = train_logistic_baseline(X_train, y_train)
    calibrated = calibrate_classifier(model, X_valid, y_valid)

    metadata = build_default_metadata()
    save_model_artifacts(
        model=calibrated,
        model_name="unit_test_model",
        metadata=metadata,
        models_dir=tmp_path,
    )

    loaded = load_model_artifacts(model_name="unit_test_model", models_dir=tmp_path)
    preds = loaded.model.predict_proba(X_valid.head(3))[:, 1]

    assert len(preds) == 3
    assert loaded.metadata["is_calibrated"] is True

