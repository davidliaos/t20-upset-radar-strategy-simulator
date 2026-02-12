"""Train, calibrate, evaluate, and persist model artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    default_model_name = "baseline_logistic_calibrated"
    parser = argparse.ArgumentParser(description="Train and save Upset Radar model artifacts.")
    parser.add_argument(
        "--model",
        choices=["logistic", "xgboost"],
        default="logistic",
        help="Base model family to train before optional calibration.",
    )
    parser.add_argument(
        "--artifact-name",
        default=default_model_name,
        help="Artifact name prefix for model and metadata files.",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Disable calibration step.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Directory for model and metadata artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Directory for metrics JSON output.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace | None = None) -> None:
    """Run training and save artifacts. Accepts pre-parsed args for testing."""
    parsed = args if args is not None else parse_args()
    _execute(parsed)


def _execute(args: argparse.Namespace) -> None:
    """Execute training pipeline with given arguments."""
    from src.data_prep import (
        assign_favorite_underdog_from_elo,
        build_team1_win_target,
        load_matches,
        time_based_split,
    )
    from src.features import build_pre_match_feature_frame
    from src.models import (
        build_default_metadata,
        calibrate_classifier,
        evaluate_binary_model,
        save_model_artifacts,
        train_logistic_baseline,
        train_xgboost_baseline,
    )

    do_calibrate = not args.no_calibrate

    df = load_matches()
    df = build_team1_win_target(df)
    df = assign_favorite_underdog_from_elo(df)
    train_df, valid_df, test_df = time_based_split(df)

    X_train, y_train = build_pre_match_feature_frame(train_df)
    X_valid, y_valid = build_pre_match_feature_frame(valid_df)
    X_test, y_test = build_pre_match_feature_frame(test_df)

    if args.model == "xgboost":
        base_model = train_xgboost_baseline(X_train, y_train)
    else:
        base_model = train_logistic_baseline(X_train, y_train)

    model = calibrate_classifier(base_model, X_valid, y_valid) if do_calibrate else base_model

    valid_metrics = evaluate_binary_model(model, X_valid, y_valid)
    test_metrics = evaluate_binary_model(model, X_test, y_test)

    metadata = build_default_metadata(
        model_family=args.model,
        is_calibrated=do_calibrate,
        valid_metrics=valid_metrics,
        test_metrics=test_metrics,
    )
    model_path, metadata_path = save_model_artifacts(
        model=model,
        model_name=args.artifact_name,
        metadata=metadata,
        models_dir=args.models_dir,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"{args.artifact_name}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"valid": valid_metrics, "test": test_metrics}, f, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Validation metrics: {valid_metrics}")
    print(f"Test metrics: {test_metrics}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()

