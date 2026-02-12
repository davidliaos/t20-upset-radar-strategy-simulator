"""Script-level tests for train_and_save.py argument paths and output files."""

from __future__ import annotations

import json
from pathlib import Path

import scripts.train_and_save as train_script
import src.data_prep as data_prep


def test_parse_args_defaults() -> None:
    """Default args resolve to expected values."""
    args = train_script.parse_args([])
    assert args.model == "logistic"
    assert args.artifact_name == "baseline_logistic_calibrated"
    assert args.no_calibrate is False


def test_parse_args_custom_paths() -> None:
    """Custom models-dir and output-dir are accepted."""
    args = train_script.parse_args([
        "--models-dir", "/tmp/models",
        "--output-dir", "/tmp/out",
    ])
    assert args.models_dir == Path("/tmp/models")
    assert args.output_dir == Path("/tmp/out")


def test_parse_args_artifact_name() -> None:
    """Custom artifact-name is accepted."""
    args = train_script.parse_args(["--artifact-name", "test_model"])
    assert args.artifact_name == "test_model"


def test_run_writes_to_tmp_path(tmp_path: Path, synthetic_matches_df, monkeypatch) -> None:
    """Run writes model, metadata, and metrics to given tmp_path without polluting repo."""
    monkeypatch.setattr(data_prep, "load_matches", lambda: synthetic_matches_df.copy())
    args = train_script.parse_args([
        "--models-dir", str(tmp_path),
        "--output-dir", str(tmp_path),
        "--artifact-name", "pytest_artifact",
        "--no-calibrate",
    ])

    train_script.run(args)

    model_path = tmp_path / "pytest_artifact.joblib"
    metadata_path = tmp_path / "pytest_artifact_metadata.json"
    metrics_path = tmp_path / "pytest_artifact_metrics.json"

    assert model_path.exists()
    assert metadata_path.exists()
    assert metrics_path.exists()


def test_run_output_files_valid(tmp_path: Path, synthetic_matches_df, monkeypatch) -> None:
    """Output files contain valid structure and expected keys."""
    monkeypatch.setattr(data_prep, "load_matches", lambda: synthetic_matches_df.copy())
    args = train_script.parse_args([
        "--models-dir", str(tmp_path),
        "--output-dir", str(tmp_path),
        "--artifact-name", "pytest_valid",
        "--no-calibrate",
    ])

    train_script.run(args)

    metadata_path = tmp_path / "pytest_valid_metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert "model_family" in metadata
    assert "feature_columns" in metadata
    assert metadata["model_family"] in ("logistic", "logistic_regression", "xgboost")

    metrics_path = tmp_path / "pytest_valid_metrics.json"
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert "valid" in metrics
    assert "test" in metrics
    assert "roc_auc" in metrics["valid"]
    assert "brier" in metrics["valid"]


def test_run_artifact_name_affects_paths(tmp_path: Path, synthetic_matches_df, monkeypatch) -> None:
    """Custom artifact-name alters output filenames."""
    artifact = "custom_prefix"
    monkeypatch.setattr(data_prep, "load_matches", lambda: synthetic_matches_df.copy())
    args = train_script.parse_args([
        "--models-dir", str(tmp_path),
        "--output-dir", str(tmp_path),
        "--artifact-name", artifact,
        "--no-calibrate",
    ])

    train_script.run(args)

    assert (tmp_path / f"{artifact}.joblib").exists()
    assert (tmp_path / f"{artifact}_metadata.json").exists()
    assert (tmp_path / f"{artifact}_metrics.json").exists()
