"""Project-level configuration and canonical paths."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "world_cup_last_30_years.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_ARTIFACT_NAME = "baseline_logistic_calibrated"

RANDOM_SEED = 42

# Default time-aware splits; can be adjusted after the first data audit.
TRAIN_END_YEAR = 2021
VALID_END_YEAR = 2023

# Core fields expected from the source dataset.
EXPECTED_COLUMNS = [
    "date",
    "season",
    "team1",
    "team2",
    "winner",
    "match_stage",
    "venue",
    "toss_winner",
    "toss_decision",
    "elo_team1",
    "elo_team2",
    "elo_diff",
    "team1_form_5",
    "team2_form_5",
    "team1_form_10",
    "team2_form_10",
    "h2h_win_pct",
]

