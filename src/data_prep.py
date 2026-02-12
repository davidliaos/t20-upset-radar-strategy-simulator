"""Data loading, target creation, upset labeling, and split helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from .config import EXPECTED_COLUMNS, RAW_DATA_PATH, TRAIN_END_YEAR, VALID_END_YEAR


def load_matches(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw match dataset and run minimal schema checks."""
    csv_path = RAW_DATA_PATH if path is None else path
    df = pd.read_csv(csv_path)

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Some rows have invalid dates; fix before modeling.")

    if "match_result" in df.columns:
        # Keep only completed matches for training labels by default.
        df = df[df["match_result"].astype(str).str.lower() == "completed"].copy()

    return df.sort_values("date").reset_index(drop=True)


def build_team1_win_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create a binary target where 1 means team1 won the match."""
    out = df.copy()
    out = out[out["winner"].notna()].copy()
    out = out[(out["winner"] == out["team1"]) | (out["winner"] == out["team2"])].copy()
    out["team1_win"] = (out["winner"] == out["team1"]).astype(int)
    return out


def assign_favorite_underdog_from_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Assign favorite/underdog using pre-match elo columns."""
    out = df.copy()
    out["elo_team1"] = pd.to_numeric(out["elo_team1"], errors="coerce")
    out["elo_team2"] = pd.to_numeric(out["elo_team2"], errors="coerce")
    out = out.dropna(subset=["elo_team1", "elo_team2"]).copy()
    out["favorite_team"] = out["team1"].where(out["elo_team1"] >= out["elo_team2"], out["team2"])
    out["underdog_team"] = out["team2"].where(out["elo_team1"] >= out["elo_team2"], out["team1"])
    out["is_upset"] = (out["winner"] == out["underdog_team"]).astype(int)
    return out


def time_based_split(
    df: pd.DataFrame,
    train_end_year: int = TRAIN_END_YEAR,
    valid_end_year: int = VALID_END_YEAR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically into train, validation, and test partitions."""
    if "year" in df.columns:
        year = df["year"]
    else:
        year = pd.to_datetime(df["date"]).dt.year

    train_df = df[year <= train_end_year].copy()
    valid_df = df[(year > train_end_year) & (year <= valid_end_year)].copy()
    test_df = df[year > valid_end_year].copy()

    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("One or more time splits are empty; adjust split boundaries.")

    return train_df, valid_df, test_df

