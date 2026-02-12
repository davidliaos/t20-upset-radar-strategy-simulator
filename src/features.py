"""Feature engineering utilities for upset and strategy modeling."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd

NUMERIC_FEATURES: List[str] = [
    "elo_team1",
    "elo_team2",
    "elo_diff",
    "team1_form_5",
    "team2_form_5",
    "team1_form_10",
    "team2_form_10",
    "h2h_win_pct",
]

CATEGORICAL_FEATURES: List[str] = [
    "team1",
    "team2",
    "match_stage",
    "venue",
    "toss_winner",
    "toss_decision",
]


def build_pre_match_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return leakage-safe feature matrix and target vector."""
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["team1_win"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    out = df.copy()
    # Avoid nulls in key categorical controls used by simulator and baseline models.
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].fillna("unknown").astype(str)

    X = out[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = out["team1_win"].astype(int)
    return X, y


def add_context_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple strategy-oriented interaction features."""
    out = df.copy()
    out["toss_stage_interaction"] = out["toss_decision"].astype(str) + "_" + out["match_stage"].astype(str)
    out["venue_stage_interaction"] = out["venue"].astype(str) + "_" + out["match_stage"].astype(str)
    return out

