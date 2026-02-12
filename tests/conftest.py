"""Pytest configuration for import paths."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root is on path for src/ and scripts/ imports.
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


@pytest.fixture
def synthetic_matches_df() -> pd.DataFrame:
    """Small deterministic dataset spanning train/valid/test years."""
    rows = [
        # train
        {
            "date": "2020-01-01",
            "year": 2020,
            "team1": "TeamA",
            "team2": "TeamB",
            "winner": "TeamA",
            "match_stage": "Group",
            "venue": "Venue1",
            "toss_winner": "TeamA",
            "toss_decision": "bat",
            "elo_team1": 1520.0,
            "elo_team2": 1490.0,
            "elo_diff": 30.0,
            "team1_form_5": 0.7,
            "team2_form_5": 0.4,
            "team1_form_10": 0.6,
            "team2_form_10": 0.5,
            "h2h_win_pct": 0.6,
            "match_result": "completed",
        },
        {
            "date": "2021-01-01",
            "year": 2021,
            "team1": "TeamA",
            "team2": "TeamB",
            "winner": "TeamB",
            "match_stage": "Group",
            "venue": "Venue2",
            "toss_winner": "TeamB",
            "toss_decision": "field",
            "elo_team1": 1510.0,
            "elo_team2": 1500.0,
            "elo_diff": 10.0,
            "team1_form_5": 0.6,
            "team2_form_5": 0.5,
            "team1_form_10": 0.55,
            "team2_form_10": 0.52,
            "h2h_win_pct": 0.58,
            "match_result": "completed",
        },
        # valid
        {
            "date": "2022-01-01",
            "year": 2022,
            "team1": "TeamA",
            "team2": "TeamC",
            "winner": "TeamA",
            "match_stage": "Semi",
            "venue": "Venue1",
            "toss_winner": "TeamC",
            "toss_decision": "field",
            "elo_team1": 1530.0,
            "elo_team2": 1480.0,
            "elo_diff": 50.0,
            "team1_form_5": 0.75,
            "team2_form_5": 0.45,
            "team1_form_10": 0.65,
            "team2_form_10": 0.48,
            "h2h_win_pct": 0.62,
            "match_result": "completed",
        },
        {
            "date": "2023-01-01",
            "year": 2023,
            "team1": "TeamC",
            "team2": "TeamA",
            "winner": "TeamA",
            "match_stage": "Final",
            "venue": "Venue3",
            "toss_winner": "TeamA",
            "toss_decision": "bat",
            "elo_team1": 1490.0,
            "elo_team2": 1540.0,
            "elo_diff": -50.0,
            "team1_form_5": 0.52,
            "team2_form_5": 0.78,
            "team1_form_10": 0.5,
            "team2_form_10": 0.7,
            "h2h_win_pct": 0.4,
            "match_result": "completed",
        },
        # test
        {
            "date": "2024-01-01",
            "year": 2024,
            "team1": "TeamB",
            "team2": "TeamC",
            "winner": "TeamB",
            "match_stage": "Group",
            "venue": "Venue2",
            "toss_winner": "TeamB",
            "toss_decision": "bat",
            "elo_team1": 1505.0,
            "elo_team2": 1495.0,
            "elo_diff": 10.0,
            "team1_form_5": 0.58,
            "team2_form_5": 0.5,
            "team1_form_10": 0.56,
            "team2_form_10": 0.52,
            "h2h_win_pct": 0.51,
            "match_result": "completed",
        },
        {
            "date": "2025-01-01",
            "year": 2025,
            "team1": "TeamC",
            "team2": "TeamB",
            "winner": "TeamB",
            "match_stage": "Semi",
            "venue": "Venue2",
            "toss_winner": "TeamC",
            "toss_decision": "field",
            "elo_team1": 1498.0,
            "elo_team2": 1512.0,
            "elo_diff": -14.0,
            "team1_form_5": 0.49,
            "team2_form_5": 0.61,
            "team1_form_10": 0.5,
            "team2_form_10": 0.59,
            "h2h_win_pct": 0.47,
            "match_result": "completed",
        },
    ]
    return pd.DataFrame(rows)
