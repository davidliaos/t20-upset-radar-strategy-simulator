"""Unit tests for app-level helper logic."""

from __future__ import annotations

import pandas as pd

from app import (
    _default_matchup_for_risk_context,
    _recommended_venue_counts,
    _recommended_venue_options,
    build_date_label,
    build_event_label,
)


def test_default_matchup_prefers_frequent_pair() -> None:
    df = pd.DataFrame(
        [
            {"team1": "A", "team2": "B", "is_upset": 0},
            {"team1": "B", "team2": "A", "is_upset": 1},
            {"team1": "A", "team2": "C", "is_upset": 0},
        ]
    )
    team1, team2 = _default_matchup_for_risk_context(df, teams=["A", "B", "C"])
    assert {team1, team2} == {"A", "B"}


def test_recommended_venue_order_and_counts() -> None:
    df = pd.DataFrame(
        [
            {"team1": "A", "team2": "B", "venue": "V1"},
            {"team1": "A", "team2": "C", "venue": "V2"},
            {"team1": "B", "team2": "C", "venue": "V3"},
            {"team1": "A", "team2": "B", "venue": "V4"},
        ]
    )
    options = _recommended_venue_options(df, "A", "B")
    pair_count, shared_count, union_count = _recommended_venue_counts(df, "A", "B")
    assert options[:2] == ["V1", "V4"]
    assert pair_count == 2
    assert shared_count == 2
    assert union_count >= shared_count


def test_event_and_date_label_fallbacks() -> None:
    row_blank_tournament = pd.Series({"tournament_name": "   ", "venue": "Venue Y", "match_stage": "Group"})
    assert build_event_label(row_blank_tournament) == "Venue Y"

    row_event = pd.Series({"tournament_name": None, "venue": "Stadium X", "match_stage": "Semi"})
    assert build_event_label(row_event) == "Stadium X"

    row_stage = pd.Series({"tournament_name": None, "venue": None, "match_stage": "Final"})
    assert build_event_label(row_stage) == "Final"

    assert build_date_label("2025-03-09") == "2025-03-09"
    assert build_date_label("not-a-date") == "Unknown Date"

