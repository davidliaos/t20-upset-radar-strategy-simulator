import pandas as pd

from src.simulation import (
    _get_global_rows,
    _get_matchup_rows,
    _get_matchup_venue_rows,
    _get_venue_rows,
    build_upset_alert,
    estimate_scenario_defaults,
    get_stage_alert_threshold,
    normalize_matchup_rows,
)


def test_normalize_matchup_rows_returns_expected_columns(synthetic_matches_df):
    df = synthetic_matches_df.copy()
    t1 = "TeamA"
    t2 = "TeamB"
    out = normalize_matchup_rows(df, t1, t2)

    expected = {
        "elo_team1",
        "elo_team2",
        "team1_form_5",
        "team2_form_5",
        "team1_form_10",
        "team2_form_10",
        "h2h_win_pct",
        "match_stage",
        "venue",
        "toss_decision",
    }
    assert expected.issubset(out.columns)


def test_estimate_scenario_defaults_range_and_keys(synthetic_matches_df):
    df = synthetic_matches_df.copy()
    t1 = "TeamA"
    t2 = "TeamB"
    venue = "Venue1"
    defaults = estimate_scenario_defaults(df, t1, t2, venue)

    for key in [
        "elo_team1",
        "elo_team2",
        "team1_form_5",
        "team2_form_5",
        "team1_form_10",
        "team2_form_10",
        "h2h_win_pct",
        "match_stage",
        "toss_decision",
    ]:
        assert key in defaults

    assert 0.0 <= float(defaults["h2h_win_pct"]) <= 1.0


def test_venue_fallback_tiers_helpers() -> None:
    """Fallback tier helpers return non-empty when data exists."""
    df = pd.DataFrame(
        [
            {"team1": "TeamA", "team2": "TeamB", "venue": "Venue1", "match_stage": "Group", "toss_decision": "bat", "elo_team1": 1600.0, "elo_team2": 1400.0, "team1_form_5": 0.9, "team2_form_5": 0.2, "team1_form_10": 0.85, "team2_form_10": 0.3, "h2h_win_pct": 0.95},
            {"team1": "TeamA", "team2": "TeamB", "venue": "Venue2", "match_stage": "Group", "toss_decision": "field", "elo_team1": 1300.0, "elo_team2": 1700.0, "team1_form_5": 0.1, "team2_form_5": 0.8, "team1_form_10": 0.2, "team2_form_10": 0.7, "h2h_win_pct": 0.15},
            {"team1": "TeamX", "team2": "TeamY", "venue": "Venue1", "match_stage": "Group", "toss_decision": "bat", "elo_team1": 1100.0, "elo_team2": 1900.0, "team1_form_5": 0.05, "team2_form_5": 0.95, "team1_form_10": 0.1, "team2_form_10": 0.9, "h2h_win_pct": 0.05},
        ]
    )
    t1 = "TeamA"
    t2 = "TeamB"
    venue = "Venue1"

    matchup_venue = _get_matchup_venue_rows(df, t1, t2, venue)
    matchup = _get_matchup_rows(df, t1, t2)
    venue_rows = _get_venue_rows(df, venue)
    global_rows = _get_global_rows(df)

    assert not matchup_venue.empty
    assert not matchup.empty
    assert not venue_rows.empty
    assert not global_rows.empty
    assert global_rows is not df


def test_estimate_scenario_defaults_metrics_bounded(synthetic_matches_df) -> None:
    """All [0,1] metrics are bounded regardless of source."""
    df = synthetic_matches_df.copy()
    t1 = "TeamA"
    t2 = "TeamB"
    venue = "unknown_venue"

    defaults = estimate_scenario_defaults(df, t1, t2, venue)

    for key in ["team1_form_5", "team2_form_5", "team1_form_10", "team2_form_10", "h2h_win_pct"]:
        assert 0.0 <= float(defaults[key]) <= 1.0, f"{key} out of bounds"


def test_estimate_scenario_defaults_empty_df() -> None:
    """Empty df returns safe defaults without error."""
    empty = pd.DataFrame()
    defaults = estimate_scenario_defaults(empty, "TeamA", "TeamB", "Venue")
    assert defaults["elo_team1"] == 1500.0
    assert defaults["h2h_win_pct"] == 0.5


def test_fallback_prefers_matchup_venue_then_matchup_then_venue() -> None:
    """Explicitly validate fallback order using distinguishable feature values."""
    df = pd.DataFrame(
        [
            # highest priority tier: matchup+venue
            {"team1": "TeamA", "team2": "TeamB", "venue": "Venue1", "match_stage": "Group", "toss_decision": "bat", "elo_team1": 2000.0, "elo_team2": 1000.0, "team1_form_5": 0.95, "team2_form_5": 0.05, "team1_form_10": 0.9, "team2_form_10": 0.1, "h2h_win_pct": 0.99},
            # matchup tier
            {"team1": "TeamA", "team2": "TeamB", "venue": "Venue2", "match_stage": "Group", "toss_decision": "field", "elo_team1": 1500.0, "elo_team2": 1500.0, "team1_form_5": 0.5, "team2_form_5": 0.5, "team1_form_10": 0.5, "team2_form_10": 0.5, "h2h_win_pct": 0.5},
            # venue tier
            {"team1": "TeamX", "team2": "TeamY", "venue": "Venue1", "match_stage": "Group", "toss_decision": "bat", "elo_team1": 1200.0, "elo_team2": 1800.0, "team1_form_5": 0.2, "team2_form_5": 0.8, "team1_form_10": 0.3, "team2_form_10": 0.7, "h2h_win_pct": 0.2},
        ]
    )
    defaults = estimate_scenario_defaults(df, "TeamA", "TeamB", "Venue1")
    assert defaults["elo_team1"] == 2000.0

    defaults_matchup = estimate_scenario_defaults(df[df["venue"] != "Venue1"], "TeamA", "TeamB", "Venue1")
    assert defaults_matchup["elo_team1"] == 1500.0

    defaults_venue = estimate_scenario_defaults(df[df["team1"] != "TeamA"], "TeamA", "TeamB", "Venue1")
    assert defaults_venue["elo_team1"] == 1200.0


def test_stage_alert_thresholds_and_levels() -> None:
    semi_t = get_stage_alert_threshold("Semi Final")
    group_t = get_stage_alert_threshold("Group Stage")
    assert semi_t < group_t

    alert = build_upset_alert(upset_risk=0.35, stage="Semi Final")
    assert alert["level"] in {"high", "medium"}

