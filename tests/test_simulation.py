import pandas as pd

from src.simulation import (
    ScenarioInput,
    _get_city_rows,
    _get_global_rows,
    _get_matchup_rows,
    _get_matchup_venue_rows,
    _get_venue_rows,
    _infer_city_for_venue,
    build_scenario_export_payload,
    build_upset_alert,
    estimate_scenario_defaults,
    estimate_scenario_defaults_with_meta,
    get_stage_alert_threshold,
    normalize_matchup_rows,
    prior_confidence_label,
    score_scenario,
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
    df["city"] = ["City1", "City2", "City1"]
    t1 = "TeamA"
    t2 = "TeamB"
    venue = "Venue1"

    matchup_venue = _get_matchup_venue_rows(df, t1, t2, venue)
    matchup = _get_matchup_rows(df, t1, t2)
    venue_rows = _get_venue_rows(df, venue)
    city_rows = _get_city_rows(df, "City1")
    global_rows = _get_global_rows(df)

    assert not matchup_venue.empty
    assert not matchup.empty
    assert not venue_rows.empty
    assert not city_rows.empty
    assert not global_rows.empty
    assert global_rows is not df
    assert _infer_city_for_venue(df, venue) == "City1"


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


def test_estimate_scenario_defaults_with_meta_tiers() -> None:
    df = pd.DataFrame(
        [
            {"team1": "TeamA", "team2": "TeamB", "venue": "Venue1", "match_stage": "Group", "toss_decision": "bat", "elo_team1": 2000.0, "elo_team2": 1000.0, "team1_form_5": 0.95, "team2_form_5": 0.05, "team1_form_10": 0.9, "team2_form_10": 0.1, "h2h_win_pct": 0.99},
            {"team1": "TeamA", "team2": "TeamB", "venue": "Venue2", "match_stage": "Group", "toss_decision": "field", "elo_team1": 1500.0, "elo_team2": 1500.0, "team1_form_5": 0.5, "team2_form_5": 0.5, "team1_form_10": 0.5, "team2_form_10": 0.5, "h2h_win_pct": 0.5},
            {"team1": "TeamX", "team2": "TeamY", "venue": "Venue1", "match_stage": "Group", "toss_decision": "bat", "elo_team1": 1200.0, "elo_team2": 1800.0, "team1_form_5": 0.2, "team2_form_5": 0.8, "team1_form_10": 0.3, "team2_form_10": 0.7, "h2h_win_pct": 0.2},
            {"team1": "TeamX", "team2": "TeamZ", "venue": "Venue3", "match_stage": "Group", "toss_decision": "field", "elo_team1": 1250.0, "elo_team2": 1750.0, "team1_form_5": 0.25, "team2_form_5": 0.75, "team1_form_10": 0.35, "team2_form_10": 0.65, "h2h_win_pct": 0.3},
        ]
    )
    df["city"] = ["City1", "City2", "City1", "City1"]
    meta_matchup_venue = estimate_scenario_defaults_with_meta(df, "TeamA", "TeamB", "Venue1")
    assert meta_matchup_venue["source_tier"] == "matchup_venue"

    meta_matchup = estimate_scenario_defaults_with_meta(df[df["venue"] != "Venue1"], "TeamA", "TeamB", "Venue1")
    assert meta_matchup["source_tier"] == "matchup"

    meta_venue = estimate_scenario_defaults_with_meta(df[df["team1"] != "TeamA"], "TeamA", "TeamB", "Venue1")
    assert meta_venue["source_tier"] == "venue"

    city_only_df = df[(df["venue"] != "Venue1") & (df["venue"] != "Venue2")]
    meta_city = estimate_scenario_defaults_with_meta(
        city_only_df, "TeamA", "TeamB", "Venue1", city="City1"
    )
    assert meta_city["source_tier"] == "city"


def test_prior_confidence_label_mapping() -> None:
    assert prior_confidence_label("matchup_venue") == "high"
    assert prior_confidence_label("matchup") == "medium"
    assert prior_confidence_label("venue") == "medium"
    assert prior_confidence_label("city") == "low"
    assert prior_confidence_label("global") == "low"


def test_build_scenario_export_payload_shape() -> None:
    current = ScenarioInput(
        team1="TeamA",
        team2="TeamB",
        venue="Venue1",
        match_stage="Group",
        toss_winner="TeamA",
        toss_decision="bat",
        elo_team1=1500.0,
        elo_team2=1450.0,
        team1_form_5=0.6,
        team2_form_5=0.5,
        team1_form_10=0.58,
        team2_form_10=0.52,
        h2h_win_pct=0.55,
    )
    alt = ScenarioInput(**{**current.__dict__, "toss_decision": "field"})
    payload = build_scenario_export_payload(
        current_scenario=current,
        current_result={"team1_win_prob": 0.61, "team2_win_prob": 0.39, "upset_risk": 0.39},
        alternative_scenario=alt,
        alternative_result={"team1_win_prob": 0.57, "team2_win_prob": 0.43, "upset_risk": 0.43},
        priors_source_tier="matchup",
        priors_source_rows=12,
        model_name="baseline_logistic_calibrated",
        model_source="loaded",
        generated_at_utc="2026-02-12T00:00:00+00:00",
    )
    assert payload["model"]["name"] == "baseline_logistic_calibrated"
    assert payload["priors"]["confidence"] == "medium"
    assert len(payload["scenarios"]) == 2


def test_score_scenario_has_underdog_and_severity_metrics() -> None:
    class DummyModel:
        def predict_proba(self, feature_row):
            _ = feature_row
            return [[0.35, 0.65]]

    feature_row = pd.DataFrame(
        [
            {
                "elo_team1": 1700.0,
                "elo_team2": 1500.0,
            }
        ]
    )
    scored = score_scenario(DummyModel(), feature_row)
    assert "underdog_win_prob" in scored
    assert "upset_severity_index" in scored
    assert abs(scored["underdog_win_prob"] - 0.35) < 1e-9
    assert 0.0 <= scored["upset_severity_index"] <= scored["underdog_win_prob"]

