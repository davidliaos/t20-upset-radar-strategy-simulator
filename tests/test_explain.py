import pandas as pd

from src.explain import (
    build_counterfactual_explanation,
    build_curated_upset_narratives,
    build_missed_upsets_audit,
    matchup_volatility_profile,
    rank_notable_upsets,
    summarize_curated_upset_patterns,
)
from src.models import train_logistic_baseline


def test_rank_notable_upsets_orders_by_abs_elo(synthetic_matches_df):
    df = synthetic_matches_df.copy()
    df["elo_diff"] = [5.0, 60.0, 25.0, 80.0, 10.0, 40.0]
    df["is_upset"] = [0, 1, 0, 1, 0, 1]
    out = rank_notable_upsets(df, top_n=2)
    assert len(out) == 2
    assert out.iloc[0]["elo_diff"] == 80.0


def test_counterfactual_explanation_returns_effects():
    X_train = pd.DataFrame(
        [
            {
                "team1": "TeamA",
                "team2": "TeamB",
                "match_stage": "Group",
                "venue": "Venue1",
                "toss_winner": "TeamA",
                "toss_decision": "bat",
                "elo_team1": 1550.0,
                "elo_team2": 1450.0,
                "elo_diff": 100.0,
                "team1_form_5": 0.8,
                "team2_form_5": 0.4,
                "team1_form_10": 0.75,
                "team2_form_10": 0.45,
                "h2h_win_pct": 0.7,
            },
            {
                "team1": "TeamA",
                "team2": "TeamB",
                "match_stage": "Semi",
                "venue": "Venue1",
                "toss_winner": "TeamB",
                "toss_decision": "field",
                "elo_team1": 1500.0,
                "elo_team2": 1500.0,
                "elo_diff": 0.0,
                "team1_form_5": 0.5,
                "team2_form_5": 0.5,
                "team1_form_10": 0.5,
                "team2_form_10": 0.5,
                "h2h_win_pct": 0.5,
            },
        ]
    )
    y_train = pd.Series([1, 0], dtype=int)
    model = train_logistic_baseline(X_train, y_train)

    explanation = build_counterfactual_explanation(model, X_train.head(1))
    assert "base_team1_win_prob" in explanation
    assert "counterfactuals" in explanation
    assert len(explanation["counterfactuals"]) >= 2


def test_matchup_volatility_profile_returns_expected_range(synthetic_matches_df):
    df = synthetic_matches_df.copy()
    df["is_upset"] = [0, 1, 0, 1, 0, 1]
    profile = matchup_volatility_profile(df, "TeamA", "TeamB")
    assert profile["matches"] >= 1
    assert 0.0 <= profile["upset_rate"] <= 1.0
    assert 0.0 <= profile["volatility_index"] <= 1.0


def test_build_missed_upsets_audit_returns_sorted_rows(synthetic_matches_df):
    df = synthetic_matches_df.copy()
    df["is_upset"] = [1, 1, 0, 0, 1, 0]
    df["pred_is_upset"] = [0, 1, 0, 0, 0, 0]
    df["favorite_team"] = df["team1"]
    df["team1_win_prob"] = [0.9, 0.2, 0.6, 0.7, 0.8, 0.4]
    out = build_missed_upsets_audit(df, top_n=5)
    assert len(out) == 2
    assert out.iloc[0]["favorite_confidence"] >= out.iloc[1]["favorite_confidence"]


def test_curated_narratives_and_summary(synthetic_matches_df):
    df = synthetic_matches_df.copy()
    df["is_upset"] = [0, 1, 0, 1, 0, 1]
    df["elo_diff"] = [10, -30, 5, -40, 8, -20]
    curated = build_curated_upset_narratives(df, top_n=3)
    assert "narrative" in curated.columns
    assert "favorite_team" in curated.columns
    assert curated["favorite_team"].isin(set(df["team1"]).union(set(df["team2"]))).all()
    summaries = summarize_curated_upset_patterns(curated)
    assert "stage_summary" in summaries
    assert "venue_summary" in summaries

