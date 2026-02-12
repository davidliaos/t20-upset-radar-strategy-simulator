import pandas as pd

from src.explain import build_counterfactual_explanation, rank_notable_upsets
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

