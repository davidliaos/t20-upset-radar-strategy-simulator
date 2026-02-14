from src.data_prep import (
    assign_favorite_underdog_from_elo,
    build_team1_win_target,
)
from src.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    add_context_interactions,
    build_pre_match_feature_frame,
)


def test_build_feature_frame_columns_and_target(synthetic_matches_df):
    df = synthetic_matches_df.copy()
    df = build_team1_win_target(df)
    df = assign_favorite_underdog_from_elo(df)
    X, y = build_pre_match_feature_frame(df)

    assert list(X.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert len(X) == len(y)
    assert "winner" not in X.columns
    assert "match_result" not in X.columns


def test_context_interactions_created(synthetic_matches_df):
    df = synthetic_matches_df.copy()
    out = add_context_interactions(df)
    assert "toss_stage_interaction" in out.columns
    assert "venue_stage_interaction" in out.columns

