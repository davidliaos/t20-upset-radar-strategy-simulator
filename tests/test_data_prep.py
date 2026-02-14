from src.data_prep import (
    assign_favorite_underdog_from_elo,
    build_team1_win_target,
    time_based_split,
)


def test_synthetic_matches_has_expected_shape_and_dates(synthetic_matches_df):
    df = synthetic_matches_df.copy()
    assert len(df) > 0
    assert "date" in df.columns
    assert df["date"].isna().sum() == 0  # date strings are present in fixture
    if "match_result" in df.columns:
        assert set(df["match_result"].str.lower().unique()) <= {"completed"}


def test_target_and_upset_columns_are_binary(synthetic_matches_df):
    df = synthetic_matches_df.copy()
    df = build_team1_win_target(df)
    df = assign_favorite_underdog_from_elo(df)

    assert set(df["team1_win"].unique()).issubset({0, 1})
    assert set(df["is_upset"].unique()).issubset({0, 1})
    assert {"favorite_team", "underdog_team"}.issubset(df.columns)


def test_time_split_non_empty_and_ordered(synthetic_matches_df):
    df = synthetic_matches_df.copy()
    train_df, valid_df, test_df = time_based_split(df)

    assert len(train_df) > 0
    assert len(valid_df) > 0
    assert len(test_df) > 0
    assert train_df["date"].max() <= valid_df["date"].max()
    assert valid_df["date"].max() <= test_df["date"].max()

