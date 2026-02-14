"""Minimal app-level runtime smoke tests."""

from __future__ import annotations

import numpy as np

import app
from src.data_prep import build_team1_win_target


class _DummyModel:
    def predict_proba(self, X):
        n = len(X)
        return np.array([[0.45, 0.55] for _ in range(n)])


def test_app_module_import_smoke() -> None:
    assert callable(app.main)


def test_prepare_eval_predictions_builds_expected_columns(synthetic_matches_df) -> None:
    eval_df = build_team1_win_target(synthetic_matches_df.copy())
    scored = app._prepare_eval_predictions(_DummyModel(), eval_df)
    expected = {
        "team1_win_prob",
        "team2_win_prob",
        "underdog_win_prob",
        "upset_severity_index",
        "pred_winner",
        "pred_is_upset",
    }
    assert expected.issubset(set(scored.columns))
