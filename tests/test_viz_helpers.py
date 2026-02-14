"""Tests for visualization interpretation helpers."""

from __future__ import annotations

import pandas as pd

from src.viz import (
    correlation_strength_label,
    roc_auc_generalization_note,
    upset_rate_by_bucket,
    venue_sample_caution_label,
)


def test_correlation_strength_label_thresholds() -> None:
    assert correlation_strength_label(0.85) == "High"
    assert correlation_strength_label(-0.6) == "Moderate"
    assert correlation_strength_label(0.2) == "Low"


def test_roc_auc_generalization_note_outputs_gap_and_text() -> None:
    gap, note = roc_auc_generalization_note(0.64, 0.62)
    assert round(gap, 3) == -0.02
    assert "generalization" in note.lower() or "drift" in note.lower()


def test_venue_sample_caution_label_default_threshold() -> None:
    assert venue_sample_caution_label(3) == "Low sample"
    assert venue_sample_caution_label(11) == "Adequate sample"


def test_upset_rate_by_bucket_outputs_expected_schema() -> None:
    df = pd.DataFrame(
        {
            "abs_elo_diff": [10, 20, 30, 40, 50, 60, 70, 80],
            "is_upset": [0, 0, 1, 0, 1, 1, 0, 1],
        }
    )
    out = upset_rate_by_bucket(df, "abs_elo_diff", "is_upset", bins=4)
    assert {"bucket", "upset_rate", "matches"}.issubset(out.columns)
    assert int(out["matches"].sum()) == len(df)
    assert out["upset_rate"].between(0.0, 1.0).all()
