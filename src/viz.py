"""Visualization helpers for upset analysis and calibration checks."""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve


def plot_calibration_curve(y_true, y_prob, n_bins: int = 10) -> Tuple[plt.Figure, plt.Axes]:
    """Create a reliability diagram for binary probabilities."""
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mean_pred, frac_pos, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    return fig, ax


def upset_rate_by_bucket(df: pd.DataFrame, score_col: str, upset_col: str, bins: int = 10) -> pd.DataFrame:
    """Aggregate observed upset rates by a numeric score bucket."""
    out = df[[score_col, upset_col]].dropna().copy()
    out["bucket"] = pd.qcut(out[score_col], q=bins, duplicates="drop")
    grouped = out.groupby("bucket", observed=True)[upset_col].agg(["mean", "count"]).reset_index()
    grouped = grouped.rename(columns={"mean": "upset_rate", "count": "matches"})
    return grouped


def correlation_strength_label(correlation: float) -> str:
    """Return a concise qualitative label for correlation magnitude."""
    magnitude = abs(float(correlation))
    if magnitude >= 0.8:
        return "High"
    if magnitude >= 0.5:
        return "Moderate"
    return "Low"


def roc_auc_generalization_note(valid_roc_auc: float, test_roc_auc: float) -> tuple[float, str]:
    """Summarize valid-vs-test ROC AUC behavior for reviewer-facing diagnostics."""
    gap = float(test_roc_auc) - float(valid_roc_auc)
    if abs(gap) <= 0.03:
        note = (
            "Valid and test ROC AUC are closely aligned, suggesting stable generalization "
            "across adjacent time windows."
        )
    elif gap > 0:
        note = (
            "Test ROC AUC is higher than validation. This can happen when the test window "
            "is slightly easier or better aligned with learned patterns."
        )
    else:
        note = (
            "Test ROC AUC is lower than validation, which may indicate mild temporal drift "
            "or reduced signal quality in the newer window."
        )
    return gap, note


def venue_sample_caution_label(match_count: int, caution_threshold: int = 10) -> str:
    """Assign a sample-size caution label for venue-level diagnostics."""
    return "Low sample" if int(match_count) < int(caution_threshold) else "Adequate sample"

