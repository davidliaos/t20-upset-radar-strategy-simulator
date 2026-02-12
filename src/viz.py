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

