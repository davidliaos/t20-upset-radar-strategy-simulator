"""Scenario construction and simulation helpers for the Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd


@dataclass
class ScenarioInput:
    """Single what-if scenario payload."""

    team1: str
    team2: str
    venue: str
    match_stage: str
    toss_winner: str
    toss_decision: str
    elo_team1: float
    elo_team2: float
    team1_form_5: float
    team2_form_5: float
    team1_form_10: float
    team2_form_10: float
    h2h_win_pct: float


def build_scenario_features(payload: ScenarioInput) -> pd.DataFrame:
    """Convert scenario inputs into a one-row feature frame."""
    row = {
        "team1": payload.team1,
        "team2": payload.team2,
        "venue": payload.venue,
        "match_stage": payload.match_stage,
        "toss_winner": payload.toss_winner,
        "toss_decision": payload.toss_decision,
        "elo_team1": payload.elo_team1,
        "elo_team2": payload.elo_team2,
        "elo_diff": payload.elo_team1 - payload.elo_team2,
        "team1_form_5": payload.team1_form_5,
        "team2_form_5": payload.team2_form_5,
        "team1_form_10": payload.team1_form_10,
        "team2_form_10": payload.team2_form_10,
        "h2h_win_pct": payload.h2h_win_pct,
    }
    return pd.DataFrame([row])


def score_scenario(model: Any, feature_row: pd.DataFrame) -> Dict[str, float]:
    """Return team1/team2 win probabilities and upset risk estimate."""
    probs = model.predict_proba(feature_row)[0]
    p_team1 = float(probs[1])
    p_team2 = 1.0 - p_team1
    upset_risk = min(p_team1, p_team2)
    return {"team1_win_prob": p_team1, "team2_win_prob": p_team2, "upset_risk": upset_risk}


def normalize_matchup_rows(df: pd.DataFrame, team1: str, team2: str) -> pd.DataFrame:
    """Return team1-perspective rows for a team pair regardless of row order."""
    pair = df[
        ((df["team1"] == team1) & (df["team2"] == team2))
        | ((df["team1"] == team2) & (df["team2"] == team1))
    ].copy()

    if pair.empty:
        return pair

    same_order = pair["team1"] == team1
    transformed = pd.DataFrame(index=pair.index)
    transformed["elo_team1"] = pair["elo_team1"].where(same_order, pair["elo_team2"])
    transformed["elo_team2"] = pair["elo_team2"].where(same_order, pair["elo_team1"])
    transformed["team1_form_5"] = pair["team1_form_5"].where(same_order, pair["team2_form_5"])
    transformed["team2_form_5"] = pair["team2_form_5"].where(same_order, pair["team1_form_5"])
    transformed["team1_form_10"] = pair["team1_form_10"].where(same_order, pair["team2_form_10"])
    transformed["team2_form_10"] = pair["team2_form_10"].where(same_order, pair["team1_form_10"])
    transformed["h2h_win_pct"] = pair["h2h_win_pct"].where(same_order, 1 - pair["h2h_win_pct"])
    transformed["match_stage"] = pair["match_stage"]
    transformed["venue"] = pair["venue"]
    transformed["toss_decision"] = pair["toss_decision"]
    # Keep categorical nulls so fallback tiers can still use matchup statistics.
    return transformed


def _get_matchup_venue_rows(df: pd.DataFrame, team1: str, team2: str, venue: str) -> pd.DataFrame:
    """Return team1-perspective rows for team1/team2 pair at selected venue."""
    matchup = normalize_matchup_rows(df, team1, team2)
    if matchup.empty:
        return matchup
    venue_str = str(venue) if pd.notna(venue) else ""
    return matchup[matchup["venue"].astype(str) == venue_str]


def _get_matchup_rows(df: pd.DataFrame, team1: str, team2: str) -> pd.DataFrame:
    """Return team1-perspective rows for team1/team2 pair across all venues."""
    return normalize_matchup_rows(df, team1, team2)


def _get_venue_rows(df: pd.DataFrame, venue: str) -> pd.DataFrame:
    """Return all rows at the given venue (all teams)."""
    venue_str = str(venue) if pd.notna(venue) else ""
    return df[df["venue"].astype(str) == venue_str].copy()


def _get_global_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return the full dataframe as global fallback."""
    return df.copy()


def _compute_defaults_from_source(source: pd.DataFrame) -> Dict[str, float | str | None]:
    """Compute default scenario values from a source dataframe."""
    required = ["elo_team1", "elo_team2", "team1_form_5", "team2_form_5", "team1_form_10", "team2_form_10", "h2h_win_pct"]
    missing = [c for c in required if c not in source.columns]
    if missing:
        raise ValueError(f"Source missing required columns: {missing}")

    def _safe_mean(series: pd.Series, default: float = 0.5) -> float:
        val = series.mean()
        return float(val) if pd.notna(val) else default

    def _safe_elo_mean(series: pd.Series) -> float:
        return _safe_mean(series, default=1500.0)

    def _safe_mode(series: pd.Series, default: str | None) -> str | None:
        mode = series.dropna().mode()
        return mode.iloc[0] if not mode.empty else default

    defaults: Dict[str, float | str | None] = {
        "elo_team1": _safe_elo_mean(source["elo_team1"]),
        "elo_team2": _safe_elo_mean(source["elo_team2"]),
        "team1_form_5": _safe_mean(source["team1_form_5"]),
        "team2_form_5": _safe_mean(source["team2_form_5"]),
        "team1_form_10": _safe_mean(source["team1_form_10"]),
        "team2_form_10": _safe_mean(source["team2_form_10"]),
        "h2h_win_pct": _safe_mean(source["h2h_win_pct"]),
        "match_stage": _safe_mode(source["match_stage"], default=None),
        "toss_decision": _safe_mode(source["toss_decision"], default="field"),
    }
    return defaults


def _clamp_metrics(defaults: Dict[str, float | str | None]) -> None:
    """Clamp [0,1] metrics in place to stay within bounds."""
    metric_keys = ["team1_form_5", "team2_form_5", "team1_form_10", "team2_form_10", "h2h_win_pct"]
    for key in metric_keys:
        value = defaults.get(key)
        if isinstance(value, (int, float)):
            defaults[key] = min(max(float(value), 0.0), 1.0)


def _select_fallback_source(
    df: pd.DataFrame, team1: str, team2: str, venue: str
) -> pd.DataFrame:
    """Select source rows using explicit fallback order: matchup+venue -> matchup -> venue -> global."""
    tiers = [
        lambda: _get_matchup_venue_rows(df, team1, team2, venue),
        lambda: _get_matchup_rows(df, team1, team2),
        lambda: _get_venue_rows(df, venue),
        lambda: _get_global_rows(df),
    ]
    for tier_fn in tiers:
        source = tier_fn()
        if not source.empty:
            return source
    return df.copy()


def estimate_scenario_defaults(df: pd.DataFrame, team1: str, team2: str, venue: str) -> Dict[str, float | str | None]:
    """Estimate robust historical priors for scenario controls.

    Uses explicit fallback order: matchup+venue -> matchup -> venue -> global.
    """
    if df.empty:
        return {
            "elo_team1": 1500.0,
            "elo_team2": 1500.0,
            "team1_form_5": 0.5,
            "team2_form_5": 0.5,
            "team1_form_10": 0.5,
            "team2_form_10": 0.5,
            "h2h_win_pct": 0.5,
            "match_stage": None,
            "toss_decision": "field",
        }
    source = _select_fallback_source(df, team1, team2, venue)
    defaults = _compute_defaults_from_source(source)
    _clamp_metrics(defaults)
    return defaults

