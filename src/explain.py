"""Local explanation helpers focused on upset narratives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd


@dataclass
class CounterfactualResult:
    """Represents one counterfactual change and its probability impact."""

    name: str
    team1_win_prob: float
    delta_vs_base: float


def rank_notable_upsets(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return top upset matches ranked by absolute pre-match ELO gap."""
    required = {"is_upset", "elo_diff"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for upset ranking: {sorted(missing)}")

    upsets = df[df["is_upset"] == 1].copy()
    if upsets.empty:
        return upsets

    upsets["abs_elo_diff"] = upsets["elo_diff"].abs()
    cols = [c for c in ["date", "team1", "team2", "winner", "match_stage", "venue", "elo_diff", "is_upset"] if c in upsets.columns]
    return upsets.sort_values("abs_elo_diff", ascending=False)[cols].head(top_n)


def build_counterfactual_explanation(
    model: Any,
    base_row: pd.DataFrame,
    *,
    fallback_stage: str = "Group",
) -> Dict[str, Any]:
    """Explain one scenario via domain counterfactuals, not black-box attributions."""
    if len(base_row) != 1:
        raise ValueError("base_row must contain exactly one row.")

    base_prob = float(model.predict_proba(base_row)[0][1])
    row = base_row.iloc[0]

    cfs: List[CounterfactualResult] = []

    def _score_with(changes: Dict[str, Any], name: str) -> None:
        cf_row = base_row.copy()
        for key, value in changes.items():
            if key in cf_row.columns:
                cf_row.at[cf_row.index[0], key] = value
        prob = float(model.predict_proba(cf_row)[0][1])
        cfs.append(CounterfactualResult(name=name, team1_win_prob=prob, delta_vs_base=prob - base_prob))

    # Toss decision switch.
    if "toss_decision" in base_row.columns:
        toss = str(row.get("toss_decision", "field"))
        _score_with(
            {"toss_decision": "field" if toss == "bat" else "bat"},
            name="Flip toss decision",
        )

    # Neutralize ELO edge.
    if "elo_team1" in base_row.columns and "elo_team2" in base_row.columns and "elo_diff" in base_row.columns:
        elo_mid = (float(row["elo_team1"]) + float(row["elo_team2"])) / 2
        _score_with(
            {"elo_team1": elo_mid, "elo_team2": elo_mid, "elo_diff": 0.0},
            name="Neutralize ELO gap",
        )

    # Neutralize recent form.
    form_changes = {}
    for col in ["team1_form_5", "team2_form_5", "team1_form_10", "team2_form_10"]:
        if col in base_row.columns:
            form_changes[col] = 0.5
    if form_changes:
        _score_with(form_changes, name="Neutralize recent form")

    # Stage to group baseline if available.
    if "match_stage" in base_row.columns:
        _score_with({"match_stage": fallback_stage}, name="Shift stage to Group")

    cfs_sorted = sorted(cfs, key=lambda x: abs(x.delta_vs_base), reverse=True)
    return {
        "base_team1_win_prob": base_prob,
        "counterfactuals": [cf.__dict__ for cf in cfs_sorted],
    }

