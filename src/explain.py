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


def matchup_volatility_profile(df: pd.DataFrame, team1: str, team2: str) -> Dict[str, float]:
    """Estimate volatility and upset tendency for a team matchup.

    This is a post-hoc diagnostic summary for historical context and reviewer UX.
    It uses observed outcomes and is not used as a model training feature.
    """
    mask = ((df["team1"] == team1) & (df["team2"] == team2)) | (
        (df["team1"] == team2) & (df["team2"] == team1)
    )
    pair = df.loc[mask].copy()
    if pair.empty:
        return {"matches": 0.0, "upset_rate": 0.0, "volatility_index": 0.0}

    # Convert winner to team1-perspective outcome for variance-based volatility.
    same_order = pair["team1"] == team1
    team1_win_view = (pair["winner"] == pair["team1"]).where(same_order, pair["winner"] == pair["team2"])
    team1_win_view = team1_win_view.astype(float)
    upset_rate = float(pair["is_upset"].mean()) if "is_upset" in pair.columns else 0.0

    # Bernoulli stddev maxes at 0.5; normalize to [0,1].
    volatility_index = float(team1_win_view.std(ddof=0)) * 2
    return {
        "matches": float(len(pair)),
        "upset_rate": upset_rate,
        "volatility_index": volatility_index,
    }


def build_missed_upsets_audit(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Return highest-confidence upset misses for post-hoc model review."""
    required = {
        "is_upset",
        "pred_is_upset",
        "favorite_team",
        "team1",
        "team1_win_prob",
        "winner",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for missed-upsets audit: {sorted(missing)}")

    audit_df = df.copy()
    audit_df["favorite_confidence"] = audit_df.apply(
        lambda r: r["team1_win_prob"] if r["favorite_team"] == r["team1"] else (1 - r["team1_win_prob"]),
        axis=1,
    )
    missed = audit_df[(audit_df["is_upset"] == 1) & (audit_df["pred_is_upset"] == 0)].copy()
    if missed.empty:
        return missed
    return missed.sort_values("favorite_confidence", ascending=False).head(top_n)


def build_curated_upset_narratives(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Build concise narratives for notable upsets."""
    notable = rank_notable_upsets(df, top_n=top_n).copy()
    if notable.empty:
        return notable

    notable["elo_gap"] = notable["elo_diff"].abs().round(1)
    # elo_diff is team1 - team2, so non-negative implies team1 was favorite.
    notable["favorite_team"] = notable.apply(
        lambda r: r["team1"] if float(r["elo_diff"]) >= 0 else r["team2"],
        axis=1,
    )

    def _narrative(row: pd.Series) -> str:
        favorite = row.get("favorite_team", "favorite")
        winner = row.get("winner", "winner")
        stage = row.get("match_stage", "Unknown Stage")
        gap = row.get("elo_gap", "n/a")
        return f"At {stage}, {winner} beat favorite {favorite} despite an ELO gap of {gap}."

    notable["narrative"] = notable.apply(_narrative, axis=1)
    return notable


def summarize_curated_upset_patterns(curated_upsets: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Summarize stage/venue context from curated upset cases."""
    if curated_upsets.empty:
        return {"stage_summary": pd.DataFrame(), "venue_summary": pd.DataFrame()}

    stage_summary = (
        curated_upsets["match_stage"]
        .value_counts(dropna=False)
        .rename("count")
        .reset_index()
        .rename(columns={"index": "match_stage"})
    )
    venue_summary = (
        curated_upsets["venue"]
        .value_counts(dropna=False)
        .rename("count")
        .reset_index()
        .rename(columns={"index": "venue"})
    )
    return {"stage_summary": stage_summary, "venue_summary": venue_summary}


def notable_upset_sentence(
    winner: str,
    loser: str,
    event_name: str,
    date_label: str,
) -> str:
    """Build a readable notable-upset sentence while avoiding duplicated team names."""
    winner_clean = str(winner).strip() or "Unknown Winner"
    loser_clean = str(loser).strip() or "Unknown Opponent"
    if winner_clean == loser_clean:
        matchup_phrase = winner_clean
    else:
        matchup_phrase = f"{winner_clean} beat {loser_clean}"
    event_clean = str(event_name).strip() or "Unknown Event"
    date_clean = str(date_label).strip() or "Unknown Date"
    return f"{matchup_phrase} at {event_clean} on {date_clean}."

