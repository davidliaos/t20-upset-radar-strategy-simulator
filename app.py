"""Streamlit dashboard for the T20 Upset Radar strategy simulator."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from html import escape
from itertools import combinations
from typing import Any, TypedDict

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.data_prep import (
    assign_favorite_underdog_from_elo,
    build_team1_win_target,
    load_matches,
    time_based_split,
)
from src.explain import (
    build_counterfactual_explanation,
    build_curated_upset_narratives,
    build_missed_upsets_audit,
    matchup_volatility_profile,
    notable_upset_sentence,
    rank_notable_upsets,
    summarize_curated_upset_patterns,
)
from src.features import build_pre_match_feature_frame
from src.models import (
    build_default_metadata,
    calibrate_classifier,
    evaluate_binary_model,
    load_model_artifacts,
    save_model_artifacts,
    train_logistic_baseline,
)
from src.simulation import (
    ScenarioInput,
    build_scenario_export_payload,
    build_scenario_features,
    build_upset_alert,
    estimate_scenario_defaults_with_meta,
    prior_confidence_label,
    score_scenario,
)
from src.viz import (
    correlation_strength_label,
    plot_calibration_curve,
    roc_auc_generalization_note,
    upset_rate_by_bucket,
    venue_sample_caution_label,
)

plt.switch_backend("Agg")


class SidebarInputs(TypedDict):
    team1: str
    team2: str
    venue: str
    stage: str
    toss_winner: str
    toss_decision: str
    elo_team1: float
    elo_team2: float
    team1_form_5: float
    team2_form_5: float
    team1_form_10: float
    team2_form_10: float
    h2h_win_pct: float
    defaults_meta: dict[str, Any]
    compact_mode: bool
    focus_mode: bool


TEAM_FLAG_EMOJI: dict[str, str] = {
    "Afghanistan": "🇦🇫",
    "Australia": "🇦🇺",
    "Bangladesh": "🇧🇩",
    "Canada": "🇨🇦",
    "England": "🇬🇧",
    "Hong Kong": "🇭🇰",
    "India": "🇮🇳",
    "Ireland": "🇮🇪",
    "Namibia": "🇳🇦",
    "Nepal": "🇳🇵",
    "Netherlands": "🇳🇱",
    "New Zealand": "🇳🇿",
    "Oman": "🇴🇲",
    "Pakistan": "🇵🇰",
    "Papua New Guinea": "🇵🇬",
    "Scotland": "🏴",
    "South Africa": "🇿🇦",
    "Sri Lanka": "🇱🇰",
    "United Arab Emirates": "🇦🇪",
    "USA": "🇺🇸",
    "West Indies": "🏝️",
    "Zimbabwe": "🇿🇼",
}


def team_flag(team_name: str) -> str:
    """Return a best-effort team flag icon."""
    return TEAM_FLAG_EMOJI.get(team_name, "🏏")


def inject_design_system(compact_mode: bool = False) -> None:
    """Inject lightweight CSS styling for clearer dashboard hierarchy."""
    st.markdown(
        """
        <style>
        .hero-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .hero-subtitle {
            color: #6b7280;
            margin-bottom: 0.6rem;
        }
        .section-kicker {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #0ea5e9;
            margin-top: 0.9rem;
            margin-bottom: 0.35rem;
        }
        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin: 0.4rem 0 0.8rem 0;
        }
        .badge-chip {
            border: 1px solid #d1d5db;
            border-radius: 999px;
            padding: 0.15rem 0.55rem;
            font-size: 0.78rem;
            color: #374151;
            background: #f9fafb;
        }
        .risk-chip {
            border-radius: 999px;
            padding: 0.15rem 0.55rem;
            font-size: 0.78rem;
            font-weight: 600;
            display: inline-block;
            margin: 0.2rem 0 0.35rem 0;
        }
        .risk-high { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
        .risk-medium { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
        .risk-low { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
        .alert-banner {
            border-radius: 0.55rem;
            padding: 0.7rem 0.9rem;
            margin: 0.45rem 0 0.9rem 0;
            font-weight: 550;
        }
        .alert-high { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
        .alert-medium { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
        .alert-low { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
        .info-card {
            border: 1px solid var(--border-color);
            border-radius: 0.6rem;
            background: var(--secondary-background-color);
            color: var(--text-color);
            padding: 0.75rem 0.9rem;
            margin: 0.45rem 0 1rem 0;
        }
        .info-card strong, .info-card span, .info-card div {
            color: inherit;
        }
        .severity-badge {
            border-radius: 999px;
            padding: 0.2rem 0.55rem;
            font-size: 0.78rem;
            font-weight: 600;
            display: inline-block;
        }
        .severity-high { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
        .severity-medium { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
        .severity-low { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
        .rank-badge {
            display: inline-block;
            min-width: 1.35rem;
            text-align: center;
            border-radius: 999px;
            background: #e0f2fe;
            color: #075985;
            border: 1px solid #bae6fd;
            font-size: 0.75rem;
            font-weight: 700;
            margin-right: 0.4rem;
            padding: 0.08rem 0.3rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if compact_mode:
        st.markdown(
            """
            <style>
            .block-container { padding-top: 1.1rem; padding-bottom: 1.1rem; }
            [data-testid="stMetricValue"] { font-size: 1.45rem; }
            </style>
            """,
            unsafe_allow_html=True,
        )


def render_section_kicker(label: str) -> None:
    st.markdown(f"<div class='section-kicker'>{escape(label)}</div>", unsafe_allow_html=True)


def render_badge_row(items: list[tuple[str, str]]) -> None:
    chips = "".join(
        f"<span class='badge-chip'><strong>{escape(k)}:</strong> {escape(v)}</span>"
        for k, v in items
    )
    st.markdown(f"<div class='badge-row'>{chips}</div>", unsafe_allow_html=True)


def render_risk_chip(level: str, message: str) -> None:
    safe_level = level if level in {"high", "medium", "low"} else "low"
    st.markdown(
        f"<span class='risk-chip risk-{safe_level}'>{escape(message)}</span>",
        unsafe_allow_html=True,
    )


def render_alert_banner(level: str, message: str) -> None:
    safe_level = level if level in {"high", "medium", "low"} else "low"
    st.markdown(
        f"<div class='alert-banner alert-{safe_level}'>{escape(message)}</div>",
        unsafe_allow_html=True,
    )


def render_info_card(title: str, lines: list[str]) -> None:
    rows = "<br/>".join(escape(line) for line in lines)
    st.markdown(
        f"<div class='info-card'><strong>{escape(title)}</strong><br/>{rows}</div>",
        unsafe_allow_html=True,
    )


def render_severity_badge(score: float) -> None:
    level = "high" if score >= 0.25 else ("medium" if score >= 0.12 else "low")
    st.markdown(
        (
            f"<span class='severity-badge severity-{level}'>"
            f"Severity (by ELO gap): {score:.1%}"
            "</span>"
        ),
        unsafe_allow_html=True,
    )


def _recommended_venue_options(df: pd.DataFrame, team1: str, team2: str) -> list[str]:
    """Return plausible venues ordered by relevance for selected teams."""
    venue_col = "venue"
    if venue_col not in df.columns:
        return []

    pair_mask = ((df["team1"] == team1) & (df["team2"] == team2)) | (
        (df["team1"] == team2) & (df["team2"] == team1)
    )
    pair_venues = set(df.loc[pair_mask, venue_col].dropna().astype(str))

    t1_mask = (df["team1"] == team1) | (df["team2"] == team1)
    t2_mask = (df["team1"] == team2) | (df["team2"] == team2)
    t1_venues = set(df.loc[t1_mask, venue_col].dropna().astype(str))
    t2_venues = set(df.loc[t2_mask, venue_col].dropna().astype(str))

    shared_venues = t1_venues & t2_venues
    union_venues = t1_venues | t2_venues

    ordered: list[str] = []
    ordered.extend(sorted(pair_venues))
    ordered.extend(sorted(shared_venues - set(ordered)))
    ordered.extend(sorted(union_venues - set(ordered)))
    return ordered


def _recommended_venue_counts(df: pd.DataFrame, team1: str, team2: str) -> tuple[int, int, int]:
    """Return pair/shared/union venue counts for helper copy."""
    venue_col = "venue"
    if venue_col not in df.columns:
        return (0, 0, 0)
    pair_mask = ((df["team1"] == team1) & (df["team2"] == team2)) | (
        (df["team1"] == team2) & (df["team2"] == team1)
    )
    pair_venues = set(df.loc[pair_mask, venue_col].dropna().astype(str))
    t1_mask = (df["team1"] == team1) | (df["team2"] == team1)
    t2_mask = (df["team1"] == team2) | (df["team2"] == team2)
    t1_venues = set(df.loc[t1_mask, venue_col].dropna().astype(str))
    t2_venues = set(df.loc[t2_mask, venue_col].dropna().astype(str))
    shared_venues = t1_venues & t2_venues
    union_venues = t1_venues | t2_venues
    return (len(pair_venues), len(shared_venues), len(union_venues))


def _default_matchup_for_risk_context(df: pd.DataFrame, teams: list[str]) -> tuple[str, str]:
    """Pick a default team pair with historical context for volatility radar."""
    if len(teams) < 2:
        raise ValueError("Need at least two teams for matchup defaults.")
    if not {"team1", "team2"}.issubset(df.columns):
        return teams[0], teams[1]

    pair = df[["team1", "team2"]].dropna().copy()
    if pair.empty:
        return teams[0], teams[1]
    pair["a"] = pair[["team1", "team2"]].min(axis=1)
    pair["b"] = pair[["team1", "team2"]].max(axis=1)

    agg_columns: dict[str, tuple[str, str]] = {"matches": ("team1", "size")}
    if "is_upset" in df.columns:
        pair["is_upset"] = pd.to_numeric(df.loc[pair.index, "is_upset"], errors="coerce")
        agg_columns["upset_rate"] = ("is_upset", "mean")
    summary = pair.groupby(["a", "b"], as_index=False).agg(**agg_columns)
    summary = summary[(summary["a"].isin(teams)) & (summary["b"].isin(teams))]
    if summary.empty:
        return teams[0], teams[1]

    if "upset_rate" in summary.columns:
        # Prefer balanced upset rates so default isn't trivially one-sided.
        summary["volatility_proxy"] = (0.5 - (summary["upset_rate"] - 0.5).abs()).clip(lower=0.0)
        summary = summary.sort_values(["matches", "volatility_proxy"], ascending=[False, False])
    else:
        summary = summary.sort_values("matches", ascending=False)
    best = summary.iloc[0]
    return str(best["a"]), str(best["b"])


def build_event_label(row: pd.Series) -> str:
    """Build an event label with robust fallbacks."""
    tournament = row.get("tournament_name")
    if pd.notna(tournament) and str(tournament).strip():
        return str(tournament).strip()
    venue = row.get("venue")
    if pd.notna(venue) and str(venue).strip():
        return str(venue).strip()
    stage = row.get("match_stage")
    if pd.notna(stage) and str(stage).strip():
        return str(stage).strip()
    return "Unknown Event"


def build_date_label(value: Any) -> str:
    """Format date values safely for user-facing labels."""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.notna(parsed):
        return parsed.strftime("%Y-%m-%d")
    return "Unknown Date"


@st.cache_data
def get_matches() -> pd.DataFrame:
    df = load_matches()
    df = build_team1_win_target(df)
    return assign_favorite_underdog_from_elo(df)


@st.cache_data
def get_split_data(df: pd.DataFrame):
    train_df, valid_df, test_df = time_based_split(df)
    X_train, y_train = build_pre_match_feature_frame(train_df)
    X_valid, y_valid = build_pre_match_feature_frame(valid_df)
    X_test, y_test = build_pre_match_feature_frame(test_df)
    return train_df, valid_df, test_df, X_train, y_train, X_valid, y_valid, X_test, y_test


@st.cache_resource
def get_baseline_model():
    df = get_matches()
    train_df, valid_df, _, X_train, y_train, X_valid, y_valid, _, _ = get_split_data(df)
    model_name = "baseline_logistic_calibrated"
    _ = (train_df, valid_df)  # keeps intent explicit for cache dependencies.

    try:
        artifacts = load_model_artifacts(model_name=model_name)
        return artifacts.model, df, "loaded", model_name, artifacts.metadata
    except FileNotFoundError:
        baseline = train_logistic_baseline(X_train, y_train)
        model = calibrate_classifier(baseline, X_valid, y_valid)
        metadata = build_default_metadata()
        save_model_artifacts(model, model_name=model_name, metadata=metadata)
        return model, df, "trained", model_name, metadata


def _build_sidebar_inputs(df: pd.DataFrame, teams: list[str], venues: list[str], stages: list[str]) -> SidebarInputs:
    with st.sidebar:
        st.header("Match Scenario")
        with st.expander("Display & priors", expanded=False):
            compact_mode = st.checkbox(
                "Compact mode",
                value=False,
                key="sidebar_compact_mode",
                help="Reduce spacing and tighten metrics for faster scanning.",
            )
            focus_mode = st.checkbox(
                "Focus mode",
                value=False,
                key="sidebar_focus_mode",
                help="Hide explanatory text for presentation-first walkthroughs.",
            )
            allow_all_venues = st.checkbox(
                "Show all venues",
                value=False,
                key="sidebar_show_all_venues",
                help="When off, venue choices are filtered to plausible team contexts.",
            )
            use_priors = st.checkbox(
                "Use matchup priors for defaults",
                value=True,
                key="sidebar_use_priors",
            )
        st.markdown("### Match Setup")
        with st.expander("Match Setup", expanded=True):
            default_team1, default_team2 = _default_matchup_for_risk_context(df, teams)
            team1_index = teams.index(default_team1) if default_team1 in teams else 0
            team1 = st.selectbox("Team 1 (favorite candidate)", options=teams, index=team1_index, key="sidebar_team1")
            team2_options = [t for t in teams if t != team1]
            team2_default_index = team2_options.index(default_team2) if default_team2 in team2_options else 0
            team2 = st.selectbox(
                "Team 2 (underdog candidate)", options=team2_options, index=team2_default_index, key="sidebar_team2"
            )
            default_vol = matchup_volatility_profile(df, default_team1, default_team2)
            st.caption(
                f"Seeded from history: {default_team1} vs {default_team2} "
                f"({int(default_vol['matches'])} matches, {default_vol['upset_rate']:.1%} upsets)."
            )
            recommended_venues = _recommended_venue_options(df, team1, team2)
            pair_venue_count, shared_venue_count, _ = _recommended_venue_counts(df, team1, team2)
            if allow_all_venues:
                venue_options = venues
            else:
                venue_options = recommended_venues
            if not venue_options:
                venue_options = venues
                st.info("No team-specific venue history found; showing all venues.")
            previous_venue = st.session_state.get("sidebar_venue")
            venue_index = venue_options.index(previous_venue) if previous_venue in venue_options else 0
            venue = st.selectbox("Venue", options=venue_options, index=venue_index, key="sidebar_venue")

        venue_city_mode = (
            df.loc[df["venue"].astype(str) == str(venue), "city"].dropna().mode()
            if "city" in df.columns
            else pd.Series(dtype="object")
        )
        venue_city = str(venue_city_mode.iloc[0]) if not venue_city_mode.empty else None
        defaults_meta = estimate_scenario_defaults_with_meta(df, team1, team2, venue, city=venue_city)
        defaults = defaults_meta["defaults"]
        source_tier = str(defaults_meta["source_tier"])
        confidence = prior_confidence_label(source_tier)
        caution_note = "Low sample size; interpret cautiously." if int(defaults_meta["source_rows"]) < 10 else "Sample size is adequate."
        st.info(
            (
                f"Priors: {source_tier} ({confidence}, {int(defaults_meta['source_rows'])} rows). "
                f"{len(venue_options)} plausible venues ({pair_venue_count} direct, {shared_venue_count} shared). "
                f"{caution_note}"
            )
        )

        st.markdown("### Match Context")
        with st.expander("Match Context", expanded=True):
            stage_index = stages.index(defaults["match_stage"]) if defaults["match_stage"] in stages else 0
            stage = st.selectbox("Match Stage", options=stages, index=stage_index, key="sidebar_stage")
            toss_winner = st.selectbox("Toss Winner", options=[team1, team2], index=0, key="sidebar_toss_winner")
            toss_decision_index = 0 if defaults["toss_decision"] == "bat" else 1
            toss_decision = st.selectbox(
                "Toss Decision", options=["bat", "field"], index=toss_decision_index, key="sidebar_toss_decision"
            )

        def prior_or(key: str, fallback: float) -> float:
            value = defaults.get(key)
            if use_priors and isinstance(value, (int, float)):
                return float(value)
            return fallback

        form_map = {"Cold": 0.35, "Mixed": 0.5, "In form": 0.65}
        strength_map = {
            "Heavy underdog": -220.0,
            "Slight underdog": -100.0,
            "Even": 0.0,
            "Slight favorite": 100.0,
            "Strong favorite": 220.0,
        }
        base_elo_team1 = prior_or("elo_team1", 1500.0)
        base_elo_team2 = prior_or("elo_team2", 1500.0)
        base_elo_mean = (base_elo_team1 + base_elo_team2) / 2
        default_strength = "Even"
        default_form_t1 = "Mixed"
        default_form_t2 = "Mixed"
        default_h2h = int(round(prior_or("h2h_win_pct", 0.5) * 100))

        st.markdown("**Quick setup**")
        strength_preset = st.select_slider(
            "Team 1 rating vs Team 2",
            options=list(strength_map.keys()),
            value=default_strength,
            key="sidebar_strength_preset",
        )
        st.caption(f"Rating: {strength_preset}.")
        team1_form_label = st.select_slider(
            "Team 1 form",
            options=list(form_map.keys()),
            value=default_form_t1,
            key="sidebar_team1_form_label",
        )
        st.caption(f"{team1} form: {team1_form_label}.")
        team2_form_label = st.select_slider(
            "Team 2 form",
            options=list(form_map.keys()),
            value=default_form_t2,
            key="sidebar_team2_form_label",
        )
        st.caption(f"{team2} form: {team2_form_label}.")
        st.markdown(f"Head-to-head advantage to {team1}: **{default_h2h}%**")
        h2h_win_pct_int = st.slider(
            "Head-to-head advantage (Team 1 %)",
            min_value=0,
            max_value=100,
            value=default_h2h,
            step=1,
            key="sidebar_h2h_pct",
        )
        st.caption(f"Head-to-head now set to {h2h_win_pct_int}% for {team1}.")

        st.caption("Fine-tune ELO, form and weighting only if you need raw controls.")
        with st.expander("Advanced sliders", expanded=False):
            use_raw_controls = st.checkbox(
                "Edit raw ELO and form values",
                value=False,
                key="sidebar_use_raw_controls",
            )
            raw_elo_team1 = st.number_input("Raw ELO Team 1", value=base_elo_team1, step=10.0, key="sidebar_raw_elo_team1")
            raw_elo_team2 = st.number_input("Raw ELO Team 2", value=base_elo_team2, step=10.0, key="sidebar_raw_elo_team2")
            raw_team1_form_5 = st.slider(
                "Raw Team 1 Form (Last 5)", 0.0, 1.0, prior_or("team1_form_5", 0.5), 0.01, key="sidebar_raw_team1_form_5"
            )
            raw_team2_form_5 = st.slider(
                "Raw Team 2 Form (Last 5)", 0.0, 1.0, prior_or("team2_form_5", 0.5), 0.01, key="sidebar_raw_team2_form_5"
            )
            raw_team1_form_10 = st.slider(
                "Raw Team 1 Form (Last 10)",
                0.0,
                1.0,
                prior_or("team1_form_10", 0.5),
                0.01,
                key="sidebar_raw_team1_form_10",
            )
            raw_team2_form_10 = st.slider(
                "Raw Team 2 Form (Last 10)",
                0.0,
                1.0,
                prior_or("team2_form_10", 0.5),
                0.01,
                key="sidebar_raw_team2_form_10",
            )
            raw_h2h_win_pct_int = st.slider(
                "Raw Team 1 Head-to-Head Win %",
                min_value=0,
                max_value=100,
                value=default_h2h,
                step=1,
                key="sidebar_raw_h2h_pct",
            )

        if use_raw_controls:
            elo_team1 = float(raw_elo_team1)
            elo_team2 = float(raw_elo_team2)
            team1_form_5 = float(raw_team1_form_5)
            team2_form_5 = float(raw_team2_form_5)
            team1_form_10 = float(raw_team1_form_10)
            team2_form_10 = float(raw_team2_form_10)
            h2h_win_pct = float(raw_h2h_win_pct_int) / 100.0
        else:
            elo_gap = strength_map[strength_preset]
            elo_team1 = base_elo_mean + (elo_gap / 2.0)
            elo_team2 = base_elo_mean - (elo_gap / 2.0)
            team1_form_5 = form_map[team1_form_label]
            team2_form_5 = form_map[team2_form_label]
            team1_form_10 = (team1_form_5 + 0.5) / 2.0
            team2_form_10 = (team2_form_5 + 0.5) / 2.0
            h2h_win_pct = float(h2h_win_pct_int) / 100.0

    return {
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "stage": stage,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "elo_team1": elo_team1,
        "elo_team2": elo_team2,
        "team1_form_5": team1_form_5,
        "team2_form_5": team2_form_5,
        "team1_form_10": team1_form_10,
        "team2_form_10": team2_form_10,
        "h2h_win_pct": h2h_win_pct,
        "defaults_meta": defaults_meta,
        "compact_mode": compact_mode,
        "focus_mode": focus_mode,
    }


def _prepare_test_eval(model, test_df: pd.DataFrame) -> pd.DataFrame:
    return _prepare_eval_predictions(model, test_df)


def _prepare_eval_predictions(model, eval_df: pd.DataFrame) -> pd.DataFrame:
    scored = assign_favorite_underdog_from_elo(eval_df.copy())
    X_eval, _ = build_pre_match_feature_frame(scored)
    scored["team1_win_prob"] = model.predict_proba(X_eval)[:, 1]
    scored["team2_win_prob"] = 1.0 - scored["team1_win_prob"]
    favorite_is_team1 = scored["favorite_team"] == scored["team1"]
    scored["underdog_win_prob"] = scored["team2_win_prob"].where(favorite_is_team1, scored["team1_win_prob"])
    elo_gap = (scored["elo_team1"] - scored["elo_team2"]).abs()
    scored["upset_severity_index"] = scored["underdog_win_prob"] * (elo_gap / 250.0).clip(0.0, 1.0)
    scored["pred_team1_win"] = (scored["team1_win_prob"] >= 0.5).astype(int)
    scored["pred_winner"] = scored.apply(
        lambda r: r["team1"] if r["pred_team1_win"] == 1 else r["team2"],
        axis=1,
    )
    scored["pred_is_upset"] = (scored["pred_winner"] == scored["underdog_team"]).astype(int)
    return scored


def _infer_city_for_venue(df: pd.DataFrame, venue: str) -> str | None:
    if "city" not in df.columns:
        return None
    mode = df.loc[df["venue"].astype(str) == str(venue), "city"].dropna().mode()
    if mode.empty:
        return None
    return str(mode.iloc[0])


def _build_watchlist(
    *,
    model,
    df: pd.DataFrame,
    selected_teams: list[str],
    selected_stage: str,
    selected_venue: str,
    max_pairs: int = 80,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    city = _infer_city_for_venue(df, selected_venue)
    pair_count = 0
    for team1, team2 in combinations(selected_teams, 2):
        if pair_count >= max_pairs:
            break
        defaults_meta = estimate_scenario_defaults_with_meta(df, team1, team2, selected_venue, city=city)
        defaults = defaults_meta["defaults"]
        try:
            scenario = ScenarioInput(
                team1=team1,
                team2=team2,
                venue=selected_venue,
                match_stage=selected_stage,
                toss_winner=team1,
                toss_decision=str(defaults.get("toss_decision", "bat")),
                elo_team1=float(defaults.get("elo_team1", 1500.0)),
                elo_team2=float(defaults.get("elo_team2", 1500.0)),
                team1_form_5=float(defaults.get("team1_form_5", 0.5)),
                team2_form_5=float(defaults.get("team2_form_5", 0.5)),
                team1_form_10=float(defaults.get("team1_form_10", 0.5)),
                team2_form_10=float(defaults.get("team2_form_10", 0.5)),
                h2h_win_pct=float(defaults.get("h2h_win_pct", 0.5)),
            )
        except (TypeError, ValueError):
            continue
        score = score_scenario(model, build_scenario_features(scenario))
        volatility = matchup_volatility_profile(df, team1, team2)
        watchlist_score = 0.7 * float(score["underdog_win_prob"]) + 0.3 * float(volatility["volatility_index"])
        favorite = team1 if scenario.elo_team1 >= scenario.elo_team2 else team2
        underdog = team2 if favorite == team1 else team1
        rows.append(
            {
                "matchup": f"{team1} vs {team2}",
                "team1": team1,
                "team2": team2,
                "favorite": favorite,
                "underdog": underdog,
                "underdog_win_prob": float(score["underdog_win_prob"]),
                "upset_severity_index": float(score["upset_severity_index"]),
                "volatility_index": float(volatility["volatility_index"]),
                "historical_matchups": int(volatility["matches"]),
                "priors_source_tier": str(defaults_meta["source_tier"]),
                "priors_source_rows": int(defaults_meta["source_rows"]),
                "watchlist_score": watchlist_score,
            }
        )
        pair_count += 1
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("watchlist_score", ascending=False).reset_index(drop=True)


def main() -> None:
    st.set_page_config(page_title="T20 Upset Radar", layout="wide")

    with st.spinner("Loading model and preparing diagnostics..."):
        model, df, model_source, model_name, model_metadata = get_baseline_model()
        train_df, valid_df, test_df, X_train, y_train, X_valid, y_valid, X_test, y_test = get_split_data(df)
        _ = (train_df, X_train)  # explicit placeholders for readability.
        test_eval = _prepare_test_eval(model, test_df)
        full_eval = _prepare_eval_predictions(model, df)

    teams = sorted(set(df["team1"]).union(set(df["team2"])))
    venues = sorted(df["venue"].dropna().astype(str).unique().tolist())
    stages = sorted(df["match_stage"].dropna().astype(str).unique().tolist())
    inputs = _build_sidebar_inputs(df, teams, venues, stages)
    inject_design_system(compact_mode=bool(inputs["compact_mode"]))
    focus_mode = bool(inputs["focus_mode"])
    st.markdown("<div class='hero-title'>T20 Upset Radar Dashboard</div>", unsafe_allow_html=True)
    with st.expander("About this tool", expanded=False):
        st.markdown(
            (
                "- Simulates T20 match scenarios with venue, toss, and stage context.\n"
                "- Surfaces upset risk with calibrated probabilities and trust diagnostics.\n"
                "- Exports scenario audits for reviewer-friendly sharing."
            )
        )
        st.caption(f"Model source: {model_source} artifact (`models/baseline_logistic_calibrated.joblib`).")
        trained_at = model_metadata.get("trained_at_utc")
        if isinstance(trained_at, str):
            st.caption(f"Model metadata timestamp (UTC): {trained_at}")

    scenario = ScenarioInput(
        team1=str(inputs["team1"]),
        team2=str(inputs["team2"]),
        venue=str(inputs["venue"]),
        match_stage=str(inputs["stage"]),
        toss_winner=str(inputs["toss_winner"]),
        toss_decision=str(inputs["toss_decision"]),
        elo_team1=float(inputs["elo_team1"]),
        elo_team2=float(inputs["elo_team2"]),
        team1_form_5=float(inputs["team1_form_5"]),
        team2_form_5=float(inputs["team2_form_5"]),
        team1_form_10=float(inputs["team1_form_10"]),
        team2_form_10=float(inputs["team2_form_10"]),
        h2h_win_pct=float(inputs["h2h_win_pct"]),
    )
    defaults_meta: dict[str, Any] = inputs["defaults_meta"]
    source_tier = str(defaults_meta["source_tier"])
    prior_conf = prior_confidence_label(source_tier)

    feature_row = build_scenario_features(scenario)
    result = score_scenario(model, feature_row)
    favorite = scenario.team1 if scenario.elo_team1 >= scenario.elo_team2 else scenario.team2
    underdog = scenario.team2 if favorite == scenario.team1 else scenario.team1

    tabs = st.tabs(["⚙️ Simulator", "📘 How It Works", "📊 Insights", "🧠 Explainability"])
    tab_sim, tab_how, tab_insights, tab_explain = tabs

    with tab_sim:
        render_section_kicker("Scenario Overview")
        render_info_card(
            "Scenario",
            [
                f"Stage: {scenario.match_stage}",
                f"Venue: {scenario.venue}",
                f"Favorite: {favorite}",
                f"Priors: {source_tier} ({prior_conf})",
            ],
        )
        if focus_mode:
            st.info("Focus mode is on: explanatory helper text is minimized.")
        else:
            st.caption("Use the controls to simulate one matchup, then scan watchlist candidates below.")

        alert = build_upset_alert(result["upset_risk"], scenario.match_stage)
        alert_level = str(alert["level"])
        render_risk_chip(alert_level, f"Upset Alert: {alert_level.upper()}")
        st.caption("Alert legend: Low <15%, Medium 15-26%, High >26% underdog chance.")
        render_alert_banner(
            alert_level,
            f"{alert['message']} (threshold: {alert['threshold']:.0%})",
        )
        st.markdown(
            f"**Result:** {underdog} has a **{result['underdog_win_prob']:.1%}** upset chance against "
            f"{favorite} under this scenario."
        )
        favorite_win_prob = result["team1_win_prob"] if favorite == scenario.team1 else result["team2_win_prob"]
        card_left, card_right = st.columns([3, 2])
        with card_left:
            c1, c2 = st.columns(2)
            c1.metric("Favorite Win %", f"{favorite_win_prob:.1%}")
            c2.metric("Underdog Win %", f"{result['underdog_win_prob']:.1%}")
        with card_right:
            render_severity_badge(float(result["upset_severity_index"]))
        if not focus_mode:
            st.caption("What these numbers mean: severity scales upset chance by ELO-gap size.")

        st.divider()
        render_section_kicker("Toss Decision Impact")
        st.subheader("Scenario Comparison")
        compare_decision = st.radio(
            "Compare toss strategy against",
            options=["bat", "field"],
            horizontal=True,
            key="sim_compare_toss_decision",
            index=1 if scenario.toss_decision == "bat" else 0,
        )
        alt_decision = str(compare_decision)
        st.caption(
            f"Current scenario: {scenario.toss_decision} first; alternative: {alt_decision} first."
        )
        alt_scenario = replace(scenario, toss_decision=alt_decision)
        alt_result = score_scenario(model, build_scenario_features(alt_scenario))
        delta_team1 = alt_result["team1_win_prob"] - result["team1_win_prob"]
        delta_underdog = alt_result["underdog_win_prob"] - result["underdog_win_prob"]
        comparison_df = pd.DataFrame(
            [
                {
                    "scenario": f"Current ({scenario.toss_decision})",
                    "team1_win_prob": result["team1_win_prob"],
                    "underdog_win_prob": result["underdog_win_prob"],
                    "upset_severity_index": result["upset_severity_index"],
                    "delta_team1": 0.0,
                    "delta_underdog": 0.0,
                },
                {
                    "scenario": f"Alternative ({alt_decision})",
                    "team1_win_prob": alt_result["team1_win_prob"],
                    "underdog_win_prob": alt_result["underdog_win_prob"],
                    "upset_severity_index": alt_result["upset_severity_index"],
                    "delta_team1": delta_team1,
                    "delta_underdog": delta_underdog,
                },
            ]
        )
        st.dataframe(
            comparison_df.style.format(
                {
                    "team1_win_prob": "{:.1%}",
                    "underdog_win_prob": "{:.1%}",
                    "upset_severity_index": "{:.1%}",
                    "delta_team1": "{:+.1%}",
                    "delta_underdog": "{:+.1%}",
                }
            ),
            width="stretch",
        )

        generated_at_utc = datetime.now(UTC).isoformat()
        export_df = comparison_df.copy()
        export_df["team1"] = scenario.team1
        export_df["team2"] = scenario.team2
        export_df["venue"] = scenario.venue
        export_df["match_stage"] = scenario.match_stage
        export_df["toss_winner"] = scenario.toss_winner
        export_df["priors_source_tier"] = source_tier
        export_df["generated_at_utc"] = generated_at_utc
        export_payload = build_scenario_export_payload(
            current_scenario=scenario,
            current_result=result,
            alternative_scenario=alt_scenario,
            alternative_result=alt_result,
            priors_source_tier=source_tier,
            priors_source_rows=int(defaults_meta["source_rows"]),
            model_name=model_name,
            model_source=model_source,
            generated_at_utc=generated_at_utc,
        )
        export_payload["model"]["metadata"] = model_metadata

        render_section_kicker("Export")
        st.subheader("Export Scenarios")
        e1, e2 = st.columns(2)
        e1.download_button(
            label="Export for slides (CSV)",
            data=export_df.to_csv(index=False),
            file_name=f"scenario_comparison_{scenario.team1}_vs_{scenario.team2}.csv".replace(" ", "_"),
            mime="text/csv",
        )
        e2.download_button(
            label="Export for audit traces (JSON)",
            data=json.dumps(export_payload, indent=2),
            file_name=f"scenario_comparison_{scenario.team1}_vs_{scenario.team2}.json".replace(" ", "_"),
            mime="application/json",
        )
        st.caption("Export for slides (CSV) or export for audit traces (JSON).")

        volatility = matchup_volatility_profile(df, scenario.team1, scenario.team2)
        st.divider()
        render_section_kicker("Risk Context")
        st.subheader("Matchup Volatility Radar")
        render_info_card(
            "Volatility Strip",
            [
                f"Historical matchups: {int(volatility['matches'])}",
                f"Historical upset rate: {volatility['upset_rate']:.1%}",
                f"Volatility index: {volatility['volatility_index']:.2f} (0 = stable, 1 = very volatile)",
            ],
        )
        if volatility["matches"] == 0:
            st.info(
                "No direct historical head-to-head rows for this matchup in current data. "
                "Volatility defaults to 0 for sparse matchups."
            )
        if not focus_mode:
            st.info(
                f"Favorite by ELO proxy: {favorite}. If {underdog} wins, it is labeled as an upset under current MVP definition."
            )

        st.divider()
        render_section_kicker("Watchlist")
        st.subheader("Upset Watchlist")
        st.caption("Rank matchups by upset likelihood and historical volatility.")
        watch_teams_default = teams[: min(8, len(teams))]
        team_game_counts = (
            pd.concat([df["team1"], df["team2"]], ignore_index=True)
            .value_counts()
            .sort_values(ascending=False)
        )
        top20_teams = team_game_counts.head(20).index.tolist()
        st.markdown("**Filters**")
        f1, f2, f3 = st.columns(3)
        quick_preset = f1.selectbox(
            "Quick preset",
            options=["Match scenario-based", "Top 10 global upsets", "Top 20 teams by games", "Fully custom"],
            key="watchlist_preset",
        )
        watch_stage_idx = stages.index(scenario.match_stage) if scenario.match_stage in stages else 0
        watch_venue_idx = venues.index(scenario.venue) if scenario.venue in venues else 0
        watch_stage = f2.selectbox("Stage filter", options=stages, index=watch_stage_idx, key="watchlist_stage")
        watch_venue = f3.selectbox("Venue filter", options=venues, index=watch_venue_idx, key="watchlist_venue")
        f4, f5 = st.columns([1, 3])
        top_n = int(f4.slider("Rows", min_value=5, max_value=20, value=10, step=1, key="watchlist_top_n"))
        selected_watch_teams = f5.multiselect(
            "Teams",
            options=teams,
            default=watch_teams_default,
            key="watchlist_teams",
        )
        with st.expander("Advanced filters", expanded=False):
            min_history = int(
                st.slider(
                    "Minimum historical head-to-head matches",
                    min_value=0,
                    max_value=10,
                    value=0,
                    step=1,
                    key="watchlist_min_history",
                )
            )
        if quick_preset == "Match scenario-based":
            watch_stage = scenario.match_stage
            watch_venue = scenario.venue
        elif quick_preset == "Top 10 global upsets":
            top_n = 10
            min_history = 0
            selected_watch_teams = teams
        elif quick_preset == "Top 20 teams by games":
            selected_watch_teams = [t for t in top20_teams if t in teams]
            top_n = min(top_n, 10)
            min_history = max(min_history, 1)
        if len(selected_watch_teams) < 2:
            st.info("Select at least two teams to build the watchlist.")
        else:
            with st.spinner("Building watchlist..."):
                watchlist = _build_watchlist(
                    model=model,
                    df=df,
                    selected_teams=selected_watch_teams,
                    selected_stage=watch_stage,
                    selected_venue=watch_venue,
                )
            if watchlist.empty:
                st.info("No watchlist scenarios could be generated from the selected filters.")
            else:
                watchlist = watchlist[watchlist["historical_matchups"] >= min_history].copy()
                if watchlist.empty:
                    st.info("No watchlist rows remain after applying the historical-matchups filter.")
                else:
                    watchlist["risk_level"] = watchlist["underdog_win_prob"].apply(
                        lambda p: build_upset_alert(float(p), watch_stage)["level"]
                    )
                    display_cols = [
                        "matchup",
                        "favorite",
                        "underdog",
                        "underdog_win_prob",
                        "risk_level",
                        "volatility_index",
                        "upset_severity_index",
                        "watchlist_score",
                        "historical_matchups",
                        "priors_source_tier",
                    ]
                    display_name_map = {
                        "rank": "Rank",
                        "matchup": "Matchup",
                        "favorite": "Favorite",
                        "underdog": "Underdog",
                        "underdog_win_prob": "Underdog Win %",
                        "risk_level": "Risk",
                        "volatility_index": "Volatility",
                        "upset_severity_index": "Severity",
                        "watchlist_score": "Watchlist Score",
                        "historical_matchups": "H2H Matches",
                        "priors_source_tier": "Priors Source",
                    }
                    watchlist["rank"] = range(1, len(watchlist) + 1)
                    display_cols_with_rank = ["rank"] + display_cols
                    display_watchlist = watchlist.head(top_n)[display_cols_with_rank].rename(columns=display_name_map)
                    st.dataframe(
                        display_watchlist.style.format(
                            {
                                "Underdog Win %": "{:.1%}",
                                "Volatility": "{:.2f}",
                                "Severity": "{:.1%}",
                                "Watchlist Score": "{:.2f}",
                            }
                        ),
                        width="stretch",
                    )
                    table_col, notes_col = st.columns([3, 2])
                    with table_col:
                        st.markdown("**Top 3 interpretation notes**")
                        for rank, (_, row) in enumerate(watchlist.head(3).iterrows(), start=1):
                            level = str(row["risk_level"])
                            st.markdown(
                                f"<span class='rank-badge'>{rank}</span>"
                                f"`{row['matchup']}`: underdog chance {row['underdog_win_prob']:.1%}, "
                                f"risk `{level}`, volatility {row['volatility_index']:.2f}, source `{row['priors_source_tier']}`.",
                                unsafe_allow_html=True,
                            )
                    with notes_col:
                        st.markdown("**Export**")
                        st.caption("Export for slides (CSV) or export for audit traces (JSON).")
                        notes_col.download_button(
                            label="Export watchlist (CSV)",
                            data=display_watchlist.to_csv(index=False),
                            file_name=f"watchlist_{watch_stage}_{watch_venue}.csv".replace(" ", "_"),
                            mime="text/csv",
                        )
                        notes_col.download_button(
                            label="Export watchlist (JSON)",
                            data=display_watchlist.to_json(orient="records", indent=2),
                            file_name=f"watchlist_{watch_stage}_{watch_venue}.json".replace(" ", "_"),
                            mime="application/json",
                        )
                    st.divider()
                    render_section_kicker("Report Snapshot")
                    st.subheader("One-Screen Summary")
                    snap_left, snap_right = st.columns(2)
                    with snap_left:
                        st.markdown("**Top Watchlist Scenarios**")
                        st.table(
                            display_watchlist[
                                ["Rank", "Matchup", "Underdog Win %", "Risk", "Volatility", "Severity"]
                            ].head(5),
                        )
                    with snap_right:
                        st.markdown("**Hot Venues**")
                        st.caption("Venues where underdogs overperform historically.")
                        venue_snapshot = full_eval.copy()
                        venue_snapshot["venue_name"] = venue_snapshot["venue"].fillna("Unknown Venue").astype(str)
                        venue_snapshot = (
                            venue_snapshot.groupby("venue_name", as_index=False)
                            .agg(matches=("is_upset", "size"), upset_rate=("is_upset", "mean"))
                            .sort_values(["upset_rate", "matches"], ascending=[False, False])
                        )
                        venue_snapshot = venue_snapshot[venue_snapshot["matches"] >= 5].head(5).rename(
                            columns={"venue_name": "Venue", "matches": "Matches", "upset_rate": "Upset Rate"}
                        )
                        st.table(
                            venue_snapshot.style.format({"Upset Rate": "{:.1%}"}),
                        )

    with tab_how:
        st.subheader("How the Logic Works")
        if focus_mode:
            st.info("Focus mode is on. Disable it in the sidebar to view full methodology notes.")
        else:
            known_stage_share = float((df["match_stage"].notna()).mean()) if "match_stage" in df.columns else 0.0
            st.markdown(
                """
                - **Favorite vs Underdog**: determined from pre-match ELO comparison.
                - **Underdog Win Probability**: explicit probability that the ELO underdog wins.
                - **Upset Severity Index**: underdog probability weighted by ELO gap size.
                - **Stage-Aware Alert**: upset alerts use dynamic thresholds by stage.
                - **Priors**: input defaults are selected by fallback tiers:
                  `matchup_venue -> matchup -> venue -> city -> global`.
                - **No Leakage Rule**: predictions use only pre-match features.
                """
            )
            st.caption(
                f"Stage coverage note: `match_stage` is present in {known_stage_share:.1%} of rows, "
                "so stage-aware thresholds only apply when stage is known."
            )
            with st.expander("What counts as an upset in this MVP?"):
                st.markdown(
                    """
                    We label an upset when the pre-match ELO underdog wins.

                    - If `elo_team1 >= elo_team2`, Team 1 is favorite and Team 2 is underdog.
                    - If `elo_team1 < elo_team2`, Team 2 is favorite and Team 1 is underdog.
                    - Upset-oriented outputs then track the underdog win chance, not just raw team win chance.
                    """
                )
            with st.expander("Home advantage note"):
                st.markdown(
                    """
                    The dataset does not provide a consistent explicit home/away flag for every match.
                    Current MVP approximates venue effects using historical priors at:
                    matchup-venue, matchup, venue, city, and global levels.
                    """
                )
            st.subheader("Review Checklist")
            st.markdown(
                """
                1. Change toss/stage/venue and verify probability movement feels plausible.
                2. Compare current vs alternative scenario and verify delta signs.
                3. Export CSV/JSON and confirm metadata and timestamp.
                4. Inspect insights and explainability tabs for alignment with simulator outputs.
                """
            )

    with tab_insights:
        st.markdown("## Model Quality & Patterns")
        render_section_kicker("Model Quality")
        st.subheader("Model Diagnostics")
        diagnostics_scope = st.radio(
            "Diagnostics data scope",
            options=["Test only", "All data"],
            horizontal=True,
            key="insights_diagnostics_scope",
        )
        diagnostics_eval = test_eval if diagnostics_scope == "Test only" else full_eval
        st.caption(
            "Scope note: selected scope applies to correlation, ELO-bucket diagnostics, and venue hotspot summaries."
        )
        valid_metrics = evaluate_binary_model(model, X_valid, y_valid)
        test_metrics = evaluate_binary_model(model, X_test, y_test)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Valid ROC AUC", f"{valid_metrics['roc_auc']:.3f}")
        m2.metric("Valid Log Loss", f"{valid_metrics['log_loss']:.3f}")
        m3.metric("Test ROC AUC", f"{test_metrics['roc_auc']:.3f}")
        m4.metric("Test Brier", f"{test_metrics['brier']:.3f}")
        roc_gap, roc_note = roc_auc_generalization_note(valid_metrics["roc_auc"], test_metrics["roc_auc"])
        st.caption(
            f"ROC AUC gap (test - valid): {roc_gap:+.3f}. {roc_note}"
        )
        with st.expander("How to interpret model trust signals"):
            st.markdown(
                """
                - **Random baseline** is ROC AUC `0.50`; current scores around `0.63` indicate moderate signal lift.
                - Use this model primarily for **ranking upset-risk scenarios**, not for exact probability certainty.
                - Combine these metrics with calibration and venue/sample-size diagnostics before decisions.
                """
            )
        st.caption(
            "Method note: validation metrics are computed on the same window used for calibration; "
            "test metrics are the best out-of-sample estimate."
        )
        if not focus_mode:
            with st.expander("Metric explanations"):
                st.markdown(
                    """
                    - **ROC AUC**: ranking quality; higher is better (0.5 random, 1.0 perfect).
                    - **Log Loss**: probability calibration + accuracy penalty; lower is better.
                    - **Brier Score**: mean squared error of predicted probabilities; lower is better.
                    """
                )

        render_section_kicker("Calibration")
        st.subheader("Calibration Curve")
        valid_probs = model.predict_proba(X_valid)[:, 1]
        fig, _ = plot_calibration_curve(y_valid, valid_probs, n_bins=10)
        st.pyplot(fig, width="content")
        plt.close(fig)

        render_section_kicker("Upset Patterns")
        st.subheader("Upset Rate by ELO Gap Bucket")
        corr_df = diagnostics_eval[["underdog_win_prob", "upset_severity_index"]].copy()
        corr_value = corr_df.corr().iloc[0, 1] if len(corr_df) >= 2 else float("nan")
        if pd.notna(corr_value):
            corr_float = float(corr_value)
            corr_label = correlation_strength_label(corr_float)
            st.caption(f"Correlation (probability vs severity): {corr_float:+.3f} ({corr_label}).")
            st.caption("Severity is intentionally related, but scaled by ELO gap.")
        bucket_df = diagnostics_eval.copy()
        bucket_df["abs_elo_diff"] = bucket_df["elo_diff"].abs()
        bucket_table = upset_rate_by_bucket(bucket_df, "abs_elo_diff", "is_upset", bins=8)
        bucket_table["predicted_upset_rate"] = bucket_table["bucket"].apply(
            lambda b: float(
                bucket_df[
                    (bucket_df["abs_elo_diff"] > float(b.left))
                    & (bucket_df["abs_elo_diff"] <= float(b.right))
                ]["underdog_win_prob"].mean()
            )
            if isinstance(b, pd.Interval)
            else float("nan")
        )
        bucket_table["elo_gap_range"] = bucket_table["bucket"].apply(
            lambda b: f"{int(round(b.left))}-{int(round(b.right))}" if isinstance(b, pd.Interval) else str(b)
        )
        st.bar_chart(
            bucket_table.set_index("elo_gap_range")[["upset_rate", "predicted_upset_rate"]],
            width="stretch",
        )
        st.table(
            bucket_table[["elo_gap_range", "upset_rate", "predicted_upset_rate", "matches"]].style.format(
                {"upset_rate": "{:.1%}", "predicted_upset_rate": "{:.1%}"}
            ),
        )

        render_section_kicker("Venue Hotspots")
        st.subheader("Venues with the Most Upsets")
        venue_df = diagnostics_eval.copy()
        venue_df["venue_name"] = venue_df["venue"].fillna("Unknown Venue").astype(str)
        venue_summary = (
            venue_df.groupby("venue_name", as_index=False)
            .agg(
                matches=("is_upset", "size"),
                upset_count=("is_upset", "sum"),
                upset_rate=("is_upset", "mean"),
                predicted_upset_rate=("underdog_win_prob", "mean"),
            )
            .sort_values(["upset_rate", "upset_count"], ascending=[False, False])
        )
        rank_mode = st.selectbox(
            "Rank venues by",
            options=["Upset rate", "Upset count"],
            index=0,
            key="venue_rank_mode",
        )
        min_venue_matches = int(
            st.slider(
                "Minimum matches per venue",
                min_value=1,
                max_value=30,
                value=5,
                step=1,
                key="venue_upset_min_matches",
            )
        )
        top_venue_rows = int(
            st.slider(
                "Top venues to show",
                min_value=5,
                max_value=25,
                value=15,
                step=1,
                key="venue_upset_top_rows",
            )
        )
        venue_filtered = venue_summary[venue_summary["matches"] >= min_venue_matches].copy()
        if venue_filtered.empty:
            st.info("No venues satisfy the selected minimum-match threshold.")
        else:
            sort_cols = ["upset_rate", "upset_count"] if rank_mode == "Upset rate" else ["upset_count", "upset_rate"]
            top_venues = venue_filtered.sort_values(sort_cols, ascending=[False, False]).head(top_venue_rows).copy()
            top_venues["sample_caution"] = top_venues["matches"].apply(
                lambda m: venue_sample_caution_label(int(m), caution_threshold=10)
            )
            if (top_venues["sample_caution"] == "Low sample").any():
                st.warning("Some displayed venues are low-sample (<10 matches). Interpret upset rates cautiously.")
            st.bar_chart(
                top_venues.set_index("venue_name")[["upset_rate", "predicted_upset_rate"]],
                width="stretch",
            )
            st.dataframe(
                top_venues[
                    ["venue_name", "matches", "sample_caution", "upset_count", "upset_rate", "predicted_upset_rate"]
                ].style.format(
                    {
                        "upset_rate": "{:.1%}",
                        "predicted_upset_rate": "{:.1%}",
                    }
                ),
                width="stretch",
            )

        render_section_kicker("Classification Focus")
        st.subheader("Upset Classification Diagnostics (Test Window)")
        upset_precision, upset_recall, upset_f1, _ = precision_recall_fscore_support(
            test_eval["is_upset"],
            test_eval["pred_is_upset"],
            average="binary",
            zero_division=0,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Upset Precision", f"{float(upset_precision):.3f}")
        c2.metric("Upset Recall", f"{float(upset_recall):.3f}")
        c3.metric("Upset F1", f"{float(upset_f1):.3f}")
        cm = confusion_matrix(test_eval["is_upset"], test_eval["pred_is_upset"])
        st.caption("Confusion matrix rows=true class (no upset/upset), columns=predicted class.")
        st.dataframe(
            pd.DataFrame(cm, index=["true_no_upset", "true_upset"], columns=["pred_no_upset", "pred_upset"]),
            width="content",
        )

        render_section_kicker("Interactive Cases")
        st.subheader("Notable Historical Upsets")
        top_n_notable = st.slider("Top list size", min_value=5, max_value=30, value=12, step=1, key="notable_top_n")
        notable = rank_notable_upsets(full_eval, top_n=top_n_notable).copy()
        if notable.empty:
            st.info("No upset rows found with current filters.")
        else:
            notable["predicted_upset_chance"] = full_eval.loc[notable.index, "underdog_win_prob"]
            notable["predicted_upset_severity"] = full_eval.loc[notable.index, "upset_severity_index"]
            notable["upset_rank"] = range(1, len(notable) + 1)
            notable["event_name"] = notable.apply(build_event_label, axis=1)
            notable["date_label"] = notable["date"].apply(build_date_label)
            notable["loser"] = notable.apply(
                lambda r: str(r["team2"]) if str(r["winner"]) == str(r["team1"]) else str(r["team1"]),
                axis=1,
            )
            notable["matchup"] = notable.apply(
                lambda r: f"{team_flag(str(r['team1']))} {r['team1']} vs {team_flag(str(r['team2']))} {r['team2']}",
                axis=1,
            )
            notable["winner_display"] = notable.apply(
                lambda r: f"{team_flag(str(r['winner']))} {r['winner']}",
                axis=1,
            )
            notable["top_list_label"] = notable.apply(
                lambda r: (
                    f"#{int(r['upset_rank'])} - {team_flag(str(r['winner']))} {r['winner']} beat "
                    f"{team_flag(str(r['loser']))} {r['loser']} "
                    f"({r['event_name']}, {r['date_label']})"
                ),
                axis=1,
            )
            display_cols = [
                "upset_rank",
                "matchup",
                "winner_display",
                "event_name",
                "date_label",
                "elo_diff",
                "predicted_upset_chance",
                "predicted_upset_severity",
                "match_stage",
                "venue",
            ]
            st.dataframe(
                notable[[c for c in display_cols if c in notable.columns]].style.format(
                    {
                        "predicted_upset_chance": "{:.1%}",
                        "predicted_upset_severity": "{:.1%}",
                    }
                ),
                width="stretch",
            )
            st.markdown("**Interactive Top List**")
            st.caption("Explore historical upsets similar to your current scenario.")
            selected_idx = st.selectbox(
                "Pick a notable upset",
                options=notable.index.tolist(),
                format_func=lambda idx: str(notable.loc[idx, "top_list_label"]),
                key="insights_notable_case",
            )
            selected_row = notable.loc[selected_idx]
            i1, i2, i3 = st.columns(3)
            i1.metric("Upset chance", f"{float(selected_row['predicted_upset_chance']):.1%}")
            i2.metric("Upset Severity", f"{float(selected_row['predicted_upset_severity']):.1%}")
            i3.metric("ELO Gap (abs)", f"{abs(float(selected_row['elo_diff'])):.1f}")
            st.progress(float(selected_row["predicted_upset_chance"]), text="Upset chance gauge")
            selected_winner = str(selected_row["winner"])
            selected_loser = str(selected_row["loser"])
            selected_event = str(selected_row["event_name"])
            selected_date = str(selected_row["date_label"])
            st.caption(
                f"{team_flag(selected_winner)} {notable_upset_sentence(selected_winner, selected_loser, selected_event, selected_date)}"
            )

    with tab_explain:
        render_section_kicker("Local What-If")
        st.subheader("Local Counterfactual Explanation")
        notable = rank_notable_upsets(full_eval, top_n=12)
        if notable.empty:
            st.info("No upset cases available for local explanation.")
        else:
            case_options = notable.index.tolist()
            selected_idx = st.selectbox(
                "Select upset case",
                options=case_options,
                format_func=lambda idx: (
                    f"{str(notable.loc[idx].get('date', 'unknown'))[:10]} | "
                    f"{notable.loc[idx].get('team1', '')} vs {notable.loc[idx].get('team2', '')}"
                ),
                key="explain_upset_case",
            )
            required_cols = [
                "team1",
                "team2",
                "match_stage",
                "venue",
                "toss_winner",
                "toss_decision",
                "elo_team1",
                "elo_team2",
                "elo_diff",
                "team1_form_5",
                "team2_form_5",
                "team1_form_10",
                "team2_form_10",
                "h2h_win_pct",
            ]
            try:
                base_row = df.loc[[selected_idx], [c for c in required_cols if c in df.columns]]
                if base_row.empty:
                    st.warning("Selected case has missing feature columns. Try another upset.")
                else:
                    local_exp = build_counterfactual_explanation(model, base_row)
                    st.metric("Base Team1 Win Probability", f"{local_exp['base_team1_win_prob']:.1%}")
                    st.dataframe(
                        pd.DataFrame(local_exp["counterfactuals"]).style.format(
                            {"team1_win_prob": "{:.1%}", "delta_vs_base": "{:+.1%}"}
                        ),
                        width="stretch",
                    )
                    st.caption(
                        "Counterfactual rows show how one targeted change shifts probability from the base case; "
                        "larger absolute delta means a stronger local influence."
                    )
            except (KeyError, ValueError, IndexError) as e:
                st.error(f"Could not generate explanation for this case: {e}")

        render_section_kicker("Missed Cases")
        st.subheader("Missed Upsets Audit")
        missed = build_missed_upsets_audit(test_eval, top_n=15)
        if missed.empty:
            st.info("No high-confidence upset misses found on current test window.")
        else:
            cols = [
                "date",
                "team1",
                "team2",
                "winner",
                "favorite_team",
                "favorite_confidence",
                "match_stage",
                "venue",
            ]
            st.dataframe(missed[[c for c in cols if c in missed.columns]], width="stretch")

        render_section_kicker("Narrative Explorer")
        st.subheader("Curated Upset Narratives")
        curated = build_curated_upset_narratives(df, top_n=10)
        if curated.empty:
            st.info("No curated narrative cases available.")
        else:
            narrative_cols = [
                "date",
                "team1",
                "team2",
                "winner",
                "favorite_team",
                "elo_gap",
                "match_stage",
                "narrative",
            ]
            st.dataframe(curated[[c for c in narrative_cols if c in curated.columns]], width="stretch")
            summaries = summarize_curated_upset_patterns(curated)
            s1, s2 = st.columns(2)
            with s1:
                st.markdown("**Stage Pattern Summary**")
                st.dataframe(summaries["stage_summary"], width="stretch")
            with s2:
                st.markdown("**Venue Pattern Summary**")
                st.dataframe(summaries["venue_summary"], width="stretch")


if __name__ == "__main__":
    main()

