"""Streamlit MVP for the T20 Upset Radar strategy simulator."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import streamlit as st

from src.data_prep import (
    assign_favorite_underdog_from_elo,
    build_team1_win_target,
    load_matches,
    time_based_split,
)
from src.features import build_pre_match_feature_frame
from src.models import (
    build_default_metadata,
    calibrate_classifier,
    load_model_artifacts,
    save_model_artifacts,
    train_logistic_baseline,
)
from src.simulation import (
    ScenarioInput,
    build_scenario_features,
    estimate_scenario_defaults,
    score_scenario,
)


@st.cache_data
def get_matches() -> pd.DataFrame:
    df = load_matches()
    df = build_team1_win_target(df)
    return assign_favorite_underdog_from_elo(df)


@st.cache_resource
def get_baseline_model():
    df = get_matches()
    train_df, valid_df, _ = time_based_split(df)
    model_name = "baseline_logistic_calibrated"

    try:
        artifacts = load_model_artifacts(model_name=model_name)
        return artifacts.model, df, "loaded"
    except FileNotFoundError:
        X_train, y_train = build_pre_match_feature_frame(train_df)
        X_valid, y_valid = build_pre_match_feature_frame(valid_df)
        baseline = train_logistic_baseline(X_train, y_train)
        model = calibrate_classifier(baseline, X_valid, y_valid)
        save_model_artifacts(model, model_name=model_name, metadata=build_default_metadata())
        return model, df, "trained"


def main() -> None:
    st.set_page_config(page_title="T20 Upset Radar", layout="wide")
    st.title("T20 Upset Radar and Strategy Simulator")
    st.caption("MVP simulator using pre-match features and a calibrated logistic baseline.")

    model, df, model_source = get_baseline_model()
    st.caption(f"Model source: {model_source} artifact (`models/baseline_logistic_calibrated.joblib`).")
    teams = sorted(set(df["team1"]).union(set(df["team2"])))
    venues = sorted(df["venue"].dropna().astype(str).unique().tolist())
    stages = sorted(df["match_stage"].dropna().astype(str).unique().tolist())

    with st.sidebar:
        st.header("Scenario Inputs")
        team1 = st.selectbox("Team 1", options=teams, index=0)
        team2_options = [t for t in teams if t != team1]
        team2 = st.selectbox("Team 2", options=team2_options, index=0)
        venue = st.selectbox("Venue", options=venues, index=0)

        defaults = estimate_scenario_defaults(df, team1, team2, venue)
        use_priors = st.checkbox("Use matchup priors for numeric defaults", value=True)

        stage_index = stages.index(defaults["match_stage"]) if defaults["match_stage"] in stages else 0
        stage = st.selectbox("Match Stage", options=stages, index=stage_index)
        toss_winner = st.selectbox("Toss Winner", options=[team1, team2], index=0)
        toss_decision_index = 0 if defaults["toss_decision"] == "bat" else 1
        toss_decision = st.selectbox("Toss Decision", options=["bat", "field"], index=toss_decision_index)

        # Lightweight proxies users can tweak for what-if analysis.
        def prior_or(key: str, fallback: float) -> float:
            value = defaults.get(key)
            if use_priors and isinstance(value, (int, float)):
                return float(value)
            return fallback

        elo_team1 = st.number_input("ELO Team 1", value=prior_or("elo_team1", 1500.0), step=10.0)
        elo_team2 = st.number_input("ELO Team 2", value=prior_or("elo_team2", 1500.0), step=10.0)
        team1_form_5 = st.slider("Team 1 Form (Last 5)", 0.0, 1.0, prior_or("team1_form_5", 0.5), 0.01)
        team2_form_5 = st.slider("Team 2 Form (Last 5)", 0.0, 1.0, prior_or("team2_form_5", 0.5), 0.01)
        team1_form_10 = st.slider("Team 1 Form (Last 10)", 0.0, 1.0, prior_or("team1_form_10", 0.5), 0.01)
        team2_form_10 = st.slider("Team 2 Form (Last 10)", 0.0, 1.0, prior_or("team2_form_10", 0.5), 0.01)
        h2h_win_pct = st.slider("Team 1 Head-to-Head Win %", 0.0, 1.0, prior_or("h2h_win_pct", 0.5), 0.01)

    scenario = ScenarioInput(
        team1=team1,
        team2=team2,
        venue=venue,
        match_stage=stage,
        toss_winner=toss_winner,
        toss_decision=toss_decision,
        elo_team1=elo_team1,
        elo_team2=elo_team2,
        team1_form_5=team1_form_5,
        team2_form_5=team2_form_5,
        team1_form_10=team1_form_10,
        team2_form_10=team2_form_10,
        h2h_win_pct=h2h_win_pct,
    )

    feature_row = build_scenario_features(scenario)
    result = score_scenario(model, feature_row)

    favorite = team1 if elo_team1 >= elo_team2 else team2
    underdog = team2 if favorite == team1 else team1

    col1, col2, col3 = st.columns(3)
    col1.metric(f"{team1} Win Probability", f"{result['team1_win_prob']:.1%}")
    col2.metric(f"{team2} Win Probability", f"{result['team2_win_prob']:.1%}")
    col3.metric("Upset Risk", f"{result['upset_risk']:.1%}")

    st.subheader("Scenario Comparison")
    alt_decision = "field" if toss_decision == "bat" else "bat"
    alt_scenario = replace(scenario, toss_decision=alt_decision)
    alt_result = score_scenario(model, build_scenario_features(alt_scenario))
    comparison_df = pd.DataFrame(
        [
            {"scenario": f"Current ({toss_decision})", "team1_win_prob": result["team1_win_prob"], "upset_risk": result["upset_risk"]},
            {"scenario": f"Alternative ({alt_decision})", "team1_win_prob": alt_result["team1_win_prob"], "upset_risk": alt_result["upset_risk"]},
        ]
    )
    st.dataframe(comparison_df.style.format({"team1_win_prob": "{:.1%}", "upset_risk": "{:.1%}"}), use_container_width=True)

    st.info(
        f"Favorite by ELO proxy: {favorite}. If {underdog} wins, it is labeled as an upset under current MVP definition."
    )


if __name__ == "__main__":
    main()

