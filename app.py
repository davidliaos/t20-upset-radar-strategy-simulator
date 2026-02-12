"""Streamlit dashboard for the T20 Upset Radar strategy simulator."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
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
from src.viz import plot_calibration_curve, upset_rate_by_bucket


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
        st.header("Scenario Inputs")
        with st.expander("Match Setup", expanded=True):
            team1 = st.selectbox("Team 1", options=teams, index=0)
            team2_options = [t for t in teams if t != team1]
            team2 = st.selectbox("Team 2", options=team2_options, index=0)
            venue = st.selectbox("Venue", options=venues, index=0)

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
        st.caption(
            "Priors source: "
            f"`{source_tier}` ({int(defaults_meta['source_rows'])} rows), confidence `{confidence}`."
        )
        if int(defaults_meta["source_rows"]) < 10:
            st.warning("Low historical sample size for priors. Use scenario outputs with caution.")
        use_priors = st.checkbox("Use matchup priors for numeric defaults", value=True)

        with st.expander("Match Context", expanded=True):
            stage_index = stages.index(defaults["match_stage"]) if defaults["match_stage"] in stages else 0
            stage = st.selectbox("Match Stage", options=stages, index=stage_index)
            toss_winner = st.selectbox("Toss Winner", options=[team1, team2], index=0)
            toss_decision_index = 0 if defaults["toss_decision"] == "bat" else 1
            toss_decision = st.selectbox("Toss Decision", options=["bat", "field"], index=toss_decision_index)

        def prior_or(key: str, fallback: float) -> float:
            value = defaults.get(key)
            if use_priors and isinstance(value, (int, float)):
                return float(value)
            return fallback

        with st.expander("Strength and Form Controls", expanded=True):
            elo_team1 = st.number_input("ELO Team 1", value=prior_or("elo_team1", 1500.0), step=10.0)
            elo_team2 = st.number_input("ELO Team 2", value=prior_or("elo_team2", 1500.0), step=10.0)
            team1_form_5 = st.slider("Team 1 Form (Last 5)", 0.0, 1.0, prior_or("team1_form_5", 0.5), 0.01)
            team2_form_5 = st.slider("Team 2 Form (Last 5)", 0.0, 1.0, prior_or("team2_form_5", 0.5), 0.01)
            team1_form_10 = st.slider("Team 1 Form (Last 10)", 0.0, 1.0, prior_or("team1_form_10", 0.5), 0.01)
            team2_form_10 = st.slider("Team 2 Form (Last 10)", 0.0, 1.0, prior_or("team2_form_10", 0.5), 0.01)
            h2h_win_pct = st.slider("Team 1 Head-to-Head Win %", 0.0, 1.0, prior_or("h2h_win_pct", 0.5), 0.01)

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


def main() -> None:
    st.set_page_config(page_title="T20 Upset Radar", layout="wide")
    st.title("T20 Upset Radar Dashboard")
    st.caption("Calibrated T20 upset simulator with explainability and model diagnostics.")

    model, df, model_source, model_name, model_metadata = get_baseline_model()
    st.caption(f"Model source: {model_source} artifact (`models/baseline_logistic_calibrated.joblib`).")
    trained_at = model_metadata.get("trained_at_utc")
    if isinstance(trained_at, str):
        st.caption(f"Model metadata timestamp (UTC): {trained_at}")
    train_df, valid_df, test_df, X_train, y_train, X_valid, y_valid, X_test, y_test = get_split_data(df)
    _ = (train_df, X_train)  # explicit placeholders for readability.
    test_eval = _prepare_test_eval(model, test_df)
    full_eval = _prepare_eval_predictions(model, df)

    teams = sorted(set(df["team1"]).union(set(df["team2"])))
    venues = sorted(df["venue"].dropna().astype(str).unique().tolist())
    stages = sorted(df["match_stage"].dropna().astype(str).unique().tolist())
    inputs = _build_sidebar_inputs(df, teams, venues, stages)

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

    feature_row = build_scenario_features(scenario)
    result = score_scenario(model, feature_row)
    favorite = scenario.team1 if scenario.elo_team1 >= scenario.elo_team2 else scenario.team2
    underdog = scenario.team2 if favorite == scenario.team1 else scenario.team1

    tabs = st.tabs(["Simulator", "How It Works", "Insights", "Explainability"])
    tab_sim, tab_how, tab_insights, tab_explain = tabs

    with tab_sim:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"{scenario.team1} Win Probability", f"{result['team1_win_prob']:.1%}")
        col1.caption("Probability this side wins under current scenario assumptions.")
        col2.metric(f"{scenario.team2} Win Probability", f"{result['team2_win_prob']:.1%}")
        col2.caption("Complementary win probability for the opposing side.")
        col3.metric("Underdog Win Probability", f"{result['underdog_win_prob']:.1%}")
        col3.caption("Probability that the ELO underdog wins this matchup.")
        col4.metric("Upset Severity Index", f"{result['upset_severity_index']:.1%}")
        col4.caption("Underdog win probability weighted by ELO gap size.")

        alert = build_upset_alert(result["upset_risk"], scenario.match_stage)
        if alert["level"] == "high":
            st.error(f"{alert['message']} (threshold: {alert['threshold']:.0%})")
        elif alert["level"] == "medium":
            st.warning(f"{alert['message']} (threshold: {alert['threshold']:.0%})")
        else:
            st.success(f"{alert['message']} (threshold: {alert['threshold']:.0%})")

        st.subheader("Scenario Comparison")
        alt_decision = "field" if scenario.toss_decision == "bat" else "bat"
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
                },
                {
                    "scenario": f"Alternative ({alt_decision})",
                    "team1_win_prob": alt_result["team1_win_prob"],
                    "underdog_win_prob": alt_result["underdog_win_prob"],
                    "upset_severity_index": alt_result["upset_severity_index"],
                },
            ]
        )
        st.dataframe(
            comparison_df.style.format(
                {
                    "team1_win_prob": "{:.1%}",
                    "underdog_win_prob": "{:.1%}",
                    "upset_severity_index": "{:.1%}",
                }
            ),
            use_container_width=True,
        )
        st.caption(
            f"Scenario delta (alternative - current): team1 win {delta_team1:+.1%}, underdog win {delta_underdog:+.1%}."
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

        st.subheader("Export Scenarios")
        e1, e2 = st.columns(2)
        e1.download_button(
            label="Download comparison CSV",
            data=export_df.to_csv(index=False),
            file_name=f"scenario_comparison_{scenario.team1}_vs_{scenario.team2}.csv".replace(" ", "_"),
            mime="text/csv",
        )
        e2.download_button(
            label="Download scenario JSON",
            data=json.dumps(export_payload, indent=2),
            file_name=f"scenario_comparison_{scenario.team1}_vs_{scenario.team2}.json".replace(" ", "_"),
            mime="application/json",
        )

        volatility = matchup_volatility_profile(df, scenario.team1, scenario.team2)
        st.subheader("Matchup Volatility Radar")
        v1, v2, v3 = st.columns(3)
        v1.metric("Historical Matchups", f"{int(volatility['matches'])}")
        v2.metric("Historical Upset Rate", f"{volatility['upset_rate']:.1%}")
        v3.metric("Volatility Index", f"{volatility['volatility_index']:.2f}")
        if volatility["matches"] == 0:
            st.info(
                "No direct historical head-to-head rows for this matchup in current data. "
                "Volatility defaults to 0 for sparse matchups."
            )
        st.info(
            f"Favorite by ELO proxy: {favorite}. If {underdog} wins, it is labeled as an upset under current MVP definition."
        )

    with tab_how:
        st.subheader("How the Logic Works")
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
        st.subheader("Model Diagnostics")
        valid_metrics = evaluate_binary_model(model, X_valid, y_valid)
        test_metrics = evaluate_binary_model(model, X_test, y_test)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Valid ROC AUC", f"{valid_metrics['roc_auc']:.3f}")
        m2.metric("Valid Log Loss", f"{valid_metrics['log_loss']:.3f}")
        m3.metric("Test ROC AUC", f"{test_metrics['roc_auc']:.3f}")
        m4.metric("Test Brier", f"{test_metrics['brier']:.3f}")
        with st.expander("Metric explanations"):
            st.markdown(
                """
                - **ROC AUC**: ranking quality; higher is better (0.5 random, 1.0 perfect).
                - **Log Loss**: probability calibration + accuracy penalty; lower is better.
                - **Brier Score**: mean squared error of predicted probabilities; lower is better.
                """
            )

        st.subheader("Calibration Curve")
        valid_probs = model.predict_proba(X_valid)[:, 1]
        fig, _ = plot_calibration_curve(y_valid, valid_probs, n_bins=10)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

        st.subheader("Upset Rate by ELO Gap Bucket")
        bucket_df = test_eval.copy()
        bucket_df["abs_elo_diff"] = bucket_df["elo_diff"].abs()
        bucket_table = upset_rate_by_bucket(bucket_df, "abs_elo_diff", "is_upset", bins=8)
        bucket_table["predicted_upset_rate"] = bucket_table["bucket"].apply(
            lambda b: float(
                bucket_df[
                    bucket_df["abs_elo_diff"].between(float(b.left), float(b.right), inclusive="right")
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
            use_container_width=True,
        )
        st.dataframe(
            bucket_table[["elo_gap_range", "upset_rate", "predicted_upset_rate", "matches"]],
            use_container_width=True,
        )

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
            use_container_width=False,
        )

        st.subheader("Notable Historical Upsets")
        notable = rank_notable_upsets(full_eval, top_n=12).copy()
        if notable.empty:
            st.info("No upset rows found with current filters.")
        else:
            notable["predicted_upset_chance"] = full_eval.loc[notable.index, "underdog_win_prob"]
            notable["predicted_upset_severity"] = full_eval.loc[notable.index, "upset_severity_index"]
            display_cols = [
                "date",
                "team1",
                "team2",
                "winner",
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
                use_container_width=True,
            )

    with tab_explain:
        st.subheader("Local Counterfactual Explanation")
        notable = rank_notable_upsets(df, top_n=12)
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
            base_row = df.loc[[selected_idx], [c for c in required_cols if c in df.columns]]
            local_exp = build_counterfactual_explanation(model, base_row)
            st.metric("Base Team1 Win Probability", f"{local_exp['base_team1_win_prob']:.1%}")
            st.dataframe(pd.DataFrame(local_exp["counterfactuals"]), use_container_width=True)

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
            st.dataframe(missed[[c for c in cols if c in missed.columns]], use_container_width=True)

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
            st.dataframe(curated[[c for c in narrative_cols if c in curated.columns]], use_container_width=True)
            summaries = summarize_curated_upset_patterns(curated)
            s1, s2 = st.columns(2)
            with s1:
                st.markdown("**Stage Pattern Summary**")
                st.dataframe(summaries["stage_summary"], use_container_width=True)
            with s2:
                st.markdown("**Venue Pattern Summary**")
                st.dataframe(summaries["venue_summary"], use_container_width=True)


if __name__ == "__main__":
    main()

