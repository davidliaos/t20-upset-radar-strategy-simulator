"""T20 Upset Radar core package."""

from .data_prep import (
    assign_favorite_underdog_from_elo,
    build_team1_win_target,
    load_matches,
    time_based_split,
)
from .data_quality import build_data_quality_report, save_data_quality_report
from .explain import (
    build_counterfactual_explanation,
    build_curated_upset_narratives,
    build_missed_upsets_audit,
    matchup_volatility_profile,
    rank_notable_upsets,
    summarize_curated_upset_patterns,
)
from .features import add_context_interactions, build_pre_match_feature_frame
from .models import (
    calibrate_classifier,
    evaluate_binary_model,
    load_model_artifacts,
    save_model_artifacts,
    train_logistic_baseline,
)
from .simulation import (
    ScenarioInput,
    build_scenario_export_payload,
    build_scenario_features,
    estimate_scenario_defaults,
    estimate_scenario_defaults_with_meta,
    normalize_matchup_rows,
    prior_confidence_label,
    score_scenario,
)

__all__ = [
    "assign_favorite_underdog_from_elo",
    "build_team1_win_target",
    "load_matches",
    "time_based_split",
    "add_context_interactions",
    "build_pre_match_feature_frame",
    "build_data_quality_report",
    "save_data_quality_report",
    "rank_notable_upsets",
    "build_missed_upsets_audit",
    "build_curated_upset_narratives",
    "summarize_curated_upset_patterns",
    "build_counterfactual_explanation",
    "matchup_volatility_profile",
    "calibrate_classifier",
    "evaluate_binary_model",
    "save_model_artifacts",
    "load_model_artifacts",
    "train_logistic_baseline",
    "ScenarioInput",
    "build_scenario_export_payload",
    "build_scenario_features",
    "normalize_matchup_rows",
    "estimate_scenario_defaults",
    "estimate_scenario_defaults_with_meta",
    "prior_confidence_label",
    "score_scenario",
]

