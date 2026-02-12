"""T20 Upset Radar core package."""

from .data_prep import (
    assign_favorite_underdog_from_elo,
    build_team1_win_target,
    load_matches,
    time_based_split,
)
from .data_quality import build_data_quality_report, save_data_quality_report
from .explain import build_counterfactual_explanation, rank_notable_upsets
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
    build_scenario_features,
    estimate_scenario_defaults,
    normalize_matchup_rows,
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
    "build_counterfactual_explanation",
    "calibrate_classifier",
    "evaluate_binary_model",
    "save_model_artifacts",
    "load_model_artifacts",
    "train_logistic_baseline",
    "ScenarioInput",
    "build_scenario_features",
    "normalize_matchup_rows",
    "estimate_scenario_defaults",
    "score_scenario",
]

