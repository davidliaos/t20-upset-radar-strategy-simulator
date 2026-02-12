# Future Roadmap Options

This document captures forward-looking implementation tracks proposed by a parallel research agent while core implementation continues.

## Track A - Research Heavy

Focus: maximize model quality, upset insight depth, and explainability.

Sprint ideas:

1. Compare logistic vs XGBoost across validation/test (AUC, Brier, log loss).
2. Run calibration analysis and reliability plots.
3. Build upset-specific evaluation (precision/recall on underdog wins).
4. Add SHAP global + local explanations for notable upsets.

Main risks:

- increased compute time for SHAP and large-sample diagnostics
- analysis complexity can outpace product hardening

## Track B - Product Heavy

Focus: simulator usability and decision support.

Sprint ideas:

1. Persist/load calibrated artifacts in `models/` (implemented baseline version now).
2. Add smarter defaults for scenario controls based on selected teams and venue.
3. Expand side-by-side scenario comparison beyond toss decision.
4. Add report export and upset alerts.

Main risks:

- UX complexity growth
- feature drift between training and inference if defaults are not validated

## Track C - Engineering Quality Heavy

Focus: reliability, reproducibility, and maintainability.

Sprint ideas:

1. Expand unit tests and add integration smoke tests.
2. Add CI checks (`pytest`, lint, optional type checks).
3. Add reproducible train/save script and metadata versioning.
4. Complete quality checklist and leakage guards in docs.

Main risks:

- slower short-term feature velocity
- up-front setup overhead

## Recommended Near-Term Blend

Use a blended path: Product + Engineering first, then deeper Research.

1. Keep shipping product-critical improvements (artifact loading, scenario quality).
2. Maintain test coverage for each added module.
3. Add research depth once persistence and quality gates are stable.

