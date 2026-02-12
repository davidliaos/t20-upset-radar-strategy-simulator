# Code Conventions

Practical standards for the T20 Upset Radar project. Keep these concise and actionable.

## Module Boundaries

- **`src/data_prep.py`**: Load data, build target, upset labels, time-based splits. No feature logic.
- **`src/features.py`**: Pre-match feature frame construction. No model training.
- **`src/models.py`**: Preprocessing, training, calibration, evaluation, artifact I/O.
- **`src/simulation.py`**: Scenario inputs → model-ready features, scoring.
- **`src/explain.py`**: Local/global explanation utilities and counterfactual narratives.
- **`src/viz.py`**: Plotting and visualization helpers.

Import from `src` in notebooks and `app.py`; avoid cross-module cycles.

## Typing and Docstrings

- Add type hints for public function parameters and return values.
- One-line docstrings for simple functions; multi-line for non-obvious behavior.
- Prefer `from __future__ import annotations` for forward references.

## Naming

- **Functions**: `snake_case` (e.g. `build_pre_match_feature_frame`).
- **Constants**: `UPPER_SNAKE_CASE` (e.g. `NUMERIC_FEATURES`, `RAW_DATA_PATH`).
- **Classes**: `PascalCase` (e.g. `ScenarioInput`).
- Private helpers: leading `_` (e.g. `_build_train_valid_sets`).

## Leakage Rules

- Only pre-match variables in model features.
- Do not use `winner`, innings, or result outcome fields in `X`.
- Use `time_based_split()` for chronological train/validation/test; never shuffle by default.
- Keep leakage exclusions documented in `docs/design.md` and `src/features.py`.

## Testing Expectations

- Unit tests in `tests/` mirror `src/` structure (e.g. `test_models.py` → `src/models.py`).
- Use pytest; fixtures for shared setup (e.g. `_build_train_valid_sets`).
- Run `pytest` from project root; CI enforces passing tests.
- Prefer assertions on observable behavior over internals.

## Notebook Hygiene

- Set `PROJECT_ROOT` and extend `sys.path` for imports from project root.
- Import from `src` modules; avoid large inline logic.
- Use `time_based_split`, `build_pre_match_feature_frame`, and model helpers from `src`.
- Include fallbacks for expensive or environment-sensitive steps (e.g. SHAP plotting) with clear messages.
- Keep cells runnable in order; avoid hidden state assumptions.

## Streamlit Patterns

- Use `@st.cache_data` for dataframe loading; `@st.cache_resource` for model loading.
- Cache keys should reflect inputs; avoid stale data when source changes.
- Prefer `st.caption` for model source and metadata.
- Use `ScenarioInput` and `build_scenario_features` from `src/simulation.py` for consistency.
