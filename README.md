# T20 Upset Radar and Strategy Simulator

This project builds a calibrated match outcome modeling pipeline for T20 World Cups with a specific focus on underdog upsets. It pairs an analysis workflow (notebooks + reusable Python modules) with an interactive Streamlit simulator to test "what-if" scenarios such as toss choice, venue, and stage context.

## Problem Framing

Most cricket forecasting work optimizes for generic winner prediction. This project targets a narrower and more useful question: when does an underdog beat a favorite, and how do pre-match conditions shift upset risk?

Two deliverables drive the work:

- Upset Radar: detect and analyze historical upsets.
- Strategy Simulator: compare scenario-level win probabilities and upset risk.

## Dataset

Primary dataset:

- `data/raw/world_cup_last_30_years.csv`

The dataset includes match metadata, teams, venue, toss context, results, and pre-match style features such as ELO values, rolling form metrics, and head-to-head indicators.

## Project Structure

- `data/raw/` raw source files
- `data/processed/` model-ready outputs
- `src/` reusable Python modules
- `notebooks/` exploratory + modeling notebooks
- `docs/` data and design notes
- `models/` serialized model artifacts
- `app.py` Streamlit strategy simulator

## Quickstart

Run all commands from the project root.

1. Create and activate a Python 3.10+ virtual environment.
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run notebook analysis in order:
   - `notebooks/01_data_audit.ipynb`
   - `notebooks/02_model_baseline.ipynb`
   - `notebooks/03_upset_explorer.ipynb`
4. Train and save calibrated artifacts (recommended before simulator):
   - `python scripts/train_and_save.py`
5. Run the simulator:
   - `streamlit run app.py`
6. Run tests:
   - `pytest`
7. Run quality gates locally:
   - `ruff check .`
   - `mypy src scripts app.py`
   - `python scripts/check_conventions.py`
8. Run tests with coverage:
   - `pytest --cov=src --cov=scripts --cov-report=term-missing --cov-fail-under=70`
9. Regenerate data quality report:
   - `python scripts/run_data_quality.py`

## Automation

- CI workflow: `.github/workflows/ci.yml`
  - dependency install
  - lint (`ruff`)
  - type check (`mypy`)
  - conventions check
  - compile check
  - test suite (`pytest`)

## Pre-commit Hooks

- Install once:
  - `pre-commit install`
- Run manually:
  - `pre-commit run --all-files`

## Development Roadmap

1. Data audit and schema validation.
2. Leakage-safe target and upset labeling.
3. Feature engineering for team strength and context effects.
4. Calibrated baseline + boosted models with time-aware splits.
5. Explainability and upset pattern analysis.
6. Scenario simulator improvements.

## Upset Definition (MVP)

For each match, the favorite is the side with stronger pre-match rating proxy (initially ELO). A match is labeled `is_upset = 1` when the underdog wins.

## Code Conventions

See [docs/code_conventions.md](docs/code_conventions.md) for module boundaries, typing, naming, leakage rules, testing, notebook hygiene, and Streamlit patterns.

## Anti-Leakage Principles

- Only pre-match variables are allowed in model features.
- Time-based train/validation/test splits are mandatory.
- Post-match outcome fields (result margins, innings outcomes, etc.) are excluded from prediction features.

