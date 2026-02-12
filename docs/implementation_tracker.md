# Implementation Tracker: T20 Upset Radar and Strategy Simulator

This is the single source of truth for implementation progress, scope decisions, and next actions.

Status legend:

- `not_started`
- `in_progress`
- `partial`
- `done`
- `blocked`

---

## Project Snapshot

- Last updated: 2026-02-12
- Primary UI: Streamlit (`app.py`) with notebook support
- Data scope (MVP): all available T20 World Cup data in `data/raw/world_cup_last_30_years.csv`
- Current phase focus: MVP completed, presentation polish in progress
- Current phase focus: deep dashboard redesign and insight UX refinement

---

## Section 0 - Project Setup and Planning

### 0.1 Define Project Scope and Success Criteria

- Status: `done`
- Objective: lock scope and MVP outcomes.
- Deliverables:
  - `README.md` problem framing and roadmap
  - `docs/design.md` system design and anti-leakage rules
- Success criteria:
  - Upset labels defined using pre-match strength proxy
  - Reusable `src/` modules used by notebooks and app
  - Simulator returns win probability + upset risk + scenario comparison

### 0.2 Repository and Environment Setup

- Status: `done`
- Deliverables:
  - repo layout (`data/`, `src/`, `notebooks/`, `docs/`, `models/`)
  - `.gitignore`, `requirements.txt`
  - `.venv` setup validated
  - CI workflow scaffold in `.github/workflows/ci.yml`
- Notes:
  - dependency install required trusted hosts in this environment.

---

## Section 1 - Data Understanding and Cleaning

### 1.1 Load and Inspect Dataset

- Status: `done`
- Objective: inspect schema, ranges, and distributions.
- Existing artifacts:
  - `notebooks/01_data_audit.ipynb`
  - `docs/data_dictionary.md`
  - `data/processed/data_quality_report.json`

### 1.2 Data Quality Audit

- Status: `done`
- Existing artifacts:
  - `docs/data_quality.md` with completed checklist and remediation log
  - `data/processed/data_quality_report.json` (quantitative audit output)

### 1.3 Outcome and Key Feature Identification

- Status: `done`
- Existing artifacts:
  - pre-match feature columns defined in `src/features.py`
  - target and upset logic in `src/data_prep.py`
- Checkpoint:
  - keep explicit leakage exclusions documented.

---

## Section 2 - Target Definition, Upset Labeling, and Splits

### 2.1 Define Win Target

- Status: `done`
- Implementation:
  - `build_team1_win_target()` in `src/data_prep.py`
  - filters invalid winner rows before label creation.

### 2.2 Define Favorite vs Underdog and Upset Label

- Status: `done`
- Implementation:
  - `assign_favorite_underdog_from_elo()` in `src/data_prep.py`
  - computes `favorite_team`, `underdog_team`, `is_upset`.

### 2.3 Time-Based Train/Validation/Test Split

- Status: `done`
- Implementation:
  - `time_based_split()` in `src/data_prep.py`
  - split boundaries controlled in `src/config.py`.

---

## Section 3 - Feature Engineering for Upsets and Strategies

### 3.1 Baseline Feature Set Construction

- Status: `done`
- Implementation:
  - `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`
  - `build_pre_match_feature_frame()` in `src/features.py`.

### 3.2 Context and Strategy-Oriented Features

- Status: `partial`
- Implementation:
  - `add_context_interactions()` available in `src/features.py`.
- Remaining work:
  - decide whether to include interaction features in the training pipeline by default.

### 3.3 Feature Engineering Validation

- Status: `partial`
- Existing:
  - basic smoke tests and model run checks completed.
- Remaining:
  - add explicit feature distribution checks and spot-check report in notebook/docs.

---

## Section 4 - Modeling and Upset-Focused Evaluation

### 4.1 Baseline Model (Logistic Regression)

- Status: `done`
- Implementation:
  - `train_logistic_baseline()` in `src/models.py`
  - used in notebook and Streamlit app.

### 4.2 Advanced Model (XGBoost/GBM)

- Status: `partial`
- Implementation:
  - `train_xgboost_baseline()` exists in `src/models.py`.
- Remaining:
  - store comparative evaluation outputs as tracked artifacts in `docs/` or `data/processed/`.

### 4.3 Probability Calibration

- Status: `done`
- Implementation:
  - calibrated classifier support in `src/models.py`
  - calibration integrated in `notebooks/02_model_baseline.ipynb`
  - calibrated artifact path integrated in `app.py`

### 4.4 Focused Upset Evaluation

- Status: `partial`
- Existing:
  - upset labels available and pipeline metrics present.
  - upset precision/recall/confusion matrix added in `notebooks/02_model_baseline.ipynb`.
- Remaining:
  - add narrative analysis for notable false negatives and false positives.

---

## Section 5 - Explainable Upset Explorer

### 5.1 Global Explainability

- Status: `done`
- Existing:
  - visualization helpers in `src/viz.py`.
  - SHAP global feature importance in `notebooks/03_upset_explorer.ipynb` (logistic baseline, fallback if SHAP unavailable).

### 5.2 Local Explanations for Famous Upsets

- Status: `done`
- Existing:
  - `src/explain.py` with `rank_notable_upsets()` and `build_counterfactual_explanation()`
  - local explanation section added in `notebooks/03_upset_explorer.ipynb`
  - matchup volatility utilities in `src/explain.py` for team-pair risk profiling
  - curated upset narratives and cross-case patterns added to `notebooks/99_final_story.ipynb`.

### 5.3 Upset Pattern Visualizations

- Status: `done`
- Existing:
  - `upset_rate_by_bucket()` in `src/viz.py`.
  - dedicated explorer notebook in `notebooks/03_upset_explorer.ipynb` with stage/venue/toss views.

---

## Section 6 - Strategy Simulator Prototype

### 6.1 Simulator Design

- Status: `done`
- Existing:
  - documented in `README.md` and `docs/design.md`
  - implemented input/output structure in `app.py`.

### 6.2 Implement Scenario Feature Builder

- Status: `done`
- Implementation:
  - `ScenarioInput`
  - `build_scenario_features()` in `src/simulation.py`.
  - fallback-source diagnostics in `estimate_scenario_defaults_with_meta()` and confidence mapping via `prior_confidence_label()`.

### 6.3 Implement Basic UI

- Status: `in_progress`
- Implementation:
  - Streamlit dashboard UI in `app.py` with tabbed navigation.
  - matchup-driven numeric defaults for ELO/form/H2H priors.
- Remaining:
  - tune visual hierarchy and spacing for final polish.

### 6.4 Scenario Comparison View

- Status: `done`
- Implementation:
  - side-by-side toss decision comparison in `app.py`.
  - stage-aware upset alert severity with threshold messaging in `app.py`.
  - explicit scenario delta narrative (alternative vs current probabilities).
  - matchup volatility radar (history count, upset rate, volatility index).
  - scenario export actions (CSV + JSON) with timestamp and model metadata.

### 6.8 Deep Dashboard IA and Reviewability

- Status: `in_progress`
- Implementation:
  - Introduced tabs: Simulator, How It Works, Insights, Explainability.
  - Added in-app review checklist and output explanation cards.
  - Added diagnostics and explainability panels for faster human review loops.
- Remaining:
  - complete final visual polish and GIF walkthrough capture.

---

## Section 7 - Final Documentation and Presentation

### 7.1 Polish README and Docs

- Status: `in_progress`
- Existing:
  - README and core docs present.
  - [docs/code_conventions.md](code_conventions.md) for development standards.
  - quality commands and pre-commit usage documented in README.
- Remaining:
  - add final MVP demo GIF from Streamlit (screenshot now added in README).

### 7.2 Final Narrative Notebook

- Status: `in_progress`
- Planned artifact:
  - `notebooks/99_final_story.ipynb` (or similarly named final report notebook).
- Existing:
  - created `notebooks/99_final_story.ipynb` with end-to-end MVP narrative scaffold.
  - added a compact missed-upsets audit table section for high-confidence upset misses.
- Remaining:
  - assemble storyline from data audit -> modeling -> explainability -> simulator.
  - expand with polished charts/tables and final written insights.
  - include concise conclusions and limitations.

### 7.3 Reflection and Future Work

- Status: `done`
- Completed outputs:
  - limitations section in `README.md`
  - future-work section in `README.md`

---

## Cross-Cutting Risks and Controls

- Leakage risk:
  - Never include post-match fields in model features (`winner`, innings stats, result fields).
- Split integrity:
  - Keep strictly chronological splits from `src/config.py`.
- Probability trust:
  - calibration must be part of final app path before presenting probabilities as decision support.
- Reproducibility:
  - persist model artifacts and metadata in `models/` to avoid implicit retraining drift.
- Conventions drift:
  - enforce via `scripts/check_conventions.py`, CI, and pre-commit hooks.

---

## Immediate Next Iteration (High Leverage)

1. Finalize dashboard visual polish (spacing, typography, panel balance).
2. Capture updated dashboard GIF walkthrough after IA redesign.
3. Run one end-to-end human review pass and collect iteration notes.
4. Convert review notes into next sprint Asana subtasks.

---

## MVP Countdown (Can Ship Soon)

### Done now

- End-to-end pipeline exists (`src/`, scripts, tests, CI, conventions).
- Time-aware modeling and calibration are implemented and validated.
- Streamlit simulator supports scenario comparison and stage-aware upset alerts.
- Explainability includes global SHAP flow and local counterfactual tooling.
- Data quality reporting is reproducible (`scripts/run_data_quality.py`).

### Remaining for MVP ship

- MVP functionality complete; remaining items are presentation polish only.
- Capture Streamlit GIF and add to `README.md` (screenshot already added).

### MVP confidence

- Status: `completed`
- Estimated effort to final portfolio polish: `1 focused iteration`

---

## Decision Log

- 2026-02-11: MVP scope set to all World Cup data in source CSV (men + women where present).
- 2026-02-11: Streamlit selected as primary simulator interface with notebook support.
- 2026-02-11: Canonical dataset location standardized to `data/raw/world_cup_last_30_years.csv`.
- 2026-02-12: Added reproducible training script `scripts/train_and_save.py`.
- 2026-02-12: Added CI workflow for compile and test checks.
- 2026-02-12: Added data-driven simulator defaults using historical matchup priors.
- 2026-02-12: Added lint/type/conventions enforcement (`ruff`, `mypy`, `pre-commit`, conventions script).
- 2026-02-12: Completed quantitative data quality audit and documented findings in `docs/data_quality.md`.
- 2026-02-12: Added coverage reporting and threshold enforcement in CI/test config.
- 2026-02-12: Added stage-aware upset alert thresholds and local counterfactual upset explainers.
- 2026-02-12: Added scripted data quality regeneration command (`scripts/run_data_quality.py`) and tests.
- 2026-02-12: Added scenario delta narrative and matchup volatility radar to Streamlit app.
- 2026-02-12: Added `notebooks/99_final_story.ipynb` MVP narrative scaffold.
- 2026-02-12: Added priors source-tier diagnostics and confidence labels for simulator defaults.
- 2026-02-12: Added scenario export support (CSV/JSON) with model metadata and timestamps.
- 2026-02-12: Extended fallback order to include city tier (`matchup_venue -> matchup -> venue -> city -> global`).
- 2026-02-12: Added compact missed-upsets audit section in `notebooks/99_final_story.ipynb`.
- 2026-02-12: Added curated upset narratives and cross-case pattern summaries to `notebooks/99_final_story.ipynb`.
- 2026-02-12: Added README model card, metrics snapshot, and limitations/future work sections.
- 2026-02-12: Marked MVP as functionally complete; remaining work is presentation assets (screenshots/GIF).
- 2026-02-12: Captured and added Streamlit MVP screenshot to `README.md`.
- 2026-02-12: Refactored Streamlit into multi-tab dashboard with insights and explainability pages.

---

## Ideas Backlog

- Add opponent-specific historical priors in scenario builder.
- Add venue fallback tiers (venue -> city -> global priors) for unseen stadiums.
- Add stage-aware thresholding for upset alerts.
- Add lightweight report export for scenario comparisons.

---

## Asana Subtasks: MVP Iteration Queue

### Section 6 - Strategy Simulator Enhancements

- **Task: 6.5 Priors Transparency and Confidence**
  - Subtask: Show fallback source tier (`matchup_venue`, `matchup`, `venue`, `global`) in Streamlit.
  - Subtask: Display confidence label (`high`/`medium`/`low`) next to priors source.
  - Subtask: Extend logic to include `city` tier between `venue` and `global`. (`done`)
  - Subtask: Add tests for source-tier selection order and confidence labels.

- **Task: 6.6 Scenario Export and Auditability**
  - Subtask: Add one-click export of scenario comparison table to CSV. (`done`)
  - Subtask: Add JSON export including full scenario inputs and model outputs. (`done`)
  - Subtask: Include timestamp + model artifact metadata in export. (`done`)
  - Subtask: Add brief export usage note in README. (`done`)

- **Task: 6.7 Upset Watchlist Panel**
  - Subtask: Build a ranking view for high-upset-risk matchups from sampled scenarios.
  - Subtask: Include volatility index and upset risk together in watchlist scoring.
  - Subtask: Add stage filter (group/knockout) and venue filter.
  - Subtask: Add simple interpretation text for top 3 watchlist entries.

- **Task: 6.8 Dashboard Redesign Sprint**
  - Subtask: Refactor app into tabs (`Simulator`, `How It Works`, `Insights`, `Explainability`). (`done`)
  - Subtask: Add diagnostics charts (calibration + upset bucket) to Insights tab. (`done`)
  - Subtask: Add counterfactual/missed-upset/narrative explainability panels. (`done`)
  - Subtask: Add reviewer-facing logic guide and checklist in app. (`done`)
  - Subtask: Final dashboard visual polish pass. (`in_progress`)

- **Task: 6.9 Metric Semantics and Trust**
  - Subtask: Replace ambiguous upset-risk display with explicit `Underdog Win Probability`. (`done`)
  - Subtask: Add `Upset Severity Index` (underdog probability weighted by ELO gap). (`done`)
  - Subtask: Update comparison table to include underdog and severity metrics. (`done`)
  - Subtask: Add tests to lock semantics for new scenario scoring outputs. (`done`)
  - Subtask: Add metric explanation expander in Insights tab (`ROC AUC`, `Log Loss`, `Brier`). (`done`)
  - Subtask: Improve ELO-bucket labels for readability in diagnostics chart/table. (`done`)
  - Subtask: Add predicted upset chance/severity columns in notable historical upset table. (`done`)
  - Subtask: Add explicit upset definition and home-advantage note in How-It-Works tab. (`done`)

- **Task: 6.10 Secondary-Agent Review Fixes**
  - Subtask: Fix curated narrative favorite-team derivation bug. (`done`)
  - Subtask: Add low-sample warning for priors source rows in sidebar. (`done`)
  - Subtask: Add upset precision/recall/F1 and confusion matrix to Insights tab. (`done`)
  - Subtask: Add predicted-vs-observed upset rate by ELO bucket in diagnostics. (`done`)
  - Subtask: Surface model metadata training timestamp in app header. (`done`)

### Section 5 - Explainability Content Completion

- **Task: 5.4 Curated Upset Narratives**
  - Subtask: Select 5-10 notable upsets by ELO gap and context diversity. (`done`)
  - Subtask: Add short narrative for each case (why upset likelihood rose). (`done`)
  - Subtask: Add one counterfactual table per case in notebook. (`done`)
  - Subtask: Summarize cross-case patterns. (`done`)

### Section 7 - Portfolio Polish

- **Task: 7.4 MVP Presentation Assets**
  - Subtask: Capture Streamlit screenshot set and one short GIF walkthrough. (`partial`: screenshot done, GIF pending)
  - Subtask: Add README model card (data window, split policy, calibration, limits). (`done`)
  - Subtask: Add concise results summary table in README. (`done`)
  - Subtask: Final QA pass on docs for consistency with tracker statuses. (`done`)

