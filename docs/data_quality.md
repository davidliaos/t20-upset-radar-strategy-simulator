# Data Quality Audit

This file captures quality checks and remediation notes for the source data.

## Checklist

- [x] Row count and date range verification
- [x] Missing value profile by column
- [x] Duplicate `match_id` check
- [x] Team name standardization review
- [x] Venue and stage normalization review
- [x] Chronological ordering checks

## Latest Audit Snapshot

Source: `data/raw/world_cup_last_30_years.csv`  
Generated report: `data/processed/data_quality_report.json`
Regeneration command: `python scripts/run_data_quality.py`
Semantics note: totals below describe the raw audit dataset; model training defaults to completed matches only (`3761` rows).

- Rows: `3865`
- Columns: `38`
- Date range: `2014-03-16` to `2026-02-09`
- Invalid dates: `0`
- Duplicate `match_id`: `0`

## Missingness Summary (non-zero columns)

| Column | Missing Count | Share |
|---|---:|---:|
| `match_stage` | 3499 | 90.53% |
| `winner` | 104 | 2.69% |
| `innings2_team` | 62 | 1.60% |
| `innings2_overs` | 62 | 1.60% |
| `innings2_wkts` | 62 | 1.60% |
| `innings2_runs` | 62 | 1.60% |
| `city` | 32 | 0.83% |

## Consistency Checks

- Team names:
  - unique raw: `106`
  - unique normalized (strip/lower/single-space): `106`
  - suspected team spelling variants: `0`
- Venue names:
  - unique raw: `321`
  - unique normalized: `321`
  - suspected venue spelling variants: `0`

## Match Result Distribution

- `completed`: `3761`
- `no result`: `77`
- `tie`: `27`

## World Cup Flag Distribution

- `False`: `2450`
- `True`: `1415`

## Known Initial Observations

- Source CSV is present at `data/raw/world_cup_last_30_years.csv`.
- Dataset includes both outcome fields and pre-match feature proxies.
- Leakage review is enforced through `scripts/check_conventions.py` and feature-set checks.

## Planned Checks

1. **Missingness**  
   Compute null counts and percentages for all columns.

2. **Uniqueness and duplicates**  
   Validate uniqueness assumptions for `match_id`.

3. **Value consistency**  
   Check for spelling variants in team and venue fields.

4. **Temporal sanity**  
   Parse dates and inspect for invalid values or ordering anomalies.

## Remediation Log

| Date | Issue | Action | Owner |
|---|---|---|---|
| 2026-02-11 | High nulls in `match_stage` (~90%) | Use safe fallback in simulator and treat stage as optional in analysis; do not drop rows globally | Agent |
| 2026-02-11 | Missing `winner` for non-completed matches | Filter to `match_result == completed` in `load_matches()` for training labels | Agent |
| 2026-02-11 | Non-completed outcomes (`no result`, `tie`) present | Keep in raw audit stats; exclude from supervised training target construction | Agent |

