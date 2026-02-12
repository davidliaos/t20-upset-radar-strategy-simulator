# Data Dictionary (Draft)

This document tracks column meanings for `data/raw/world_cup_last_30_years.csv`.

## Core Match Metadata

| Column | Type (expected) | Description | Pre-match safe |
|---|---|---|---|
| `match_id` | int/string | Unique match identifier | yes |
| `date` | date | Match date | yes |
| `season` | int/string | Season label | yes |
| `tournament_name` | string | Tournament label | yes |
| `is_worldcup` | bool | World Cup flag | yes |
| `match_stage` | string | Stage (group/knockout/etc.) | yes |
| `format` | string | Match format (T20) | yes |

## Teams, Venue, Toss

| Column | Type (expected) | Description | Pre-match safe |
|---|---|---|---|
| `team1` | string | Team 1 name | yes |
| `team2` | string | Team 2 name | yes |
| `venue` | string | Ground/stadium | yes |
| `city` | string | Host city | yes |
| `toss_winner` | string | Toss winner team | yes |
| `toss_decision` | string | Toss choice (bat/field) | yes |

## Outcome and Innings Fields

| Column | Type (expected) | Description | Pre-match safe |
|---|---|---|---|
| `winner` | string | Match winner | no |
| `result_type` | string | Win type | no |
| `match_result` | string | Result status | no |
| `innings1_runs` | numeric | First innings runs | no |
| `innings2_runs` | numeric | Second innings runs | no |
| `first_innings_score` | numeric | First innings score | no |
| `second_innings_score` | numeric | Second innings score | no |

## Existing Derived Feature Columns

| Column | Type (expected) | Description | Pre-match safe |
|---|---|---|---|
| `elo_team1` | numeric | Team1 pre-match ELO proxy | yes |
| `elo_team2` | numeric | Team2 pre-match ELO proxy | yes |
| `elo_diff` | numeric | `elo_team1 - elo_team2` | yes |
| `team1_form_5` | numeric | Team1 rolling form (5 matches) | yes |
| `team2_form_5` | numeric | Team2 rolling form (5 matches) | yes |
| `team1_form_10` | numeric | Team1 rolling form (10 matches) | yes |
| `team2_form_10` | numeric | Team2 rolling form (10 matches) | yes |
| `h2h_win_pct` | numeric | Team1 historical head-to-head win pct | yes |

## Notes

- This is an initial draft and should be updated after detailed audit checks.
- Any column used for model training must be verified as known before ball one.

