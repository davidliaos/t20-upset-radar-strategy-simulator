"""Reusable data quality audit helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def build_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Build a JSON-serializable data quality report."""
    report: Dict[str, Any] = {}
    report["rows"] = int(len(df))
    report["columns"] = int(df.shape[1])

    if "date" in df.columns:
        parsed_dates = pd.to_datetime(df["date"], errors="coerce")
        report["date_min"] = str(parsed_dates.min())
        report["date_max"] = str(parsed_dates.max())
        report["invalid_dates"] = int(parsed_dates.isna().sum())
    else:
        report["date_min"] = None
        report["date_max"] = None
        report["invalid_dates"] = None

    report["duplicate_match_id"] = int(df["match_id"].duplicated().sum()) if "match_id" in df.columns else None

    report["missing_by_column"] = {
        col: int(count)
        for col, count in df.isna().sum().sort_values(ascending=False).items()
        if int(count) > 0
    }

    if {"team1", "team2"}.issubset(df.columns):
        teams = pd.concat([df["team1"], df["team2"]], axis=0).dropna().astype(str)
        teams_norm = teams.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
        report["unique_teams_raw"] = int(teams.nunique())
        report["unique_teams_normalized"] = int(teams_norm.nunique())
        report["team_name_variants_suspected"] = int(report["unique_teams_raw"] - report["unique_teams_normalized"])

    if "venue" in df.columns:
        venues = df["venue"].dropna().astype(str)
        venues_norm = venues.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
        report["unique_venues_raw"] = int(venues.nunique())
        report["unique_venues_normalized"] = int(venues_norm.nunique())
        report["venue_name_variants_suspected"] = int(report["unique_venues_raw"] - report["unique_venues_normalized"])

    if "match_result" in df.columns:
        report["match_result_distribution"] = {
            str(k): int(v) for k, v in df["match_result"].fillna("missing").value_counts().to_dict().items()
        }

    if "is_worldcup" in df.columns:
        report["is_worldcup_distribution"] = {
            str(k): int(v) for k, v in df["is_worldcup"].fillna("missing").value_counts().to_dict().items()
        }

    return report


def save_data_quality_report(report: Dict[str, Any], output_path: Path) -> Path:
    """Write report JSON to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path

