"""Lightweight conventions compliance checks."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _check_leakage_features(violations: list[str]) -> None:
    from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

    banned_exact = {
        "winner",
        "result_type",
        "match_result",
        "first_innings_score",
        "second_innings_score",
        "batting_first",
        "chasing_team",
    }
    feature_columns = {c.lower() for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES}
    for col in sorted(feature_columns):
        if col in banned_exact:
            violations.append(f"Leakage feature present in model columns: {col}")
        if col.startswith("innings"):
            violations.append(f"Post-match innings field present in model columns: {col}")


def _check_required_docs(violations: list[str]) -> None:
    required_docs = [
        PROJECT_ROOT / "docs" / "code_conventions.md",
        PROJECT_ROOT / "docs" / "design.md",
        PROJECT_ROOT / "docs" / "implementation_tracker.md",
    ]
    for path in required_docs:
        if not path.exists():
            violations.append(f"Required documentation missing: {path.relative_to(PROJECT_ROOT)}")


def _check_required_tests(violations: list[str]) -> None:
    required_tests = [
        PROJECT_ROOT / "tests" / "test_data_prep.py",
        PROJECT_ROOT / "tests" / "test_features.py",
        PROJECT_ROOT / "tests" / "test_models.py",
        PROJECT_ROOT / "tests" / "test_simulation.py",
    ]
    for path in required_tests:
        if not path.exists():
            violations.append(f"Required test file missing: {path.relative_to(PROJECT_ROOT)}")


def run_checks() -> int:
    violations: list[str] = []
    _check_leakage_features(violations)
    _check_required_docs(violations)
    _check_required_tests(violations)

    if violations:
        print("Conventions check failed:")
        for v in violations:
            print(f"- {v}")
        return 1

    print("Conventions check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_checks())

