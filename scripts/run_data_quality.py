"""Generate a reproducible data quality report from raw CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate data quality report JSON.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "world_cup_last_30_years.csv",
        help="Path to raw CSV input.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "data_quality_report.json",
        help="Path to output report JSON.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace | None = None) -> Path:
    parsed = args if args is not None else parse_args()

    import pandas as pd

    from src.data_quality import build_data_quality_report, save_data_quality_report

    df = pd.read_csv(parsed.input)
    report = build_data_quality_report(df)
    path = save_data_quality_report(report, parsed.output)

    print(path)
    print(json.dumps(report, indent=2))
    return path


if __name__ == "__main__":
    run()

