from pathlib import Path

import pandas as pd

from src.data_quality import build_data_quality_report, save_data_quality_report


def test_build_data_quality_report_core_keys(synthetic_matches_df):
    report = build_data_quality_report(synthetic_matches_df)
    assert report["rows"] == len(synthetic_matches_df)
    assert report["columns"] == synthetic_matches_df.shape[1]
    assert report["invalid_dates"] == 0
    assert "missing_by_column" in report


def test_save_data_quality_report_writes_json(tmp_path: Path):
    df = pd.DataFrame({"date": ["2020-01-01"], "match_id": [1]})
    report = build_data_quality_report(df)
    out = save_data_quality_report(report, tmp_path / "dq.json")
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("{")

