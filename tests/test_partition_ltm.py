# Tests for LTM year-partitioning — pure logic, no network.
import pandas as pd

import partition_ltm as pt


def test_build_creates_year_partitions(tmp_path, monkeypatch):
    ltm = tmp_path / "ltm"
    ltm.mkdir()
    parts = tmp_path / "parts"
    monkeypatch.setattr(pt, "LTM_DIR", ltm)
    monkeypatch.setattr(pt, "PARTS_DIR", parts)
    df = pd.DataFrame(
        {
            "Symbol": ["A", "A", "B", "B"],
            "Date": pd.to_datetime(["2024-06-01", "2025-06-01", "2024-06-01", "2025-06-01"]),
            "Open": 1.0,
            "High": 1.0,
            "Low": 1.0,
            "Close": 1.0,
            "Volume": 10,
        }
    )
    df.to_parquet(ltm / "IN.parquet", index=False)

    res = pt.build(verbose=False)
    assert res["partitions"] == 2 and res["rows"] == 4  # 2024 + 2025
    assert (parts / "market=IN" / "year=2024" / "data.parquet").exists()
    assert (parts / "market=IN" / "year=2025" / "data.parquet").exists()
    # row-preserving + no 'year' column leaked into the data
    got = pd.read_parquet(parts / "market=IN" / "year=2024" / "data.parquet")
    assert len(got) == 2 and "year" not in got.columns
    assert pt.available() is True


def test_available_false_when_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "PARTS_DIR", tmp_path / "none")
    assert pt.available() is False


def test_dataset_glob_shape(monkeypatch, tmp_path):
    monkeypatch.setattr(pt, "PARTS_DIR", tmp_path / "p")
    assert pt.dataset_glob().endswith("*.parquet")
