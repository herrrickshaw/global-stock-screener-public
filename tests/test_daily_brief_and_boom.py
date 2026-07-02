# Tests for F-16 (daily brief) and F-10 (sector-boom gating) — pure logic, mocked.
import pandas as pd

import auto_screener as a
import daily_brief as db


def test_brief_renders_regime_table(monkeypatch):
    frame = pd.DataFrame(
        {
            "Symbol": ["A", "B", "C"],
            "LastDate": "2026-06-30",
            "Ret252": [40, 30, 20],
            "Above200DMA": [True, True, True],
            "Liquidity": ["High", "High", "Medium"],
        }
    )
    monkeypatch.setattr(db.sl, "serving", lambda m: frame)
    monkeypatch.setattr(db.datalink, "MARKETS", ["IN"])
    text = db.build_brief()
    assert "Daily Market Brief" in text
    assert "| IN |" in text and "Bull" in text  # 100% breadth + +ret → Bull


def test_brief_handles_no_data(monkeypatch):
    monkeypatch.setattr(db.sl, "serving", lambda m: pd.DataFrame())
    monkeypatch.setattr(db.datalink, "MARKETS", ["IN"])
    assert "No serving views" in db.build_brief()


def test_boom_sector_members(monkeypatch):
    import sector_map as sm

    strong = [
        {"Symbol": s, "Ret252": 40, "Ret63": 10, "Above200DMA": True} for s in ["A", "B", "C"]
    ]
    weak = [
        {"Symbol": s, "Ret252": -25, "Ret63": -8, "Above200DMA": False} for s in ["X", "Y", "Z"]
    ]
    serving = pd.DataFrame(strong + weak)
    smap = {"A": "Tech", "B": "Tech", "C": "Tech", "X": "Metals", "Y": "Metals", "Z": "Metals"}
    # Tech booming, Metals weak
    monkeypatch.setattr(sm, "sector_map", lambda m: smap)
    monkeypatch.setattr(a.sl, "serving", lambda m: serving)
    booms = a.boom_sectors("IN")
    assert "Tech" in booms
    members = a.boom_sector_members("IN")
    assert {"A", "B", "C"} <= members and "X" not in members


def test_boom_empty_without_sector_map(monkeypatch):
    import sector_map as sm

    monkeypatch.setattr(sm, "sector_map", lambda m: {})
    assert a.boom_sectors("IN") == set() and a.boom_sector_members("IN") == set()
