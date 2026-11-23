import importlib

import pandas as pd
import pytest

import reward_overlay_mailer as M


def _picks():
    return pd.DataFrame(
        [
            {
                "market": "HK",
                "screen": "low-vol × rising volume",
                "ticker": "0005.HK",
                "liq_6_6": 0.25,
                "ret_6m": 0.1,
                "last_price": 1234.5,
            },
            {
                "market": "HK",
                "screen": "low-vol × rising volume",
                "ticker": "9999.HK",
                "liq_6_6": float("nan"),
                "ret_6m": -0.05,
                "last_price": 3.0,
            },
            {
                "market": "UK",
                "screen": "low-vol × rising volume",
                "ticker": "ULVR.L",
                "liq_6_6": -0.1,
                "ret_6m": 0.0,
                "last_price": 40.0,
            },
            {
                "market": "ZZ",
                "screen": "x",
                "ticker": "Q",
                "liq_6_6": 0.1,
                "ret_6m": 0.1,
                "last_price": 1.0,
            },
        ]
    )


@pytest.fixture
def picks_file(tmp_path, monkeypatch):
    p = tmp_path / "p.csv"
    _picks().to_csv(p, index=False)
    monkeypatch.setattr(M, "PICKS", str(p))


@pytest.fixture
def no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "PICKS", str(tmp_path / "none.csv"))


def test_base_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("REWARD_BASE", str(tmp_path))
    try:
        mod = importlib.reload(M)
        assert mod.BASE == str(tmp_path)
        assert mod.PICKS == str(tmp_path / "reports" / "reward_overlay_picks.csv")
    finally:
        monkeypatch.delenv("REWARD_BASE")
        importlib.reload(M)


def test_load(no_file):
    assert M._load() is None


def test_html_no_file(no_file):
    h = M.section_html()
    assert h.endswith("</div>") and "<table" not in h and "Liquidity-Confirmed" in h


def test_html_empty_file(tmp_path, monkeypatch):
    p = tmp_path / "e.csv"
    _picks().iloc[0:0].to_csv(p, index=False)
    monkeypatch.setattr(M, "PICKS", str(p))
    assert "<table" not in M.section_html()
    assert len(M.section_text().splitlines()) == 3


def test_html_with_picks(picks_file):
    h = M.section_html()
    assert "Hong" not in h and "HSBC" in h and "Unilever" in h
    assert "overlay +8.9pp OOS alpha" in h and "overlay +3.5pp OOS alpha" in h
    assert "HK$1,234.50" in h and "£40.00" in h
    assert "+25%" in h and "-5%" in h and "nan%" in h
    assert "#2e7d32" in h and "#c62828" in h
    assert "ZZ" not in h
    assert h.index("🇭🇰") < h.index("🇬🇧")  # FLAG order
    assert h.endswith("</div>") and "reward_overlay_picks.csv" in h


def test_html_top(picks_file):
    h = M.section_html(top=1)
    assert "HSBC" in h and "9999.HK" not in h


def test_table_zero_return_green():
    rows = _picks().iloc[[2]]
    h = M._table("UK", rows)
    assert "#2e7d32" in h and "#c62828" not in h and "+0%" in h


def test_table_unknown_name_dash_and_ccy():
    rows = pd.DataFrame([{"ticker": "T", "liq_6_6": 0.1, "ret_6m": 0.1, "last_price": 2.0}])
    h = M._table("ZZ", rows)
    assert "—</td>" in h and ">2.00<" in h


def test_text_no_file(no_file):
    assert M.section_text().splitlines()[0].startswith("LIQUIDITY-CONFIRMED")
    assert len(M.section_text().splitlines()) == 3


def test_text_with_picks(picks_file):
    t = M.section_text()
    assert "HK — low-vol × rising volume (overlay +8.9pp OOS)" in t
    assert "0005.HK    HSBC" in t and "Δvol +25%" in t
    assert "UK — " in t and "ZZ" not in t
    assert "9999.HK" in t
    assert "9999.HK" not in M.section_text(top=1)
