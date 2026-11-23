import pandas as pd
import pytest

import reward_picks_mailer as M


def _picks():
    return pd.DataFrame(
        [
            {"market": "US", "ticker": "NVDA", "ret_6m": 0.25, "last_price": 1234.5},
            {"market": "US", "ticker": "ZZZZ", "ret_6m": -0.1, "last_price": 3.0},
            {"market": "US", "ticker": "AAPL", "ret_6m": float("nan"), "last_price": 9.0},
            {"market": "CA", "ticker": "TD.TO", "ret_6m": 0.0, "last_price": 80.0},
            {"market": "XX", "ticker": "Q", "ret_6m": 0.1, "last_price": 1.0},
        ]
    )


@pytest.fixture
def picks_file(tmp_path, monkeypatch):
    p = tmp_path / "picks.csv"
    _picks().to_csv(p, index=False)
    monkeypatch.setattr(M, "PICKS", str(p))
    return p


@pytest.fixture
def no_picks(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "PICKS", str(tmp_path / "missing.csv"))


def test_load_missing_is_none(no_picks):
    assert M._load() is None


def test_load_reads(picks_file):
    assert len(M._load()) == 5


def test_explain_html_mentions_oos():
    h = M._explain_html()
    assert "out-of-sample" in h and "<b>What is an out-of-sample screener?</b>" in h


def test_section_html_degrades_without_csv(no_picks):
    h = M.section_html()
    assert h.endswith("</div>") and "Reward-Optimised" in h
    assert "<table" not in h


def test_section_html_with_picks(picks_file):
    h = M.section_html()
    assert "NVIDIA" in h and "TD Bank" in h
    assert "+25%" in h and "-10%" in h
    assert "$1,234.50" in h and "C$80.00" in h
    # CN/IN/TW have no rows -> no header; unknown market XX not in STRATEGY
    assert "TW —" not in h and "XX" not in h
    assert "Why here:" in h
    assert "#2e7d32" in h and "#c62828" in h  # green and red
    assert "—</td>" in h  # unknown name placeholder


def test_section_html_top_limits(picks_file):
    h = M.section_html(top=1)
    assert "NVDA" in h and "ZZZZ" not in h


def test_table_zero_return_is_green_nan_is_red():
    rows = pd.DataFrame(
        [
            {"ticker": "A", "ret_6m": 0.0, "last_price": 1.0},
            {"ticker": "B", "ret_6m": float("nan"), "last_price": 1.0},
        ]
    )
    h = M._table("ZZ", rows)
    first, second = h.split("<tr>")[2:4]
    assert "#2e7d32" in first
    assert "#c62828" not in first
    # nan >= 0 is False -> red
    assert "#c62828" in second
    assert "+0%" in first and "nan" in second.lower()


def test_table_unknown_market_no_currency():
    rows = pd.DataFrame([{"ticker": "A", "ret_6m": 0.5, "last_price": 2.0}])
    assert ">2.00<" in M._table("ZZ", rows)


def test_section_text_without_csv(no_picks):
    t = M.section_text()
    assert t.startswith("REWARD-OPTIMISED SCREENERS")
    assert len(t.splitlines()) == 3


def test_section_text_with_picks(picks_file):
    t = M.section_text()
    assert "US — cash-flow quality  (out-of-sample alpha +4.5%/yr · 67% win)" in t
    assert "NVDA         NVIDIA" in t and "6mo +25%" in t
    assert "CA — 12-month momentum" in t
    assert "TW —" not in t


def test_section_text_top(picks_file):
    t = M.section_text(top=1)
    assert "NVDA" in t and "ZZZZ" not in t


def test_strategy_order_and_keys():
    assert list(M.STRATEGY) == ["US", "CA", "TW", "IN", "CN"]
    assert set(M.FLAG) == set(M.CCY) == set(M.STRATEGY)
