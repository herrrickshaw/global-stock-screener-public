# Tests for the yfinance fundamentals mapping — pure logic, mocked info (no network).
import fundamental_metrics as fm


def test_yf_mapping(monkeypatch):
    fake = {
        "trailingPE": 18.5,
        "returnOnEquity": 0.22,
        "debtToEquity": 45.0,
        "netIncomeToCommon": 5e8,
        "totalRevenue": 3e9,
        "trailingEps": 4.1,
        "marketCap": 9e9,
        "grossMargins": 0.41,
        "earningsGrowth": 0.15,
    }

    class _T:
        def __init__(self, *a, **k):
            self.info = fake

    import sys
    import types

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=_T))
    out = fm._yf_fundamentals("EU", "MT.AS")
    assert out["pe"] == 18.5
    assert round(out["roe"], 0) == 22  # 0.22 → 22%
    assert out["debt_to_equity"] == 0.45  # 45 (pct) → 0.45
    assert round(out["gross_margin"], 0) == 41
    assert round(out["profit_growth"], 0) == 15


def test_yf_empty_info(monkeypatch):
    class _T:
        def __init__(self, *a, **k):
            self.info = {}

    import sys
    import types

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=_T))
    assert fm._yf_fundamentals("EU", "X.PA") == {}
