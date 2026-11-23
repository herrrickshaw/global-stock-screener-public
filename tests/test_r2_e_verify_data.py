"""Offline tests for verify_data (free-first price cross-check)."""

import datetime as dt
import math

import pandas as pd
import pytest

import verify_data as vd


def _hist(price, date="2024-05-02"):
    return pd.DataFrame({"Close": [price - 1, price]}, index=pd.to_datetime(["2024-05-01", date]))


# ---- _pct_diff ----
def test_pct_diff():
    assert vd._pct_diff(100, 101) == pytest.approx(1.0)
    assert vd._pct_diff(200, 100) == pytest.approx(-50.0)
    assert math.isnan(vd._pct_diff(0, 5))
    assert math.isnan(vd._pct_diff(float("nan"), 5))
    assert math.isnan(vd._pct_diff(None, 5))


# ---- stooq ----
def test_stooq_ok_mismatch_and_miss(monkeypatch):
    calls = {}

    def fake(syms, min_bars, verbose):
        calls["a"] = (syms, min_bars, verbose)
        return {"AAA": _hist(100.0), "EMPTY": pd.DataFrame()}

    monkeypatch.setattr(vd, "stooq_fetch", fake)
    r = vd._verify_us_via_stooq("AAA", 101.0)
    assert r["status"] == "OK"  # exactly 1% is within tolerance
    assert r["verify_price"] == 100.0 and r["source_price"] == 101.0
    assert r["pct_diff"] == pytest.approx(1.0)
    assert r["verify_source"] == "Stooq (free)" and r["verify_date"] == "2024-05-02"
    assert calls["a"] == (["AAA"], 1, False)
    assert vd._verify_us_via_stooq("AAA", 101.01)["status"] == "MISMATCH"
    assert vd._verify_us_via_stooq("AAA", 99.0)["status"] == "OK"
    assert vd._verify_us_via_stooq("AAA", 98.9)["status"] == "MISMATCH"
    assert vd._verify_us_via_stooq("EMPTY", 1.0) is None
    assert vd._verify_us_via_stooq("ZZZ", 1.0) is None


# ---- databento ----
class FakeStore:
    def __init__(self, df):
        self.df = df

    def to_df(self):
        return self.df


class FakeHist:
    def __init__(self, cost=0.0001, df=None, boom=None, rec=None):
        self.cost, self.df, self.boom, self.rec = cost, df, boom, rec
        outer = self

        class Meta:
            def get_cost(self, **kw):
                outer.rec["cost"] = kw
                if outer.boom == "cost":
                    raise RuntimeError("cost fail")
                return outer.cost

        class TS:
            def get_range(self, **kw):
                outer.rec["range"] = kw
                return FakeStore(outer.df)

        self.metadata = Meta()
        self.timeseries = TS()


class FakeDB:
    def __init__(self, hist):
        self._h = hist

    def Historical(self):
        return self._h


def _bars():
    ts = pd.to_datetime(["2024-05-01 20:00", "2024-05-02 20:00", "2024-05-02 20:00"], utc=True)
    return pd.DataFrame(
        {"close": [10.0, 50.0, 51.0], "volume": [5, 100, 300]},
        index=pd.Index(ts, name="ts_event"),
    )


def test_databento_not_installed(monkeypatch):
    monkeypatch.setattr(vd, "db", None)
    r = vd._verify_us_via_databento("A", 1.0, 1.0)
    assert r == {"symbol": "A", "status": "SKIPPED", "reason": "databento package not installed"}


def test_databento_key_missing(monkeypatch):
    monkeypatch.setattr(vd, "db", object())
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    r = vd._verify_us_via_databento("A", 1.0, 1.0)
    assert r["status"] == "SKIPPED" and r["reason"] == "DATABENTO_API_KEY not set"
    monkeypatch.setenv("DATABENTO_API_KEY", "")
    assert vd._verify_us_via_databento("A", 1.0, 1.0)["status"] == "SKIPPED"


@pytest.fixture
def dbenv(monkeypatch):
    monkeypatch.setenv("DATABENTO_API_KEY", "test-not-a-real-key")


def test_databento_ok_picks_highest_volume_latest_day(monkeypatch, dbenv):
    rec = {}
    monkeypatch.setattr(vd, "db", FakeDB(FakeHist(df=_bars(), rec=rec)))
    r = vd._verify_us_via_databento("AAA", 51.5, 1.0)
    assert r["status"] == "OK"
    assert r["verify_price"] == 51.0  # volume 300 beats 100 on the latest day
    assert r["verify_source"] == "Databento"
    assert r["verify_date"] == "2024-05-02"
    assert r["pct_diff"] == pytest.approx((51.5 - 51.0) / 51.0 * 100)
    # request window: 10 days ending yesterday
    end = dt.date.today() - dt.timedelta(days=1)
    assert rec["range"]["end"] == end.isoformat()
    assert rec["range"]["start"] == (end - dt.timedelta(days=10)).isoformat()
    assert rec["range"]["dataset"] == vd.US_DATASET
    assert rec["range"]["schema"] == "ohlcv-1d" and rec["range"]["symbols"] == ["AAA"]
    assert rec["cost"]["dataset"] == vd.US_DATASET


def test_databento_mismatch(monkeypatch, dbenv):
    monkeypatch.setattr(vd, "db", FakeDB(FakeHist(df=_bars(), rec={})))
    assert vd._verify_us_via_databento("A", 60.0, 1.0)["status"] == "MISMATCH"


def test_databento_cost_gate_boundary(monkeypatch, dbenv):
    rec = {}
    monkeypatch.setattr(vd, "db", FakeDB(FakeHist(cost=1.0, df=_bars(), rec=rec)))
    assert vd._verify_us_via_databento("A", 51.0, 1.0)["status"] == "OK"  # == threshold allowed
    assert "range" in rec
    rec.clear()
    monkeypatch.setattr(vd, "db", FakeDB(FakeHist(cost=1.5, df=_bars(), rec=rec)))
    r = vd._verify_us_via_databento("A", 51.0, 1.0)
    assert r["status"] == "SKIPPED"
    assert r["reason"] == "estimated cost $1.5000 over $1.00 threshold"
    assert "range" not in rec  # never fetched


def test_databento_no_data_and_error(monkeypatch, dbenv):
    monkeypatch.setattr(vd, "db", FakeDB(FakeHist(df=pd.DataFrame(), rec={})))
    assert vd._verify_us_via_databento("A", 1.0, 1.0) == {"symbol": "A", "status": "NO_DATA"}
    monkeypatch.setattr(vd, "db", FakeDB(FakeHist(boom="cost", rec={})))
    r = vd._verify_us_via_databento("A", 1.0, 1.0)
    assert r == {"symbol": "A", "status": "ERROR", "reason": "cost fail"}


# ---- verify_us ----
def test_verify_us_free_first(monkeypatch):
    def stooq(syms, min_bars, verbose):
        return {s: _hist(100.0) for s in syms if s in ("OK", "BAD")}

    called = []

    def dbento(sym, price, thr):
        called.append((sym, price, thr))
        return {"symbol": sym, "status": "OK", "verify_price": 77.0}

    monkeypatch.setattr(vd, "stooq_fetch", stooq)
    monkeypatch.setattr(vd, "_verify_us_via_databento", dbento)
    res = vd.verify_us(
        [
            {"symbol": "OK", "price": 100.0},
            {"symbol": "BAD", "price": 110.0},
            {"symbol": "NONE", "price": 5.0},
        ],
        cost_threshold_usd=0.5,
    )
    assert [r["symbol"] for r in res] == ["OK", "BAD", "NONE"]
    assert res[0]["verify_source"] == "Stooq (free)" and "escalation_reason" not in res[0]
    assert called == [("BAD", 110.0, 0.5), ("NONE", 5.0, 0.5)]
    assert res[1]["escalation_reason"] == (
        "Stooq itself showed a +10.00% discrepancy -- escalating for confirmation"
    )
    assert res[1]["stooq_price"] == 100.0
    assert res[2]["escalation_reason"] == "Stooq had no data for this symbol"
    assert "stooq_price" not in res[2]


def test_verify_us_default_threshold_and_empty(monkeypatch):
    monkeypatch.setattr(vd, "stooq_fetch", lambda *a, **k: {})
    seen = []
    monkeypatch.setattr(
        vd, "_verify_us_via_databento", lambda s, p, t: seen.append(t) or {"symbol": s}
    )
    assert vd.verify_us([]) == []
    vd.verify_us([{"symbol": "A", "price": 1.0}])
    assert seen == [1.0]


# ---- verify_india ----
def test_verify_india(monkeypatch):
    frames = {"GOOD": _hist(200.0), "OFF": _hist(200.0), "EMPTY": pd.DataFrame()}
    asked = []

    def gs(sym, n_days):
        asked.append((sym, n_days))
        return frames.get(sym)

    monkeypatch.setattr(vd.bh, "get_symbol", gs)
    res = vd.verify_india(
        [
            {"symbol": "GOOD", "price": 202.0},
            {"symbol": "OFF", "price": 205.0},
            {"symbol": "EMPTY", "price": 1.0},
            {"symbol": "NONE", "price": 1.0},
        ]
    )
    assert asked[0] == ("GOOD", 10)
    assert res[0]["status"] == "OK" and res[0]["verify_source"] == "NSE Bhavcopy"
    assert res[0]["verify_date"] == "2024-05-02" and res[0]["verify_price"] == 200.0
    assert res[1]["status"] == "MISMATCH" and res[1]["pct_diff"] == pytest.approx(2.5)
    assert res[2] == {"symbol": "EMPTY", "status": "NO_DATA"}
    assert res[3] == {"symbol": "NONE", "status": "NO_DATA"}
    assert vd.verify_india([]) == []


# ---- html / text ----
def _res(sym, status, **kw):
    base = {
        "symbol": sym,
        "status": status,
        "source_price": 10.0,
        "verify_price": 12.0,
        "pct_diff": -16.67,
        "verify_source": "Databento",
        "verify_date": "2024-05-02",
    }
    base.update(kw)
    return base


def test_section_html_empty():
    assert vd._section_html("L", []) == (
        "<p style='font-size:12px;color:#666'>L: no picks to verify today.</p>"
    )


def test_section_html_all_branches():
    res = [
        _res("A", "OK"),
        _res("B", "MISMATCH"),
        _res("C", "SKIPPED"),
        _res("D", "NO_DATA", escalation_reason="r"),
        _res("E", "ERROR"),
        _res("F", "MISMATCH", verify_source=None),
    ]
    res[5].pop("verify_source")
    h = vd._section_html("L", res)
    assert "<b>L</b> — 1/6 confirmed" in h
    assert "2 discrepancy(ies)" in h
    assert ", 3 skipped" in h
    assert "(1 escalated to a paid source)" in h
    assert "⚠ B: report=10.00 vs Databento=12.00 (-16.67%, as of 2024-05-02)" in h
    assert "⚠ F: report=10.00 vs ?=12.00" in h


def test_section_html_clean():
    h = vd._section_html("L", [_res("A", "OK")])
    assert h.count("<p") == 1
    assert "discrepancy" not in h and "skipped" not in h and "escalated" not in h
    assert "1/1 confirmed" in h


def test_verification_html_composes():
    h = vd.verification_html([_res("A", "OK")], [])
    assert h.startswith("<h3")
    assert "Data Verification" in h
    assert "🇺🇸 US</b> — 1/1 confirmed" in h
    assert "🇮🇳 India: no picks to verify today." in h
    assert h.index("🇺🇸") < h.index("🇮🇳")


def test_verification_text():
    us = [_res("A", "OK", escalation_reason="x"), _res("B", "MISMATCH")]
    india = []
    t = vd.verification_text(us, india)
    lines = t.split("\n")
    assert lines[0] == "Data Verification (yfinance vs independent source, free-first):"
    assert lines[1] == "  US: 1/2 confirmed (1 escalated to a paid source)"
    assert lines[2] == "    ! B: report=10.00 vs Databento=12.00 (-16.67%)"
    assert lines[3] == "  India: no picks to verify today."
    assert len(lines) == 4
    t2 = vd.verification_text([_res("A", "OK")], [_res("Z", "MISMATCH", verify_source=None)])
    assert "  US: 1/1 confirmed\n" in t2 + "\n"
    assert "(0 escalated" not in t2
    assert "vs None=12.00" in t2
