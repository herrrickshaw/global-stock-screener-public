import sys

import pandas as pd
import pytest

from tests.r2g_support import fake_module, loader  # noqa: F401

P1, P2 = pd.Timestamp("2024-03-31"), pd.Timestamp("2024-06-30")


def stmt(rows, cols=(P1, P2)):
    return pd.DataFrame(rows, index=list(cols)).T


def ed_index(*dates, tz="Asia/Kolkata"):
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates]).tz_localize(tz)
    return pd.DataFrame({"EPS": [1.0] * len(dates)}, index=idx)


def good(ed=None, cf=None, ed_exc=False, cf_exc=False):
    bs = stmt(
        {
            "Total Assets": [100.0, 110.0],
            "Share Issued": [5.0, 5.0],
            "Retained Earnings": [float("nan"), 9.0],
        }
    )
    inc = stmt({"Total Revenue": [10.0, 12.0], "Net Income": [1.0, 2.0]})
    if cf is None:
        cf = stmt({"Operating Cash Flow": [3.0, 4.0], "Capital Expenditure": [-2.0, -1.0]})
    return {"bs": bs, "inc": inc, "cf": cf, "ed": ed, "ed_exc": ed_exc, "cf_exc": cf_exc}


@pytest.fixture
def pit(loader, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(sys, "path", list(sys.path))
    data = {}
    seen = []

    def Ticker(tk):
        seen.append(tk)
        d = data.get(tk)
        if isinstance(d, Exception):
            raise d
        d = d or {"bs": None, "inc": None}

        class T:
            quarterly_balance_sheet = d["bs"]
            quarterly_income_stmt = d["inc"]

            @property
            def quarterly_cashflow(self_):
                if d.get("cf_exc"):
                    raise RuntimeError("cf")
                return d.get("cf")

            @property
            def earnings_dates(self_):
                if d.get("ed_exc"):
                    raise RuntimeError("ed")
                return d.get("ed")

        return T()

    m = loader(
        "collect_yfinance_pit_fundamentals", {"yfinance": fake_module("yfinance", Ticker=Ticker)}
    )
    monkeypatch.setattr(m, "BASE", str(tmp_path))
    monkeypatch.setattr(m, "OUT_DIR", str(tmp_path / "fh"))
    monkeypatch.setattr(m, "OUT", str(tmp_path / "fh" / "IN_yfinance.parquet"))
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    m.data, m.seen, m.tmp = data, seen, tmp_path
    return m


def test_pick(pit):
    s = stmt({"A": [1.0, float("nan")], "B": [2.0, 3.0]})
    assert pit._pick(s, ["A"], P1) == 1.0
    assert pit._pick(s, ["A"], P2) is None  # NaN
    assert pit._pick(s, ["A", "B"], P2) == 3.0  # falls through NaN to next label
    assert pit._pick(s, ["Z"], P1) is None
    assert isinstance(pit._pick(s, ["B"], P1), float)


def test_fetch_ticker_full_row(pit):
    pit.data["RELIANCE.NS"] = good(ed=(ed_index("2024-04-20", "2024-07-25", "2024-02-01")))
    rows, suffix = pit.fetch_ticker("RELIANCE")
    assert suffix == ".NS" and pit.seen == ["RELIANCE.NS"]
    assert [r["fy_end"] for r in rows] == [P1, P2]
    a, b = rows
    assert a["ticker"] == "RELIANCE" and a["yf_ticker"] == "RELIANCE.NS"
    assert a["total_assets"] == 100.0 and a["shares"] == 5.0 and a["revenue"] == 10.0
    assert a["retained_earnings"] is None and b["retained_earnings"] == 9.0
    assert a["cfo"] == 3.0 and a["capex"] == 2.0 and b["capex"] == 1.0  # abs()
    assert a["d_and_a"] is None
    # nearest announcement on/after period end
    assert a["filed"] == pd.Timestamp("2024-04-20") and b["filed"] == pd.Timestamp("2024-07-25")
    assert a["filed"].tzinfo is None


def test_fetch_ticker_earnings_tolerance_boundary(pit):
    # exactly 120 days after P1 (2024-03-31 + 120d = 2024-07-29) is accepted; 121 is not
    pit.data["A.NS"] = good(ed=(ed_index("2024-07-29")))
    rows, _ = pit.fetch_ticker("A")
    assert rows[0]["filed"] == pd.Timestamp("2024-07-29")
    pit.data["B.NS"] = good(ed=(ed_index("2024-07-30")))
    rows, _ = pit.fetch_ticker("B")
    assert rows[0]["filed"] is None
    # on-the-day announcement counts (>=)
    pit.data["C.NS"] = good(ed=(ed_index("2024-03-31")))
    assert pit.fetch_ticker("C")[0][0]["filed"] == pd.Timestamp("2024-03-31")


def test_fetch_ticker_no_earnings_or_cashflow(pit):
    pit.data["A.NS"] = good(ed=None, cf=pd.DataFrame())
    rows, suffix = pit.fetch_ticker("A")
    assert all(r["filed"] is None for r in rows)
    assert all(r[k] is None for r in rows for k in ("cfo", "capex", "d_and_a"))
    pit.data["B.NS"] = good(ed=pd.DataFrame())
    assert pit.fetch_ticker("B")[0][0]["filed"] is None
    # cashflow lacking a given period column -> None for that period only
    cf = stmt({"Operating Cash Flow": [3.0]}, cols=(P1,))
    pit.data["C.NS"] = good(cf=cf)
    rows, _ = pit.fetch_ticker("C")
    assert rows[0]["cfo"] == 3.0 and rows[1]["cfo"] is None


def test_fetch_ticker_cashflow_and_earnings_exceptions(pit):
    pit.data["A.NS"] = good(cf_exc=True, ed_exc=True)
    rows, suffix = pit.fetch_ticker("A")
    assert suffix == ".NS" and rows[0]["cfo"] is None and rows[0]["filed"] is None


def test_fetch_ticker_falls_back_to_bo(pit):
    pit.data["X.NS"] = {"bs": pd.DataFrame(), "inc": pd.DataFrame()}
    pit.data["X.BO"] = good()
    rows, suffix = pit.fetch_ticker("X")
    assert suffix == ".BO" and pit.seen == ["X.NS", "X.BO"]
    assert rows[0]["yf_ticker"] == "X.BO"


def test_fetch_ticker_skips_bad_and_raises(pit):
    g = good()
    pit.data["A.NS"] = {"bs": g["bs"], "inc": None}  # inc missing
    pit.data["A.BO"] = RuntimeError("boom")  # raising ticker -> continue
    assert pit.fetch_ticker("A") == ([], None)
    pit.data["B.NS"] = {
        "bs": stmt({"Total Assets": [1.0]}, cols=(P1,)),
        "inc": stmt({"Net Income": [1.0]}, cols=(P2,)),
    }
    pit.data["B.BO"] = {"bs": None, "inc": g["inc"]}
    assert pit.fetch_ticker("B") == ([], None)  # no common periods
    assert pit.fetch_ticker("NONE") == ([], None)


def test_fetch_ticker_only_common_periods(pit):
    d = good()
    d["inc"] = stmt({"Total Revenue": [10.0]}, cols=(P2,))
    pit.data["A.NS"] = d
    rows, _ = pit.fetch_ticker("A")
    assert [r["fy_end"] for r in rows] == [P2]


def test_universe_delegates(pit, monkeypatch):
    import collect_nse_results as cn

    monkeypatch.setattr(cn, "_universe", lambda n: ["U", n])
    assert pit._universe(7) == ["U", 7]


def go(pit, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["x", *argv])
    pit.main()


def rec(t):
    return [{"ticker": t, "yf_ticker": t + ".NS", "fy_end": P1, "filed": None}]


def test_main_tickers_retry_and_resume(pit, monkeypatch, capsys):
    calls = []

    def ft(t):
        calls.append(t)
        if t == "BAD":
            return [], None
        if t == "FLAKY" and calls.count("FLAKY") == 1:
            return [], None
        return rec(t), ".NS"

    monkeypatch.setattr(pit, "fetch_ticker", ft)
    go(pit, monkeypatch, "--tickers", "ok", "flaky", "bad")
    out = capsys.readouterr().out
    assert calls == ["OK", "FLAKY", "FLAKY", "BAD", "BAD"]  # one retry each on failure
    assert "[3/3] BAD: no data from yfinance (.NS or .BO), skipping" in out
    assert "3/3 | 2 ok | 1 failed | 2 rows" in out and "(2 rows, 2 tickers)" in out
    assert set(pd.read_parquet(pit.OUT)["ticker"]) == {"OK", "FLAKY"}
    calls.clear()
    go(pit, monkeypatch, "--tickers", "OK", "NEW")
    assert calls == ["NEW"] and "2 already done, 1 to fetch" in capsys.readouterr().out
    assert set(pd.read_parquet(pit.OUT)["ticker"]) == {"OK", "FLAKY", "NEW"}


def test_main_smoke_ignores_existing_and_universe(pit, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(pit, "fetch_ticker", lambda t: calls.append(t) or (rec(t), ".NS"))
    pd.DataFrame({"ticker": ["RELIANCE"]}).pipe(
        lambda d: (
            pit.OUT_DIR and __import__("os").makedirs(pit.OUT_DIR, exist_ok=True),
            d.to_parquet(pit.OUT),
        )[1]
    )
    go(pit, monkeypatch, "--smoke-test")
    assert len(calls) == 20 and calls[0] == "RELIANCE" and calls[-1] == "20MICRONS"
    assert "20 tickers, 0 already done, 20 to fetch" in capsys.readouterr().out
    # default universe path
    calls.clear()
    monkeypatch.setattr(pit, "_universe", lambda n: ["Q", "R"][: n or 2])
    go(pit, monkeypatch, "--limit", "1")
    assert calls == ["Q"] or calls == []  # Q may already be absent -> fetched


def test_main_checkpoint_every_100_and_nothing(pit, monkeypatch, capsys):
    monkeypatch.setattr(pit, "fetch_ticker", lambda t: (rec(t), ".NS"))
    go(pit, monkeypatch, "--tickers", *[f"T{i:03d}" for i in range(101)])
    out = capsys.readouterr().out
    assert "100/101 | 100 ok | 0 failed | 100 rows" in out and "101/101 | 101 ok" in out
    import os

    os.remove(pit.OUT)
    monkeypatch.setattr(pit, "fetch_ticker", lambda t: ([], None))
    go(pit, monkeypatch, "--tickers", "Z")
    assert not os.path.exists(pit.OUT)
    assert "(0 rows, 0 tickers)" in capsys.readouterr().out
