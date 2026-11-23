import sys
import types

import pandas as pd
import pytest

from tests.r2g_support import fake_module, loader  # noqa: F401

D1, D2 = pd.Timestamp("2023-03-31 05:00"), pd.Timestamp("2024-03-31")


def stmt(rows):
    return pd.DataFrame(rows, index=[D1, D2]).T


class FakeYF:
    def __init__(self):
        self.data = {}

    def Ticker(self, tk):
        d = self.data.get(tk)
        if isinstance(d, Exception):
            raise d
        d = d or {}
        return types.SimpleNamespace(
            income_stmt=d.get("inc"), balance_sheet=d.get("bal"), cashflow=d.get("cash")
        )


def full_data():
    nan = float("nan")
    return {
        "inc": stmt({"Net Income": [1, 2], "Total Revenue": [10, nan], "Operating Income": [3, 4]}),
        "bal": stmt({"Total Assets": [100, 200], "Share Issued": [5, 6], "Total Liab": [1, 1]}),
        "cash": stmt({"Operating Cash Flow": [7, 8], "Capital Expenditures": [-1, -2]}),
    }


def full_multi():
    one = [1.0, 2.0]
    return {
        "inc": stmt(
            {"Net Income": [1, 2], "Total Revenue": one, "Gross Profit": one, "EBIT": [3, 4]}
        ),
        "bal": stmt(
            {
                "Total Assets": [100, 200],
                "Current Assets": one,
                "Current Liabilities": one,
                "Long Term Debt": one,
                "Share Issued": one,
                "Stockholders Equity": one,
                "Cash And Cash Equivalents": one,
                "Total Liab": [1, 1],
                "Retained Earnings": one,
            }
        ),
        "cash": stmt(
            {"Operating Cash Flow": one, "Capital Expenditures": [-1, -2], "Depreciation": one}
        ),
    }


@pytest.fixture
def mods(loader, tmp_path, monkeypatch):  # noqa: F811
    yf = fake_module("yfinance")
    f = FakeYF()
    yf.Ticker = f.Ticker
    india = loader("india_fundamentals_yf", {"yfinance": yf})
    multi = loader("yf_fundamentals", {"yfinance": yf})
    out = tmp_path / "out"
    monkeypatch.setattr(india, "OUT_DIR", str(out))
    monkeypatch.setattr(india, "OUT", str(out / "IN.parquet"))
    monkeypatch.setattr(india, "UNIVERSE", str(tmp_path / "IN.csv"))
    monkeypatch.setattr(india, "LTM", str(tmp_path / "ltm_IN.parquet"))
    monkeypatch.setattr(india, "BASE", str(tmp_path))
    monkeypatch.setattr(multi, "FUND_DIR", str(out))
    monkeypatch.setattr(multi, "UNI_DIR", str(tmp_path / "uni"))
    monkeypatch.setattr(multi, "LTM_DIR", str(tmp_path / "ltm"))
    (tmp_path / "uni").mkdir()
    (tmp_path / "ltm").mkdir()
    return types.SimpleNamespace(india=india, multi=multi, f=f, tmp=tmp_path, out=out)


@pytest.mark.parametrize("which", ["india", "multi"])
def test_pick(mods, which):
    m = getattr(mods, which)
    df = stmt({"A": [1, float("nan")], "B": [3, 4]})
    assert m._pick(None, ["A"]) == {}
    assert m._pick(pd.DataFrame(), ["A"]) == {}
    assert m._pick(df, ["zz"]) == {}
    assert m._pick(df, ["A"]) == {D1.normalize(): 1.0}  # NaN dropped, date normalised
    assert m._pick(df, ["zz", "B"]) == {D1.normalize(): 3, D2: 4}
    assert m._pick(df, ["B", "A"])[D2] == 4  # first present label wins


def test_india_collect_ticker(mods):
    mods.f.data["REL.NS"] = full_data()
    rows = mods.india.collect_ticker("REL.NS")
    assert [r["fy_end"] for r in rows] == ["2023-03-31", "2024-03-31"]
    r0, r1 = rows
    assert r0["ticker"] == "REL" and r0["net_income"] == 1.0 and r0["total_assets"] == 100.0
    assert r0["shares"] == 5.0 and r0["cfo"] == 7.0
    assert "revenue" not in r1 and r1["net_income"] == 2.0
    assert "ebit" not in r0 and "capex" not in r0  # india schema is smaller
    assert r0["filed"] == "2023-06-29" and r1["filed"] == "2024-06-29"
    assert mods.india.collect_ticker("X.BO") == []
    assert mods.india.collect_ticker("lower.bo") == []


def test_multi_collect_ticker(mods):
    mods.f.data["7203.T"] = full_data()
    mods.f.data["REL.NS"] = full_data()
    r = mods.multi.collect_ticker("7203.T", False)
    assert r[0]["ticker"] == "7203.T" and r[0]["ebit"] == 3.0 and r[0]["capex"] == -1.0
    assert r[0]["total_liabilities"] == 1.0
    assert mods.multi.collect_ticker("REL.NS", True)[0]["ticker"] == "REL"
    assert mods.multi.collect_ticker("REL.NS", False)[0]["ticker"] == "REL.NS"
    assert mods.multi.collect_ticker("zz.l", False) == []


def _uni(m, tickers):
    pd.DataFrame({"yf_ticker": tickers}).to_csv(m.tmp / "IN.csv", index=False)


def test_india_universe_limit(mods, tmp_path):
    _uni(mods, ["A.NS", "B.NS", "C.BO", "A.NS"])
    a = types.SimpleNamespace(limit=0)
    assert mods.india._universe(a) == ["A.NS", "B.NS", "C.BO"]
    a.limit = 2  # LTM missing -> no filtering
    assert mods.india._universe(a) == ["A.NS", "B.NS", "C.BO"]
    pd.DataFrame(
        {
            "Symbol": ["A", "A", "B", "C", "D"],
            "Close": [1, 1, 100, 50, 999],
            "Volume": [1, 1, 10, 10, 1],
        }
    ).to_parquet(mods.india.LTM)
    # dvol means: A=1, B=1000, C=500, D=999 ; top2 = B, D ; universe has B only
    assert mods.india._universe(a) == ["B.NS"]
    a.limit = 3
    assert mods.india._universe(a) == ["B.NS", "C.BO"]


def run_india(mods, monkeypatch, argv=()):
    monkeypatch.setattr(sys, "argv", ["x", *argv])
    mods.india.main()


def test_india_main_fetch_and_resume(mods, monkeypatch, capsys):
    _uni(mods, ["A.NS", "B.NS", "BAD.NS"])
    mods.f.data["A.NS"] = full_data()
    mods.f.data["B.NS"] = {}
    mods.f.data["BAD.NS"] = RuntimeError("x")
    run_india(mods, monkeypatch, ["--workers", "2"])
    out = capsys.readouterr().out
    assert "3 names, 0 done, 3 to fetch (2 threads)" in out
    assert "3/3 | 1 with data | 2 rows" in out and "done → " in out
    df = pd.read_parquet(mods.india.OUT)
    assert set(df["ticker"]) == {"A"} and len(df) == 2
    # resume: A done, only B and BAD retried; data for B now appears
    mods.f.data["B.NS"] = full_data()
    mods.f.data["BAD.NS"] = {}
    run_india(mods, monkeypatch)
    out = capsys.readouterr().out
    assert "3 names, 1 done, 2 to fetch (12 threads)" in out
    assert set(pd.read_parquet(mods.india.OUT)["ticker"]) == {"A", "B"}


def test_india_main_legacy_schema_ignored_and_empty(mods, monkeypatch, capsys):
    _uni(mods, ["A.NS"])
    mods.out.mkdir()
    pd.DataFrame({"ticker": ["A"], "net_income": [1.0]}).to_parquet(mods.india.OUT)
    mods.f.data["A.NS"] = {}  # nothing fetched, nothing to write
    run_india(mods, monkeypatch)
    out = capsys.readouterr().out
    assert "1 names, 0 done, 1 to fetch" in out
    assert "1/1 | 0 with data | 0 rows" in out
    assert list(pd.read_parquet(mods.india.OUT).columns) == ["ticker", "net_income"]  # untouched


def test_india_main_nothing_to_do(mods, monkeypatch, capsys):
    _uni(mods, [])
    run_india(mods, monkeypatch)
    out = capsys.readouterr().out
    assert "0 names, 0 done, 0 to fetch" in out and "done → " in out


def test_india_checkpoint_every_200(mods, monkeypatch, capsys):
    _uni(mods, [f"T{i:03d}.NS" for i in range(201)])
    monkeypatch.setattr(
        mods.india, "collect_ticker", lambda t: [{"ticker": t, "total_assets": 1.0}]
    )
    run_india(mods, monkeypatch, ["--workers", "1"])
    out = capsys.readouterr().out
    assert "200/201 | 200 with data | 200 rows" in out and "201/201 | 201 with data" in out


# ---------------------------------------------------------------- yf_fundamentals
def ns(**k):
    d = dict(market=None, markets=None, all=False, workers=2)
    d.update(k)
    return types.SimpleNamespace(**d)


def test_markets_selection(mods):
    m = mods.multi
    assert m._markets(ns(markets=" jp, de ,uk")) == ["JP", "DE", "UK"]
    assert m._markets(ns(market="jp")) == ["JP"]
    assert m._markets(ns(markets="x", market="y")) == ["X"]
    for n in ("US", "JP", "DE", "IN"):
        pd.DataFrame({"Symbol": ["a"]}).to_parquet(mods.tmp / "ltm" / f"{n}.parquet")
    assert m._markets(ns(all=True)) == ["DE", "IN", "JP"]


def test_collect_market_no_source(mods, capsys):
    mods.multi.collect_market("JP", 1)
    assert "JP: no universe CSV or ltm panel, skipping" in capsys.readouterr().out


def test_collect_market_from_csv_and_resume(mods, capsys):
    m = mods.multi
    pd.DataFrame({"yf_ticker": ["7203.T", "6758.T"]}).to_csv(
        mods.tmp / "uni" / "JP.csv", index=False
    )
    mods.f.data["7203.T"] = full_multi()
    mods.f.data["6758.T"] = RuntimeError("x")
    m.collect_market("JP", 2)
    out = capsys.readouterr().out
    assert "JP: 2 names, 0 done, 2 to fetch" in out and "JP: done — 1 names with data" in out
    df = pd.read_parquet(mods.out / "JP.parquet")
    assert set(df["ticker"]) == {"7203.T"}
    mods.f.data["6758.T"] = full_multi()
    m.collect_market("JP", 2)
    out = capsys.readouterr().out
    assert "2 names, 1 done, 1 to fetch" in out
    assert set(pd.read_parquet(mods.out / "JP.parquet")["ticker"]) == {"7203.T", "6758.T"}


def test_collect_market_ltm_fallback_and_bare_key(mods, capsys):
    m = mods.multi
    pd.DataFrame({"Symbol": ["AAA.NS", "BBB.NS"]}).to_parquet(mods.tmp / "ltm" / "IN.parquet")
    mods.f.data["AAA.NS"] = full_multi()
    mods.f.data["BBB.NS"] = full_multi()
    m.collect_market("IN", 1)
    assert set(pd.read_parquet(mods.out / "IN.parquet")["ticker"]) == {"AAA", "BBB"}
    capsys.readouterr()
    m.collect_market("IN", 1)  # bare key match => both done
    assert "2 names, 2 done, 0 to fetch" in capsys.readouterr().out


def test_collect_market_refetches_when_schema_old(mods, capsys):
    m = mods.multi
    pd.DataFrame({"yf_ticker": ["7203.T"]}).to_csv(mods.tmp / "uni" / "JP.csv", index=False)
    mods.out.mkdir()
    pd.DataFrame({"ticker": ["7203.T"], "total_assets": [1.0]}).to_parquet(mods.out / "JP.parquet")
    mods.f.data["7203.T"] = full_data()
    m.collect_market("JP", 1)
    out = capsys.readouterr().out
    assert "new fields since last collection" in out and "'cash'" in out
    assert "1 names, 0 done, 1 to fetch" in out
    # one field missing is enough to trigger the self-heal
    cols = {c: [1.0] for c in set(m.INCOME) | set(m.BAL) | set(m.CASH)}
    cols["ticker"] = ["7203.T"]
    pd.DataFrame(cols).drop(columns=["capex"]).to_parquet(mods.out / "JP.parquet")
    m.collect_market("JP", 1)
    assert "['capex']" in capsys.readouterr().out
    pd.DataFrame(cols).to_parquet(mods.out / "JP.parquet")
    m.collect_market("JP", 1)
    assert "1 names, 1 done, 0 to fetch" in capsys.readouterr().out


def test_collect_market_checkpoint_300(mods, monkeypatch):
    m = mods.multi
    pd.DataFrame({"yf_ticker": [f"T{i:03d}" for i in range(301)]}).to_csv(
        mods.tmp / "uni" / "JP.csv", index=False
    )
    calls = []
    monkeypatch.setattr(
        m, "collect_ticker", lambda t, b: calls.append(t) or [{"ticker": t, "total_assets": 1.0}]
    )
    writes = []
    orig = pd.DataFrame.to_parquet
    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda self, *a, **k: (writes.append(len(self)), orig(self, *a, **k))[1],
    )
    m.collect_market("JP", 1)
    assert len(writes) == 2 and writes[-1] == 301 and writes[0] >= 300 - 0


def test_multi_main(mods, monkeypatch, capsys):
    got = []
    monkeypatch.setattr(mods.multi, "collect_market", lambda mk, w: got.append((mk, w)))
    monkeypatch.setattr(sys, "argv", ["x", "--markets", "jp,kr", "--workers", "4"])
    mods.multi.main()
    assert got == [("JP", 4), ("KR", 4)]
    out = capsys.readouterr().out
    assert "yfinance fundamentals — 2 markets: JP, KR" in out and out.rstrip().endswith("done")
    got.clear()
    monkeypatch.setattr(sys, "argv", ["x", "--market", "de"])
    mods.multi.main()
    assert got == [("DE", 16)]
