"""Offline tests for exchange_reference (paths redirected into tmp_path)."""

import datetime
import os
import sys

import pandas as pd
import pytest

import exchange_reference as xr

TODAY = datetime.date(2026, 11, 20)


def set_mtime(path, d):
    ts = datetime.datetime(d.year, d.month, d.day, 12).timestamp()
    os.utime(path, (ts, ts))


@pytest.fixture
def ref(tmp_path, monkeypatch):
    base = tmp_path / "base"
    (base / "data" / "global_universe").mkdir(parents=True)
    monkeypatch.setattr(xr, "BASE", str(base))
    monkeypatch.setattr(xr, "CURATED", str(base / "data" / "curated"))
    monkeypatch.setattr(xr, "UNIVERSE_GLOB", str(base / "data" / "global_universe" / "*.csv"))
    monkeypatch.setattr(xr, "FUND_DIR", str(base / "fund"))
    (base / "fund").mkdir()
    return base


def uni(ref, market, df, date=TODAY):
    p = ref / "data" / "global_universe" / f"{market}.csv"
    df.to_csv(p, index=False)
    set_mtime(p, date)
    return p


def scan(ref, patt_dir, name, date):
    d = ref / patt_dir
    d.mkdir(exist_ok=True)
    p = d / name
    p.write_text("x")
    set_mtime(p, date)
    return p


def test_mtime_date_and_latest(ref):
    assert xr._mtime_date(None) is None
    assert xr._mtime_date(str(ref / "nope")) is None
    p = ref / "f.txt"
    p.write_text("x")
    set_mtime(p, TODAY)
    assert xr._mtime_date(str(p)) == TODAY
    d = ref / "d"
    d.mkdir()
    a, b = d / "a_1.txt", d / "a_2.txt"
    for f, day in ((a, 1), (b, 2)):
        f.write_text("x")
        set_mtime(f, datetime.date(2026, 1, day))
    assert xr._latest("d/a_*.txt") == str(b)
    assert xr._latest("d/zzz*") is None


def test_fundamentals_status(ref):
    assert xr._fundamentals_status("US") == {}
    p = ref / "fund" / "US.parquet"
    pd.DataFrame(
        {
            "ticker": ["A", "A", "B"],
            "fy_end": ["2020-12-31", "2021-12-31", "2019-06-30"],
        }
    ).to_parquet(p)
    set_mtime(p, TODAY)
    s = xr._fundamentals_status("US")
    assert s == {
        "fund_source": "SEC EDGAR",
        "n_fund_tickers": 2,
        "fund_rows": 3,
        "fund_year_min": 2019,
        "fund_year_max": 2021,
        "fund_collected_on": "2026-11-20",
    }
    # unknown market source -> dash ; unparsable years -> None
    p2 = ref / "fund" / "JP.parquet"
    pd.DataFrame({"ticker": ["X"], "fy_end": ["garbage"]}).to_parquet(p2)
    s = xr._fundamentals_status("JP")
    assert s["fund_source"] == "—" and s["fund_year_min"] is None and s["fund_year_max"] is None
    # empty frame
    p3 = ref / "fund" / "KR.parquet"
    pd.DataFrame({"ticker": [], "fy_end": []}).to_parquet(p3)
    assert xr._fundamentals_status("KR") == {}
    # corrupt file
    (ref / "fund" / "CN.parquet").write_text("junk")
    assert xr._fundamentals_status("CN") == {}


def test_build_reference_empty(ref):
    assert xr.build_reference(TODAY).empty


def test_build_reference_statuses_and_order(ref):
    uni(
        ref,
        "US",
        pd.DataFrame(
            {
                "yf_ticker": ["AAPL", "F", "MSFT", None, "AAPL"],
                "exchange": ["K", "N", "K", "K", "K"],
            }
        ),
    )
    uni(ref, "JP", pd.DataFrame({"code": ["7203.T", "6758.T"]}), date=datetime.date(2026, 11, 15))
    uni(ref, "EU", pd.DataFrame({"yf_ticker": ["ASML.AS"], "exchange": ["Euronext"]}))
    uni(ref, "IN", pd.DataFrame({"yf_ticker": ["TCS"], "exchange": ["NSE"]}))
    scan(
        ref, "us_full_scan", "us_full_scan_1.xlsx", TODAY - datetime.timedelta(days=1)
    )  # age 1 FRESH
    scan(
        ref,
        "european_scan",
        "european_market_scan_broad_1.xlsx",
        TODAY - datetime.timedelta(days=2),
    )  # STALE
    scan(ref, "indian_full_scan", "indian_full_scan_1.xlsx", TODAY)
    pq = ref / "fund" / "US.parquet"
    pd.DataFrame({"ticker": ["AAPL"], "fy_end": ["2020-12-31"]}).to_parquet(pq)
    df = xr.build_reference(TODAY)
    # order: STALE, FRESH(IN, US...), UNIVERSE_ONLY
    assert list(df["status"]) == ["STALE", "FRESH", "FRESH", "FRESH", "UNIVERSE_ONLY"]
    assert list(df["market"]) == ["EU", "IN", "US", "US", "JP"]
    us = df[df["market"] == "US"].set_index("exchange")
    assert set(us.index) == {"NYSE", "Nasdaq"}
    assert us.loc["Nasdaq", "n_tickers"] == 2 and us.loc["NYSE", "n_tickers"] == 1
    assert us.loc["Nasdaq", "tickers"] == ["AAPL", "MSFT"]
    jp = df[df["market"] == "JP"].iloc[0]
    assert jp["exchange"] == "JP" and pd.isna(jp["data_updated"])
    assert jp["age_days"] == 5 and pd.isna(jp["scan_source"])
    assert jp["n_tickers"] == 2 and jp["tickers"] == ["6758.T", "7203.T"]
    eu = df[df["market"] == "EU"].iloc[0]
    assert eu["age_days"] == 2 and eu["data_updated"] == "2026-11-18"
    assert eu["scan_source"].endswith("european_market_scan_broad_1.xlsx")
    assert eu["universe_source"] == os.path.join("data", "global_universe", "EU.csv")
    assert eu["checked"] == "2026-11-20"
    assert pd.isna(eu["n_fund_tickers"])


def test_build_reference_age_boundary(ref):
    uni(ref, "US", pd.DataFrame({"yf_ticker": ["A"], "exchange": ["N"]}))
    p = scan(ref, "us_full_scan", "us_full_scan_1.xlsx", TODAY - datetime.timedelta(days=1))
    assert xr.build_reference(TODAY)["status"].iloc[0] == "FRESH"
    set_mtime(p, TODAY - datetime.timedelta(days=2))
    assert xr.build_reference(TODAY)["status"].iloc[0] == "STALE"
    set_mtime(p, TODAY)
    assert xr.build_reference(TODAY)["age_days"].iloc[0] == 0


def test_build_reference_us_exchange_fix_and_fundamentals(ref):
    uni(
        ref,
        "US",
        pd.DataFrame(
            {
                "yf_ticker": ["A", "B", "C", "D", "E", "F"],
                "exchange": ["N", "A", "P", "Z", "K", "Q"],
            }
        ),
    )
    pq = ref / "fund" / "US.parquet"
    pd.DataFrame({"ticker": ["A", "B"], "fy_end": ["2018-12-31", "2022-12-31"]}).to_parquet(pq)
    df = xr.build_reference(TODAY)
    assert set(df["exchange"]) == {"NYSE", "NYSE American", "NYSE Arca", "Cboe BZX", "Nasdaq", "Q"}
    assert (df["fund_source"] == "SEC EDGAR").all()
    assert (df["fund_year_min"] == 2018).all() and (df["fund_year_max"] == 2022).all()


def test_build_reference_default_today(ref):
    uni(ref, "JP", pd.DataFrame({"x": ["a"]}), date=datetime.date.today())
    df = xr.build_reference()
    assert (
        df["checked"].iloc[0] == datetime.date.today().isoformat() and df["age_days"].iloc[0] == 0
    )


def test_write_parquet(ref):
    out = xr.write_parquet(pd.DataFrame({"a": [1]}))
    assert out == os.path.join(xr.CURATED, "exchange_reference.parquet")
    assert pd.read_parquet(out)["a"].tolist() == [1]


def test_checklist_md(ref):
    df = pd.DataFrame(
        [
            {
                "status": "STALE",
                "market": "EU",
                "exchange": "Euronext",
                "n_tickers": 1500,
                "data_updated": "2026-11-18",
                "age_days": 2,
                "n_fund_tickers": None,
                "fund_year_min": None,
                "fund_year_max": None,
                "fund_source": None,
                "fund_collected_on": None,
            },
            {
                "status": "FRESH",
                "market": "US",
                "exchange": "NYSE",
                "n_tickers": 20,
                "data_updated": "2026-11-20",
                "age_days": 0,
                "n_fund_tickers": 3000.0,
                "fund_year_min": 2010.0,
                "fund_year_max": 2024.0,
                "fund_source": "SEC EDGAR",
                "fund_collected_on": "2026-11-19",
            },
            {
                "status": "UNIVERSE_ONLY",
                "market": "JP",
                "exchange": "JP",
                "n_tickers": 5,
                "data_updated": None,
                "age_days": float("nan"),
                "n_fund_tickers": None,
                "fund_year_min": None,
                "fund_year_max": None,
                "fund_source": None,
                "fund_collected_on": None,
            },
            {
                "status": "NO_DATA",
                "market": "KR",
                "exchange": "KR",
                "n_tickers": 1,
                "data_updated": None,
                "age_days": 3,
                "n_fund_tickers": None,
                "fund_year_min": None,
                "fund_year_max": None,
                "fund_source": None,
                "fund_collected_on": None,
            },
            {
                "status": "WEIRD",
                "market": "ZZ",
                "exchange": "ZZ",
                "n_tickers": 1,
                "data_updated": None,
                "age_days": 3,
                "n_fund_tickers": None,
                "fund_year_min": None,
                "fund_year_max": None,
                "fund_source": None,
                "fund_collected_on": None,
            },
        ]
    )
    df["data_updated"] = df["data_updated"].astype(object).where(df["data_updated"].notna(), None)
    md = xr.checklist_md(df, TODAY)
    lines = md.split("\n")
    assert lines[0] == "# 📋 Daily Data Freshness Checklist — 2026-11-20"
    assert (
        "**5 exchanges** · ✅ 1 fresh · ⚠️ 1 stale · 📋 1 universe-only · 1,527 tickers tracked"
        in md
    )
    assert (
        "**Dated fundamentals:** US 3,000 names (2010–2024, SEC EDGAR, collected 2026-11-19)" in md
    )
    assert "> ⚠️ **STALE — refresh needed:** Euronext" in md
    assert "| ⚠️ STALE | EU | Euronext | 1,500 | 2026-11-18 | — | — | 2 |" in md
    assert "| ✅ FRESH | US | NYSE | 20 | 2026-11-20 | 3,000 | 2010–2024 | 0 |" in md
    assert "| 📋 UNIVERSE_ONLY | JP | JP | 5 | — | — | — |  |" in md
    assert "| ❌ NO_DATA | KR |" in md
    assert "|  WEIRD | ZZ |" in md
    assert md.endswith("|\n")


def test_checklist_md_minimal(ref):
    df = pd.DataFrame(
        [
            {
                "status": "FRESH",
                "market": "US",
                "exchange": "NYSE",
                "n_tickers": 2,
                "data_updated": "d",
                "age_days": 0,
                "n_fund_tickers": None,
                "fund_year_min": None,
                "fund_year_max": None,
                "fund_source": None,
                "fund_collected_on": None,
            }
        ]
    )
    md = xr.checklist_md(df, TODAY)
    assert "Dated fundamentals" not in md and "STALE —" not in md


def test_run(ref, capsys):
    df, pq = xr.run(TODAY)
    assert df.empty and pq is None
    assert "No universe files found" in capsys.readouterr().out
    uni(ref, "JP", pd.DataFrame({"x": ["a", "b"]}))
    df, pq = xr.run(TODAY)
    assert pq == os.path.join(xr.CURATED, "exchange_reference.parquet") and os.path.exists(pq)
    md_path = os.path.join(xr.CURATED, "daily_checklist_2026-11-20.md")
    assert open(md_path).read().startswith("# 📋 Daily Data Freshness Checklist — 2026-11-20")
    out = capsys.readouterr().out
    assert "parquet → " + os.path.join("data", "curated", "exchange_reference.parquet") in out
    assert "| Status" not in out
    os.remove(md_path)
    df, pq = xr.run(TODAY, write=False)
    assert pq is None and not os.path.exists(md_path)
    assert "(not written)" in capsys.readouterr().out
    xr.run()  # default today


def test_main_flags(ref, monkeypatch, capsys):
    uni(ref, "US", pd.DataFrame({"yf_ticker": ["A"], "exchange": ["N"]}))
    scan(
        ref,
        "us_full_scan",
        "us_full_scan_1.xlsx",
        datetime.date.today() - datetime.timedelta(days=5),
    )
    monkeypatch.setattr(sys, "argv", ["x", "--no-write"])
    xr.main()
    assert not os.path.exists(xr.CURATED)
    monkeypatch.setattr(sys, "argv", ["x", "--stale"])
    with pytest.raises(SystemExit) as e:
        xr.main()
    assert e.value.code == 1
    scan(ref, "us_full_scan", "us_full_scan_2.xlsx", datetime.date.today())
    xr.main()  # no stale flag
    monkeypatch.setattr(sys, "argv", ["x", "--stale"])
    xr.main()  # fresh -> no exit
    capsys.readouterr()


def test_main_stale_empty_df_no_exit(ref, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--stale"])
    xr.main()  # empty universe: no SystemExit


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="exchange_reference.py:~178 `r['data_updated'] or '—'` treats a NaN (what pandas>=3 "
    "string columns hold for missing) as truthy and prints 'nan'",
)
def test_checklist_md_nan_data_updated_renders_dash():
    row = {
        "status": "UNIVERSE_ONLY",
        "market": "JP",
        "exchange": "JP",
        "n_tickers": 5,
        "data_updated": float("nan"),
        "age_days": 1.0,
        "n_fund_tickers": None,
        "fund_year_min": None,
        "fund_year_max": None,
        "fund_source": None,
        "fund_collected_on": None,
    }
    md = xr.checklist_md(pd.DataFrame([row]), TODAY)
    assert "| JP | JP | 5 | — |" in md
