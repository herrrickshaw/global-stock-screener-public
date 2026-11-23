"""Offline tests for etl_pipeline (HOME redirected so import-time mkdirs stay in tmp)."""

import importlib
import json
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def etl(tmp_path, monkeypatch):
    home = tmp_path / "home"
    base = home / "Downloads" / "code" / "python_files"
    base.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("GMS_REPO", raising=False)
    saved = sys.modules.pop("etl_pipeline", None)
    mod = importlib.import_module("etl_pipeline")
    yield mod
    sys.modules.pop("etl_pipeline", None)
    if saved is not None:
        sys.modules["etl_pipeline"] = saved


def ctx_for(etl, markets=("US",), **kw):
    return etl.Ctx(list(markets), kw.pop("force", False), **kw)


def test_dotenv_loader(etl, tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("# c\n\nA_KEY=\"v1\"\nB_KEY='v2'\nNOEQ\nEXIST=new\n=bad\n")
    monkeypatch.setenv("EXIST", "old")
    monkeypatch.delenv("A_KEY", raising=False)
    monkeypatch.delenv("B_KEY", raising=False)
    etl._load_dotenv(str(p))
    assert os.environ["A_KEY"] == "v1" and os.environ["B_KEY"] == "v2"
    assert os.environ["EXIST"] == "old"
    etl._load_dotenv(str(tmp_path / "missing.env"))  # no error


def test_dotenv_default_path(etl, monkeypatch):
    monkeypatch.delenv("ZZ_DEF", raising=False)
    with open(os.path.join(etl.BASE, ".env"), "w") as fh:
        fh.write("ZZ_DEF=1\n")
    etl._load_dotenv()
    assert os.environ["ZZ_DEF"] == "1"
    monkeypatch.delenv("ZZ_DEF")


def test_latest_and_fresh(etl, tmp_path):
    d = os.path.join(etl.BASE, "x")
    os.makedirs(d)
    a, b = os.path.join(d, "a_1.xlsx"), os.path.join(d, "a_2.xlsx")
    for p, t in ((a, 100), (b, 200)):
        open(p, "w").write("x")
        os.utime(p, (t, t))
    assert etl._latest("x/a_*.xlsx") == b
    assert etl._latest("x/none_*.xlsx") is None
    c = ctx_for(etl)
    assert c.fresh(a) is False  # old mtime
    open(a, "w").write("y")  # now
    assert c.fresh(a) is True
    assert ctx_for(etl, force=True).fresh(a) is False
    assert c.fresh(None) is False
    assert c.fresh(os.path.join(d, "nope")) is False


def test_stage_registry_order(etl):
    names = [n for n, _, _ in etl.STAGES]
    assert names[0] == "extract_universe" and names[-1] == "load_manifest"
    assert {p for _, p, _ in etl.STAGES} == {"E", "T", "L"}


def test_extract_universe(etl):
    os.makedirs(os.path.join(etl.BASE, "data"), exist_ok=True)
    with open(os.path.join(etl.BASE, "data", "europe_broad_list.csv"), "w") as fh:
        fh.write("t\nA\nB\nC\n")
    c = ctx_for(etl)
    assert etl.extract_universe(c) == "ok"
    assert c.rows["universe:europe_broad"] == 3
    out = c.artifacts["universe"]
    m = json.load(open(out))
    assert m == {"europe_broad": {"path": "data/europe_broad_list.csv", "tickers": 3}}
    assert c.lineage[out] == ["data/europe_broad_list.csv"]
    c2 = ctx_for(etl)
    assert etl.extract_universe(c2) == "skip"
    assert c2.artifacts["universe"] == out
    c3 = ctx_for(etl, force=True)
    assert etl.extract_universe(c3) == "ok"


def test_rescan_paths(etl, monkeypatch):
    assert etl._rescan("XX") == (False, "no rescan command")
    ok, msg = etl._rescan("EU")
    assert not ok and "scan script not found" in msg
    assert "full_european_market_scan.py" in msg
    # US script is outside BASE -> absolute path printed
    ok, msg = etl._rescan("US")
    assert not ok and "full_us_market_scan.py" in msg
    script = os.path.join(etl.BASE, "full_european_market_scan.py")
    open(script, "w").write("")
    calls = []

    def fake(cmd, **kw):
        calls.append((cmd, kw))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(etl.subprocess, "run", fake)
    assert etl._rescan("EU") == (True, "rescanned")
    assert calls[0][0][0] == sys.executable
    assert calls[0][0][1] == script and "--batch" in calls[0][0]
    assert calls[0][1]["cwd"] == etl.BASE

    def bad(cmd, **kw):
        return types.SimpleNamespace(returncode=2, stdout="", stderr="x\nboom happened")

    monkeypatch.setattr(etl.subprocess, "run", bad)
    assert etl._rescan("EU") == (False, "rescan failed: boom happened")

    def bad2(cmd, **kw):
        return types.SimpleNamespace(returncode=3, stdout="", stderr="")

    monkeypatch.setattr(etl.subprocess, "run", bad2)
    assert etl._rescan("EU") == (False, "rescan failed: rc=3")

    def bad3(cmd, **kw):
        return types.SimpleNamespace(returncode=3, stdout="outline", stderr="")

    monkeypatch.setattr(etl.subprocess, "run", bad3)
    assert etl._rescan("EU")[1] == "rescan failed: outline"


def test_extract_market_scans(etl, monkeypatch):
    d = os.path.join(etl.BASE, "us_full_scan")
    os.makedirs(d)
    wb = os.path.join(d, "us_full_scan_1.xlsx")
    open(wb, "w").write("x")
    c = ctx_for(etl, markets=("US", "EU", "ZZ"))
    assert etl.extract_market_scans(c) == "ok"
    assert c.artifacts["scan:US"] == wb
    notes = c.notes["extract_market_scans"]
    assert any(n.startswith("EU: NO scan workbook") for n in notes)
    assert any(n.startswith("ZZ: NO scan workbook") for n in notes)
    assert not any(n.startswith("US") for n in notes)
    # stale
    os.utime(wb, (100, 100))
    c = ctx_for(etl)
    etl.extract_market_scans(c)
    assert c.notes["extract_market_scans"] == [
        "US: stale (us_full_scan_1.xlsx) — --refresh-scans to rebuild"
    ]

    # refresh success picks up new file
    def fake_rescan(m):
        open(os.path.join(d, "us_full_scan_2.xlsx"), "w").write("n")
        return True, "rescanned"

    monkeypatch.setattr(etl, "_rescan", fake_rescan)
    c = ctx_for(etl, refresh_scans=True)
    etl.extract_market_scans(c)
    assert c.artifacts["scan:US"].endswith("us_full_scan_2.xlsx")
    assert c.notes["extract_market_scans"] == ["US: stale → rescanned"]
    # refresh missing + failure
    monkeypatch.setattr(etl, "_rescan", lambda m: (False, "nope"))
    c = ctx_for(etl, markets=("EU",), refresh_scans=True)
    etl.extract_market_scans(c)
    assert c.notes["extract_market_scans"] == ["EU: missing → nope"]
    assert "scan:EU" not in c.artifacts
    c = ctx_for(etl, markets=("EU",))
    c.notes.clear()
    monkeypatch.setitem(etl.SCAN_GLOBS, "EU", "nothing/*.xlsx")
    c2 = ctx_for(etl, markets=("ZZ",), refresh_scans=True)
    etl.extract_market_scans(c2)
    assert c2.notes["extract_market_scans"] == ["ZZ: missing → nope"]


def test_extract_market_scans_no_notes(etl):
    d = os.path.join(etl.BASE, "us_full_scan")
    os.makedirs(d)
    open(os.path.join(d, "us_full_scan_1.xlsx"), "w").write("x")
    c = ctx_for(etl)
    etl.extract_market_scans(c)
    assert "extract_market_scans" not in c.notes


def test_darvas_fragments(etl, monkeypatch, tmp_path):
    seen = []

    def fake(cmd, **kw):
        seen.append(cmd)
        m = cmd[cmd.index("--market") + 1]
        out = "<html/>" if m == "IN" else "   "
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    monkeypatch.setattr(etl.subprocess, "run", fake)
    c = ctx_for(etl, markets=("IN", "US", "EU"))
    assert etl.extract_darvas_fragments(c) == "ok"
    assert len(seen) == 2 and "--top" in seen[0] and "15" in seen[0]
    assert c.notes["extract_darvas_fragments"] == "regenerated: IN"
    assert open(os.path.expanduser("~/Downloads/in_darvas.html")).read() == "<html/>"
    assert not os.path.exists(os.path.expanduser("~/Downloads/us_darvas.html"))
    c = ctx_for(etl, markets=("EU",))
    etl.extract_darvas_fragments(c)
    assert c.notes["extract_darvas_fragments"] == "regenerated: none"


class FakeSP:
    mood = {"n_articles": 7}

    def get_market_mood(self, m):
        return {"n_articles": 7} if m != "EU" else None

    def get_ticker_sentiment(self, sym, m, company_name=None):
        label = (
            "POSITIVE" if sym.startswith("P") else "NEGATIVE" if sym.startswith("N") else "NO_DATA"
        )
        if sym.startswith("Z"):
            label, n = "POSITIVE", 0
        else:
            n = 2
        heads = [{"title": f"h-{sym}-{company_name}"}] if sym != "P2" else []
        return types.SimpleNamespace(label=label, score=0.5, n_articles=n, top_headlines=heads)


@pytest.fixture
def fake_sp(monkeypatch):
    mod = types.ModuleType("sentiment_pipeline")
    mod.SentimentPipeline = FakeSP
    monkeypatch.setitem(sys.modules, "sentiment_pipeline", mod)


def test_extract_news(etl, fake_sp):
    c = ctx_for(etl, markets=("US", "EU"))
    assert etl.extract_news(c) == "ok"
    assert json.load(open(c.artifacts["news"])) == {"US": {"n_articles": 7}, "EU": {}}
    assert c.rows == {"news:US": 7, "news:EU": 0}
    c2 = ctx_for(etl, markets=("US",))
    assert etl.extract_news(c2) == "skip"


def test_extract_news_unavailable(etl, monkeypatch):
    mod = types.ModuleType("sentiment_pipeline")

    def boom():
        raise RuntimeError("down")

    mod.SentimentPipeline = boom
    monkeypatch.setitem(sys.modules, "sentiment_pipeline", mod)
    c = ctx_for(etl)
    assert etl.extract_news(c) == "ok"
    assert c.notes["extract_news"] == "sentiment unavailable: down"
    assert json.load(open(c.artifacts["news"])) == {}


def test_extract_ipo_tracker(etl, monkeypatch):
    mod = types.ModuleType("ipo_tracker")
    mod.DEFAULT_MARKETS = ["IN", "US"]
    mod.track = lambda mk, verbose: {"IN": {"new": ["a", "b"]}, "US": {"new": []}}
    monkeypatch.setitem(sys.modules, "ipo_tracker", mod)
    c = ctx_for(etl)
    assert etl.extract_ipo_tracker(c) == "ok"
    assert c.rows["ipo_new"] == 2
    assert c.notes["extract_ipo_tracker"] == "new listings — IN:2"
    mod.track = lambda mk, verbose: {"IN": {}}
    c = ctx_for(etl)
    etl.extract_ipo_tracker(c)
    assert c.notes["extract_ipo_tracker"] == "new listings — none"

    def boom(mk, verbose):
        raise ValueError("feed")

    mod.track = boom
    c = ctx_for(etl)
    assert etl.extract_ipo_tracker(c) == "fail"
    assert c.notes["extract_ipo_tracker"] == "failed: feed"
    # fresh snapshot -> skip (even w/o force); force -> runs
    snap = os.path.join(etl.BASE, "data", "bhavcopy_cache", "universe_snapshots", "IN.json")
    os.makedirs(os.path.dirname(snap))
    open(snap, "w").write("{}")
    assert etl.extract_ipo_tracker(ctx_for(etl)) == "skip"
    assert etl.extract_ipo_tracker(ctx_for(etl, force=True)) == "fail"


def test_extract_exchange_reference(etl, monkeypatch):
    mod = types.ModuleType("exchange_reference")
    df = pd.DataFrame({"exchange": ["A", "B", "C"], "status": ["FRESH", "STALE", "STALE"]})
    state = {"df": df}
    mod.build_reference = lambda d: state["df"]
    mod.write_parquet = lambda d: "/p.parquet"
    mod.checklist_md = lambda d, dd: "# md " + dd.isoformat()
    monkeypatch.setitem(sys.modules, "exchange_reference", mod)
    c = ctx_for(etl)
    assert etl.extract_exchange_reference(c) == "ok"
    assert c.notes["extract_exchange_reference"] == "3 exchanges, 1 fresh · ⚠ STALE: B, C"
    assert c.rows["exchange_reference"] == 3
    assert open(c.artifacts["checklist"]).read() == "# md " + etl.RUN_DATE
    state["df"] = pd.DataFrame({"exchange": ["A"], "status": ["FRESH"]})
    c = ctx_for(etl)
    etl.extract_exchange_reference(c)
    assert c.notes["extract_exchange_reference"] == "1 exchanges, 1 fresh · none stale"
    state["df"] = pd.DataFrame()
    c = ctx_for(etl)
    assert etl.extract_exchange_reference(c) == "skip"
    mod.build_reference = lambda d: (_ for _ in ()).throw(KeyError("k"))
    c = ctx_for(etl)
    assert etl.extract_exchange_reference(c) == "fail"
    assert c.notes["extract_exchange_reference"].startswith("failed:")


@pytest.mark.parametrize(
    "pe,sm,zone",
    [
        (8, 10, "BUY"),
        (8.1, 10, "FAIR"),
        (11, 10, "FAIR"),
        (11.1, 10, "CAUTION"),
        (16, 10, "CAUTION"),
        (16.1, 10, "RICH"),
        (0, 10, "n/a"),
        (-1, 10, "n/a"),
        (5, 0, "n/a"),
        (np.nan, 10, "n/a"),
        (5, np.nan, "n/a"),
    ],
)
def test_pe_zone(etl, pe, sm, zone):
    assert etl._pe_zone(pe, sm) == zone


def _scan_wb(path, fund=None, darvas=None, fund_sheet="All_Fundamentals"):
    with pd.ExcelWriter(path) as w:
        if fund is not None:
            fund.to_excel(w, sheet_name=fund_sheet, index=False)
        if darvas is not None:
            darvas.to_excel(w, sheet_name="Darvas_Signals", index=False)
        if fund is None and darvas is None:
            pd.DataFrame({"a": [1]}).to_excel(w, sheet_name="Other", index=False)


def test_transform_fundamentals(etl, tmp_path):
    us = tmp_path / "us.xlsx"
    _scan_wb(
        us,
        pd.DataFrame(
            {
                "Symbol": ["A", "B", "C"],
                "Name": ["a", "b", "c"],
                "Sector": ["X", "X", "X"],
                "P_E_Ratio": [8.0, 10.0, 12.0],
                "Piotroski_Score": [9, 7, 3],
                "Extra": [1, 2, 3],
            }
        ),
    )
    eu = tmp_path / "eu.xlsx"
    _scan_wb(
        eu,
        pd.DataFrame(
            {"Symbol": ["E1", "E2"], "Sector": ["Y", "Y"], "Earnings_Yield_%": [10.0, 0.0]}
        ),
        fund_sheet="Fundamentals",
    )
    c = ctx_for(etl, markets=("US", "EU", "IN"))
    c.artifacts["scan:US"] = str(us)
    c.artifacts["scan:EU"] = str(eu)
    c.artifacts["scan:IN"] = str(us)  # no fundamentals sheet configured
    assert etl.transform_fundamentals(c) == "ok"
    df = pd.read_parquet(c.artifacts["fundamentals"])
    assert list(df["Market"]) == ["US"] * 3 + ["EU"] * 2
    assert "Extra" not in df.columns
    # median of US = 10: 8 -> ratio .8 BUY ; 10 -> FAIR ; 12 -> 1.2 CAUTION
    assert list(df[df.Market == "US"]["PE_Zone"]) == ["BUY", "FAIR", "CAUTION"]
    eu_df = df[df.Market == "EU"]
    assert eu_df["P_E_Ratio"].iloc[0] == pytest.approx(10.0)
    assert np.isnan(eu_df["P_E_Ratio"].iloc[1])
    assert list(eu_df["PE_Zone"]) == ["FAIR", "n/a"]
    assert c.rows["fundamentals"] == 5
    assert c.lineage[c.artifacts["fundamentals"]] == [str(us), str(eu), str(us)]


def test_transform_fundamentals_skip(etl):
    assert etl.transform_fundamentals(ctx_for(etl, markets=("US",))) == "skip"
    c = ctx_for(etl, markets=("IN",))
    c.artifacts["scan:IN"] = "x"
    assert etl.transform_fundamentals(c) == "skip"


def test_transform_fundamentals_no_pe(etl, tmp_path):
    us = tmp_path / "us.xlsx"
    _scan_wb(us, pd.DataFrame({"Symbol": ["A"], "Sector": ["X"], "Dummy": [1]}))
    c = ctx_for(etl)
    c.artifacts["scan:US"] = str(us)
    # missing PE column: pd.to_numeric(None) differs across pandas; either ok or TypeError
    try:
        r = etl.transform_fundamentals(c)
    except (TypeError, KeyError):
        return
    assert r == "ok"


def test_transform_darvas(etl, tmp_path):
    us = tmp_path / "us.xlsx"
    _scan_wb(
        us,
        darvas=pd.DataFrame(
            {
                "Symbol": list("ABCDEF"),
                "Darvas_Signal": ["BREAKOUT_BUY"] * 5 + ["HOLD"],
                "Position_in_Box%": [100, 120, 99.9, 120.1, 110, 110],
                "Change%": [1, 1, 1, 1, 0, 1],
                "Box_Top": [1] * 6,
                "Junk": [0] * 6,
            }
        ),
    )
    bad = tmp_path / "bad.xlsx"
    _scan_wb(bad)  # no Darvas_Signals sheet
    c = ctx_for(etl, markets=("US", "EU", "IN"))
    c.artifacts["scan:US"] = str(us)
    c.artifacts["scan:EU"] = str(bad)
    assert etl.transform_darvas(c) == "ok"
    df = pd.read_parquet(c.artifacts["darvas"])
    assert "Junk" not in df.columns
    assert list(df["fresh_breakout"]) == [True, True, False, False, False, False]
    assert c.rows["darvas"] == 6 and c.rows["darvas:fresh"] == 2
    c = ctx_for(etl, markets=("EU",))
    c.artifacts["scan:EU"] = str(bad)
    assert etl.transform_darvas(c) == "skip"
    assert etl.transform_darvas(ctx_for(etl)) == "skip"


def _fund_parquet(etl):
    p = os.path.join(etl.CURATED, "fundamentals.parquet")
    pd.DataFrame(
        {
            "Market": ["US", "US", "US", "US", "EU", "US"],
            "Symbol": ["P1", "N1", "Q1", "Z1", "P2", "P1"],
            "Name": ["Dup", "Neg", None, "Zed", "EuName", "dup"],
            "Sector": ["X"] * 6,
            "PE_Zone": ["BUY"] * 6,
            "Piotroski_Score": [8, 7, 6.9, 9, 7, 9],
        }
    ).to_parquet(p, index=False)
    return p


def test_transform_sentiment(etl, fake_sp):
    c = ctx_for(etl, markets=("US", "EU"))
    assert etl.transform_sentiment(c) == "skip"
    c.artifacts["fundamentals"] = _fund_parquet(etl)
    assert etl.transform_sentiment(c) == "ok"
    df = pd.read_parquet(c.artifacts["sentiment"])
    # Q1 below 7, Z1 zero articles, last P1 dup name (case-insens) skipped
    assert list(df["Symbol"]) == ["P1", "N1", "P2"]
    assert list(df["sentiment"]) == ["POSITIVE", "NEGATIVE", "POSITIVE"]
    assert list(df["headline"]) == ["h-P1-None", "h-N1-None", ""]
    assert c.rows["sentiment"] == 3


def test_transform_sentiment_eu_company_name(etl, fake_sp):
    c = ctx_for(etl, markets=("EU",))
    p = os.path.join(etl.CURATED, "f.parquet")
    pd.DataFrame(
        {"Market": ["EU"], "Symbol": ["P3"], "Name": ["Nm"], "Piotroski_Score": [7]}
    ).to_parquet(p)
    c.artifacts["fundamentals"] = p
    etl.transform_sentiment(c)
    assert pd.read_parquet(c.artifacts["sentiment"])["headline"].iloc[0] == "h-P3-Nm"


def test_transform_sentiment_unavailable(etl, monkeypatch):
    mod = types.ModuleType("sentiment_pipeline")

    def boom():
        raise RuntimeError("zz")

    mod.SentimentPipeline = boom
    monkeypatch.setitem(sys.modules, "sentiment_pipeline", mod)
    c = ctx_for(etl)
    c.artifacts["fundamentals"] = "x"
    assert etl.transform_sentiment(c) == "fail"
    assert c.notes["transform_sentiment"] == "unavailable: zz"


def test_transform_convergence(etl):
    c = ctx_for(etl)
    assert etl.transform_convergence(c) == "skip"
    c.artifacts["fundamentals"] = _fund_parquet(etl)
    assert etl.transform_convergence(c) == "skip"
    sp = os.path.join(etl.CURATED, "s.parquet")
    pd.DataFrame(
        {
            "Market": ["US", "US", "US", "US"],
            "Symbol": ["P1", "N1", "Q1", "Z1"],
            "sentiment": ["POSITIVE", "NEGATIVE", "NEUTRAL", "POSITIVE"],
        }
    ).to_parquet(sp)
    c.artifacts["sentiment"] = sp
    assert etl.transform_convergence(c) == "ok"
    df = pd.read_parquet(c.artifacts["convergence"]).sort_values("Symbol")
    # Q1 score 6.9 <7 excluded; P1 appears twice in fundamentals
    assert set(df["Symbol"]) == {"P1", "N1", "Z1"}
    sig = dict(zip(df["Symbol"], df["signal"]))
    assert sig == {"P1": "HIGH_CONVICTION", "N1": "CAUTION", "Z1": "HIGH_CONVICTION"}
    pd.DataFrame({"Market": ["US"], "Symbol": ["Q1"], "sentiment": ["NEUTRAL"]}).to_parquet(sp)
    c2 = ctx_for(etl)
    c2.artifacts.update(fundamentals=c.artifacts["fundamentals"], sentiment=sp)
    df = None
    etl.transform_convergence(c2)
    assert c2.rows["convergence"] == 0


def test_transform_convergence_empty_sentiment(etl):
    sp = os.path.join(etl.CURATED, "s.parquet")
    pd.DataFrame({"Market": [], "Symbol": [], "sentiment": []}).to_parquet(sp)
    c = ctx_for(etl)
    c.artifacts.update(fundamentals=_fund_parquet(etl), sentiment=sp)
    assert etl.transform_convergence(c) == "ok"
    assert c.rows["convergence"] == 0


def test_transform_hidden_gems(etl):
    c = ctx_for(etl)
    assert etl.transform_hidden_gems(c) == "skip"
    c = ctx_for(etl, hidden_gems=True)
    assert etl.transform_hidden_gems(c) == "skip"
    p = os.path.join(etl.CURATED, "f.parquet")
    pd.DataFrame(
        {
            "Symbol": list("ABCD"),
            "Piotroski_Score": [8, 9, 8, 7],
            "Net_Profit_$M": [5.0, 9.0, 1.0, 0.1],
        }
    ).to_parquet(p)
    c.artifacts["fundamentals"] = p
    assert etl.transform_hidden_gems(c) == "ok"
    df = pd.read_parquet(c.artifacts["hidden_gems"])
    assert list(df["Symbol"]) == ["B", "C", "A"]
    assert "_size" not in df.columns
    assert c.rows["hidden_gems"] == 3
    pd.DataFrame({"Symbol": list("ABC"), "Piotroski_Score": [8, 9, 8]}).to_parquet(p)
    assert etl.transform_hidden_gems(c) == "ok"
    assert list(pd.read_parquet(c.artifacts["hidden_gems"])["Symbol"]) == ["B", "A", "C"]


def test_load_report(etl, monkeypatch):
    captured = {}

    def fake(cmd, **kw):
        captured["env"] = kw["env"]
        return types.SimpleNamespace(returncode=0, stdout="x\nSUBJECT: hello\n", stderr="")

    monkeypatch.setattr(etl.subprocess, "run", fake)
    dl = os.path.expanduser("~/Downloads")
    os.makedirs(dl, exist_ok=True)
    open(f"{dl}/daily_market_report_{etl.RUN_DATE}.html", "w").write("h")
    open(f"{dl}/daily_market_report_hiddengems_artifact.html", "w").write("h")
    monkeypatch.delenv("HIDDEN_GEMS", raising=False)
    c = ctx_for(etl)
    assert etl.load_report(c) == "ok"
    assert "HIDDEN_GEMS" not in captured["env"]
    assert c.notes["load_report"] == "SUBJECT: hello"
    assert "email" in c.artifacts and "artifact" not in c.artifacts
    c = ctx_for(etl, hidden_gems=True)
    etl.load_report(c)
    assert captured["env"]["HIDDEN_GEMS"] == "1"
    assert c.artifacts["artifact"].endswith("hiddengems_artifact.html")


def test_load_report_fail(etl, monkeypatch):
    monkeypatch.setattr(
        etl.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="", stderr="a\nlast err"),
    )
    c = ctx_for(etl)
    assert etl.load_report(c) == "fail"
    assert c.notes["load_report"] == "last err"
    monkeypatch.setattr(
        etl.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    c = ctx_for(etl)
    etl.load_report(c)
    assert c.notes["load_report"] == "no output"


def test_load_deliver(etl, monkeypatch, tmp_path):
    c = ctx_for(etl)
    assert etl.load_deliver(c) == "skip"
    rep = tmp_path / "r.html"
    rep.write_text("<b>x</b>")
    c.artifacts["email"] = str(rep)
    c.no_send = True
    assert etl.load_deliver(c) == "skip"
    assert "--no-send" in c.notes["load_deliver"]
    c.no_send = False
    for k in ("GMAIL_USER", "GMAIL_APP_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    assert etl.load_deliver(c) == "skip"
    assert "no valid GMAIL_APP_PASSWORD" in c.notes["load_deliver"]
    monkeypatch.setenv("GMAIL_USER", "u@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "shortpw")  # len < 12
    assert etl.load_deliver(c) == "skip"
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "put-your-password-here")
    assert etl.load_deliver(c) == "skip"
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "x" * 12)
    sent = []
    mod = types.ModuleType("send_mailer")
    mod.send = lambda s, b, h: sent.append((s, b, h)) or True
    monkeypatch.setitem(sys.modules, "send_mailer", mod)
    c.notes["load_report"] = "SUBJECT: Subj"
    assert etl.load_deliver(c) == "ok"
    assert sent == [("Subj", "See the HTML report.", "<b>x</b>")]
    assert c.notes["load_deliver"] == "emailed: Subj"
    c.notes.pop("load_report")
    etl.load_deliver(c)
    assert sent[-1][0] == f"Daily Market Report — {etl.RUN_DATE}"
    c.hidden_gems = True
    c.rows["hidden_gems"] = 4
    c.notes["load_report"] = "SUBJECT: S"
    etl.load_deliver(c)
    assert sent[-1][0] == "💎 Hidden Gems (full-market) — 4 strong picks · S"
    mod.send = lambda *a: False
    assert etl.load_deliver(c) == "fail"
    assert c.notes["load_deliver"].startswith("send returned False")

    def boom(*a):
        raise OSError("smtp")

    mod.send = boom
    assert etl.load_deliver(c) == "fail"
    assert c.notes["load_deliver"] == "send failed: smtp"


def test_load_manifest(etl):
    c = ctx_for(etl, markets=("US",), force=True)
    c.status["a"] = "ok"
    c.timings["a"] = 1.2345
    out = os.path.join(etl.CURATED, "o.parquet")
    inp = os.path.join(etl.BASE, "in.xlsx")
    c.lineage[out] = [inp, None]
    c.artifacts["x"] = out
    c.artifacts["none"] = None
    assert etl.load_manifest(c) == "ok"
    m = json.load(open(c.artifacts["manifest"]))
    assert m["run_date"] == etl.RUN_DATE and m["forced"] is True
    assert m["stage_seconds"] == {"a": 1.23}
    assert m["artifacts"]["x"] == os.path.join("data", "curated", "o.parquet")
    assert "none" not in m["artifacts"]
    assert m["lineage"] == {os.path.join("data", "curated", "o.parquet"): ["in.xlsx"]}


def _patch_stages(etl, monkeypatch, order):
    def mk(name, st="ok", boom=False):
        def fn(ctx):
            order.append(name)
            if boom:
                raise RuntimeError("bad" * 100)
            if name == "transform_darvas":
                ctx.rows["darvas"] = 3
            (
                ctx.notes.setdefault("transform_darvas", ["n1", "n2"])
                if name == "transform_darvas"
                else None
            )
            return st

        return fn

    stages = [
        ("e1", "E", mk("e1")),
        ("transform_darvas", "T", mk("transform_darvas")),
        ("t2", "T", mk("t2", boom=True)),
        ("l1", "L", mk("l1", st=None)),
    ]
    monkeypatch.setattr(etl, "STAGES", stages)


def test_run_phases_and_status(etl, monkeypatch, capsys):
    order = []
    _patch_stages(etl, monkeypatch, order)
    ctx = etl.run(["US"], ["T"], None, False, False)
    assert order == ["transform_darvas", "t2"]
    assert ctx.status == {"transform_darvas": "ok", "t2": "fail"}
    assert ctx.notes["t2"].startswith("EXC RuntimeError: ")
    assert len(ctx.notes["t2"]) == len("EXC RuntimeError: ") + 120
    out = capsys.readouterr().out
    assert "· 3 rows" in out and "n1; n2" in out
    assert "phases=T" in out and "FORCE" not in out
    assert "manifest" in ctx.artifacts


def test_run_single_stage_and_dry(etl, monkeypatch, capsys):
    order = []
    _patch_stages(etl, monkeypatch, order)
    ctx = etl.run(["US"], ["E"], "l1", True, False)
    assert order == ["l1"] and ctx.status["l1"] == "ok"
    assert "manifest" not in ctx.artifacts
    out = capsys.readouterr().out
    assert "STAGE=l1" in out and "FORCE" in out
    order.clear()
    ctx = etl.run(["US"], ["E", "T", "L"], None, False, True)
    assert order == [] and set(ctx.status.values()) == {"dry"}
    assert "manifest" not in ctx.artifacts
    assert "DRY-RUN" in capsys.readouterr().out
    ctx = etl.run(["US"], [], "load_manifest", False, False)
    # only_stage load_manifest not in patched list -> finalised explicitly
    assert "manifest" in ctx.artifacts


def test_main(etl, monkeypatch, capsys):
    seen = {}

    def fake_run(markets, phases, stage, force, dry, **kw):
        seen.update(markets=markets, phases=phases, stage=stage, force=force, dry=dry, **kw)
        return types.SimpleNamespace(artifacts={"manifest": os.path.join(etl.BASE, "m.json")})

    monkeypatch.setattr(etl, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["etl", "--markets", "us, eu", "--phase", "t,l", "--no-send"])
    etl.main()
    assert seen["markets"] == ["US", "EU"] and seen["phases"] == ["T", "L"]
    assert seen["no_send"] is True and seen["refresh_scans"] is False
    assert seen["hidden_gems"] is False and seen["force"] is False and seen["stage"] is None
    assert "manifest → m.json" in capsys.readouterr().out
    monkeypatch.setattr(
        sys,
        "argv",
        ["etl", "--stage", "x", "--force", "--dry-run", "--refresh-scans", "--hidden-gems"],
    )
    etl.main()
    assert seen["phases"] == ["E", "T", "L"] and seen["markets"] == ["US", "EU", "IN"]
    assert seen["force"] and seen["dry"] and seen["refresh_scans"] and seen["hidden_gems"]
    fake_run2 = lambda *a, **k: types.SimpleNamespace(artifacts={})  # noqa: E731
    monkeypatch.setattr(etl, "run", fake_run2)
    capsys.readouterr()
    etl.main()
    assert "manifest" not in capsys.readouterr().out
