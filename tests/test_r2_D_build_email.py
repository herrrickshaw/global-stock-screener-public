"""build_email.py is a top-level script: run it with HOME redirected to tmp_path and synthetic inputs."""

import datetime
import json
import runpy
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = str(Path(__file__).resolve().parent.parent / "build_email.py")


def _us_scan(base):
    d = base / "us_full_scan"
    d.mkdir(parents=True)
    rows = [
        # Symbol, Exchange, Name, Sector, EarningsYield
        ("AAA", "NASDAQ", "Alpha Inc", "Tech", 10.0),  # PE 10
        ("BBB", "NASDAQ", "Beta Inc", "Tech", 5.0),  # PE 20
        ("CCC", "NASDAQ", "Gamma Inc", "Tech", 2.5),  # PE 40
        ("HHH", "NASDAQ", "Eta Inc", "Tech", 100 / 28),  # PE 28
        ("DDD", "NYSE", "Delta Corp", "Fin", 8.0),
        ("EEE", "AMEX", None, "Fin", 0.0),  # PE NaN, name NaN
        ("FFF", "NYSE", "Foxtrot", None, 6.25),  # NaN sector -> universe median
        ("GGG", None, "Golf", "Fin", 4.0),  # exchange NaN -> "US"
    ]
    pd.DataFrame(
        rows, columns=["Symbol", "Exchange", "Name", "Sector", "Earnings_Yield_%"]
    ).to_excel(d / "us_full_scan_20260708_1238.xlsx", sheet_name="All_Fundamentals", index=False)


def _eu_rows(n_strong=35, fresh=True):
    suffixes = [".DE", ".L", ".PA", ".SW", ".ST", ".WA", ".IS", ".PR", ".BD", ".MC", ".TL", ".XX"]
    fund = []
    for i in range(n_strong):
        sfx = suffixes[i % len(suffixes)]
        fund.append(
            {
                "Symbol": f"S{i}{sfx}",
                "Name": f"Comp{i} AG,0P0001,1060" if i % 5 == 0 else f"Comp{i} AG",
                "Sector": "Ind" if i % 2 else "Fin",
                "LTP": 10.0 + i,
                "P_E_Ratio": [5.0, 12.0, 15.0, 30.0, np.nan, -3.0][i % 6],
                "Piotroski_Score": 9 - (i % 3),
                "CoffeeCan_Score": f"{i % 7}/10",
                "CoffeeCan_Class": "PASS" if i % 2 == 0 else "FAIL",
                "Darvas_Signal": "BREAKOUT_BUY" if i % 4 == 0 else "NONE",
            }
        )
    fund.append(  # weak piotroski -> excluded from picks
        dict(fund[0], Symbol="WEAK.DE", Name="Weak AG", Piotroski_Score=3)
    )
    dv = []

    def d(sym, name, sig, trend, pos, chg):
        dv.append(
            {
                "Symbol": sym,
                "Name": name,
                "LTP": 20.0,
                "Change%": chg,
                "Box_Top": 19.0,
                "Position_in_Box%": pos,
                "Darvas_Signal": sig,
                "Trend_Signal": trend,
            }
        )

    if fresh:
        d("F1.DE", "Fresh1", "BREAKOUT_BUY", "Uptrend", 110, 2.0)
        d("F2.L", "Fresh2", "BREAKOUT_BUY", "Strong Uptrend", 101, 5.0)
    else:
        d("X1.DE", "Ext1", "BREAKOUT_BUY", "Uptrend", 150, 2.0)
        d("X2.DE", "Ext2", "BREAKOUT_BUY", "Uptrend", 130, -1.0)
    d("D1.DE", "Down", "BREAKOUT_BUY", "Downtrend", 105, 3.0)  # not uptrend
    d("E1.IS", "Ist1", "BREAKOUT_BUY", "Uptrend", 150, 9.0)
    d("E2.PR", "Pra1", "BREAKOUT_BUY", "Uptrend", 150, 7.0)
    d("E3.BD", "Bud1", "BREAKOUT_BUY", "Uptrend", 150, 8.0)
    d("B1.DE", "Brk1", "BREAKDOWN_SELL", "Downtrend", 50, -7.0)
    d("B2.L", "Brk2", "BREAKDOWN_SELL", "Downtrend", 50, -2.0)
    return pd.DataFrame(fund), pd.DataFrame(dv)


def _eu_scan(path, fund, dv, all_stocks=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as w:
        fund.to_excel(w, sheet_name="Fundamentals", index=False)
        dv.to_excel(w, sheet_name="Darvas_Signals", index=False)
        if all_stocks:
            pd.DataFrame({"Symbol": ["A.DE", "B.DE", "C.L", "D.PA", "E.PA"]}).to_excel(
                w, sheet_name="All_Stocks", index=False
            )


def _jsons(base):
    cr = base / "combined_report_results"
    cr.mkdir(parents=True)
    in_picks = [
        {"Symbol": f"IN{i}", "LTP": 100.5 + i, "Tier": "Single-Screen", "Screens": "Darvas"}
        for i in range(30)
    ]
    inj = {
        "picks": in_picks,
        "mood": {
            "mood": "NEGATIVE",
            "score": -0.2,
            "n_articles": 5,
            "by_outlet": {"ET": 1, "MC": 2},
        },
        "talk": {
            "IN0": {
                "n_articles": 2,
                "label": "POSITIVE",
                "score": 0.3,
                "headlines": ["<b>head</b>" + "x" * 100],
            },
            "IN1": {"n_articles": 0, "label": "NEUTRAL", "score": 0, "headlines": []},
            "IN2": {"n_articles": 3, "label": "NO_DATA", "score": 0},
            "IN3": {"n_articles": 1, "label": "WEIRD", "score": -0.1, "headlines": []},
        },
        "convergence": [],
    }
    us_picks = [
        {"Symbol": "AAA", "Tier": "Triple Hit", "Screens": "D+P+C", "LTP": 12.5, "Piotroski": 8.0},
        {
            "Symbol": "BBB",
            "Tier": "Multi-Screen",
            "Screens": "D+P",
            "LTP": 1234.5,
            "Piotroski": None,
        },
        {
            "Symbol": "CCC",
            "Tier": "Multi-Screen",
            "Screens": "x",
            "LTP": 3.0,
            "Piotroski": float("nan"),
        },
        {"Symbol": "HHH", "Tier": "Triple Hit", "Screens": "x", "LTP": 3.0},
        {"Symbol": "DDD", "Tier": "Multi-Screen", "Screens": "x", "LTP": 3.0, "Piotroski": 7.0},
        {"Symbol": "EEE", "Tier": "Triple Hit", "Screens": "x", "LTP": 3.0, "Piotroski": 7.0},
        {"Symbol": "FFF", "Tier": "Triple Hit", "Screens": "x", "LTP": 3.0, "Piotroski": 7.0},
        {
            "Symbol": "ZZZ",
            "Tier": "Triple Hit",
            "Screens": "x",
            "LTP": 3.0,
            "Piotroski": 7.0,
        },  # not in meta
        {"Symbol": "SGL", "Tier": "Single-Screen", "Screens": "x", "LTP": 3.0},  # filtered out
    ]
    usj = {
        "picks": us_picks,
        "mood": {"mood": "POSITIVE", "score": 0.25, "n_articles": 9},
        "talk": {
            "AAA": {"n_articles": 4, "label": "NEUTRAL", "score": 0.0, "headlines": ["us head"]}
        },
        "convergence": [
            {
                "Symbol": "AAA",
                "Convergence": "BOTH BULLISH",
                "Tier": "Triple Hit",
                "News_Score": 0.5,
                "N_Articles": 3,
                "Top_Headline": "good <news>",
            },
            {
                "Symbol": "DDD",
                "Convergence": "FUND STRONG / NEGATIVE NEWS",
                "Tier": "Multi-Screen",
                "News_Score": -0.4,
                "N_Articles": 2,
                "Top_Headline": "bad",
            },
            {
                "Symbol": "BBB",
                "Convergence": "NEUTRAL",
                "Tier": "x",
                "News_Score": 0,
                "N_Articles": 1,
                "Top_Headline": "n",
            },
        ],
    }
    (cr / "combined_IN_20260709_0808.json").write_text(json.dumps(inj))
    (cr / "combined_US_20260709_0809.json").write_text(json.dumps(usj))
    return inj, usj


class FakeSP:
    mood = {"mood": "POSITIVE", "score": 0.1, "n_articles": 12, "by_outlet": {"BBC": 1}}
    calls = []
    raise_on_init = False

    def __init__(self):
        if FakeSP.raise_on_init:
            raise RuntimeError("x" * 300)

    def get_market_mood(self, m):
        assert m == "EU"
        return dict(FakeSP.mood, by_outlet=dict(FakeSP.mood["by_outlet"]))

    def get_ticker_sentiment(self, sym, market, company_name=""):
        FakeSP.calls.append((sym, market, company_name))
        n = (
            int(company_name.replace("Comp", "").split(" ")[0])
            if company_name.startswith("Comp")
            else 0
        )
        lab = ["POSITIVE", "NEGATIVE", "NEUTRAL", "NO_DATA"][n % 4]
        return types.SimpleNamespace(
            n_articles=0 if lab == "NO_DATA" else 2,
            label=lab,
            score={"POSITIVE": 0.4, "NEGATIVE": -0.3, "NEUTRAL": 0.0, "NO_DATA": 0}[lab],
            top_headlines=[{"title": f"headline {n}"}] if n % 8 != 2 else [],
        )


_n = [0]


def _run(
    tmp_path,
    monkeypatch,
    *,
    fresh=True,
    n_strong=35,
    all_stocks=True,
    sp="ok",
    emerg=None,
    hidden=False,
    frags=True,
    broad=True,
    us_scan=True,
):
    _n[0] += 1
    home = tmp_path / f"home{_n[0]}"
    base = home / "Downloads" / "code" / "python_files"
    base.mkdir(parents=True)
    if us_scan:
        _us_scan(base)
    _jsons(base)
    fund, dv = _eu_rows(n_strong, fresh)
    name = (
        "european_market_scan_broad_20260709.xlsx"
        if broad
        else "european_market_scan_20260709.xlsx"
    )
    _eu_scan(base / "european_scan" / name, fund, dv, all_stocks)
    if broad:  # an older non-broad file that must be ignored
        _eu_scan(base / "european_scan" / "european_market_scan_old.xlsx", fund.head(1), dv.head(1))
    if frags:
        (home / "Downloads" / "in_darvas.html").write_text("<p>IN-DARVAS-FRAG</p>")
    (home / "Downloads").mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    if hidden:
        monkeypatch.setenv("HIDDEN_GEMS", "1")
    else:
        monkeypatch.delenv("HIDDEN_GEMS", raising=False)
    FakeSP.calls = []
    FakeSP.raise_on_init = sp == "raise"
    if sp == "noimport":
        monkeypatch.setitem(sys.modules, "sentiment_pipeline", None)
    else:
        mod = types.ModuleType("sentiment_pipeline")
        mod.SentimentPipeline = FakeSP
        monkeypatch.setitem(sys.modules, "sentiment_pipeline", mod)
    ef = types.ModuleType("emerging_fundamentals")
    ef.available_source = lambda: emerg
    monkeypatch.setitem(sys.modules, "emerging_fundamentals", ef if emerg != "missing" else None)
    if emerg == "missing":
        monkeypatch.setitem(sys.modules, "emerging_fundamentals", None)
    g = runpy.run_path(SCRIPT, run_name="build_email_under_test")
    return g, home


@pytest.fixture
def run(tmp_path, monkeypatch):
    return lambda **kw: _run(tmp_path, monkeypatch, **kw)


def test_full_report_outputs(run, capsys):
    g, home = run()
    out = capsys.readouterr().out
    today = datetime.date.today()
    rpt = home / "Downloads" / f"daily_market_report_{today.isoformat()}.html"
    art = home / "Downloads" / "daily_market_report_artifact.html"
    assert rpt.exists() and art.exists()
    assert f"PATH: {rpt}" in out and f"ARTIFACT: {art}" in out
    email = rpt.read_text()
    subj = [l for l in out.splitlines() if l.startswith("SUBJECT:")][0]
    assert f"Daily Market Report — {today.strftime('%d %b %Y')}" in subj
    assert "Street mood: POSITIVE" in subj
    # fundamentals: 8 of 9 us picks pass tier filter + EU picks
    n_fund = g["n_fund"]
    assert n_fund == 8 + 35 and f"Fundamentals: {n_fund} picks" in subj
    assert f"Convergence: {g['n_conv']}" in subj and g["n_conv"] == 1 + len(g["eu_conv_hi"])
    assert "IN-DARVAS-FRAG" in email
    assert "us_darvas.html not found this run" in email
    assert f"BYTES: {len(email)}" in out
    assert (
        "MARKET DAILY" in art.read_text()
        and "Fundamentals · Street Talk · Darvas" in art.read_text()
    )
    assert 'class="trail"><table' in art.read_text()


def test_pe_zone_and_us_helpers(run):
    g, _ = run()
    pz = g["pe_zone"]
    assert pz("AAA")[0].startswith("🟢 BUY (PE 10 vs 24)") and pz("AAA")[1] == "#1a7f37"
    assert pz("BBB")[0].startswith("🟡 FAIR") and pz("BBB")[1] == "#b58a00"
    assert pz("HHH")[0].startswith("🟠 CAUTION") and pz("HHH")[1] == "#c05a00"
    assert pz("CCC")[0].startswith("🔴 RICH") and pz("CCC")[1] == "#c0392b"
    assert pz("EEE") == ("—", "#888")  # NaN PE
    assert pz("NOPE") == ("—", "#888")  # not in meta
    assert pz("FFF")[0].startswith("🟢")  # PE 16 vs universe median (NaN sector)
    assert (
        g["usname"]("AAA") == "Alpha Inc"
        and g["usname"]("EEE") == "EEE"
        and g["usname"]("ZZZ") == "ZZZ"
    )
    assert (
        g["usexch"]("AAA") == "NASDAQ" and g["usexch"]("GGG") == "US" and g["usexch"]("ZZZ") == "US"
    )


def test_pe_zone_boundaries(run):
    g, _ = run()
    # fu PEs: AAA10 BBB20 CCC40 HHH28 -> Tech median 24; 0.8*24=19.2, 1.1*24=26.4, 1.6*24=38.4
    meta = g["meta"]
    sm = g["sec_med"]["Tech"]
    assert sm == 24
    for pe, expect in [
        (19.2, "BUY"),
        (19.3, "FAIR"),
        (26.4, "FAIR"),
        (26.5, "CAUTION"),
        (38.4, "CAUTION"),
        (38.5, "RICH"),
    ]:
        meta["AAA"]["PE"] = pe
        assert expect in g["pe_zone"]("AAA")[0], (pe, g["pe_zone"]("AAA"))


def test_eu_helpers(run):
    g, _ = run()
    assert g["eu_exch"]("ADS.DE") == "XETRA" and g["eu_exch"]("rio.l") == "London SE"
    assert g["eu_exch"]("X.F") == "Frankfurt" and g["eu_exch"]("X.MC") == "BME Madrid"
    assert g["eu_exch"]("FOO") == "Europe" and g["eu_exch"]("A.ZZ") == "Europe"
    assert g["eu_exch"]("A.IC") == "Nasdaq Iceland"  # longest-first, not shadowed by shorter suffix
    assert (
        g["eu_cur"]("A.L") == "£" and g["eu_cur"]("A.SW") == "CHF " and g["eu_cur"]("A.WA") == "zł"
    )
    assert g["eu_cur"]("A.DE") == "€" and g["eu_cur"]("A.TL") == "€"
    assert g["eu_name"]("ADS.DE,0P00,1060") == "ADS.DE" and g["eu_name"]("  Name  ") == "Name"
    assert g["eu_name"](",x") == ",x"  # empty head falls back to full string
    assert g["esc"]("<a&>") == "&lt;a&amp;&gt;"


def test_eu_pe_zone_and_screens(run):
    g, _ = run()
    z = g["eu_pe_zone"]
    med = g["eu_sec_med"]["Fin"]
    assert (
        z(np.nan, "Fin") == ("—", "#888")
        and z(0, "Fin") == ("—", "#888")
        and z(-1, "Fin") == ("—", "#888")
    )
    assert z(med * 0.8, "Fin")[0].startswith("🟢") and z(med * 0.81, "Fin")[0].startswith("🟡")
    assert z(med * 1.1, "Fin")[0].startswith("🟡") and z(med * 1.2, "Fin")[0].startswith("🟠")
    assert z(med * 1.6, "Fin")[0].startswith("🟠") and z(med * 1.7, "Fin")[0].startswith("🔴")
    assert "vs" in z(10, "UnknownSector")[0]  # falls back to universe median
    s = g["eu_screens"]
    assert (
        s({"Darvas_Signal": "BREAKOUT_BUY", "Piotroski_Score": 7, "CoffeeCan_Class": "PASS"})
        == "Darvas+Piotroski+CoffeeCan"
    )
    assert s({"Darvas_Signal": "x", "Piotroski_Score": 6, "CoffeeCan_Class": "FAIL"}) == "—"
    assert s({"Piotroski_Score": np.nan}) == "—" and s({"Piotroski_Score": 7}) == "Piotroski"


def test_us_c1_tables(run):
    g, _ = run()
    c1 = g["us_c1"]
    assert "US — NASDAQ" in c1 and "US — NYSE" in c1 and "Other (ADR/AMEX)" in c1
    nas = c1.split("US — NYSE")[0]
    assert (
        nas.index("ALPHA INC") < nas.index("ETA INC") < nas.index("BETA INC")
    )  # Triple before Multi
    assert "$1,234.50" in nas and "$12.50" in nas
    assert "SGL" not in c1  # single-screen filtered
    assert '<td style="padding:6px 8px;border-bottom:1px solid #eee;text-align:center">8</td>' in c1
    other = c1.split("Other (ADR/AMEX)")[1]
    assert "[EEE · AMEX]" in other and "[ZZZ · US]" in other
    # NaN / None piotroski render blank
    assert 'text-align:center"></td>' in c1


def test_us_c1_nyse_empty_message(run, tmp_path, monkeypatch):
    g, home = run()
    rows = g["us_c1_rows"]([], ("Triple Hit",))
    assert rows == ""
    assert (
        g["us_c1_rows"](
            [{"Symbol": "AAA", "Tier": "Other", "Screens": "x", "LTP": 1}], ("Triple Hit",)
        )
        == ""
    )


def test_india_component(run):
    g, _ = run()
    assert g["in_c1"].count("[IN") == 25  # capped at 25
    assert "IN24 " in g["in_c1"] and "IN25 " not in g["in_c1"]
    assert "₹100.50" in g["in_c1"] and "no Triple-Hit or" in g["in_c1"]


def test_europe_picks_cap_and_sort(run):
    g, home = run()
    assert len(g["eu_pick_all"]) == 35 and len(g["eu_pick"]) == 30 and g["eu_more"] == 5
    assert "(+5 more in the CSV)" in g["eu_c1"]
    assert "<b>35</b> names cleared" in g["eu_c1"] and "strongest 30" in g["eu_c1"]
    pa = g["eu_pick_all"]
    assert bool(pa["is_triple"].iloc[0]) and pa["Piotroski_Score"].iloc[0] == 9
    # sorted: triple desc, then piotroski desc
    flags = list(pa["is_triple"])
    assert flags == sorted(flags, reverse=True)
    assert "WEAK.DE" not in g["eu_c1"]
    assert "5 stocks across 3 exchanges" in g["eu_c1"]  # All_Stocks sheet: .DE, .L, .PA
    csv = home / "Downloads" / "eu_broad_picks_2026-07-09.csv"
    assert csv.exists() and len(pd.read_csv(csv)) == 35
    assert "COMP0 AG" in g["eu_c1"]  # name cleaned of ",0P0001,1060"
    assert "0P0001" not in g["eu_c1"]
    assert "£" in g["eu_c1"] and "CHF " in g["eu_c1"] and "zł" in g["eu_c1"] and "kr" in g["eu_c1"]


def test_europe_no_more_and_all_stocks_fallback(run):
    g, _ = run(n_strong=4, all_stocks=False)
    assert g["eu_more"] == 0 and "more in the CSV" not in g["eu_c1"]
    # no All_Stocks sheet -> uses Darvas rows
    assert f"{len(g['eu_dv'])} stocks across" in g["eu_c1"]


def test_europe_darvas_fresh(run):
    g, _ = run(fresh=True)
    assert g["eu_fresh_mode"] == "fresh"
    assert list(g["eu_fresh"]["Name"]) == ["Fresh2", "Fresh1"]  # sorted by Change% desc
    d = g["eu_darvas"]
    assert "price just clearing the box top" in d
    assert "Ist1" in d and "Pra1" in d and "Bud1" in d  # emerging block
    assert d.index("Ist1") < d.index("Bud1") < d.index("Pra1")  # by Change% desc: 9, 8, 7
    assert "2 total" in d and "Brk1 (-7.0%), Brk2 (-2.0%)" in d  # breakdowns, worst first
    assert "Down" not in d.split("Emerging")[0]  # downtrend breakout excluded
    assert "momentum view" not in d and "FMP_API_KEY" in d


def test_europe_darvas_least_extended_fallback(run):
    g, _ = run(fresh=False)
    assert g["eu_fresh_mode"] == "least-extended"
    names = list(g["eu_fresh"]["Position_in_Box%"])
    assert names == sorted(names)  # least extended first
    assert "already extended above their boxes" in g["eu_darvas"]


def test_emerging_source_note(run):
    g, _ = run(emerg="FMP")
    assert "sourced from <b>FMP</b>" in g["eu_darvas"] and "FMP_API_KEY" not in g["eu_darvas"]
    g, _ = run(emerg="missing")
    assert "FMP_API_KEY" in g["eu_darvas"]


def test_eu_sentiment_dedupe_and_conviction(run):
    g, _ = run()
    talk = g["eu_talk"]
    assert FakeSP.calls and all(c[1] == "EU" for c in FakeSP.calls)
    assert len(FakeSP.calls) <= 40
    labels = {t["label"] for t in talk.values()}
    assert labels <= {"POSITIVE", "NEGATIVE", "NEUTRAL"}  # NO_DATA dropped
    assert len(g["eu_conv_hi"]) == sum(t["label"] == "POSITIVE" for t in talk.values())
    assert len(g["eu_conv_cau"]) == sum(t["label"] == "NEGATIVE" for t in talk.values())
    names = [t["name"] for t in talk.values()]
    assert len(names) == len(set(names))
    assert g["eu_mood"]["mood"] == "POSITIVE"
    assert {t["tier"] for t in talk.values()} <= {"Triple Hit", "Multi-Screen"}


def test_eu_sentiment_failure_and_missing(run):
    g, _ = run(sp="raise")
    assert g["eu_mood"] == {"error": "x" * 120}
    assert g["eu_talk"] == {} and g["eu_conv_hi"] == []
    assert "feed unavailable this run" in g["eu_mood_html"]()
    g, _ = run(sp="noimport")
    assert "error" in g["eu_mood"]


def test_conv_section(run):
    g, _ = run()
    c = g["conv_section"]()
    assert "&lt;news&gt;" in c  # escaped
    assert "ALPHA INC" in c and "+0.50" in c and "-0.40" in c
    assert "High-conviction" in c and "Caution" in c
    assert "BETA INC" not in c.split("Europe —")[0]  # NEUTRAL convergence not listed


def test_conv_section_empty(run):
    g, _ = run()
    g["usj"]["convergence"] = []
    g["eu_conv_hi"].clear()
    g["eu_conv_cau"].clear()
    assert (
        g["conv_section"]()
        == "<p>No fundamental picks had matching news coverage today (India · US · Europe).</p>"
    )
    g["usj"]["convergence"] = [{"Symbol": "AAA", "Convergence": "x"}]
    assert g["conv_section"]().startswith(
        "<p>No fundamental picks"
    )  # non-matching entries still count as none
    g["inj"]["convergence"] = [{"x": 1}]
    assert "None today." in g["conv_section"]()
    g["inj"]["convergence"] = []
    g["eu_conv_hi"].append(
        {
            "name": "n",
            "exch": "e",
            "tier": "t",
            "score": 0.5,
            "n_articles": 1,
            "headline": "h" * 200,
        }
    )
    c = g["conv_section"]()
    assert "<b>N</b> [e]" in c and ("h" * 90 + "…") in c and ("h" * 91) not in c


def test_mood_and_talk_tables(run):
    g, _ = run()
    m = g["mood_html"](g["inj"], "🇮🇳 India")
    assert "#c62828" in m and "NEGATIVE (-0.20)" in m and "5 live articles (ET · MC)" in m
    m = g["mood_html"](g["usj"], "US")
    assert "#2e7d32" in m and "wire feeds" in m and "+0.25" in m
    unknown = {"mood": {"mood": "???", "score": 0, "n_articles": 0}}
    assert "#555" in g["mood_html"](unknown, "X")
    t = g["talk_table"](g["inj"], "₹")
    assert "<b>IN0</b>" in t and "IN1" not in t and "IN2" not in t  # n=0 and NO_DATA skipped
    assert "IN3" in t and "#555" in t
    assert (
        "&lt;b&gt;head" in t and ("x" * 60) in t and ("x" * 70) not in t
    )  # escaped, then cut at 90 chars
    t = g["talk_table"](g["usj"], "$")
    assert "<b>ALPHA INC</b>" in t
    empty = g["talk_table"]({"talk": {}}, "$")
    assert "No per-ticker news coverage matched" in empty


def test_eu_mood_html_and_talk_table(run):
    g, _ = run()
    h = g["eu_mood_html"]()
    assert "POSITIVE (+0.10)" in h and "12 live articles (BBC)" in h
    g["eu_mood"].pop("by_outlet")
    assert "EU feeds" in g["eu_mood_html"]()
    g["eu_mood"]["mood"] = "WEIRD"
    assert "#555" in g["eu_mood_html"]()
    t = g["eu_talk_table"]()
    scores = [
        float(x) for x in __import__("re").findall(r"text-align:right'>([+-]\d\.\d\d)</td>", t)
    ]
    assert scores == sorted(scores, reverse=True)
    g["eu_talk"].clear()
    assert "No European pick was named" in g["eu_talk_table"]()


def test_read_frag_and_placeholder(run, tmp_path):
    g, home = run()
    assert g["_read_frag"]("in_darvas.html") == "<p>IN-DARVAS-FRAG</p>"
    assert "nothing.html not found this run" in g["_read_frag"]("nothing.html")
    assert g["in_darvas"] == "<p>IN-DARVAS-FRAG</p>"


def test_hidden_gems_variant(run):
    g, home = run(hidden=True)
    assert (home / "Downloads" / "daily_market_report_hiddengems_artifact.html").exists()
    assert not (home / "Downloads" / "daily_market_report_artifact.html").exists()
    art = (home / "Downloads" / "daily_market_report_hiddengems_artifact.html").read_text()
    assert (
        "💎 HIDDEN GEMS" in art
        and "hidden-gems scan, weekday evenings" in art
        and "IN / US / EU" in art
    )
    assert g["HIDDEN_GEMS"] is True


def test_default_variant_flags(run):
    g, _ = run()
    assert g["HIDDEN_GEMS"] is False and "Regenerated each weekday morning" in g["_foot"]


def test_wrap_tables(run):
    g, _ = run()
    assert (
        g["wrap_tables"]("a<table>x</table>b<table>\ny</table>")
        == 'a<div class="trail"><table>x</table></div>b<div class="trail"><table>\ny</table></div>'
    )


def test_latest_eu_prefers_broad(run):
    g, home = run()
    assert g["EU_SCAN"].endswith("european_market_scan_broad_20260709.xlsx")


def test_latest_eu_fallback_any(run):
    g, home = run(broad=False)
    assert g["EU_SCAN"].endswith("european_market_scan_20260709.xlsx")


def test_latest_eu_default_missing(tmp_path, monkeypatch):
    home = tmp_path / "home"
    base = home / "Downloads" / "code" / "python_files"
    base.mkdir(parents=True)
    _us_scan(base)
    _jsons(base)
    monkeypatch.setenv("HOME", str(home))
    with pytest.raises(FileNotFoundError, match="european_market_scan_20260709_0842"):
        runpy.run_path(SCRIPT, run_name="build_email_under_test")
