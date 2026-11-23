import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

import darvas_breakouts as db


def _row(
    sym,
    sig="BREAKOUT_BUY",
    gc="ABOVE_200DMA",
    pos=110.0,
    chg=1.0,
    dp=300,
    gap=10.0,
    ltp=110.0,
    bt=100.0,
    bb=50.0,
    suffix=".NS",
):
    return {
        "Symbol": sym, "Darvas_Signal": sig, "GC_Signal": gc, "Position_in_Box%": pos,
        "Change%": chg, "Data_Points": dp, "DMA_Gap%": gap, "LTP": ltp, "Box_Top": bt,
        "Box_Bottom": bb, "Suffix": suffix,
    }  # fmt: skip


def test_cname_esc():
    assert db.cname("GD") == "GENERAL DYNAMICS CORP"
    assert db.cname("ZZZ") == "ZZZ"
    assert db.esc("<a&>") == "&lt;a&amp;&gt;"


def test_yf_symbol():
    assert db._yf_symbol(" AB ", "US", ".BO") == "AB"
    assert db._yf_symbol("AB", "IN", ".bo") == "AB.bo"
    assert db._yf_symbol("AB", "IN", ".NS") == "AB.NS"
    assert db._yf_symbol("AB", "IN", ".XX") == "AB.NS"
    assert db._yf_symbol("AB", "IN") == "AB.NS"
    assert db._yf_symbol("AB", "IN", None) == "AB.NS"


def test_exchange_for():
    assert db.exchange_for({"Suffix": ".bo"}, "IN") == "BSE"
    assert db.exchange_for({"Suffix": ".NS"}, "IN") == "NSE"
    assert db.exchange_for({"Suffix": None}, "IN") == "NSE"
    assert db.exchange_for({}, "IN") == "NSE"
    assert db.exchange_for({"Suffix": ".BO"}, "US") == "US"


def test_latest_scan(tmp_path):
    a, b = tmp_path / "x_1.xlsx", tmp_path / "x_2.xlsx"
    a.write_text("a")
    b.write_text("b")
    os.utime(a, (1000, 1000))
    os.utime(b, (2000, 2000))
    cfg = {"scan_dirs": [str(tmp_path), str(tmp_path)], "glob": "x_*.xlsx"}
    assert db.latest_scan(cfg) == str(b)
    assert db.latest_scan({"scan_dirs": [str(tmp_path / "no")], "glob": "*"}) is None


def _fake_yf(monkeypatch, data=None, exc=None):
    m = types.ModuleType("yfinance")
    calls = []

    def download(syms, **kw):
        calls.append((syms, kw))
        if exc:
            raise exc
        return data

    m.download = download
    monkeypatch.setitem(sys.modules, "yfinance", m)
    return calls


def test_fetch_live_no_yfinance(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "yfinance", None)
    assert db.fetch_live_quotes([("A", "A")]) == {}
    assert "yfinance not installed" in capsys.readouterr().err


def test_fetch_live_empty_and_failure(monkeypatch, capsys):
    calls = _fake_yf(monkeypatch, exc=RuntimeError("boom"))
    assert db.fetch_live_quotes([]) == {}
    assert calls == []
    assert db.fetch_live_quotes([("A", "A.NS")]) == {}
    assert "live download failed (boom)" in capsys.readouterr().err
    assert calls[0][1]["period"] == "5d" and calls[0][1]["auto_adjust"] is False


def test_fetch_live_single(monkeypatch):
    data = pd.DataFrame({"Close": [np.nan, 10.0, 12.0]})
    _fake_yf(monkeypatch, data=data)
    assert db.fetch_live_quotes([("A", "A.NS")]) == {"A": (12.0, 10.0)}
    _fake_yf(monkeypatch, data=pd.DataFrame({"Close": [7.0]}))
    assert db.fetch_live_quotes([("A", "A")]) == {"A": (7.0, 7.0)}
    _fake_yf(monkeypatch, data=pd.DataFrame({"Close": [np.nan]}))
    assert db.fetch_live_quotes([("A", "A")]) == {}
    _fake_yf(monkeypatch, data=pd.DataFrame({"Close": [0.0]}))
    assert db.fetch_live_quotes([("A", "A")]) == {}  # ltp must be > 0


def test_fetch_live_multi(monkeypatch):
    cols = pd.MultiIndex.from_product([["A.NS", "B.NS", "C.NS"], ["Close"]])
    data = pd.DataFrame([[1.0, np.nan, 5.0], [2.0, np.nan, 6.0]], columns=cols)
    _fake_yf(monkeypatch, data=data)
    out = db.fetch_live_quotes([("A", "A.NS"), ("B", "B.NS"), ("C", "C.NS"), ("D", "D.NS")])
    assert out == {"A": (2.0, 1.0), "C": (6.0, 5.0)}


def test_fetch_live_bad_subframe_is_skipped(monkeypatch):
    cols = pd.MultiIndex.from_product([["A.NS", "B.NS"], ["Open"]])
    data = pd.DataFrame([[1.0, 2.0]], columns=cols)  # no Close -> KeyError swallowed
    _fake_yf(monkeypatch, data=data)
    assert db.fetch_live_quotes([("A", "A.NS"), ("B", "B.NS")]) == {}


def test_apply_live(monkeypatch):
    df = pd.DataFrame(
        [
            _row("NEAR", pos=111), _row("FAR", pos=140), _row("LOW", pos=84.9),
            _row("HI", pos=150), _row("LOW85", pos=85), _row("SHORT", dp=249),
            _row("SELL", sig="BREAKDOWN_SELL"), _row("BAD", gc="BELOW"),
            _row("GCX", gc="GOLDEN_CROSS", pos=100, suffix=".BO"),
        ]
    )  # fmt: skip
    seen = {}

    def fq(pairs):
        seen["pairs"] = pairs
        return {"NEAR": (120.0, 100.0), "FAR": (90.0, 0.0), "GCX": (75.0, 50.0)}

    monkeypatch.setattr(db, "fetch_live_quotes", fq)
    out, n, k = db.apply_live(df, "IN", 200)
    assert n == 5 and k == 3  # NEAR, FAR, HI(no quote), LOW85, GCX
    assert [p[0] for p in seen["pairs"]] == ["NEAR", "GCX", "LOW85", "FAR", "HI"]
    assert dict(seen["pairs"])["GCX"] == "GCX.BO"
    o = out.set_index("Symbol")
    assert o.loc["NEAR", "LTP"] == 120.0
    assert o.loc["NEAR", "Change%"] == pytest.approx(20.0)
    assert o.loc["NEAR", "Position_in_Box%"] == pytest.approx(140.0)
    assert o.loc["FAR", "LTP"] == 90.0 and o.loc["FAR", "Change%"] == 1.0  # prev 0 keeps
    assert o.loc["FAR", "Position_in_Box%"] == pytest.approx(80.0)
    assert o.loc["SHORT", "LTP"] == 110.0


def test_apply_live_limits_and_empty(monkeypatch):
    df = pd.DataFrame([_row("A"), _row("B", pos=100)])
    monkeypatch.setattr(db, "fetch_live_quotes", lambda p: {})
    assert db.apply_live(df, "US", 1)[1:] == (1, 0)
    assert db.apply_live(df, "US", 0)[1:] == (0, 0)
    assert db.apply_live(df, "US", -5)[1:] == (0, 0)


def test_apply_live_no_suffix_and_degenerate_box(monkeypatch):
    df = pd.DataFrame([_row("A", bt=50.0, bb=50.0), _row("B", bt=np.nan)]).drop(columns="Suffix")
    monkeypatch.setattr(db, "fetch_live_quotes", lambda p: {"A": (10.0, 0.0), "B": (10.0, 5.0)})
    out, n, k = db.apply_live(df, "IN", 10)
    assert (n, k) == (2, 2)
    assert out["Position_in_Box%"].tolist() == [110.0, 110.0]  # unchanged
    assert out.loc[1, "Change%"] == pytest.approx(100.0)


def test_build_tiers():
    df = pd.DataFrame(
        [
            _row("T1A", gc="GOLDEN_CROSS", pos=130, chg=1),
            _row("T1B", gc="GOLDEN_CROSS", pos=130.1, chg=5),
            _row("T1C", gc="GOLDEN_CROSS", pos=90, chg=3),
            _row("T2A", pos=100, chg=0.5, gap=200),
            _row("T2B", pos=120, chg=2, gap=-100),
            _row("X_LOWPOS", pos=99.9),
            _row("X_HIPOS", pos=120.1),
            _row("X_DOWN", chg=0),
            _row("X_SHORT", dp=249),
            _row("X_UNHEALTHY", gc="BELOW"),
            _row("BD1", sig="BREAKDOWN_SELL", chg=-5),
            _row("BD2", sig="BREAKDOWN_SELL", chg=-9),
        ]
    )
    t1, t2, bd = db.build(df)
    assert t1["Symbol"].tolist() == ["T1C", "T1A"]  # 130.1 excluded, sorted by Change% desc
    assert set(t2["Symbol"]) == {"T2A", "T2B"}
    s = t2.set_index("Symbol")["score"]
    assert s["T2A"] == pytest.approx(0.5 + 20 * 0.15 + 100 * 0.05)  # gap clipped to 100
    assert s["T2B"] == pytest.approx(2 + 0 + (-50) * 0.05)
    assert t2["Symbol"].tolist() == ["T2A", "T2B"]
    assert bd["Symbol"].tolist() == ["BD2", "BD1"]


def test_build_india_encoding_no_gap():
    df = pd.DataFrame([_row("IND", gc="NONE", pos=105, chg=1)]).drop(columns="DMA_Gap%")
    df["DMA50_above_200"] = True
    t1, t2, bd = db.build(df)
    assert t1.empty and bd.empty
    assert t2["Symbol"].tolist() == ["IND"]
    assert t2["score"].iloc[0] == pytest.approx(1 + 15 * 0.15)
    df["DMA50_above_200"] = False
    assert db.build(df)[1].empty


def test_rows_html_and_to_html():
    rows = pd.DataFrame(
        [_row("GD", chg=2.0, gap=5.0, ltp=1234.5, suffix=".BO"), _row("X<", chg=-1.0, gap=np.nan)]
    )
    h = db.rows_html(rows, "IN", "₹")
    assert "GENERAL DYNAMICS CORP" in h and "[GD · BSE]" in h
    assert "₹1,234.50" in h and "+2.0%" in h and "+5%" in h and "color:#1a7f37" in h
    assert "-1.0%" in h and "color:#c0392b" in h and ">n/a<" in h and "X&lt;" in h
    nogap = db.rows_html(rows.drop(columns="DMA_Gap%"), "US", "$")
    assert nogap.count(">n/a<") == 2 and "· US]" in nogap
    zero = pd.DataFrame([_row("Z", chg=0.0)])
    assert "color:#1a7f37" in db.rows_html(zero, "US", "$")
    assert db.rows_html(rows.iloc[0:0], "US", "$") == ""


def test_to_html():
    cfg = db.MARKETS["IN"]
    empty = pd.DataFrame(columns=list(_row("A")))
    bd = pd.DataFrame([_row(f"S{i}", sig="BREAKDOWN_SELL", chg=-i) for i in range(10)])
    h = db.to_html(empty, empty, bd, "scan.xlsx", cfg, "IN", top=3, live_note="<LIVE>")
    assert h.count("None today.") == 2
    assert "INDIA DARVAS BREAKOUTS" in h and "<LIVE>" in h and "(top 3)" in h
    assert "SELL side, 10 total" in h and "S7 (-7%)" in h and "S8" not in h
    t2 = pd.DataFrame([_row(f"T{i}") for i in range(5)])
    h2 = db.to_html(empty, t2, bd.iloc[0:0], "s", cfg, "IN", top=2)
    assert "[T1 ·" in h2 and "[T2 ·" not in h2 and h2.count("None today.") == 1


def _xlsx(tmp_path, rows):
    p = tmp_path / "scan.xlsx"
    with pd.ExcelWriter(p) as w:
        pd.DataFrame(rows).to_excel(w, sheet_name="Darvas_Signals", index=False)
    return p


def test_main_no_file(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sys, "argv", ["d", "--market", "IN", "--file", str(tmp_path / "nope")])
    with pytest.raises(SystemExit) as e:
        db.main()
    assert e.value.code == 0
    cap = capsys.readouterr()
    assert "No IN scan xlsx found" in cap.err and "India Darvas breakouts: no scan" in cap.out
    monkeypatch.setattr(db, "latest_scan", lambda cfg: None)
    monkeypatch.setattr(sys, "argv", ["d"])
    with pytest.raises(SystemExit):
        db.main()
    assert "US Darvas breakouts" in capsys.readouterr().out


def test_main_end_to_end(monkeypatch, capsys, tmp_path):
    p = _xlsx(
        tmp_path,
        [_row("AA", gc="GOLDEN_CROSS", pos=105), _row("BB", pos=110), _row("CC", sig="BREAKDOWN_SELL")],
    )  # fmt: skip
    csv = tmp_path / "o.csv"
    monkeypatch.setattr(sys, "argv", ["d", "--market", "IN", "--file", str(p), "--csv", str(csv)])
    db.main()
    cap = capsys.readouterr()
    assert "T1=1 T2=1 breakdowns=1" in cap.err
    out = pd.read_csv(csv)
    assert out["Tier"].tolist() == ["T1_GoldenCross", "T2_Fresh"]
    assert list(out.columns)[:2] == ["Symbol", "Suffix"]
    assert "Live prices" not in cap.out and "IN" in cap.out


def test_main_live_and_default_csv(monkeypatch, capsys, tmp_path):
    p = _xlsx(tmp_path, [_row("BB", pos=110)])
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "Downloads").mkdir()
    monkeypatch.setattr(db, "fetch_live_quotes", lambda pairs: {"BB": (101.0, 100.0)})
    monkeypatch.setattr(
        sys, "argv", ["d", "--market", "US", "--file", str(p), "--live", "--live-max", "5"]
    )
    db.main()
    cap = capsys.readouterr()
    assert "LIVE refresh: candidates=1 refreshed=1" in cap.err
    assert "Live prices as of" in cap.out and "refreshed for 1 of 1" in cap.out
    import datetime

    f = tmp_path / "Downloads" / f"us_darvas_breakouts_{datetime.date.today().isoformat()}.csv"
    assert f.exists()
