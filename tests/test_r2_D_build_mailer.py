"""Section-helper tests for build_mailer.py (offline; cwd in tmp_path)."""

import os
import sys
import types

import pandas as pd
import pytest

import build_mailer as bm


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _annot(df):
    df = df.copy()
    liq = {"TRI": "High", "MUL": "Low", "SGL": "Medium", "OTH": "Unknown"}
    df["Liquidity"] = df["Symbol"].map(liq)
    df["Turnover_USD"] = [2.5e6 if s == "TRI" else float("nan") for s in df["Symbol"]]
    return df


def test_tv():
    assert bm._tv(2.55e6) == "$2.5M" or bm._tv(2.55e6) == "$2.6M"
    assert bm._tv(float("nan")) == "—"
    assert bm._tv(None) == "—"


def test_table_structure():
    t = bm._table(["A", "B"], ["<tr><td>1</td></tr>"])
    assert t.count("<th") == 2 and "<td>1</td>" in t and t.startswith("<table")


def test_combined_report_latest(tmp_path):
    assert bm._combined_report("IN") is None
    d = tmp_path / "combined_report_results"
    d.mkdir()
    (d / "combined_IN_20260101.json").write_text('{"v": 1}')
    (d / "combined_IN_20260102.json").write_text('{"v": 2}')
    (d / "combined_US_20260105.json").write_text('{"v": 9}')
    assert bm._combined_report("IN") == {"v": 2}
    assert bm._combined_report("US") == {"v": 9}


def test_picks_rows_empty():
    assert bm._picks_rows(None, "IN") == []
    assert bm._picks_rows({}, "IN") == []
    assert bm._picks_rows({"picks": []}, "IN") == []


def test_picks_rows_sorted_and_limited(monkeypatch):
    monkeypatch.setattr(bm.liq, "annotate", _annot)
    picks = [
        {"Symbol": "SGL", "Tier": "Single-Screen", "Screens": 1},
        {"Symbol": "TRI", "Tier": "Triple Hit", "Screens": 3},
        {"Symbol": "MUL", "Tier": "Multi-Screen", "Screens": 2},
        {"Symbol": "OTH", "Tier": "Other", "Screens": 0},
    ]
    rows = bm._picks_rows({"picks": picks}, "IN")
    order = [r.split("<b>")[1].split("</b>")[0] for r in rows]
    assert order == ["TRI", "MUL", "SGL", "OTH"]
    assert "$2.5M" in rows[0] or "$2.6M" in rows[0]
    assert "—" in rows[1]
    assert "#1b7f37" in rows[0]  # High liquidity colour
    assert len(bm._picks_rows({"picks": picks}, "IN", limit=2)) == 2


def test_picks_rows_liquidity_tiebreak(monkeypatch):
    def ann(df):
        df = df.copy()
        df["Liquidity"] = ["Low", "High", "Medium"]
        df["Turnover_USD"] = 1e6
        return df

    monkeypatch.setattr(bm.liq, "annotate", ann)
    picks = [{"Symbol": s, "Tier": "Triple Hit", "Screens": 3} for s in "ABC"]
    rows = bm._picks_rows({"picks": picks}, "US")
    assert [r.split("<b>")[1][0] for r in rows] == ["B", "C", "A"]


def test_convergence_rows():
    assert bm._convergence_rows(None) == []
    assert bm._convergence_rows({"convergence": []}) == []
    d = {
        "convergence": [
            {
                "Symbol": "X",
                "Convergence": "BULLISH",
                "Fundamental": "Strong",
                "News": "Pos",
                "News_Score": 0.5,
                "Top_Headline": "h" * 100,
            },
            {
                "Symbol": "Y",
                "Convergence": "MIXED",
                "Fundamental": "Weak",
                "News": "Neg",
                "News_Score": -0.25,
            },
        ]
    }
    rows = bm._convergence_rows(d)
    assert "#1b7f37" in rows[0] and "#b8860b" in rows[1]
    assert "+0.50" in rows[0] and "-0.25" in rows[1]
    assert "h" * 60 in rows[0] and "h" * 61 not in rows[0]


def test_ai_html_variants():
    assert "no discovered screen" in bm._ai_html("India", None)
    assert "no discovered screen" in bm._ai_html("India", {"picks": []})
    base = {
        "picks": ["AAA", "BBB"],
        "rl_correction": False,
        "n": 3,
        "tape": "bull",
        "reward": 1.234,
        "overlap": 0.456,
    }
    h = bm._ai_html("India", base)
    assert "RL-corrected" not in h and "thin sample" not in h
    assert "reward 1.23" in h and "46% overlap" in h and "bull tape" in h
    assert "<b>AAA</b>, <b>BBB</b>" in h
    assert "thin sample" in bm._ai_html("US", dict(base, n=2))
    assert "thin sample" not in bm._ai_html("US", dict(base, n=3))
    assert "RL-corrected" in bm._ai_html("US", dict(base, rl_correction=True))


def test_ai_screener_section(tmp_path):
    assert bm._ai_screener_section("IN") is None
    d = tmp_path / "cache_seed" / "discovered_screens"
    d.mkdir(parents=True)
    (d / "IN.json").write_text("{}")
    s = bm._ai_screener_section("IN")
    assert s == {
        "tape": "n/a",
        "reward": 0,
        "n": 0,
        "overlap": 0,
        "rl_correction": False,
        "picks": [],
    }
    import json

    (d / "US.json").write_text(
        json.dumps(
            {
                "regime": {"tape": "bear"},
                "metrics": {"reward": 2, "n": 5, "overlap": 0.5},
                "picks": list("ABCDEFGHIJKL"),
                "rl_correction_applied": True,
            }
        )
    )
    s = bm._ai_screener_section("US")
    assert s["tape"] == "bear" and s["rl_correction"] is True and s["picks"] == list("ABCDEFGHIJ")
    (d / "XX.json").write_text("not json")
    assert bm._ai_screener_section("XX") is None


def _mk_gms(tmp_path, monkeypatch, folder, sheets, name="a.xlsx"):
    d = tmp_path / "gms" / folder
    d.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(d / name) as w:
        for k, v in sheets.items():
            v.to_excel(w, sheet_name=k, index=False)
    monkeypatch.setattr(bm, "GMS_REPO", str(tmp_path / "gms"))


def test_gms_scan_none(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "GMS_REPO", str(tmp_path / "nothing"))
    assert bm._gms_scan("US") is None


def test_gms_scan_us_all_sheets(tmp_path, monkeypatch):
    _mk_gms(
        tmp_path,
        monkeypatch,
        "us_full_scan",
        {
            "Triple_Hits": pd.DataFrame({"Symbol": ["T1"]}),
            "Multi_Screen_Hits": pd.DataFrame({"Symbol": ["M1", "M2"]}),
            "ML_Bullish": pd.DataFrame({"Symbol": ["B"], "ML_Confidence": [0.9]}),
        },
        name="old.xlsx",
    )
    _mk_gms(
        tmp_path,
        monkeypatch,
        "us_full_scan",
        {"Triple_Hits": pd.DataFrame({"Symbol": ["T2"]})},
        name="z_new.xlsx",
    )
    s = bm._gms_scan("US")
    assert s["file"] == "z_new.xlsx"  # sorted by name, last wins
    assert list(s["triple"]["Symbol"]) == ["T2"]
    assert s["multi"].empty and s["ml_bull"].empty


def test_gms_scan_india_folder_and_placeholder(tmp_path, monkeypatch):
    _mk_gms(
        tmp_path, monkeypatch, "indian_full_scan", {"Triple_Hits": pd.DataFrame({"Note": ["none"]})}
    )
    s = bm._gms_scan("IN")
    assert s["triple"].empty and s["file"] == "a.xlsx"
    # US folder doesn't exist -> None
    assert bm._gms_scan("US") is None


def test_gms_scan_corrupt(tmp_path, monkeypatch):
    d = tmp_path / "gms" / "us_full_scan"
    d.mkdir(parents=True)
    (d / "bad.xlsx").write_text("not excel")
    monkeypatch.setattr(bm, "GMS_REPO", str(tmp_path / "gms"))
    assert bm._gms_scan("US") is None


def test_gms_html():
    assert "no global-market-scanners run cached" in bm._gms_html("India", None)
    empty = pd.DataFrame()
    h = bm._gms_html("US", {"triple": empty, "multi": empty, "ml_bull": empty, "file": "f.xlsx"})
    assert (
        "f.xlsx" in h
        and "No Triple Hits today" in h
        and "Multi-Screen" not in h
        and "ML Ridge" not in h
    )
    full = {
        "triple": pd.DataFrame({"Symbol": [f"T{i}" for i in range(12)]}),
        "multi": pd.DataFrame({"Symbol": [f"M{i}" for i in range(12)]}),
        "ml_bull": pd.DataFrame(
            {
                "Symbol": list("ABCDEFG"),
                "ML_Confidence": [1, 2, 3, 4, 5, 6, 7],
                "ML_Pred_Ret%": [1.0, 2.0, 3.0, 4.0, 5.0, float("nan"), 7.0],
            }
        ),
        "file": "g.xlsx",
    }
    h = bm._gms_html("US", full)
    assert "<b>T9</b>" in h and "T10" not in h
    assert "top 10 of 12" in h and "<b>M9</b>" in h and "M10" not in h
    assert "<b>G</b> (+7.0%)" in h and "<b>F</b>," in h  # NaN pred -> bare symbol
    assert "<b>C</b>" in h and "<b>B</b>" not in h  # only top 5 by confidence
    nopred = dict(full, ml_bull=pd.DataFrame({"Symbol": ["A"], "ML_Confidence": [1.0]}))
    assert "<b>A</b>" in bm._gms_html("US", nopred)


def test_europe_scan(tmp_path):
    assert bm._europe_scan() is None
    d = tmp_path / "european_scan"
    d.mkdir()
    old = d / "european_market_scan_zzz.xlsx"
    new = d / "european_market_scan_aaa.xlsx"
    for p, n in ((old, 1), (new, 3)):
        with pd.ExcelWriter(p) as w:
            pd.DataFrame({"Symbol": ["a"] * n}).to_excel(w, sheet_name="All_Stocks", index=False)
    os.utime(old, (1, 1))
    os.utime(new, (2_000_000_000, 2_000_000_000))
    s = bm._europe_scan()
    assert s["file"] == new.name  # newest mtime, not filename order
    assert s["n_scanned"] == 3 and s["triple"].empty and s["darvas"].empty
    (d / "european_market_scan_bad.xlsx").write_text("junk")
    os.utime(d / "european_market_scan_bad.xlsx", (3_000_000_000, 3_000_000_000))
    assert bm._europe_scan() is None


def test_europe_html():
    assert "No Europe scan cached" in bm._europe_html(None)
    e = pd.DataFrame()
    h = bm._europe_html({"triple": e, "darvas": e, "n_scanned": 7, "file": "f"})
    assert "7 stocks scanned" in h and "No Triple Hits today" in h and "Darvas breakouts" not in h
    tri = pd.DataFrame(
        {
            "Symbol": ["AB"],
            "Name": ["Abc"],
            "Sector": ["Tech"],
            "Piotroski_Score": [8],
            "CoffeeCan_Score": [5],
            "LTP": [12.345],
        }
    )
    dar = pd.DataFrame({"Symbol": [f"D{i}" for i in range(10)], "Upside_to_Top%": list(range(10))})
    h = bm._europe_html({"triple": tri, "darvas": dar, "n_scanned": 1, "file": "f"})
    assert "8/9" in h and "€12.35" in h and "10 Darvas breakouts" in h
    assert "<b>D9</b>, <b>D8</b>" in h and "D1</b>" not in h.split("upside-to-box-top:")[1]
    tri2 = tri.drop(columns=["Piotroski_Score", "CoffeeCan_Score"])
    assert "/9" in bm._europe_html({"triple": tri2, "darvas": e, "n_scanned": 1, "file": "f"})


def test_regime_section(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "regime_picks", types.SimpleNamespace(section_html=lambda: "<S>")
    )
    assert bm._regime_section() == "<S>"

    def boom():
        raise RuntimeError

    monkeypatch.setitem(sys.modules, "regime_picks", types.SimpleNamespace(section_html=boom))
    assert bm._regime_section() == ""


def test_verification_blocks_ok(monkeypatch):
    seen = {}
    rp = types.SimpleNamespace(picks_for=lambda m: [{"symbol": m + "1", "price": 1.5, "extra": 1}])
    vd = types.SimpleNamespace(
        verify_us=lambda p: seen.setdefault("us", p) or "US",
        verify_india=lambda p: seen.setdefault("in", p) or "IN",
        verification_html=lambda a, b: f"H{a}{b}",
        verification_text=lambda a, b: f"T{a}{b}",
    )
    monkeypatch.setitem(sys.modules, "regime_picks", rp)
    monkeypatch.setitem(sys.modules, "verify_data", vd)
    h, t = bm._verification_blocks()
    assert seen["us"] == [{"symbol": "US1", "price": 1.5}]
    assert seen["in"] == [{"symbol": "IN1", "price": 1.5}]
    assert h.startswith("H[") and t.startswith("T[")


def test_verification_blocks_failure(monkeypatch):
    def boom(m):
        raise RuntimeError("nope")

    monkeypatch.setitem(sys.modules, "regime_picks", types.SimpleNamespace(picks_for=boom))
    monkeypatch.setitem(sys.modules, "verify_data", types.SimpleNamespace())
    h, t = bm._verification_blocks()
    assert "unavailable: nope" in h and t == "Data verification unavailable."


def _ccc(tmp_path, df):
    (tmp_path / "cache_seed").mkdir(exist_ok=True)
    df.to_parquet(tmp_path / "cache_seed" / "india_ccc_screen.parquet")


def test_ccc_rows_missing_file():
    assert bm._ccc_rows() == []


def test_ccc_rows_filters_and_sorts(tmp_path):
    _ccc(
        tmp_path,
        pd.DataFrame(
            {
                "Symbol": list("ABCDE"),
                "Cash_Cycle": ["5", "-2", "x", "9", "1"],
                "Liquidity": ["High", "Medium", "High", "Low", "Medium"],
                "ROCE": [1, 2, 3, 4, 5],
            }
        ),
    )
    rows = bm._ccc_rows()
    syms = [r.split("<b>")[1].split("</b>")[0] for r in rows]
    # D (Low liquidity) dropped; C (NaN cycle, High liquidity) sorted last
    assert syms == ["B", "E", "A", "C"]
    assert "-2.0" in rows[0]


def test_ccc_rows_nan_last_and_limit(tmp_path):
    n = 15
    _ccc(
        tmp_path,
        pd.DataFrame(
            {
                "Symbol": [f"S{i}" for i in range(n)],
                "Cash_Cycle": [str(i) for i in range(n)],
                "Liquidity": ["High"] * n,
            }
        ),
    )
    rows = bm._ccc_rows()
    assert len(rows) == 10 and "<b>S0</b>" in rows[0] and "<b>S9</b>" in rows[9]


def test_ccc_rows_annotates_when_no_liquidity(tmp_path, monkeypatch):
    _ccc(tmp_path, pd.DataFrame({"Symbol": ["A", "B"], "Cash_Cycle": ["1", "2"]}))
    got = {}

    def ann(df):
        got["m"] = list(df["Market"])
        df = df.copy()
        df["Liquidity"] = ["High", "Low"]
        return df

    monkeypatch.setattr(bm.liq, "annotate", ann)
    rows = bm._ccc_rows()
    assert got["m"] == ["IN", "IN"] and len(rows) == 1 and "<b>A</b>" in rows[0]


def test_momentum_rows_top15():
    df = pd.DataFrame(
        {
            "Market": ["US"] * 20,
            "Symbol": [f"S{i}" for i in range(20)],
            "ret_126": [float(i) for i in range(20)],
            "rsi14": [50] * 20,
        }
    )
    rows = bm._momentum_rows(df)
    assert len(rows) == 15 and "S0" in rows[0] and "S14" in rows[14] and "+0" in rows[0]


def test_other_market_rows(monkeypatch):
    def ann(df):
        df = df.copy()
        df["Liquidity"] = "High"
        return df

    monkeypatch.setattr(bm.liq, "annotate", ann)
    df = pd.DataFrame(
        {
            "Market": ["KR", "KR", "JP", "IN", "ZZ"],
            "Symbol": ["k1", "k2", "j1", "i1", "z1"],
            "ret_126": [10.0, 30.0, 5.0, 99.0, 99.0],
            "rsi14": [1] * 5,
        }
    )
    rows = bm._other_market_rows(df)
    text = "".join(rows)
    assert len(rows) == 2
    assert "Korea" in text and "<b>k2</b>" in text and "+30%" in text and "k1" not in text
    assert "Japan" in text and "i1" not in text and "z1" not in text
    # order follows MARKET_NAMES (KR before JP)
    assert text.index("Korea") < text.index("Japan")


def test_other_market_rows_failure(monkeypatch):
    def boom(df):
        raise RuntimeError

    monkeypatch.setattr(bm.liq, "annotate", boom)
    assert bm._other_market_rows(pd.DataFrame({"Market": ["KR"]})) == []


def test_scoreboard_rows(monkeypatch):
    monkeypatch.setattr(
        bm.mp,
        "load",
        lambda: pd.DataFrame(
            {
                "Market": ["IN", "US"],
                "Index": ["N", "S"],
                "CAGR%": [1.5, 2.5],
                "Return_1y%": [3, 4],
                "Sharpe": [0.5, 0.6],
            }
        ),
    )
    rows = bm._scoreboard_rows()
    assert len(rows) == 2 and "<td>1.5</td>" in rows[0] and "<td>0.6</td>" in rows[1]
