"""Extra coverage for morning_mailer.py and daily_brief.py (offline, tmp_path redirected)."""

import datetime as dt
import sys
import types

import numpy as np
import pandas as pd
import pytest

import daily_brief as db
import morning_mailer as mm


@pytest.fixture(autouse=True)
def _redirect(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "RESULTS", tmp_path / "res")
    (tmp_path / "res").mkdir()
    monkeypatch.setattr(mm, "COMBINED_DIR", tmp_path / "comb")
    monkeypatch.setattr(db, "RESULTS", tmp_path / "res")


class _D(dt.date):
    @classmethod
    def today(cls):
        return cls(2026, 3, 4)


def _mod(monkeypatch, name, **kw):
    m = types.ModuleType(name)
    for k, v in kw.items():
        setattr(m, k, v)
    monkeypatch.setitem(sys.modules, name, m)


def _boom(*a, **k):
    raise RuntimeError("boom")


# ---------------- morning_mailer ----------------


def test_latest_combined(tmp_path):
    assert mm._latest_combined("IN") is None  # dir missing
    (tmp_path / "comb").mkdir()
    assert mm._latest_combined("IN") is None  # no frags
    (tmp_path / "comb" / "combined_IN_20260101.html").write_text("a")
    (tmp_path / "comb" / "combined_IN_20260102.html").write_text("b")
    (tmp_path / "comb" / "combined_US_20260109.html").write_text("c")
    assert mm._latest_combined("IN").name == "combined_IN_20260102.html"


def test_assemble_combined(tmp_path, monkeypatch):
    monkeypatch.setattr(mm._dt, "date", _D)
    assert mm.assemble_combined(["IN"]) is None
    (tmp_path / "comb").mkdir()
    (tmp_path / "comb" / "combined_IN_1.html").write_text("<i>IN-FRAG</i>")
    (tmp_path / "comb" / "combined_XX_1.html").write_text("<i>XX-FRAG</i>")
    subj, text, html = mm.assemble_combined(["IN", "US", "XX"])
    assert subj == "📈 Daily Combined Report — 2026-03-04"
    assert (
        "<i>IN-FRAG</i>" in html
        and "<h2 style='border-top:2px solid #ccc;padding-top:8px'>🇮🇳 India</h2>" in html
    )
    assert (
        "<h2 style='border-top:2px solid #ccc;padding-top:8px'>XX</h2>" in html
    )  # unknown market uses raw code
    assert "United States" not in html  # US has no fragment
    assert "🇮🇳 India — see HTML view. (source: combined_IN_1.html)" in text
    assert text.endswith("Educational/research only. NOT investment advice.")
    assert html.endswith("</body></html>")


def test_top_buys(monkeypatch):
    seen = {}

    def rec(m, top, verbose):
        seen.update(m=m, top=top, verbose=verbose)
        return {"picks": list("ABCDEFGHIJ")}

    _mod(monkeypatch, "auto_screener", recommend=rec)
    assert mm._top_buys("IN", n=3) == ["A", "B", "C"]
    assert seen == {"m": "IN", "top": 3, "verbose": False}
    assert mm._top_buys("IN") == list("ABCDEFGH")
    _mod(monkeypatch, "auto_screener", recommend=lambda *a, **k: {})
    assert mm._top_buys("IN") == []
    _mod(monkeypatch, "auto_screener", recommend=_boom)
    assert mm._top_buys("IN") == []


def test_assemble_combined_preferred_and_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(mm._dt, "date", _D)
    _mod(monkeypatch, "daily_brief_x")
    _mod(monkeypatch, "daily_brief", build_brief=lambda m: "BRIEF " + ",".join(m))
    _mod(
        monkeypatch,
        "watchlist_delta",
        diff=lambda m: (
            (None, {"new_buys": 2, "downgrades": 1, "new_flags": 0})
            if m == "IN"
            else (None, {"note": "no prior"})
        ),
    )
    monkeypatch.setattr(mm, "_top_buys", lambda m: ["T1", "T2"] if m == "IN" else [])
    subj, text, html = mm.assemble(fmt="brief")  # default markets IN, US
    assert subj == "📈 Morning Market Brief — 2026-03-04"
    assert "BRIEF IN,US" in text
    assert "- **IN**: 2 new BUYs · 1 downgrades · 0 new risk flags" in text
    assert "US**: " not in text.split("## Top BUY")[0]
    assert "- **IN**: T1, T2" in text and "(none)" not in text
    assert "<h3>Watchlist changes</h3>" in html and "<li>" in html
    # combined preferred when fragment exists
    (tmp_path / "comb").mkdir()
    (tmp_path / "comb" / "combined_IN_1.html").write_text("F")
    assert mm.assemble(["IN"])[0].startswith("📈 Daily Combined")
    assert mm.assemble(["IN"], fmt="brief")[0].startswith("📈 Morning")
    # combined requested but no fragments -> brief
    assert mm.assemble(["US"])[0].startswith("📈 Morning")


def test_assemble_failures(monkeypatch):
    _mod(monkeypatch, "daily_brief", build_brief=_boom)
    _mod(monkeypatch, "watchlist_delta", diff=_boom)
    monkeypatch.setattr(mm, "_top_buys", lambda m: [])
    _, text, _ = mm.assemble(["IN"], fmt="brief")
    assert "_brief unavailable_" in text
    assert "- (no prior snapshot)" in text and "- (none)" in text


def test_md_to_html_table_and_lists():
    md = "# T\n## S\n- a **b**\n- c\n\ntext\n| A | B |\n|---|:--|\n| 1 | 2 |\n| 3 | 4 |\nafter"
    h = mm._md_to_html(md)
    assert "<h2>T</h2>" in h and "<h3>S</h3>" in h
    assert "<ul>\n<li>a b</li>\n<li>c</li>\n</ul>" in h  # ** stripped, single ul
    assert "<tr><th>A</th><th>B</th></tr>" in h  # first row is header
    assert "<tr><td>1</td><td>2</td></tr>" in h and "<tr><td>3</td><td>4</td></tr>" in h
    assert "---" not in h  # separator dropped
    assert h.count("<table") == 1 and "</table>\n<p>after</p>" in h
    assert "<p>text</p>" in h
    assert h.startswith("<html><body") and h.endswith("</body></html>")


def test_md_to_html_unclosed_endings():
    assert "</ul>" in mm._md_to_html("- x")
    assert mm._md_to_html("|a|b|\n|c|d|").count("</table>") == 1
    # a lone pipe line is not a table row
    assert "<table" not in mm._md_to_html("| only")


def test_send_dry(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mm, "assemble", lambda m, fmt: ("S", "TXT", "<h/>"))
    assert mm.send(["IN"], dry=True) is False
    assert (tmp_path / "res" / "morning_brief.html").read_text() == "<h/>"
    assert (tmp_path / "res" / "morning_brief.md").read_text() == "TXT"
    out = capsys.readouterr().out
    assert "TXT" in out and "(dry run)" in out
    mm.send(["IN"], dry=True, verbose=False)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("ok,msg", [(True, "sent ✓"), (False, "no GMAIL creds")])
def test_send_live(monkeypatch, capsys, ok, msg):
    monkeypatch.setattr(mm, "assemble", lambda m, fmt: ("S", "T", "H"))
    got = {}
    _mod(monkeypatch, "send_mailer", send=lambda s, t, h: got.update(a=(s, t, h)) or ok)
    assert mm.send(["IN"]) is ok
    assert got["a"] == ("S", "T", "H") and msg in capsys.readouterr().out
    assert mm.send(["IN"], verbose=False) is ok
    assert capsys.readouterr().out == ""


def test_send_error(monkeypatch, capsys):
    monkeypatch.setattr(mm, "assemble", lambda m, fmt: ("S", "T", "H"))

    def bad(*a):
        raise RuntimeError("x" * 80)

    _mod(monkeypatch, "send_mailer", send=bad)
    assert mm.send(["IN"]) is False
    out = capsys.readouterr().out
    assert "send failed (" + "x" * 50 + ")" in out
    assert mm.send(["IN"], verbose=False) is False


@pytest.mark.parametrize(
    "argv,exp",
    [
        ([], (None, False, "combined")),
        (["--market", "IN", "US", "--dry", "--format", "brief"], (["IN", "US"], True, "brief")),
    ],
)
def test_morning_main(monkeypatch, argv, exp):
    got = {}
    monkeypatch.setattr(
        mm, "send", lambda m, dry=False, fmt="combined": got.update(a=(m, dry, fmt))
    )
    monkeypatch.setattr(sys, "argv", ["morning_mailer.py"] + argv)
    assert mm.main() == 0
    assert got["a"] == exp


# ---------------- daily_brief ----------------


@pytest.mark.parametrize(
    "b,r,exp",
    [
        (0.56, 1, "Bull"),
        (0.55, 1, "Neutral"),
        (0.9, 0, "Neutral"),
        (0.9, -0.1, "Neutral"),
        (0.34, 5, "Bear"),
        (0.35, 5, "Neutral"),
        (0.5, -5.1, "Bear"),
        (0.5, -5, "Neutral"),
    ],
)
def test_tape_boundaries(b, r, exp):
    assert db._tape(b, r) == exp


def _view(sym, ret, above, liq, last="2026-03-01"):
    return pd.DataFrame(
        {"Symbol": sym, "Ret252": ret, "Above200DMA": above, "Liquidity": liq, "LastDate": last}
    )


def test_build_brief_rows_and_order(monkeypatch):
    views = {
        "IN": _view(
            list("ABCD"), [50.0, 40, 30, 20], [1, 1, 1, 0], ["High", "Low", "Medium", "High"]
        ),
        "US": _view(list("EF"), [-10.0, -20], [0, 0], ["High", "High"], last="2026-03-05"),
        "JP": _view(list("GH"), [1.0, 2], [1, 0], ["Low", "Low"]),
        "EMPTY": pd.DataFrame(),
    }
    monkeypatch.setattr(db.sl, "serving", lambda m: views[m])
    out = db.build_brief(["US", "JP", "IN", "EMPTY"])
    assert out.startswith("# Daily Market Brief — 2026-03-05")
    assert "**Tape:** 1 Bull · 1 Neutral · 1 Bear" in out
    assert "Bull: IN  |  Bear: US" in out
    lines = [l for l in out.splitlines() if l.startswith("| ") and "Market" not in l]
    assert [l.split("|")[1].strip() for l in lines] == ["IN", "JP", "US"]  # breadth desc
    inl = lines[0]
    assert "| 4 |" in inl and "75%" in inl and "| 35% |" in inl and "Bull" in inl
    assert inl.rstrip().endswith("A, C, D |")  # Low-liquidity B excluded, ranked by Ret252
    assert out.endswith("_Educational/research only. Not investment advice._")


@pytest.mark.xfail(
    strict=True,
    raises=KeyError,
    reason="daily_brief.py:53 guards a missing Ret252 column but line 55 then sorts by it unguarded",
)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_build_brief_missing_columns(monkeypatch):
    df = pd.DataFrame({"Symbol": ["A", "B"], "LastDate": ["2026-01-01", "2026-01-02"]})
    monkeypatch.setattr(db.sl, "serving", lambda m: df)
    out = db.build_brief(["IN"])
    assert "n/a" in out and "0%" in out and "Neutral" in out
    assert "# Daily Market Brief — 2026-01-02" in out


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_build_brief_nan_medret_sorts_last(monkeypatch):
    views = {
        "A": _view(["x"], [np.nan], [1], ["High"]),
        "B": _view(["y"], [3.0], [1], ["High"]),
    }
    monkeypatch.setattr(db.sl, "serving", lambda m: views[m])
    out = db.build_brief(["A", "B"])
    rows = [
        l.split("|")[1].strip()
        for l in out.splitlines()
        if l.startswith("| ") and "Market" not in l
    ]
    assert rows == ["B", "A"]


def test_build_brief_empty(monkeypatch):
    monkeypatch.setattr(db.sl, "serving", lambda m: pd.DataFrame())
    monkeypatch.setattr(db._dt, "date", _D)
    out = db.build_brief(["IN"])
    assert out.startswith("# Daily Market Brief — 2026-03-04")
    assert "No serving views built" in out


def test_build_brief_default_markets(monkeypatch):
    seen = []
    monkeypatch.setattr(db.sl, "serving", lambda m: seen.append(m) or pd.DataFrame())
    db.build_brief()
    assert seen == list(db.datalink.MARKETS)


def test_write_brief(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(db, "build_brief", lambda m=None: "BODY")
    p = db.write_brief(["IN"])
    assert p == tmp_path / "res" / "daily_brief.md" and p.read_text() == "BODY"
    assert "BODY" in capsys.readouterr().out
    db.write_brief(verbose=False)
    assert capsys.readouterr().out == ""


def test_mail_brief(monkeypatch):
    got = {}
    _mod(monkeypatch, "morning_mailer", send=lambda m, verbose: got.update(a=(m, verbose)) or True)
    assert db.mail_brief(["IN"]) is True and got["a"] == (["IN"], False)
    _mod(monkeypatch, "morning_mailer", send=_boom)
    assert db.mail_brief() is False


@pytest.mark.parametrize(
    "argv,mailed,msg",
    [([], None, ""), (["--mail"], True, "mailed ✓"), (["--mail"], False, "mail skipped")],
)
def test_daily_main(monkeypatch, capsys, argv, mailed, msg):
    called = []
    monkeypatch.setattr(db, "write_brief", lambda m: called.append(m))
    monkeypatch.setattr(db, "mail_brief", lambda m: mailed)
    monkeypatch.setattr(
        sys, "argv", ["daily_brief.py"] + argv + (["--market", "US"] if argv else [])
    )
    assert db.main() == 0
    assert called == ([["US"]] if argv else [None])
    out = capsys.readouterr().out
    assert (msg in out) if msg else ("mail" not in out)
