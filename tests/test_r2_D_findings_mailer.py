"""Offline tests for screener_findings_mailer.py (pure HTML assembly; send path is faked)."""

import datetime as dt
import runpy
import sys
import types

import pytest

import screener_findings_mailer as sfm


class _D(dt.date):
    @classmethod
    def today(cls):
        return cls(2026, 5, 6)


@pytest.fixture(autouse=True)
def _fixed_date(monkeypatch):
    monkeypatch.setattr(sfm, "date", _D)


def test_table_structure_and_empty():
    t = sfm._table(["A", "B"], [["1", "2"], ["3", "4"]])
    assert t.count("<th") == 2 and t.count("<tr>") == 2 and t.count("<td") == 4
    assert "<td" in t and ">3</td>" in t and t.endswith("</table>")
    e = sfm._table(["A"], [])
    assert "<tr>" not in e and "<th" in e


def test_regime_rows_colour_and_format():
    rows = sfm._regime_rows()
    assert len(rows) == len(sfm.REGIME_MAP)
    for r, (mkt, flag, screen, edge, note, regime, why) in zip(rows, sfm.REGIME_MAP):
        assert r[0] == f"{flag} <b>{mkt}</b>" and r[1] == f"<code>{screen}</code>" and r[4] == why
        assert f"{edge:+.2f}pp" in r[2]
        assert ("#2e7d32" in r[2]) == (edge > 0) and ("#bf360c" in r[2]) == (edge <= 0)
        assert note in r[3]


def test_regime_rows_zero_edge_is_not_green(monkeypatch):
    monkeypatch.setattr(
        sfm,
        "REGIME_MAP",
        [
            ("XX", "F", "s", 0.0, "n", "r", "w"),
            ("YY", "F", "s", -1.0, "n", "r", "w"),
            ("ZZ", "F", "s", 0.01, "n", "r", "w"),
        ],
    )
    r = sfm._regime_rows()
    assert "#bf360c" in r[0][2] and "+0.00pp" in r[0][2]
    assert "#bf360c" in r[1][2] and "-1.00pp" in r[1][2]
    assert "#2e7d32" in r[2][2]


def test_fund_rows():
    rows = sfm._fund_rows()
    assert len(rows) == len(sfm.US_FUNDAMENTAL) == 6
    assert rows[0][0] == "<code>net_net</code>" and "+12.4pp" in rows[0][2] and rows[0][3] == "60%"
    assert "+0.2pp" in rows[-1][2] and rows[-1][3] == "52%"
    assert all("#2e7d32" in r[2] for r in rows)


def test_regime_map_data_sanity():
    assert len({m[0] for m in sfm.REGIME_MAP}) == len(sfm.REGIME_MAP)  # unique markets
    assert all(len(m) == 7 for m in sfm.REGIME_MAP)
    for key in ("US", "IN", "KR", "CN", "JP"):
        assert any(m[0] == key for m in sfm.REGIME_MAP)


def test_build_outputs():
    subj, text, html = sfm.build()
    assert subj == "🔬 Screener Backtest Briefing — best screen per market — 2026-05-06"
    assert text.startswith("Screener Backtest Briefing — 2026-05-06\n")
    assert "NOT investment advice" in text and "NOT investment advice" in html
    assert "Screener Backtest Briefing — 2026-05-06</h1>" in html
    assert html.count("<table") == 2
    for m in sfm.REGIME_MAP:
        assert f"<b>{m[0]}</b>" in html
    for slug, *_ in sfm.US_FUNDAMENTAL:
        assert f"<code>{slug}</code>" in html
    assert "{BLUE}" not in html and "{regime_rows}" not in html
    assert sfm.BLUE in html and "{today}" not in html


@pytest.mark.parametrize(
    "send_result,expected",
    [(True, "sent"), (False, "not sent (no creds)"), ("raise", "send failed (boom)")],
)
def test_main_send(monkeypatch, tmp_path, capsys, send_result, expected):
    monkeypatch.chdir(tmp_path)
    got = {}
    m = types.ModuleType("send_mailer")

    def send(s, t, h):
        got["args"] = (s, t, h)
        if send_result == "raise":
            raise RuntimeError("boom")
        return send_result

    m.send = send
    monkeypatch.setitem(sys.modules, "send_mailer", m)
    monkeypatch.setattr(sys, "argv", ["x", "--send"])
    runpy.run_path(sfm.__file__, run_name="__main__")
    out = capsys.readouterr().out
    assert expected in out and "html bytes:" in out
    written = (tmp_path / "screener_findings.html").read_text()
    if send_result != "raise":
        assert got["args"][2] == written
    else:
        assert "saving html" in out


def test_main_without_send(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    m = types.ModuleType("send_mailer")
    m.send = lambda *a: (_ for _ in ()).throw(AssertionError("must not send"))
    monkeypatch.setitem(sys.modules, "send_mailer", m)
    monkeypatch.setattr(sys, "argv", ["x"])
    runpy.run_path(sfm.__file__, run_name="__main__")
    out = capsys.readouterr().out
    assert "→ screener_findings.html" in out and "sent" not in out.replace("html bytes", "")
    assert (tmp_path / "screener_findings.html").exists()
