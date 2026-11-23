"""Offline tests for repo_dashboard section builders, assemble/send/main."""

import datetime as dt
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

import repo_dashboard as rd


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "HERE", tmp_path)
    monkeypatch.setattr(rd, "RESULTS", tmp_path / "results")
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(rd, "COMBINED_DIR", tmp_path / "combined")
    return tmp_path


def fake_mod(monkeypatch, name, **attrs):
    m = ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    monkeypatch.setitem(sys.modules, name, m)
    return m


def _status(rows):
    return pd.DataFrame(rows, columns=["market", "tier", "symbols", "start", "end", "rows"])


# ---- _safe / _card ----
def test_safe_truncates_error_message():
    def boom():
        raise ValueError("x" * 200)

    out = rd._safe(boom, "fb")
    assert out.startswith("<p style='color:#999'>fb <i>(")
    assert "x" * 80 in out and "x" * 81 not in out
    assert rd._safe(lambda: "ok", "fb") == "ok"


def test_card():
    c = rd._card("T", "<b>B</b>")
    assert "<h2 style='margin:0 0 10px;font-size:17px'>T</h2><b>B</b></div>" in c


# ---- _kpis ----
def test_kpis_all_sources(env, monkeypatch):
    (env / "tests").mkdir()
    for n in ("test_a.py", "test_b.py", "other.py"):
        (env / "tests" / n).write_text("")
    (env / "cache_seed").mkdir()
    (env / "cache_seed" / "CHECKSUMS.sha256").write_text("a  x\n\nb  y\n   \nc  z\n")
    fake_mod(
        monkeypatch,
        "governance",
        run=lambda verbose: {"r1": (True, ""), "r2": (False, "x"), "r3": (True, "")},
    )
    fake_mod(
        monkeypatch,
        "backlog",
        _load=lambda: [
            {"status": "done"},
            {"status": "done"},
            {"status": "todo"},
            {"status": "doing"},
            {},
        ],
    )
    html, m = rd._kpis(["IN", "US", "JP"])
    assert m == {
        "gov_pass": 2,
        "gov_total": 3,
        "gov_fail": ["r2"],
        "tests": 2,
        "checksums": 3,
        "done": 2,
        "todo": 1,
        "doing": 1,
    }
    assert ">2/3<" in html and "#c62828" in html and "#2e7d32" in html
    assert ">Governance<" in html and ">Integrity files<" in html
    assert html.count("display:inline-block") == 6
    assert (
        "<div style='font-size:22px;font-weight:700'>3</div><div style='font-size:11px;opacity:.9'>Markets"
        in html
    )


def test_kpis_governance_green(env, monkeypatch):
    fake_mod(monkeypatch, "governance", run=lambda verbose: {"a": (True, "")})
    fake_mod(monkeypatch, "backlog", _load=lambda: [])
    html, m = rd._kpis(["IN"])
    assert m["gov_pass"] == m["gov_total"] == 1
    assert "#c62828" not in html


def test_kpis_all_fail_gracefully(env, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError

    fake_mod(monkeypatch, "governance", run=boom)
    fake_mod(monkeypatch, "backlog", _load=boom)
    html, m = rd._kpis(["IN"])  # no tests dir, no manifest
    assert m == {
        "gov_pass": 0,
        "gov_total": 0,
        "gov_fail": [],
        "tests": 0,
        "checksums": 0,
        "done": 0,
        "todo": 0,
        "doing": 0,
    }
    assert ">0/0<" in html and "#c62828" not in html  # 0 == 0 counts as green


# ---- _data_coverage ----
def test_data_coverage(monkeypatch):
    fake_mod(
        monkeypatch,
        "market_memory",
        status=lambda markets: _status(
            [["IN", "stm", 10, "2024-01-01", "2024-02-01", 1234567], ["US", "ltm", 5, "a", "b", 0]]
        ),
    )
    h = rd._data_coverage(["IN", "US"])
    assert h.startswith("<table") and h.endswith("</table>")
    assert (
        "<td><b>IN</b></td><td>STM</td><td>10</td><td>2024-01-01</td><td>2024-02-01</td><td>1,234,567</td>"
        in h
    )
    assert "<td>LTM</td>" in h and "<td>0</td></tr>" in h
    assert h.count("<tr>") == 3


# ---- _market_tape ----
def test_market_tape(monkeypatch):
    got = {}
    fake_mod(monkeypatch, "daily_brief", build_brief=lambda mk: got.setdefault("m", mk) and "BRIEF")
    fake_mod(monkeypatch, "morning_mailer", _md_to_html=lambda s: f"<x>{s}</x>")
    assert rd._market_tape(["IN"]) == "<x>BRIEF</x>"
    assert got["m"] == ["IN"]


# ---- _stock_lists ----
def test_stock_lists_combined(env):
    d = env / "combined"
    d.mkdir()
    (d / "combined_IN_20240101.html").write_text("OLD")
    (d / "combined_IN_20240201.html").write_text("NEW")
    (d / "combined_US_1.html").write_text("USX")
    (d / "combined_JP_1.html").write_text("JPX")
    out = rd._stock_lists(["IN", "US", "JP", "KR"])
    assert "🇮🇳 India</h3>NEW" in out and "OLD" not in out
    assert "🇺🇸 United States</h3>USX" in out
    assert ">JP</h3>JPX" in out
    assert "KR" not in out


def test_stock_lists_fallback_auto_screener(env, monkeypatch):
    calls = []

    def rec(mk, top, verbose):
        calls.append((mk, top, verbose))
        return {"picks": [f"{mk}{i}" for i in range(15)]} if mk == "IN" else {}

    fake_mod(monkeypatch, "auto_screener", recommend=rec)
    out = rd._stock_lists(["IN", "US"])
    assert calls == [("IN", 10, False), ("US", 10, False)]
    assert "<p><b>IN</b>: IN0, IN1, IN2, IN3, IN4, IN5, IN6, IN7, IN8, IN9</p>" in out
    assert "IN10" not in out
    assert "<p><b>US</b>: —</p>" in out


def test_stock_lists_dir_exists_but_no_matches(env, monkeypatch):
    (env / "combined").mkdir()
    fake_mod(monkeypatch, "auto_screener", recommend=lambda mk, top, verbose: {"picks": ["Z"]})
    assert rd._stock_lists(["IN"]) == "<p><b>IN</b>: Z</p>"


# ---- _backlog_table ----
def _bk(monkeypatch, items):
    fake_mod(
        monkeypatch,
        "backlog",
        _load=lambda: items,
        ranked=lambda its, status: [i for i in its if i.get("status") == status],
        wsjf=lambda i: i.get("w", 1.5),
    )


def test_backlog_table_clear(monkeypatch):
    _bk(monkeypatch, [{"status": "done", "id": 1, "title": "t"}])
    assert rd._backlog_table() == "<p>✅ Backlog clear — 0 todo / 0 doing.</p>"


def test_backlog_table_rows_and_cap(monkeypatch):
    items = [{"status": "doing", "id": "D1", "title": "doing it", "epic": "E"}]
    items += [{"status": "todo", "id": f"T{i}", "title": f"t{i}", "w": i} for i in range(15)]
    _bk(monkeypatch, items)
    h = rd._backlog_table()
    assert h.index("D1") < h.index("T0")  # doing first
    assert "<td>1.50</td><td>D1</td><td>doing it</td><td>doing</td><td>E</td>" in h
    assert "T11" in h and "T12" not in h  # 12 todo max
    assert "<td>2.00</td><td>T2</td><td>t2</td><td>todo</td><td></td>" in h
    assert h.count("<tr>") == 1 + 1 + 12


def test_backlog_table_only_doing(monkeypatch):
    _bk(monkeypatch, [{"status": "doing", "id": "D", "title": "x"}])
    assert "<td>D</td>" in rd._backlog_table()


# ---- _challenges ----
def _mm(monkeypatch, rows):
    fake_mod(monkeypatch, "market_memory", status=lambda markets: _status(rows))


def test_challenges_stale_boundaries(monkeypatch):
    today = dt.date.today()
    rows = [
        ["IN", "stm", 1, "s", (today - dt.timedelta(days=5)).isoformat(), 10],
        ["US", "stm", 1, "s", (today - dt.timedelta(days=6)).isoformat(), 10],
        ["JP", "stm", 1, "s", (today - dt.timedelta(days=30)).isoformat() + "T00:00", 10],
        ["KR", "ltm", 1, "s", "2000-01-01", 10],
        ["CN", "stm", 0, "s", "2000-01-01", 0],
        ["XX", "stm", 1, "s", "garbage", 10],
    ]
    _mm(monkeypatch, rows)
    out = rd._challenges(["IN"], {})
    assert "<b>IN</b> data is" not in out  # exactly 5 days: not stale
    assert "<b>US</b> data is 6 days stale" in out
    assert "<b>JP</b> data is 30 days stale" in out
    assert "KR" not in out  # non-stm tier skipped
    assert "No data for <b>CN</b> (STM empty)" in out
    assert "XX" not in out  # unparsable date swallowed
    assert out.startswith("<ul>") and out.endswith("</ul>")


def test_challenges_governance_and_backlog_order(monkeypatch):
    _mm(monkeypatch, [])
    out = rd._challenges([], {"gov_fail": ["a", "b"], "todo": 3})
    assert out.index("<b>a</b>") < out.index("<b>b</b>") < out.index("3 backlog item(s)")
    assert out.count("<li>") == 3
    out2 = rd._challenges([], {"todo": 0, "gov_fail": []})
    assert out2 == "<p>✅ No open challenges — all green.</p>"


def test_challenges_market_status_failure(monkeypatch):
    def boom(m):
        raise RuntimeError

    fake_mod(monkeypatch, "market_memory", status=boom)
    assert rd._challenges(["IN"], {}) == "<p>✅ No open challenges — all green.</p>"


# ---- _recent_activity ----
def test_recent_activity(env, monkeypatch):
    seen = {}

    def run(cmd, **kw):
        seen["cmd"], seen["kw"] = cmd, kw
        return SimpleNamespace(stdout="abc|2024-01-01|fix: a | b\nbad line\nd|2024-01-02|s\n")

    monkeypatch.setattr(rd.subprocess, "run", run)
    h = rd._recent_activity(3)
    assert seen["cmd"][:2] == ["git", "log"] and "-3" in seen["cmd"]
    assert seen["kw"]["cwd"] == env and seen["kw"]["timeout"] == 10
    assert "<td>2024-01-01</td><td><code>abc</code></td><td>fix: a | b</td>" in h
    assert "bad line" not in h
    assert h.count("<tr>") == 3
    rd._recent_activity()
    assert "-12" in seen["cmd"]


def test_recent_activity_git_failure(monkeypatch):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired("git", 10)

    monkeypatch.setattr(rd.subprocess, "run", boom)
    assert rd._recent_activity() == "<p style='color:#999'>git log unavailable</p>"


# ---- assemble ----
def _stub_all(monkeypatch):
    fake_mod(monkeypatch, "governance", run=lambda verbose: {"a": (True, "")})
    fake_mod(
        monkeypatch, "backlog", _load=lambda: [], ranked=lambda i, status: [], wsjf=lambda i: 0
    )
    fake_mod(monkeypatch, "market_memory", status=lambda m: _status([]))
    fake_mod(monkeypatch, "daily_brief", build_brief=lambda m: "B")
    fake_mod(monkeypatch, "morning_mailer", _md_to_html=lambda s: "TAPE")
    fake_mod(monkeypatch, "auto_screener", recommend=lambda mk, top, verbose: {"picks": ["P"]})
    monkeypatch.setattr(rd.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=""))


def test_assemble(env, monkeypatch):
    _stub_all(monkeypatch)
    subject, text, html = rd.assemble()
    today = dt.date.today().isoformat()
    assert subject == f"📊 Repo Activity Dashboard — {today}"
    assert "markets: IN, US" in html
    assert "Markets: IN, US" in text and "Governance: 1/1 | Tests: 0 | Integrity files: 0" in text
    assert "Backlog: 0 done / 0 doing / 0 todo" in text
    for card in (
        "Challenges &amp; Alerts",
        "Market Tape",
        "Stock Lists",
        "Data Coverage",
        "SAFe Backlog",
        "Recent Activity",
    ):
        assert card in html
    assert "TAPE" in html and "<b>IN</b>: P" in html
    assert html.startswith("<html>") and html.endswith("</html>")
    _, t2, h2 = rd.assemble(["JP"])
    assert "Markets: JP" in t2 and "markets: JP" in h2


def test_assemble_sections_degrade(env, monkeypatch):
    _stub_all(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("section down")

    fake_mod(monkeypatch, "daily_brief", build_brief=boom)
    fake_mod(monkeypatch, "market_memory", status=boom)
    html = rd.assemble(["IN"])[2]
    assert "tape unavailable" in html and "coverage unavailable" in html
    assert "section down" in html


# ---- send / main ----
def test_send_dry(env, monkeypatch, capsys):
    _stub_all(monkeypatch)
    assert rd.send(["IN"], dry=True) is False
    assert (env / "results" / "repo_dashboard.html").read_text().startswith("<html>")
    assert "Repo Activity Dashboard" in (env / "results" / "repo_dashboard.txt").read_text()
    out = capsys.readouterr().out
    assert "(dry run) →" in out and "repo_dashboard.html" in out
    rd.send(["IN"], dry=True, verbose=False)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("ok,msg", [(True, "sent ✓"), (False, "no GMAIL creds")])
def test_send_via_mailer(env, monkeypatch, capsys, ok, msg):
    _stub_all(monkeypatch)
    sent = {}

    def fake_send(subject, text, html):
        sent["a"] = (subject, text, html)
        return ok

    fake_mod(monkeypatch, "send_mailer", send=fake_send)
    assert rd.send(["IN"]) is ok
    assert sent["a"][0].startswith("📊 Repo Activity Dashboard")
    assert msg in capsys.readouterr().out
    assert rd.send(["IN"], verbose=False) is ok
    assert capsys.readouterr().out == ""


def test_send_mailer_failure(env, monkeypatch, capsys):
    _stub_all(monkeypatch)

    def boom(*a):
        raise OSError("smtp down")

    fake_mod(monkeypatch, "send_mailer", send=boom)
    assert rd.send() is False
    assert "send failed (smtp down)" in capsys.readouterr().out
    assert rd.send(verbose=False) is False
    assert capsys.readouterr().out == ""


def test_main(monkeypatch):
    calls = []
    monkeypatch.setattr(rd, "send", lambda m, dry: calls.append((m, dry)))
    monkeypatch.setattr(sys, "argv", ["repo_dashboard.py"])
    assert rd.main() == 0
    monkeypatch.setattr(sys, "argv", ["repo_dashboard.py", "--dry", "--market", "IN", "US"])
    assert rd.main() == 0
    assert calls == [(None, False), (["IN", "US"], True)]
