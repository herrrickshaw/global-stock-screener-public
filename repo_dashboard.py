#!/usr/bin/env python3
# repo_dashboard.py
# =================
# One email that shows *everything happening in the repo*: data coverage, the
# market tape, today's stock lists, engineering health (tests / governance /
# integrity), the SAFe backlog, open challenges, and recent git activity.
#
#   python3 repo_dashboard.py               # build + send (or save HTML if no creds)
#   python3 repo_dashboard.py --dry         # build + print + save, do not send
#   python3 repo_dashboard.py --market IN US
#
# Every section is best-effort: a source that errors degrades to a "—" note
# rather than failing the whole dashboard. Pure/offline-safe for CI.
# ⚠️ Research/education only. Not advice.

from __future__ import annotations

import argparse
import datetime as _dt
import subprocess
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
COMBINED_DIR = HERE / "combined_report_results"


# ── section builders (each returns an HTML fragment; never raises) ──────────────
def _safe(fn, fallback: str) -> str:
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return f"<p style='color:#999'>{fallback} <i>({str(e)[:80]})</i></p>"


def _card(title: str, body: str) -> str:
    return (
        "<div style='border:1px solid #e0e0e0;border-radius:8px;padding:14px 18px;"
        "margin:14px 0;background:#fafafa'>"
        f"<h2 style='margin:0 0 10px;font-size:17px'>{title}</h2>{body}</div>"
    )


def _kpis(markets: List[str]) -> Tuple[str, dict]:
    """Top-of-dashboard KPI strip; also returns raw metrics for the challenges scan."""
    metrics: dict = {}

    # governance
    try:
        import governance as g

        res = g.run(verbose=False)
        passed = sum(1 for ok, _ in res.values() if ok)
        metrics["gov_pass"], metrics["gov_total"] = passed, len(res)
        metrics["gov_fail"] = [n for n, (ok, _) in res.items() if not ok]
    except Exception:
        metrics["gov_pass"] = metrics["gov_total"] = 0
        metrics["gov_fail"] = []

    # tests
    try:
        metrics["tests"] = len(list((HERE / "tests").glob("test_*.py")))
    except Exception:
        metrics["tests"] = 0

    # integrity file count
    try:
        man = HERE / "cache_seed" / "CHECKSUMS.sha256"
        metrics["checksums"] = sum(1 for _ in man.read_text().splitlines() if _.strip())
    except Exception:
        metrics["checksums"] = 0

    # backlog
    try:
        import backlog as b

        items = b._load()
        metrics["done"] = sum(1 for i in items if i.get("status") == "done")
        metrics["todo"] = sum(1 for i in items if i.get("status") == "todo")
        metrics["doing"] = sum(1 for i in items if i.get("status") == "doing")
    except Exception:
        metrics["done"] = metrics["todo"] = metrics["doing"] = 0

    def chip(label: str, value: str, color: str) -> str:
        return (
            "<div style='display:inline-block;min-width:96px;margin:4px;padding:10px 14px;"
            f"border-radius:8px;background:{color};color:#fff;text-align:center'>"
            f"<div style='font-size:22px;font-weight:700'>{value}</div>"
            f"<div style='font-size:11px;opacity:.9'>{label}</div></div>"
        )

    gov = f"{metrics['gov_pass']}/{metrics['gov_total']}"
    gov_color = "#2e7d32" if metrics["gov_pass"] == metrics["gov_total"] else "#c62828"
    strip = (
        chip("Governance", gov, gov_color)
        + chip("Tests", str(metrics["tests"]), "#1565c0")
        + chip("Integrity files", str(metrics["checksums"]), "#5e35b1")
        + chip("Backlog done", str(metrics["done"]), "#2e7d32")
        + chip("Backlog todo", str(metrics["todo"]), "#ef6c00")
        + chip("Markets", str(len(markets)), "#00838f")
    )
    return f"<div style='text-align:center'>{strip}</div>", metrics


def _data_coverage(markets: List[str]) -> str:
    import market_memory as mm

    df = mm.status(markets)
    rows = [
        "<tr><th>Market</th><th>Tier</th><th>Symbols</th><th>Start</th><th>End</th><th>Rows</th></tr>"
    ]
    for _, r in df.iterrows():
        rows.append(
            f"<tr><td><b>{r['market']}</b></td><td>{r['tier'].upper()}</td>"
            f"<td>{r['symbols']}</td><td>{r['start']}</td><td>{r['end']}</td>"
            f"<td>{int(r['rows']):,}</td></tr>"
        )
    return (
        "<table border=1 cellpadding=4 style='border-collapse:collapse'>"
        + "".join(rows)
        + "</table>"
    )


def _market_tape(markets: List[str]) -> str:
    import daily_brief as db

    brief = db.build_brief(markets)
    # brief is markdown; reuse the mailer's converter for consistent styling
    import morning_mailer as m

    return m._md_to_html(brief)


def _stock_lists(markets: List[str]) -> str:
    """Today's stock lists — reuse the combined-report fragments if present."""
    frags = []
    if COMBINED_DIR.is_dir():
        for mk in markets:
            fs = sorted(COMBINED_DIR.glob(f"combined_{mk}_*.html"))
            if fs:
                title = {"IN": "🇮🇳 India", "US": "🇺🇸 United States"}.get(mk, mk)
                frags.append(f"<h3 style='margin:12px 0 4px'>{title}</h3>" + fs[-1].read_text())
    if frags:
        return "".join(frags)
    # fallback: auto-screener top BUYs
    import auto_screener as a

    out = []
    for mk in markets:
        picks = a.recommend(mk, top=10, verbose=False).get("picks", [])[:10]
        out.append(f"<p><b>{mk}</b>: {', '.join(picks) if picks else '—'}</p>")
    return "".join(out)


def _backlog_table() -> str:
    import backlog as b

    items = b._load()
    todo = b.ranked(items, status="todo")
    doing = [i for i in items if i.get("status") == "doing"]
    if not todo and not doing:
        return "<p>✅ Backlog clear — 0 todo / 0 doing.</p>"
    rows = ["<tr><th>WSJF</th><th>ID</th><th>Title</th><th>Status</th><th>Epic</th></tr>"]
    for i in doing + todo[:12]:
        rows.append(
            f"<tr><td>{b.wsjf(i):.2f}</td><td>{i['id']}</td><td>{i['title']}</td>"
            f"<td>{i.get('status')}</td><td>{i.get('epic', '')}</td></tr>"
        )
    return (
        "<table border=1 cellpadding=4 style='border-collapse:collapse'>"
        + "".join(rows)
        + "</table>"
    )


def _challenges(markets: List[str], metrics: dict) -> str:
    """Actionable issues surfaced from the aggregated state."""
    issues: List[str] = []
    for name in metrics.get("gov_fail", []):
        issues.append(f"🔴 Governance non-conformance: <b>{name}</b>")
    if metrics.get("todo", 0):
        issues.append(f"🟠 {metrics['todo']} backlog item(s) still open (see backlog card)")
    # data staleness / gaps
    try:
        import market_memory as mm

        df = mm.status(markets)
        today = _dt.date.today()
        for _, r in df.iterrows():
            if r["tier"] != "stm":
                continue
            if int(r["rows"]) == 0:
                issues.append(f"🔴 No data for <b>{r['market']}</b> (STM empty)")
                continue
            try:
                end = _dt.date.fromisoformat(str(r["end"])[:10])
                lag = (today - end).days
                if lag > 5:
                    issues.append(
                        f"🟠 <b>{r['market']}</b> data is {lag} days stale (last {r['end']})"
                    )
            except Exception:
                pass
    except Exception:
        pass
    if not issues:
        return "<p>✅ No open challenges — all green.</p>"
    return "<ul>" + "".join(f"<li>{i}</li>" for i in issues) + "</ul>"


def _recent_activity(n: int = 12) -> str:
    try:
        out = subprocess.run(
            ["git", "log", f"-{n}", "--pretty=format:%h|%ad|%s", "--date=short"],
            cwd=HERE,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except Exception:
        return "<p style='color:#999'>git log unavailable</p>"
    rows = ["<tr><th>Date</th><th>Commit</th><th>Summary</th></tr>"]
    for line in out.splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3:
            h, d, s = parts
            rows.append(f"<tr><td>{d}</td><td><code>{h}</code></td><td>{s}</td></tr>")
    return (
        "<table border=1 cellpadding=4 style='border-collapse:collapse'>"
        + "".join(rows)
        + "</table>"
    )


# ── assembly ────────────────────────────────────────────────────────────────
def assemble(markets: Optional[List[str]] = None) -> Tuple[str, str, str]:
    """Return (subject, text, html) for the repo activity dashboard."""
    markets = markets or ["IN", "US"]
    today = _dt.date.today().isoformat()
    subject = f"📊 Repo Activity Dashboard — {today}"

    kpi_html, metrics = _kpis(markets)

    body = [
        "<html><body style='font-family:sans-serif;max-width:820px;margin:0 auto'>",
        "<h1 style='margin-bottom:2px'>📊 Repo Activity Dashboard</h1>",
        f"<p style='color:#666;margin-top:0'>{today} · markets: {', '.join(markets)}</p>",
        kpi_html,
        _card(
            "⚠️ Challenges &amp; Alerts",
            _safe(lambda: _challenges(markets, metrics), "challenges unavailable"),
        ),
        _card("🗞️ Market Tape", _safe(lambda: _market_tape(markets), "tape unavailable")),
        _card(
            "📈 Today's Stock Lists",
            _safe(lambda: _stock_lists(markets), "stock lists unavailable"),
        ),
        _card("💾 Data Coverage", _safe(lambda: _data_coverage(markets), "coverage unavailable")),
        _card("🧭 SAFe Backlog (open, WSJF-ranked)", _safe(_backlog_table, "backlog unavailable")),
        _card("🛠️ Recent Activity", _safe(lambda: _recent_activity(12), "git log unavailable")),
        "<p style='font-size:11px;color:#bf360c'>⚠️ Research/education only. NOT investment advice. "
        "Consult a SEBI-registered advisor.</p>",
        "</body></html>",
    ]
    html = "\n".join(body)

    # compact plain-text alternative
    text = "\n".join(
        [
            f"Repo Activity Dashboard — {today}",
            f"Markets: {', '.join(markets)}",
            f"Governance: {metrics['gov_pass']}/{metrics['gov_total']} | Tests: {metrics['tests']} | "
            f"Integrity files: {metrics['checksums']}",
            f"Backlog: {metrics['done']} done / {metrics['doing']} doing / {metrics['todo']} todo",
            "",
            "Open the HTML view for the full dashboard (tape, stock lists, coverage, backlog, activity).",
            "",
            "Research/education only. NOT investment advice.",
        ]
    )
    return subject, text, html


def send(markets: Optional[List[str]] = None, dry: bool = False, verbose: bool = True) -> bool:
    subject, text, html = assemble(markets)
    (RESULTS / "repo_dashboard.html").write_text(html)
    (RESULTS / "repo_dashboard.txt").write_text(text)
    if dry:
        if verbose:
            print(text)
            print(f"\n  (dry run) → {RESULTS / 'repo_dashboard.html'}")
        return False
    try:
        import send_mailer

        ok = send_mailer.send(subject, text, html)
        if verbose:
            print("  sent ✓" if ok else "  no GMAIL creds — saved results/repo_dashboard.html")
        return bool(ok)
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"  send failed ({str(e)[:50]}); saved results/repo_dashboard.html")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Repo activity dashboard mailer")
    ap.add_argument("--market", nargs="*")
    ap.add_argument("--dry", action="store_true", help="build + save HTML, do not send")
    args = ap.parse_args()
    send(args.market, dry=args.dry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
