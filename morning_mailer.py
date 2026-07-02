#!/usr/bin/env python3
# morning_mailer.py
# =================
# The daily morning mailer: assembles one email from the market brief, watchlist
# decision-deltas, and the day's top BUY picks per market, and sends it via
# send_mailer (Gmail app-password from .env). Runs after the daily pipeline.
#
#   python3 morning_mailer.py              # build + send (or save HTML if no creds)
#   python3 morning_mailer.py --market IN US --dry   # build only, print, no send
#
# Schedule ~8:00 AM (after the overnight data refresh) — see the launchd/cron footer.
# ⚠️ Research/education only. Not advice.

from __future__ import annotations

import argparse
import datetime as _dt
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

warnings.filterwarnings("ignore")

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

COMBINED_DIR = Path(__file__).parent / "combined_report_results"
_MARKET_TITLES = {"IN": "🇮🇳 India", "US": "🇺🇸 United States"}


def _latest_combined(market: str) -> Optional[Path]:
    """Newest combined-report HTML fragment for a market, if daily_combined_report ran."""
    if not COMBINED_DIR.is_dir():
        return None
    frags = sorted(COMBINED_DIR.glob(f"combined_{market}_*.html"))
    return frags[-1] if frags else None


def assemble_combined(markets: List[str]) -> Optional[Tuple[str, str, str]]:
    """Old combined-report format: stitch per-market fundamentals + street-talk fragments.

    Returns (subject, text, html), or None if no fragments are available (caller
    then falls back to the brief format).
    """
    frags = [(m, _latest_combined(m)) for m in markets]
    frags = [(m, f) for m, f in frags if f is not None]
    if not frags:
        return None

    today = _dt.date.today().isoformat()
    subject = f"📈 Daily Combined Report — {today}"

    parts = [
        "<html><body style='font-family:sans-serif;max-width:760px'>",
        f"<h1>📈 Daily Combined Report — {today}</h1>",
    ]
    text_bits = [f"Daily Combined Report — {today}", ""]
    for m, f in frags:
        title = _MARKET_TITLES.get(m, m)
        parts.append(f"<h2 style='border-top:2px solid #ccc;padding-top:8px'>{title}</h2>")
        parts.append(f.read_text())
        text_bits.append(f"{title} — see HTML view. (source: {f.name})")
    parts.append("<p><i>Educational/research only. Not investment advice.</i></p></body></html>")
    text_bits += ["", "Educational/research only. NOT investment advice."]
    return subject, "\n".join(text_bits), "\n".join(parts)


def _top_buys(market: str, n: int = 8) -> list:
    """Top BUY tickers for a market from the auto-screener recommendation."""
    try:
        import auto_screener as a

        return a.recommend(market, top=n, verbose=False).get("picks", [])[:n]
    except Exception:
        return []


def assemble(markets: Optional[List[str]] = None, fmt: str = "combined") -> Tuple[str, str, str]:
    """Return (subject, text, html) for the morning mail.

    fmt="combined" (default) uses the daily_combined_report fragments when present,
    matching the long-standing email layout; it falls back to the brief format if
    no fragments exist. fmt="brief" always uses the compact brief format.
    """

    markets = markets or ["IN", "US"]
    if fmt == "combined":
        combined = assemble_combined(markets)
        if combined is not None:
            return combined

    today = _dt.date.today().isoformat()
    subject = f"📈 Morning Market Brief — {today}"

    # 1) market brief
    try:
        import daily_brief as db

        brief = db.build_brief(markets)
    except Exception:
        brief = "_brief unavailable_"

    # 2) watchlist deltas + 3) top buys, per market
    deltas, buys = [], []
    for m in markets:
        try:
            import watchlist_delta as wd

            d, s = wd.diff(m)
            if "note" not in s:
                deltas.append(
                    f"- **{m}**: {s['new_buys']} new BUYs · {s['downgrades']} downgrades · {s['new_flags']} new risk flags"
                )
        except Exception:
            pass
        tb = _top_buys(m)
        if tb:
            buys.append(f"- **{m}**: {', '.join(tb)}")

    md = [
        f"# {subject}",
        "",
        brief,
        "",
        "## Watchlist changes",
        *(deltas or ["- (no prior snapshot)"]),
        "",
        "## Top BUY candidates",
        *(buys or ["- (none)"]),
        "",
        "_Educational/research only. Not investment advice._",
    ]
    text = "\n".join(md)
    html = _md_to_html(text)
    return subject, text, html


def _md_to_html(md: str) -> str:
    """Minimal, dependency-free markdown→HTML (headings, bold, lists, tables)."""
    out, in_ul, in_tbl = [], False, False
    for line in md.splitlines():
        s = line.rstrip()
        if s.startswith("|") and "|" in s[1:]:
            cells = [c.strip() for c in s.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue  # separator row
            tag = "th" if not in_tbl else "td"
            if not in_tbl:
                out.append("<table border=1 cellpadding=4 style='border-collapse:collapse'>")
                in_tbl = True
            out.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_tbl:
            out.append("</table>")
            in_tbl = False
        if s.startswith("# "):
            out.append(f"<h2>{s[2:]}</h2>")
        elif s.startswith("## "):
            out.append(f"<h3>{s[3:]}</h3>")
        elif s.startswith("- "):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{s[2:]}</li>")
        else:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            out.append(f"<p>{s}</p>" if s else "")
    if in_ul:
        out.append("</ul>")
    if in_tbl:
        out.append("</table>")
    html = "\n".join(out).replace("**", "")  # strip bold markers (already structural)
    return f"<html><body style='font-family:sans-serif'>{html}</body></html>"


def send(
    markets: Optional[List[str]] = None,
    dry: bool = False,
    verbose: bool = True,
    fmt: str = "combined",
) -> bool:
    subject, text, html = assemble(markets, fmt=fmt)
    (RESULTS / "morning_brief.html").write_text(html)
    (RESULTS / "morning_brief.md").write_text(text)
    if dry:
        if verbose:
            print(text)
            print(f"\n  (dry run) → {RESULTS / 'morning_brief.html'}")
        return False
    try:
        import send_mailer

        ok = send_mailer.send(subject, text, html)
        if verbose:
            print("  sent ✓" if ok else "  no GMAIL creds — saved results/morning_brief.html")
        return bool(ok)
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"  send failed ({str(e)[:50]}); saved results/morning_brief.html")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Daily morning market mailer")
    ap.add_argument("--market", nargs="*")
    ap.add_argument("--dry", action="store_true", help="build + save HTML, do not send")
    ap.add_argument(
        "--format",
        choices=["combined", "brief"],
        default="combined",
        help="combined = daily_combined_report layout (default); brief = compact brief",
    )
    args = ap.parse_args()
    send(args.market, dry=args.dry, fmt=args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ── Schedule ~8:00 AM (after overnight data refresh) ────────────────────────────
# cron:  0 8 * * 1-5 cd /path/to/repo && python3 morning_mailer.py >> ~/mailer.log 2>&1
# launchd: StartCalendarInterval {Hour 8, Minute 0} running `python3 morning_mailer.py`
# (set GMAIL_USER / GMAIL_APP_PASSWORD / MAIL_TO in .env first)
