#!/usr/bin/env python3
"""
reward_picks_mailer.py — HTML/text section for the daily mailer that (1) explains the
reward-optimised, out-of-sample-validated screener in plain English, and (2) lists the
names currently passing it in the markets where the edge held out of sample.

The per-market strategy metadata lives in STRATEGY (the final, verified screen + its
out-of-sample statistic); the current names come from reports/reward_picks.csv
(reward_screener_picks.py). US is the screen re-optimised on realized portfolio ALPHA
(cash-flow quality); the others carry their discovery out-of-sample median excess. If the
picks CSV is missing the section degrades to just the explanation so the mailer never breaks.

  from reward_picks_mailer import section_html, section_text
  python3 reward_picks_mailer.py            # text preview

⚠ Educational/research only — NOT investment advice.
"""
import os

import pandas as pd

BASE = os.path.expanduser("~/Downloads/code/python_files")
FONT = "-apple-system,Segoe UI,Roboto,Arial,sans-serif"
PICKS = os.path.join(BASE, "reports", "reward_picks.csv")

FLAG = {"US": "🇺🇸", "CA": "🇨🇦", "TW": "🇹🇼", "IN": "🇮🇳", "CN": "🇨🇳"}
CCY = {"US": "$", "CA": "C$", "TW": "NT$", "IN": "₹", "CN": "¥"}
# market -> (label, out-of-sample badge, why here). Order = display order (strongest first).
STRATEGY = {
    "US": (
        "cash-flow quality",
        "out-of-sample alpha +4.5%/yr · 67% win",
        "top cash-flow/assets — cash-generative quality (re-optimised on realized return; "
        "beats value & low-vol on actual alpha)",
    ),
    "CA": (
        "12-month momentum",
        "out-of-sample excess +4.5% · 53% win",
        "last year's winners keep winning in this market",
    ),
    "TW": (
        "low volatility",
        "out-of-sample excess +3.2% · 57% win",
        "the calmest names beat the cross-section here",
    ),
    "IN": (
        "uptrend (above 200-day average)",
        "out-of-sample excess +1.5% · 52% win",
        "names trading above their 200-day average — note: the India edge is modest and "
        "liquidity-sensitive (it weakens among the very largest caps)",
    ),
    "CN": (
        "low volatility",
        "out-of-sample excess +0.6% · 51% win (marginal)",
        "calmest names — a thin, marginal edge",
    ),
}
# company-name map for the large caps that surface most; unknown → ticker only
NAMES = {
    "MU": "Micron",
    "NVDA": "NVIDIA",
    "MSFT": "Microsoft",
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "GOOG": "Alphabet",
    "META": "Meta",
    "PLTR": "Palantir",
    "QCOM": "Qualcomm",
    "AMAT": "Applied Materials",
    "NFLX": "Netflix",
    "TD.TO": "TD Bank",
    "SU.TO": "Suncor",
    "CM.TO": "CIBC",
    "CVE.TO": "Cenovus",
    "CLS.TO": "Celestica",
    "MG.TO": "Magna",
    "ABX.TO": "Barrick",
    "ATZ.TO": "Aritzia",
    "2330.TW": "TSMC",
    "2317.TW": "Hon Hai",
    "2882.TW": "Cathay Fin",
    "2881.TW": "Fubon",
    "2382.TW": "Quanta",
    "2891.TW": "CTBC",
    "2357.TW": "Asus",
    "300750.SZ": "CATL",
    "600519.SS": "Kweichow Moutai",
    "000063.SZ": "ZTE",
    "601899.SS": "Zijin Mining",
    "300059.SZ": "East Money",
    "002371.SZ": "NAURA",
    "ADANIPOWER": "Adani Power",
    "ADANIGREEN": "Adani Green",
    "ADANIENT": "Adani Ent",
    "ADANIPORTS": "Adani Ports",
    "BHEL": "BHEL",
    "POWERINDIA": "Hitachi Energy India",
    "MCX": "MCX",
    "BSE": "BSE Ltd",
    "NETWEB": "Netweb",
    "MTARTECH": "MTAR Tech",
    "HSCL": "Himadri Speciality",
    "APOLLO": "Apollo Micro",
}


def _load():
    return pd.read_csv(PICKS) if os.path.exists(PICKS) else None


def _explain_html():
    return (
        f'<div style="background:#eef3fb;border-left:4px solid #1a73e8;padding:9px 13px;'
        f'margin:6px 0 12px;font-size:12px;color:#20364f;line-height:1.55;font-family:{FONT}">'
        "<b>What is an out-of-sample screener?</b> Rather than trusting a fixed recipe "
        "(Piotroski, Magic Formula…), we let the data choose. Across ~20 point-in-time "
        "factors — value, quality, growth, momentum, volatility — a search builds the "
        "screen that most rewards a real edge. The guard against curve-fitting: every "
        "filter is <b>trained on early years and then tested on later years it never saw</b> "
        "(that unseen test is the <b>out-of-sample</b>, or OOS, check). Only screens whose "
        "edge <b>persisted out-of-sample</b> are shown. It survives in a handful of markets: "
        "on realized returns the US edge is <b>cash-flow quality</b> (not value or low-vol); "
        "elsewhere it is a <b>price/risk</b> factor (momentum, low-volatility, trend). "
        '<span style="color:#8a6d00">Edges are modest and survivorship-flattered — '
        "research only, not advice.</span></div>"
    )


def _table(mkt, rows):
    ccy = CCY.get(mkt, "")
    th = "text-align:right;padding:3px 7px;border-bottom:2px solid #1a73e8;font-size:10.5px;color:#555"
    head = (
        '<table style="border-collapse:collapse;width:100%;margin:2px 0 10px">'
        f'<tr><th style="text-align:left;padding:3px 7px;border-bottom:2px solid #1a73e8;'
        f'font-size:10.5px;color:#555">Name</th>'
        f'<th style="{th}">Ticker</th><th style="{th}">6-mo</th><th style="{th}">Last</th></tr>'
    )
    body = ""
    for _, r in rows.iterrows():
        tk = str(r["ticker"])
        nm = NAMES.get(tk, "")
        ret = r["ret_6m"] * 100 if pd.notna(r["ret_6m"]) else float("nan")
        rc = "#2e7d32" if ret >= 0 else "#c62828"
        td = "padding:3px 7px;border-bottom:1px solid #eee;font-size:11.5px"
        body += (
            f'<tr><td style="{td};color:#333">{nm or "—"}</td>'
            f'<td style="{td};text-align:right;font-family:monospace;color:#1a73e8">{tk}</td>'
            f'<td style="{td};text-align:right;color:{rc}">{ret:+.0f}%</td>'
            f'<td style="{td};text-align:right;color:#333">{ccy}{r["last_price"]:,.2f}</td></tr>'
        )
    return head + body + "</table>"


def section_html(top=8):
    picks = _load()
    head = (
        f'<div style="font-family:{FONT};max-width:760px;color:#1a1a1a">'
        '<h2 style="font-size:16px;border-bottom:2px solid #1a73e8;padding-bottom:3px;'
        'margin-top:22px">🧪 Reward-Optimised Screeners — Out-of-Sample (OOS) Robust Picks</h2>'
    )
    if picks is None:
        return head + _explain_html() + "</div>"
    out = [head, _explain_html()]
    for mkt, (label, badge, why) in STRATEGY.items():
        rows = picks[picks["market"] == mkt].head(top)
        if not len(rows):
            continue
        out.append(
            f'<h3 style="font-size:13.5px;margin:14px 0 1px">{FLAG.get(mkt,"")} {mkt} — {label} '
            f'<span style="font-size:10px;color:#fff;background:#1a73e8;padding:1px 7px;'
            f'border-radius:10px">{badge}</span></h3>'
            f'<p style="font-size:11px;color:#666;margin:0 0 3px">Why here: {why}.</p>'
        )
        out.append(_table(mkt, rows))
    out.append(
        '<p style="font-size:10.5px;color:#8a6d00;margin:6px 0 0">Ranked by liquidity within '
        "each screen (not the most-extreme factor value, which surfaces pumps/distress). "
        "“Out-of-sample” (OOS) = tested on later years the screen was never fit on. "
        "Full list: reports/reward_picks.csv.</p></div>"
    )
    return "".join(out)


def section_text(top=8):
    picks = _load()
    lines = ["REWARD-OPTIMISED SCREENERS — OUT-OF-SAMPLE (OOS) ROBUST PICKS", ""]
    lines.append(
        "Data-chosen screens (search over ~20 factors) validated on later years the screen "
        "was never fit on (that is the out-of-sample, or OOS, test). On realized returns the "
        "US edge is cash-flow quality; elsewhere price/risk factors (momentum, low-vol, "
        "trend). Modest, survivorship-flattered edges. Research only, not advice."
    )
    if picks is None:
        return "\n".join(lines)
    for mkt, (label, badge, _why) in STRATEGY.items():
        rows = picks[picks["market"] == mkt].head(top)
        if not len(rows):
            continue
        lines.append("")
        lines.append(f"{mkt} — {label}  ({badge})")
        for _, r in rows.iterrows():
            nm = NAMES.get(str(r["ticker"]), "")
            lines.append(f"  {str(r['ticker']):<12} {nm:<20} 6mo {r['ret_6m']*100:+.0f}%")
    return "\n".join(lines)


if __name__ == "__main__":
    print(section_text())
