#!/usr/bin/env python3
"""
reward_overlay_mailer.py — daily-mailer section for the LIQUIDITY-CONFIRMED picks: names
that pass BOTH a market's out-of-sample-robust price screen AND its confirming liquidity
trend (reward_overlay). Reads reports/reward_overlay_picks.csv (REWARD_BASE-aware). Degrades
to nothing if the file is absent, so the mailer never breaks.

  from reward_overlay_mailer import section_html, section_text
  python3 reward_overlay_mailer.py       # text preview

⚠ Educational/research only — NOT investment advice.
"""
import os

import pandas as pd

BASE = os.environ.get("REWARD_BASE") or os.path.expanduser("~/Downloads/code/python_files")
FONT = "-apple-system,Segoe UI,Roboto,Arial,sans-serif"
PICKS = os.path.join(BASE, "reports", "reward_overlay_picks.csv")

FLAG = {"HK": "🇭🇰", "UK": "🇬🇧", "JP": "🇯🇵", "TW": "🇹🇼", "SG": "🇸🇬", "CA": "🇨🇦", "SE": "🇸🇪"}
CCY = {"HK": "HK$", "UK": "£", "JP": "¥", "TW": "NT$", "SG": "S$", "CA": "C$", "SE": "kr"}
# per-market OOS alpha uplift from the liquidity overlay (reward_overlay sweep, 2022-24)
UPLIFT = {
    "HK": "+8.9pp",
    "UK": "+3.5pp",
    "JP": "+3.1pp",
    "TW": "+2.8pp",
    "CA": "+2.6pp",
    "SE": "+2.6pp",
    "SG": "+2.3pp",
}
NAMES = {
    "0939.HK": "China Construction Bank",
    "0941.HK": "China Mobile",
    "1398.HK": "ICBC",
    "2388.HK": "BOC Hong Kong",
    "0005.HK": "HSBC",
    "1288.HK": "Agri Bank of China",
    "ULVR.L": "Unilever",
    "LGEN.L": "Legal & General",
    "NXT.L": "Next",
    "BEZ.L": "Beazley",
    "6201.T": "Toyota Industries",
    "9005.T": "Tokyu",
    "1878.T": "Daito Trust",
    "5938.T": "LIXIL",
    "2633.TW": "Taiwan High Speed Rail",
    "2006.TW": "Tung Ho Steel",
    "1723.TW": "China Steel Chem",
    "D05.SI": "DBS",
    "O39.SI": "OCBC",
    "C38U.SI": "CapitaLand Ascendas",
    "S68.SI": "SGX",
    "RY.TO": "Royal Bank",
    "ENB.TO": "Enbridge",
    "TD.TO": "TD Bank",
    "BMO.TO": "Bank of Montreal",
    "AAK.ST": "AAK",
    "SINCH.ST": "Sinch",
    "MILDEF.ST": "MilDef",
    "FABG.ST": "Fabege",
}


def _load():
    return pd.read_csv(PICKS) if os.path.exists(PICKS) else None


def _table(mkt, rows):
    ccy = CCY.get(mkt, "")
    th = "text-align:right;padding:3px 7px;border-bottom:2px solid #6a1b9a;font-size:10.5px;color:#555"
    head = (
        '<table style="border-collapse:collapse;width:100%;margin:2px 0 10px">'
        f'<tr><th style="text-align:left;padding:3px 7px;border-bottom:2px solid #6a1b9a;'
        f'font-size:10.5px;color:#555">Name</th><th style="{th}">Ticker</th>'
        f'<th style="{th}">6-mo Δvol</th><th style="{th}">6-mo ret</th><th style="{th}">Last</th></tr>'
    )
    body = ""
    for _, r in rows.iterrows():
        tk = str(r["ticker"])
        nm = NAMES.get(tk, "")
        vol = r["liq_6_6"] * 100 if pd.notna(r["liq_6_6"]) else float("nan")
        ret = r["ret_6m"] * 100 if pd.notna(r["ret_6m"]) else float("nan")
        rc = "#2e7d32" if ret >= 0 else "#c62828"
        td = "padding:3px 7px;border-bottom:1px solid #eee;font-size:11.5px"
        body += (
            f'<tr><td style="{td};color:#333">{nm or "—"}</td>'
            f'<td style="{td};text-align:right;font-family:monospace;color:#6a1b9a">{tk}</td>'
            f'<td style="{td};text-align:right;color:#555">{vol:+.0f}%</td>'
            f'<td style="{td};text-align:right;color:{rc}">{ret:+.0f}%</td>'
            f'<td style="{td};text-align:right;color:#333">{ccy}{r["last_price"]:,.2f}</td></tr>'
        )
    return head + body + "</table>"


def section_html(top=8):
    picks = _load()
    head = (
        f'<div style="font-family:{FONT};max-width:760px;color:#1a1a1a">'
        '<h2 style="font-size:16px;border-bottom:2px solid #6a1b9a;padding-bottom:3px;'
        'margin-top:22px">💧 Liquidity-Confirmed Picks — price screen × volume trend</h2>'
        f'<div style="background:#f3e5f5;border-left:4px solid #6a1b9a;padding:9px 13px;'
        f'margin:6px 0 12px;font-size:12px;color:#3a1a4f;line-height:1.55">'
        "A price signal is stronger when the <b>liquidity trend confirms it</b>. Overlaying "
        "each market's out-of-sample-robust price screen with its volume trend improved "
        "realized out-of-sample alpha in <b>8 of 9 markets</b> (the Piotroski×Darvas pattern: "
        "liquidity works as a <b>confirmation</b>, not a stand-alone screen). Below: names "
        "passing the price screen <b>and</b> its confirming volume trend — mostly low-vol / "
        "breakout blue chips with <b>building</b> participation (Canada & Sweden: <b>quieting</b>). "
        '<span style="color:#8a6d00">Modest, survivorship-flattered edges — research only, '
        "not advice.</span></div>"
    )
    if picks is None or picks.empty:
        return head + "</div>"
    out = [head]
    for mkt in [m for m in FLAG if m in set(picks["market"])]:
        rows = picks[picks["market"] == mkt].head(top)
        if not len(rows):
            continue
        label = rows.iloc[0]["screen"]
        out.append(
            f'<h3 style="font-size:13.5px;margin:14px 0 1px">{FLAG.get(mkt,"")} {mkt} — {label} '
            f'<span style="font-size:10px;color:#fff;background:#6a1b9a;padding:1px 7px;'
            f'border-radius:10px">overlay {UPLIFT.get(mkt,"")} OOS alpha</span></h3>'
        )
        out.append(_table(mkt, rows))
    out.append(
        '<p style="font-size:10.5px;color:#8a6d00;margin:6px 0 0">“6-mo Δvol” = change in '
        "6-month dollar-volume vs the prior 6 months (the confirming liquidity trend). "
        "Full list: reports/reward_overlay_picks.csv.</p></div>"
    )
    return "".join(out)


def section_text(top=8):
    picks = _load()
    lines = ["LIQUIDITY-CONFIRMED PICKS — price screen × volume trend", ""]
    lines.append(
        "Names passing a market's OOS-robust price screen AND its confirming liquidity trend. "
        "The overlay improved realized out-of-sample alpha in 8/9 markets. Research only."
    )
    if picks is None or picks.empty:
        return "\n".join(lines)
    for mkt in [m for m in FLAG if m in set(picks["market"])]:
        rows = picks[picks["market"] == mkt].head(top)
        if not len(rows):
            continue
        lines.append("")
        lines.append(f"{mkt} — {rows.iloc[0]['screen']} (overlay {UPLIFT.get(mkt,'')} OOS)")
        for _, r in rows.iterrows():
            nm = NAMES.get(str(r["ticker"]), "")
            lines.append(f"  {str(r['ticker']):<10} {nm:<24} Δvol {r['liq_6_6']*100:+.0f}%")
    return "\n".join(lines)


if __name__ == "__main__":
    print(section_text())
