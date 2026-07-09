#!/usr/bin/env python3
"""Assemble the daily market report email from the combined-report JSONs,
US fundamentals scan, and the two Darvas fragments."""
import datetime
import html
import json
import os

import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/Downloads/code/python_files")
SP = "/private/tmp/claude-501/-Users-umashankar/f9e77393-1920-4a43-b9bd-676ad0b5a584/scratchpad"
IN_JSON = os.path.join(BASE, "combined_report_results/combined_IN_20260709_0808.json")
US_JSON = os.path.join(BASE, "combined_report_results/combined_US_20260709_0809.json")
US_SCAN = os.path.join(BASE, "us_full_scan/us_full_scan_20260708_1238.xlsx")
import glob as _glob


def _latest_eu():
    broad = sorted(
        _glob.glob(os.path.join(BASE, "european_scan/european_market_scan_broad_*.xlsx")),
        key=os.path.getmtime,
    )
    if broad:
        return broad[-1]
    any_eu = sorted(
        _glob.glob(os.path.join(BASE, "european_scan/european_market_scan_*.xlsx")),
        key=os.path.getmtime,
    )
    return (
        any_eu[-1]
        if any_eu
        else os.path.join(BASE, "european_scan/european_market_scan_20260709_0842.xlsx")
    )


EU_SCAN = _latest_eu()


def esc(s):
    return html.escape(str(s))


inj = json.load(open(IN_JSON))
usj = json.load(open(US_JSON))

# ---- US fundamentals enrichment: exchange, name, sector, PE, sector-relative zone
fu = pd.read_excel(US_SCAN, sheet_name="All_Fundamentals")
fu["PE"] = fu["Earnings_Yield_%"].apply(lambda e: 100.0 / e if pd.notna(e) and e > 0 else np.nan)
sec_med = fu.groupby("Sector")["PE"].median()
uni_med = fu["PE"].median()
meta = fu.set_index("Symbol")[["Exchange", "Name", "Sector", "PE"]].to_dict("index")


def pe_zone(sym):
    m = meta.get(sym)
    if not m or pd.isna(m["PE"]):
        return "—", "#888"
    pe = m["PE"]
    sm = sec_med.get(m["Sector"], uni_med)
    if pd.isna(sm) or sm <= 0:
        return f"PE {pe:.0f}", "#888"
    r = pe / sm
    if r <= 0.8:
        return f"🟢 BUY (PE {pe:.0f} vs {sm:.0f})", "#1a7f37"
    if r <= 1.1:
        return f"🟡 FAIR (PE {pe:.0f} vs {sm:.0f})", "#b58a00"
    if r <= 1.6:
        return f"🟠 CAUTION (PE {pe:.0f} vs {sm:.0f})", "#c05a00"
    return f"🔴 RICH (PE {pe:.0f} vs {sm:.0f})", "#c0392b"


def usname(sym):
    m = meta.get(sym)
    n = m["Name"] if m else None
    return str(n) if (n is not None and not (isinstance(n, float) and pd.isna(n))) else sym


def usexch(sym):
    m = meta.get(sym)
    e = m["Exchange"] if m else None
    return str(e) if (e is not None and not (isinstance(e, float) and pd.isna(e))) else "US"


# ---- Europe: broad multi-exchange scan (Fundamentals + Darvas_Signals sheets)
# Suffix ordered longest-first so ".MC" isn't shadowed by ".C"-style prefixes.
EU_EXCH = {
    ".DE": "XETRA",
    ".F": "Frankfurt",
    ".PA": "Euronext Paris",
    ".AS": "Euronext Amsterdam",
    ".BR": "Euronext Brussels",
    ".MI": "Borsa Italiana",
    ".MC": "BME Madrid",
    ".HE": "Nasdaq Helsinki",
    ".ST": "Nasdaq Stockholm",
    ".CO": "Nasdaq Copenhagen",
    ".OL": "Oslo Bors",
    ".LS": "Euronext Lisbon",
    ".IR": "Euronext Dublin",
    ".SW": "SIX Swiss",
    ".VI": "Vienna",
    ".L": "London SE",
    ".IS": "Borsa Istanbul",
    ".PR": "Prague SE",
    ".BD": "Budapest SE",
    ".TL": "Nasdaq Tallinn",
    ".RG": "Nasdaq Riga",
    ".VS": "Nasdaq Vilnius",
    ".IC": "Nasdaq Iceland",
    ".WA": "Warsaw GPW",
    ".AT": "Athens SE",
}


def eu_exch(sym):
    su = str(sym).upper()
    for suf in sorted(EU_EXCH, key=len, reverse=True):
        if su.endswith(suf):
            return EU_EXCH[suf]
    return "Europe"


# Local currency by suffix (Baltics .TL/.RG/.VS are eurozone → €).
EU_CUR = {
    ".L": "£",
    ".SW": "CHF ",
    ".ST": "kr",
    ".OL": "kr",
    ".CO": "kr",
    ".IC": "kr",
    ".WA": "zł",
    ".IS": "₺",
    ".PR": "Kč",
    ".BD": "Ft",
}


def eu_cur(sym):
    su = str(sym).upper()
    for suf in sorted(EU_CUR, key=len, reverse=True):
        if su.endswith(suf):
            return EU_CUR[suf]
    return "€"


def eu_name(s):
    """Clean a company name for display — some yfinance shortNames come back
    mangled as 'TICKER.XX,0P00...,1060'; keep the readable part before the comma."""
    n = str(s).split(",")[0].strip()
    return n or str(s)


eu_fu = pd.read_excel(EU_SCAN, sheet_name="Fundamentals")
eu_fu["P_E_Ratio"] = pd.to_numeric(eu_fu["P_E_Ratio"], errors="coerce")
eu_sec_med = eu_fu.groupby("Sector")["P_E_Ratio"].median()
eu_uni_med = eu_fu["P_E_Ratio"].median()
eu_dv = pd.read_excel(EU_SCAN, sheet_name="Darvas_Signals")


def eu_pe_zone(pe, sector):
    if pd.isna(pe) or pe <= 0:
        return "—", "#888"
    sm = eu_sec_med.get(sector, eu_uni_med)
    if pd.isna(sm) or sm <= 0:
        return f"PE {pe:.0f}", "#888"
    r = pe / sm
    if r <= 0.8:
        return f"🟢 BUY (PE {pe:.0f} vs {sm:.0f})", "#1a7f37"
    if r <= 1.1:
        return f"🟡 FAIR (PE {pe:.0f} vs {sm:.0f})", "#b58a00"
    if r <= 1.6:
        return f"🟠 CAUTION (PE {pe:.0f} vs {sm:.0f})", "#c05a00"
    return f"🔴 RICH (PE {pe:.0f} vs {sm:.0f})", "#c0392b"


def eu_screens(row):
    s = []
    if str(row.get("Darvas_Signal")) == "BREAKOUT_BUY":
        s.append("Darvas")
    if pd.notna(row.get("Piotroski_Score")) and row["Piotroski_Score"] >= 7:
        s.append("Piotroski")
    if str(row.get("CoffeeCan_Class")) == "PASS":
        s.append("CoffeeCan")
    return "+".join(s) or "—"


# ---------------------------------------------------------------- Component 1
def us_c1_rows(picks, tiers):
    rows = ""
    for p in picks:
        if p["Tier"] not in tiers:
            continue
        sym = p["Symbol"]
        zt, zc = pe_zone(sym)
        pio = (
            ""
            if p.get("Piotroski") in (None,)
            or (isinstance(p.get("Piotroski"), float) and np.isnan(p["Piotroski"]))
            else f'{p["Piotroski"]:.0f}'
        )
        rows += (
            f'<tr><td style="padding:6px 8px;border-bottom:1px solid #eee"><b>{esc(usname(sym).upper())}</b>'
            f' <span style="color:#666;font-size:12px">[{esc(sym)} · {esc(usexch(sym))}]</span></td>'
            f'<td style="padding:6px 8px;border-bottom:1px solid #eee;text-align:right">${p["LTP"]:,.2f}</td>'
            f'<td style="padding:6px 8px;border-bottom:1px solid #eee">{esc(p["Tier"])}</td>'
            f'<td style="padding:6px 8px;border-bottom:1px solid #eee;font-size:12px">{esc(p["Screens"])}</td>'
            f'<td style="padding:6px 8px;border-bottom:1px solid #eee;text-align:center">{pio}</td>'
            f'<td style="padding:6px 8px;border-bottom:1px solid #eee;color:{zc};font-size:12px">{zt}</td></tr>'
        )
    return rows


C1_HDR = (
    '<tr style="background:#1f2937;color:#fff">'
    '<th style="padding:8px;text-align:left">Company [Ticker · Exch]</th>'
    '<th style="padding:8px;text-align:right">LTP</th><th style="padding:8px;text-align:left">Tier</th>'
    '<th style="padding:8px;text-align:left">Screens</th><th style="padding:8px">Piotroski</th>'
    '<th style="padding:8px;text-align:left">PE Zone (sector-rel.)</th></tr>'
)

# US split by exchange
us_picks = [p for p in usj["picks"] if p["Tier"] in ("Triple Hit", "Multi-Screen")]


def by_exch(picks, exch):
    return [p for p in picks if usexch(p["Symbol"]) == exch]


nasd = by_exch(us_picks, "NASDAQ")
nyse = by_exch(us_picks, "NYSE")
other = [p for p in us_picks if usexch(p["Symbol"]) not in ("NASDAQ", "NYSE")]


def order(picks):  # Triple Hits first
    return [p for p in picks if p["Tier"] == "Triple Hit"] + [
        p for p in picks if p["Tier"] == "Multi-Screen"
    ]


us_c1 = "<h3 style='margin:14px 0 4px;font-size:15px'>🇺🇸 US — NASDAQ</h3>"
us_c1 += f'<table style="width:100%;border-collapse:collapse;font-size:13px">{C1_HDR}{us_c1_rows(order(nasd),("Triple Hit","Multi-Screen"))}</table>'
us_c1 += "<h3 style='margin:14px 0 4px;font-size:15px'>🇺🇸 US — NYSE</h3>"
nyse_rows = us_c1_rows(order(nyse), ("Triple Hit", "Multi-Screen"))
us_c1 += f'<table style="width:100%;border-collapse:collapse;font-size:13px">{C1_HDR}{nyse_rows or "<tr><td colspan=6 style=padding:6px;color:#666>No NYSE multi-screen hits today.</td></tr>"}</table>'
if other:
    us_c1 += "<h3 style='margin:14px 0 4px;font-size:15px'>🇺🇸 US — Other (ADR/AMEX)</h3>"
    us_c1 += f'<table style="width:100%;border-collapse:collapse;font-size:13px">{C1_HDR}{us_c1_rows(order(other),("Triple Hit","Multi-Screen"))}</table>'

# India Component 1 — only Single-Screen Darvas picks exist; show top 25, tag NSE
in_top = inj["picks"][:25]
in_rows = ""
for p in in_top:
    in_rows += (
        f'<tr><td style="padding:6px 8px;border-bottom:1px solid #eee"><b>{esc(p["Symbol"])}</b>'
        f' <span style="color:#666;font-size:12px">[{esc(p["Symbol"])} · NSE]</span></td>'
        f'<td style="padding:6px 8px;border-bottom:1px solid #eee;text-align:right">₹{p["LTP"]:,.2f}</td>'
        f'<td style="padding:6px 8px;border-bottom:1px solid #eee">{esc(p["Tier"])}</td>'
        f'<td style="padding:6px 8px;border-bottom:1px solid #eee;font-size:12px">{esc(p["Screens"])}</td>'
        f'<td style="padding:6px 8px;border-bottom:1px solid #eee;text-align:center">—</td>'
        f'<td style="padding:6px 8px;border-bottom:1px solid #eee;color:#888;font-size:12px">n/a</td></tr>'
    )
in_c1 = (
    "<p style='font-size:12px;color:#666;margin:6px 0'>India's full-universe scan produced <b>no Triple-Hit or "
    "Multi-Screen fundamental picks</b> today — only single-screen Darvas/Golden-Cross momentum names, with no "
    "Piotroski or sector-PE fundamentals attached in this run. Top 25 single-screen names shown (all NSE-listed):</p>"
)
in_c1 += (
    f'<table style="width:100%;border-collapse:collapse;font-size:13px">{C1_HDR}{in_rows}</table>'
)

# Europe Component 1 — Triple Hit (P>=7 & CoffeeCan PASS) first, then strong Piotroski>=7.
# Broad universe yields many P>=7 names; cap the table to the strongest, note the rest.
eu_pick_all = eu_fu[eu_fu["Piotroski_Score"] >= 7].copy()
eu_pick_all["is_triple"] = (
    (eu_pick_all["Piotroski_Score"] >= 7)
    & (eu_pick_all["CoffeeCan_Class"] == "PASS")
    & (eu_pick_all["Darvas_Signal"] == "BREAKOUT_BUY")
)
eu_pick_all["_cc"] = pd.to_numeric(
    eu_pick_all["CoffeeCan_Score"].astype(str).str.split("/").str[0], errors="coerce"
).fillna(0)
eu_pick_all = eu_pick_all.sort_values(["is_triple", "Piotroski_Score", "_cc"], ascending=False)
EU_CAP = 30
eu_pick = eu_pick_all.head(EU_CAP)
eu_more = len(eu_pick_all) - len(eu_pick)
eu_rows = ""
for _, r in eu_pick.iterrows():
    tier = "Triple Hit" if r["is_triple"] else "Multi-Screen"
    zt, zc = eu_pe_zone(r["P_E_Ratio"], r["Sector"])
    eu_rows += (
        f'<tr><td style="padding:6px 8px;border-bottom:1px solid #eee"><b>{esc(eu_name(r["Name"]).upper())}</b>'
        f' <span style="color:#666;font-size:12px">[{esc(r["Symbol"])} · {esc(eu_exch(r["Symbol"]))}]</span></td>'
        f'<td style="padding:6px 8px;border-bottom:1px solid #eee;text-align:right">{eu_cur(r["Symbol"])}{r["LTP"]:,.2f}</td>'
        f'<td style="padding:6px 8px;border-bottom:1px solid #eee">{tier}</td>'
        f'<td style="padding:6px 8px;border-bottom:1px solid #eee;font-size:12px">{esc(eu_screens(r))}</td>'
        f'<td style="padding:6px 8px;border-bottom:1px solid #eee;text-align:center">{r["Piotroski_Score"]:.0f}</td>'
        f'<td style="padding:6px 8px;border-bottom:1px solid #eee;color:{zc};font-size:12px">{zt}</td></tr>'
    )
try:
    eu_all = pd.read_excel(EU_SCAN, sheet_name="All_Stocks")
    eu_scanned = len(eu_all)
    eu_n_exch = len({eu_exch(s) for s in eu_all["Symbol"]})
except Exception:
    eu_scanned = len(eu_dv)
    eu_n_exch = len({eu_exch(s) for s in eu_dv["Symbol"]})
eu_c1 = (
    f"<p style='font-size:12px;color:#666;margin:6px 0'>Broad European universe — <b>{eu_scanned} stocks across {eu_n_exch} exchanges</b> "
    "(LSE, Deutsche Börse, Euronext Paris/Amsterdam/Brussels/Lisbon/Dublin, Borsa Italiana, BME Madrid, Nasdaq Nordic &amp; Baltic, "
    "SIX Swiss, Vienna, Warsaw, Athens, Borsa Istanbul, Prague, Budapest, Iceland). "
    "<b>Triple Hit</b> = Darvas breakout + Piotroski≥7 + Coffee Can PASS; <b>Multi-Screen</b> = Piotroski≥7 breakout. "
    f"PE Zone is sector-relative within the EU universe; LTP in each market's local currency. "
    f"<b>{len(eu_pick_all)}</b> names cleared Piotroski≥7 — showing the strongest {len(eu_pick)}"
    f"{f' (+{eu_more} more in the CSV)' if eu_more else ''}.</p>"
)
eu_c1 += (
    f'<table style="width:100%;border-collapse:collapse;font-size:13px">{C1_HDR}{eu_rows}</table>'
)
# Persist the full Europe pick list (all Piotroski≥7) so the capped table has a source.
try:
    _euc = [
        "Symbol",
        "Name",
        "Sector",
        "LTP",
        "P_E_Ratio",
        "Piotroski_Score",
        "CoffeeCan_Score",
        "CoffeeCan_Class",
        "Darvas_Signal",
        "is_triple",
    ]
    eu_pick_all[[c for c in _euc if c in eu_pick_all.columns]].to_csv(
        os.path.expanduser(f"~/Downloads/eu_broad_picks_{datetime.date(2026,7,9).isoformat()}.csv"),
        index=False,
    )
except Exception:
    pass

# Europe Darvas — with the broad universe there are genuine fresh breakouts.
# Prefer position 100–120% + uptrend + up-on-day; fall back to least-extended uptrend.
eu_b = eu_dv[eu_dv["Darvas_Signal"] == "BREAKOUT_BUY"].copy()
eu_up = eu_b["Trend_Signal"].astype(str).str.contains("Uptrend", na=False)
eu_fresh = (
    eu_b[eu_up & eu_b["Position_in_Box%"].between(100, 120) & (eu_b["Change%"] > 0)]
    .sort_values("Change%", ascending=False)
    .head(12)
)
eu_fresh_mode = "fresh"
if eu_fresh.empty:
    eu_fresh = eu_b[eu_up].sort_values("Position_in_Box%").head(10)
    eu_fresh_mode = "least-extended"
eu_bd = eu_dv[eu_dv["Darvas_Signal"] == "BREAKDOWN_SELL"].sort_values("Change%").head(10)


def eu_dv_rows(rows):
    out = ""
    for _, r in rows.iterrows():
        cur = eu_cur(r["Symbol"])
        out += (
            f'<tr><td style="padding:6px 8px;border-bottom:1px solid #eee"><b>{esc(eu_name(r["Name"]))}</b>'
            f' <span style="color:#666;font-size:12px">[{esc(r["Symbol"])} · {esc(eu_exch(r["Symbol"]))}]</span></td>'
            f'<td style="padding:6px 8px;border-bottom:1px solid #eee;text-align:right">{cur}{r["LTP"]:,.2f}</td>'
            f'<td style="padding:6px 8px;border-bottom:1px solid #eee;text-align:right;color:{"#1a7f37" if r["Change%"]>=0 else "#c0392b"}">{r["Change%"]:+.1f}%</td>'
            f'<td style="padding:6px 8px;border-bottom:1px solid #eee;text-align:right">{cur}{r["Box_Top"]:,.2f}</td>'
            f'<td style="padding:6px 8px;border-bottom:1px solid #eee;text-align:center">{r["Position_in_Box%"]:.0f}%</td></tr>'
        )
    return out


eu_dv_hdr = (
    '<tr style="background:#1f2937;color:#fff"><th style="padding:8px;text-align:left">Company [Ticker · Exch]</th>'
    '<th style="padding:8px;text-align:right">LTP</th><th style="padding:8px;text-align:right">Day%</th>'
    '<th style="padding:8px;text-align:right">Box Top</th><th style="padding:8px">Box Pos</th></tr>'
)
eu_bd_total = int((eu_dv["Darvas_Signal"] == "BREAKDOWN_SELL").sum())
eu_bd_names = ", ".join(f'{esc(r["Name"])} ({r["Change%"]:+.1f}%)' for _, r in eu_bd.iterrows())

# Emerging Europe (Istanbul/Prague/Budapest): yfinance has their OHLC but NOT
# their financial statements, so these names produce Darvas momentum signals only
# (no Piotroski/Coffee Can → absent from Component 1). Surface their top breakouts.
_EMERG = {".IS": 5, ".PR": 3, ".BD": 4}  # per-exchange cap so all three show
_em_parts = []
for _suf, _n in _EMERG.items():
    _sub = (
        eu_b[eu_b["Symbol"].astype(str).str.upper().str.endswith(_suf)]
        .sort_values("Change%", ascending=False)
        .head(_n)
    )
    _em_parts.append(_sub)
eu_em = pd.concat(_em_parts).sort_values("Change%", ascending=False) if _em_parts else eu_b.head(0)
try:
    import emerging_fundamentals as _ef

    _emerg_src = _ef.available_source()
except Exception:
    _emerg_src = None
_emerg_note = (
    f"Fundamentals for these markets are now sourced from <b>{esc(_emerg_src)}</b> (yfinance lacks them), "
    "so qualifying names also appear in the fundamental picks above; this block is the momentum view."
    if _emerg_src
    else "These markets have price data but no yfinance financial statements, so they appear here as momentum only — "
    "not in the fundamental picks above. Set <code>FMP_API_KEY</code> or <code>TWELVEDATA_KEY</code> to add their fundamentals."
)
eu_emerging_html = ""
if not eu_em.empty:
    eu_emerging_html = (
        '<p style="font-size:13px;font-weight:700;margin:14px 0 4px">🆕 Emerging-Europe breakouts (Istanbul · Prague · Budapest)</p>'
        f'<p style="font-size:11px;color:#888;margin:0 0 4px">{_emerg_note}</p>'
        f'<table style="width:100%;border-collapse:collapse;font-size:13px">{eu_dv_hdr}{eu_dv_rows(eu_em)}</table>'
    )
eu_intro = (
    "Fresh box breakouts across the broad European universe — price just clearing the box top "
    "(position 100–120%), in an uptrend, up on the day."
    if eu_fresh_mode == "fresh"
    else "Most breakouts today are already extended above their boxes, so no tight/fresh entries qualified — "
    "showing uptrend breakouts nearest their box top (least-extended first)."
)
eu_darvas = (
    '<h2 style="font-size:17px;color:#111;border-bottom:2px solid #eee;padding-bottom:4px;margin:22px 0 8px">📈 🇪🇺 EUROPE DARVAS BREAKOUTS</h2>'
    f'<p style="font-size:12px;color:#666;margin:2px 0 8px">{eu_intro}</p>'
    f'<table style="width:100%;border-collapse:collapse;font-size:13px">{eu_dv_hdr}{eu_dv_rows(eu_fresh) or "<tr><td colspan=5 style=padding:6px;color:#666>None in a clean uptrend today.</td></tr>"}</table>'
    f"{eu_emerging_html}"
    f'<p style="font-size:12px;color:#c0392b;margin:10px 0 2px"><b>🔻 Darvas breakdowns (SELL side, {eu_bd_total} total):</b> {eu_bd_names}</p>'
    '<p style="font-size:11px;color:#888;margin:6px 0">Momentum signals, NOT buy recommendations. European box positions scale differently from NSE/US; treat these as trend context.</p>'
)

# ---------------------------------------------------------------- Europe news sentiment
# Run the free EU RSS sentiment pipeline on the fundamental picks (by company name),
# so Europe gets a market mood, a per-ticker talk table, and its own convergence.
eu_mood, eu_talk, eu_conv_hi, eu_conv_cau = {}, {}, [], []
try:
    from sentiment_pipeline import SentimentPipeline

    _sp = SentimentPipeline()
    eu_mood = _sp.get_market_mood("EU") or {}
    # Score the strongest ~40 EU picks (name-matched); RSS feeds are cached per run.
    # Dedupe by company name so dual listings (e.g. AIR.PA & AIR.F) count once.
    _eu_seen = set()
    for _, r in eu_pick_all.head(40).iterrows():
        sym, nm = r["Symbol"], str(r["Name"])
        _nk = nm.strip().upper()
        if _nk in _eu_seen:
            continue
        s = _sp.get_ticker_sentiment(sym, "EU", company_name=nm)
        if s.n_articles > 0 and s.label != "NO_DATA":
            _eu_seen.add(_nk)
            hl = s.top_headlines[0]["title"] if s.top_headlines else ""
            eu_talk[sym] = {
                "name": nm,
                "label": s.label,
                "score": s.score,
                "n_articles": s.n_articles,
                "headline": hl,
                "exch": eu_exch(sym),
                "tier": "Triple Hit" if r["is_triple"] else "Multi-Screen",
            }
            if s.label == "POSITIVE":
                eu_conv_hi.append(eu_talk[sym])
            elif s.label == "NEGATIVE":
                eu_conv_cau.append(eu_talk[sym])
except Exception as _e:
    eu_mood = {"error": str(_e)[:120]}


# ---------------------------------------------------------------- Convergence
def conv_section():
    out = ""
    # US convergence data
    hi = [c for c in usj["convergence"] if "BOTH BULLISH" in c.get("Convergence", "")]
    cau = [c for c in usj["convergence"] if "NEGATIVE" in c.get("Convergence", "")]
    in_conv = inj.get("convergence", [])
    if not hi and not cau and not in_conv and not eu_conv_hi and not eu_conv_cau:
        return "<p>No fundamental picks had matching news coverage today (India · US · Europe).</p>"
    out += "<p style='margin:6px 0'><b>🇮🇳 India:</b> No fundamental picks had matching news coverage today.</p>"
    out += "<p style='margin:10px 0 4px'><b>🇺🇸 US — 🎯 High-conviction (fundamentally strong AND positive news):</b></p>"
    if hi:
        out += "<ul style='margin:4px 0'>"
        for c in hi:
            out += (
                f"<li style='background:#e8f5e9;padding:4px 8px;margin:3px 0;border-radius:4px'>"
                f"<b>{esc(usname(c['Symbol']).upper())}</b> [{esc(c['Symbol'])} · {esc(usexch(c['Symbol']))}] — "
                f"{esc(c['Tier'])}, news <b>+{c['News_Score']:.2f}</b> ({c['N_Articles']} art.): "
                f"<i>{esc(c['Top_Headline'])}…</i></li>"
            )
        out += "</ul>"
    else:
        out += "<p style='color:#666'>None today.</p>"
    out += "<p style='margin:10px 0 4px'><b>🇺🇸 US — ⚠️ Caution (strong fundamentals BUT negative news):</b></p>"
    if cau:
        out += "<ul style='margin:4px 0'>"
        for c in cau:
            out += (
                f"<li style='background:#fdecea;padding:4px 8px;margin:3px 0;border-radius:4px'>"
                f"<b>{esc(usname(c['Symbol']).upper())}</b> [{esc(c['Symbol'])} · {esc(usexch(c['Symbol']))}] — "
                f"{esc(c['Tier'])}, news <b>{c['News_Score']:.2f}</b> ({c['N_Articles']} art.): "
                f"<i>{esc(c['Top_Headline'])}…</i></li>"
            )
        out += "</ul>"
    else:
        out += "<p style='color:#666'>None today.</p>"
    # Europe convergence (from the EU RSS sentiment pipeline)
    out += "<p style='margin:12px 0 4px'><b>🇪🇺 Europe — 🎯 High-conviction (fundamentally strong AND positive news):</b></p>"
    if eu_conv_hi:
        out += "<ul style='margin:4px 0'>"
        for c in eu_conv_hi:
            out += (
                f"<li style='background:#e8f5e9;padding:4px 8px;margin:3px 0;border-radius:4px'>"
                f"<b>{esc(c['name'].upper())}</b> [{esc(c['exch'])}] — {esc(c['tier'])}, "
                f"news <b>+{c['score']:.2f}</b> ({c['n_articles']} art.): <i>{esc(c['headline'])[:90]}…</i></li>"
            )
        out += "</ul>"
    else:
        out += "<p style='color:#666'>None today.</p>"
    out += "<p style='margin:10px 0 4px'><b>🇪🇺 Europe — ⚠️ Caution (strong fundamentals BUT negative news):</b></p>"
    if eu_conv_cau:
        out += "<ul style='margin:4px 0'>"
        for c in eu_conv_cau:
            out += (
                f"<li style='background:#fdecea;padding:4px 8px;margin:3px 0;border-radius:4px'>"
                f"<b>{esc(c['name'].upper())}</b> [{esc(c['exch'])}] — {esc(c['tier'])}, "
                f"news <b>{c['score']:.2f}</b> ({c['n_articles']} art.): <i>{esc(c['headline'])[:90]}…</i></li>"
            )
        out += "</ul>"
    else:
        out += "<p style='color:#666'>None today.</p>"
    return out


# ---------------------------------------------------------------- Component 2
def mood_html(j, region):
    m = j["mood"]
    col = {"POSITIVE": "#2e7d32", "NEGATIVE": "#c62828", "NEUTRAL": "#f57f17"}.get(
        m["mood"], "#555"
    )
    outlets = " · ".join(m.get("by_outlet", {}).keys()) or "wire feeds"
    return (
        f"<p style='margin:6px 0'><b>{region} market mood:</b> "
        f"<b style='color:{col}'>{m['mood']} ({m['score']:+.2f})</b> from {m['n_articles']} live articles ({esc(outlets)})</p>"
    )


def talk_table(j, cur):
    rows = ""
    for sym, t in j["talk"].items():
        if t.get("n_articles", 0) <= 0 or t.get("label") == "NO_DATA":
            continue
        lab = t["label"]
        col = {"POSITIVE": "#2e7d32", "NEGATIVE": "#c62828", "NEUTRAL": "#f57f17"}.get(lab, "#555")
        hl = (t.get("headlines") or [""])[0]
        nm = usname(sym).upper() if cur == "$" else sym
        rows += (
            f"<tr><td style='padding:6px 8px;border-bottom:1px solid #eee'><b>{esc(nm)}</b> "
            f"<span style='color:#666;font-size:12px'>[{esc(sym)}]</span></td>"
            f"<td style='padding:6px 8px;border-bottom:1px solid #eee;color:{col}'><b>{esc(lab)}</b></td>"
            f"<td style='padding:6px 8px;border-bottom:1px solid #eee;text-align:right'>{t['score']:+.2f}</td>"
            f"<td style='padding:6px 8px;border-bottom:1px solid #eee;text-align:center'>{t['n_articles']}</td>"
            f"<td style='padding:6px 8px;border-bottom:1px solid #eee;font-size:12px'>{esc(hl)[:90]}</td></tr>"
        )
    hdr = (
        "<tr style='background:#374151;color:#fff'><th style='padding:8px;text-align:left'>Company [Ticker]</th>"
        "<th style='padding:8px'>Sentiment</th><th style='padding:8px'>Score</th>"
        "<th style='padding:8px'>Articles</th><th style='padding:8px;text-align:left'>Top headline</th></tr>"
    )
    body = (
        rows
        or "<tr><td colspan=5 style='padding:6px 8px;color:#666'>No per-ticker news coverage matched today's picks.</td></tr>"
    )
    return f"<table style='width:100%;border-collapse:collapse;font-size:13px'>{hdr}{body}</table>"


def eu_mood_html():
    if not eu_mood or "mood" not in eu_mood:
        return (
            "<p style='margin:6px 0'><b>🇪🇺 Europe market mood:</b> feed unavailable this run.</p>"
        )
    col = {"POSITIVE": "#2e7d32", "NEGATIVE": "#c62828", "NEUTRAL": "#f57f17"}.get(
        eu_mood["mood"], "#555"
    )
    outlets = " · ".join(eu_mood.get("by_outlet", {}).keys()) or "EU feeds"
    return (
        f"<p style='margin:6px 0'><b>🇪🇺 Europe market mood:</b> "
        f"<b style='color:{col}'>{eu_mood['mood']} ({eu_mood['score']:+.2f})</b> "
        f"from {eu_mood['n_articles']} live articles ({esc(outlets)})</p>"
    )


def eu_talk_table():
    rows = ""
    for sym, t in sorted(eu_talk.items(), key=lambda x: -x[1]["score"]):
        col = {"POSITIVE": "#2e7d32", "NEGATIVE": "#c62828", "NEUTRAL": "#f57f17"}.get(
            t["label"], "#555"
        )
        rows += (
            f"<tr><td style='padding:6px 8px;border-bottom:1px solid #eee'><b>{esc(t['name'].upper())}</b> "
            f"<span style='color:#666;font-size:12px'>[{esc(sym)} · {esc(t['exch'])}]</span></td>"
            f"<td style='padding:6px 8px;border-bottom:1px solid #eee;color:{col}'><b>{esc(t['label'])}</b></td>"
            f"<td style='padding:6px 8px;border-bottom:1px solid #eee;text-align:right'>{t['score']:+.2f}</td>"
            f"<td style='padding:6px 8px;border-bottom:1px solid #eee;text-align:center'>{t['n_articles']}</td>"
            f"<td style='padding:6px 8px;border-bottom:1px solid #eee;font-size:12px'>{esc(t['headline'])[:90]}</td></tr>"
        )
    hdr = (
        "<tr style='background:#374151;color:#fff'><th style='padding:8px;text-align:left'>Company [Ticker · Exch]</th>"
        "<th style='padding:8px'>Sentiment</th><th style='padding:8px'>Score</th>"
        "<th style='padding:8px'>Articles</th><th style='padding:8px;text-align:left'>Top headline</th></tr>"
    )
    body = (
        rows
        or "<tr><td colspan=5 style='padding:6px 8px;color:#666'>No European pick was named in today's news flow.</td></tr>"
    )
    return f"<table style='width:100%;border-collapse:collapse;font-size:13px'>{hdr}{body}</table>"


def _read_frag(name):
    """Darvas fragments: prefer a stable ~/Downloads copy, fall back to the
    session scratchpad, else an empty-section placeholder."""
    for p in (os.path.expanduser(f"~/Downloads/{name}"), os.path.join(SP, name)):
        if os.path.exists(p):
            return open(p).read()
    return f'<p style="font-size:12px;color:#888">📈 Darvas fragment {name} not found this run.</p>'


in_darvas = _read_frag("in_darvas.html")
us_darvas = _read_frag("us_darvas.html")

today = datetime.date(2026, 7, 9)
datestr = today.strftime("%d %b %Y")
now = datetime.datetime.now().strftime("%d %b %Y %H:%M")
n_conv = len([c for c in usj["convergence"] if "BOTH BULLISH" in c.get("Convergence", "")]) + len(
    eu_conv_hi
)
n_fund = len(us_picks) + len(eu_pick_all)
mood_lbl = usj["mood"]["mood"]

SECT = "<h2 style='font-size:18px;color:#111;border-bottom:2px solid #e5e7eb;padding-bottom:5px;margin:26px 0 10px'>{}</h2>"

email = f"""<div style="font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:860px;margin:0 auto;color:#1a1a1a;line-height:1.5">
<div style="background:#111827;color:#fff;padding:18px 22px;border-radius:8px">
  <div style="font-size:22px;font-weight:800">📈 Daily Market Report</div>
  <div style="font-size:13px;color:#cbd5e1;margin-top:3px">{datestr} · generated {now} IST · India · US · Europe</div>
</div>

<div style="background:#fff8e1;border:1px solid #ffe082;border-radius:6px;padding:10px 14px;margin:12px 0;font-size:12px;color:#7a5c00">
  <b>⚠️ Disclaimer:</b> Educational / research use only. This is <b>NOT investment advice</b>. Signals come from mechanical
  screeners and noisy news sentiment. Past results ≠ future returns. Consult a SEBI-registered advisor before acting.
</div>

<div style="display:flex;gap:10px;flex-wrap:wrap;margin:12px 0">
  <div style="flex:1;min-width:230px;background:#f3f4f6;border-radius:6px;padding:10px 14px;font-size:13px">
    <b>🇮🇳 India snapshot</b><br>Nifty 50: <b>23,882</b> vs 200-DMA 24,858 — <span style="color:#c0392b">BELOW (cautious)</span><br>
    India VIX: <b>14.68</b> (calm)</div>
  <div style="flex:1;min-width:230px;background:#f3f4f6;border-radius:6px;padding:10px 14px;font-size:13px">
    <b>🇺🇸 US snapshot</b><br>S&P 500: <b>7,483</b> vs 200-DMA 6,956 — <span style="color:#1a7f37">ABOVE (bullish)</span></div>
  <div style="flex:1;min-width:230px;background:#f3f4f6;border-radius:6px;padding:10px 14px;font-size:13px">
    <b>🇪🇺 Europe snapshot</b><br>EURO STOXX 50: <b>6,205</b> vs 200-DMA 5,848 — <span style="color:#1a7f37">ABOVE (bullish)</span></div>
</div>

{SECT.format("⭐ CONVERGENCE — Where Fundamentals & The Street Agree")}
<p style="font-size:12px;color:#666;margin:2px 0 8px">The highest-value cross-check: names flagged by BOTH the fundamental screeners and the news sentiment engine. <b>Convergence is explicitly NOT a buy signal</b> — it only marks where two independent, imperfect views happen to line up.</p>
{conv_section()}

{SECT.format("📊 COMPONENT 1 — Stock Picks Based on Fundamentals")}
<p style="font-size:12px;color:#666;margin:2px 0 8px">Multi-screen confluence from the full-universe scan. <b>Triple Hit</b> = Darvas + Piotroski≥7 + Coffee Can; <b>Multi-Screen</b> = 3+ of the 6 screeners. PE Zone is sector-relative (pick PE vs. its sector median across the scanned universe).</p>
{us_c1}
<h3 style="margin:18px 0 4px;font-size:15px">🇪🇺 Europe</h3>
{eu_c1}
<h3 style="margin:18px 0 4px;font-size:15px">🇮🇳 India</h3>
{in_c1}

{SECT.format("📈 Darvas Breakouts")}
<p style="font-size:12px;color:#666;margin:2px 0 8px">Momentum signals — fresh box breakouts only (price just clearing the box top, in an uptrend). These are <b>NOT buy recommendations</b>; false breakouts are common. India first, then US, then Europe.</p>
{in_darvas}
{us_darvas}
{eu_darvas}

{SECT.format("🗞️ COMPONENT 2 — Talk on the Street")}
{mood_html(inj, "🇮🇳 India")}
{talk_table(inj, "₹")}
{mood_html(usj, "🇺🇸 US")}
{talk_table(usj, "$")}
{eu_mood_html()}
{eu_talk_table()}
<p style="font-size:11px;color:#888;margin:4px 0">🇪🇺 Europe sentiment: free English-language EU business feeds (CityAM · Guardian · BBC · CNBC-Europe · Euronews · FT · DW), company-name matched, VADER-scored. General market-news flow, so only picks named in today's headlines appear.</p>
<p style="font-size:11px;color:#666;margin:6px 0"><span style="color:#2e7d32">■</span> POSITIVE &nbsp; <span style="color:#f57f17">■</span> NEUTRAL &nbsp; <span style="color:#c62828">■</span> NEGATIVE</p>

{SECT.format("📚 Reference — The Six Screeners")}
<ul style="font-size:12px;color:#444;line-height:1.7">
<li><b>Darvas Box</b> — momentum: buys breakouts above a consolidation box (higher reward, higher whipsaw risk).</li>
<li><b>Piotroski F-Score</b> — 9-point financial-health check; ≥7 = fundamentally strong, lower balance-sheet risk.</li>
<li><b>Coffee Can</b> — buy-and-hold quality: durable revenue growth + high ROE (low churn, long horizon).</li>
<li><b>Bull Cartel</b> — combined momentum + quality confluence screen.</li>
<li><b>Magic Formula</b> — Greenblatt: high ROIC + high earnings yield (value + quality).</li>
<li><b>Golden Cross</b> — 50-DMA crossing above 200-DMA; trend-confirmation overlay.</li>
</ul>

<div style="background:#fff8e1;border:1px solid #ffe082;border-radius:6px;padding:12px 16px;margin:16px 0;font-size:12px;color:#7a5c00">
  <b>⚠️ Full disclaimer.</b> This report is for <b>educational and research purposes only</b> and is <b>NOT investment advice</b>.
  It combines mechanical screeners (which ignore qualitative context) with automated news sentiment (which is noisy and can
  misread headlines). Convergence between the two is <b>NOT a buy signal</b>. Past performance does not guarantee future results.
  Markets carry risk of loss. Always do your own due diligence and consult a <b>SEBI-registered investment advisor</b> before making
  any investment decision.
</div>
<p style="font-size:11px;color:#999;text-align:center;margin:10px 0">Automated daily report · fundamentals + street-talk + Darvas momentum · India &amp; US</p>
</div>"""

out_path = os.path.expanduser(f"~/Downloads/daily_market_report_{today.isoformat()}.html")
open(out_path, "w").write(email)

# ---------------------------------------------------------------- Artifact page
# The report is a "printed market sheet" (light document, single-theme by intent)
# resting on a theme-aware desk. Body content only — the Artifact host adds the
# doctype/head/body skeleton.
import re as _re


def wrap_tables(hpage):
    return _re.sub(r"(<table\b.*?</table>)", r'<div class="trail">\1</div>', hpage, flags=_re.S)


artifact = f"""<style>
:root {{
  --desk:#e9ebef; --desk-edge:#d6dae1; --ink:#0f1720; --muted:#5b6672;
  --accent:#0e7490; --paper:#ffffff;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --desk:#0e1218; --desk-edge:#232a33; --ink:#e7ebf0; --muted:#8b97a5; --accent:#38bdf8; }}
}}
:root[data-theme="light"] {{ --desk:#e9ebef; --desk-edge:#d6dae1; --ink:#0f1720; --muted:#5b6672; --accent:#0e7490; }}
:root[data-theme="dark"]  {{ --desk:#0e1218; --desk-edge:#232a33; --ink:#e7ebf0; --muted:#8b97a5; --accent:#38bdf8; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; }}
.desk {{
  background:var(--desk); min-height:100vh; padding:28px 16px 56px;
  -webkit-font-smoothing:antialiased; font-variant-numeric:tabular-nums;
}}
.deskhead {{
  max-width:900px; margin:0 auto 16px; display:flex; align-items:baseline;
  justify-content:space-between; gap:12px; flex-wrap:wrap; color:var(--ink);
}}
.wordmark {{ font-family:Georgia,"Times New Roman",serif; font-size:15px; font-weight:700;
  letter-spacing:.02em; }}
.wordmark .tick {{ color:var(--accent); }}
.eyebrow {{ font:600 11px/1 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  text-transform:uppercase; letter-spacing:.16em; color:var(--muted); }}
.paper {{
  max-width:900px; margin:0 auto; background:var(--paper); color:#1a1a1a;
  border:1px solid var(--desk-edge); border-radius:12px;
  box-shadow:0 1px 2px rgba(15,23,32,.06),0 12px 34px rgba(15,23,32,.10);
  padding:26px 30px 30px; overflow:hidden;
}}
.paper table {{ font-variant-numeric:tabular-nums; min-width:640px; }}
/* every table sits in its own scroll rail; the page body never scrolls sideways */
.paper .trail {{ overflow-x:auto; max-width:100%; scrollbar-width:thin; margin:2px 0; }}
.deskfoot {{ max-width:900px; margin:18px auto 0; color:var(--muted);
  font:400 11px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif; text-align:center; }}
@media (max-width:640px) {{ .paper {{ padding:18px 14px 22px; border-radius:9px; }} }}
</style>
<div class="desk">
  <div class="deskhead">
    <div class="wordmark"><span class="tick">▲</span> MARKET DAILY</div>
    <div class="eyebrow">Fundamentals · Street Talk · Darvas · IN / US / EU</div>
  </div>
  <article class="paper">
  {wrap_tables(email)}
  </article>
  <div class="deskfoot">Educational / research only — not investment advice. Regenerated each weekday morning.</div>
</div>"""
art_path = os.path.expanduser("~/Downloads/daily_market_report_artifact.html")
open(art_path, "w").write(artifact)

# also print subject line for the sender step
subj = f"Daily Market Report — {datestr} | Fundamentals: {n_fund} picks | Street mood: {mood_lbl} | Convergence: {n_conv}"
print("SUBJECT:", subj)
print("PATH:", out_path)
print("ARTIFACT:", art_path)
print("BYTES:", len(email), "ARTBYTES:", len(artifact))
