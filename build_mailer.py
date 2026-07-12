#!/usr/bin/env python3
# build_mailer.py
# ===============
# Assemble the Daily Market Brief HTML (+ plain-text summary) from the latest
# cached outputs, so the morning job / any run produces a consistent mailer:
#   1. India screener (liquid picks, with liquidity tier)
#   2. India Cash Conversion Cycle screen (screener.in 228040, low-CCC + liquidity)
#   3. US screener (combined_US_*.json) + fundamentals/news convergence
#   4. AI-based hybrid screener picks (auto_screener.py), India + US
#   5. global-market-scanners Triple/Multi-Screen Hits + ML signal, India + US
#      (github.com/herrrickshaw/global-market-scanners, cloned to ~/repos/)
#   6. Europe full-universe screen (full_european_market_scan.py, 22 exchanges)
#   7. Global momentum top-15 (all markets)
#   8. 20-market 5-year scoreboard
#   + educational-only / NOT-investment-advice disclaimer.
#
#   from build_mailer import build; subj, text, html = build()

from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

import liquidity as liq
import market_performance as mp
import run_global_analysis as rga

GMS_REPO = os.path.expanduser("~/repos/global-market-scanners")

GMS_REPO = os.path.expanduser("~/repos/global-market-scanners")

_COL = {"High": "#1b7f37", "Medium": "#b8860b", "Low": "#b00"}


def _tv(x):
    return f"${x/1e6:.1f}M" if pd.notna(x) else "—"


def _table(headers, rows):
    h = "".join(f"<th align='left' style='padding:5px 8px'>{x}</th>" for x in headers)
    return (
        f"<table style='border-collapse:collapse;width:100%;font-size:13px'>"
        f"<tr style='background:#eef'>{h}</tr>{''.join(rows)}</table>"
    )


def _combined_report(market: str):
    """Load the latest combined_report_results/combined_<MKT>_*.json, if any."""
    cj = sorted(glob.glob(f"combined_report_results/combined_{market}_*.json"))
    if not cj:
        return None
    return json.load(open(cj[-1]))


def _picks_rows(d: dict, market: str, limit: int = 12):
    if not d or not d.get("picks"):
        return []
    p = pd.DataFrame(d["picks"])
    p["Market"] = market
    p = liq.annotate(p)
    rank = {"Triple Hit": 0, "Multi-Screen": 1, "Single-Screen": 2}
    lr = {"High": 0, "Medium": 1, "Low": 2, "Unknown": 3}
    p["_o"] = p["Tier"].map(rank).fillna(3)
    p["_l"] = p["Liquidity"].map(lr)
    rows = []
    for _, r in p.sort_values(["_o", "_l"]).head(limit).iterrows():
        rows.append(
            f"<tr><td style='padding:4px 8px'><b>{r.Symbol}</b></td>"
            f"<td>{r.Tier}</td><td>{r.Screens}</td><td>{_tv(r.get('Turnover_USD'))}</td>"
            f"<td style='color:{_COL.get(r.Liquidity,'#777')};font-weight:600'>{r.Liquidity}</td></tr>"
        )
    return rows


def _convergence_rows(d: dict):
    if not d or not d.get("convergence"):
        return []
    rows = []
    for r in d["convergence"]:
        col = "#1b7f37" if "BULLISH" in r.get("Convergence", "") else "#b8860b"
        rows.append(
            f"<tr><td style='padding:4px 8px'><b>{r['Symbol']}</b></td>"
            f"<td style='color:{col};font-weight:600'>{r['Convergence']}</td>"
            f"<td>{r['Fundamental']}</td><td>{r['News']} ({r['News_Score']:+.2f})</td>"
            f"<td style='font-size:11px'>{r.get('Top_Headline','')[:60]}</td></tr>"
        )
    return rows


def _ai_html(label: str, ai: dict | None) -> str:
    if not ai or not ai["picks"]:
        return f"<p style='font-size:12px;color:#666'>{label}: no discovered screen cached.</p>"
    picks = ", ".join(f"<b>{s}</b>" for s in ai["picks"])
    rl = " (RL-corrected)" if ai["rl_correction"] else ""
    warn = ""
    if ai["n"] <= 2:
        warn = " ⚠️ thin sample — treat as noise, not a signal."
    return (
        f"<p style='font-size:12px;margin:4px 0'><b>{label}</b> — {ai['tape']} tape, "
        f"reward {ai['reward']:.2f}{rl}, {ai['n']} names, {ai['overlap']*100:.0f}% overlap "
        f"w/ known-good.{warn}<br>{picks}</p>"
    )


def _ai_screener_section(market: str):
    """auto_screener.py's discovered rule + top picks for a market, if cached."""
    path = f"cache_seed/discovered_screens/{market}.json"
    try:
        d = json.load(open(path))
    except Exception:
        return None
    regime = d.get("regime", {})
    metrics = d.get("metrics", {})
    picks = d.get("picks", [])
    return {
        "tape": regime.get("tape", "n/a"),
        "reward": metrics.get("reward", 0),
        "n": metrics.get("n", 0),
        "overlap": metrics.get("overlap", 0),
        "rl_correction": d.get("rl_correction_applied", False),
        "picks": picks[:10],
    }


def _gms_scan(market: str):
    """Latest global-market-scanners full-universe scan for a market (US/IN),
    from ~/repos/global-market-scanners/{us_full_scan,indian_full_scan}/*.xlsx."""
    folder = "us_full_scan" if market == "US" else "indian_full_scan"
    files = sorted(glob.glob(os.path.join(GMS_REPO, folder, "*.xlsx")))
    if not files:
        return None
    path = files[-1]
    try:
        xl = pd.ExcelFile(path)
        triple = xl.parse("Triple_Hits") if "Triple_Hits" in xl.sheet_names else pd.DataFrame()
        multi = (
            xl.parse("Multi_Screen_Hits")
            if "Multi_Screen_Hits" in xl.sheet_names
            else pd.DataFrame()
        )
        ml_bull = xl.parse("ML_Bullish") if "ML_Bullish" in xl.sheet_names else pd.DataFrame()
        if "Note" in triple.columns:  # placeholder sheet when there are 0 triple hits
            triple = pd.DataFrame()
        return {
            "triple": triple,
            "multi": multi,
            "ml_bull": ml_bull,
            "file": os.path.basename(path),
        }
    except Exception:
        return None


def _gms_html(label: str, scan: dict | None) -> str:
    if not scan:
        return (
            f"<p style='font-size:12px;color:#666'>{label}: no global-market-scanners run cached "
            f"(clone/run github.com/herrrickshaw/global-market-scanners).</p>"
        )
    parts = [f"<p style='font-size:12px;margin:6px 0 2px'><b>{label}</b> — {scan['file']}</p>"]
    if not scan["triple"].empty:
        names = ", ".join(f"<b>{s}</b>" for s in scan["triple"]["Symbol"].head(10))
        parts.append(f"<p style='font-size:12px;margin:2px 0'>🎯 Triple Hits: {names}</p>")
    else:
        parts.append("<p style='font-size:12px;margin:2px 0;color:#666'>No Triple Hits today.</p>")
    if not scan["multi"].empty:
        names = ", ".join(f"<b>{s}</b>" for s in scan["multi"]["Symbol"].head(10))
        parts.append(
            f"<p style='font-size:12px;margin:2px 0'>Multi-Screen Hits (top 10 of {len(scan['multi'])}): {names}</p>"
        )
    if not scan["ml_bull"].empty:
        top = scan["ml_bull"].sort_values("ML_Confidence", ascending=False).head(5)
        names = []
        for _, r in top.iterrows():
            pred = r.get("ML_Pred_Ret%")
            names.append(
                f"<b>{r['Symbol']}</b> ({pred:+.1f}%)"
                if pd.notna(pred)
                else f"<b>{r['Symbol']}</b>"
            )
        parts.append(
            f"<p style='font-size:12px;margin:2px 0'>ML Ridge signal, top bullish: {', '.join(names)}</p>"
        )
    return "".join(parts)


def _europe_scan():
    """Latest full_european_market_scan.py run, from european_scan/*.xlsx
    (22-exchange broad universe -- data/europe_broad_list.csv)."""
    files = glob.glob("european_scan/european_market_scan_*.xlsx")
    if not files:
        return None
    # sort by mtime, not filename -- different --label values don't sort
    # chronologically as strings (e.g. "europe_restart" > "europe_final")
    path = max(files, key=os.path.getmtime)
    try:
        xl = pd.ExcelFile(path)
        triple = xl.parse("Triple_Hits") if "Triple_Hits" in xl.sheet_names else pd.DataFrame()
        darvas = (
            xl.parse("Darvas_Signals") if "Darvas_Signals" in xl.sheet_names else pd.DataFrame()
        )
        all_stocks = xl.parse("All_Stocks") if "All_Stocks" in xl.sheet_names else pd.DataFrame()
        return {
            "triple": triple,
            "darvas": darvas,
            "n_scanned": len(all_stocks),
            "file": os.path.basename(path),
        }
    except Exception:
        return None


def _europe_html(scan: dict | None) -> str:
    if not scan:
        return "<p style='font-size:12px;color:#666'>No Europe scan cached (run full_european_market_scan.py).</p>"
    parts = [
        f"<p style='font-size:12px;margin:2px 0 6px'>{scan['n_scanned']} stocks scanned across 22 exchanges — {scan['file']}</p>"
    ]
    if not scan["triple"].empty:
        rows = []
        for _, r in scan["triple"].iterrows():
            rows.append(
                f"<tr><td style='padding:4px 8px'><b>{r['Symbol']}</b></td><td>{r['Name']}</td>"
                f"<td>{r['Sector']}</td><td>{r.get('Piotroski_Score','')}/9</td>"
                f"<td>{r.get('CoffeeCan_Score','')}</td><td>€{r['LTP']:.2f}</td></tr>"
            )
        parts.append(_table(["Symbol", "Name", "Sector", "F-Score", "CoffeeCan", "LTP"], rows))
    else:
        parts.append("<p style='font-size:12px;color:#666'>No Triple Hits today.</p>")
    if not scan["darvas"].empty:
        top = scan["darvas"].sort_values("Upside_to_Top%", ascending=False).head(8)
        names = ", ".join(f"<b>{r.Symbol}</b>" for _, r in top.iterrows())
        parts.append(
            f"<p style='font-size:12px;margin:6px 0 0'>{len(scan['darvas'])} Darvas breakouts. Top by upside-to-box-top: {names}</p>"
        )
    return "".join(parts)


def _combined_report(market: str):
    """Load the latest combined_report_results/combined_<MKT>_*.json, if any."""
    cj = sorted(glob.glob(f"combined_report_results/combined_{market}_*.json"))
    if not cj:
        return None
    return json.load(open(cj[-1]))


def _picks_rows(d: dict, market: str, limit: int = 12):
    if not d or not d.get("picks"):
        return []
    p = pd.DataFrame(d["picks"])
    p["Market"] = market
    p = liq.annotate(p)
    rank = {"Triple Hit": 0, "Multi-Screen": 1, "Single-Screen": 2}
    lr = {"High": 0, "Medium": 1, "Low": 2, "Unknown": 3}
    p["_o"] = p["Tier"].map(rank).fillna(3)
    p["_l"] = p["Liquidity"].map(lr)
    rows = []
    for _, r in p.sort_values(["_o", "_l"]).head(limit).iterrows():
        rows.append(
            f"<tr><td style='padding:4px 8px'><b>{r.Symbol}</b></td>"
            f"<td>{r.Tier}</td><td>{r.Screens}</td><td>{_tv(r.get('Turnover_USD'))}</td>"
            f"<td style='color:{_COL.get(r.Liquidity,'#777')};font-weight:600'>{r.Liquidity}</td></tr>"
        )
    return rows


def _convergence_rows(d: dict):
    if not d or not d.get("convergence"):
        return []
    rows = []
    for r in d["convergence"]:
        col = "#1b7f37" if "BULLISH" in r.get("Convergence", "") else "#b8860b"
        rows.append(
            f"<tr><td style='padding:4px 8px'><b>{r['Symbol']}</b></td>"
            f"<td style='color:{col};font-weight:600'>{r['Convergence']}</td>"
            f"<td>{r['Fundamental']}</td><td>{r['News']} ({r['News_Score']:+.2f})</td>"
            f"<td style='font-size:11px'>{r.get('Top_Headline','')[:60]}</td></tr>"
        )
    return rows


def _ai_html(label: str, ai: dict | None) -> str:
    if not ai or not ai["picks"]:
        return f"<p style='font-size:12px;color:#666'>{label}: no discovered screen cached.</p>"
    picks = ", ".join(f"<b>{s}</b>" for s in ai["picks"])
    rl = " (RL-corrected)" if ai["rl_correction"] else ""
    warn = ""
    if ai["n"] <= 2:
        warn = " ⚠️ thin sample — treat as noise, not a signal."
    return (
        f"<p style='font-size:12px;margin:4px 0'><b>{label}</b> — {ai['tape']} tape, "
        f"reward {ai['reward']:.2f}{rl}, {ai['n']} names, {ai['overlap']*100:.0f}% overlap "
        f"w/ known-good.{warn}<br>{picks}</p>"
    )


def _ai_screener_section(market: str):
    """auto_screener.py's discovered rule + top picks for a market, if cached."""
    path = f"cache_seed/discovered_screens/{market}.json"
    try:
        d = json.load(open(path))
    except Exception:
        return None
    regime = d.get("regime", {})
    metrics = d.get("metrics", {})
    picks = d.get("picks", [])
    return {
        "tape": regime.get("tape", "n/a"),
        "reward": metrics.get("reward", 0),
        "n": metrics.get("n", 0),
        "overlap": metrics.get("overlap", 0),
        "rl_correction": d.get("rl_correction_applied", False),
        "picks": picks[:10],
    }


def _gms_scan(market: str):
    """Latest global-market-scanners full-universe scan for a market (US/IN),
    from ~/repos/global-market-scanners/{us_full_scan,indian_full_scan}/*.xlsx."""
    folder = "us_full_scan" if market == "US" else "indian_full_scan"
    files = sorted(glob.glob(os.path.join(GMS_REPO, folder, "*.xlsx")))
    if not files:
        return None
    path = files[-1]
    try:
        xl = pd.ExcelFile(path)
        triple = xl.parse("Triple_Hits") if "Triple_Hits" in xl.sheet_names else pd.DataFrame()
        multi = (
            xl.parse("Multi_Screen_Hits")
            if "Multi_Screen_Hits" in xl.sheet_names
            else pd.DataFrame()
        )
        ml_bull = xl.parse("ML_Bullish") if "ML_Bullish" in xl.sheet_names else pd.DataFrame()
        if "Note" in triple.columns:  # placeholder sheet when there are 0 triple hits
            triple = pd.DataFrame()
        return {
            "triple": triple,
            "multi": multi,
            "ml_bull": ml_bull,
            "file": os.path.basename(path),
        }
    except Exception:
        return None


def _gms_html(label: str, scan: dict | None) -> str:
    if not scan:
        return (
            f"<p style='font-size:12px;color:#666'>{label}: no global-market-scanners run cached "
            f"(clone/run github.com/herrrickshaw/global-market-scanners).</p>"
        )
    parts = [f"<p style='font-size:12px;margin:6px 0 2px'><b>{label}</b> — {scan['file']}</p>"]
    if not scan["triple"].empty:
        names = ", ".join(f"<b>{s}</b>" for s in scan["triple"]["Symbol"].head(10))
        parts.append(f"<p style='font-size:12px;margin:2px 0'>🎯 Triple Hits: {names}</p>")
    else:
        parts.append("<p style='font-size:12px;margin:2px 0;color:#666'>No Triple Hits today.</p>")
    if not scan["multi"].empty:
        names = ", ".join(f"<b>{s}</b>" for s in scan["multi"]["Symbol"].head(10))
        parts.append(
            f"<p style='font-size:12px;margin:2px 0'>Multi-Screen Hits (top 10 of {len(scan['multi'])}): {names}</p>"
        )
    if not scan["ml_bull"].empty:
        top = scan["ml_bull"].sort_values("ML_Confidence", ascending=False).head(5)
        names = []
        for _, r in top.iterrows():
            pred = r.get("ML_Pred_Ret%")
            names.append(
                f"<b>{r['Symbol']}</b> ({pred:+.1f}%)"
                if pd.notna(pred)
                else f"<b>{r['Symbol']}</b>"
            )
        parts.append(
            f"<p style='font-size:12px;margin:2px 0'>ML Ridge signal, top bullish: {', '.join(names)}</p>"
        )
    return "".join(parts)


def _europe_scan():
    """Latest full_european_market_scan.py run, from european_scan/*.xlsx
    (22-exchange broad universe -- data/europe_broad_list.csv)."""
    files = glob.glob("european_scan/european_market_scan_*.xlsx")
    if not files:
        return None
    # sort by mtime, not filename -- different --label values don't sort
    # chronologically as strings (e.g. "europe_restart" > "europe_final")
    path = max(files, key=os.path.getmtime)
    try:
        xl = pd.ExcelFile(path)
        triple = xl.parse("Triple_Hits") if "Triple_Hits" in xl.sheet_names else pd.DataFrame()
        darvas = (
            xl.parse("Darvas_Signals") if "Darvas_Signals" in xl.sheet_names else pd.DataFrame()
        )
        all_stocks = xl.parse("All_Stocks") if "All_Stocks" in xl.sheet_names else pd.DataFrame()
        return {
            "triple": triple,
            "darvas": darvas,
            "n_scanned": len(all_stocks),
            "file": os.path.basename(path),
        }
    except Exception:
        return None


def _europe_html(scan: dict | None) -> str:
    if not scan:
        return "<p style='font-size:12px;color:#666'>No Europe scan cached (run full_european_market_scan.py).</p>"
    parts = [
        f"<p style='font-size:12px;margin:2px 0 6px'>{scan['n_scanned']} stocks scanned across 22 exchanges — {scan['file']}</p>"
    ]
    if not scan["triple"].empty:
        rows = []
        for _, r in scan["triple"].iterrows():
            rows.append(
                f"<tr><td style='padding:4px 8px'><b>{r['Symbol']}</b></td><td>{r['Name']}</td>"
                f"<td>{r['Sector']}</td><td>{r.get('Piotroski_Score','')}/9</td>"
                f"<td>{r.get('CoffeeCan_Score','')}</td><td>€{r['LTP']:.2f}</td></tr>"
            )
        parts.append(_table(["Symbol", "Name", "Sector", "F-Score", "CoffeeCan", "LTP"], rows))
    else:
        parts.append("<p style='font-size:12px;color:#666'>No Triple Hits today.</p>")
    if not scan["darvas"].empty:
        top = scan["darvas"].sort_values("Upside_to_Top%", ascending=False).head(8)
        names = ", ".join(f"<b>{r.Symbol}</b>" for _, r in top.iterrows())
        parts.append(
            f"<p style='font-size:12px;margin:6px 0 0'>{len(scan['darvas'])} Darvas breakouts. Top by upside-to-box-top: {names}</p>"
        )
    return "".join(parts)


def build():
    today = _dt.date.today().strftime("%d %b %Y")

    # 1. India screener
    d_in = _combined_report("IN")
    picks_rows = _picks_rows(d_in, "IN")
    mood = (d_in or {}).get("mood", {"mood": "n/a", "score": 0, "n_articles": 0})
    conv_in_rows = _convergence_rows(d_in)

    # 1b. US screener + convergence
    d_us = _combined_report("US")
    us_picks_rows = _picks_rows(d_us, "US")
    us_mood = (d_us or {}).get("mood", {"mood": "n/a", "score": 0, "n_articles": 0})
    conv_us_rows = _convergence_rows(d_us)

    # 1c. AI-based hybrid screener (auto_screener.py) — India + US
    ai_in = _ai_screener_section("IN")
    ai_us = _ai_screener_section("US")

    # 1d. global-market-scanners full-universe scan — India + US
    gms_in = _gms_scan("IN")
    gms_us = _gms_scan("US")

    # 1e. Europe full-universe scan (22 exchanges)
    europe = _europe_scan()

    # 1f. Regime picks — each market screened by its best-backtested filter
    try:
        import regime_picks

        regime_section = regime_picks.section_html()
    except Exception:
        regime_section = ""

    # 1g. Data verification — cross-check this report's yfinance-sourced
    # prices (the same regime picks just used above) against an independent
    # authoritative source: Databento (US, paid/metered) and NSE Bhavcopy
    # (India, free/official). See verify_data.py's own docstring for why
    # Databento can't cover India (confirmed against its real dataset list,
    # not marketing copy) and Bhavcopy substitutes there instead.
    try:
        import regime_picks
        import verify_data

        us_picks_for_verify = [
            {"symbol": p["symbol"], "price": p["price"]} for p in regime_picks.picks_for("US")
        ]
        in_picks_for_verify = [
            {"symbol": p["symbol"], "price": p["price"]} for p in regime_picks.picks_for("IN")
        ]
        us_verify = verify_data.verify_us(us_picks_for_verify)
        in_verify = verify_data.verify_india(in_picks_for_verify)
        verification_section = verify_data.verification_html(us_verify, in_verify)
        verification_text_block = verify_data.verification_text(us_verify, in_verify)
    except Exception as e:  # noqa: BLE001 - verification is an annotation, never blocks the mailer
        verification_section = (
            f"<p style='font-size:11px;color:#888'>Data verification unavailable: {e}</p>"
        )
        verification_text_block = "Data verification unavailable."

    # 2. India CCC screen (screener.in)
    ccc_rows = []
    try:
        cdf = pd.read_parquet("cache_seed/india_ccc_screen.parquet")
        cdf["Cash_Cycle"] = pd.to_numeric(cdf["Cash_Cycle"], errors="coerce")
        if "Liquidity" not in cdf.columns:
            # ccc_screen() doesn't annotate liquidity itself -- do it here so this
            # section isn't silently empty.
            cdf["Market"] = "IN"
            cdf = liq.annotate(cdf)
        cdf = cdf[cdf["Liquidity"].isin(["High", "Medium"])].sort_values("Cash_Cycle").head(10)
        for _, r in cdf.iterrows():
            ccc_rows.append(
                f"<tr><td style='padding:4px 8px'><b>{r.Symbol}</b></td>"
                f"<td>{r.get('Name','')}</td><td>{r.Cash_Cycle:.1f}</td>"
                f"<td>{r.get('ROCE','')}</td>"
                f"<td style='color:{_COL.get(r.Liquidity,'#777')};font-weight:600'>{r.Liquidity}</td></tr>"
            )
    except Exception:
        pass

    # 3. Global momentum (top 15 overall)
    allh = rga.load_highlights()
    g = allh.head(15)
    g_rows = [
        f"<tr><td style='padding:4px 8px'>{r.Market}</td><td>{r.Symbol}</td>"
        f"<td>{r.ret_126:+.0f}</td><td>{r.rsi14}</td></tr>"
        for _, r in g.iterrows()
    ]

    # 3b. Other markets — top tradable mover per market (world tour, ex-IN/US)
    other_rows = []
    try:
        h = liq.annotate(allh.copy())
        names = {
            "TW": "Taiwan",
            "KR": "Korea",
            "JP": "Japan",
            "CN": "China",
            "HK": "Hong Kong",
            "CA": "Canada",
            "AU": "Australia",
            "UK": "UK",
            "DE": "Germany",
            "CH": "Switzerland",
            "SE": "Sweden",
            "FI": "Finland",
            "DK": "Denmark",
            "SG": "Singapore",
            "EU": "Euronext",
            "BR": "Brazil",
            "SA": "Saudi",
            "ZA": "S.Africa",
        }
        for m in [x for x in names if x in set(h.Market)]:
            sub = h[h.Market == m].sort_values("ret_126", ascending=False).head(1)
            if sub.empty:
                continue
            r = sub.iloc[0]
            other_rows.append(
                f"<tr><td style='padding:4px 8px'>{names[m]}</td>"
                f"<td><b>{r.Symbol}</b></td><td>{r.ret_126:+.0f}%</td>"
                f"<td style='color:{_COL.get(r.Liquidity,'#777')};font-weight:600'>{r.Liquidity}</td></tr>"
            )
    except Exception:
        pass

    # 4. 5y scoreboard
    p5 = mp.load()
    p_rows = [
        f"<tr><td style='padding:4px 8px'>{r.Market}</td><td>{r.Index}</td>"
        f"<td>{r['CAGR%']}</td><td>{r['Return_1y%']}</td><td>{r.Sharpe}</td></tr>"
        for _, r in p5.iterrows()
    ]

    html = f"""<div style="font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:740px;color:#1a1a1a">
<h1 style="font-size:21px;margin:0 0 2px">📈 Daily Market Brief — {today}</h1>
<p style="color:#666;font-size:13px;margin:0 0 14px">India + US screeners · fundamentals/news convergence · AI-discovered screens · cash-conversion-cycle · global momentum + 5-year scoreboard · Databento/Bhavcopy-verified · data as of last close</p>
{regime_section}
{verification_section}
<h2 style="font-size:16px;border-bottom:2px solid #1a73e8;padding-bottom:3px">🇮🇳 India — Daily Screener (most tradable)</h2>
{_table(["Symbol","Tier","Screens","Turnover","Liquidity"], picks_rows) if picks_rows else "<p>no picks</p>"}
<p style="font-size:11px;color:#666;margin:3px 0">Mood: <b style="color:#2e7d32">{mood['mood']} ({mood['score']:+.2f})</b> from {mood.get('n_articles',0)} articles. Liquidity (India): High ≥$5M/day · Medium $0.5–5M · Low &lt;$0.5M.</p>
{_table(["Symbol","Convergence","Fundamental","News","Headline"], conv_in_rows) if conv_in_rows else ""}
<h2 style="font-size:16px;border-bottom:2px solid #1a73e8;padding-bottom:3px;margin-top:22px">🇺🇸 US — Daily Screener (most tradable)</h2>
{_table(["Symbol","Tier","Screens","Turnover","Liquidity"], us_picks_rows) if us_picks_rows else "<p>no picks</p>"}
<p style="font-size:11px;color:#666;margin:3px 0">Mood: <b style="color:#2e7d32">{us_mood['mood']} ({us_mood['score']:+.2f})</b> from {us_mood.get('n_articles',0)} articles.</p>
<h3 style="font-size:13px;margin:10px 0 4px">⭐ Fundamentals + News Convergence</h3>
{_table(["Symbol","Convergence","Fundamental","News","Headline"], conv_us_rows) if conv_us_rows else "<p style='font-size:12px'>no convergence today</p>"}
<h2 style="font-size:16px;border-bottom:2px solid #1a73e8;padding-bottom:3px;margin-top:22px">🤖 AI-Discovered Screens (auto_screener.py)</h2>
<p style="font-size:11px;color:#555;margin:2px 0">Supervised anchor + unsupervised discovery + RL correction over the serving view. Research/education only — not a recommendation.</p>
{_ai_html("India", ai_in)}
{_ai_html("US", ai_us)}
<h2 style="font-size:16px;border-bottom:2px solid #1a73e8;padding-bottom:3px;margin-top:22px">🔬 Global-Market-Scanners — Full-Universe Scan</h2>
<p style="font-size:11px;color:#555;margin:2px 0">Darvas+Piotroski+Coffee-Can Triple Hits, Multi-Screen Hits, and (US) a Ridge-regression ML signal, from a standalone full-universe scanner (github.com/herrrickshaw/global-market-scanners).</p>
{_gms_html("India", gms_in)}
{_gms_html("US", gms_us)}
<h2 style="font-size:16px;border-bottom:2px solid #1a73e8;padding-bottom:3px;margin-top:22px">🇪🇺 Europe — Full-Universe Scan (22 exchanges)</h2>
<p style="font-size:11px;color:#555;margin:2px 0">Darvas + Piotroski + Coffee-Can across London, Frankfurt, Paris, Milan, Madrid, Amsterdam, Athens, Warsaw, Istanbul, Nordics, and more (full_european_market_scan.py).</p>
{_europe_html(europe)}
<h2 style="font-size:16px;border-bottom:2px solid #1a73e8;padding-bottom:3px;margin-top:22px">💵 India — Cash Conversion Cycle (screener.in 228040)</h2>
<p style="font-size:12px;color:#555">Lowest/negative CCC = collects from customers before paying suppliers. Tradable (High/Medium) only.</p>
{_table(["Symbol","Name","CCC days","ROCE","Liquidity"], ccc_rows) if ccc_rows else "<p>n/a</p>"}
<h2 style="font-size:16px;border-bottom:2px solid #1a73e8;padding-bottom:3px;margin-top:22px">🌍 Global Momentum — Top 15 (20 markets)</h2>
{_table(["Mkt","Symbol","6mo %","RSI"], g_rows)}
<h2 style="font-size:16px;border-bottom:2px solid #1a73e8;padding-bottom:3px;margin-top:22px">🗺️ Other Markets — top tradable mover each</h2>
<p style="font-size:12px;color:#555">Best liquid (≥$1M/day) 6-month performer per market.</p>
{_table(["Market","Symbol","6mo %","Liquidity"], other_rows) if other_rows else "<p>n/a</p>"}
<h2 style="font-size:16px;border-bottom:2px solid #1a73e8;padding-bottom:3px;margin-top:22px">🗓️ 20-Market 5-Year Scoreboard</h2>
{_table(["Mkt","Index","5y CAGR%","1y %","Sharpe"], p_rows)}
<p style="font-size:11px;color:#bf360c;border-top:1px solid #eee;padding-top:10px;margin-top:18px">⚠️ Educational/research only. NOT investment advice. Screener results are mechanical filters, not buy/sell signals. AI-discovered screens are historical associations from an experimental pipeline, not predictions. Liquidity/CCC are estimates; index figures price-only, local currency. Past performance does not guarantee future returns. Consult a SEBI-registered investment advisor.</p>
</div>"""

    text = (
        f"Daily Market Brief — {today}\n"
        f"India mood: {mood['mood']} ({mood['score']:+.2f}). US mood: {us_mood['mood']} ({us_mood['score']:+.2f}).\n"
        f"India + US screeners, fundamentals/news convergence, AI-discovered screens (auto_screener.py),\n"
        f"global-market-scanners full-universe scan, Europe 22-exchange scan,\n"
        f"India CCC screen (screener.in 228040), global momentum + 20-market 5y scoreboard.\n"
        f"{verification_text_block}\n"
        f"Educational/research only. NOT investment advice."
    )
    subject = f"📈 Daily Market Brief — {today}"
    return subject, text, html


if __name__ == "__main__":
    s, t, h = build()
    open("brief_today.html", "w").write(h)
    print(s, "\n", t, "\nhtml bytes:", len(h))
