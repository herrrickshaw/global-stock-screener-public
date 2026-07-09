#!/usr/bin/env python3
"""
Darvas breakouts section for the daily market report (India + US).

Reads the most recent {market}_full_scan xlsx (Darvas_Signals sheet) and extracts
*genuine* Darvas breakouts, not the raw "BREAKOUT_BUY" flags (which just mean
"trading above the box"). A real Darvas breakout = price has just cleared the box
top, in an uptrend, on positive momentum.

Tiers:
  T1  Golden Cross + Darvas breakout that is FRESH (box position <= 130%)
  T2  Fresh healthy breakouts: box position 100-120%, above 200-DMA, up on day,
      >=250 daily bars (a real box, not a recent IPO)

Outputs:
  - HTML fragment to stdout (embed directly in the email), and
  - a ranked CSV to --csv (default ~/Downloads/{market}_darvas_breakouts_<date>.csv)

Usage:
  python3 darvas_breakouts.py --market US   [--top 15]
  python3 darvas_breakouts.py --market IN   [--top 15]
  python3 darvas_breakouts.py --market IN --file /path/to/scan.xlsx
"""
import argparse
import datetime
import glob
import html
import os
import sys

import pandas as pd

MARKETS = {
    "US": {
        "label": "US",
        "cur": "$",
        "flag": "🇺🇸",
        "scan_dirs": ["~/Downloads/code/python_files/us_full_scan"],
        "glob": "us_full_scan_*.xlsx",
    },
    "IN": {
        "label": "India",
        "cur": "₹",
        "flag": "🇮🇳",
        "scan_dirs": [
            "~/Downloads/code/python_files/indian_full_scan",
            "~/Downloads/data/indian_full_scan",
        ],
        "glob": "indian_full_scan_*.xlsx",
    },
}

# Confident ticker -> registered name map (extend as needed). Unknown -> ticker.
NAMES = {
    # US
    "GD": "GENERAL DYNAMICS CORP",
    "ABVX": "ABIVAX SA",
    "RBRK": "RUBRIK INC",
    "NEO": "NEOGENOMICS INC",
    "RCEL": "AVITA MEDICAL INC",
    "BOC": "BOSTON OMAHA CORP",
    "BV": "BRIGHTVIEW HOLDINGS INC",
    "CANE": "TEUCRIUM SUGAR FUND (ETF)",
    "PRVA": "PRIVIA HEALTH GROUP INC",
    "BSET": "BASSETT FURNITURE INDUSTRIES INC",
    "GIBO": "GIBO HOLDINGS LTD",
    "RKLB": "ROCKET LAB CORP",
    "ALAB": "ASTERA LABS INC",
    "INSW": "INTERNATIONAL SEAWAYS INC",
    "ASX": "ASE TECHNOLOGY HOLDING",
    "TRMD": "TORM PLC",
    "KOD": "KODIAK SCIENCES INC",
    "AAMI": "ACADIAN ASSET MANAGEMENT",
    "RDW": "REDWIRE CORP",
    "SKLZ": "SKILLZ INC",
    "WATT": "ENERGOUS CORP",
    "IAG": "IAMGOLD CORP",
    "CVGI": "COMMERCIAL VEHICLE GROUP",
    "LCUT": "LIFETIME BRANDS INC",
    "SB": "SAFE BULKERS INC",
    "EQ": "EQUILLIUM INC",
    # India (confident only)
    "ASIANPAINT": "ASIAN PAINTS LTD",
    "APOLLOHOSP": "APOLLO HOSPITALS ENTERPRISE LTD",
    "ADANIPORTS": "ADANI PORTS & SEZ LTD",
    "ADANIENSOL": "ADANI ENERGY SOLUTIONS LTD",
    "ATUL": "ATUL LTD",
    "ATULAUTO": "ATUL AUTO LTD",
    "AARTIIND": "AARTI INDUSTRIES LTD",
    "APARINDS": "APAR INDUSTRIES LTD",
    "ANGELONE": "ANGEL ONE LTD",
    "ANANDRATHI": "ANAND RATHI WEALTH LTD",
    "AEGISLOG": "AEGIS LOGISTICS LTD",
    "ADOR": "ADOR WELDING LTD",
    "ARVIND": "ARVIND LTD",
    "ANTHEM": "ANTHEM BIOSCIENCES LTD",
    "ASHAPURMIN": "ASHAPURA MINECHEM LTD",
    "APCOTEXIND": "APCOTEX INDUSTRIES LTD",
    "BAJAJCON": "BAJAJ CONSUMER CARE LTD",
    "ASTRAMICRO": "ASTRA MICROWAVE PRODUCTS LTD",
    "ACMESOLAR": "ACME SOLAR HOLDINGS LTD",
    "J&KBANK": "JAMMU & KASHMIR BANK LTD",
    "DATAPATTNS": "DATA PATTERNS (INDIA) LTD",
    "ADFFOODS": "ADF FOODS LTD",
    "ANDHRSUGAR": "THE ANDHRA SUGARS LTD",
}


def cname(sym):
    return NAMES.get(sym, sym)


def esc(s):
    return html.escape(str(s))


# ─────────────────────────── real-time refresh ───────────────────────────
# The batch scan xlsx is our "legacy" layer: it carries the expensive, history-
# dependent insights — Box_Top / Box_Bottom (needs the full box history), the
# 50/200-DMA gap, Data_Points (a real box vs. a fresh IPO), and the GC flag.
# Live mode keeps all of that but overlays a fresh quote so the *breakout
# freshness* (LTP, Day%, position inside the legacy box) reflects right now,
# not whenever the batch ran. A name that was 105% of its box at batch time may
# now be 125% (extended) or back under 100% (failed) — the refresh catches that.


def _yf_symbol(sym, market, suffix=None):
    """Map a scan symbol to its yfinance ticker."""
    sym = str(sym).strip()
    if market == "IN":
        suf = str(suffix or "").strip()
        if suf.upper() not in (".NS", ".BO"):
            suf = ".NS"  # NSE is the default listing for Indian names
        return f"{sym}{suf}"
    return sym  # US tickers are used bare


def fetch_live_quotes(pairs):
    """
    pairs: list of (orig_symbol, yf_symbol).
    Returns {orig_symbol: (ltp, prev_close)} using one batched yfinance call.
    Missing/failed tickers are simply omitted (caller keeps the legacy value).
    """
    try:
        import yfinance as yf
    except ImportError:
        sys.stderr.write(
            "[darvas] yfinance not installed — cannot run --live; "
            "falling back to legacy batch prices.\n"
        )
        return {}
    out, yf_syms = {}, [y for _, y in pairs]
    if not yf_syms:
        return out
    try:
        data = yf.download(
            yf_syms,
            period="5d",
            interval="1d",
            group_by="ticker",
            threads=True,
            progress=False,
            auto_adjust=False,
        )
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[darvas] live download failed ({e}); using legacy prices.\n")
        return out
    single = len(yf_syms) == 1
    lvl0 = set() if single else set(data.columns.get_level_values(0))
    for orig, y in pairs:
        try:
            sub = data if single else (data[y] if y in lvl0 else None)
            if sub is None:
                continue
            closes = sub["Close"].dropna()
            if closes.empty:
                continue
            ltp = float(closes.iloc[-1])
            prev = float(closes.iloc[-2]) if len(closes) >= 2 else ltp
            if ltp > 0:
                out[orig] = (ltp, prev)
        except Exception:  # noqa: BLE001
            continue
    return out


def apply_live(df, market, live_max):
    """
    Overlay live quotes onto the legacy scan for the plausible-breakout band,
    recomputing LTP, Day%, and Position_in_Box% against the legacy box levels.
    DMA_Gap% is left from the batch — 50/200-DMAs barely move intraday.
    Returns (df, n_candidates, n_refreshed).
    """
    healthy = df["GC_Signal"].isin(["ABOVE_200DMA", "GOLDEN_CROSS"])
    cand = df[
        (df["Darvas_Signal"] == "BREAKOUT_BUY")
        & healthy
        & (df["Data_Points"] >= 250)
        & (df["Position_in_Box%"].between(85, 150))
    ].copy()
    # Prefer names sitting nearest the fresh-breakout zone (~110% of the box).
    cand["_prox"] = (cand["Position_in_Box%"] - 110).abs()
    cand = cand.sort_values("_prox").head(max(0, live_max))
    if cand.empty:
        return df, 0, 0

    suf_col = "Suffix" if "Suffix" in df.columns else None
    pairs = [
        (r["Symbol"], _yf_symbol(r["Symbol"], market, r.get(suf_col) if suf_col else None))
        for _, r in cand.iterrows()
    ]
    quotes = fetch_live_quotes(pairs)

    refreshed = 0
    for idx, r in cand.iterrows():
        q = quotes.get(r["Symbol"])
        if not q:
            continue
        ltp, prev = q
        df.at[idx, "LTP"] = ltp
        if prev and prev > 0:
            df.at[idx, "Change%"] = (ltp - prev) / prev * 100.0
        bt, bb = r.get("Box_Top"), r.get("Box_Bottom")
        if pd.notna(bt) and pd.notna(bb) and (bt - bb) != 0:
            df.at[idx, "Position_in_Box%"] = (ltp - bb) / (bt - bb) * 100.0
        refreshed += 1
    return df, len(cand), refreshed


def latest_scan(cfg):
    files = []
    for d in cfg["scan_dirs"]:
        files += glob.glob(os.path.join(os.path.expanduser(d), cfg["glob"]))
    files = sorted(set(files), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def exchange_for(row, market):
    """India rows carry a Suffix column (.BO = BSE, else NSE). US is just 'US'."""
    if market == "IN":
        suf = str(row.get("Suffix", "") or "").upper()
        return "BSE" if suf == ".BO" else "NSE"
    return "US"


def build(df):
    b = df[df["Darvas_Signal"] == "BREAKOUT_BUY"].copy()
    # US scans mark the uptrend via GC_Signal == ABOVE_200DMA; the India scan
    # instead carries a DMA50_above_200 boolean and only populates GC_Signal for
    # GOLDEN_CROSS. Accept either encoding so both markets produce breakouts.
    above_dma = (b["DMA50_above_200"]) if "DMA50_above_200" in b.columns else False
    b["healthy"] = b["GC_Signal"].isin(["ABOVE_200DMA", "GOLDEN_CROSS"]) | above_dma
    t1 = b[(b["GC_Signal"] == "GOLDEN_CROSS") & (b["Position_in_Box%"] <= 130)].copy()
    t1 = t1.sort_values("Change%", ascending=False)
    t2 = b[
        (b["Position_in_Box%"].between(100, 120))
        & b["healthy"]
        & (b["Change%"] > 0)
        & (b["Data_Points"] >= 250)
        & (b["GC_Signal"] != "GOLDEN_CROSS")
    ].copy()
    # DMA_Gap% is a US-only column; drop that term from the score when absent.
    gap = t2["DMA_Gap%"].clip(-50, 100) if "DMA_Gap%" in t2.columns else 0
    t2["score"] = t2["Change%"] + (120 - t2["Position_in_Box%"]) * 0.15 + gap * 0.05
    t2 = t2.sort_values("score", ascending=False)
    bd = df[df["Darvas_Signal"] == "BREAKDOWN_SELL"].sort_values("Change%")
    return t1, t2, bd


def rows_html(rows, market, cur):
    out = []
    has_gap = "DMA_Gap%" in rows.columns
    for _, r in rows.iterrows():
        exch = exchange_for(r, market)
        gap_cell = f'{r["DMA_Gap%"]:+.0f}%' if has_gap and pd.notna(r["DMA_Gap%"]) else "n/a"
        out.append(
            f'<tr><td style="padding:6px 8px;border-bottom:1px solid #eee">'
            f'<b>{esc(cname(r["Symbol"]))}</b> <span style="color:#666;font-size:12px">[{esc(r["Symbol"])} · {exch}]</span></td>'
            f'<td style="padding:6px 8px;border-bottom:1px solid #eee;text-align:right">{cur}{r["LTP"]:,.2f}</td>'
            f'<td style="padding:6px 8px;border-bottom:1px solid #eee;text-align:right;color:{"#1a7f37" if r["Change%"]>=0 else "#c0392b"}">{r["Change%"]:+.1f}%</td>'
            f'<td style="padding:6px 8px;border-bottom:1px solid #eee;text-align:right">{cur}{r["Box_Top"]:,.2f}</td>'
            f'<td style="padding:6px 8px;border-bottom:1px solid #eee;text-align:center">{r["Position_in_Box%"]:.0f}%</td>'
            f'<td style="padding:6px 8px;border-bottom:1px solid #eee;text-align:center">{gap_cell}</td></tr>'
        )
    return "".join(out)


def to_html(t1, t2, bd, scan_name, cfg, market, top=15, live_note=""):
    cur = cfg["cur"]
    hdr = (
        '<tr style="background:#1f2937;color:#fff">'
        '<th style="padding:8px;text-align:left">Company [Ticker · Exch]</th>'
        '<th style="padding:8px;text-align:right">LTP</th>'
        '<th style="padding:8px;text-align:right">Day%</th>'
        '<th style="padding:8px;text-align:right">Box Top</th>'
        '<th style="padding:8px">Box Pos</th>'
        '<th style="padding:8px">50/200 Gap</th></tr>'
    )
    t1h = (
        rows_html(t1, market, cur)
        or '<tr><td colspan="6" style="padding:6px 8px;color:#666">None today.</td></tr>'
    )
    t2h = (
        rows_html(t2.head(top), market, cur)
        or '<tr><td colspan="6" style="padding:6px 8px;color:#666">None today.</td></tr>'
    )
    bd_names = ", ".join(
        f'{esc(r["Symbol"])} ({r["Change%"]:+.0f}%)' for _, r in bd.head(8).iterrows()
    )
    return f"""
<h2 style="font-size:17px;color:#111;border-bottom:2px solid #eee;padding-bottom:4px;margin:22px 0 8px">📈 {cfg["flag"]} {esc(cfg["label"].upper())} DARVAS BREAKOUTS</h2>{live_note}
<p style="font-size:12px;color:#666;margin:2px 0 8px">Fresh box breakouts from {esc(scan_name)}. The raw scan flags every stock above its box as "BREAKOUT_BUY"; this section keeps only <b>fresh</b> breakouts — price just clearing the box top (position 100–120%), in an uptrend, up on the day. Box Pos &gt;130% = already extended (excluded).</p>
<p style="font-size:13px;font-weight:700;margin:10px 0 4px">🥇 Golden Cross + Darvas breakout (highest-conviction confluence)</p>
<table style="width:100%;border-collapse:collapse;font-size:13px">{hdr}{t1h}</table>
<p style="font-size:13px;font-weight:700;margin:14px 0 4px">🥈 Fresh healthy breakouts (top {top})</p>
<table style="width:100%;border-collapse:collapse;font-size:13px">{hdr}{t2h}</table>
<p style="font-size:12px;color:#c0392b;margin:10px 0 2px"><b>🔻 Darvas breakdowns (SELL side, {len(bd)} total):</b> {bd_names}</p>
<p style="font-size:11px;color:#888;margin:6px 0">Darvas breakouts are momentum signals, NOT buy recommendations. False breakouts are common; confirm with volume and your own risk rules.</p>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["US", "IN"], default="US")
    ap.add_argument("--file", default=None, help="explicit xlsx path (overrides scan dirs)")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument(
        "--live",
        action="store_true",
        help="Overlay real-time quotes on the legacy box/DMA insights "
        "before selecting breakouts (needs yfinance).",
    )
    ap.add_argument(
        "--live-max",
        type=int,
        default=200,
        help="Max candidate tickers to refresh live (default 200).",
    )
    args = ap.parse_args()

    cfg = MARKETS[args.market]
    path = args.file or latest_scan(cfg)
    if not path or not os.path.exists(path):
        sys.stderr.write(f"[darvas] No {args.market} scan xlsx found in {cfg['scan_dirs']}\n")
        # Emit a graceful placeholder so the section can still be skipped cleanly.
        print(
            f'<p style="font-size:12px;color:#888">📈 {cfg["label"]} Darvas breakouts: no scan file available.</p>'
        )
        sys.exit(0)

    df = pd.read_excel(path, sheet_name="Darvas_Signals")

    live_note = ""
    if args.live:
        df, n_cand, n_ref = apply_live(df, args.market, args.live_max)
        now = datetime.datetime.now().strftime("%d %b %Y %H:%M")
        scan_dt = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%d %b %Y %H:%M")
        sys.stderr.write(
            f"[darvas] LIVE refresh: candidates={n_cand} refreshed={n_ref} "
            f"(legacy scan {scan_dt})\n"
        )
        live_note = (
            f'<p style="font-size:12px;color:#0b5;background:#e8faf0;border:1px solid #8fd9b0;'
            f'border-radius:5px;padding:6px 9px;margin:6px 0"><b>⏱ Live prices as of {now}</b> — '
            f"LTP, Day% and box position refreshed for {n_ref} of {n_cand} candidate names. "
            f"Box tops, 50/200-DMA gap and box validity are carried from the legacy batch scan "
            f"({scan_dt}); those need a full year of history and do not change intraday.</p>"
        )

    t1, t2, bd = build(df)

    date = datetime.date.today().isoformat()
    csv_path = args.csv or os.path.expanduser(
        f"~/Downloads/{args.market.lower()}_darvas_breakouts_{date}.csv"
    )
    keep = [
        "Symbol",
        "LTP",
        "Change%",
        "Box_Top",
        "Box_Bottom",
        "Position_in_Box%",
        "DMA_Gap%",
        "GC_Signal",
        "Data_Points",
    ]
    if "Suffix" in df.columns:
        keep.insert(1, "Suffix")
    cols = [c for c in keep if c in df.columns]
    out = pd.concat([t1.assign(Tier="T1_GoldenCross"), t2.assign(Tier="T2_Fresh")])
    out[cols + ["Tier"]].to_csv(csv_path, index=False)
    sys.stderr.write(
        f"[darvas] market={args.market} scan={os.path.basename(path)} "
        f"T1={len(t1)} T2={len(t2)} breakdowns={len(bd)} csv={csv_path}\n"
    )

    print(
        to_html(
            t1, t2, bd, os.path.basename(path), cfg, args.market, top=args.top, live_note=live_note
        )
    )


if __name__ == "__main__":
    main()
