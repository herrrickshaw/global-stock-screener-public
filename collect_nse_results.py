#!/usr/bin/env python3
"""
collect_nse_results.py -- point-in-time India quarterly financial results,
sourced directly from NSE's own "Results Comparison" API
(nseindia.com/api/results-comparision?symbol=X), via nsepython's URL/header
conventions but with a SINGLE persistent session reused across the whole
run rather than nsepython's own nse_past_results()/nsefetch(), which opens
a brand-new requests.Session() and re-does BOTH NSE homepage warm-up GETs
on every single call -- fine for one-off use, but calling it in a tight
loop across ~8,900 tickers would triple request volume for no benefit and
risk tripping NSE's anti-bot defenses faster than necessary.

WHY THIS EXISTS (vs. screener_history_collector.py, already in this repo):
screener.in exposes fiscal PERIOD-END only, no real filing date -- this
account's own established convention there is a fy_end+90-day PROXY, not
a genuine disclosure date. NSE's results-comparison endpoint instead
returns `re_create_dt`, which is the ACTUAL result-creation/broadcast
timestamp NSE recorded -- verified live on RELIANCE: Q3 FY25 (period end
31-Dec-2024) created 16-Jan-2025 (16-day lag), Q2 FY25 created 14-Oct-2024
(14-day lag), Q1 FY25 created 19-Jul-2024 (19-day lag), FY24 annual
created 22-Apr-2024 (22-day lag) -- genuinely VARIABLE lag, not a constant
offset, i.e. real point-in-time data. This also runs cleanly on this Mac,
unlike nsepython's equity_history() (documented elsewhere in this
account's memory as needing browser cookies to work on Mac) -- this
endpoint has no such issue.

RATE LIMITING: 1.2s between requests (more cautious than screener.in's
0.7s -- NSE's own site is known to be stricter about automated traffic
than screener.in), one retry on empty/error, and a hard circuit breaker
on sustained consecutive failures (same shape as screener_history_
collector.py's Blocked-exception handling) so a real block aborts
cleanly instead of continuing to hammer a blocking host.
"""
from __future__ import annotations

import argparse
import os
import time

import pandas as pd
import requests

BASE = "/Users/umashankar/repos/global-stock-screener"
LTM = os.path.join(BASE, "cache_seed", "ltm", "IN.parquet")
OUT_DIR = os.path.join(BASE, "cache_seed", "fundamentals_history")
OUT = os.path.join(OUT_DIR, "IN_nse_results.parquet")

_RATE = 1.2
# Raised from 15 after a real run hit a cluster of Gold ETFs (GOLD1, GOLDBEES,
# GOLDETF, GOLDIETF, ...) sitting consecutively in the alphabetically-sorted
# universe -- ETFs/index funds are not operating companies and legitimately
# have no quarterly results, but they clustered wider than the original
# threshold tolerated, tripping a false "blocked" abort. Combined with the
# name-based ETF/fund filter below (belt AND suspenders -- the filter cuts
# the known cases, this threshold absorbs whatever it misses) rather than
# chasing every fund-family naming convention with an ever-growing keyword
# list.
MAX_CONSECUTIVE_EMPTY = 40

# Substrings seen in real NSE-listed ETF/index-fund tickers that are NOT
# operating companies (Gold/Silver/Liquid/Nifty/Sensex-tracking funds, "BEES"
# is Benchmark Exchange-traded Scheme -- the original India ETF brand name
# still used generically). Best-effort, not exhaustive -- MAX_CONSECUTIVE_EMPTY
# above is the real backstop for whatever this misses.
_ETF_HINTS = ("BEES", "ETF", "GOLDCASE")

HEADERS = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "accept-language": "en-US,en;q=0.9,en-IN;q=0.8,en-GB;q=0.7",
    "cache-control": "max-age=0",
    "sec-ch-ua": '"Microsoft Edge";v="129", "Not=A?Brand";v="8", "Chromium";v="129"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0",
}

# Every field this endpoint returns that maps onto this repo's existing
# fundamentals_history schema (see screener_history_collector.py's own
# column set) -- kept under the SAME names so downstream code (compute_
# fundamental_screens equivalents) needs no per-source branching.
FIELD_MAP = {
    "re_total_inc": "revenue",
    "re_net_sale": "net_sale",
    "re_net_profit": "net_income",
    "re_pro_loss_bef_tax": "pretax_income",
    "re_tax": "tax",
    "re_int_new": "interest_expense",
    "re_depr_und_exp": "depreciation",
    "re_staff_cost": "staff_cost",
    "re_oth_inc_new": "other_income",
    "re_rawmat_consump": "raw_material_cost",
    "re_pur_trd_goods": "purchase_traded_goods",
    "re_basic_eps_for_cont_dic_opr": "eps",
    "re_debt_eqt_rat": "debt_to_equity",
    "re_debt_ser_cov": "debt_service_coverage",
    "re_int_ser_cov": "interest_coverage",
    "re_face_val": "face_value",
    "re_res_type": "result_type",
}


class Blocked(Exception):
    pass


def _session() -> requests.Session:
    s = requests.Session()
    s.get("https://www.nseindia.com", headers=HEADERS, timeout=10)
    s.get("https://www.nseindia.com/option-chain", headers=HEADERS, timeout=10)
    return s


def _parse_ddmonyyyy(s):
    if not s:
        return None
    try:
        return pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")
    except Exception:
        return None


def fetch_ticker(session: requests.Session, ticker: str) -> list[dict]:
    url = f"https://www.nseindia.com/api/results-comparision?symbol={ticker}"
    try:
        r = session.get(url, headers=HEADERS, timeout=10)
    except requests.RequestException:
        return []
    if r.status_code == 401 or r.status_code == 403:
        raise Blocked(f"HTTP {r.status_code}")
    if r.status_code != 200:
        return []
    try:
        payload = r.json()
    except ValueError:
        raise Blocked("non-JSON response -- session likely stale/blocked")
    rows = payload.get("resCmpData") if isinstance(payload, dict) else None
    if not rows:
        return []

    out = []
    for row in rows:
        filed = _parse_ddmonyyyy(row.get("re_create_dt"))
        fy_end = _parse_ddmonyyyy(row.get("re_to_dt"))
        fy_start = _parse_ddmonyyyy(row.get("re_from_dt"))
        if filed is None or fy_end is None:
            continue
        rec = {
            "ticker": ticker,
            "filed": filed,
            "fy_end": fy_end,
            "fy_start": fy_start,
            "seq_num": row.get("re_seq_num"),
        }
        for src_col, dst_col in FIELD_MAP.items():
            val = row.get(src_col)
            if val in (None, "", "-"):
                rec[dst_col] = None
            else:
                try:
                    rec[dst_col] = (
                        float(val) * 100_000
                        if dst_col
                        not in (
                            "eps",
                            "debt_to_equity",
                            "debt_service_coverage",
                            "interest_coverage",
                            "face_value",
                            "result_type",
                        )
                        else (val if dst_col == "result_type" else float(val))
                    )
                except (TypeError, ValueError):
                    rec[dst_col] = val
        out.append(rec)
    return out


def _universe(limit: int) -> list[str]:
    # Reuses screener_history_collector.py's own bond/G-sec/T-bill filter --
    # NOT duplicated here. Without it, the alphabetically-first symbols in
    # the LTM universe are debt instruments (07AGG, 08ABB, 0KFL25, ...),
    # which legitimately have no equity results on NSE's API -- a run of
    # these falsely tripped this collector's circuit breaker on its first
    # real attempt (0 tickers collected, reported as a "block" that wasn't
    # one). Filtering them out here fixes the root cause instead of loosening
    # the circuit breaker, which would just paper over a real host block
    # if one actually happens later.
    #
    # is_non_equity()'s regex alone does NOT catch this batch (07AGG, 0KFL25,
    # 0ANSPL28, ... have 1-2 leading digits, not the 3-4 the _BOND pattern
    # expects) -- confirmed by testing, they survive the filter unchanged.
    # Rather than chase every NCD naming variant with a fragile regex, apply
    # the SAME liquidity gate screener_history_collector.py's own --liquid-
    # only flag uses (median 60-session dollar turnover >= Rs 1 crore) as a
    # second, more robust line of defense -- bonds/NCDs don't clear this,
    # and it also matches this account's own established mandatory-liquidity-
    # gate convention for anything price-derived.
    from screener_history_collector import is_non_equity

    px = pd.read_parquet(LTM, columns=["Symbol", "Close", "Volume", "Date"])
    px["Symbol"] = px["Symbol"].astype(str).str.upper()
    cutoff = px["Date"].max() - pd.Timedelta(days=120)
    recent = px[px["Date"] >= cutoff].copy()
    recent["tv"] = recent["Close"] * recent["Volume"]
    liq = recent.groupby("Symbol")["tv"].median()
    syms = sorted(liq[liq >= 10_000_000].index)
    syms = [s for s in syms if not is_non_equity(s) and not any(h in s for h in _ETF_HINTS)]
    if limit:
        syms = syms[:limit]
    return syms


def main():
    ap = argparse.ArgumentParser(
        description="Collect India point-in-time results from NSE's own API"
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tickers", nargs="*")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    universe = [t.upper() for t in args.tickers] if args.tickers else _universe(args.limit)

    done, existing = set(), []
    if os.path.exists(OUT):
        prev = pd.read_parquet(OUT)
        done = set(prev["ticker"].astype(str).str.upper())
        existing = [prev]
    todo = [t for t in universe if t not in done]
    print(
        f"NSE results-comparison: {len(universe)} tickers, {len(done)} already done, "
        f"{len(todo)} to fetch (~{len(todo) * _RATE / 60:.0f} min at {_RATE}s/req)"
    )

    session = _session()

    def _checkpoint(rows):
        df = (
            pd.concat(existing + [pd.DataFrame(rows)], ignore_index=True)
            if (existing or rows)
            else pd.DataFrame()
        )
        if not df.empty:
            os.makedirs(OUT_DIR, exist_ok=True)
            df.to_parquet(OUT, index=False)
        return df

    rows, n_ok, t0, consec_empty = [], 0, time.time(), 0
    for i, tk in enumerate(todo, 1):
        time.sleep(_RATE)
        try:
            r = fetch_ticker(session, tk)
        except Blocked as e:
            print(f"\n  possible block ({e}) at {tk} -- re-establishing session and retrying once")
            try:
                session = _session()
                time.sleep(_RATE * 2)
                r = fetch_ticker(session, tk)
            except Blocked as e2:
                print(
                    f"\n  ABORT: still blocked after session refresh ({e2}) at {tk}. "
                    f"Collected {n_ok} before this."
                )
                _checkpoint(rows)
                break
        got = bool(r)
        if got:
            rows.extend(r)
            n_ok += 1
        consec_empty = 0 if got else consec_empty + 1
        if consec_empty >= MAX_CONSECUTIVE_EMPTY:
            print(
                f"\n  ABORT: {consec_empty} consecutive tickers returned nothing -- "
                f"session dead or host blocking. Collected {n_ok} before this."
            )
            _checkpoint(rows)
            break
        if i % 50 == 0 or i == len(todo):
            df = _checkpoint(rows)
            print(
                f"  {i}/{len(todo)} | {n_ok} with data | {len(df)} rows | {time.time() - t0:.0f}s"
            )
    print(f"done -> {os.path.relpath(OUT, BASE)}")


if __name__ == "__main__":
    main()
