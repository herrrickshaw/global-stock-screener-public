#!/usr/bin/env python3
"""
sec_history_collector.py — collect POINT-IN-TIME historical annual financials for
US tickers from SEC EDGAR companyfacts, so the fundamental screeners (Piotroski,
Coffee Can) can be BACKTESTED without look-ahead.

Unlike sec_fundamentals.py (which returns only the latest year), this extracts the
FULL annual (10-K/FY) time-series per concept, each row carrying the fiscal
period-end AND the SEC filing date (`filed`) — the filing date is when the data
became public, i.e. the correct as-of point for a point-in-time backtest.

Output: cache_seed/fundamentals_history/US.parquet — long format, one row per
(ticker, fy_end) with the Piotroski/Coffee-Can inputs + `filed`. Resumable: reuses
already-collected tickers on re-run.

  python3 sec_history_collector.py                 # US universe from data/global_universe/US.csv
  python3 sec_history_collector.py --limit 200     # test on the first 200
  python3 sec_history_collector.py --tickers AAPL MSFT
"""
import argparse
import os
import time

import pandas as pd
import requests

# BASE hardcoded ~/Downloads/code/python_files — migrated away from AND periodically
# mass-wiped, so the universe CSV, the LTM parquet and OUT_DIR all resolved to nothing.
# Same bug as screener_history_collector.py. Resolve against this file's own repo.
_HERE = os.path.dirname(os.path.abspath(__file__))
_LEGACY = os.path.expanduser("~/Downloads/code/python_files")
BASE = _HERE if os.path.exists(os.path.join(_HERE, "cache_seed")) else _LEGACY
OUT_DIR = os.environ.get("FUND_HIST_DIR") or os.path.join(
    BASE, "cache_seed", "fundamentals_history"
)
OUT = os.path.join(OUT_DIR, "US.parquet")
_UA = {"User-Agent": "market-research umashankartd1991@gmail.com"}  # SEC requires a contact
_RATE = 0.13  # ~8 req/s (SEC allows 10)

# us-gaap concepts → our field names (first present wins for alternates).
CONCEPTS = {
    "net_income": ["NetIncomeLoss"],
    "total_assets": ["Assets"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    # 🔴 ORDER MATTERS AND WAS WRONG. RevenueFromContractWithCustomerExcludingAssessedTax
    # was FIRST, but it is the ASC-606 CONTRACT-revenue subset, not total revenue. For a
    # multi-segment filer it omits derivative/trading revenue entirely:
    #     ADM FY2023 stored $25.69B against a 10-K total of ~$93.9B — wrong by 3.7x.
    # `cost_of_revenue` matched the filing exactly; only revenue didn't, which is why
    # `cost_of_revenue > revenue` fires on 262/5,418 rows (4.8%) — arithmetically
    # impossible, e.g. CPT FY2023 revenue ~$0 against $550M of cost.
    # Revenue feeds P/S, every margin and every growth rate, so a 3.7x error propagates
    # into everything downstream.
    # `Revenues` is the US-GAAP TOTAL and now goes first; the ASC-606 tag remains as a
    # fallback for filers who only report that one.
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ],
    # GrossProfit is tagged by only ~41% of filers — many report the COST line and
    # leave the subtraction to the reader, so asking for one tag threw away more than
    # half the universe. Both are collected; gross_profit is DERIVED from
    # revenue - cost_of_revenue when the direct tag is absent (see collect_ticker).
    # Same pattern `revenue` above already uses, and the same move that recovered EBIT
    # for India (pbt + interest) — build from reported components rather than hoping
    # for one convenience tag.
    "gross_profit": ["GrossProfit"],
    "cost_of_revenue": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
        "CostOfServices",
    ],
    "ebit": ["OperatingIncomeLoss"],
    "equity": ["StockholdersEquity"],
    "shares": ["CommonStockSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingBasic"],
}


def _ticker_cik(sess):
    r = sess.get("https://www.sec.gov/files/company_tickers.json", headers=_UA, timeout=30)
    r.raise_for_status()
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in r.json().values()}


def _series(facts, concepts):
    """Full annual (10-K/FY) series for the first present concept → {fy_end: (val, filed)}.
    Each period is re-reported as a comparative in later 10-Ks; keep the EARLIEST
    filing (when the number first became public) so the backtest is point-in-time."""
    for concept in concepts:
        node = facts.get("us-gaap", {}).get(concept)
        if not node:
            continue
        out = {}
        for unit_vals in node.get("units", {}).values():
            for r in unit_vals:
                if r.get("form") == "10-K" and r.get("fp") == "FY" and r.get("val") is not None:
                    end, filed = r["end"], (r.get("filed") or "9999-99-99")
                    if end not in out or filed < out[end][1]:  # earliest filing wins
                        out[end] = (r["val"], filed)
        if out:
            return out
    return {}


def collect_ticker(sess, ticker, cik):
    r = sess.get(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers=_UA, timeout=30
    )
    if r.status_code != 200:
        return []
    facts = r.json().get("facts", {})
    series = {field: _series(facts, cs) for field, cs in CONCEPTS.items()}
    fy_ends = sorted({e for s in series.values() for e in s})
    rows = []
    for fy in fy_ends:
        row = {"ticker": ticker, "fy_end": fy}
        filed = None
        for field, s in series.items():
            if fy in s:
                row[field] = s[fy][0]
                filed = filed or s[fy][1]
        row["filed"] = filed
        # Derive gross_profit when the filer never tagged GrossProfit. ~59% of filers
        # do not, but nearly all tag a cost line — so the field was absent by
        # COLLECTION METHOD, not by disclosure. This is subtraction of two reported
        # figures, NOT the ratio-inversion that previously produced absurd ROCE of
        # 95-137%. Where BOTH exist they must agree, which is checkable and is the
        # reason the direct tag always wins.
        # Sanity flag, not a silent repair: cost > revenue is impossible and means the
        # revenue tag captured a subset. Surfaced so a consumer can exclude the row
        # rather than quietly inheriting a 3.7x error.
        _r, _c = row.get("revenue"), row.get("cost_of_revenue")
        if _r is not None and _c is not None and _r > 0:
            row["revenue_suspect"] = bool(_c > _r)
        if row.get("gross_profit") is None:
            rev, cost = row.get("revenue"), row.get("cost_of_revenue")
            if rev is not None and cost is not None:
                row["gross_profit"] = rev - cost
                row["gross_profit_derived"] = True
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Collect US historical financials from SEC EDGAR")
    ap.add_argument("--tickers", nargs="*", help="explicit tickers (else the US universe)")
    ap.add_argument("--limit", type=int, default=0, help="cap number of tickers (testing)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    sess = requests.Session()
    cik = _ticker_cik(sess)

    if args.tickers:
        universe = [t.upper() for t in args.tickers]
    else:
        uni_csv = os.path.join(BASE, "data", "global_universe", "US.csv")
        if os.path.exists(uni_csv):
            universe = sorted(set(pd.read_csv(uni_csv)["yf_ticker"].astype(str).str.upper()))
        else:
            # The universe CSV is a build artifact and is usually absent. Two better
            # sources are already to hand, so crashing on it is pointless:
            #   * the LTM price panel — the TRADED universe, and analysis needs prices
            #     anyway, so a filer with no price data cannot be used;
            #   * SEC's own ticker->CIK map (fetched above) as the outer bound.
            ltm = os.path.join(BASE, "cache_seed", "ltm", "US.parquet")
            if os.path.exists(ltm):
                universe = sorted(set(pd.read_parquet(ltm, columns=["Symbol"])["Symbol"]
                                      .astype(str).str.upper()))
                print(f"  universe from LTM price panel: {len(universe):,} symbols")
            else:
                universe = sorted(cik.keys())
                print(f"  universe from SEC ticker->CIK map: {len(universe):,} filers")
    universe = [t for t in universe if t in cik]
    if args.limit:
        universe = universe[: args.limit]

    done = set()
    existing = []
    if os.path.exists(OUT):
        prev = pd.read_parquet(OUT)
        done = set(prev["ticker"].unique())
        existing = [prev]
    todo = [t for t in universe if t not in done]
    print(
        f"SEC history: {len(universe)} tickers in universe, {len(done)} already done, "
        f"{len(todo)} to fetch (~{len(todo)*_RATE/60:.0f} min)"
    )

    rows, n_ok, t0 = [], 0, time.time()
    for i, tk in enumerate(todo, 1):
        time.sleep(_RATE)
        try:
            r = collect_ticker(sess, tk, cik[tk])
            if r:
                rows.extend(r)
                n_ok += 1
        except Exception:
            pass
        if i % 250 == 0 or i == len(todo):
            # checkpoint to disk so the run is resumable / crash-safe
            df = (
                pd.concat(existing + [pd.DataFrame(rows)], ignore_index=True)
                if (existing or rows)
                else pd.DataFrame()
            )
            if not df.empty:
                os.makedirs(OUT_DIR, exist_ok=True)  # survive a mid-run dir wipe
                df.to_parquet(OUT, index=False)
            print(f"  {i}/{len(todo)} | {n_ok} with data | {len(df)} rows | {time.time()-t0:.0f}s")
    print(f"done → {os.path.relpath(OUT, BASE)}")


if __name__ == "__main__":
    main()
