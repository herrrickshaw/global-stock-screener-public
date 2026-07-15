#!/usr/bin/env python3
"""
screener_history_collector.py — collect POINT-IN-TIME historical ANNUAL financials
for INDIAN (NSE/BSE) tickers from screener.in's per-company Excel export, so the
fundamental screeners (Piotroski, Coffee Can) can be BACKTESTED on India without
look-ahead — the India analogue of sec_history_collector.py (which does US via
SEC EDGAR).

Each company's screener.in export carries a "Data Sheet" with ANNUAL line items
aligned to a fiscal "Report Date" axis (Mar-YYYY, ~10 years). We extract the
Piotroski/Coffee-Can inputs per fiscal year:
  net_income, revenue, cfo, borrowings, reserves, equity_capital, other_liab,
  inventory, receivables, cash, operating_profit, shares.

CAVEAT vs the US collector: screener.in exposes the fiscal PERIOD-END but NOT the
filing date. Indian issuers must file audited annual results within ~60 days of the
FY-end (SEBI LODR); the backtest therefore enters `fy_end + reporting-lag` days
later (default 90) as a conservative, no-look-ahead proxy — there is no earlier
as-of date available, so this is the honest best we can do.

Output: cache_seed/fundamentals_history/IN.parquet — long format, one row per
(ticker, fy_end). Resumable: reuses already-collected tickers on re-run.

  python3 screener_history_collector.py                 # NSE universe (data/global_universe/IN.csv)
  python3 screener_history_collector.py --limit 200      # test on the first 200
  python3 screener_history_collector.py --tickers RELIANCE TCS INFY
  python3 screener_history_collector.py --liquid-only     # only names present in cache_seed/ltm/IN.parquet
"""
import argparse
import io
import os
import time

import pandas as pd

# BASE used to hardcode ~/Downloads/code/python_files. That tree was migrated away
# from AND is periodically mass-wiped by a parallel session, so all three derived
# paths (UNIVERSE, LTM, OUT_DIR) resolved to nothing — and OUT_DIR pointing there
# meant a 30-minute collection would write straight into the wipe zone.
# Resolve against this file's own repo, which is where cache_seed/ltm/IN.parquet
# actually lives; fall back to the legacy path only if it still exists.
_HERE = os.path.dirname(os.path.abspath(__file__))
_LEGACY = os.path.expanduser("~/Downloads/code/python_files")
BASE = _HERE if os.path.exists(os.path.join(_HERE, "cache_seed")) else _LEGACY
OUT_DIR = os.environ.get("FUND_HIST_DIR") or os.path.join(
    BASE, "cache_seed", "fundamentals_history"
)
OUT = os.path.join(OUT_DIR, "IN.parquet")
UNIVERSE = os.path.join(BASE, "data", "global_universe", "IN.csv")
LTM = os.path.join(BASE, "cache_seed", "ltm", "IN.parquet")
_RATE = 0.7  # polite delay between screener.in fetches (ToS-friendly)
# Consecutive empty fetches before aborting. A real universe has gaps (BSE-only names
# 404 on the NSE-shaped URL), but ~30 in a row is a block, not a coincidence.
MAX_CONSECUTIVE_EMPTY = 30

# Data-Sheet row label → our field name (annual series only; first present wins).
FIELDS = {
    "net_income": ["Net profit", "Net Profit"],
    "revenue": ["Sales", "Sales+", "Revenue", "Total Revenue"],
    # "Operating Profit" is DELIBERATELY NOT mapped, despite existing in the export.
    # It appears ONLY inside the "Quarters" block (row ~49), whose Report-Date axis
    # carries Jun/Sep/Dec and is correctly rejected by _is_annual_axis. screener.in
    # does not publish an annual Operating Profit row at all: the "PROFIT & LOSS"
    # block runs Sales -> ... -> Interest -> Profit before tax -> Net profit with no
    # operating-profit line. Mapping the label would silently splice a QUARTERLY
    # figure into an annual series — a far worse bug than the missing column, because
    # it would look populated and be wrong by ~4x.
    #
    # EBIT is instead DERIVED from two directly-reported ANNUAL line items:
    #     EBIT = profit_before_tax + interest
    # which is the textbook definition ROCE takes as its numerator. This is addition
    # of reported figures, NOT the ratio-inversion trap (equity = net_income/roe)
    # that previously produced absurd ROCE of 95-137%.
    "profit_before_tax": ["Profit before tax", "Profit Before Tax", "PBT"],
    "interest": ["Interest", "Interest Expense", "Finance Costs"],
    # COGS components. screener.in has no "Gross Profit" or "COGS" row — it itemises
    # the cost lines — so gross profit is built from them (see collect_ticker).
    "raw_material_cost": ["Raw Material Cost"],
    "change_in_inventory": ["Change in Inventory"],
    "power_and_fuel": ["Power and Fuel"],
    "other_mfr_exp": ["Other Mfr. Exp", "Other Mfr Exp"],
    "cfo": ["Cash from Operating Activity", "Cash from Operating Activity+"],
    "borrowings": ["Borrowings", "Borrowings+", "Total Debt"],
    "reserves": ["Reserves"],
    "equity_capital": ["Equity Capital", "Equity Share Capital", "Share Capital"],
    "other_liab": ["Other Liabilities", "Other Liabilities+"],
    "inventory": ["Inventory", "Inventories"],
    "receivables": ["Trade receivables", "Debtors", "Receivables"],
    "cash": ["Cash & Bank", "Cash Equivalents"],
    "shares": ["No. of Equity Shares", "No of Equity Shares"],
}


def _num(x):
    try:
        return float(str(x).replace(",", "").replace("%", "").strip())
    except Exception:
        return None


def _is_annual_axis(vals):
    """A 'Report Date' row is the ANNUAL axis iff every date is a fiscal year-end
    (month == 3). Quarterly Report-Date rows carry Jun/Sep/Dec and are skipped so we
    never mistake a quarterly Sales/Net-profit row for the annual one."""
    dates = [v for v in vals if isinstance(v, pd.Timestamp) or hasattr(v, "month")]
    if not dates:
        return None
    if all(getattr(d, "month", 0) == 3 for d in dates):
        return [pd.Timestamp(d).normalize() for d in dates]
    return None


def _extract_annual(df):
    """Walk the Data Sheet top→bottom, tracking the current annual Report-Date axis,
    and pull each wanted field's values aligned to fiscal year-ends. Returns
    {fy_end: {field: value}}. First annual occurrence of a field wins (the annual
    block precedes the quarterly block for Sales/Net profit)."""
    # invert label → field for quick lookup
    label_to_field = {}
    for field, labels in FIELDS.items():
        for lab in labels:
            label_to_field[lab.strip().lower()] = field

    axis = None  # current annual fiscal axis (list of Timestamps) or None if quarterly
    out = {}  # fy_end -> {field: val}
    seen = set()  # fields already captured (first annual wins)
    for _, r in df.iterrows():
        cells = r.tolist()
        label = str(cells[0]).strip().lower()
        vals = cells[1:]
        if label == "report date":
            axis = _is_annual_axis(vals)
            continue
        if axis is None:
            continue
        field = label_to_field.get(label)
        if field is None or field in seen:
            continue
        seen.add(field)
        for dt, v in zip(axis, vals):
            n = _num(v)
            if n is not None:
                out.setdefault(pd.Timestamp(dt), {})[field] = n
    return out


class Blocked(RuntimeError):
    """screener.in stopped serving us. Raised so the caller can ABORT, not grind on."""


def collect_ticker(sess_mod, ticker):
    """Fetch + parse one company. Returns list of per-fiscal-year row dicts.

    Raises Blocked when the host refuses the connection, so main() can stop.
    Previously every failure was swallowed into `return []`, which turned a block into a
    silent no-op: on 2026-07-15 a run was blocked ~1 MINUTE in, then spent 40 MINUTES
    walking 2,367 tickers whose every fetch failed, and reported "2367/2367 | 31 with
    data" — a completion message for a job that collected nothing after the first
    minute. A failure that looks like progress is worse than a crash.
    """
    try:
        raw = sess_mod.company_export_bytes(ticker)
    except Exception as e:
        if "Max retries" in str(e) or "Connection" in type(e).__name__:
            raise Blocked(f"host refused connection at {ticker}: {str(e)[:60]}") from e
        return []
    if not raw:
        return []
    try:
        xl = pd.ExcelFile(io.BytesIO(raw))
        sheet = next((s for s in xl.sheet_names if "data" in s.lower()), xl.sheet_names[0])
        df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet, header=None)
    except Exception:
        return []
    annual = _extract_annual(df)
    rows = []
    for fy, fields in sorted(annual.items()):
        row = {"ticker": ticker, "fy_end": fy.strftime("%Y-%m-%d")}
        row.update(fields)
        # EBIT and capital employed, derived from reported annual line items so
        # downstream ROCE consumers never re-derive them inconsistently.
        pbt, interest = row.get("profit_before_tax"), row.get("interest")
        if pbt is not None and interest is not None:
            row["ebit"] = pbt + interest
        # Gross profit. screener.in publishes no Gross Profit / COGS row, so COGS is
        # assembled from the itemised annual cost lines. The sign convention is NOT
        # guessed — it is pinned by reconstructing reported Profit-before-tax:
        #   Sales - RawMat + ChgInv - Power - OtherMfr - Employee - Selling - OtherExp
        #         + OtherIncome - Depreciation - Interest  ==  PBT
        # On RELIANCE FY2017 that reproduces the reported PBT of 40,777 EXACTLY, and
        # only when Change in Inventory is ADDED — it is a credit to profit, hence a
        # REDUCTION of cost. The same chain independently reproduces EBIT 43,500,
        # matching pbt+interest above, so the two derivations corroborate each other.
        #
        # Employee Cost and Selling/admin are EXCLUDED: they are SG&A. Folding them in
        # would compute an OPERATING margin and label it a gross margin — which is the
        # kind of plausible-but-wrong field that is worse than a missing one.
        # GATED ON BUSINESS MODEL — this COGS definition is MANUFACTURER-ONLY.
        #
        # I previously claimed the PBT identity proved gross margin "isn't a judgement
        # call". That was wrong. The identity pins the ARITHMETIC (it proved Change in
        # Inventory is a credit, reproducing reported PBT to the rupee) but says nothing
        # about WHICH lines are cost-of-goods for a given business. TCS exposed it:
        #     Raw Material Cost 94  vs  Sales 92,693  (0.1%)  — because TCS's real cost
        #     of service is Employee Cost 48,116.
        # Formula A would report ~98% gross margin for TCS against a true ~42%, i.e. a
        # plausible-looking, badly wrong number for every IT/services firm in the universe.
        #
        # Validated against yfinance's TRUE Gross Profit (64 company-years, 17 tickers):
        #     rm/sales >30% (manufacturers)  error  -0.1pp   <- essentially exact
        #     rm/sales 5-30% (mixed)         error  +6.7pp
        #     rm/sales <5%  (services)       NO DATA to validate against
        #
        # So the formula is applied ONLY where it was proven, and SKIPPED elsewhere.
        # Skipping is not a cop-out: weigh() drops a None test from both numerator and
        # denominator, so a services firm scores on 7 tests and stays comparable. A
        # wrong-but-populated field is the failure mode this repo keeps hitting (the
        # quarterly Operating Profit splice, the ratio-inverted ROCE at 95-137%, the
        # current-ratio proxy at 62.1% vs a 57.6% coin flip).
        #
        # A services COGS (employee cost as cost of revenue) is defensible in principle
        # but screener.in does not split Employee Cost into direct vs SG&A, and there is
        # no validation sample here for it. Left unimplemented rather than guessed.
        MFR_RM_SHARE_MIN = 0.30
        rm, chg = row.get("raw_material_cost"), row.get("change_in_inventory")
        pf, omf = row.get("power_and_fuel"), row.get("other_mfr_exp")
        sales = row.get("revenue")
        if rm is not None and sales:
            row["rm_share"] = rm / sales
            if row["rm_share"] >= MFR_RM_SHARE_MIN:
                cogs = rm - (chg or 0) + (pf or 0) + (omf or 0)
                row["cogs"] = cogs
                row["gross_profit"] = sales - cogs
                row["gross_margin"] = row["gross_profit"] / sales
            # else: not a manufacturer -> gross_margin stays absent, test 8 is skipped
        # Capital employed = shareholders' funds + debt. screener.in's balance
        # sheet gives the pieces directly (Equity Share Capital + Reserves +
        # Borrowings), so this is a sum of reported figures, not an inversion.
        eq, res, bor = (row.get("equity_capital"), row.get("reserves"),
                        row.get("borrowings"))
        if eq is not None and res is not None and bor is not None:
            row["capital_employed"] = eq + res + bor
        if row.get("ebit") is not None and row.get("capital_employed"):
            row["roce"] = row["ebit"] / row["capital_employed"]
        rows.append(row)
    return rows


def _universe(args):
    if args.tickers:
        return [t.upper() for t in args.tickers]
    if os.path.exists(UNIVERSE):
        uni = pd.read_csv(UNIVERSE)
        syms = sorted(set(uni["yf_ticker"].astype(str)
                          .str.replace(r"\.(NS|BO)$", "", regex=True).str.upper()))
        if args.liquid_only and os.path.exists(LTM):
            liq = set(pd.read_parquet(LTM, columns=["Symbol"])["Symbol"].astype(str).str.upper())
            syms = [s for s in syms if s in liq]
        return syms
    # The universe CSV is a build artifact and is often absent. The LTM price
    # parquet is the better source anyway: it IS the traded universe, and
    # --liquid-only only ever intersected against it. Crashing on a missing
    # artifact when the authoritative list is right there is pointless.
    if not os.path.exists(LTM):
        raise SystemExit(f"no universe: neither {UNIVERSE} nor {LTM} exists")
    px = pd.read_parquet(LTM, columns=["Symbol", "Close", "Volume", "Date"])
    if not args.liquid_only:
        return sorted(set(px["Symbol"].astype(str).str.upper()))
    # --liquid-only MUST still filter here. The first cut of this fallback ignored
    # the flag and returned all 8,944 symbols in the LTM parquet, turning a ~28-minute
    # job into a ~3.5-HOUR one against a rate-limited host — and silently, because the
    # count is only printed once. The flag exists precisely to avoid hammering
    # screener.in for names too illiquid to trade.
    #
    # Liquidity is the same gate used everywhere else in this repo: MEDIAN daily
    # turnover >= Rs 1 crore, median so one block deal cannot lift a dead stock over
    # the bar. Applied on the last 60 sessions.
    cutoff = px["Date"].max() - pd.Timedelta(days=120)
    recent = px[px["Date"] >= cutoff].copy()
    recent["tv"] = recent["Close"] * recent["Volume"]
    liq = recent.groupby("Symbol")["tv"].median()
    keep = liq[liq >= 10_000_000].index
    return sorted(set(pd.Series(list(keep)).astype(str).str.upper()))


def main():
    ap = argparse.ArgumentParser(description="Collect India historical financials from screener.in")
    ap.add_argument("--tickers", nargs="*", help="explicit NSE symbols (else the India universe)")
    ap.add_argument("--limit", type=int, default=0, help="cap number of tickers (testing)")
    ap.add_argument(
        "--liquid-only", action="store_true", help="only names present in cache_seed/ltm/IN.parquet"
    )
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    import screener_in_auth as sess_mod

    sess_mod.session()  # login once (cached); raises if creds missing/invalid

    universe = _universe(args)
    if args.limit:
        universe = universe[: args.limit]

    done, existing = set(), []
    if os.path.exists(OUT):
        prev = pd.read_parquet(OUT)
        done = set(prev["ticker"].astype(str).str.upper())
        existing = [prev]
    todo = [t for t in universe if t not in done]
    print(
        f"screener.in history: {len(universe)} tickers, {len(done)} already done, "
        f"{len(todo)} to fetch (~{len(todo)*_RATE/60:.0f} min at {_RATE}s/req)"
    )

    rows, n_ok, t0 = [], 0, time.time()
    for i, tk in enumerate(todo, 1):
        time.sleep(_RATE)
        # CIRCUIT BREAKER. Consecutive empties mean the session died or the host is
        # blocking; continuing just deepens the block and produces a fake progress bar.
        if not got:
            _consec_empty += 1
            if _consec_empty >= MAX_CONSECUTIVE_EMPTY:
                print(f"\n  ABORT: {_consec_empty} consecutive tickers returned nothing "
                      f"— session dead or host blocking. Collected {n_ok} before this.")
                break
        else:
            _consec_empty = 0
        try:
            r = collect_ticker(sess_mod, tk)
            if r:
                rows.extend(r)
                n_ok += 1
        except Exception:
            pass
        if i % 100 == 0 or i == len(todo):
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
