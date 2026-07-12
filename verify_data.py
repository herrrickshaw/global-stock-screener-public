#!/usr/bin/env python3
"""
verify_data.py
==============
Cross-checks the mailer's yfinance-sourced Close prices against an
independent source, per market -- FREE sources first, Databento (paid,
metered) only as a backup when a free source can't confirm:

  US -> 1. Stooq (free, no key -- already this repo's own data_sources.py
           fallback provider). If Stooq's Close confirms the report's price
           within tolerance, that's the whole check -- Databento is never
           called, and nothing is billed.
        2. Databento (paid, metered -- cost-gated the same way as
           Bazartalks_Py2Cplus's libs/storage/databento_client: estimate
           cost via the free metadata.get_cost() call first, only proceed
           under a conservative threshold) -- ONLY reached if Stooq had no
           data for the symbol, or Stooq's own price also disagreed with
           the report (an escalation, not a routine check).
  IN -> NSE Bhavcopy (free, official EOD bulk files, via the already-built
        bhavcopy_history.py -- see that module's own get_symbol()). Already
        free, so there's no cheaper fallback to try first.

Databento has NO India coverage at all (confirmed directly against its real
MetadataListDatasets()/MetadataListPublishers() API, not marketing copy --
every dataset is US equities/futures or European derivatives), which is why
India verification uses a different, free source rather than forcing a
mismatched vendor into both halves.

All checks are READ-ONLY cross-references: none of them write back to any
source, they only annotate the report with a confirm/discrepancy verdict.

Usage:
    from verify_data import verify_us, verify_india, verification_html, verification_text
    us = verify_us([{"symbol": "AAPL", "price": 210.50}, ...])
    india = verify_india([{"symbol": "HFCL", "price": 212.85}, ...])
    html_section = verification_html(us, india)
"""
from __future__ import annotations

import datetime as _dt
import os

try:
    import databento as db
except ImportError:  # pragma: no cover - optional dependency
    db = None

import bhavcopy_history as bh
from data_sources import stooq_fetch

US_DATASET = "DBEQ.BASIC"
TOLERANCE_PCT = 1.0  # flag a discrepancy beyond this relative %


def _pct_diff(reference: float, candidate: float) -> float:
    if not reference or reference != reference:
        return float("nan")
    return (candidate - reference) / reference * 100.0


def _verify_us_via_stooq(sym: str, src_price: float) -> dict | None:
    """Free, no-key check via this repo's own stooq_fetch(). Returns a result
    dict if Stooq had data for the symbol (whether it confirms or disagrees),
    or None if Stooq simply had nothing (unsupported ticker, rate-limited,
    HTML/limit page instead of CSV -- stooq_fetch() already treats all of
    these as a plain miss).

    KNOWN CURRENT LIMITATION (confirmed 2026-07-12): Stooq is now serving a
    JavaScript bot-verification challenge page instead of CSV for direct
    requests-based fetches from this environment -- every call currently
    returns None and escalates to Databento, so this free-first design isn't
    saving real cost right now. This is Stooq's own anti-bot measure, not a
    bug here; per this session's standing rule, bot-detection is not
    something to work around. The fallback chain itself is still correct and
    will resume saving cost automatically the moment Stooq is reachable
    again (or if a different free source is substituted here) -- no code
    change needed either way.
    """
    data = stooq_fetch([sym], min_bars=1, verbose=False)
    if sym not in data or data[sym].empty:
        return None
    row = data[sym].iloc[-1]
    stooq_price = float(row["Close"])
    diff = _pct_diff(stooq_price, src_price)
    status = "OK" if abs(diff) <= TOLERANCE_PCT else "MISMATCH"
    return {
        "symbol": sym,
        "status": status,
        "source_price": src_price,
        "verify_price": stooq_price,
        "pct_diff": diff,
        "verify_source": "Stooq (free)",
        "verify_date": str(data[sym].index[-1].date()),
    }


def _verify_us_via_databento(sym: str, src_price: float, cost_threshold_usd: float) -> dict:
    """Paid, metered fallback -- only called when Stooq (free) couldn't
    confirm the price itself. Cost-gated: a real (tiny, ~$0.0001/symbol for
    a 10-day OHLCV-1d window) but real charge -- estimate first via the
    free metadata.get_cost(), skip (not fetch) if the estimate exceeds
    cost_threshold_usd."""
    if db is None:
        return {"symbol": sym, "status": "SKIPPED", "reason": "databento package not installed"}
    if not os.environ.get("DATABENTO_API_KEY"):
        return {"symbol": sym, "status": "SKIPPED", "reason": "DATABENTO_API_KEY not set"}

    client = db.Historical()
    # DBEQ.BASIC's EOD data lags "today" by roughly a session (confirmed via
    # a real 422 data_end_after_available_end error querying through
    # date.today() itself) -- back off by a day so the window's end is
    # reliably within the dataset's published range. The per-day grouping
    # below already finds whatever the latest ACTUAL bar within the window
    # is, so this is just a safe upper bound, not a precision requirement.
    end = _dt.date.today() - _dt.timedelta(days=1)
    start = end - _dt.timedelta(days=10)
    try:
        cost = client.metadata.get_cost(
            dataset=US_DATASET,
            symbols=[sym],
            schema="ohlcv-1d",
            start=start.isoformat(),
            end=end.isoformat(),
        )
        if cost > cost_threshold_usd:
            return {
                "symbol": sym,
                "status": "SKIPPED",
                "reason": f"estimated cost ${cost:.4f} over ${cost_threshold_usd:.2f} threshold",
            }

        store = client.timeseries.get_range(
            dataset=US_DATASET,
            symbols=[sym],
            schema="ohlcv-1d",
            start=start.isoformat(),
            end=end.isoformat(),
        )
        df = store.to_df()
        if df.empty:
            return {"symbol": sym, "status": "NO_DATA"}
        # DBEQ.BASIC returns one row per (date, publisher/venue) -- take the
        # highest-volume venue print per day as the representative bar, then
        # the most recent day. See databento_client_comparison.md's Finding
        # #2 -- the C++ ingest had the same multi-row exposure, fixed there
        # to match this exact convention.
        df = df.reset_index()
        df["_date"] = df["ts_event"].dt.date
        last_date = df["_date"].max()
        day_rows = df[df["_date"] == last_date]
        row = day_rows.loc[day_rows["volume"].idxmax()]
        db_price = float(row["close"])
        diff = _pct_diff(db_price, src_price)
        status = "OK" if abs(diff) <= TOLERANCE_PCT else "MISMATCH"
        return {
            "symbol": sym,
            "status": status,
            "source_price": src_price,
            "verify_price": db_price,
            "pct_diff": diff,
            "verify_source": "Databento",
            "verify_date": str(last_date),
        }
    except Exception as e:  # noqa: BLE001 - report per-symbol, don't abort the batch
        return {"symbol": sym, "status": "ERROR", "reason": str(e)}


def verify_us(picks: list[dict], cost_threshold_usd: float = 1.0) -> list[dict]:
    """picks: [{"symbol": str, "price": float}, ...] -- the mailer's own
    US-section prices.

    FREE-FIRST, PAID-BACKUP: tries Stooq (free, no key) for each symbol
    first. If Stooq confirms the report's price within tolerance, that's
    the whole check for that symbol -- Databento is never called, nothing
    is billed. Databento is only reached per-symbol when Stooq had no data,
    or when Stooq's own price ALSO disagreed with the report (a genuine
    escalation, not a routine per-run cost).
    """
    results = []
    for p in picks:
        sym, src_price = p["symbol"], p["price"]

        stooq_result = _verify_us_via_stooq(sym, src_price)
        if stooq_result is not None and stooq_result["status"] == "OK":
            results.append(stooq_result)
            continue

        # Stooq either had nothing, or itself disagreed -- escalate to the
        # paid, metered backup for an authoritative tie-breaker.
        escalation_reason = (
            "Stooq had no data for this symbol"
            if stooq_result is None
            else f"Stooq itself showed a {stooq_result['pct_diff']:+.2f}% discrepancy -- escalating for confirmation"
        )
        databento_result = _verify_us_via_databento(sym, src_price, cost_threshold_usd)
        databento_result["escalation_reason"] = escalation_reason
        if stooq_result is not None:
            databento_result["stooq_price"] = stooq_result["verify_price"]
        results.append(databento_result)
    return results


def verify_india(picks: list[dict]) -> list[dict]:
    """Same shape as verify_us but against NSE Bhavcopy (free, no cost-gate
    needed) via bhavcopy_history.get_symbol(). Requires the bhavcopy cache to
    already be populated -- see README: refresh with
    `BHAV_CACHE=... python3 -c "import bhavcopy_history as bh; bh.fetch_history(n_days=5)"`.
    """
    results = []
    for p in picks:
        sym, src_price = p["symbol"], p["price"]
        df = bh.get_symbol(sym, n_days=10)
        if df is None or df.empty:
            results.append({"symbol": sym, "status": "NO_DATA"})
            continue
        bhav_price = float(df.iloc[-1]["Close"])
        diff = _pct_diff(bhav_price, src_price)
        status = "OK" if abs(diff) <= TOLERANCE_PCT else "MISMATCH"
        results.append(
            {
                "symbol": sym,
                "status": status,
                "source_price": src_price,
                "verify_price": bhav_price,
                "pct_diff": diff,
                "verify_source": "NSE Bhavcopy",
                "verify_date": str(df.index[-1].date()),
            }
        )
    return results


def _section_html(label: str, results: list[dict]) -> str:
    # `verify_source` now varies per row (Stooq for most US symbols, Databento
    # only for the ones that escalated) -- shown per-row, not as one section-
    # wide source string.
    if not results:
        return f"<p style='font-size:12px;color:#666'>{label}: no picks to verify today.</p>"
    ok = sum(1 for r in results if r["status"] == "OK")
    mismatches = [r for r in results if r["status"] == "MISMATCH"]
    skipped = [r for r in results if r["status"] in ("SKIPPED", "NO_DATA", "ERROR")]
    escalated = sum(1 for r in results if "escalation_reason" in r)
    head = f"<p style='font-size:12px;margin:4px 0'><b>{label}</b> — {ok}/{len(results)} confirmed"
    if mismatches:
        head += (
            f", <span style='color:#b00;font-weight:600'>{len(mismatches)} discrepancy(ies)</span>"
        )
    if skipped:
        head += f", {len(skipped)} skipped"
    if escalated:
        head += f" <span style='color:#888'>({escalated} escalated to a paid source)</span>"
    head += "</p>"
    rows = [head]
    for r in mismatches:
        source = r.get("verify_source", "?")
        rows.append(
            f"<p style='font-size:11px;margin:2px 0 2px 12px;color:#b00'>⚠ {r['symbol']}: "
            f"report={r['source_price']:.2f} vs {source}={r['verify_price']:.2f} "
            f"({r['pct_diff']:+.2f}%, as of {r['verify_date']})</p>"
        )
    return "".join(rows)


def verification_html(us_results: list[dict], india_results: list[dict]) -> str:
    return (
        "<h3 style='margin:10px 0 4px'>🔍 Data Verification</h3>"
        "<p style='font-size:11px;color:#888;margin:0 0 6px'>Cross-checking this report's "
        "yfinance-sourced prices against an independent source per market — "
        "US via Stooq first (free, no key), Databento (paid, metered) only when Stooq can't "
        "confirm; India via NSE Bhavcopy (free, official EOD, no paid fallback needed). "
        "A discrepancy flags a data-quality question, not necessarily an error in either source "
        "(stale cache, different session cutoff, corporate action, etc.).</p>"
        + _section_html("🇺🇸 US", us_results)
        + _section_html("🇮🇳 India", india_results)
    )


def verification_text(us_results: list[dict], india_results: list[dict]) -> str:
    lines = ["Data Verification (yfinance vs independent source, free-first):"]
    for label, results in (("US", us_results), ("India", india_results)):
        if not results:
            lines.append(f"  {label}: no picks to verify today.")
            continue
        ok = sum(1 for r in results if r["status"] == "OK")
        escalated = sum(1 for r in results if "escalation_reason" in r)
        suffix = f" ({escalated} escalated to a paid source)" if escalated else ""
        lines.append(f"  {label}: {ok}/{len(results)} confirmed{suffix}")
        for r in results:
            if r["status"] == "MISMATCH":
                source = r.get("verify_source", "?")
                lines.append(
                    f"    ! {r['symbol']}: report={r['source_price']:.2f} vs "
                    f"{source}={r['verify_price']:.2f} ({r['pct_diff']:+.2f}%)"
                )
    return "\n".join(lines)
