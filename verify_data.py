#!/usr/bin/env python3
"""
verify_data.py
==============
Cross-checks the mailer's yfinance-sourced Close prices against an
independent, authoritative source, per market:

  US -> Databento (paid, metered -- cost-gated the same way as
        Bazartalks_Py2Cplus's libs/storage/databento_client: estimate cost
        via the free metadata.get_cost() call first, only proceed under a
        conservative threshold).
  IN -> NSE Bhavcopy (free, official EOD bulk files, via the already-built
        bhavcopy_history.py -- see that module's own get_symbol()).

Databento has NO India coverage at all (confirmed directly against its real
MetadataListDatasets()/MetadataListPublishers() API, not marketing copy --
every dataset is US equities/futures or European derivatives), which is why
India verification uses a different, free source rather than forcing a
mismatched vendor into both halves.

Both checks are READ-ONLY cross-references: neither writes back to any
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

US_DATASET = "DBEQ.BASIC"
TOLERANCE_PCT = 1.0  # flag a discrepancy beyond this relative %


def _pct_diff(reference: float, candidate: float) -> float:
    if not reference or reference != reference:
        return float("nan")
    return (candidate - reference) / reference * 100.0


def verify_us(picks: list[dict], cost_threshold_usd: float = 1.0) -> list[dict]:
    """picks: [{"symbol": str, "price": float}, ...] -- the mailer's own
    US-section prices. Fetches Databento OHLCV-1d for a small recent window
    per symbol and compares against the most recent session's Close.

    Cost-gated per call: a real (tiny, ~$0.0001/symbol for a 10-day OHLCV-1d
    window) but real charge -- estimate first via the free metadata.get_cost(),
    skip (not fetch) if the estimate exceeds cost_threshold_usd.
    """
    if db is None:
        return [
            {
                "symbol": p["symbol"],
                "status": "SKIPPED",
                "reason": "databento package not installed",
            }
            for p in picks
        ]
    if not os.environ.get("DATABENTO_API_KEY"):
        return [
            {"symbol": p["symbol"], "status": "SKIPPED", "reason": "DATABENTO_API_KEY not set"}
            for p in picks
        ]

    client = db.Historical()
    # DBEQ.BASIC's EOD data lags "today" by roughly a session (confirmed via
    # a real 422 data_end_after_available_end error querying through
    # date.today() itself) -- back off by a day so the window's end is
    # reliably within the dataset's published range. get_range()'s own
    # per-day grouping in this module already finds whatever the latest
    # ACTUAL bar within the window is, so this is just a safe upper bound,
    # not a precision requirement.
    end = _dt.date.today() - _dt.timedelta(days=1)
    start = end - _dt.timedelta(days=10)
    results = []
    for p in picks:
        sym, src_price = p["symbol"], p["price"]
        try:
            cost = client.metadata.get_cost(
                dataset=US_DATASET,
                symbols=[sym],
                schema="ohlcv-1d",
                start=start.isoformat(),
                end=end.isoformat(),
            )
            if cost > cost_threshold_usd:
                results.append(
                    {
                        "symbol": sym,
                        "status": "SKIPPED",
                        "reason": f"estimated cost ${cost:.4f} over ${cost_threshold_usd:.2f} threshold",
                    }
                )
                continue

            store = client.timeseries.get_range(
                dataset=US_DATASET,
                symbols=[sym],
                schema="ohlcv-1d",
                start=start.isoformat(),
                end=end.isoformat(),
            )
            df = store.to_df()
            if df.empty:
                results.append({"symbol": sym, "status": "NO_DATA"})
                continue
            # DBEQ.BASIC returns one row per (date, publisher/venue) -- take
            # the highest-volume venue print per day as the representative
            # bar, then the most recent day.
            df = df.reset_index()
            df["_date"] = df["ts_event"].dt.date
            last_date = df["_date"].max()
            day_rows = df[df["_date"] == last_date]
            row = day_rows.loc[day_rows["volume"].idxmax()]
            db_price = float(row["close"])
            diff = _pct_diff(db_price, src_price)
            status = "OK" if abs(diff) <= TOLERANCE_PCT else "MISMATCH"
            results.append(
                {
                    "symbol": sym,
                    "status": status,
                    "source_price": src_price,
                    "verify_price": db_price,
                    "pct_diff": diff,
                    "verify_source": "Databento",
                    "verify_date": str(last_date),
                }
            )
        except Exception as e:  # noqa: BLE001 - report per-symbol, don't abort the batch
            results.append({"symbol": sym, "status": "ERROR", "reason": str(e)})
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


def _section_html(label: str, results: list[dict], source: str) -> str:
    if not results:
        return f"<p style='font-size:12px;color:#666'>{label}: no picks to verify today.</p>"
    ok = sum(1 for r in results if r["status"] == "OK")
    mismatches = [r for r in results if r["status"] == "MISMATCH"]
    skipped = [r for r in results if r["status"] in ("SKIPPED", "NO_DATA", "ERROR")]
    head = f"<p style='font-size:12px;margin:4px 0'><b>{label}</b> (vs {source}) — {ok}/{len(results)} confirmed"
    if mismatches:
        head += (
            f", <span style='color:#b00;font-weight:600'>{len(mismatches)} discrepancy(ies)</span>"
        )
    if skipped:
        head += f", {len(skipped)} skipped"
    head += "</p>"
    rows = [head]
    for r in mismatches:
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
        "yfinance-sourced prices against an independent authoritative source per market — "
        "US via Databento (paid, metered), India via NSE Bhavcopy (free, official EOD). "
        "A discrepancy flags a data-quality question, not necessarily an error in either source "
        "(stale cache, different session cutoff, corporate action, etc.).</p>"
        + _section_html("🇺🇸 US", us_results, "Databento")
        + _section_html("🇮🇳 India", india_results, "NSE Bhavcopy")
    )


def verification_text(us_results: list[dict], india_results: list[dict]) -> str:
    lines = ["Data Verification (yfinance vs independent source):"]
    for label, results, source in (
        ("US", us_results, "Databento"),
        ("India", india_results, "NSE Bhavcopy"),
    ):
        if not results:
            lines.append(f"  {label}: no picks to verify today.")
            continue
        ok = sum(1 for r in results if r["status"] == "OK")
        lines.append(f"  {label} (vs {source}): {ok}/{len(results)} confirmed")
        for r in results:
            if r["status"] == "MISMATCH":
                lines.append(
                    f"    ! {r['symbol']}: report={r['source_price']:.2f} vs "
                    f"{source}={r['verify_price']:.2f} ({r['pct_diff']:+.2f}%)"
                )
    return "\n".join(lines)
