#!/usr/bin/env python3
"""
reward_backtest.py — turn the reward-optimised screens into an actual PORTFOLIO backtest
and verify realized returns (not just the cohort-neutral median excess used to discover
them). For each market whose liquidity-gated whole-universe screen was OOS-robust
(reports/reward_universe_liq50.csv), we:

  • every 30 Jun (2017…2024) rank the liquid universe on the discovered factor, buy the
    top/bottom keep-fraction EQUAL-WEIGHT, hold 252 trading days, book the realized return;
  • benchmark = equal-weight of the same liquid universe that year;
  • compound annual portfolio vs benchmark → cumulative return, CAGR, annual alpha, and
    the fraction of years the portfolio beat the benchmark — reported for the FULL sample
    and for the HELD-OUT years (2022-2024) separately.

Realized yearly returns are winsorised 1/99 within the year so a single data-error / lottery
name doesn't fake the equal-weight mean; the equity curve compounds those winsorised means.
Screens rank WITHIN each year's cross-section (no look-ahead in the threshold).

Educational/research only. Survivorship caveat: the price store holds only still-listed
names, so ALL curves (portfolio AND benchmark) are lifted — read the ALPHA, not the level.

  MARKET_DB=/path/to/market_data.duckdb python3 reward_backtest.py
"""
import os

import numpy as np
import pandas as pd

import reward_universe_check as U

BASE = U.BASE
LIQ = 0.50
OOS_FROM = 2022
LIQ_CSV = os.path.join(BASE, "reports", "reward_universe_liq50.csv")


def _parse(label):
    """'vol_60↓10%' -> ('vol_60', -1, 0.10)."""
    if "↑" in label:
        f, rest = label.split("↑")
        d = +1
    else:
        f, rest = label.split("↓")
        d = -1
    return f, d, int(rest.replace("%", "")) / 100.0


def _compound(rets):
    return float(np.prod([1 + r for r in rets]) - 1) if len(rets) else float("nan")


def _cagr(rets):
    if not len(rets):
        return float("nan")
    return float(np.prod([1 + r for r in rets]) ** (1 / len(rets)) - 1)


def backtest_market(market, label):
    f, d, keep = _parse(label)
    df = U._panel(market)
    if df.empty:
        return None
    df = df[df["dvol"] >= df["dvol"].quantile(LIQ)]  # tradeable universe
    years = sorted(df["year"].unique())
    rows = []
    for y in years:
        sub = df[df["year"] == y]
        if len(sub) < 20:
            continue
        lo, hi = sub["fwd"].quantile([0.01, 0.99])
        w = sub["fwd"].clip(lo, hi)
        thr = sub[f].quantile(1 - keep) if d > 0 else sub[f].quantile(keep)
        sel = (sub[f] >= thr) if d > 0 else (sub[f] <= thr)
        if sel.sum() < 5:
            continue
        rows.append(
            {
                "year": int(y),
                "n": int(sel.sum()),
                "port": float(w[sel].mean()),
                "bench": float(w.mean()),
            }
        )
    if not rows:
        return None
    bt = pd.DataFrame(rows)
    bt["alpha"] = bt["port"] - bt["bench"]
    oos = bt[bt["year"] >= OOS_FROM]
    return {
        "market": market,
        "factor": label,
        "years": len(bt),
        "port_cagr": _cagr(bt["port"].tolist()),
        "bench_cagr": _cagr(bt["bench"].tolist()),
        "port_cum": _compound(bt["port"].tolist()),
        "bench_cum": _compound(bt["bench"].tolist()),
        "alpha_yr": float(bt["alpha"].mean()),
        "win_yrs": float((bt["alpha"] > 0).mean()),
        "oos_port_cagr": _cagr(oos["port"].tolist()),
        "oos_bench_cagr": _cagr(oos["bench"].tolist()),
        "oos_alpha_yr": float(oos["alpha"].mean()) if len(oos) else float("nan"),
        "oos_win": float((oos["alpha"] > 0).mean()) if len(oos) else float("nan"),
        "detail": bt,
    }


def main():
    liq = pd.read_csv(LIQ_CSV)
    robust = liq[liq["robust"]][["market", "factor"]].values.tolist()
    print(f"\n{'='*100}")
    print(
        " PORTFOLIO BACKTEST — reward-optimised screens (annual rebalance, equal-weight, 1-yr hold)"
    )
    print(f"{'='*100}")
    print(
        f" {'mkt':<4}{'factor':<18}{'yrs':>4}   FULL: {'port':>7}{'bench':>7}{'alpha/yr':>9}{'win':>5}"
        f"    OOS22-24: {'port':>7}{'bench':>7}{'alpha/yr':>9}{'win':>5}"
    )
    print(f" {'-'*98}")
    out = []
    for mkt, label in robust:
        r = backtest_market(mkt, label)
        if not r:
            print(f" {mkt:<4} (no backtest)")
            continue
        print(
            f" {r['market']:<4}{r['factor']:<18}{r['years']:>4}   "
            f"CAGR {r['port_cagr']*100:>6.1f}%{r['bench_cagr']*100:>6.1f}%{r['alpha_yr']*100:>8.1f}%"
            f"{r['win_yrs']*100:>4.0f}%    "
            f"CAGR {r['oos_port_cagr']*100:>6.1f}%{r['oos_bench_cagr']*100:>6.1f}%"
            f"{r['oos_alpha_yr']*100:>8.1f}%{r['oos_win']*100:>4.0f}%"
        )
        out.append({k: v for k, v in r.items() if k != "detail"})
    print(f"{'='*100}")
    beat = [o for o in out if o["oos_alpha_yr"] > 0]
    print(
        f" Portfolio beat benchmark OOS (positive alpha 2022-24): {len(beat)}/{len(out)} — "
        f"{', '.join(o['market'] for o in beat)}"
    )
    print(f"{'='*100}")
    path = os.path.join(BASE, "reports", "reward_backtest.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(out).to_csv(path, index=False)
    print(f" written → {os.path.relpath(path, BASE)}")
    print(" ⚠ Survivorship lifts both curves — read ALPHA, not level. Research only, not advice.\n")


if __name__ == "__main__":
    main()
