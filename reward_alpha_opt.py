#!/usr/bin/env python3
"""
reward_alpha_opt.py — re-optimise a market's screener directly on REALIZED PORTFOLIO
ALPHA (equal-weight portfolio return − equal-weight benchmark return), not the cohort
median-excess that reward_screener_opt used. Built to answer: once we score on the metric
that actually failed for US & JP (mean alpha, not median), what screen — if any — beats
the benchmark out-of-sample?

Method (annual rebalance 30 Jun 2017-2024, liquid universe, 1-yr hold):
  • Candidate factors: the 5 price factors (momentum, low-vol, trend, reversal, near-high)
    PLUS — where deep dated fundamentals exist (US SEC) — point-in-time value/quality
    factors (book/market, earnings yield, ROA, gross margin, CFO/assets, low leverage),
    joined as the latest 10-K filed BEFORE each rebalance. JP has no deep dated
    fundamentals, so it is price-only.
  • Reward of a screen = MEAN of the yearly alphas (winsorised within year). Each filter
    ranks WITHIN the year (relative keep-fraction → no threshold look-ahead).
  • Selection is cross-validated: alpha must be positive in BOTH train folds (early/late).
    Greedy up to 2 filters. TEST = held-out 2022-2024 mean alpha + CAGR.

Educational/research only; survivorship lifts port and benchmark alike — read the alpha.

  MARKET_DB=/path/... python3 reward_alpha_opt.py --markets US,JP
"""
import argparse
import os

import numpy as np
import pandas as pd

import reward_universe_check as U

BASE = U.BASE
LIQ = 0.50
SPLIT = 2021
OOS_FROM = 2022
KEEP = [0.5, 0.4, 0.3, 0.2, 0.1]
# fundamental factor -> +1 high-good / -1 low-good
FUND_DIRS = {
    "book_to_market": +1,
    "earnings_yield": +1,
    "roa": +1,
    "gross_margin": +1,
    "cfo_ta": +1,
    "leverage": -1,
}


def _with_fundamentals(df, market):
    """US only: join the latest SEC filing known at each 30-Jun rebalance and derive
    point-in-time value/quality factors using the rebalance close."""
    if market != "US":
        return df, list(U.FACTORS)
    fp = os.path.join(BASE, "cache_seed", "fundamentals_history", "US.parquet")
    if not os.path.exists(fp):
        return df, list(U.FACTORS)
    f = pd.read_parquet(fp)
    f["filed"] = pd.to_datetime(f["filed"], errors="coerce")
    f = f.dropna(subset=["filed"]).sort_values("filed")
    out = []
    for yr, g in df.groupby("year"):
        asof = pd.Timestamp(int(yr), 6, 30)
        fy = f[f["filed"] <= asof].groupby("ticker").tail(1)
        m = g.merge(fy, left_on="symbol", right_on="ticker", how="left", suffixes=("", "_f"))
        out.append(m)
    d = pd.concat(out, ignore_index=True)
    cap = d["shares"] * d["close"]
    d["book_to_market"] = np.where(cap > 0, d["equity"] / cap, np.nan)
    d["earnings_yield"] = np.where(cap > 0, d["net_income"] / cap, np.nan)
    d["roa"] = d["net_income"] / d["total_assets"]
    d["gross_margin"] = d["gross_profit"] / d["revenue"].where(d["revenue"] > 0)
    d["cfo_ta"] = d["cfo"] / d["total_assets"]
    d["leverage"] = d["long_term_debt"] / d["total_assets"]
    for k in FUND_DIRS:
        d[k] = pd.to_numeric(d[k], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return d, list(U.FACTORS) + list(FUND_DIRS)


def _dirs(factors):
    d = dict(U.FACTORS)
    d.update(FUND_DIRS)
    return {f: d[f] for f in factors}


def _mask(sub, f, direction, keep):
    s = sub[f]
    thr = s.quantile(1 - keep) if direction > 0 else s.quantile(keep)
    m = (s >= thr) if direction > 0 else (s <= thr)
    return m & s.notna()


def _alpha_over(df, years, filters):
    """Mean winsorised yearly alpha of the equal-weight portfolio for a set of filters."""
    alphas, ns = [], []
    for y in years:
        sub = df[df["year"] == y]
        if len(sub) < 20:
            continue
        lo, hi = sub["fwd"].quantile([0.01, 0.99])
        w = sub["fwd"].clip(lo, hi)
        m = pd.Series(True, index=sub.index)
        for f, direction, keep in filters:
            m &= _mask(sub, f, direction, keep)
        if m.sum() < 5:
            continue
        alphas.append(float(w[m].mean() - w.mean()))
        ns.append(int(m.sum()))
    if len(alphas) < 2:
        return None
    return {
        "alpha": float(np.mean(alphas)),
        "yrs": len(alphas),
        "win": float(np.mean([a > 0 for a in alphas])),
        "n": int(np.mean(ns)),
        "cagr": float(np.prod([1 + a for a in alphas]) ** (1 / len(alphas)) - 1),
    }


def optimise(market):
    df = U._panel(market)
    if df.empty:
        return None
    df = df[df["dvol"] >= df["dvol"].quantile(LIQ)].copy()
    df, factors = _with_fundamentals(df, market)
    dirs = _dirs(factors)
    tr_years = [y for y in sorted(df["year"].unique()) if y <= SPLIT]
    te_years = [y for y in sorted(df["year"].unique()) if y >= OOS_FROM]
    if len(tr_years) < 4 or len(te_years) < 2:
        return None
    mid = tr_years[len(tr_years) // 2]
    fa, fb = [y for y in tr_years if y < mid], [y for y in tr_years if y >= mid]

    def cv(filters):
        ra, rb = _alpha_over(df, fa, filters), _alpha_over(df, fb, filters)
        if not ra or not rb:
            return None
        return min(ra["alpha"], rb["alpha"])

    # univariate pool: best keep per factor whose worse-fold alpha > 0
    pool = {}
    for f, direction in dirs.items():
        if df[f].notna().sum() < 200:
            continue
        for keep in KEEP:
            c = cv([(f, direction, keep)])
            if c is not None and c > 0 and (f not in pool or c > pool[f][1]):
                pool[f] = ((f, direction, keep), c)
    ranked = sorted(pool.values(), key=lambda x: -x[1])
    # greedy up to 2 filters on worse-fold alpha
    chosen, cur = [], -1e9
    used = set()
    while len(chosen) < 2 and ranked:
        best = None
        for filt, _ in ranked:
            if filt[0] in used:
                continue
            c = cv(chosen + [filt])
            if c is not None and c > (best[1] if best else -1e9):
                best = (filt, c)
        if not best or best[1] <= cur + 0.004:
            break
        chosen.append(best[0])
        used.add(best[0][0])
        cur = best[1]
    if not chosen and ranked:
        chosen = [ranked[0][0]]
    return {
        "market": market,
        "chosen": chosen,
        "train": _alpha_over(df, tr_years, chosen),
        "test": _alpha_over(df, te_years, chosen),
        "full": _alpha_over(df, sorted(df["year"].unique()), chosen),
        "pool": ranked,
    }


def _lbl(filters):
    return " & ".join(f"{f}{'↑' if d>0 else '↓'}{k:.0%}" for f, d, k in filters) or "(none)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="US,JP")
    args = ap.parse_args()
    print(f"\n{'='*84}")
    print(" RE-OPTIMISED ON REALIZED PORTFOLIO ALPHA (mean yearly, equal-weight, 1-yr hold)")
    print(f"{'='*84}")
    out = []
    for m in [x.strip().upper() for x in args.markets.split(",")]:
        r = optimise(m)
        if not r:
            print(f"\n {m}: no data")
            continue
        te, tr = r["test"], r["train"]
        print(f"\n {m} — alpha-optimal screen: {_lbl(r['chosen'])}")
        print(
            "   top alpha factors (worse-fold):  "
            + ", ".join(f"{_lbl([f])}={c*100:+.1f}%" for f, c in r["pool"][:5])
        )
        if tr:
            print(
                f"   TRAIN  alpha/yr {tr['alpha']*100:+5.1f}%  win {tr['win']*100:3.0f}%  (~{tr['n']} names)"
            )
        if te:
            robust = te["alpha"] > 0 and te["win"] >= 0.5
            print(
                f"   OOS    alpha/yr {te['alpha']*100:+5.1f}%  win {te['win']*100:3.0f}%  "
                f"CAGR-alpha {te['cagr']*100:+.1f}%   {'✅ beats benchmark OOS' if robust else '⚠ no OOS alpha'}"
            )
            out.append(
                {
                    "market": m,
                    "screen": _lbl(r["chosen"]),
                    "train_alpha": tr["alpha"] if tr else None,
                    "oos_alpha": te["alpha"],
                    "oos_win": te["win"],
                    "robust": bool(robust),
                }
            )
    print(f"\n{'='*84}")
    if out:
        path = os.path.join(BASE, "reports", "reward_alpha_reopt.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame(out).to_csv(path, index=False)
        print(f" written → {os.path.relpath(path, BASE)}")
    print(" ⚠ Research only; survivorship lifts both curves — read the alpha.\n")


if __name__ == "__main__":
    main()
