#!/usr/bin/env python3
"""
ml_factor_screener.py — an ML/AI factor-weighting screener for ALL markets.
===========================================================================
Instead of hand-picking one screen, this LEARNS how much each factor matters in
each market from 10 years of price history, then filters on the weighted score.

Pipeline (per market, from cache_seed/ltm/<MKT>.parquet):
  1. FACTORS  — compute standard price/technical factors per (ticker, date):
       mom_63, mom_21   momentum (multi-horizon return)
       rsi14            mean-reversion (oversold ↔ overbought)
       trend            distance above/below the 200-DMA
       near_high        proximity to the 252-day high (breakout/strength)
       near_low         proximity to the 252-day low (value/washout)
       vol21            realised volatility (risk)
  2. WEIGHT   — fit a Ridge regression of the *standardised* factors on the
       forward return (default 21 trading days). The coefficients ARE the learned,
       signed factor weights — the "AI gives weights to factors" step. Reported
       per market with the in-sample information coefficient (IC).
  3. FILTER   — apply the learned weights to each stock's LATEST factors →
       composite score; rank; keep the top-N per market.

This is deliberately regime-aware by construction: the same factor gets a POSITIVE
weight where it worked (e.g. momentum in India) and a NEGATIVE one where it didn't
(e.g. near-high in the US), because the weights are learned per market.

Educational/research only. Survivorship caveat: ltm holds only currently-listed names.

  python3 ml_factor_screener.py                       # all markets, 21d horizon, top 20
  python3 ml_factor_screener.py --markets US,IN,EU --top 15
  python3 ml_factor_screener.py --horizon 63 --weights-only   # just print learned weights
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

BASE = os.path.expanduser("~/Downloads/code/python_files")
LTM = os.path.join(BASE, "cache_seed", "ltm")
OUT = os.path.join(BASE, "data", "curated", "ml_factor_picks.parquet")
WEIGHTS_OUT = os.path.join(BASE, "data", "curated", "ml_factor_weights.parquet")
FACTORS = ["mom_63", "mom_21", "rsi14", "trend", "near_high", "near_low", "vol21"]


def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _add_factors(g):
    """g = one symbol's OHLC sorted by date. Adds factor columns."""
    c = g["Close"]
    g["mom_63"] = c / c.shift(63) - 1
    g["mom_21"] = c / c.shift(21) - 1
    g["rsi14"] = _rsi(c, 14)
    g["trend"] = c / c.rolling(200).mean() - 1
    g["near_high"] = c / c.rolling(126).max() - 1  # 6-mo high (short-history markets qualify)
    g["near_low"] = c / c.rolling(126).min() - 1
    g["vol21"] = c.pct_change().rolling(21).std() * np.sqrt(252)
    g["dvol"] = (c * g["Volume"]).rolling(63).mean()  # 63-day avg dollar volume (liquidity)
    return g


def _panel(market, sample):
    p = os.path.join(LTM, f"{market}.parquet")
    if not os.path.exists(p):
        return None
    d = pd.read_parquet(p, columns=["Date", "Close", "Volume", "Symbol"])
    d["Date"] = pd.to_datetime(d["Date"])
    syms = d["Symbol"].drop_duplicates()
    if sample and len(syms) > sample:
        syms = syms.sample(sample, random_state=7)
        d = d[d["Symbol"].isin(set(syms))]
    d = d.sort_values(["Symbol", "Date"])
    return d.groupby("Symbol", group_keys=False).apply(_add_factors)


def train_weights(market, horizon, sample=500):
    """Learn signed factor weights (Ridge on standardised factors → fwd return)."""
    d = _panel(market, sample)
    if d is None or d.empty:
        return None
    d["fwd"] = d.groupby("Symbol", group_keys=False)["Close"].transform(
        lambda s: s.shift(-horizon) / s - 1
    )
    train = d.dropna(subset=FACTORS + ["fwd"])
    # need a real sample AND breadth — tiny/short-history markets overfit wildly.
    if len(train) < 5000 or train["Symbol"].nunique() < 60:
        return None
    scaler = StandardScaler().fit(train[FACTORS])
    X = scaler.transform(train[FACTORS])
    X = np.clip(X, -8, 8)  # cap extreme z-scores (near-constant factors) so weights stay sane
    # winsorize the forward return to its 1st–99th pct so a few lottery/survivorship
    # movers don't let raw volatility masquerade as a return factor.
    y = train["fwd"].clip(*train["fwd"].quantile([0.01, 0.99]).to_numpy()).to_numpy()
    model = Ridge(alpha=50.0).fit(X, y)  # strong regularization → stable, comparable weights
    # sanity: on standardized factors + winsorized returns a real coef is ~O(0.01–0.1).
    # Anything huge means a degenerate/near-constant factor on a thin market — reject.
    if np.abs(model.coef_).max() > 0.5:
        return None
    composite = X @ model.coef_
    ic = np.corrcoef(composite, y)[0, 1]  # in-sample information coefficient
    return {
        "weights": dict(zip(FACTORS, model.coef_)),
        "mean": dict(zip(FACTORS, scaler.mean_)),
        "scale": dict(zip(FACTORS, scaler.scale_)),
        "ic": float(ic),
        "n": len(train),
    }


def score_current(market, fit, top):
    """Apply learned weights to each stock's LATEST factors, split by LIQUIDITY tier
    (per-market terciles of 63-day avg dollar volume), and return the top-N movers
    within each of High / Medium / Low liquidity."""
    d = _panel(market, sample=0)  # full universe for scoring
    if d is None:
        return pd.DataFrame()
    latest = d.dropna(subset=FACTORS + ["dvol"]).groupby("Symbol").tail(1)
    if latest.empty:
        return pd.DataFrame()
    z = (latest[FACTORS] - pd.Series(fit["mean"])) / pd.Series(fit["scale"])
    w = pd.Series(fit["weights"])
    latest = latest.copy()
    latest["score"] = z[FACTORS].mul(w, axis=1).sum(axis=1)
    latest["market"] = market
    q1, q2 = latest["dvol"].quantile([1 / 3, 2 / 3])
    latest["liq_tier"] = np.where(
        latest["dvol"] >= q2, "High", np.where(latest["dvol"] >= q1, "Medium", "Low")
    )
    out = (
        latest.sort_values("score", ascending=False).groupby("liq_tier", group_keys=False).head(top)
    )
    return out[["market", "liq_tier", "Symbol", "Date", "Close", "dvol", "score"] + FACTORS]


def main():
    ap = argparse.ArgumentParser(description="ML factor-weighting screener (all markets)")
    ap.add_argument("--markets", default="", help="comma list (default: all ltm markets)")
    ap.add_argument("--horizon", type=int, default=21, help="forward days to learn weights on")
    ap.add_argument("--top", type=int, default=20, help="picks per market")
    ap.add_argument("--sample", type=int, default=500, help="training tickers/market")
    ap.add_argument("--weights-only", action="store_true", help="print learned weights, no picks")
    args = ap.parse_args()

    if args.markets:
        markets = [m.strip().upper() for m in args.markets.split(",")]
    else:
        markets = sorted(
            os.path.basename(p)[:-8] for p in glob.glob(os.path.join(LTM, "*.parquet"))
        )

    print(f"ML factor screener — {args.horizon}d horizon — learning per-market factor weights\n")
    print("  " + "Mkt".ljust(5) + "IC".rjust(6) + "   " + "  ".join(f.rjust(9) for f in FACTORS))
    wrows, picks = [], []
    for m in markets:
        fit = train_weights(m, args.horizon, args.sample)
        if not fit:
            print(f"  {m:<5}  (insufficient data)")
            continue
        w = fit["weights"]
        print(f"  {m:<5}{fit['ic']:>6.3f}   " + "  ".join(f"{w[f]*1000:>+9.2f}" for f in FACTORS))
        wrows.append(
            {"market": m, "ic": fit["ic"], "n": fit["n"], **{f"w_{f}": w[f] for f in FACTORS}}
        )
        if not args.weights_only:
            picks.append(score_current(m, fit, args.top))

    if wrows:
        os.makedirs(os.path.dirname(WEIGHTS_OUT), exist_ok=True)
        pd.DataFrame(wrows).to_parquet(WEIGHTS_OUT, index=False)
    print("\n  (weights ×1000 for readability; sign = direction the factor pays in that market)")
    print(f"  weights → {os.path.relpath(WEIGHTS_OUT, BASE)}")

    if picks and not args.weights_only:
        allp = pd.concat([p for p in picks if not p.empty], ignore_index=True)
        allp.to_parquet(OUT, index=False)
        # print top movers per liquidity tier, ordered High → Medium → Low
        show = min(5, args.top)
        for tier in ("High", "Medium", "Low"):
            t = allp[allp["liq_tier"] == tier]
            print(f"\n  ═══ {tier} liquidity movers (top {show}/market) ═══")
            for m in sorted(t["market"].unique()):
                names = t[t["market"] == m].head(show)
                cells = "  ".join(f"{r['Symbol']}({r['score']:+.2f})" for _, r in names.iterrows())
                print(f"    {m:<4} {cells}")
        print(
            f"\n  picks → {os.path.relpath(OUT, BASE)} "
            f"({len(allp)} rows · {allp['market'].nunique()} markets · High/Med/Low tiers)"
        )


if __name__ == "__main__":
    main()
