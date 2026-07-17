#!/usr/bin/env python3
"""
reward_liquidity_check.py — is LIQUIDITY TREND (accumulation / distribution) a stock-picking
edge? Everything so far was priced off returns (momentum, low-vol, trend, cash-flow). This
tests changes in trading liquidity over time, market by market, on the same rigorous frame
as reward_universe_check: whole price universe, annual rebalance, forward-252d return,
cohort-neutral excess, fold-CV inside train + a held-out test, liquidity-gated.

Liquidity-trend factors (from Close·Volume dollar volume; direction is SEARCHED, since
rising vs falling liquidity is not a priori bullish):
  • liq_3_12   mean $vol last 3mo / prior 9mo − 1      (recent surge vs the year)
  • liq_6_6    mean $vol last 6mo / prior 6mo − 1       (half-over-half acceleration)
  • amihud_trend  fractional DROP in Amihud illiquidity (|ret|/$vol) recent vs prior
                  ( >0 = getting more liquid / tighter )
  • vol_surge  mean share-volume last 1mo / last 1yr − 1

For each market the best (factor, direction, decile) whose WORSE train fold pays is chosen
and scored on the held-out years (2022-24). Also lists the names with the strongest current
liquidity trend (liquidity-gated). Educational/research only — survivorship caveat applies.

  MARKET_DB=/path/... python3 reward_liquidity_check.py            # sweep all markets
  MARKET_DB=/path/... python3 reward_liquidity_check.py --picks US,IN --top 12
"""
import argparse
import os

import numpy as np
import pandas as pd

import reward_universe_check as U

BASE = U.BASE
HORIZON = U.HORIZON
REBAL_YEARS = U.REBAL_YEARS
LIQ_GATE = 0.50  # keep the most-liquid half (drop microcaps where trends are noise)
KEEP = [0.5, 0.4, 0.3, 0.2, 0.1]
N_FOLD = 80
FACTORS = ["liq_3_12", "liq_6_6", "amihud_trend", "vol_surge"]  # direction searched ±1


def _panel(market):
    """Per (symbol, annual rebalance): liquidity-trend factors, forward return, liquidity."""
    px = U._prices(market).sort_values(["Symbol", "Date"])
    rebal = [np.datetime64(f"{y}-06-30") for y in REBAL_YEARS]
    rows = []
    for sym, g in px.groupby("Symbol"):
        d = g["Date"].to_numpy()
        c = pd.to_numeric(g["Close"], errors="coerce").to_numpy()
        v = pd.to_numeric(g["Volume"], errors="coerce").to_numpy()
        m = np.isfinite(c) & (c > 0)
        d, c, v = d[m], c[m], np.where(np.isfinite(v[m]), v[m], 0.0)
        if len(c) < 300:
            continue
        dvol = c * v
        illiq = np.abs(np.diff(c)) / c[:-1] / np.where(dvol[1:] > 0, dvol[1:], np.nan)
        for rd in rebal:
            i = int(np.searchsorted(d, rd))
            j = i + HORIZON
            if i < 252 or j >= len(c) or c[i] <= 0:
                continue

            def _ratio(a, b):
                a, b = np.nanmean(a), np.nanmean(b)
                return a / b - 1 if b and np.isfinite(b) and b != 0 else np.nan

            liq_3_12 = _ratio(dvol[i - 63 : i], dvol[i - 252 : i - 63])
            liq_6_6 = _ratio(dvol[i - 126 : i], dvol[i - 252 : i - 126])
            am_recent = np.nanmean(illiq[i - 63 : i - 1])
            am_prior = np.nanmean(illiq[i - 252 : i - 63])
            amihud_trend = (
                (am_prior - am_recent) / am_prior if am_prior and np.isfinite(am_prior) else np.nan
            )
            vol_surge = _ratio(v[i - 21 : i], v[i - 252 : i])
            rows.append(
                (
                    sym,
                    int(str(rd)[:4]),
                    liq_3_12,
                    liq_6_6,
                    amihud_trend,
                    vol_surge,
                    c[j] / c[i] - 1,
                    float(np.nanmedian(dvol[i - 60 : i])),
                )
            )
    cols = ["symbol", "year"] + FACTORS + ["fwd", "dvol"]
    df = pd.DataFrame(rows, columns=cols).replace([np.inf, -np.inf], np.nan).dropna(subset=["fwd"])
    if df.empty:
        return df
    lo, hi = df["fwd"].quantile([0.01, 0.99])
    df["fwd_w"] = df["fwd"].clip(lo, hi)
    df["excess"] = df["fwd"] - df.groupby("year")["fwd"].transform("median")
    return df


def _reward(sub):
    if len(sub) < 60:
        return None
    return {
        "n": len(sub),
        "med": float(sub["excess"].median()),
        "win": float((sub["excess"] > 0).mean()),
    }


def _best(df, split=2021):
    """Best (factor, direction, keep) whose WORSE train-fold median-excess > 0; scored OOS."""
    train, test = df[df.year <= split], df[df.year > split]
    ty = sorted(train.year.unique())
    if len(ty) < 2 or test.empty:
        return None
    mid = ty[len(ty) // 2]
    fa, fb = train[train.year < mid], train[train.year >= mid]
    best = None
    for f in FACTORS:
        s = train[f].dropna()
        if len(s) < 200:
            continue
        for d in (+1, -1):
            for keep in KEEP:
                thr = s.quantile(1 - keep) if d > 0 else s.quantile(keep)
                op = (df[f] >= thr) if d > 0 else (df[f] <= thr)
                ra, rb = _reward(fa[op.loc[fa.index]]), _reward(fb[op.loc[fb.index]])
                if not ra or not rb or ra["n"] < N_FOLD or rb["n"] < N_FOLD:
                    continue
                worse = min(ra["med"], rb["med"])
                # select the strongest in-sample (worse-fold) factor REGARDLESS of sign, so
                # every market reports a best-attempt; robustness is judged separately.
                if best is None or worse > best["worse"]:
                    te = _reward(test[op.loc[test.index]])
                    if te:
                        best = {
                            "factor": f,
                            "dir": d,
                            "keep": keep,
                            "worse": worse,
                            "train": _reward(train[op.loc[train.index]]),
                            "test": te,
                        }
    return best


def sweep(markets):
    print(f"\n{'='*92}")
    print(f" LIQUIDITY-TREND EDGE — whole universe, liquidity-gated, {HORIZON}d, held-out test")
    print(f"{'='*92}")
    print(
        f" {'mkt':<4}{'names':>7}{'obs':>8}{'oos_n':>7}  {'best liquidity factor':<26}"
        f"{'TRAIN':>8}{'OOS':>8}{'win':>6} V"
    )
    print(f" {'-'*90}")
    robust, out = [], []
    for m in markets:
        try:
            df = _panel(m)
        except SystemExit:
            print(f" {m:<4} (no data)")
            continue
        if df.empty:
            print(f" {m:<4} (empty)")
            continue
        df = df[df["dvol"] >= df["dvol"].quantile(LIQ_GATE)]
        b = _best(df)
        names = df["symbol"].nunique()
        if not b:
            print(f" {m:<4}{names:>7}{len(df):>8}{'':>7}  {'— none robust —':<26}")
            out.append({"market": m, "factor": None, "robust": False})
            continue
        arrow = "↑ rising" if b["dir"] > 0 else "↓ falling"
        # robust = pays in BOTH train folds (worse>0) AND holds out-of-sample
        ok = b["worse"] > 0 and b["test"]["med"] > 0.005 and b["test"]["win"] >= 0.50
        lbl = f"{b['factor']} {arrow} {b['keep']:.0%}"
        print(
            f" {m:<4}{names:>7}{len(df):>8}{b['test']['n']:>7}  {lbl:<26}"
            f"{b['train']['med']*100:>7.1f}%{b['test']['med']*100:>7.1f}%"
            f"{b['test']['win']*100:>5.0f}% {'✅' if ok else '⚠'}"
        )
        out.append(
            {
                "market": m,
                "factor": b["factor"],
                "direction": "rising" if b["dir"] > 0 else "falling",
                "keep": b["keep"],
                "train_med": round(b["train"]["med"], 4),
                "oos_med": round(b["test"]["med"], 4),
                "oos_win": round(b["test"]["win"], 3),
                "oos_n": b["test"]["n"],
                "robust": ok,
            }
        )
        if ok:
            robust.append(m)
    print(f"{'='*92}")
    print(f" liquidity-trend robust in {len(robust)}/{len(markets)}: {', '.join(robust) or 'none'}")
    print(f"{'='*92}")
    path = os.path.join(BASE, "reports", "reward_liquidity.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(out).to_csv(path, index=False)
    print(f" written → {os.path.relpath(path, BASE)}\n")
    return {r["market"]: r for r in out}


def picks(markets, results, top):
    print(f"{'='*70}\n CURRENT LIQUIDITY-TREND PICKS (research only — NOT advice)\n{'='*70}")
    for m in markets:
        r = results.get(m)
        if not r or not r.get("robust"):
            print(f"\n  {m}: no robust liquidity-trend edge — skipping picks")
            continue
        f, d, keep = r["factor"], r["direction"], r["keep"]
        df = _panel(m)
        latest = df[df["year"] == df["year"].max()].copy()
        latest = latest[latest["dvol"] >= latest["dvol"].quantile(LIQ_GATE)]
        asc = d == "falling"
        thr = latest[f].quantile(keep if asc else 1 - keep)
        sel = latest[latest[f] <= thr] if asc else latest[latest[f] >= thr]
        sel = sel.sort_values("dvol", ascending=False).head(top)
        print(f"\n  {m} — {f} {d} (top {keep:.0%}, liquidity-gated)")
        print(f"  {'ticker':<14}{f:>12}")
        for _, x in sel.iterrows():
            print(f"  {x['symbol']:<14}{x[f]*100:>11.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default=",".join(U.ALL))
    ap.add_argument("--picks", default=None, help="comma markets to list current picks for")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    mkts = [x.strip().upper() for x in args.markets.split(",")]
    results = sweep(mkts)
    if args.picks:
        picks([x.strip().upper() for x in args.picks.split(",")], results, args.top)


if __name__ == "__main__":
    main()
