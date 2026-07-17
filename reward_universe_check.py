#!/usr/bin/env python3
"""
reward_universe_check.py — stress-test the reward-optimised PRICE screeners against the
WHOLE price universe (every listed name), not just the ~2k fundamental-covered subset the
discovery ran on. Four of the five OOS-robust screens are price-only (CA 12-mo momentum,
TW & CN low-volatility, IN >200-DMA trend), so they can be re-checked at full breadth.

Method (price-only, no fundamentals → every ticker qualifies):
  • Rebalance annually (30 Jun 2017 … 2024). At each rebalance, for every symbol with
    enough history, compute price factors (mom_12_1, vol_60, above_200dma, reversal_1m,
    dist_52w_high) and the forward 252-day return; de-mean the forward return by
    rebalance cohort → market-neutral `excess`.
  • TRAIN = rebalances <= 2021, TEST = 2022-2024 (held out). Selection is cross-validated
    within train (early vs late fold); a factor must pay in the worse fold. Report each
    market's best OOS-robust price factor at full-universe breadth, with and without a
    liquidity floor (median is outlier-robust; winsorised mean shown for context).

Educational/research only. Survivorship caveat unchanged (only still-listed names priced).

  MARKET_DB=/path/to/market_data.duckdb python3 reward_universe_check.py
"""
import argparse
import os

import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/Downloads/code/python_files")
FACTORS = {  # name -> +1 high-good / -1 low-good
    "mom_12_1": +1,
    "vol_60": -1,
    "above_200dma": +1,
    "reversal_1m": -1,
    "dist_52w_high": +1,
}
KEEP = [0.5, 0.4, 0.3, 0.2, 0.1]
HORIZON = 252
REBAL_YEARS = list(range(2017, 2025))
N_FOLD = 80
ALL = [
    "US",
    "CN",
    "JP",
    "HK",
    "KR",
    "IN",
    "TW",
    "CA",
    "AU",
    "EU",
    "UK",
    "SG",
    "SE",
    "DE",
    "SA",
    "BR",
    "CH",
    "ZA",
    "FI",
    "DK",
]


def _prices(market):
    if market == "US":
        px = pd.read_parquet(
            os.path.join(BASE, "cache_seed", "ltm", "US.parquet"),
            columns=["Date", "Close", "Volume", "Symbol"],
        )
        px["Date"] = pd.to_datetime(px["Date"])
        return px
    import warehouse as wh

    return wh.prices(market)[["Date", "Close", "Volume", "Symbol"]]


def _panel(market):
    """Whole-universe price panel: one row per (symbol, annual rebalance) with factors,
    forward 252d return and dollar-volume."""
    px = _prices(market).sort_values(["Symbol", "Date"])
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
        for rd in rebal:
            i = int(np.searchsorted(d, rd))
            j = i + HORIZON
            if i < 273 or j >= len(c) or c[i] <= 0:
                continue
            r60 = np.diff(c[i - 60 : i + 1]) / c[i - 60 : i]
            rows.append(
                (
                    sym,
                    int(str(rd)[:4]),
                    c[i - 21] / c[i - 273] - 1,  # mom_12_1
                    float(np.nanstd(r60)),  # vol_60
                    c[i] / np.mean(c[i - 200 : i]) - 1,  # above_200dma
                    c[i] / c[i - 21] - 1,  # reversal_1m
                    c[i] / c[max(0, i - 252) : i + 1].max() - 1,  # dist_52w_high
                    c[j] / c[i] - 1,  # forward return
                    float(np.nanmedian(c[i - 60 : i] * v[i - 60 : i])),  # dvol
                    float(c[i]),  # close at rebalance (for point-in-time value ratios)
                )
            )
    cols = ["symbol", "year"] + list(FACTORS) + ["fwd", "dvol", "close"]
    df = pd.DataFrame(rows, columns=cols).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["fwd"])
    lo, hi = df["fwd"].quantile([0.01, 0.99])
    df["fwd_w"] = df["fwd"].clip(lo, hi)
    df["excess"] = df["fwd"] - df.groupby("year")["fwd"].transform("median")
    df["excess_w"] = df["fwd_w"] - df.groupby("year")["fwd_w"].transform("median")
    return df


def _reward(sub):
    if len(sub) < 60:
        return None
    return {
        "n": len(sub),
        "med": float(sub["excess"].median()),
        "mean": float(sub["excess_w"].mean()),
        "win": float((sub["excess"] > 0).mean()),
    }


def _best_factor(df, split=2021):
    """Best price factor whose worse-train-fold edge > 0, evaluated OOS on the holdout."""
    train, test = df[df.year <= split], df[df.year > split]
    tyears = sorted(train.year.unique())
    if len(tyears) < 2 or test.empty:
        return None
    mid = tyears[len(tyears) // 2]
    fa, fb = train[train.year < mid], train[train.year >= mid]
    best = None
    for f, d in FACTORS.items():
        s = train[f].dropna()
        if len(s) < 200:
            continue
        for keep in KEEP:
            thr = s.quantile(1 - keep) if d > 0 else s.quantile(keep)
            op = (df[f] >= thr) if d > 0 else (df[f] <= thr)
            ra, rb = _reward(fa[op.loc[fa.index]]), _reward(fb[op.loc[fb.index]])
            if not ra or not rb or ra["n"] < N_FOLD or rb["n"] < N_FOLD:
                continue
            worse = min(ra["med"], rb["med"])
            if worse > 0 and (best is None or worse > best["worse"]):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default=",".join(ALL))
    ap.add_argument(
        "--liq", type=float, default=0.0, help="drop least-liquid frac (0=whole universe)"
    )
    args = ap.parse_args()
    mkts = [m.strip().upper() for m in args.markets.split(",")]
    print(f"\n{'='*94}")
    print(
        f" WHOLE-UNIVERSE CHECK — reward-optimised PRICE screeners  ({HORIZON}d, annual rebalance)"
    )
    print(
        f" liquidity floor: {'none (entire universe)' if args.liq == 0 else f'drop least-liquid {args.liq:.0%}'}"
    )
    print(f"{'='*94}")
    print(
        f" {'mkt':<4}{'names':>7}{'obs':>8}{'oos_n':>7}  {'best price factor':<20}{'TRAIN':>9}{'OOS':>9}{'win':>6}  V"
    )
    print(f" {'-'*90}")
    robust, out_rows = [], []
    for m in mkts:
        try:
            df = _panel(m)
        except SystemExit:
            print(f" {m:<4} (no data)")
            continue
        if args.liq > 0 and not df.empty:
            df = df[df["dvol"] >= df["dvol"].quantile(args.liq)]
        b = _best_factor(df) if not df.empty else None
        names = df["symbol"].nunique() if not df.empty else 0
        if not b:
            print(
                f" {m:<4}{names:>7}{len(df):>8}{'':>7}  {'— none robust —':<20}{'':>9}{'':>9}{'':>6}  ⚠"
            )
            out_rows.append(
                {"market": m, "names": names, "obs": len(df), "factor": None, "robust": False}
            )
            continue
        arrow = "↑" if b["dir"] > 0 else "↓"
        ok = b["test"]["med"] > 0.005 and b["test"]["win"] >= 0.50
        tag = "✅" if ok else "⚠"
        label = f"{b['factor']}{arrow}{b['keep']:.0%}"
        out_rows.append(
            {
                "market": m,
                "names": names,
                "obs": len(df),
                "factor": label,
                "train_med": round(b["train"]["med"], 4),
                "oos_med": round(b["test"]["med"], 4),
                "oos_win": round(b["test"]["win"], 3),
                "oos_n": b["test"]["n"],
                "robust": ok,
            }
        )
        print(
            f" {m:<4}{names:>7}{len(df):>8}{b['test']['n']:>7}  {label:<20}"
            f"{b['train']['med']*100:>8.1f}%{b['test']['med']*100:>8.1f}%{b['test']['win']*100:>5.0f}%  {tag}"
        )
        if ok:
            robust.append(m)
    print(f"{'='*94}")
    print(
        f" OOS-robust at full universe: {len(robust)}/{len(mkts)} — {', '.join(robust) or 'none'}"
    )
    print(f"{'='*94}\n")
    tag = "whole" if args.liq == 0 else f"liq{int(args.liq*100)}"
    out = os.path.join(BASE, "reports", f"reward_universe_{tag}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out, index=False)
    print(f" written → {os.path.relpath(out, BASE)}\n")


if __name__ == "__main__":
    main()
