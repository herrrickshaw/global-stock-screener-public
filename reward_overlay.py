#!/usr/bin/env python3
"""
reward_overlay.py — does a LIQUIDITY-TREND confirmation improve the PRICE screens?

The Piotroski×Darvas idea, one layer over: take each market's out-of-sample-robust PRICE
screen (from reward_universe_check, liquidity-gated whole universe) and require, on top of
it, a liquidity-trend condition — RISING liquidity (accumulation confirms the move) or
FALLING liquidity (attention fading). Then compare realized equal-weight portfolio alpha:

    price alone   vs   price × rising-liquidity   vs   price × falling-liquidity

Annual rebalance (30 Jun 2017-2024), 1-yr hold, benchmark = equal-weight liquid universe,
reported for the full sample and the held-out years (2022-2024). Overlay = liq_6_6 above /
below the yearly median (a 50% split, so the overlay refines rather than decimates the book).

Educational/research only; survivorship lifts port and benchmark alike — read the alpha.

  MARKET_DB=/path/... python3 reward_overlay.py
"""
import os

import numpy as np
import pandas as pd

import reward_liquidity_check as L
import reward_universe_check as U

BASE = U.BASE
LIQ_GATE = 0.50
OOS_FROM = 2022
MIN_NAMES = 5

# markets where the liquidity overlay improved the price screen OOS (reward_overlay sweep):
# market -> (label, price_factor, side, keep, overlay_direction)
OVERLAY_PICKS = {
    "HK": ("low-vol × rising volume", "vol_60", "bottom", 0.10, "rising"),
    "UK": ("low-vol × rising volume", "vol_60", "bottom", 0.20, "rising"),
    "JP": ("low-vol × rising volume", "vol_60", "bottom", 0.10, "rising"),
    "TW": ("low-vol × rising volume", "vol_60", "bottom", 0.10, "rising"),
    "SG": ("breakout × rising volume", "dist_52w_high", "top", 0.40, "rising"),
    "CA": ("low-vol × quieting volume", "vol_60", "bottom", 0.10, "falling"),
    "SE": ("breakout × quieting volume", "dist_52w_high", "top", 0.30, "falling"),
}
MIN_PRICE = {"HK": 1.0, "UK": 50.0, "JP": 100.0, "TW": 10.0, "SG": 0.2, "CA": 2.0, "SE": 10.0}


def _current(market):
    """Latest-bar price factor + liq_6_6 + liquidity per symbol (for live overlay picks)."""
    px = U._prices(market).sort_values(["Symbol", "Date"])
    rows = []
    for sym, g in px.groupby("Symbol"):
        c = pd.to_numeric(g["Close"], errors="coerce").to_numpy()
        v = pd.to_numeric(g["Volume"], errors="coerce").to_numpy()
        m = np.isfinite(c) & (c > 0)
        c, v = c[m], np.where(np.isfinite(v[m]), v[m], 0.0)
        if len(c) < 273 or c[-1] <= 0:
            continue
        dvol = c * v
        r60 = np.diff(c[-61:]) / c[-61:-1]
        liq_6_6 = (
            np.nanmean(dvol[-126:]) / np.nanmean(dvol[-252:-126]) - 1
            if np.nanmean(dvol[-252:-126]) > 0
            else np.nan
        )
        rows.append(
            (
                sym,
                float(np.nanstd(r60)),
                c[-1] / c[max(0, len(c) - 252) :].max() - 1,
                liq_6_6,
                float(np.nanmedian(dvol[-60:])),
                c[-1],
                c[-1] / c[-126] - 1,
            )
        )
    return pd.DataFrame(
        rows, columns=["ticker", "vol_60", "dist_52w_high", "liq_6_6", "dvol", "last", "ret_6m"]
    ).replace([np.inf, -np.inf], np.nan)


def overlay_picks(top=10):
    """Current names passing BOTH the price screen and its confirming liquidity overlay."""
    out = []
    for m, (label, fac, side, keep, odir) in OVERLAY_PICKS.items():
        df = _current(m).dropna(subset=[fac, "liq_6_6", "ret_6m"])
        if df.empty:
            continue
        df = df[df["last"] >= MIN_PRICE.get(m, 0)]
        df = df[df["dvol"] >= df["dvol"].quantile(1 - LIQ_GATE)]  # most-liquid half
        if df.empty:
            continue
        thr = df[fac].quantile(1 - keep) if side == "top" else df[fac].quantile(keep)
        price = df[fac] >= thr if side == "top" else df[fac] <= thr
        med = df["liq_6_6"].median()
        conf = df["liq_6_6"] >= med if odir == "rising" else df["liq_6_6"] < med
        sel = df[price & conf].sort_values("dvol", ascending=False).head(top)
        for _, r in sel.iterrows():
            out.append(
                {
                    "market": m,
                    "screen": label,
                    "ticker": r["ticker"],
                    "liq_6_6": round(float(r["liq_6_6"]), 4),
                    "ret_6m": round(float(r["ret_6m"]), 4),
                    "last_price": round(float(r["last"]), 2),
                }
            )
    path = os.path.join(BASE, "reports", "reward_overlay_picks.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(out).to_csv(path, index=False)
    return out, path


def _merged(market):
    """Price-factor panel + liquidity factors merged on (symbol, year)."""
    p = U._panel(market)
    q = L._panel(market)
    if p.empty or q.empty:
        return p
    return p.merge(q[["symbol", "year"] + L.FACTORS], on=["symbol", "year"], how="inner")


def _price_screen(df):
    """Market's OOS-robust price screen (factor, dir, keep) via fold-CV; None if none."""
    b = U._best_factor(df)
    return (b["factor"], b["dir"], b["keep"]) if b else None


def _price_mask(sub, screen):
    f, d, keep = screen
    thr = sub[f].quantile(1 - keep) if d > 0 else sub[f].quantile(keep)
    return (sub[f] >= thr) if d > 0 else (sub[f] <= thr)


def _alpha(df, screen, years):
    """Per-year winsor alpha for price-only, price×rising-liq, price×falling-liq."""
    rows = []
    for y in years:
        sub = df[df["year"] == y]
        if len(sub) < 20:
            continue
        lo, hi = sub["fwd"].quantile([0.01, 0.99])
        w = sub["fwd"].clip(lo, hi)
        bench = w.mean()
        pmask = _price_mask(sub, screen) & sub[screen[0]].notna()
        if pmask.sum() < MIN_NAMES:
            continue
        liq = sub["liq_6_6"]
        med = liq.median()
        rise = pmask & (liq >= med)
        fall = pmask & (liq < med)
        rows.append(
            {
                "price": w[pmask].mean() - bench,
                "rise": (w[rise].mean() - bench) if rise.sum() >= MIN_NAMES else np.nan,
                "fall": (w[fall].mean() - bench) if fall.sum() >= MIN_NAMES else np.nan,
                "n_p": int(pmask.sum()),
                "n_r": int(rise.sum()),
                "n_f": int(fall.sum()),
            }
        )
    return pd.DataFrame(rows)


def run(market):
    df = _merged(market)
    if df.empty:
        return None
    df = df[df["dvol"] >= df["dvol"].quantile(LIQ_GATE)]
    screen = _price_screen(df)
    if screen is None:
        return {"market": market, "screen": None}
    oos = _alpha(df, screen, [y for y in sorted(df["year"].unique()) if y >= OOS_FROM])
    if oos.empty:
        return {"market": market, "screen": None}
    f, d, keep = screen
    return {
        "market": market,
        "screen": f"{f}{'↑' if d > 0 else '↓'}{keep:.0%}",
        "price": float(oos["price"].mean()),
        "rise": float(oos["rise"].mean()),
        "fall": float(oos["fall"].mean()),
        "n_p": int(oos["n_p"].mean()),
        "n_r": int(oos["n_r"].mean()),
    }


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--picks", action="store_true", help="also write current overlay picks")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    # markets with an OOS-robust price screen (reward_universe liq-gated) + the majors
    markets = ["US", "JP", "HK", "TW", "CA", "AU", "UK", "SG", "SE"]
    print(f"\n{'='*96}")
    print(" LIQUIDITY-TREND OVERLAY ON THE PRICE SCREENS — OOS (2022-24) realized alpha/yr")
    print(f"{'='*96}")
    print(
        f" {'mkt':<4}{'price screen':<20}{'price':>9}{'×rising':>10}{'×falling':>10}"
        f"   best overlay Δ"
    )
    print(f" {'-'*94}")
    rows = []
    for m in markets:
        r = run(m)
        if not r or not r.get("screen"):
            print(f" {m:<4} (no robust price screen)")
            continue
        best = max(
            [("rising", r["rise"]), ("falling", r["fall"])],
            key=lambda x: (x[1] if x[1] == x[1] else -9),
        )
        delta = best[1] - r["price"] if best[1] == best[1] else float("nan")
        flag = "✅ helps" if delta > 0.005 else ("– neutral" if delta > -0.005 else "✗ hurts")
        print(
            f" {m:<4}{r['screen']:<20}{r['price']*100:>8.1f}%{r['rise']*100:>9.1f}%"
            f"{r['fall']*100:>9.1f}%   {best[0]} {delta*100:+.1f}pp  {flag}"
        )
        rows.append(
            {
                **{k: r[k] for k in ("market", "screen", "price", "rise", "fall")},
                "best_overlay": best[0],
                "delta": round(delta, 4),
            }
        )
    print(f"{'='*96}")
    helped = [x for x in rows if x["delta"] > 0.005]
    print(
        f" overlay IMPROVES the price screen OOS in {len(helped)}/{len(rows)}: "
        f"{', '.join(x['market'] for x in helped) or 'none'}"
    )
    print(f"{'='*96}")
    out = os.path.join(BASE, "reports", "reward_overlay.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f" written → {os.path.relpath(out, BASE)}")
    if args.picks:
        picks, ppath = overlay_picks(args.top)
        print(f" {len(picks)} liquidity-confirmed picks → {os.path.relpath(ppath, BASE)}")
    print(" ⚠ Research only; survivorship lifts both curves — read the alpha delta.\n")


if __name__ == "__main__":
    main()
