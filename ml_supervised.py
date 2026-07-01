#!/usr/bin/env python3
# ml_supervised.py
# ================
# LAYER 1 of ML_Stock_Screening_System.docx — supervised forward-return classifier.
#
# Trains on the DEEP history (5y LTM) with WALK-FORWARD validation (no lookahead)
# to predict the doc's 4 return classes from fundamentals-free price/liquidity
# features. Its output (predicted Strong-Buy/Buy names) becomes the supervised
# ANCHOR that auto_screener measures unsupervised discoveries against.
#
#   label (doc Code 1, scaled to the horizon actually available):
#     forward return ≥ strong → 3 Strong Buy | ≥ buy → 2 Buy | ≥ 0 → 1 Hold | else 0 Avoid
#
# Engine: sklearn HistGradientBoostingClassifier (same GBT family as XGBoost, no new
# dep). If `xgboost` is installed it is used instead. Walk-forward = sort by as-of
# date, train past → test future (TimeSeriesSplit), so no future data leaks.
#
#   python3 ml_supervised.py --market IN --train      # train + report + save model
#   python3 ml_supervised.py --market IN              # predict current Strong/Buy names
#
# ⚠️ Research/education only. Predictions are historical associations, not advice.
# NOTE: with only ~1y of real LTM depth the horizon is a quarter (63d); accuracy
# improves and the horizon can widen toward 1y as the LTM deepens (see --horizon).

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import datalink

MODEL_DIR = Path(__file__).parent / "cache_seed" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["rsi14", "ret21", "ret63", "ret126", "pct_from_high", "pct_from_low",
            "above_200dma", "golden_cross", "vol_ratio", "log_turnover"]
CLASS_NAMES = {0: "Avoid", 1: "Hold", 2: "Buy", 3: "Strong Buy"}


# ── features at a point in time (only data up to t — no lookahead) ───────────────
def _feat_at(close: np.ndarray, vol: np.ndarray, t: int) -> Optional[dict]:
    if t < 200:
        return None
    c = close[: t + 1]
    v = vol[: t + 1]
    last = c[-1]
    if last <= 0:
        return None
    # RSI(14) Wilder
    d = np.diff(c[-60:])
    up = np.clip(d, 0, None).mean() if len(d) else 0.0
    dn = (-np.clip(d, None, 0)).mean() if len(d) else 0.0
    rsi = 100 - 100 / (1 + up / dn) if dn > 0 else 100.0
    sma50, sma200 = c[-50:].mean(), c[-200:].mean()
    hi, lo = c[-252:].max(), c[-252:].min()
    v20 = v[-20:].mean()
    v60 = v[-60:].mean() if len(v) >= 60 else v20
    return {
        "rsi14": rsi,
        "ret21": (last / c[-21] - 1) * 100 if len(c) > 21 and c[-21] else 0.0,
        "ret63": (last / c[-63] - 1) * 100 if len(c) > 63 and c[-63] else 0.0,
        "ret126": (last / c[-126] - 1) * 100 if len(c) > 126 and c[-126] else 0.0,
        "pct_from_high": (last / hi - 1) * 100 if hi else 0.0,
        "pct_from_low": (last / lo - 1) * 100 if lo else 0.0,
        "above_200dma": float(last > sma200),
        "golden_cross": float(sma50 > sma200),
        "vol_ratio": v20 / v60 if v60 else 1.0,
        "log_turnover": float(np.log1p((c[-20:] * v[-20:]).mean())),
    }


def _label(fwd_ret_pct: float, strong: float, buy: float) -> int:
    if fwd_ret_pct >= strong:
        return 3
    if fwd_ret_pct >= buy:
        return 2
    if fwd_ret_pct >= 0:
        return 1
    return 0


# ── build a walk-forward dataset from the LTM ────────────────────────────────────
def build_dataset(
    market: str, horizon: int = 63, n_asof: int = 8, strong: float = 12.0, buy: float = 6.0
) -> pd.DataFrame:
    data = datalink.load_market(market, tier="ltm") or datalink.load_market(market)
    rows = []
    for sym, df in data.items():
        if df is None or len(df) <= 200 + horizon:
            continue
        close = df["Close"].to_numpy("float64")
        vol = df["Volume"].to_numpy("float64")
        dates = df.index
        n = len(close)
        last_t = n - horizon - 1  # last as-of with a full forward window
        if last_t <= 200:
            continue
        # sample as-of points that leave room for the forward horizon
        pts = np.linspace(200, last_t, n_asof).astype(int)
        for t in np.unique(pts):
            f = _feat_at(close, vol, t)
            if f is None:
                continue
            fwd = (close[t + horizon] / close[t] - 1) * 100 if close[t] else 0.0
            f.update({"Symbol": sym, "asof": dates[t], "label": _label(fwd, strong, buy)})
            rows.append(f)
    return pd.DataFrame(rows).sort_values("asof").reset_index(drop=True)


def _make_model():
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss")
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_depth=5, learning_rate=0.05, max_iter=300)


# ── train with walk-forward validation ───────────────────────────────────────────
def train(market: str, horizon: int = 63, verbose: bool = True) -> dict:
    import joblib
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import TimeSeriesSplit

    ds = build_dataset(market, horizon=horizon)
    if len(ds) < 200:
        raise RuntimeError(f"too few training samples for {market} ({len(ds)})")
    X, y = ds[FEATURES].to_numpy(), ds["label"].to_numpy()

    accs = []
    tscv = TimeSeriesSplit(n_splits=5)
    for tr, te in tscv.split(X):  # walk-forward: past → future
        m = _make_model()
        m.fit(X[tr], y[tr])
        accs.append(accuracy_score(y[te], m.predict(X[te])))

    model = _make_model()
    model.fit(X, y)  # final fit on all data
    joblib.dump({"model": model, "features": FEATURES, "horizon": horizon}, MODEL_DIR / f"{market}.pkl")

    imp = _importances(model, X, y)
    if verbose:
        print(f"  trained {market}: {len(ds)} samples, walk-forward acc "
              f"{np.mean(accs):.3f}±{np.std(accs):.3f}")
        print("  top features: " + ", ".join(f"{FEATURES[i]}={imp[i]:.2f}" for i in np.argsort(imp)[::-1][:5]))
        print("  label mix: " + ", ".join(f"{CLASS_NAMES[c]}={int((y==c).sum())}" for c in sorted(set(y))))
    return {"market": market, "samples": len(ds), "cv_acc": float(np.mean(accs))}


def _importances(model, X, y) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    from sklearn.inspection import permutation_importance

    return permutation_importance(model, X, y, n_repeats=3, random_state=0).importances_mean


# ── predict current names ────────────────────────────────────────────────────────
def _load(market: str):
    import joblib

    p = MODEL_DIR / f"{market}.pkl"
    return joblib.load(p) if p.exists() else None


def predict_current(market: str) -> pd.DataFrame:
    bundle = _load(market)
    if bundle is None:
        return pd.DataFrame()
    data = datalink.load_market(market, tier="ltm") or datalink.load_market(market)
    rows = []
    for sym, df in data.items():
        if df is None or len(df) < 200:
            continue
        f = _feat_at(df["Close"].to_numpy("float64"), df["Volume"].to_numpy("float64"), len(df) - 1)
        if f:
            f["Symbol"] = sym
            rows.append(f)
    if not rows:
        return pd.DataFrame()
    cur = pd.DataFrame(rows)
    model = bundle["model"]
    proba = model.predict_proba(cur[FEATURES].to_numpy())
    cur["pred_class"] = proba.argmax(1)
    cur["pred_name"] = cur["pred_class"].map(CLASS_NAMES)
    cur["score"] = proba[:, 2:].sum(1)  # P(Buy) + P(Strong Buy)
    return cur.sort_values("score", ascending=False).reset_index(drop=True)


def known_good_supervised(market: str, min_score: float = 0.5) -> set:
    """Supervised anchor: names the model predicts as Buy/Strong-Buy."""
    cur = predict_current(market)
    if cur.empty:
        return set()
    return set(cur[cur["score"] >= min_score]["Symbol"])


def main() -> int:
    ap = argparse.ArgumentParser(description="Supervised forward-return classifier (doc Layer 1)")
    ap.add_argument("--market", default="IN")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--horizon", type=int, default=63, help="forward return horizon in trading days")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    if args.train:
        train(args.market, horizon=args.horizon)
        return 0

    cur = predict_current(args.market)
    if cur.empty:
        print("no model — run --train first")
        return 0
    cols = [c for c in ["Symbol", "pred_name", "score", "rsi14", "ret63", "pct_from_high"] if c in cur.columns]
    print(f"\nTop {args.top} supervised Buy/Strong-Buy — {args.market}:")
    print(cur[cur["score"] >= 0.5][cols].head(args.top).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
