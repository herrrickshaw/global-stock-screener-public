#!/usr/bin/env python3
# factor_analysis.py
# ==================
# F-07 — Fama-French 3-factor attribution: regress a stock's excess returns on
# Mkt-RF, SMB, HML → alpha (skill) + factor betas + R². Uses reference_seed/
# french_ff3.parquet. The OLS is pure + tested; return alignment is best-effort.
# Research/education only. Not advice.
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
FF3 = HERE / "reference_seed" / "french_ff3.parquet"


def ols(y: List[float], factors: Dict[str, List[float]]) -> dict:
    """OLS of y on the factor columns (+ intercept). Returns alpha, betas, r2."""
    yv = np.asarray(y, float)
    names = list(factors)
    X = np.column_stack([np.ones(len(yv))] + [np.asarray(factors[n], float) for n in names])
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((yv - pred) ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2)) or 1.0
    return {
        "alpha": round(float(coef[0]), 4),
        "betas": {n: round(float(c), 3) for n, c in zip(names, coef[1:])},
        "r2": round(1 - ss_res / ss_tot, 3),
        "n": len(yv),
    }


def _load_ff3() -> "any":
    import pandas as pd

    d = pd.read_parquet(FF3)
    for c in ("Mkt-RF", "SMB", "HML", "RF"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def analyze_returns(monthly_excess: List[float], ff_rows: List[dict]) -> dict:
    """Regress a monthly excess-return series on aligned FF3 rows (same length)."""
    n = min(len(monthly_excess), len(ff_rows))
    if n < 6:
        return {}
    y = monthly_excess[-n:]
    f = {k: [r[k] for r in ff_rows[-n:]] for k in ("Mkt-RF", "SMB", "HML")}
    return ols(y, f)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fama-French 3-factor attribution")
    ap.add_argument("--demo", action="store_true", help="run on synthetic returns")
    args = ap.parse_args()
    if args.demo:
        rng = np.random.default_rng(0)
        mkt = rng.normal(1, 3, 60)
        y = 0.4 + 1.1 * mkt + rng.normal(0, 0.5, 60)
        print(ols(list(y), {"Mkt-RF": list(mkt), "SMB": list(rng.normal(0, 1, 60))}))
    else:
        print("provide a return series via analyze_returns(); FF3 rows:", len(_load_ff3()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
