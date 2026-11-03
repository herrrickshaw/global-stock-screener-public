"""Characterisation of MLSignalEngine.predict on seeded synthetic OHLC (all model types)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ml_signal_engine as mse

GOLDEN = Path(__file__).parent / "fixtures" / "golden_ml_predict.json"


def _ohlc(seed: int, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.003, n)),
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": rng.uniform(1e5, 1e6, n),
        },
        index=pd.bdate_range("2023-01-02", periods=n),
    )


@pytest.mark.skipif(not mse._SKLEARN_OK, reason="sklearn missing")
def test_predict_matches_golden():
    snap = {}
    for model in ("ridge", "lr"):
        eng = mse.MLSignalEngine(model_type=model)
        for seed, n in [(1, 600), (2, 450), (3, 380), (4, 120), (5, 700), (6, 520)]:
            snap[f"{model}/{seed}/{n}"] = eng.predict(f"S{seed}", _ohlc(seed, n))
        snap[f"{model}/none"] = eng.predict("NONE", None)
        bad = _ohlc(9, 500).drop(columns=["Close"])
        snap[f"{model}/bad"] = {k: v for k, v in eng.predict("BAD", bad).items() if k != "error"}
        snap[f"{model}/bad_err"] = "error" in eng.predict("BAD", bad)
    snap = json.loads(json.dumps(snap))
    if not GOLDEN.exists() and os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(snap, indent=1, sort_keys=True))
    assert snap == json.loads(GOLDEN.read_text())
