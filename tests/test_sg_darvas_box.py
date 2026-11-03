"""Characterisation of sg_stock_daily_report.compute_darvas_box on synthetic OHLC.

Pins the owner's design rule: the box is formed from history only (current bar excluded).
Golden captured from the pre-pass-2 code in tests/fixtures/golden_sg_darvas.json.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GOLDEN = Path(__file__).parent / "fixtures" / "golden_sg_darvas.json"


def _frames() -> dict:
    out = {"none": None, "empty": pd.DataFrame(), "short": pd.DataFrame({"High": [1] * 5, "Low": [1] * 5, "Close": [1] * 5})}
    out["no_cols"] = pd.DataFrame({"A": range(30)})
    out["alt_cols"] = pd.DataFrame(
        {"DAYHIGH": np.linspace(10, 12, 30), "DAYLOW": np.linspace(9, 11, 30), "LAST": np.linspace(9.5, 11.5, 30)}
    )
    for seed in range(24):
        rng = np.random.default_rng(seed)
        n = 15 + seed * 3
        close = 50 + np.cumsum(rng.normal(0, 1, n))
        high = close + rng.uniform(0.1, 1.0, n)
        low = close - rng.uniform(0.1, 1.0, n)
        if seed % 4 == 1:  # breakout on the last bar
            close[-1] = high[:-1].max() + 3
            high[-1] = close[-1] + 0.5
        elif seed % 4 == 2:  # breakdown on the last bar
            close[-1] = low[:-1].min() - 3
            low[-1] = close[-1] - 0.5
        elif seed % 4 == 3:  # zeros / NaNs in the data
            high[3] = 0
            low[5] = np.nan
            close[7] = np.nan
        out[f"rw{seed}"] = pd.DataFrame({"High": high, "Low": low, "Close": close})
    flat = pd.DataFrame({"High": [10.0] * 20, "Low": [9.0] * 20, "Close": [9.5] * 20})
    out["flat"] = flat
    return out


def test_compute_darvas_box_matches_golden(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("sg_stock_daily_report", None)
    mod = importlib.import_module("sg_stock_daily_report")
    snap = {k: mod.compute_darvas_box(v) for k, v in _frames().items()}
    snap["confirm5"] = mod.compute_darvas_box(_frames()["rw20"], confirm=5)
    snap = json.loads(json.dumps(snap, default=float))
    if not GOLDEN.exists() and os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(snap, indent=1, sort_keys=True))
    assert snap == json.loads(GOLDEN.read_text())
    assert len({v["signal"] for v in snap.values()}) >= 4
