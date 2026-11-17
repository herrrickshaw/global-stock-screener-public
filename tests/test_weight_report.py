"""Characterisation of backtest_weight_optimization generate_report console output."""

from __future__ import annotations

import json
import os
from pathlib import Path

import backtest_weight_optimization as bwo

GOLDEN = Path(__file__).parent / "fixtures" / "golden_weight_report.json"


def _m(scale, ret):
    return {
        "n_signals": 1000 * scale,
        "n_outperform": 1500,
        "tp": 600 * scale,
        "fp": 400 * scale,
        "fn": 900,
        "tn": 5000,
        "precision": 0.4 * scale / 2 + 0.1,
        "recall": 0.3 + 0.05 * scale,
        "f1": 0.33 + 0.02 * scale * ret,
        "accuracy": 0.7,
        "avg_buy_return": 3.1 * ret,
        "avg_sell_return": -1.2,
        "return_spread": 4.3 * ret,
    }


def _cases():
    return {
        "baseline_only": {"baseline": _m(1, 1)},
        "optimized_only": {"optimized": _m(2, 1)},
        "significant": {"baseline": _m(1, 1), "optimized": _m(2, 4)},
        "modest": {"baseline": _m(1, 1), "optimized": {**_m(1, 1), "f1": _m(1, 1)["f1"] + 0.01}},
        "none": {"baseline": _m(2, 1), "optimized": _m(1, 1)},
        "tie": {"baseline": _m(1, 1), "optimized": _m(1, 1)},
        "empty": {},
    }


def test_generate_report_matches_golden(capsys):
    cls = next(
        v for v in vars(bwo).values() if isinstance(v, type) and hasattr(v, "generate_report")
    )
    obj = cls.__new__(cls)
    obj.data = list(range(10000))
    snap = {}
    for name, res in _cases().items():
        obj.generate_report(res)
        snap[name] = capsys.readouterr().out
    if not GOLDEN.exists() and os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(snap, indent=1, sort_keys=True))
    assert snap == json.loads(GOLDEN.read_text())
