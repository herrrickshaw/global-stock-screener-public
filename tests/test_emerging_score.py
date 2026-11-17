"""Characterisation of emerging_fundamentals.score_from_statements over seeded random inputs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import emerging_fundamentals as ef

GOLDEN = Path(__file__).parent / "fixtures" / "golden_emerging_score.json"


def _maybe(rng, v, p_none=0.12):
    return None if rng.random() < p_none else float(v)


def _cases():
    rng = np.random.default_rng(7)
    cases = [
        ("no_inc", [], [{"total_assets": 1}], [], {}),
        ("no_bal", [{"revenue": 1}], [], [], None),
        ("pe_zero", [{"revenue": 1}], [{"total_assets": 1}], [], {"pe": 0, "sector": "X"}),
        ("pe_str", [{"revenue": 1}], [{"total_assets": 1}], [], {"pe": "12.345"}),
    ]
    for k in range(60):
        n = int(rng.integers(1, 5))
        inc = [
            {
                "net_income": _maybe(rng, rng.normal(50, 60)),
                "revenue": _maybe(rng, rng.uniform(500, 2000)),
                "gross_profit": _maybe(rng, rng.uniform(100, 700)),
                "ebit": _maybe(rng, rng.uniform(20, 400)),
                "shares": _maybe(rng, rng.uniform(90, 110)),
            }
            for _ in range(n)
        ]
        bal = [
            {
                "total_assets": _maybe(rng, rng.uniform(500, 3000)),
                "long_term_debt": _maybe(rng, rng.uniform(0, 800)),
                "current_assets": _maybe(rng, rng.uniform(100, 900)),
                "current_liabilities": _maybe(rng, rng.uniform(100, 700)),
            }
            for _ in range(int(rng.integers(1, 5)))
        ]
        cf = [
            {
                "operating_cash_flow": _maybe(rng, rng.normal(80, 60)),
                "free_cash_flow": _maybe(rng, rng.normal(40, 60), 0.5),
                "capex": _maybe(rng, -rng.uniform(5, 80)),
            }
        ][: int(rng.integers(0, 2))]
        profile = {"pe": _maybe(rng, rng.uniform(-5, 40)), "sector": f"S{k % 3}"}
        cases.append((f"r{k}", inc, bal, cf, profile))
    return cases


def test_score_from_statements_matches_golden():
    snap = {n: ef.score_from_statements(i, b, c, p) for n, i, b, c, p in _cases()}
    snap = json.loads(json.dumps(snap))
    if not GOLDEN.exists() and os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(snap, indent=1, sort_keys=True))
    assert snap == json.loads(GOLDEN.read_text())
    assert len({v["f_score"] for v in snap.values()}) > 3
