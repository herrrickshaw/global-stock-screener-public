# Tests for F-07 factor regression, F-05 deepen, F-17 API routes — pure logic.
import numpy as np

import api
import factor_analysis as fa


def test_ols_recovers_betas():
    rng = np.random.default_rng(1)
    mkt = rng.normal(1, 3, 200)
    smb = rng.normal(0, 1, 200)
    y = 0.5 + 1.2 * mkt - 0.4 * smb + rng.normal(0, 0.2, 200)
    r = fa.ols(list(y), {"Mkt-RF": list(mkt), "SMB": list(smb)})
    assert abs(r["alpha"] - 0.5) < 0.1
    assert abs(r["betas"]["Mkt-RF"] - 1.2) < 0.05
    assert abs(r["betas"]["SMB"] + 0.4) < 0.05
    assert r["r2"] > 0.95


def test_analyze_returns_needs_min_obs():
    assert fa.analyze_returns([1, 2], [{"Mkt-RF": 1, "SMB": 0, "HML": 0}] * 2) == {}


def test_api_routes_importable():
    assert set(["health", "screen", "decide", "brief"]) <= set(api.ROUTES)
    h = api.health()
    assert "status" in h  # runs without FastAPI installed
