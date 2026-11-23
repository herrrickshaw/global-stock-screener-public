import numpy as np
import pandas as pd
import pytest

import factor_analysis as fa


def test_ols_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    m, s = rng.normal(0, 1, 200), rng.normal(0, 1, 200)
    y = 0.5 + 1.2 * m - 0.7 * s
    r = fa.ols(list(y), {"Mkt-RF": list(m), "SMB": list(s)})
    assert r["alpha"] == 0.5
    assert r["betas"] == {"Mkt-RF": 1.2, "SMB": -0.7}
    assert r["r2"] == 1.0 and r["n"] == 200


def test_ols_rounding():
    x = [0.0, 1.0, 2.0, 3.0]
    y = [0.12345, 1.0, 2.0, 3.1234]
    r = fa.ols(y, {"f": x})
    assert r["alpha"] == round(r["alpha"], 4)
    assert r["betas"]["f"] == round(r["betas"]["f"], 3)
    assert r["r2"] == round(r["r2"], 3)


def test_ols_constant_y_has_zero_r2_denominator_guard():
    r = fa.ols([2.0] * 5, {"f": [1.0, 2.0, 3.0, 4.0, 5.0]})
    assert r["alpha"] == 2.0 and r["betas"]["f"] == 0.0
    assert r["r2"] == 1.0  # ss_res 0 / ss_tot guarded to 1 -> r2 = 1


def test_ols_noise_lowers_r2():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 300)
    y = x + rng.normal(0, 3, 300)
    assert fa.ols(list(y), {"f": list(x)})["r2"] < 0.3


def test_analyze_returns_too_short():
    rows = [{"Mkt-RF": 1.0, "SMB": 0.0, "HML": 0.0}] * 10
    assert fa.analyze_returns([1.0] * 5, rows) == {}
    assert fa.analyze_returns([1.0] * 10, rows[:5]) == {}


def test_analyze_returns_boundary_and_alignment():
    rng = np.random.default_rng(2)
    n = 12
    rows = [
        {"Mkt-RF": float(a), "SMB": float(b), "HML": float(c)}
        for a, b, c in rng.normal(0, 1, (n, 3))
    ]
    y = [0.3 + 1.0 * r["Mkt-RF"] + 0.5 * r["HML"] for r in rows]
    r = fa.analyze_returns(y, rows)
    assert r["n"] == 12 and r["alpha"] == 0.3
    assert r["betas"] == {"Mkt-RF": 1.0, "SMB": 0.0, "HML": 0.5}
    # exactly 6 pairs is the minimum; the TAIL of the longer series is used
    r6 = fa.analyze_returns([99.0] * 4 + y[-6:], rows[-6:])
    assert r6["n"] == 6 and r6["alpha"] == pytest.approx(0.3, abs=1e-3)
    assert fa.analyze_returns(y[:5], rows[:5]) == {}


def test_analyze_returns_uses_last_n_of_each():
    rows = [{"Mkt-RF": float(i), "SMB": float(i % 3), "HML": float((i * 7) % 5)} for i in range(20)]
    y = [1.0 + 2.0 * r["Mkt-RF"] for r in rows]
    out = fa.analyze_returns(y + [1000.0, 1000.0], rows)  # y longer by 2 -> last 20 of y
    assert out["n"] == 20
    out2 = fa.analyze_returns(y, rows + rows[:3])
    assert out2["n"] == 20


def test_load_ff3(tmp_path, monkeypatch):
    p = tmp_path / "ff3.parquet"
    pd.DataFrame(
        {
            "Mkt-RF": ["1.5", "x"],
            "SMB": [0.1, 0.2],
            "HML": [0.3, 0.4],
            "RF": ["0.01", "0.02"],
            "extra": ["a", "b"],
        }
    ).to_parquet(p)
    monkeypatch.setattr(fa, "FF3", p)
    d = fa._load_ff3()
    assert d["Mkt-RF"].iloc[0] == 1.5 and np.isnan(d["Mkt-RF"].iloc[1])
    assert d["RF"].dtype.kind == "f" and d["extra"].iloc[0] == "a"


def test_main_demo(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["x", "--demo"])
    assert fa.main() == 0
    out = capsys.readouterr().out
    assert "'betas'" in out and "'Mkt-RF'" in out and "'n': 60" in out
    # demo is seeded: market beta ~1.1
    import ast

    d = ast.literal_eval(out.strip())
    assert d["betas"]["Mkt-RF"] == pytest.approx(1.1, abs=0.1)
    assert d["alpha"] == pytest.approx(0.4, abs=0.2)


def test_main_default(tmp_path, monkeypatch, capsys):
    p = tmp_path / "ff3.parquet"
    pd.DataFrame(
        {"Mkt-RF": [1.0, 2.0, 3.0], "SMB": [0.0] * 3, "HML": [0.0] * 3, "RF": [0.0] * 3}
    ).to_parquet(p)
    monkeypatch.setattr(fa, "FF3", p)
    monkeypatch.setattr("sys.argv", ["x"])
    assert fa.main() == 0
    assert "FF3 rows: 3" in capsys.readouterr().out
