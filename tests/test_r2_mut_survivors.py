"""Boundary tests added after the first mutation round (each pins a previously surviving mutant)."""

import numpy as np
import pandas as pd
import pytest

import backtest_piotroski_in as pin
import backtest_weight_optimization as wopt
import fundamental_metrics as fm
import reward_liquidity_check as L
import screener_in_auth as sa
from tests.r2b_support import load_with_fake_sklearn, synthetic_ohlc
from tests.test_r2_a_piotroski_in import _s
from tests.test_r2_D_sentiment_pipeline import IN_URL, _rss


# ---- backtest_piotroski_in: current-ratio improvement must be strict (cr > cr1) -----
def test_screener_current_ratio_must_strictly_improve():
    y1 = (10, 100, 8, 50, 90, 10, 50, 10, 20, 10, 100)
    same = (10, 100, 8, 50, 90, 10, 50, 10, 20, 10, 100)  # cur ratio unchanged
    better = (10, 100, 8, 50, 90, 10, 50, 15, 20, 10, 100)  # more inventory -> higher ratio
    f = []
    for y2 in (same, better):
        df = pd.DataFrame([_s("A", "2018-03-31", *y1), _s("A", "2019-03-31", *y2)])
        f.append(pin._piotroski(df)["F"].iloc[0])
    assert f[1] - f[0] == 1.0


# ---- backtest_weight_optimization: empty denominators give 0, not NaN -------------------
def test_evaluate_signals_no_outperformers_gives_zero_recall():
    b = wopt.WeightBacktester.__new__(wopt.WeightBacktester)
    flat = pd.Series([1.0] * 6)  # nothing exceeds the median -> tp + fn == 0
    r = b._evaluate_signals(np.array([1, 0, 1, 0, 1, 0]), flat, "flat")
    assert r["recall"] == 0 and r["f1"] == 0 and r["precision"] == 0
    none = b._evaluate_signals(np.array([0, 0, 0, 0, 0, 0]), pd.Series([1.0, 2, 3, 4, 5, 6]), "n")
    assert none["precision"] == 0 and none["f1"] == 0


# ---- reward_liquidity_check.picks: threshold row is included (<= / >=) -----------------
@pytest.mark.parametrize("direction", ["rising", "falling"])
def test_picks_threshold_row_included(monkeypatch, capsys, direction):
    df = pd.DataFrame(
        {
            "symbol": list("ABCDE"),
            "year": [2024] * 5,
            "dvol": [10.0] * 5,
            "liq_3_12": [0.1, 0.2, 0.3, 0.4, 0.5],  # keep=0.5 -> threshold is exactly 0.3 (C)
        }
    )
    monkeypatch.setattr(L, "_panel", lambda m: df)
    res = {"XX": {"robust": True, "factor": "liq_3_12", "direction": direction, "keep": 0.5}}
    L.picks(["XX"], res, 10)
    out = capsys.readouterr().out
    want = {"rising": "CDE", "falling": "ABC"}[direction]
    got = [s for s in "ABCDE" if f"  {s:<14}" in out]
    assert "".join(got) == want


# ---- ml_signal_engine.evaluate_on_backtest: both window ends are inclusive -------------
def test_evaluate_window_is_inclusive_on_both_ends(monkeypatch):
    mod = load_with_fake_sklearn(
        __import__("pathlib")
        .Path(__file__)
        .resolve()
        .parent.parent.joinpath("ml_signal_engine.py"),
        "mse_r2_mut",
        monkeypatch,
    )
    df = synthetic_ohlc(2, 520)
    idx = mod.compute_features(df).index
    day = str(idx[-30].date())
    r = mod.MLSignalEngine("lr").evaluate_on_backtest({"A": df}, day, day)
    assert r["n_predictions"] == 1


# ---- sentiment_pipeline: 3-letter tokens count towards the company-name phrase ---------
def test_rss_three_letter_name_token_is_kept(tmp_path, monkeypatch):
    feeds = {
        IN_URL: [
            {"title": "Tata Ltd results", "summary": ""},
            {"title": "Tata Motors rises", "summary": ""},
        ]
    }
    p = _rss(tmp_path, monkeypatch, feeds).IndianRSSProvider()
    out = p.fetch_news("TATA", "IN", company_name="Tata Ltd")
    assert [a.title for a in out] == ["Tata Ltd results"]


# ---- fundamental_metrics.FUND_SCREENS: every threshold is strict ------------------------
def _d(**kw):
    return pd.DataFrame({k: [v] for k, v in kw.items()})


@pytest.mark.parametrize(
    ("screen", "edge", "pass_"),
    [
        ("piotroski_9", {"piotroski": 8}, {"piotroski": 7}),
        ("high_roce", {"roce": 20.0}, {"roce": 20.5}),
        ("high_roe", {"roe": 20.0}, {"roe": 20.5}),
        ("debt_free", {"debt_to_equity": 0.1}, {"debt_to_equity": 0.09}),
        ("low_debt_equity", {"debt_to_equity": 0.3}, {"debt_to_equity": 0.29}),
        ("fcf_yield", {"fcf_yield": 5.0}, {"fcf_yield": 5.5}),
        ("high_dividend", {"div_yield": 3.0}, {"div_yield": 3.5}),
        ("eps_growth", {"eps_growth": 15.0}, {"eps_growth": 15.5}),
    ],
)
def test_fund_screen_thresholds(screen, edge, pass_):
    fn = fm.FUND_SCREENS[screen]
    if screen == "piotroski_9":  # >= 8: 8 passes, 7 fails
        assert fn(_d(**edge)).tolist() == [True] and fn(_d(**pass_)).tolist() == [False]
    else:  # strict: the edge value fails, just beyond it passes
        assert fn(_d(**edge)).tolist() == [False] and fn(_d(**pass_)).tolist() == [True]


def test_fund_screen_magic_formula_and_coffee_can_boundaries():
    mf = fm.FUND_SCREENS["magic_formula"]
    assert mf(_d(earnings_yield=8.0, roce=16.0)).tolist() == [False]
    assert mf(_d(earnings_yield=8.5, roce=15.0)).tolist() == [False]
    assert mf(_d(earnings_yield=8.5, roce=15.5)).tolist() == [True]
    cc = fm.FUND_SCREENS["coffee_can"]
    assert cc(_d(roe=15.0, debt_to_equity=0.1)).tolist() == [False]
    assert cc(_d(roe=16.0, debt_to_equity=0.5)).tolist() == [False]
    assert cc(_d(roe=16.0, debt_to_equity=0.49)).tolist() == [True]


# ---- screener_in_auth._load_dotenv: comment lines are skipped even if they contain '=' ---
def test_dotenv_comment_with_equals_is_not_loaded(tmp_path, monkeypatch):
    import os

    monkeypatch.setattr(sa, "__file__", str(tmp_path / "screener_in_auth.py"))
    (tmp_path / ".env").write_text("#R2H_CMT=1\nR2H_REAL=2\n")
    try:
        sa._load_dotenv()
        assert os.environ.get("R2H_REAL") == "2"
        assert "#R2H_CMT" not in os.environ and "R2H_CMT" not in os.environ
    finally:
        os.environ.pop("R2H_REAL", None)
        os.environ.pop("#R2H_CMT", None)
