# Tests for the delivery-% accumulation scoring — pure logic, no network.
import numpy as np
import pandas as pd

import delivery_signal as ds


def _hist(sym, series):
    n = len(series)
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"Date": dates, "Symbol": sym, "DelivPct": series, "Volume": np.full(n, 1000)}
    )


def test_accumulation_detected():
    # low base (~40%) then rising to high (~70%) → accumulation
    s = ds.accumulation_scores(_hist("ACC", [40] * 20 + [70] * 5))
    row = s[s.Symbol == "ACC"].iloc[0]
    assert row.AccumScore > 1.1
    assert "ACCUMULATION" in row.Signal


def test_distribution_detected():
    s = ds.accumulation_scores(_hist("DIS", [70] * 20 + [45] * 5))
    row = s[s.Symbol == "DIS"].iloc[0]
    assert row.AccumScore <= 0.8
    assert row.Signal == "DISTRIBUTION"


def test_short_history_skipped():
    assert ds.accumulation_scores(_hist("SHORT", [50, 55, 60])).empty


def test_empty_input():
    assert ds.accumulation_scores(pd.DataFrame()).empty
