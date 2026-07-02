# Tests for F-01 (flows/VIX regime) and F-13 (watchlist delta) — pure logic, no network.
import pandas as pd

import flows_regime as fr
import watchlist_delta as wd


def test_regime_risk_on_off_neutral():
    assert fr.regime({"vix": 11, "fii_net": 2500, "dii_net": 500})["regime"] == "Risk-On"
    assert fr.regime({"vix": 26, "fii_net": -3000})["regime"] == "Risk-Off"
    assert fr.regime({"vix": 17, "fii_net": 0})["regime"] == "Neutral"
    assert fr.regime({})["regime"] == "Neutral"  # no data → neutral


def test_regime_bias_range(monkeypatch):
    monkeypatch.setattr(fr, "_load", lambda: {"vix": 12, "fii_net": 2000})
    b = fr.regime_bias()
    assert -1 <= b <= 1 and b > 0  # risk-on → positive tilt


def test_watchlist_diff(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "SNAP", tmp_path)
    pd.DataFrame(
        {"Symbol": ["A", "B", "C"], "Signal": ["HOLD", "BUY", "HOLD"], "Flags": ["", "", ""]}
    ).to_parquet(tmp_path / "IN_2026-06-29.parquet", index=False)
    pd.DataFrame(
        {
            "Symbol": ["A", "B", "C"],
            "Signal": ["BUY", "SELL", "HOLD"],
            "Flags": ["", "", "IMPAIRED"],
        }
    ).to_parquet(tmp_path / "IN_2026-06-30.parquet", index=False)
    d, s = wd.diff("IN")
    changes = dict(zip(d["Symbol"], d["Change"]))
    assert changes["A"] == "UPGRADE"  # HOLD→BUY
    assert changes["B"] == "DOWNGRADE"  # BUY→SELL
    assert changes["C"] == "NEW-FLAG"  # same signal, new IMPAIRED flag
    assert s["new_buys"] == 1 and s["downgrades"] == 1 and s["new_flags"] == 1


def test_diff_needs_two_snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "SNAP", tmp_path)
    d, s = wd.diff("IN")
    assert d.empty and "note" in s
