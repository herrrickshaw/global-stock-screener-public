# Tests for F-11 portfolio analytics + F-06 total return — pure logic, no network.
import portfolio_analytics as pa
import total_return as tr


def test_weights_and_hhi():
    assert pa._weights(["A", "B"]) == {"A": 0.5, "B": 0.5}
    assert pa._weights({"A": 3, "B": 1}) == {"A": 0.75, "B": 0.25}
    assert pa._hhi({"A": 1.0}) == 1.0
    assert pa._hhi({"A": 0.5, "B": 0.5}) == 0.5


def test_rebalance_actions():
    rows = [
        {"symbol": "X", "weight": 0.2, "signal": "BUY", "flags": "", "sector": "Tech"},
        {"symbol": "W", "weight": 0.25, "signal": "BUY", "flags": "", "sector": "Tech"},
        {"symbol": "Y", "weight": 0.05, "signal": "SELL", "flags": "IMPAIRED", "sector": "Bank"},
    ]
    acts = " | ".join(pa.rebalance_actions(rows))
    assert "TRIM/EXIT Y" in acts  # SELL + IMPAIRED
    assert "REDUCE X" in acts  # 20% > 10% single-name cap
    assert "REDUCE sector Tech" in acts  # Tech 45% > 30% sector cap


def test_total_return_reinvests_dividends():
    # flat price 100→100 but a 5 dividend at t1 → total return 5%, price return 0%
    assert tr.price_return_from([100, 100]) == 0.0
    assert tr.total_return_from([100, 100], [0, 5]) == 5.0
    # pure appreciation 100→110 → both 10% (no divs)
    assert tr.total_return_from([100, 110]) == 10.0
    assert tr.total_return_from([100], None) == 0.0  # too short
