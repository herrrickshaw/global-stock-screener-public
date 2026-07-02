# Tests for F-14 consistency, F-18 smart holdings, F-15 results calendar — pure logic.
import datetime as dt

import consistency as cons
import results_calendar as rc
import smart_holdings as sh


def test_consistency_metrics():
    assert cons.growth_consistency([1, 2, 3, 4]) == 1.0  # always up
    assert cons.growth_consistency([4, 3, 2, 1]) == 0.0  # always down
    assert cons.positive_years([1, -1, 2, 3]) == 0.75
    assert cons.cagr([100, 121]) == 21.0
    s = cons.compounder_score([10, 12, 15, 18, 22])
    assert 0 <= s["score"] <= 100 and s["growth_consistency"] == 1.0


def test_bulk_deal_net_and_screen():
    deals = [
        {"symbol": "AAA", "buySell": "BUY", "quantity": 1000},
        {"symbol": "AAA", "buySell": "SELL", "quantity": 400},
        {"symbol": "BBB", "buySell": "SELL", "quantity": 500},
    ]
    net = sh.net_buy_by_symbol(deals)
    assert net["AAA"] == 600 and net["BBB"] == -500
    assert sh.conviction_screen(deals) == ["AAA"]  # only net buyer


def test_calendar_date_logic():
    today = dt.date(2026, 7, 1)
    assert rc.days_to_event("2026-07-06", today) == 5
    assert rc.near_event("2026-07-04", window=5, today=today) is True
    assert rc.near_event("2026-07-20", window=5, today=today) is False
    ev = {"X": "2026-07-05", "Y": "2026-07-20", "Z": "2026-06-30"}
    assert set(rc.upcoming(ev, within=7, today=today)) == {"X"}  # Y too far, Z past
