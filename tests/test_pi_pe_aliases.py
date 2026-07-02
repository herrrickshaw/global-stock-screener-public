# Tests for F-03 (P/E clamp) and F-02 (ticker aliases) — pure logic, no network.
import portfolio_evaluator as pe
import ticker_aliases as ta


def test_pe_clamp_discards_absurd():
    # the ABFRL glitch: near-zero EPS → P/E in the hundreds of millions → discarded
    assert pe._pe_of({"pe": 836202011.6}, None) is None
    assert pe._pe_of({"pe": 21.9}, None) == 21.9  # normal kept
    assert pe._pe_of({"pe": -5.0}, None) == -5.0  # loss-making kept (engine flags)
    assert pe._sane_pe(1500.0) is None and pe._sane_pe(950.0) == 950.0


def test_pe_from_eps_clamped():
    # price 500 / eps 0.0005 = 1,000,000 → absurd → None
    assert pe._pe_of({"eps_ttm": 0.0005}, 500) is None
    assert pe._pe_of({"eps_ttm": 25.0}, 500) == 20.0  # 500/25


def test_aliases_known_renames():
    assert ta.canonical("ZOMATO") == "ETERNAL"
    assert ta.canonical("PVR") == "PVRINOX"
    assert ta.canonical("HDFC") == "HDFCBANK"
    assert ta.canonical("RELIANCE") == "RELIANCE"  # unchanged


def test_apply_aliases_dedups():
    out = ta.apply_aliases(["SRTRANSFIN", "SHRIRAMCIT", "INFY"])  # both → SHRIRAMFIN
    assert out.count("SHRIRAMFIN") == 1 and "INFY" in out
