# Tests for the portfolio evaluator — pure logic, no network.
import pandas as pd


def test_parse_tickers():
    import portfolio_evaluator as pe

    assert pe.parse_tickers("reliance, tcs\ninfy;wipro") == ["RELIANCE", "TCS", "INFY", "WIPRO"]
    assert pe.parse_tickers("AAPL AAPL MSFT") == ["AAPL", "MSFT"]  # dedup


def test_clean_syms_strips_suffix_and_isin():
    import portfolio_evaluator as pe

    got = pe._clean_syms(pd.Series(["RELIANCE.NS", "TCS.BO", "INE002A01018", "  infy  "]))
    assert "RELIANCE" in got and "TCS" in got and "INFY" in got
    assert "INE002A01018" in got  # ISIN kept as-is token (still a valid identifier)


def test_ticker_column_autodetect():
    import portfolio_evaluator as pe

    df = pd.DataFrame({"Qty": [1], "Scrip Name": ["TCS"], "P&L": [10]})
    assert pe._ticker_col(df) == "Scrip Name"
    assert pe._ticker_col(pd.DataFrame({"foo": [1]})) is None


def test_decision_logic_by_pe():
    import portfolio_evaluator as pe

    med = 20.0
    assert pe._decide(12, med, 2, True)[0] == "BUY"  # cheap + uptrend
    assert pe._decide(40, med, 0, True)[0] == "SELL"  # 40 > 1.5*20
    assert pe._decide(-5, med, 0, False)[0] == "SELL"  # loss-making downtrend
    assert pe._decide(20, med, 1, True)[0] == "HOLD"  # fair value
    assert pe._decide(None, med, 0, True)[0] == "HOLD"  # no PE, uptrend
    assert pe._decide(None, med, 0, False)[0] == "SELL"  # no PE, downtrend
