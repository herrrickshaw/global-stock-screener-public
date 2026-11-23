"""Tests for the small helpers in stock_utils.py: statement extraction, frame hygiene,
yfinance bulk extraction/retry, parallel_map and the numeric helpers.

Everything is synthetic and offline: yfinance is replaced by a fake downloader and
``time.sleep`` is recorded instead of waited on.
"""

import numpy as np
import pandas as pd
import pytest

import stock_utils as su

# ---- statement helpers ------------------------------------------------------------------


def _stmt():
    cols = pd.to_datetime(["2024-03-31", "2023-03-31", "2022-03-31"])
    return pd.DataFrame(
        {cols[0]: [300.0, 30.0], cols[1]: [200.0, np.nan], cols[2]: [100.0, 10.0]},
        index=["Total Revenue", "Net Income"],
    )


class _Ticker:
    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


def test_first_df_skips_none_empty_and_non_frames():
    good = _stmt()
    t = _Ticker(a=None, b=pd.DataFrame(), c="not a frame", d=good, e=_stmt() * 2)
    assert su.first_df(t, "a", "b", "c", "d", "e") is good  # first usable one wins
    assert su.first_df(t, "missing", "a", "b") is None
    assert su.first_df(t) is None


def test_row_reads_by_name_and_column():
    df = _stmt()
    assert su.row(df, "Total Revenue") == 300.0
    assert su.row(df, "Total Revenue", col=2) == 100.0
    assert su.row(df, "Nope", "Total Revenue") == 300.0  # falls through to the next name
    assert isinstance(su.row(df, "Total Revenue"), float)


def test_row_returns_none_for_nan_missing_or_bad_input():
    df = _stmt()
    assert su.row(df, "Net Income", col=1) is None  # NaN value
    assert su.row(df, "Nope") is None
    assert su.row(df, "Total Revenue", col=9) is None  # column out of range is swallowed
    assert su.row(None, "Total Revenue") is None
    assert su.row(pd.DataFrame(), "Total Revenue") is None


def test_series_returns_non_nan_values_in_column_order():
    df = _stmt()
    assert su.series(df, "Total Revenue") == [300.0, 200.0, 100.0]
    assert su.series(df, "Net Income") == [30.0, 10.0]  # the NaN period is dropped
    assert su.series(df, "Nope", "Net Income") == [30.0, 10.0]
    assert su.series(df, "Nope") == []
    assert su.series(None, "Total Revenue") == []
    assert su.series(pd.DataFrame(), "Total Revenue") == []


def test_fin_metric_default_only_applies_when_missing():
    df = _stmt()
    assert su.fin_metric(df, ("Total Revenue",)) == 300.0
    assert su.fin_metric(df, ("Nope",), default=-1.0) == -1.0
    assert su.fin_metric(df, ("Nope",)) is None
    zero = pd.DataFrame({pd.Timestamp("2024-03-31"): [0.0]}, index=["Net Income"])
    assert su.fin_metric(zero, ("Net Income",), default=5.0) == 0.0  # a real 0 is kept


# ---- clean_financials -------------------------------------------------------------------


def test_clean_financials_orders_newest_period_first():
    df = _stmt()
    scrambled = df[[df.columns[2], df.columns[0], df.columns[1]]]
    out = su.clean_financials(scrambled)
    assert list(out.columns) == sorted(df.columns, reverse=True)
    assert su.row(out, "Total Revenue") == 300.0  # what row()/fin_metric() assume for col=0


def test_clean_financials_drops_empty_columns_and_coerces_numbers():
    df = pd.DataFrame(
        {
            pd.Timestamp("2024-03-31"): ["300", "30"],
            pd.Timestamp("2023-03-31"): [np.nan, np.nan],  # duplicate/empty period
        },
        index=["Total Revenue", "Net Income"],
    )
    out = su.clean_financials(df)
    assert list(out.columns) == [pd.Timestamp("2024-03-31")]
    assert out.loc["Total Revenue"].iloc[0] == 300.0


def test_clean_financials_none_when_nothing_usable():
    assert su.clean_financials(None) is None
    assert su.clean_financials(pd.DataFrame()) is None
    junk = pd.DataFrame({"x": ["a", "b"]}, index=["r1", "r2"])
    assert su.clean_financials(junk) is None  # everything coerces to NaN


def test_clean_financials_tolerates_unsortable_columns():
    df = pd.DataFrame([[1.0, 2.0]], columns=["FY24", 2023], index=["Total Revenue"])
    out = su.clean_financials(df)  # str vs int columns cannot be sorted
    assert list(out.columns) == ["FY24", 2023]


# ---- extract_ticker_df ------------------------------------------------------------------


def _bars(n=40, start=100.0):
    closes = start + np.arange(n, dtype=float)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes * 1.01,
            "Low": closes * 0.99,
            "Close": closes,
            "Volume": np.full(n, 100_000),
        },
        index=idx,
    )


def _multi(tickers, n=40):
    frames = {t: _bars(n, start=100.0 + 50 * i) for i, t in enumerate(tickers)}
    return pd.concat(frames, axis=1).swaplevel(axis=1).sort_index(axis=1)


def test_extract_ticker_df_handles_multiindex_and_flat_layouts():
    raw = _multi(["AAA", "BBB"])
    a = su.extract_ticker_df(raw, "AAA")
    b = su.extract_ticker_df(raw, "BBB")
    assert a is not None and b is not None
    assert a["Close"].iloc[0] == 100.0 and b["Close"].iloc[0] == 150.0  # right ticker each time
    flat = su.extract_ticker_df(_bars(), "ANY")  # single-ticker download has flat columns
    assert flat is not None and len(flat) == 40


def test_extract_ticker_df_none_for_unknown_ticker_or_empty_frame():
    raw = _multi(["AAA"])
    assert su.extract_ticker_df(raw, "ZZZ") is None
    assert su.extract_ticker_df(pd.DataFrame(), "AAA") is None
    allnan = _bars().astype(float) * np.nan
    assert su.extract_ticker_df(allnan, "AAA") is None


def test_extract_ticker_df_unclean_path_applies_min_bars_and_keeps_ohlcv():
    df = _bars(10)
    df["Dividends"] = 0.0
    assert su.extract_ticker_df(df, "AAA", min_bars=20, clean=False) is None
    kept = su.extract_ticker_df(df, "AAA", min_bars=5, clean=False)
    assert list(kept.columns) == su.OHLCV_COLS  # extra yfinance columns are dropped
    other = pd.DataFrame({"foo": [1.0, 2.0]})
    assert list(su.extract_ticker_df(other, "AAA", clean=False).columns) == ["foo"]


# ---- bulk_download (fake yfinance) ------------------------------------------------------


class _FakeYF:
    """Stands in for the yfinance module: scripted responses/exceptions per call."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def download(self, batch, **kwargs):
        self.calls.append((list(batch), kwargs))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture
def fake_yf(monkeypatch):
    sleeps = []
    monkeypatch.setattr(su, "_YF_OK", True)
    monkeypatch.setattr(su.time, "sleep", lambda s: sleeps.append(s))

    def install(*responses):
        fy = _FakeYF(*responses)
        monkeypatch.setattr(su, "yf", fy, raising=False)
        return fy, sleeps

    return install


def test_bulk_download_returns_nothing_without_yfinance(monkeypatch):
    monkeypatch.setattr(su, "_YF_OK", False)
    assert su.bulk_download(["AAA.NS"]) == {}


def test_bulk_download_batches_strips_suffix_and_pauses_between_batches(fake_yf):
    fy, sleeps = fake_yf(_multi(["AAA.NS", "BBB.BO"]), _multi(["CCC"]))
    out = su.bulk_download(
        ["AAA.NS", "BBB.BO", "CCC"], batch_size=2, sleep_between=0.5, verbose=False
    )
    assert sorted(out) == ["AAA", "BBB", "CCC"]  # .NS/.BO suffixes stripped from the keys
    assert [c[0] for c in fy.calls] == [["AAA.NS", "BBB.BO"], ["CCC"]]
    assert sleeps == [0.5]  # one pause, between the two batches only
    assert fy.calls[0][1]["period"] == "1y" and "start" not in fy.calls[0][1]


def test_bulk_download_uses_start_instead_of_period(fake_yf):
    fy, _ = fake_yf(_multi(["AAA"]))
    su.bulk_download(["AAA"], start="2024-01-01", period="5y", verbose=False)
    kwargs = fy.calls[0][1]
    assert kwargs["start"] == "2024-01-01" and "period" not in kwargs


def test_bulk_download_skips_short_histories_and_empty_batches(fake_yf):
    fy, _ = fake_yf(_multi(["AAA"], n=10), pd.DataFrame())
    out = su.bulk_download(["AAA", "BBB"], batch_size=1, min_bars=30, verbose=False)
    assert out == {}  # AAA has too few bars; BBB's batch came back empty
    assert len(fy.calls) == 2


def test_bulk_download_retries_rate_limits_with_growing_backoff(fake_yf):
    fy, sleeps = fake_yf(Exception("Too Many Requests"), Exception("429"), _multi(["AAA"]))
    out = su.bulk_download(["AAA"], max_retries=3, sleep_between=0, verbose=False)
    assert list(out) == ["AAA"]
    assert len(fy.calls) == 3
    assert sleeps[:2] == [20, 40]  # 20s then 40s back-off


def test_bulk_download_gives_up_after_max_retries(fake_yf):
    fy, sleeps = fake_yf(Exception("Rate limited"), Exception("Rate limited"))
    out = su.bulk_download(["AAA"], max_retries=2, sleep_between=0, verbose=False)
    assert out == {}
    assert len(fy.calls) == 2 and sleeps == [20]  # waited once; the last attempt just fails


def test_bulk_download_does_not_retry_other_errors(fake_yf):
    fy, sleeps = fake_yf(ValueError("boom"))
    out = su.bulk_download(["AAA"], max_retries=3, sleep_between=0, verbose=False)
    assert out == {} and len(fy.calls) == 1 and sleeps == []


def test_bulk_download_verbose_reports_progress(fake_yf, capsys):
    fake_yf(_multi(["AAA"]))
    su.bulk_download(["AAA"], sleep_between=0)
    out = capsys.readouterr().out
    assert "bulk_download: 1 tickers" in out and "complete: 1/1 loaded" in out


# ---- small helpers ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "msg,expected",
    [
        ("Rate limited. Try after a while", True),
        ("HTTP Error 429", True),
        ("Too Many Requests", True),
        ("Invalid Crumb", True),
        ("connection reset by peer", False),
    ],
)
def test_is_rate_limited(msg, expected):
    assert su.is_rate_limited(Exception(msg)) is expected


@pytest.mark.parametrize(
    "ticker,expected",
    [("RELIANCE.NS", "RELIANCE"), ("TCS.BO", "TCS"), ("AAPL", "AAPL"), ("7203.T", "7203.T")],
)
def test_strip_suffix_only_removes_indian_suffixes(ticker, expected):
    assert su._strip_suffix(ticker) == expected


def test_parallel_map_collects_non_none_and_ignores_failures(capsys):
    def work(x):
        if x == 3:
            raise RuntimeError("bad item")
        return None if x % 2 == 0 else x * 10

    out = su.parallel_map(work, [1, 2, 3, 4, 5], workers=2, progress_every=2, label="things")
    assert sorted(out) == [10, 50]  # evens -> None, 3 -> exception: both dropped
    printed = capsys.readouterr().out
    assert "things processed" in printed and "5/5" in printed
    assert su.parallel_map(work, [1], verbose=False) == [10]
    assert su.parallel_map(work, [], verbose=False) == []


def test_cagr_matches_hand_calculation_and_rejects_bad_inputs():
    assert su.cagr(200.0, 100.0, 1) == pytest.approx(100.0)
    assert su.cagr(121.0, 100.0, 2) == pytest.approx(10.0)  # 1.21 ** 0.5 - 1
    assert su.cagr(100.0, 200.0, 1) == pytest.approx(-50.0)  # shrinking is allowed
    assert su.cagr(100.0, 0.0, 3) is None
    assert su.cagr(100.0, -5.0, 3) is None
    assert su.cagr(100.0, None, 3) is None
    assert su.cagr(200.0, 100.0, 0) is None


def test_normalise_debt_to_equity_converts_percent_style_values():
    assert su.normalise_debt_to_equity(None) is None
    assert su.normalise_debt_to_equity(45.2) == pytest.approx(0.452)  # yfinance percent style
    assert su.normalise_debt_to_equity(0.8) == 0.8  # already a ratio
    assert su.normalise_debt_to_equity(10) == 10  # threshold is strictly > 10


def test_market_cap_crores():
    assert su.market_cap_crores(1e9) == pytest.approx(100.0)  # 1e9 / 1e7
    assert su.market_cap_crores(None) == 0.0
    assert su.market_cap_crores(0) == 0.0
