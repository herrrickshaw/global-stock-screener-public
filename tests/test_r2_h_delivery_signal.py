import datetime as dt
import sys
import types

import numpy as np
import pandas as pd
import pytest

import delivery_signal as ds


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "CACHE", tmp_path)
    return tmp_path


class _Resp:
    def __init__(self, status=200, content=b""):
        self.status_code = status
        self.content = content


def _requests(monkeypatch, fn):
    m = types.ModuleType("requests")
    m.get = fn
    monkeypatch.setitem(sys.modules, "requests", m)


CSV = (
    b"SYMBOL, SERIES, DELIV_PER, TTL_TRD_QNTY\n"
    b"AAA, EQ, 55.5, 100\n"
    b"BBB, BE, 40, 50\n"
    b"CCC, EQ, -, 10\n"
    b"DDD,  EQ , 70, 20\n"
)


def test_fetch_day_parses_eq_and_drops_nan(monkeypatch):
    urls = []

    def get(url, headers, timeout):
        urls.append((url, timeout))
        return _Resp(200, CSV)

    _requests(monkeypatch, get)
    out = ds._fetch_day(dt.date(2024, 3, 5))
    assert urls[0][0].endswith("sec_bhavdata_full_05032024.csv")
    assert urls[0][1] == 25
    assert out["Symbol"].tolist() == ["AAA", "DDD"]
    assert out["DelivPct"].tolist() == [55.5, 70.0]
    assert out["Volume"].tolist() == [100, 20]
    assert (out["Date"] == pd.Timestamp("2024-03-05")).all()


def test_fetch_day_failures(monkeypatch):
    _requests(monkeypatch, lambda *a, **k: _Resp(404, b"x"))
    assert ds._fetch_day(dt.date(2024, 3, 5)) is None
    _requests(monkeypatch, lambda *a, **k: _Resp(200, b""))
    assert ds._fetch_day(dt.date(2024, 3, 5)) is None

    def boom(*a, **k):
        raise OSError

    _requests(monkeypatch, boom)
    assert ds._fetch_day(dt.date(2024, 3, 5)) is None
    _requests(monkeypatch, lambda *a, **k: _Resp(200, b"A,B\n1,2\n"))
    assert ds._fetch_day(dt.date(2024, 3, 5)) is None  # missing columns -> exception swallowed


class _FixedDate(dt.date):
    @classmethod
    def today(cls):
        return cls(2024, 3, 6)  # Wednesday


def test_fetch_skips_weekends_and_known(cache, monkeypatch, capsys):
    monkeypatch.setattr(ds._dt, "date", _FixedDate)
    seen = []

    def fd(day):
        seen.append(day)
        if day == dt.date(2024, 3, 5):
            return None
        if day == dt.date(2024, 3, 1):
            return pd.DataFrame()
        return pd.DataFrame(
            {"Date": [pd.Timestamp(day)], "Symbol": ["A"], "DelivPct": [50.0], "Volume": [1.0]}
        )

    monkeypatch.setattr(ds, "_fetch_day", fd)
    got = ds.fetch(days=7)
    # days: 6(Wed),5(Tue),4(Mon),3(Sun),2(Sat),1(Fri),29 Feb(Thu)
    assert [d.day for d in seen] == [6, 5, 4, 1, 29]
    assert got == 3
    saved = pd.read_parquet(cache / "IN.parquet")
    assert len(saved) == 3
    assert "+3 new day(s), IN.parquet now 3 rows" in capsys.readouterr().out
    # second run: known dates skipped
    seen.clear()
    got = ds.fetch(days=7, verbose=False)
    assert [d.day for d in seen] == [5, 1]
    assert got == 0
    assert capsys.readouterr().out == ""
    assert len(pd.read_parquet(cache / "IN.parquet")) == 3


def test_fetch_no_data_writes_nothing(cache, monkeypatch):
    monkeypatch.setattr(ds._dt, "date", _FixedDate)
    monkeypatch.setattr(ds, "_fetch_day", lambda d: None)
    assert ds.fetch(days=3, verbose=False) == 0
    assert not (cache / "IN.parquet").exists()


def _hist(vals, sym="A"):
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=len(vals)),
            "Symbol": sym,
            "DelivPct": vals,
        }
    )


def test_scores_signal_thresholds():
    # base 20 @ 40 then recent 5 @ x
    def sig(x):
        r = ds.accumulation_scores(_hist([40.0] * 20 + [x] * 5))
        return r.iloc[0]

    assert sig(46.0)["Signal"] == "ACCUMULATION"  # ratio 1.15 exactly
    assert sig(45.9)["Signal"] == "NEUTRAL"
    assert sig(32.0)["Signal"] == "DISTRIBUTION"  # 0.8 exactly
    assert sig(32.1)["Signal"] == "NEUTRAL"
    r = sig(46.0)
    assert r["DelivRecent"] == 46.0 and r["DelivBase"] == 40.0 and r["AccumScore"] == 1.15


def test_strong_accumulation():
    r = ds.accumulation_scores(_hist([50.0] * 20 + [60.0] * 5)).iloc[0]
    assert r["Signal"] == "STRONG ACCUMULATION"
    r = ds.accumulation_scores(_hist([50.0] * 20 + [59.9] * 5)).iloc[0]
    assert r["Signal"] == "ACCUMULATION"
    r = ds.accumulation_scores(_hist([60.0] * 20 + [65.9] * 5)).iloc[0]  # ratio < 1.1
    assert r["Signal"] == "NEUTRAL"
    r = ds.accumulation_scores(_hist([60.0] * 20 + [66.0] * 5)).iloc[0]  # ratio == 1.1
    assert r["Signal"] == "STRONG ACCUMULATION"


def test_scores_min_history_and_short_base():
    assert ds.accumulation_scores(_hist([50.0] * 7)).empty  # < recent + 3
    out = ds.accumulation_scores(_hist([40.0] * 3 + [60.0] * 5))  # 8 rows -> base = first 3
    assert len(out) == 1
    assert out.iloc[0]["DelivBase"] == 40.0


def test_scores_invalid_base_skipped_and_nan():
    assert ds.accumulation_scores(_hist([0.0] * 20 + [50.0] * 5)).empty
    assert ds.accumulation_scores(_hist([-5.0] * 20 + [50.0] * 5)).empty
    out = ds.accumulation_scores(_hist([np.nan, 40.0] * 10 + [50.0] * 5))
    assert out.iloc[0]["DelivBase"] == 40.0
    assert ds.accumulation_scores(_hist([np.nan] * 25)).empty


def test_scores_sorted_and_multi_symbol_and_params():
    df = pd.concat(
        [_hist([40.0] * 20 + [44.0] * 5, "LOW"), _hist([40.0] * 20 + [60.0] * 5, "HIGH")]
    )
    out = ds.accumulation_scores(df)
    assert out["Symbol"].tolist() == ["HIGH", "LOW"]
    out = ds.accumulation_scores(df, recent=2, base=3)
    assert len(out) == 2
    # unsorted dates are sorted per symbol
    shuffled = df.sample(frac=1, random_state=1)
    assert ds.accumulation_scores(shuffled)["Symbol"].tolist() == ["HIGH", "LOW"]


def test_scores_empty_columns():
    out = ds.accumulation_scores(pd.DataFrame())
    assert list(out.columns) == ["Symbol", "DelivRecent", "DelivBase", "AccumScore", "Signal"]


def test_delivery_map_and_report(cache, capsys):
    assert ds.delivery_map() == {}
    assert ds.report().empty
    assert "no delivery cache" in capsys.readouterr().out
    assert ds.report(verbose=False).empty
    assert capsys.readouterr().out == ""
    df = pd.concat(
        [
            _hist([40.0] * 20 + [60.0] * 5, "UP"),
            _hist([50.0] * 20 + [30.0] * 5, "DOWN"),
            _hist([40.0] * 20 + [40.0] * 5, "FLAT"),
        ]
    )
    df.to_parquet(cache / "IN.parquet", index=False)
    assert ds.delivery_map() == {"UP": 60.0, "DOWN": 30.0, "FLAT": 40.0}
    sc = ds.report(top=1)
    out = capsys.readouterr().out
    assert "UP" in out
    assert "DOWN" not in out.split("1 accumulating")[0]  # top=1 trims to best
    assert "1 accumulating, 1 distributing" in out
    assert len(sc) == 3
    ds.report(verbose=False)
    assert capsys.readouterr().out == ""


def test_delivery_map_empty_scores(cache):
    _hist([40.0] * 3).to_parquet(cache / "IN.parquet", index=False)
    assert ds.delivery_map() == {}
    assert ds.report(verbose=False).empty


def test_main(monkeypatch):
    calls = []
    monkeypatch.setattr(ds, "fetch", lambda d: calls.append(("f", d)))
    monkeypatch.setattr(ds, "report", lambda t: calls.append(("r", t)))
    monkeypatch.setattr(sys, "argv", ["ds"])
    assert ds.main() == 0
    assert calls == [("r", 20)]
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["ds", "--fetch", "--days", "9", "--top", "4"])
    assert ds.main() == 0
    assert calls == [("f", 9), ("r", 4)]
