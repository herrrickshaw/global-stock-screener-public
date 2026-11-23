"""Offline tests for serving_layer: dirs redirected to tmp_path, datalink faked."""

import sys
import types

import numpy as np
import pandas as pd
import pytest

import serving_layer as sl


@pytest.fixture
def sv(tmp_path, monkeypatch):
    monkeypatch.setattr(sl, "SERVING_DIR", tmp_path / "serving")
    monkeypatch.setattr(sl, "CDC_DIR", tmp_path / "cdc")
    sl.SERVING_DIR.mkdir()
    sl.CDC_DIR.mkdir()
    monkeypatch.setattr(sl, "_SERVE_CACHE", {})
    return sl


def bars(n, start=100.0, step=1.0, vol=10.0):
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    c = start + step * np.arange(n)
    return pd.DataFrame({"Close": c, "Volume": [vol] * n}, index=idx)


def test_features_too_short_or_none(sv):
    assert sl._features("A", "IN", None) is None
    assert sl._features("A", "IN", bars(19)) is None
    assert sl._features("A", "IN", bars(20)) is not None


def test_features_values(sv):
    d = bars(300)
    f = sl._features("A", "IN", d)
    c = d["Close"]
    assert f["Symbol"] == "A" and f["Market"] == "IN" and f["Bars"] == 300
    assert f["Close"] == c.iloc[-1]
    assert f["SMA20"] == pytest.approx(c.tail(20).mean())
    assert f["SMA50"] == pytest.approx(c.tail(50).mean())
    assert f["SMA200"] == pytest.approx(c.tail(200).mean())
    assert f["High252"] == c.tail(252).max() and f["Low252"] == c.tail(252).min()
    assert f["PctFromHigh"] == pytest.approx(0.0)
    assert f["PctFromLow"] == pytest.approx((c.iloc[-1] / c.tail(252).min() - 1) * 100)
    assert f["Ret21"] == pytest.approx((c.iloc[-1] / c.iloc[-21] - 1) * 100)
    assert f["Ret252"] == pytest.approx((c.iloc[-1] / c.iloc[-252] - 1) * 100)
    assert f["Above200DMA"] is True and f["GoldenCross"] is True
    assert f["TurnoverLocal"] == pytest.approx((c.tail(20) * 10).mean())
    assert f["LastDate"] == str(d.index[-1])[:10]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="serving_layer.py:~66 RSI divides by dn.replace(0, nan): a series with no "
    "losses yields NaN instead of RSI=100",
)
def test_features_rsi_all_gains_is_100(sv):
    f = sl._features("A", "IN", bars(300))
    assert f["RSI14"] == pytest.approx(100.0)


def test_features_rsi_mixed(sv):
    d = bars(40)
    d["Close"] = [100 + (i % 2) * 2 + i * 0.1 for i in range(40)]
    r = sl._features("A", "IN", d)["RSI14"]
    assert 50 < r < 100


def test_features_short_history_nans_and_falling(sv):
    f = sl._features("A", "IN", bars(30, start=200.0, step=-1.0))
    assert np.isnan(f["SMA50"]) and np.isnan(f["SMA200"])
    assert np.isnan(f["Ret63"]) and np.isnan(f["Ret252"])
    assert not np.isnan(f["Ret21"])
    assert f["Above200DMA"] is False and f["GoldenCross"] is False
    assert f["RSI14"] == pytest.approx(0.0)
    # 14 bars -> rsi needs n>=15 ; n here is >=20 always, rsi defined
    assert not np.isnan(f["RSI14"])


def test_features_ret_boundary_and_zero_close(sv):
    # n == k: ret(k) requires n > k
    f = sl._features("A", "IN", bars(21))
    assert np.isnan(f["Ret21"])
    f = sl._features("A", "IN", bars(22))
    assert not np.isnan(f["Ret21"])
    d = bars(25)
    d.iloc[-21, 0] = 0.0
    assert np.isnan(sl._features("A", "IN", d)["Ret21"])


def test_features_sma_boundary_and_golden_cross(sv):
    f = sl._features("A", "IN", bars(50))
    assert not np.isnan(f["SMA50"]) and np.isnan(f["SMA200"])
    assert f["GoldenCross"] is False  # needs sma200
    assert not np.isnan(sl._features("A", "IN", bars(200))["SMA200"])
    # falling then spike: sma50 > sma200 false when price collapsing
    f = sl._features("A", "IN", bars(300, start=500.0, step=-1.0))
    assert f["GoldenCross"] is False and f["Above200DMA"] is False
    assert f["PctFromLow"] == pytest.approx(0.0)


def test_source_prefers_ltm_then_seed(sv, monkeypatch):
    monkeypatch.setattr(sl.datalink, "load_market", lambda m, tier: {"X": bars(25)})
    assert set(sl._source("IN")) == {"X"}
    monkeypatch.setattr(sl.datalink, "load_market", lambda m, tier: {})
    frames = sys.modules["frames"]
    seen = {}

    def fake_read_seed(p):
        seen["p"] = p
        return {"S": bars(25)}

    monkeypatch.setattr(frames, "read_seed", fake_read_seed)
    assert set(sl._source("IN")) == {"S"}
    assert seen["p"] == sl.datalink.SEED_DIR / sl.datalink._seed_name("IN")


def test_build_serving(sv, monkeypatch, capsys):
    monkeypatch.setattr(sl, "_source", lambda m: {"A": bars(30), "SHORT": bars(5), "B": bars(40)})
    ann = types.ModuleType("liquidity")
    ann.annotate = lambda df: df.assign(Liquidity="High")
    monkeypatch.setitem(sys.modules, "liquidity", ann)
    assert sl.build_serving("IN") == 2
    df = pd.read_parquet(sl.SERVING_DIR / "IN.parquet")
    assert set(df["Symbol"]) == {"A", "B"} and (df["Liquidity"] == "High").all()
    assert "serving[IN]: 2 symbols → IN.parquet" in capsys.readouterr().out
    sl.build_serving("IN", verbose=False)
    assert capsys.readouterr().out == ""


def test_build_serving_annotate_failure_and_empty(sv, monkeypatch):
    monkeypatch.setattr(sl, "_source", lambda m: {"A": bars(30)})
    ann = types.ModuleType("liquidity")

    def boom(df):
        raise RuntimeError("x")

    ann.annotate = boom
    monkeypatch.setitem(sys.modules, "liquidity", ann)
    assert sl.build_serving("US", verbose=False) == 1
    assert "Liquidity" not in pd.read_parquet(sl.SERVING_DIR / "US.parquet").columns
    monkeypatch.setattr(sl, "_source", lambda m: {"A": bars(3)})
    assert sl.build_serving("EU") == 0
    assert not (sl.SERVING_DIR / "EU.parquet").exists()


def _write_serving(market, rows):
    pd.DataFrame(rows).to_parquet(sl.SERVING_DIR / f"{market}.parquet", index=False)


def test_capture_deltas_flow(sv, capsys):
    assert sl.capture_deltas("IN") == 0
    _write_serving(
        "IN",
        [
            {"Symbol": "A", "Close": 1.0, "LastDate": "d1"},
            {"Symbol": "B", "Close": 2.0, "LastDate": "d1"},
        ],
    )
    assert sl.capture_deltas("IN") == 2
    log = pd.read_parquet(sl.CDC_DIR / "IN.parquet")
    assert set(log["op"]) == {"INSERT"} and set(log["key"]) == {"A", "B"}
    assert "+2 deltas → IN.parquet" in capsys.readouterr().out
    # unchanged -> no changes
    assert sl.capture_deltas("IN") == 0
    assert "cdc[IN]: no changes" in capsys.readouterr().out
    assert sl.capture_deltas("IN", verbose=False) == 0
    assert capsys.readouterr().out == ""
    # A changes, B same, C new
    _write_serving(
        "IN",
        [
            {"Symbol": "A", "Close": 1.5, "LastDate": "d2"},
            {"Symbol": "B", "Close": 2.0, "LastDate": "d2"},
            {"Symbol": "C", "Close": 3.0, "LastDate": "d2"},
        ],
    )
    assert sl.capture_deltas("IN") == 2
    log = pd.read_parquet(sl.CDC_DIR / "IN.parquet")
    assert len(log) == 4
    new = log.tail(2).set_index("key")["op"].to_dict()
    assert new == {"A": "UPDATE", "C": "INSERT"}
    # uses the *latest* logged value per key: back to 1.5 -> no delta for A
    assert sl.capture_deltas("IN") == 0


def test_serving_memoised(sv, monkeypatch):
    assert sl.serving("IN").empty
    _write_serving("IN", [{"Symbol": "A", "Close": 1.0}])
    d1 = sl.serving("IN")
    assert sl.serving("IN") is d1
    calls = []
    orig = pd.read_parquet
    monkeypatch.setattr(pd, "read_parquet", lambda p: calls.append(p) or orig(p))
    sl.serving("IN")
    assert calls == []
    _write_serving("IN", [{"Symbol": "A", "Close": 1.0}, {"Symbol": "B", "Close": 2.0}])
    assert len(sl.serving("IN")) == 2


def _serve_df():
    return pd.DataFrame(
        {
            "Symbol": list("ABCDE"),
            "Above200DMA": [True, True, False, True, True],
            "RSI14": [50, 60, 40, 65, 70],
            "Ret252": [10.0, 30.0, 20.0, 5.0, 40.0],
        }
    )


def test_screen_fast(sv):
    assert sl.screen_fast({}, "IN").empty
    _serve_df().to_parquet(sl.SERVING_DIR / "IN.parquet", index=False)
    out = sl.screen_fast({"Above200DMA": ("==", True), "RSI14": ("<", 70)})
    assert list(out["Symbol"]) == ["B", "A", "D"]  # sorted by Ret252 desc
    assert list(out.index) == [0, 1, 2]
    # boundaries
    assert list(sl.screen_fast({"RSI14": ("<=", 60)}, sort="RSI14")["Symbol"]) == ["B", "A", "C"]
    assert list(sl.screen_fast({"RSI14": (">", 65)})["Symbol"]) == ["E"]
    assert set(sl.screen_fast({"RSI14": (">=", 65)})["Symbol"]) == {"D", "E"}
    assert list(sl.screen_fast({"RSI14": ("!=", 50)}, top=2)["Symbol"]) == ["E", "B"]
    assert list(sl.screen_fast({"RSI14": ("==", 40)})["Symbol"]) == ["C"]
    # unknown column / op ignored
    assert len(sl.screen_fast({"Nope": ("<", 1), "RSI14": ("~", 1)})) == 5
    # top None / 0 -> everything; unknown sort keeps order
    assert len(sl.screen_fast({}, top=None)) == 5
    assert len(sl.screen_fast({}, top=0)) == 5
    assert list(sl.screen_fast({}, sort="zzz", top=2)["Symbol"]) == ["A", "B"]


def test_refresh(sv, monkeypatch, capsys):
    built = []
    monkeypatch.setattr(sl, "build_serving", lambda m, v: built.append(m) or (m != "EU"))
    monkeypatch.setattr(sl, "capture_deltas", lambda m, v: 3)
    monkeypatch.setattr(sl.datalink, "changed_markets", lambda: ["IN", "EU"])
    monkeypatch.setattr(sl, "MARKETS", ["IN", "US", "EU"])
    # US not changed but missing serving file -> built; IN changed; EU changed build fails
    r = sl.refresh(sl.MARKETS)
    assert built == ["IN", "US", "EU"] and r == {"views": 2, "deltas": 6}
    assert "serving refresh: 2 views, 6 deltas" in capsys.readouterr().out
    # existing view + unchanged -> skipped
    _write_serving("US", [{"Symbol": "A", "Close": 1.0}])
    built.clear()
    sl.refresh(sl.MARKETS, verbose=False)
    assert built == ["IN", "EU"]
    # explicit subset: all considered changed
    built.clear()
    r = sl.refresh(["US"], verbose=False)
    assert built == ["US"] and r["views"] == 1


def test_refresh_default_markets(sv, monkeypatch):
    built = []
    monkeypatch.setattr(sl, "build_serving", lambda m, v: built.append(m) or 1)
    monkeypatch.setattr(sl, "capture_deltas", lambda m, v: 0)
    monkeypatch.setattr(sl.datalink, "changed_markets", lambda: [])
    monkeypatch.setattr(sl, "MARKETS", ["IN", "US"])
    sl.refresh(verbose=False)
    assert built == ["IN", "US"]  # no serving files -> built regardless


def test_main_refresh_and_screen(sv, monkeypatch, capsys):
    got = {}
    monkeypatch.setattr(sl, "refresh", lambda m: got.setdefault("m", m))
    monkeypatch.setattr(sys, "argv", ["sl", "--refresh", "--market", "IN", "US"])
    assert sl.main() == 0 and got["m"] == ["IN", "US"]
    got.clear()
    monkeypatch.setattr(sys, "argv", ["sl"])
    sl.main()
    assert got["m"] is None
    monkeypatch.setattr(sys, "argv", ["sl", "--screen", "IN"])
    assert sl.main() == 0
    assert "no serving view — run --refresh" in capsys.readouterr().out
    pd.DataFrame(
        {
            "Symbol": ["A", "B", "C"],
            "Close": [1.0, 2.0, 3.0],
            "Above200DMA": [True, True, False],
            "RSI14": [50.0, 75.0, 40.0],
            "PctFromHigh": [-5.0, -1.0, -3.0],
            "Ret252": [1.0, 2.0, 3.0],
        }
    ).to_parquet(sl.SERVING_DIR / "IN.parquet", index=False)
    assert sl.main() == 0
    out = capsys.readouterr().out
    assert " A " in out or out.split()[0] == "Symbol"
    assert "B" not in out.split("\n", 1)[1] and "C" not in out.split("\n", 1)[1]
    assert "Liquidity" not in out
