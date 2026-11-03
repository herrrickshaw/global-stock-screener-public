"""Characterisation of bhavcopy_history.fetch_history (cold, warm, partially-cached runs).

Golden: tests/fixtures/golden_bhavcopy_history.json, captured before pass-2 splitting.
All network is faked; caches live in a temp dir (BHAV_CACHE); today is frozen.
"""

from __future__ import annotations

import datetime as dt
import importlib
import json
import os
import sys
import types
from pathlib import Path

import pandas as pd

GOLDEN = Path(__file__).parent / "fixtures" / "golden_bhavcopy_history.json"
TODAY = dt.date(2026, 1, 15)
HOLIDAY = dt.date(2026, 1, 5)  # calendar says no trading
DEAD = dt.date(2026, 1, 8)  # calendar says trading but neither exchange published


class _FixedDate(dt.date):
    @classmethod
    def today(cls):
        return cls(TODAY.year, TODAY.month, TODAY.day)


def _csv_rows(symbols, day, series_col, series_val):
    rows = []
    for k, s in enumerate(symbols):
        base = 100 + 10 * k + day.toordinal() % 17
        rows.append(
            {
                "TckrSymb": s,
                series_col: series_val,
                "OpnPric": base,
                "HghPric": base + 2,
                "LwPric": base - 2,
                "ClsPric": base + 1,
                "TtlTradgVol": 1000 + k,
            }
        )
    return pd.DataFrame(rows)


class _NSEClient:
    def __init__(self, tmp, **_kw):
        self.tmp = tmp

    def equityBhavcopy(self, date):
        d = date.date()
        if d in (DEAD, TODAY):
            raise RuntimeError("not published")
        p = self.tmp / f"raw_{d:%Y%m%d}.csv"
        _csv_rows(["AAA", "BBB", "CCC"], d, "SctySrs", "EQ").to_csv(p, index=False)
        return str(p)

    def exit(self):
        pass


class _Resp:
    def __init__(self, status, content=b""):
        self.status_code, self.content = status, content


class _Sess:
    def __init__(self):
        self.headers = {}

    def get(self, url, timeout=0):
        day = dt.datetime.strptime(url.split("_")[-3], "%Y%m%d").date()
        if day in (DEAD, TODAY):
            return _Resp(404)
        df = _csv_rows(["CCC", "DDD", "EEE"], day, "FinInstrmTp", "STK")
        return _Resp(200, df.to_csv(index=False).encode() + b" " * 1000)


def _load(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setenv("BHAV_CACHE", str(cache))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "nse",
        types.SimpleNamespace(NSE=lambda download_folder: _NSEClient(Path(download_folder).parent)),
    )
    monkeypatch.setitem(
        sys.modules, "bhavcopy_store", types.SimpleNamespace(build=lambda *a, **k: None)
    )
    monkeypatch.setitem(
        sys.modules,
        "market_calendar",
        types.SimpleNamespace(
            trading_days=lambda cal: [d for d in cal if d.weekday() < 5 and d != HOLIDAY]
        ),
    )
    sys.modules.pop("bhavcopy_history", None)
    mod = importlib.import_module("bhavcopy_history")
    monkeypatch.setattr(
        mod,
        "_dt",
        types.SimpleNamespace(date=_FixedDate, timedelta=dt.timedelta, datetime=dt.datetime),
    )
    monkeypatch.setattr(mod.requests, "Session", _Sess)
    return mod, cache


def _summ(out):
    return {
        s: {
            "n": len(g),
            "first": str(g.index[0].date()),
            "last": str(g.index[-1].date()),
            "close_sum": round(float(g["Close"].sum()), 6),
            "vol_sum": float(g["Volume"].sum()),
        }
        for s, g in sorted(out.items())
    }


def _files(cache):
    return sorted(str(p.relative_to(cache)) for p in cache.rglob("*") if p.is_file())


def test_fetch_history_matches_golden(tmp_path, monkeypatch, capsys):
    mod, cache = _load(tmp_path, monkeypatch)
    snap = {}
    cold = mod.fetch_history(n_days=14, workers=2, min_bars=3)
    snap["cold"] = _summ(cold)
    snap["cold_files"] = _files(cache)
    snap["no_data"] = json.loads((cache / "no_data_dates.json").read_text())
    warm = mod.fetch_history(n_days=14, workers=2, min_bars=3)  # fast path
    snap["warm"] = _summ(warm)
    # incremental: drop the cleaned + assembled caches' newest dates by deleting cleaned only
    (cache / "cleaned_long.parquet").unlink()
    again = mod.fetch_history(n_days=14, workers=2, min_bars=3)
    snap["rebuild_from_assembled"] = _summ(again)
    # fresh-clone path: only cleaned cache present
    (cache / "assembled_long.parquet").unlink()
    pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-13", "2026-01-14"]),
            "Open": [1.0, 2.0],
            "High": [2.0, 3.0],
            "Low": [0.5, 1.5],
            "Close": [1.5, 2.5],
            "Volume": [10.0, 20.0],
            "Symbol": ["ZZZ", "ZZZ"],
        }
    ).to_parquet(cache / "cleaned_long.parquet", index=False)
    seeded = mod.fetch_history(n_days=14, workers=2, min_bars=1, exchanges=("NSE",))
    snap["seeded_nse_only"] = _summ(seeded)
    snap["stdout"] = sorted(ln for ln in capsys.readouterr().out.splitlines() if ln.strip())
    snap["empty_stdout_marker"] = os.environ.get("BHAV_CACHE") is not None
    if not GOLDEN.exists() and os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(snap, indent=1, sort_keys=True))
    assert json.loads(json.dumps(snap)) == json.loads(GOLDEN.read_text())
