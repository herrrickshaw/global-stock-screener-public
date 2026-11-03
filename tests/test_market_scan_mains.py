"""Characterisation of the Europe/Japan/Korea scan `main()` entry points.

Golden JSON (tests/fixtures/golden_market_scans.json) was captured from the code BEFORE
pass-2 splitting. Everything external (network, universe, fundamentals) is faked; output
goes to a temp dir. Rows are compared order-insensitively because the worker pool may
complete in any order.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

GOLDEN = Path(__file__).parent / "fixtures" / "golden_market_scans.json"
N_DAYS = 260


def _series(kind: int) -> pd.DataFrame:
    rng = np.random.default_rng(100 + kind)
    base = 100 + np.cumsum(rng.normal(0, 0.6, N_DAYS))
    if kind % 3 == 0:  # flat then jump on last bar -> breakout candidate
        base[:-1] = 100 + rng.normal(0, 0.3, N_DAYS - 1)
        base[-1] = 130
    elif kind % 3 == 1:  # flat then crash on last bar -> breakdown candidate
        base[:-1] = 100 + rng.normal(0, 0.3, N_DAYS - 1)
        base[-1] = 70
    close = base
    idx = pd.bdate_range("2025-01-01", periods=N_DAYS)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": np.full(N_DAYS, 1000.0),
        },
        index=idx,
    )


def _fake_scan_result(i: int) -> dict:
    return {
        "f_score": 9 - (i % 3) * 2,
        "f_strong": (9 - (i % 3) * 2) >= 7,
        "f_interpretation": "x",
        "cc_qualifies": "PASS" if i % 2 == 0 else "FAIL",
        "cc_score": i % 5,
        "cc_rev_cagr": 10.0 + i,
        "cc_roce_avg": 15.0 + i,
        "cc_debt_equity": 0.5,
        "market_cap_b_jpy": 1.5 + i,
        "market_cap_b_krw": 2.5 + i,
        "market_cap_b": 3.5 + i,
        "pe_ratio": 12.0,
        "pb_ratio": 1.2,
        "dividend_yield": 2.0,
        "pegy_ratio": 1.1,
        "sector": "Tech",
        "error": "",
    }


_TS = re.compile(r"\d{2} \w{3} \d{4}\s+\d{2}:\d{2}:\d{2}")
_PATH = re.compile(r"(saved|Excel saved) → .*")


def _snapshot(out_dir: Path, stdout: str) -> dict:
    files = list(out_dir.glob("*.xlsx"))
    assert len(files) == 1
    sheets = pd.read_excel(files[0], sheet_name=None)
    snap_sheets = {}
    for name, df in sheets.items():
        df = df.astype(object).where(df.notna(), None)
        rows = sorted((json.dumps(r, default=str, sort_keys=True) for r in df.to_dict("records")))
        snap_sheets[name] = {"columns": list(df.columns), "rows": rows}
    lines = sorted(
        _PATH.sub(r"\1 → <path>", _TS.sub("<ts>", ln)) for ln in stdout.splitlines() if ln.strip()
    )
    return {"sheets": snap_sheets, "stdout": lines}


def _load(name: str, tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "pykrx", types.SimpleNamespace(stock=types.SimpleNamespace()))
    sys.modules.pop(name, None)
    mod = importlib.import_module(name)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(mod, "DOWNLOAD_DIR", out)
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
    return mod, out


def _run_japan(tmp_path, monkeypatch, capsys):
    mod, out = _load("full_japan_market_scan", tmp_path, monkeypatch)
    tickers = [f"{1000 + i}.T" for i in range(9)]
    uni = [
        {"yf_ticker": t, "code": t[:-2], "name": f"Co{i}", "sector": "S"}
        for i, t in enumerate(tickers)
    ]
    frames = {t: _series(i) for i, t in enumerate(tickers)}
    monkeypatch.setattr(mod, "build_ticker_universe", lambda: uni)
    monkeypatch.setattr(mod, "bulk_download_ohlc", lambda tk, period="3mo": {t: frames[t] for t in tk})
    monkeypatch.setattr(mod, "fundamental_scan", lambda t: _fake_scan_result(int(t[:4]) % 10))
    monkeypatch.setattr(sys, "argv", ["x", "--workers", "1"])
    mod.main()
    return _snapshot(out, capsys.readouterr().out)


def _run_korea(tmp_path, monkeypatch, capsys):
    mod, out = _load("full_korea_market_scan", tmp_path, monkeypatch)
    uni = [
        {
            "code": f"{5000 + i:06d}",
            "name": f"Co{i}",
            "market": "KOSPI" if i % 2 else "KOSDAQ",
            "yf_suffix": ".KS" if i % 2 else ".KQ",
        }
        for i in range(9)
    ]
    frames = {s["code"]: _series(i) for i, s in enumerate(uni)}
    monkeypatch.setattr(mod, "build_krx_universe", lambda kospi_only=False: list(uni))
    monkeypatch.setattr(mod, "fetch_krx_ohlc", lambda u, months=3: dict(frames))
    monkeypatch.setattr(mod, "fundamental_scan", lambda c, s: _fake_scan_result(int(c) % 10))
    monkeypatch.setattr(sys, "argv", ["x", "--workers", "1"])
    mod.main()
    return _snapshot(out, capsys.readouterr().out)


def _run_europe(tmp_path, monkeypatch, capsys):
    mod, out = _load("full_european_market_scan", tmp_path, monkeypatch)
    syms = list(mod.EURO_STOXX_50_META)[:9]
    frames = {s: _series(i) for i, s in enumerate(syms)}

    def fake_download(chunk, **_kw):
        return pd.concat({s: frames[s] for s in chunk}, axis=1)

    monkeypatch.setattr(mod.yf, "download", fake_download)
    monkeypatch.setattr(mod, "fundamental_scan", lambda s: _fake_scan_result(syms.index(s)))
    monkeypatch.setattr(mod, "MAX_WORKERS", 1)
    monkeypatch.setattr(sys, "argv", ["x", "--top", "9", "--batch", "4"])
    mod.main()
    return _snapshot(out, capsys.readouterr().out)


RUNNERS = {"japan": _run_japan, "korea": _run_korea, "europe": _run_europe}


@pytest.mark.parametrize("market", sorted(RUNNERS))
def test_scan_main_matches_golden(market, tmp_path, monkeypatch, capsys):
    snap = RUNNERS[market](tmp_path, monkeypatch, capsys)
    if not GOLDEN.exists() or (
        market not in json.loads(GOLDEN.read_text()) and __import__("os").environ.get("UPDATE_GOLDEN")
    ):
        data = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
        data[market] = snap
        GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True))
    golden = json.loads(GOLDEN.read_text())[market]
    assert json.loads(json.dumps(snap)) == golden
