"""Characterisation of build_mailer.build() (HTML/text assembly only; nothing is sent).

Section helpers, liquidity, scoreboards and optional modules are stubbed; the CCC section
reads a synthetic parquet from a temp cwd. Golden: tests/fixtures/golden_build_mailer.json.
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

GOLDEN = Path(__file__).parent / "fixtures" / "golden_build_mailer.json"


class _FixedDate(dt.date):
    @classmethod
    def today(cls):
        return cls(2026, 1, 15)


def _annotate(df):
    df = df.copy()
    df["Liquidity"] = ["High", "Medium", "Low"] * (len(df) // 3 + 1)
    df["Liquidity"] = df["Liquidity"].iloc[: len(df)].values if len(df) else df.get("Liquidity")
    return df


def _setup(tmp_path, monkeypatch, *, with_ccc: bool, with_extras: bool):
    monkeypatch.chdir(tmp_path)
    if with_ccc:
        (tmp_path / "cache_seed").mkdir()
        pd.DataFrame(
            {
                "Symbol": list("ABCDEF"),
                "Name": [f"N{i}" for i in range(6)],
                "Cash_Cycle": ["10.5", "-3", "x", "40", "5", "7"],
                "ROCE": [12, 15, 18, 20, 22, 24],
            }
        ).to_parquet(tmp_path / "cache_seed" / "india_ccc_screen.parquet")
    sys.modules.pop("build_mailer", None)
    mod = importlib.import_module("build_mailer")
    monkeypatch.setattr(mod, "_dt", types.SimpleNamespace(date=_FixedDate))
    monkeypatch.setattr(mod.liq, "annotate", _annotate)
    hl = pd.DataFrame(
        {
            "Market": ["IN", "US", "KR", "JP", "ZZ"] * 4,
            "Symbol": [f"S{i}" for i in range(20)],
            "ret_126": [50.0 - i for i in range(20)],
            "rsi14": [60 + i for i in range(20)],
        }
    )
    monkeypatch.setattr(mod.rga, "load_highlights", lambda: hl)
    monkeypatch.setattr(
        mod.mp,
        "load",
        lambda: pd.DataFrame(
            {
                "Market": ["IN"],
                "Index": ["NIFTY"],
                "CAGR%": [12.5],
                "Return_1y%": [8.1],
                "Sharpe": [0.9],
            }
        ),
    )
    if with_extras:
        reports = {
            "IN": {"mood": {"mood": "bullish", "score": 0.31, "n_articles": 12}, "picks": [1]},
            "US": {"mood": {"mood": "neutral", "score": -0.02, "n_articles": 5}},
        }
        monkeypatch.setattr(mod, "_combined_report", lambda m: reports.get(m))
        monkeypatch.setattr(
            mod, "_picks_rows", lambda d, m, limit=12: [f"<tr><td>{m}pick</td></tr>"] if d else []
        )
        monkeypatch.setattr(
            mod,
            "_convergence_rows",
            lambda d: ["<tr><td>conv</td></tr>"] if d and d.get("picks") else [],
        )
        monkeypatch.setattr(mod, "_ai_screener_section", lambda m: None)
        monkeypatch.setattr(mod, "_gms_scan", lambda m: None)
        monkeypatch.setattr(mod, "_europe_scan", lambda: None)
        rp = types.SimpleNamespace(
            section_html=lambda: "<div>REGIME</div>",
            picks_for=lambda m: [{"symbol": f"{m}1", "price": 10.0}],
        )
        vd = types.SimpleNamespace(
            verify_us=lambda p: ("us", p),
            verify_india=lambda p: ("in", p),
            verification_html=lambda a, b: f"<div>VERIFY {a} {b}</div>",
            verification_text=lambda a, b: "VERIFY-TEXT",
        )
        monkeypatch.setitem(sys.modules, "regime_picks", rp)
        monkeypatch.setitem(sys.modules, "verify_data", vd)
    return mod


def test_build_matches_golden(tmp_path, monkeypatch):
    snap = {}
    for name, ccc, extras in [("bare", False, False), ("full", True, True)]:
        d = tmp_path / name
        d.mkdir()
        mod = _setup(d, monkeypatch, with_ccc=ccc, with_extras=extras)
        if not extras:
            monkeypatch.setitem(sys.modules, "regime_picks", None)  # import fails -> except path
            monkeypatch.setitem(sys.modules, "verify_data", None)
            monkeypatch.setattr(mod, "_ai_screener_section", lambda m: None)
            monkeypatch.setattr(mod, "_gms_scan", lambda m: None)
            monkeypatch.setattr(mod, "_europe_scan", lambda: None)
        s, t, h = mod.build()
        snap[name] = {"subject": s, "text": t, "html": h}
    if not GOLDEN.exists() and os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(snap, indent=1, sort_keys=True))
    assert snap == json.loads(GOLDEN.read_text())
