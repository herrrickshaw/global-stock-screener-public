"""Pins the 20-country universe table (content and order) of Global20CountryScreener."""

from __future__ import annotations

import json
import os
from pathlib import Path

from global_20country_universe_screener import Global20CountryScreener

GOLDEN = Path(__file__).parent / "fixtures" / "golden_universe_countries.json"


def test_countries_config_matches_golden():
    cfg = Global20CountryScreener().countries_config
    snap = [[k, v] for k, v in cfg.items()]
    if not GOLDEN.exists() and os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(snap, indent=1))
    assert snap == json.loads(GOLDEN.read_text())
    a, b = Global20CountryScreener(), Global20CountryScreener()
    assert a.countries_config is not b.countries_config  # fresh dict per instance
    assert a.countries_config["USA"]["sectors"] is not b.countries_config["USA"]["sectors"]
