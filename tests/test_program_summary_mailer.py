"""Characterisation of program_summary_mailer.build() (pure HTML/text; nothing is sent)."""

from __future__ import annotations

import importlib
import json
import os
import sys
from datetime import date
from pathlib import Path

GOLDEN = Path(__file__).parent / "fixtures" / "golden_program_summary_mailer.json"


class _FixedDate(date):
    @classmethod
    def today(cls):
        return cls(2026, 1, 15)


def test_build_matches_golden(monkeypatch):
    sys.modules.pop("program_summary_mailer", None)
    mod = importlib.import_module("program_summary_mailer")
    monkeypatch.setattr(mod, "date", _FixedDate)
    s, t, h = mod.build()
    snap = {"subject": s, "text": t, "html": h}
    if not GOLDEN.exists() and os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(snap, indent=1, sort_keys=True))
    assert snap == json.loads(GOLDEN.read_text())
