#!/usr/bin/env python3
# flows_regime.py
# ===============
# F-01 — India market-regime signal from FII/DII cash flows + India VIX.
#
# Institutional flows and volatility set the risk backdrop screeners should respect:
#   • FII net buying + low VIX      → Risk-On   (favour momentum/beta)
#   • FII net selling + high VIX    → Risk-Off  (favour quality/defensives, trim)
#
# Source: nsepython (FII/DII cash-segment activity + India VIX). Best-effort fetch +
# cache (cache_seed/flows/IN.json); the regime SCORING runs offline on the cache.
#
#   python3 flows_regime.py --fetch     # pull latest FII/DII + VIX → cache
#   python3 flows_regime.py             # print the current regime signal
#
# ⚠️ Research/education only. Not advice.

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
CACHE = HERE / "cache_seed" / "flows"
CACHE.mkdir(parents=True, exist_ok=True)
STORE = CACHE / "IN.json"


# ── fetch (best-effort) ──────────────────────────────────────────────────────────
def fetch(verbose: bool = True) -> dict:
    """Pull latest FII/DII net (₹cr) + India VIX via nsepython → cache."""
    out: dict = {}
    try:
        import nsepython as nse

        try:
            vix = float(nse.nse_get_index_quote("INDIA VIX")["lastPrice"])
            out["vix"] = vix
        except Exception:
            pass
        try:
            fd = nse.nse_fiidii()  # list of dicts: FII/DII buy/sell/net
            for row in fd if isinstance(fd, list) else []:
                cat = str(row.get("category", "")).upper()
                net = float(row.get("netValue", row.get("net", 0)) or 0)
                if "FII" in cat or "FPI" in cat:
                    out["fii_net"] = net
                elif "DII" in cat:
                    out["dii_net"] = net
        except Exception:
            pass
    except Exception:
        pass
    if out:
        import datetime as _dt

        out["asof"] = _dt.date.today().isoformat()
        STORE.write_text(json.dumps(out, indent=2))
    if verbose:
        print(f"  flows: {out or 'fetch failed (nsepython/network)'}")
    return out


def _load() -> dict:
    return json.loads(STORE.read_text()) if STORE.exists() else {}


# ── regime scoring (offline, testable) ───────────────────────────────────────────
def regime(data: Optional[dict] = None) -> dict:
    """Risk regime from flows + VIX. Returns {regime, score, drivers}.

    score in [-100, 100]: >25 Risk-On, <-25 Risk-Off, else Neutral. Missing inputs
    are simply skipped (graceful)."""
    d = data if data is not None else _load()
    score, drivers = 0.0, []
    vix = d.get("vix")
    if vix is not None:
        # low VIX (<14) risk-on; high (>20) risk-off; linear between
        v = max(-40, min(40, (17 - vix) * 4))
        score += v
        drivers.append(f"VIX {vix:.1f}")
    fii = d.get("fii_net")
    if fii is not None:
        score += max(-40, min(40, fii / 50))  # ±₹2000cr ≈ ±40
        drivers.append(f"FII net ₹{fii:,.0f}cr")
    dii = d.get("dii_net")
    if dii is not None:
        score += max(-20, min(20, dii / 100))
        drivers.append(f"DII net ₹{dii:,.0f}cr")
    reg = "Risk-On" if score > 25 else ("Risk-Off" if score < -25 else "Neutral")
    return {"regime": reg, "score": round(score, 1), "drivers": drivers, "asof": d.get("asof", "")}


def regime_bias() -> float:
    """A [-1, 1] tilt usable by screeners/decisions (positive = risk-on)."""
    return round(regime()["score"] / 100, 2)


def main() -> int:
    ap = argparse.ArgumentParser(description="India FII/DII + VIX regime signal")
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        fetch()
    r = regime()
    print(
        f"India regime: {r['regime']}  (score {r['score']}) — {', '.join(r['drivers']) or 'no data — run --fetch'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
