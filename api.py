#!/usr/bin/env python3
# api.py
# ======
# F-17 — a thin HTTP surface over the screener so the notebooks/tools become a
# shareable service. Route logic lives in plain functions (testable without a server);
# FastAPI just wraps them, and is an OPTIONAL dependency.
#
#   pip install fastapi uvicorn
#   uvicorn api:app --reload         # then GET /health, /screen/IN/darvas, /brief, /decide?tickers=RELIANCE,TCS
#
# Research/education only. Not advice.
from __future__ import annotations

import warnings
from typing import List, Optional

warnings.filterwarnings("ignore")


# ── route logic (pure-ish, importable, testable without FastAPI) ─────────────────
def health() -> dict:
    try:
        import preflight

        ok, rep = preflight.check(verbose=False)
        return {"status": "ok" if ok else "degraded", "conditions": rep}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "detail": str(e)[:80]}


def screen(market: str, strategy: str, top: int = 25) -> list:
    import screener_kit as k

    df = k.screen(strategy, market, top=top)
    return df.to_dict("records") if not df.empty else []


def decide(tickers: List[str], market: str = "IN") -> list:
    import portfolio_evaluator as pe

    out = pe.evaluate(
        [t.strip().upper() for t in tickers if t.strip()], market, fetch=False, verbose=False
    )
    return out.to_dict("records") if not out.empty else []


def brief(markets: Optional[List[str]] = None) -> dict:
    import daily_brief as db

    return {"markdown": db.build_brief(markets)}


ROUTES = {"health": health, "screen": screen, "decide": decide, "brief": brief}


# ── FastAPI wrapper (only if installed) ──────────────────────────────────────────
def create_app():
    from fastapi import FastAPI, Query  # noqa: F401

    app = FastAPI(title="Global Stock Screener API", version="1.0")

    @app.get("/health")
    def _health():
        return health()

    @app.get("/screen/{market}/{strategy}")
    def _screen(market: str, strategy: str, top: int = 25):
        return screen(market, strategy, top)

    @app.get("/decide")
    def _decide(tickers: str, market: str = "IN"):
        return decide(tickers.split(","), market)

    @app.get("/brief")
    def _brief(markets: str = "IN,US"):
        return brief(markets.split(","))

    return app


try:  # expose `app` for `uvicorn api:app` when FastAPI is present
    app = create_app()
except Exception:  # noqa: BLE001
    app = None


if __name__ == "__main__":
    print("routes:", list(ROUTES))
    print("health:", health()["status"])
