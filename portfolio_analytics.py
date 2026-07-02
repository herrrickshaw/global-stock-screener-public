#!/usr/bin/env python3
# portfolio_analytics.py
# ======================
# F-11 — portfolio-level analytics: roll up a holdings list into concentration,
# valuation, signal mix, risk flags, and concrete rebalance suggestions. Where the
# per-name evaluator answers "is this a buy?", this answers "how healthy is the whole
# book?".
#
#   from portfolio_analytics import analyze
#   analyze(["RELIANCE","TCS","INFY"], market="IN")     # equal-weight
#   analyze({"RELIANCE":0.4,"TCS":0.6}, market="IN")    # weighted
#
#   python3 portfolio_analytics.py --tickers RELIANCE,TCS,INFY --market IN
#
# ⚠️ Research/education only. Not advice.

from __future__ import annotations

import argparse
import warnings
from typing import Dict, List, Union

warnings.filterwarnings("ignore")


def _weights(holdings: Union[List[str], Dict[str, float]]) -> Dict[str, float]:
    if isinstance(holdings, dict):
        tot = sum(holdings.values()) or 1.0
        return {k.upper(): v / tot for k, v in holdings.items()}
    syms = [s.upper() for s in holdings]
    w = 1.0 / len(syms) if syms else 0.0
    return {s: w for s in syms}


def _hhi(weights: Dict[str, float]) -> float:
    """Herfindahl concentration index (0=diversified, 1=single name)."""
    return round(sum(v * v for v in weights.values()), 3)


def rebalance_actions(
    rows: List[dict], max_name: float = 0.10, max_sector: float = 0.30
) -> List[str]:
    """Suggest trims: SELL/IMPAIRED holdings, oversized names, oversized sectors."""
    actions = []
    sector_w: Dict[str, float] = {}
    for r in rows:
        sector_w[r.get("sector", "?")] = sector_w.get(r.get("sector", "?"), 0) + r["weight"]
    for r in rows:
        if r["signal"] in ("SELL",) or "IMPAIRED" in r.get("flags", ""):
            actions.append(f"TRIM/EXIT {r['symbol']} — {r['signal']} {r.get('flags','')}".strip())
        elif r["weight"] > max_name:
            actions.append(
                f"REDUCE {r['symbol']} — {r['weight']:.0%} > {max_name:.0%} single-name cap"
            )
    for sec, w in sector_w.items():
        if sec not in ("?", "") and w > max_sector:
            actions.append(f"REDUCE sector {sec} — {w:.0%} > {max_sector:.0%} sector cap")
    return actions


def analyze(
    holdings: Union[List[str], Dict[str, float]], market: str = "IN", verbose: bool = True
) -> dict:
    import portfolio_evaluator as pe

    w = _weights(holdings)
    ev = pe.evaluate(list(w), market, fetch=False, verbose=False).set_index("Symbol")
    try:
        import sector_map as sm

        smap = sm.sector_map(market)
    except Exception:
        smap = {}

    rows, wpe_num, wpe_den = [], 0.0, 0.0
    sig_mix = {"BUY": 0.0, "HOLD": 0.0, "SELL": 0.0, "NO DATA": 0.0}
    for s, wt in w.items():
        r = ev.loc[s] if s in ev.index else None
        sig = r["Signal"] if r is not None else "NO DATA"
        pe_v = (
            float(r["PE"]) if r is not None and r.get("PE") == r.get("PE") and r.get("PE") else None
        )
        sig_mix[sig] = sig_mix.get(sig, 0) + wt
        if pe_v and 0 < pe_v < 1000:
            wpe_num += pe_v * wt
            wpe_den += wt
        rows.append(
            {
                "symbol": s,
                "weight": wt,
                "signal": sig,
                "flags": (r.get("Flags", "") if r is not None else ""),
                "sector": smap.get(s, "?"),
            }
        )

    sector_w: Dict[str, float] = {}
    for r in rows:
        sector_w[r["sector"]] = sector_w.get(r["sector"], 0) + r["weight"]
    out = {
        "market": market,
        "n": len(w),
        "name_hhi": _hhi(w),
        "sector_hhi": _hhi(sector_w),
        "weighted_pe": round(wpe_num / wpe_den, 1) if wpe_den else None,
        "signal_mix": {k: round(v, 3) for k, v in sig_mix.items() if v},
        "top_sectors": dict(sorted(sector_w.items(), key=lambda kv: -kv[1])[:5]),
        "flagged": [r["symbol"] for r in rows if r["flags"]],
        "actions": rebalance_actions(rows),
    }
    if verbose:
        print(f"=== Portfolio analytics ({market}) — {out['n']} holdings ===")
        print(f"  concentration: name HHI {out['name_hhi']}, sector HHI {out['sector_hhi']}")
        print(f"  weighted P/E: {out['weighted_pe']}")
        print(f"  signal mix (by weight): {out['signal_mix']}")
        print(f"  top sectors: { {k: round(v,2) for k,v in out['top_sectors'].items()} }")
        print("  rebalance actions:")
        for a in out["actions"] or ["  (none — book looks balanced)"]:
            print(f"    • {a}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Portfolio-level analytics")
    ap.add_argument("--tickers", required=True, help="comma list")
    ap.add_argument("--market", default="IN")
    args = ap.parse_args()
    analyze([t.strip() for t in args.tickers.split(",") if t.strip()], args.market)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
