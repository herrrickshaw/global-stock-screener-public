#!/usr/bin/env python3
# backlog.py
# ==========
# SAFe backlog management, made executable: a WSJF-prioritised backlog for the
# project. Items are Epics/Features/Stories with Cost-of-Delay components and a job
# size; WSJF = Cost of Delay / Job Size ranks what to pull into the next Program
# Increment (PI). See AGILE.md for the operating model.
#
#   WSJF = (user_business_value + time_criticality + risk_reduction) / job_size
#          (Fibonacci-scaled inputs; higher WSJF = do first — shortest, highest value)
#
#   python backlog.py                 # print the WSJF-ranked backlog
#   python backlog.py --pi            # propose the next PI (fits a capacity budget)
#   python backlog.py --add ...       # (see --help)
#
# Backlog persisted in backlog.json next to this file.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

HERE = Path(__file__).parent
STORE = HERE / "backlog.json"

# seed backlog = the real roadmap (from DATA_SOURCES.md opportunities + open items).
# value / time_criticality / risk (Cost-of-Delay components) and job_size are
# Fibonacci-scaled (1,2,3,5,8,13,20). status: todo | doing | done.
_SEED: List[dict] = [
    {
        "id": "F-01",
        "type": "Feature",
        "title": "FII/DII flows + India VIX regime signal",
        "value": 13,
        "time_criticality": 8,
        "risk": 8,
        "job_size": 8,
        "status": "todo",
        "epic": "Smart-money signals",
    },
    {
        "id": "F-02",
        "type": "Feature",
        "title": "Map 45 NO-DATA delisted/renamed tickers",
        "value": 8,
        "time_criticality": 5,
        "risk": 3,
        "job_size": 3,
        "status": "todo",
        "epic": "Data quality",
    },
    {
        "id": "F-03",
        "type": "Feature",
        "title": "Clamp absurd P/E (near-zero EPS) in decision engine",
        "value": 5,
        "time_criticality": 5,
        "risk": 5,
        "job_size": 1,
        "status": "todo",
        "epic": "Data quality",
    },
    {
        "id": "F-04",
        "type": "Feature",
        "title": "Trendlyne DVM scores + promoter pledge into decision factors",
        "value": 13,
        "time_criticality": 5,
        "risk": 5,
        "job_size": 5,
        "status": "todo",
        "epic": "Fundamentals depth",
    },
    {
        "id": "F-05",
        "type": "Feature",
        "title": "Deepen thin LTMs past 252 bars (full 1y momentum)",
        "value": 8,
        "time_criticality": 3,
        "risk": 3,
        "job_size": 5,
        "status": "todo",
        "epic": "Data depth",
    },
    {
        "id": "F-06",
        "type": "Feature",
        "title": "Dividend/split-adjusted total-return series",
        "value": 8,
        "time_criticality": 2,
        "risk": 3,
        "job_size": 5,
        "status": "todo",
        "epic": "Data quality",
    },
    {
        "id": "F-07",
        "type": "Feature",
        "title": "Factor regressions (Fama-French) attribution",
        "value": 8,
        "time_criticality": 2,
        "risk": 2,
        "job_size": 8,
        "status": "todo",
        "epic": "Analytics",
    },
    {
        "id": "F-08",
        "type": "Feature",
        "title": "PPO/SB3 upgrade path for the RL trade agent",
        "value": 5,
        "time_criticality": 1,
        "risk": 2,
        "job_size": 13,
        "status": "todo",
        "epic": "ML",
    },
    # already delivered this program increment (kept for velocity/history):
    {
        "id": "F-D1",
        "type": "Feature",
        "title": "Delivery-% accumulation signal",
        "value": 13,
        "time_criticality": 8,
        "risk": 8,
        "job_size": 5,
        "status": "done",
        "epic": "Smart-money signals",
    },
    {
        "id": "F-D2",
        "type": "Feature",
        "title": "Sector classification + rotation",
        "value": 13,
        "time_criticality": 5,
        "risk": 5,
        "job_size": 5,
        "status": "done",
        "epic": "Analytics",
    },
    {
        "id": "F-D3",
        "type": "Feature",
        "title": "Trendlyne fundamentals source (P/E, ROE, +)",
        "value": 13,
        "time_criticality": 8,
        "risk": 5,
        "job_size": 5,
        "status": "done",
        "epic": "Fundamentals depth",
    },
]


def _load() -> List[dict]:
    if STORE.exists():
        try:
            return json.loads(STORE.read_text())
        except Exception:
            pass
    STORE.write_text(json.dumps(_SEED, indent=2))
    return list(_SEED)


def _save(items: List[dict]) -> None:
    STORE.write_text(json.dumps(items, indent=2))


def wsjf(item: dict) -> float:
    """WSJF = Cost of Delay / Job Size. Higher = higher priority."""
    cod = item.get("value", 0) + item.get("time_criticality", 0) + item.get("risk", 0)
    size = item.get("job_size", 0) or 1
    return round(cod / size, 2)


def ranked(items: List[dict], status: str = "todo") -> List[dict]:
    todo = [i for i in items if i.get("status") == status]
    return sorted(todo, key=wsjf, reverse=True)


def next_pi(items: List[dict], capacity: int = 15) -> List[dict]:
    """Fit the highest-WSJF items into a PI capacity budget (sum of job_size)."""
    plan, used = [], 0
    for i in ranked(items):
        if used + i["job_size"] <= capacity:
            plan.append(i)
            used += i["job_size"]
    return plan


def _print_backlog(items: List[dict]) -> None:
    print("=== Program backlog (WSJF-ranked) ===")
    print(f"  {'WSJF':>5}  {'ID':5} {'size':>4}  Title  [epic]")
    for i in ranked(items):
        print(f"  {wsjf(i):>5}  {i['id']:5} {i['job_size']:>4}  {i['title']}  [{i['epic']}]")
    done = [i for i in items if i.get("status") == "done"]
    print(
        f"\n  delivered this PI (velocity): {sum(i['job_size'] for i in done)} pts / {len(done)} features"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="SAFe WSJF-prioritised backlog")
    ap.add_argument("--pi", action="store_true", help="propose the next Program Increment")
    ap.add_argument("--capacity", type=int, default=15, help="PI capacity (job-size budget)")
    ap.add_argument("--json", action="store_true", help="emit ranked backlog as JSON")
    args = ap.parse_args()
    items = _load()
    if args.json:
        print(json.dumps([{**i, "wsjf": wsjf(i)} for i in ranked(items)], indent=2))
        return 0
    if args.pi:
        plan = next_pi(items, args.capacity)
        print(f"=== Proposed next PI (capacity {args.capacity} pts) ===")
        for i in plan:
            print(f"  WSJF {wsjf(i):>5}  {i['id']}  {i['title']} ({i['job_size']} pts)")
        print(f"  committed: {sum(i['job_size'] for i in plan)} pts across {len(plan)} features")
        return 0
    _print_backlog(items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
