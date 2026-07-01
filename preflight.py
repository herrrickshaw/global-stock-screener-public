#!/usr/bin/env python3
# preflight.py
# ============
# The single entry point that establishes and verifies the INITIAL CONDITIONS the
# rest of the system (code + Colab notebooks) needs to run — so everything is usable
# out of the box and testable in CI.
#
#   ensure()  — make the conditions true: set BHAV_CACHE, create dirs, derive the 1y
#               STM seeds from the committed 5y LTM, build the serving views.
#   check()   — verify the conditions and return a pass/fail report (LFS data pulled,
#               packages importable, dirs writable, core imports OK, data present).
#
#   python preflight.py                 # health check (exit 1 if a critical cond fails)
#   python preflight.py --ensure        # establish conditions, then check
#   python preflight.py --ensure --markets IN US
#   # in code / notebooks:
#   import preflight; preflight.ensure(["IN", "US"])
#
# This is the one call notebooks make in their setup cell — no ad-hoc bootstrapping.

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).parent
SEED_DIR = HERE / "cache_seed"
LTM_DIR = SEED_DIR / "ltm"

REQUIRED_PKGS = ["pandas", "numpy", "pyarrow"]  # hard requirements to do anything
OPTIONAL_PKGS = ["lmdb", "sklearn", "yfinance", "nsepython", "openpyxl", "vaderSentiment"]
CORE_MODULES = ["frames", "datalink", "market_memory", "serving_layer", "screener_kit"]


# ── establish the conditions ─────────────────────────────────────────────────────
def ensure(
    markets: List[str] | None = None, build_serving: bool = True, verbose: bool = True
) -> dict:
    """Make the initial conditions true. Idempotent; safe to call repeatedly."""
    markets = markets or ["IN"]
    # 1) working cache location (default under /content on Colab, else ~/Downloads)
    if not os.environ.get("BHAV_CACHE"):
        default = (
            "/content/cache"
            if Path("/content").exists()
            else str(Path.home() / "Downloads" / "data" / "bhavcopy_cache")
        )
        os.environ["BHAV_CACHE"] = default
    Path(os.environ["BHAV_CACHE"]).mkdir(parents=True, exist_ok=True)

    # 2) derive the 1y STM seeds from the committed 5y LTM (repo ships only LTM)
    try:
        import market_memory as mm

        mm.ensure_stm_seeds(verbose=False)
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"  ensure: STM derive skipped ({str(e)[:50]})")

    # 3) build the materialised serving views the screeners/tools read
    n = 0
    if build_serving:
        try:
            import serving_layer as sl

            for m in markets:
                n += 1 if sl.build_serving(m, verbose=False) else 0
                sl.capture_deltas(m, verbose=False)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  ensure: serving build skipped ({str(e)[:50]})")
    if verbose:
        print(f"  ensure: cache={os.environ['BHAV_CACHE']}, serving views built for {n} market(s)")
    return {"cache": os.environ["BHAV_CACHE"], "serving_built": n}


# ── verify the conditions ────────────────────────────────────────────────────────
def _is_lfs_pointer(p: Path) -> bool:
    try:
        with open(p, "rb") as fh:
            return fh.read(40).startswith(b"version https://git-lfs")
    except Exception:
        return False


def check(markets: List[str] | None = None, verbose: bool = True) -> Tuple[bool, Dict[str, str]]:
    """Verify initial conditions. Returns (all_critical_ok, {condition: status})."""
    markets = markets or ["IN"]
    r: Dict[str, str] = {}
    critical_ok = True

    # required packages
    missing = [p for p in REQUIRED_PKGS if not _importable(p)]
    r["required packages"] = "OK" if not missing else f"MISSING {missing}"
    critical_ok &= not missing
    opt_missing = [p for p in OPTIONAL_PKGS if not _importable(p)]
    r["optional packages"] = (
        "all present" if not opt_missing else f"missing (some features off): {opt_missing}"
    )

    # LFS data actually pulled (not pointers)
    ltm = list(LTM_DIR.glob("*.parquet"))
    if not ltm:
        r["LFS data"] = "CRITICAL: no LTM seeds — run `git lfs pull`"
        critical_ok = False
    elif any(_is_lfs_pointer(p) for p in ltm):
        r["LFS data"] = "CRITICAL: LTM are LFS pointers — run `git lfs pull`"
        critical_ok = False
    else:
        r["LFS data"] = f"OK ({len(ltm)} LTM seeds pulled)"

    # cache dir writable
    cache = Path(
        os.environ.get("BHAV_CACHE", Path.home() / "Downloads" / "data" / "bhavcopy_cache")
    )
    try:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / ".w").write_text("x")
        (cache / ".w").unlink()
        r["cache writable"] = f"OK ({cache})"
    except Exception as e:  # noqa: BLE001
        r["cache writable"] = f"FAIL ({str(e)[:40]})"
        critical_ok = False

    # core modules import
    bad = []
    for m in CORE_MODULES:
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            bad.append(f"{m}:{str(e)[:25]}")
    r["core imports"] = "OK" if not bad else f"FAIL {bad}"
    critical_ok &= not bad

    # serving views available for the requested markets (usability of screeners)
    try:
        import serving_layer as sl

        built = [m for m in markets if not sl.serving(m).empty]
        r["serving views"] = (
            f"{len(built)}/{len(markets)} ready"
            if built
            else "none built — call preflight.ensure()"
        )
    except Exception:
        r["serving views"] = "unknown"

    if verbose:
        print("=== preflight: initial conditions ===")
        for k, v in r.items():
            mark = (
                "✓"
                if v.startswith(("OK", "all", "ready"))
                or "/" in v
                or v.startswith(str(len(markets)))
                else ("•" if not v.startswith(("CRITICAL", "FAIL", "MISSING")) else "✗")
            )
            print(f"  {mark} {k:20} {v}")
        print(f"  → {'READY ✓' if critical_ok else 'NOT READY ✗ (fix CRITICAL items above)'}")
    return critical_ok, r


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Establish + verify the system's initial conditions")
    ap.add_argument("--ensure", action="store_true", help="establish conditions before checking")
    ap.add_argument("--markets", nargs="*", default=["IN"])
    args = ap.parse_args()
    if args.ensure:
        ensure(args.markets)
    ok, _ = check(args.markets)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
