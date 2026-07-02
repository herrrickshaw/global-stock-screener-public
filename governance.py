#!/usr/bin/env python3
# governance.py
# =============
# TOGAF Phase-G Architecture Governance, made executable: verify that the project's
# architecture principles (see TOGAF.md) are actually upheld by enforceable controls.
# Run in CI / pre-commit so non-conformances are caught before they land.
#
#   python governance.py            # print a compliance report; exit 1 on any breach
#   python governance.py --quiet
#
# Each check maps to a principle (P1–P10) in TOGAF.md.

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).parent


def _tracked() -> List[str]:
    return subprocess.run(
        ["git", "-C", str(HERE), "ls-files"], capture_output=True, text=True
    ).stdout.splitlines()


def _check_integrity() -> Tuple[bool, str]:  # P4
    m = HERE / "cache_seed" / "CHECKSUMS.sha256"
    if not m.exists():
        return False, "CHECKSUMS.sha256 missing (run integrity.py --generate)"
    try:
        import integrity

        ok = integrity.verify(verbose=False) == 0
        return ok, "SHA-256 manifest verified" if ok else "integrity drift — files changed"
    except Exception as e:  # noqa: BLE001
        return False, f"integrity check errored: {str(e)[:40]}"


def _check_no_secrets() -> Tuple[bool, str]:  # P5
    tracked = _tracked()
    if ".env" in tracked:
        return False, ".env is TRACKED — secrets leak risk"
    import re

    pat = re.compile(r"(password|api[_-]?key|secret|token)\s*=\s*['\"][A-Za-z0-9]{12,}", re.I)
    hits = []
    for f in tracked:
        if f.endswith((".py", ".md", ".txt", ".yml", ".yaml", ".cfg", ".ini")):
            try:
                if pat.search((HERE / f).read_text(errors="ignore")):
                    hits.append(f)
            except Exception:
                pass
    return (not hits), (
        "no hard-coded secrets in tracked files" if not hits else f"possible secret in {hits}"
    )


def _check_dep_map() -> Tuple[bool, str]:  # P2
    return (
        (HERE / "DEPENDENCIES.md").exists(),
        (
            "dependency map present"
            if (HERE / "DEPENDENCIES.md").exists()
            else "DEPENDENCIES.md missing"
        ),
    )


def _check_tests() -> Tuple[bool, str]:  # P6
    n = len(list((HERE / "tests").glob("test_*.py"))) if (HERE / "tests").exists() else 0
    return (n >= 5, f"{n} test modules present")


def _check_ci_governance() -> Tuple[bool, str]:  # P10
    wf = HERE / ".github" / "workflows"
    have = {p.name for p in wf.glob("*.yml")} if wf.exists() else set()
    need = {"tests.yml", "integrity.yml"}
    missing = need - have
    return (
        not missing,
        "CI governance workflows present" if not missing else f"missing workflow(s): {missing}",
    )


def _check_docs() -> Tuple[bool, str]:  # P9 explainability / EA content
    need = ["TOGAF.md", "DATA_SCHEMA_AND_ARCHITECTURE.md", "SECURITY.md", "DATA_SOURCES.md"]
    missing = [d for d in need if not (HERE / d).exists()]
    return (not missing, "EA content docs present" if not missing else f"missing docs: {missing}")


CHECKS = {
    "P4 Data integrity (integrity manifest)": _check_integrity,
    "P5 Security (no secrets in VCS)": _check_no_secrets,
    "P2 Single source of truth (dependency map)": _check_dep_map,
    "P6 Testability (test suite present)": _check_tests,
    "P10 Governance (CI conformance gates)": _check_ci_governance,
    "P9 Explainability (EA content docs)": _check_docs,
}


def run(verbose: bool = True) -> Dict[str, Tuple[bool, str]]:
    results = {name: fn() for name, fn in CHECKS.items()}
    if verbose:
        print("=== TOGAF Phase-G Architecture Governance ===")
        for name, (ok, msg) in results.items():
            print(f"  {'✓' if ok else '✗'} {name:44} {msg}")
        passed = sum(1 for ok, _ in results.values() if ok)
        print(
            f"  → {passed}/{len(results)} controls passed"
            f" — {'CONFORMANT ✓' if passed == len(results) else 'NON-CONFORMANCE ✗'}"
        )
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="TOGAF architecture-governance compliance check")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    results = run(verbose=not args.quiet)
    return 0 if all(ok for ok, _ in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
