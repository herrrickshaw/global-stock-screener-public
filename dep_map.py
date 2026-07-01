#!/usr/bin/env python3
# dep_map.py
# ==========
# Auto-generate the repository's interdependency map — for BOTH code and data — so a
# change's blast radius is obvious and bugs are faster to localise.
#
#   CODE:  who-imports-whom (forward) + who-breaks-if-I-change-X (reverse/impact)
#   DATA:  which module PRODUCES vs CONSUMES each data artifact (cache_seed / reference_seed)
#
#   python3 dep_map.py            # regenerate DEPENDENCIES.md
#   python3 dep_map.py --print    # also print a summary to stdout
#
# Static analysis (ast for imports; regex for data-file references + write ops), so
# it's deterministic and safe to run in CI / pre-commit.

from __future__ import annotations

import argparse
import ast
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

HERE = Path(__file__).parent
OUT = HERE / "DEPENDENCIES.md"

# data artifacts we track (filename fragment → friendly label)
DATA_ARTIFACTS = {
    "cleaned_long": "STM seeds (1y)",
    "ltm/": "LTM seeds (5y)",
    "ohlcv.lmdb": "LMDB store",
    "assembled_long": "assembled raw (IN)",
    "serving/": "serving views",
    "cdc/": "CDC delta log",
    "fundamentals/": "fundamentals cache",
    "public_screens/": "public screens cache",
    "india_ccc_screen": "India CCC screen",
    "liquidity_index": "liquidity index",
    "company_list": "company list",
    "data_manifest": "data manifest",
    "pipeline_state": "pipeline registry",
    "models/": "ML models",
    "discovered_screens": "discovered screens",
    "CHECKSUMS": "integrity manifest",
    "reference_seed": "reference (Damodaran/French)",
    "nse_holidays": "NSE holidays",
    "market_performance": "market performance",
}
WRITE_HINTS = (
    "to_parquet",
    "to_csv",
    ".write_text",
    ".write(",
    "json.dump",
    "joblib.dump",
    "txn.put",
    ".to_excel",
    "open(",
    "df.to_",
    "write_seed",
)


def _modules() -> List[str]:
    out = subprocess.run(
        ["git", "-C", str(HERE), "ls-files", "*.py"], capture_output=True, text=True
    ).stdout.split()
    return [f for f in out if not f.startswith("tests/")]


def _local_names(mods: List[str]) -> Set[str]:
    names = set()
    for m in mods:
        p = Path(m)
        names.add(p.stem if p.parent == Path(".") else p.parent.name)
    return names


def code_graph(mods: List[str]):
    local = _local_names(mods)
    fwd: Dict[str, Set[str]] = defaultdict(set)
    for m in mods:
        mod = Path(m).stem
        try:
            tree = ast.parse((HERE / m).read_text())
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    top = a.name.split(".")[0]
                    if top in local and top != mod:
                        fwd[mod].add(top)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                if top in local and top != mod:
                    fwd[mod].add(top)
    rev: Dict[str, Set[str]] = defaultdict(set)
    for m, deps in fwd.items():
        for d in deps:
            rev[d].add(m)
    return fwd, rev


READ_HINTS = (
    "read_parquet",
    "read_csv",
    "read_text",
    "read_excel",
    "pd.read",
    "joblib.load",
    "feather.read",
    "glob(",
    ".exists(",
    "read_seed",
    '"rb"',
    "'rb'",
)


def data_graph(mods: List[str]):
    produces: Dict[str, Set[str]] = defaultdict(set)  # artifact → modules that write it
    consumes: Dict[str, Set[str]] = defaultdict(set)  # artifact → modules that read it
    assign = re.compile(r"^\s*([A-Za-z_][\w, ]*?)\s*=\s*(.+)$")  # incl. tuple targets
    for m in mods:
        mod = Path(m).stem
        try:
            lines = (HERE / m).read_text().splitlines()
        except Exception:
            continue
        # 1) map module-level path variables → artifact label (e.g. CLEANED = ... "cleaned_long")
        var2label: Dict[str, str] = {}
        for line in lines:
            am = assign.match(line)
            if not am:
                continue
            targets, rhs = am.group(1), am.group(2)
            for frag, label in DATA_ARTIFACTS.items():
                if frag in rhs:
                    for var in re.findall(r"[A-Za-z_]\w*", targets):
                        var2label[var] = label
        # 2) attribute reads/writes: a line referencing a path var or literal fragment
        for line in lines:
            hit = None
            for frag, label in DATA_ARTIFACTS.items():
                if frag in line:
                    hit = label
                    break
            if hit is None:
                for var, label in var2label.items():
                    if re.search(rf"\b{var}\b", line):
                        hit = label
                        break
            if hit is None:
                continue
            if any(h in line for h in WRITE_HINTS):
                produces[hit].add(mod)
            elif any(h in line for h in READ_HINTS):
                consumes[hit].add(mod)
    return produces, consumes


def _table(rows):
    return "\n".join(rows)


def build(verbose: bool = False) -> None:
    mods = _modules()
    fwd, rev = code_graph(mods)
    produces, consumes = data_graph(mods)

    lines = [
        "# Interdependency map (code + data)",
        "",
        "_Auto-generated by `dep_map.py` — do not edit by hand; run `python dep_map.py`._",
        "",
        "Use it to size a change's blast radius before editing. The data section is a",
        "static heuristic (path constants + read/write ops); dynamically-built paths may be",
        "missed — treat §1/§2 (imports) as exact.",
        "",
        "## 1. Code — impact (who breaks if you change a module)",
        "",
        "| Module | Directly imported by (fix these if you change it) |",
        "|--------|----------------------------------------------------|",
    ]
    for mod in sorted(rev):
        importers = ", ".join(f"`{x}`" for x in sorted(rev[mod]))
        lines.append(f"| `{mod}` | {importers} |")
    leaves = sorted(set(Path(m).stem for m in mods) - set(rev))
    lines += [
        "",
        "**Leaf modules** (nothing imports them — safe to change in isolation): "
        + ", ".join(f"`{x}`" for x in leaves),
        "",
    ]

    lines += [
        "## 2. Code — dependencies (what each module needs)",
        "",
        "| Module | Imports (local) |",
        "|--------|-----------------|",
    ]
    for mod in sorted(fwd):
        lines.append(f"| `{mod}` | " + ", ".join(f"`{x}`" for x in sorted(fwd[mod])) + " |")

    lines += [
        "",
        "## 3. Data — producers → consumers",
        "",
        "| Data artifact | Produced by | Consumed by |",
        "|---------------|-------------|-------------|",
    ]
    for label in sorted(set(produces) | set(consumes)):
        p = ", ".join(f"`{x}`" for x in sorted(produces.get(label, []))) or "—"
        c = ", ".join(f"`{x}`" for x in sorted(consumes.get(label, []))) or "—"
        lines.append(f"| {label} | {p} | {c} |")

    lines += [
        "",
        "## How to use",
        "",
        "- **Changing code?** Check §1 for the module's importers — those are the files to",
        "  re-test. Leaf modules are safe to change alone.",
        "- **Bad/at data?** §3 shows which module *wrote* it (start debugging there) and",
        "  which *read* it (what's affected downstream).",
        "",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    if verbose:
        print(
            f"  wrote {OUT.name}: {len(mods)} modules, "
            f"{len(produces)|len(consumes)} data artifacts"
        )
        hot = sorted(rev.items(), key=lambda kv: -len(kv[1]))[:5]
        print("  most-depended-on modules:")
        for mod, imp in hot:
            print(f"    {mod:18} ← {len(imp)} importers")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the code+data interdependency map")
    ap.add_argument("--print", action="store_true", dest="show")
    build(verbose=ap.parse_args().show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
