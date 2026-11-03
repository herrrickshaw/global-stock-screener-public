"""Helpers shared by the Europe/Japan/Korea market-scan scripts (pure or I/O at the edge)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

SheetSpec = tuple[list[dict], str, "str | None"]


def trend_signal(dist_ma, fallback: str) -> str:
    """Label the % distance from the 200-day MA (None/0 -> `fallback`; +-5% is the band)."""
    return (
        "Above 200MA (Uptrend)"
        if dist_ma and dist_ma > 5
        else (
            "Below 200MA (Downtrend)"
            if dist_ma and dist_ma < -5
            else "Near 200MA (Consolidation)" if dist_ma else fallback
        )
    )


def write_scan_workbook(
    path: Path,
    sheets: Sequence[SheetSpec],
    style_sheet: Callable | None = None,
) -> None:
    """Write (rows, sheet_name, sort_col) specs to an .xlsx; sort descending, style if given."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for rows, name, sort_col in sheets:
            if not rows:
                pd.DataFrame().to_excel(writer, sheet_name=name, index=False)
                continue
            df = pd.DataFrame(rows)
            if sort_col and sort_col in df.columns:
                df = df.sort_values(sort_col, ascending=False)
            df.to_excel(writer, sheet_name=name, index=False)
            if style_sheet is not None:
                style_sheet(writer.sheets[name])
