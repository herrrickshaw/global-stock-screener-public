# Tests for sector-rotation analytics — pure logic, no network.
import pandas as pd

import sector_map as sm


def _serving(rows):
    return pd.DataFrame(rows)


def test_rotation_ranks_and_phases():
    serving = _serving(
        [
            {"Symbol": "A", "Ret252": 40, "Ret63": 10, "Above200DMA": True},
            {"Symbol": "B", "Ret252": 30, "Ret63": 8, "Above200DMA": True},
            {"Symbol": "C", "Ret252": 35, "Ret63": 12, "Above200DMA": True},  # Tech: strong
            {"Symbol": "D", "Ret252": -20, "Ret63": -8, "Above200DMA": False},
            {"Symbol": "E", "Ret252": -15, "Ret63": -6, "Above200DMA": False},
            {"Symbol": "F", "Ret252": -25, "Ret63": -10, "Above200DMA": False},  # Metals: weak
        ]
    )
    smap = {"A": "Tech", "B": "Tech", "C": "Tech", "D": "Metals", "E": "Metals", "F": "Metals"}
    out = sm.rotation(serving, smap)
    tech = out[out.Sector == "Tech"].iloc[0]
    metals = out[out.Sector == "Metals"].iloc[0]
    assert tech.Breadth == 1.0 and "Boom" in tech.Phase
    assert metals.Phase == "Laggard"
    # ranked: Tech (higher breadth) before Metals
    assert out.iloc[0].Sector == "Tech"


def test_min_names_filter():
    serving = _serving([{"Symbol": "A", "Ret252": 10, "Ret63": 2, "Above200DMA": True}])
    out = sm.rotation(serving, {"A": "Solo"})
    assert out.empty  # < min_names (3) → excluded


def test_empty_inputs():
    assert sm.rotation(pd.DataFrame(), {}).empty
    assert sm.rotation(pd.DataFrame([{"Symbol": "A"}]), {}).empty
