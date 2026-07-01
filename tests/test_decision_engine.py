# Tests for the multi-factor Buy/Hold/Sell engine — pure logic, no network.
import decision_engine as de

HEALTHY = {
    "Close": 120,
    "SMA200": 100,
    "Above200DMA": True,
    "GoldenCross": True,
    "Ret252": 45,
    "Ret63": 12,
    "PctFromHigh": -3,
    "PctFromLow": 60,
    "Liquidity": "High",
}
BROKEN = {
    "Close": 10,
    "SMA200": 40,
    "Above200DMA": False,
    "GoldenCross": False,
    "Ret252": -72,
    "Ret63": -25,
    "PctFromHigh": -85,
    "PctFromLow": 3,
    "Liquidity": "Low",
}


def test_valuation_score_cheaper_is_higher():
    assert de.valuation_score(10, 20) > de.valuation_score(20, 20) > de.valuation_score(40, 20)
    assert de.valuation_score(-5, 20) <= 10  # losses score low
    assert de.valuation_score(None, 20) == 40  # unknown → neutral-low


def test_recovery_low_for_broken_high_for_healthy():
    assert de.recovery_score(BROKEN, pe=-4, fund={"roe": -15}) < 35
    assert de.recovery_score(HEALTHY, pe=14, fund={"roe": 24}) > 80


def test_impaired_detection():
    assert (
        de.is_impaired(
            BROKEN, pe=-4, fund={"roe": -15}, rec=de.recovery_score(BROKEN, -4, {"roe": -15})
        )
        is True
    )
    assert (
        de.is_impaired(
            HEALTHY, pe=14, fund={"roe": 24}, rec=de.recovery_score(HEALTHY, 14, {"roe": 24})
        )
        is False
    )


def test_decide_signals():
    imp = de.decide(BROKEN, pe=-4, med_pe=20, fundamentals={"roe": -15})
    assert imp["signal"] == "SELL" and "IMPAIRED" in imp["flags"]
    assert "recover" in imp["rationale"].lower()

    buy = de.decide(
        HEALTHY,
        pe=14,
        med_pe=22,
        fundamentals={"roe": 24, "roce": 22, "piotroski": 8, "debt_to_equity": 0.2},
    )
    assert buy["signal"] == "BUY" and not buy["flags"]

    # expensive + weak trend → SELL, no impaired flag (it's just overvalued)
    rich = {**HEALTHY, "Above200DMA": False, "GoldenCross": False, "Ret252": -5}
    r = de.decide(rich, pe=80, med_pe=20, fundamentals={"roe": 10})
    assert r["signal"] == "SELL"


def test_quality_none_without_fundamentals():
    d = de.decide(HEALTHY, pe=15, med_pe=20, fundamentals=None)
    assert d["quality"] is None  # renormalised without quality
    assert d["signal"] in ("BUY", "HOLD", "SELL")
