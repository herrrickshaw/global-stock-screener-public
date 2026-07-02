# Tests for F-04: DVM durability/momentum + promoter-pledge in the decision engine.
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


def test_durability_lifts_quality():
    base = de.quality_score({"roe": 10})
    with_dur = de.quality_score({"roe": 10, "dvm_durability": 90})
    assert with_dur > base  # high durability raises quality


def test_dvm_momentum_blends_into_momentum():
    plain = de.momentum_score(HEALTHY)
    boosted = de.momentum_score({**HEALTHY, "dvm_momentum": 100})
    weak = de.momentum_score({**HEALTHY, "dvm_momentum": 0})
    assert weak < plain < boosted or weak < boosted  # DVM momentum shifts the score


def test_high_pledge_flags_and_vetoes_buy():
    # a cheap, uptrending stock would be BUY; heavy pledge de-risks it to HOLD
    d = de.decide(HEALTHY, pe=8, med_pe=30, fundamentals={"roe": 20, "promoter_pledge": 60})
    assert any("PLEDGE" in f for f in d["flags"])
    assert d["signal"] == "HOLD"


def test_moderate_pledge_flags_only():
    d = de.decide(HEALTHY, pe=8, med_pe=30, fundamentals={"roe": 20, "promoter_pledge": 30})
    assert any("PLEDGE-30%" == f for f in d["flags"])
    assert d["signal"] == "BUY"  # 30% flagged but not vetoed (<50)


def test_no_pledge_no_flag():
    d = de.decide(HEALTHY, pe=8, med_pe=30, fundamentals={"roe": 20})
    assert not any("PLEDGE" in f for f in d["flags"])
