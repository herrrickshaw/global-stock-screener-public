# Test delivery-% accumulation wired into the decision engine — pure logic.
import decision_engine as de

BASE = {
    "Close": 100,
    "SMA200": 98,
    "Above200DMA": True,
    "GoldenCross": True,
    "Ret252": 20,
    "Ret63": 5,
    "PctFromHigh": -6,
    "PctFromLow": 40,
    "Liquidity": "High",
}


def test_accumulation_lifts_and_flags():
    plain = de.decide(BASE, pe=18, med_pe=20)
    acc = de.decide({**BASE, "accum_score": 1.5}, pe=18, med_pe=20)
    assert acc["composite"] >= plain["composite"]
    assert "ACCUMULATION" in acc["flags"]


def test_distribution_penalises_and_flags():
    dis = de.decide({**BASE, "accum_score": 0.6}, pe=18, med_pe=20)
    assert "DISTRIBUTION" in dis["flags"]
    assert dis["composite"] <= de.decide(BASE, pe=18, med_pe=20)["composite"]


def test_neutral_accum_no_flag():
    d = de.decide({**BASE, "accum_score": 1.0}, pe=18, med_pe=20)
    assert "ACCUMULATION" not in d["flags"] and "DISTRIBUTION" not in d["flags"]
