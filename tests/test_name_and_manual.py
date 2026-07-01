# Tests for name→ticker matching and manual-input decisions. Pure logic, no network.
import name_matcher as nm
import portfolio_evaluator as pe


def test_clean_strips_corporate_noise():
    assert nm.clean("Reliance Industries Ltd") == "reliance"
    assert nm.clean("HDFC Bank Limited") == "hdfc bank"
    assert nm.clean("Tata Consultancy Services Pvt Ltd") == "tata consultancy"


def test_match_exact_contains_fuzzy(monkeypatch):
    idx = (
        {"reliance": "RELIANCE", "hdfc bank": "HDFCBANK", "bajaj finance": "BAJFINANCE"},
        {},
        ("reliance", "hdfc bank", "bajaj finance"),
    )
    monkeypatch.setattr(nm, "_index", lambda market: idx)
    assert nm.match("Reliance Industries Ltd", "IN")[0] == "RELIANCE"  # exact after clean
    assert nm.match("HDFC Bank", "IN")[0] == "HDFCBANK"  # contains
    assert nm.match("Bajaj Financ", "IN")[0] == "BAJFINANCE"  # fuzzy
    assert nm.match("Totally Unknown Co", "IN") is None


def test_resolve_reports_unmatched(monkeypatch):
    idx = ({"infosys": "INFY"}, {}, ("infosys",))
    monkeypatch.setattr(nm, "_index", lambda market: idx)
    hits, misses = nm.resolve(["Infosys Ltd", "Zzzz Corp"], "IN")
    assert hits["Infosys Ltd"][0] == "INFY"
    assert misses == ["Zzzz Corp"]


def test_row_from_manual_derives_drawdown():
    row, p, fund = pe._row_from_manual(
        {"price": 80, "pe": 15, "high_52w": 100, "low_52w": 70, "sma200": 90, "roe": 18}
    )
    assert round(row["PctFromHigh"], 0) == -20  # 80/100-1
    assert row["Above200DMA"] is False  # 80 < 90
    assert p == 15 and fund["roe"] == 18


def test_evaluate_manual_signals():
    out = pe.evaluate_manual(
        [
            {
                "ticker": "GOOD",
                "price": 120,
                "pe": 12,
                "high_52w": 125,
                "low_52w": 70,
                "sma200": 100,
                "ret_1y": 45,
                "roe": 22,
            },
            {
                "ticker": "TRAP",
                "price": 8,
                "pe": -3,
                "high_52w": 60,
                "low_52w": 7,
                "sma200": 30,
                "ret_1y": -75,
                "roe": -12,
                "liquidity": "Low",
            },
        ],
        med_pe=22,
        verbose=False,
    )
    sig = dict(zip(out["Ticker"], out["Signal"]))
    assert sig["GOOD"] == "BUY"
    assert sig["TRAP"] == "SELL"
    assert "IMPAIRED" in out[out["Ticker"] == "TRAP"]["Flags"].iloc[0]
