import pytest

import emerging_fundamentals as ef


class R:
    def __init__(self, js=None, status=200):
        self._js, self.status_code = js, status

    def json(self):
        return self._js


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(ef, "_PROVIDERS", None)
    monkeypatch.setattr(ef.time, "sleep", lambda t: None)
    for k in ("FMP_API_KEY", "TWELVEDATA_KEY"):
        monkeypatch.delenv(k, raising=False)


# ── throttle ─────────────────────────────────────────────────────────────────────
def test_throttle(monkeypatch):
    slept = []
    clock = {"t": 100.0}
    monkeypatch.setattr(ef.time, "time", lambda: clock["t"])
    monkeypatch.setattr(ef.time, "sleep", lambda t: slept.append(round(t, 3)))
    monkeypatch.setitem(ef._LAST_CALL, "t", 99.9)  # 0.1s ago -> sleep 0.1
    ef._throttle()
    assert slept == [0.1]
    assert ef._LAST_CALL["t"] == 100.0
    clock["t"] = 100.2  # exactly the interval: no sleep
    ef._throttle()
    assert slept == [0.1]
    clock["t"] = 105.0
    ef._throttle()
    assert slept == [0.1]


# ── _g ───────────────────────────────────────────────────────────────────────────
def test_g():
    assert ef._g({"a": "3.5"}, "a") == 3.5
    assert ef._g({"a": 0}, "a") == 0.0
    assert ef._g({"a": None}, "a") is None
    assert ef._g({"a": "x"}, "a") is None
    assert ef._g({"a": [1]}, "a") is None
    assert ef._g({}, "a") is None
    assert ef._g(None, "a") is None


# ── piotroski ────────────────────────────────────────────────────────────────────
def _inc(ni=10, rev=100, gp=40, ebit=20, sh=10):
    return {"net_income": ni, "revenue": rev, "gross_profit": gp, "ebit": ebit, "shares": sh}


def _bal(ta=100, ltd=10, ca=50, cl=25):
    return {
        "total_assets": ta,
        "long_term_debt": ltd,
        "current_assets": ca,
        "current_liabilities": cl,
    }


def _cf(ocf=15, fcf=10, capex=-5):
    return {"operating_cash_flow": ocf, "free_cash_flow": fcf, "capex": capex}


def test_piotroski_perfect():
    inc = [_inc(ni=20, rev=200, gp=100, sh=9), _inc(ni=5, rev=100, gp=40, sh=10)]
    bal = [_bal(ta=100, ltd=5, ca=60, cl=20), _bal(ta=100, ltd=20, ca=50, cl=25)]
    assert ef._piotroski(inc, bal, [_cf(ocf=30)]) == 9


def test_piotroski_each_point_off():
    inc1 = [_inc(), _inc()]
    bal1 = [_bal(), _bal()]
    # identical years: f1 (roa>0)=1, f2 (ocf>0)=1, f3 0, f4 (ocf/a > roa) 1, f5 0 (not <),
    # f6 0, f7 (sh0<=sh1)=1, f8 0, f9 0
    assert ef._piotroski(inc1, bal1, [_cf(ocf=15)]) == 4
    # no prior year: f7 defaults to 1 when shares missing
    assert ef._piotroski([_inc()], [_bal()], [_cf(ocf=15)]) == 3 + 1  # f1,f2,f4,f7
    assert ef._piotroski([_inc()], [_bal()], []) == 2  # f1 + f7 (no cf)
    # losses
    assert ef._piotroski([_inc(ni=-5)], [_bal()], [_cf(ocf=-1)]) == 2  # f4 (ocf/a > roa) + f7
    # shares dilution
    assert ef._piotroski([_inc(sh=11), _inc(sh=10)], [_bal(), _bal()], [_cf(ocf=0)]) == 1
    # f4 boundary: ocf/a == roa is not >
    assert ef._piotroski([_inc(ni=10)], [_bal(ta=100)], [_cf(ocf=10)]) == 3  # f1,f2,f7


def test_piotroski_f3_f5_f6_f8_f9_individually():
    base_inc = [_inc(ni=10, rev=100, gp=40, sh=10), _inc(ni=10, rev=100, gp=40, sh=10)]
    base_bal = [_bal(), _bal()]
    base = ef._piotroski(base_inc, base_bal, [_cf(ocf=0)])
    # f3: roa up
    inc = [_inc(ni=20), _inc(ni=10)]
    assert ef._piotroski(inc, base_bal, [_cf(ocf=0)]) == base + 1
    # f5: leverage down
    bal = [_bal(ltd=5), _bal(ltd=10)]
    assert ef._piotroski(base_inc, bal, [_cf(ocf=0)]) == base + 1
    # f6: current ratio up
    bal = [_bal(ca=80), _bal(ca=50)]
    assert ef._piotroski(base_inc, bal, [_cf(ocf=0)]) == base + 1
    # f8: gross margin up
    inc = [_inc(gp=50), _inc(gp=40)]
    assert ef._piotroski(inc, base_bal, [_cf(ocf=0)]) == base + 1
    # f9: asset turnover up
    inc = [_inc(rev=150), _inc(rev=100)]
    assert ef._piotroski(inc, base_bal, [_cf(ocf=0)]) == base + 1


def test_piotroski_raises_on_empty():
    with pytest.raises(IndexError):
        ef._piotroski([], [_bal()], [])


# ── coffee can ───────────────────────────────────────────────────────────────────
def test_coffee_can_all_pass():
    inc = [
        _inc(rev=200, gp=100, ebit=50),
        _inc(rev=150, gp=60, ebit=45),
        _inc(rev=100, gp=30, ebit=40),
    ]
    bal = [_bal(ta=200, ltd=20, cl=100)] * 3
    out = ef._coffee_can(inc, bal, [_cf(fcf=5)])
    assert out["cc_qualifies"] == "PASS" and out["cc_score"] == "6/6"
    assert out["cc_rev_cagr"] == pytest.approx(round(((200 / 100) ** 0.5 - 1) * 100, 2))
    assert out["cc_roce_avg"] == pytest.approx(
        round(sum([50 / 100, 45 / 100, 40 / 100]) / 3 * 100, 2)
    )
    assert out["cc_debt_equity"] == 0.2


def test_coffee_can_each_criterion_fails():
    inc = [_inc(rev=105, gp=10, ebit=1, ni=-1), _inc(rev=100, gp=40, ebit=1, ni=5)]
    bal = [_bal(ta=100, ltd=200, cl=20)] * 2
    out = ef._coffee_can(inc, bal, [_cf(fcf=-1)])
    assert out["cc_qualifies"] == "FAIL" and out["cc_score"] == "0/6"
    assert out["cc_debt_equity"] == 2.5


def test_coffee_can_thresholds():
    # C1: cagr exactly 10 fails, >10 passes (two points -> 1 period)
    def c1(r0):
        return ef._coffee_can([_inc(rev=r0), _inc(rev=100)], [_bal(), _bal()], [])["cc_score"]

    # compare C1 via cc_rev_cagr values
    assert ef._coffee_can([_inc(rev=110), _inc(rev=100)], [_bal()] * 2, [])["cc_rev_cagr"] == 10.0
    s9 = int(c1(109).split("/")[0])
    s11 = int(c1(111).split("/")[0])
    assert s11 == s9 + 1

    # C3: de == 1.0 fails, < 1.0 passes
    def c3(ltd):
        return ef._coffee_can([_inc()], [_bal(ta=100, ltd=ltd, cl=20)], [])["cc_debt_equity"]

    assert c3(80) == 1.0
    b = ef._coffee_can([_inc(ni=0)], [_bal(ta=100, ltd=80, cl=20)], [])
    b2 = ef._coffee_can([_inc(ni=0)], [_bal(ta=100, ltd=79, cl=20)], [])
    assert int(b2["cc_score"].split("/")[0]) == int(b["cc_score"].split("/")[0]) + 1
    # C2: roce avg exactly 15 fails
    r15 = ef._coffee_can([_inc(ebit=12)], [_bal(ta=100, cl=20)], [])
    r16 = ef._coffee_can([_inc(ebit=12.1)], [_bal(ta=100, cl=20)], [])
    assert int(r16["cc_score"].split("/")[0]) == int(r15["cc_score"].split("/")[0]) + 1


def test_coffee_can_fcf_fallback_and_missing():
    base = ef._coffee_can([_inc()], [_bal()], [{"operating_cash_flow": 10, "capex": -4}])
    nofcf = ef._coffee_can([_inc()], [_bal()], [{"operating_cash_flow": 10}])
    assert int(base["cc_score"].split("/")[0]) == int(nofcf["cc_score"].split("/")[0]) + 1
    neg = ef._coffee_can([_inc()], [_bal()], [{"operating_cash_flow": 3, "capex": 4}])
    assert int(neg["cc_score"].split("/")[0]) == int(nofcf["cc_score"].split("/")[0])
    # explicit fcf wins over ocf-capex
    win = ef._coffee_can(
        [_inc()], [_bal()], [{"free_cash_flow": 1, "operating_cash_flow": 0, "capex": 5}]
    )
    assert int(win["cc_score"].split("/")[0]) == int(base["cc_score"].split("/")[0])


def test_coffee_can_rounding_and_none():
    out = ef._coffee_can([_inc(ebit=None, ni=None)], [_bal(ltd=None)], [])
    assert (
        out["cc_rev_cagr"] is None and out["cc_roce_avg"] is None and out["cc_debt_equity"] is None
    )
    out = ef._coffee_can([_inc(ebit=10)], [_bal(ta=300, cl=0, ltd=0)], [])
    assert out["cc_roce_avg"] == 3.33
    assert out["cc_debt_equity"] is None  # de == 0 -> falsy -> None


def test_coffee_can_negative_oldest_revenue():
    inc = [_inc(rev=100), _inc(rev=-5)]
    assert ef._coffee_can(inc, [_bal()] * 2, [])["cc_rev_cagr"] is None


def test_coffee_can_margin_trend_c4():
    up = ef._coffee_can([_inc(gp=50), _inc(gp=40)], [_bal()] * 2, [])
    dn = ef._coffee_can([_inc(gp=40), _inc(gp=50)], [_bal()] * 2, [])
    eq = ef._coffee_can([_inc(gp=40), _inc(gp=40)], [_bal()] * 2, [])
    n = lambda o: int(o["cc_score"].split("/")[0])  # noqa: E731
    assert n(up) == n(dn) + 1 == n(eq) + 1


# ── score_from_statements ────────────────────────────────────────────────────────
def test_score_no_statements():
    r = ef.score_from_statements([], [_bal()], [], {"sector": "S", "pe": 12.345})
    assert r["error"] == "no_financial_statements" and r["f_score"] is None
    assert r["sector"] == "S" and r["pe_ratio"] == 12.35
    assert ef.score_from_statements([_inc()], [], [], None)["error"] == "no_financial_statements"


def test_score_pe_rules():
    f = lambda pe: ef.score_from_statements([], [], [], {"pe": pe})["pe_ratio"]  # noqa: E731
    assert f(None) is None and f(0) is None and f(-3) is None
    assert f("7.5") == 7.5
    assert f(0.001) == 0.0


def test_score_full():
    r = ef.score_from_statements([_inc()], [_bal()], [_cf()], {"pe": 10, "sector": "Tech"})
    assert r["error"] is None
    assert r["f_score"] == ef._piotroski([_inc()], [_bal()], [_cf()])
    assert r["cc_score"].endswith("/6")
    assert r["sector"] == "Tech"


def test_score_scorer_exceptions_swallowed(monkeypatch):
    def boom(*a):
        raise ValueError

    monkeypatch.setattr(ef, "_piotroski", boom)
    monkeypatch.setattr(ef, "_coffee_can", boom)
    r = ef.score_from_statements([_inc()], [_bal()], [], {})
    assert r["f_score"] is None and r["cc_score"] is None and r["cc_qualifies"] == "FAIL"


# ── FMP provider ─────────────────────────────────────────────────────────────────
def _requests(monkeypatch, handler):
    calls = []

    def get(url, params=None, headers=None, timeout=None):
        calls.append((url, dict(params), headers, timeout))
        return handler(url, params)

    monkeypatch.setattr(ef.requests, "get", get)
    return calls


def test_fmp_available(monkeypatch):
    assert not ef._FMP().available
    monkeypatch.setenv("FMP_API_KEY", "  k  ")
    p = ef._FMP()
    assert p.available and p.key == "k"
    monkeypatch.setenv("FMP_API_KEY", "   ")
    assert not ef._FMP().available


def test_fmp_get(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "k")
    p = ef._FMP()
    calls = _requests(monkeypatch, lambda u, pr: R([1]))
    assert p._get("x/Y", period="annual") == [1]
    assert calls[0][0] == f"{p.base}/x/Y"
    assert calls[0][1] == {"period": "annual", "apikey": "k"}
    assert calls[0][3] == ef._TIMEOUT
    _requests(monkeypatch, lambda u, pr: R([1], status=429))
    assert p._get("x") is None
    _requests(monkeypatch, lambda u, pr: R({"Error Message": "bad"}))
    assert p._get("x") is None
    _requests(monkeypatch, lambda u, pr: R({"ok": 1}))
    assert p._get("x") == {"ok": 1}


FMP_DATA = {
    "income-statement": [
        {"netIncome": 5, "revenue": 100, "grossProfit": 40, "operatingIncome": 0, "ebitda": 9,
         "weightedAverageShsOut": 7},
    ],
    "balance-sheet-statement": [
        {"totalAssets": 100, "longTermDebt": 5, "totalCurrentAssets": 50, "totalCurrentLiabilities": 20},
    ],
    "cash-flow-statement": [
        {"operatingCashFlow": 8, "freeCashFlow": 3, "capitalExpenditure": -5},
    ],
    "profile": [{"sector": "Industrials"}],
    "quote": [{"pe": 11.0}],
}  # fmt: skip


def _fmp_handler(data):
    def h(url, params):
        for k, v in data.items():
            if f"/{k}" in url:
                return R(v)
        return R(None, status=404)

    return h


def test_fmp_fetch(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "k")
    calls = _requests(monkeypatch, _fmp_handler(FMP_DATA))
    inc, bal, cf, prof = ef._FMP().fetch("AKBNK.IS")
    assert inc == [
        {"net_income": 5, "revenue": 100, "gross_profit": 40, "ebit": 9, "shares": 7}
    ]  # ebitda fallback
    assert bal[0]["total_assets"] == 100 and bal[0]["current_liabilities"] == 20
    assert cf == [{"operating_cash_flow": 8, "free_cash_flow": 3, "capex": -5}]
    assert prof == {"pe": 11.0, "sector": "Industrials"}
    assert calls[0][1]["limit"] == 5 and calls[0][1]["period"] == "annual"


def test_fmp_fetch_missing_parts(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "k")
    d = {k: v for k, v in FMP_DATA.items() if k not in ("cash-flow-statement", "profile", "quote")}
    _requests(monkeypatch, _fmp_handler(d))
    inc, bal, cf, prof = ef._FMP().fetch("X")
    assert cf == [] and prof == {"pe": None, "sector": None}
    for drop in ("income-statement", "balance-sheet-statement"):
        d = {k: v for k, v in FMP_DATA.items() if k != drop}
        _requests(monkeypatch, _fmp_handler(d))
        assert ef._FMP().fetch("X") is None


# ── Twelve Data provider ─────────────────────────────────────────────────────────
def test_td_get(monkeypatch):
    monkeypatch.setenv("TWELVEDATA_KEY", "t")
    p = ef._TwelveData()
    assert p.available
    calls = _requests(monkeypatch, lambda u, pr: R({"a": 1}))
    assert p._get("quote", symbol="X") == {"a": 1}
    assert calls[0][0] == f"{p.base}/quote" and calls[0][1] == {"symbol": "X", "apikey": "t"}
    _requests(monkeypatch, lambda u, pr: R({}, status=500))
    assert p._get("quote") is None
    _requests(monkeypatch, lambda u, pr: R({"code": 400, "status": "error"}))
    assert p._get("quote") is None
    _requests(monkeypatch, lambda u, pr: R({"code": 200, "status": "ok"}))
    assert p._get("quote") == {"code": 200, "status": "ok"}
    _requests(monkeypatch, lambda u, pr: R({"status": "error"}))  # no code -> passes through
    assert p._get("quote") == {"status": "error"}


TD_DATA = {
    "income_statement": {"income_statement": [
        {"net_income": 5, "sales": 100, "revenue": 1, "gross_profit": 40, "operating_income": 12,
         "basic_shares_outstanding": 7},
        {"net_income": 4, "revenue": 90},
    ]},
    "balance_sheet": {"balance_sheet": [
        {"assets": {"total_assets": 100}, "long_term_debt": 5, "total_current_assets": 50,
         "total_current_liabilities": 20},
        {"total_assets": 80},
    ]},
    "cash_flow": {"cash_flow": [
        {"operating_cash_flow": 8, "free_cash_flow": 3, "capital_expenditures": -5},
    ]},
    "quote": {"pe": 0},
    "profile": {"pe": 13.0, "sector": "Energy"},
}  # fmt: skip


def _td_handler(data):
    def h(url, params):
        path = url.rsplit("/", 1)[1]
        return R(data[path]) if path in data else R({}, status=404)

    return h


def test_td_fetch(monkeypatch):
    monkeypatch.setenv("TWELVEDATA_KEY", "t")
    _requests(monkeypatch, _td_handler(TD_DATA))
    inc, bal, cf, prof = ef._TwelveData().fetch("X")
    assert inc[0] == {"net_income": 5, "revenue": 100, "gross_profit": 40, "ebit": 12, "shares": 7}
    assert inc[1]["revenue"] == 90
    assert bal[0]["total_assets"] == 100 and bal[1]["total_assets"] == 80
    assert cf == [{"operating_cash_flow": 8, "free_cash_flow": 3, "capex": -5}]
    assert prof == {"pe": 13.0, "sector": "Energy"}  # quote pe 0 -> profile pe


def test_td_fetch_missing(monkeypatch):
    monkeypatch.setenv("TWELVEDATA_KEY", "t")
    d = {k: v for k, v in TD_DATA.items() if k not in ("cash_flow", "quote", "profile")}
    _requests(monkeypatch, _td_handler(d))
    inc, bal, cf, prof = ef._TwelveData().fetch("X")
    assert cf == [] and prof == {"pe": None, "sector": None}
    for drop in ("income_statement", "balance_sheet"):
        d = {k: v for k, v in TD_DATA.items() if k != drop}
        _requests(monkeypatch, _td_handler(d))
        assert ef._TwelveData().fetch("X") is None
    d = dict(TD_DATA, income_statement={"income_statement": []})
    _requests(monkeypatch, _td_handler(d))
    assert ef._TwelveData().fetch("X") is None


# ── provider selection / get_fundamentals ────────────────────────────────────────
def test_providers_none_and_cached(monkeypatch):
    assert ef._providers() == []
    assert ef.available_source() is None
    assert ef.get_fundamentals("X") is None
    monkeypatch.setenv("FMP_API_KEY", "k")
    assert ef._providers() == []  # cached empty list


def test_providers_order(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "k")
    monkeypatch.setenv("TWELVEDATA_KEY", "t")
    assert [p.name for p in ef._providers()] == ["FMP", "TwelveData"]
    assert ef.available_source() == "FMP"


def test_providers_only_td(monkeypatch):
    monkeypatch.setenv("TWELVEDATA_KEY", "t")
    assert ef.available_source() == "TwelveData"


class _P:
    def __init__(self, name, result=None, exc=None):
        self.name, self.result, self.exc = name, result, exc
        self.available = True

    def fetch(self, t):
        if self.exc:
            raise self.exc
        return self.result


def test_get_fundamentals_fallthrough(monkeypatch):
    good = ([_inc()], [_bal()], [_cf()], {"pe": 9, "sector": "S"})
    bad_scored = ([], [], [], {})  # f_score None -> next provider
    monkeypatch.setattr(
        ef,
        "_PROVIDERS",
        [
            _P("A", exc=RuntimeError()),
            _P("B", None),
            _P("C", bad_scored),
            _P("D", good),
            _P("E", good),
        ],
    )
    out = ef.get_fundamentals("X")
    assert out["source"] == "D" and out["f_score"] is not None and out["sector"] == "S"
    monkeypatch.setattr(ef, "_PROVIDERS", [_P("C", bad_scored)])
    assert ef.get_fundamentals("X") is None
