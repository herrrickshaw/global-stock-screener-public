# Tests for Trendlyne parsing — pure logic on a mocked SSR blob, no network/login.
import trendlyne_auth as tl

SAMPLE = (
    'junk <span>{"title": "PE TTM", "value": 21.9, "color": "g"} more '
    '{"title": "ROE Annual %", "value": 8.9} {"title": "Market Capitalization", "value": 1770056.1} '
    '{"title": "Debt to Equity", "value": 0.41} {"title": "Dividend Yield", "value": 0.5} '
    '{"title": "Some Text Metric", "value": "NA"}'
)


def test_parse_blobs():
    b = tl._parse_blobs(SAMPLE)
    assert b["PE TTM"] == 21.9
    assert b["ROE Annual %"] == 8.9
    assert b["Market Capitalization"] == 1770056.1
    assert "Some Text Metric" not in b  # non-numeric value skipped


def test_to_canonical_mapping():
    c = tl.to_canonical(tl._parse_blobs(SAMPLE))
    assert c["pe"] == 21.9
    assert c["roe"] == 8.9
    assert c["market_cap"] == 1770056.1
    assert c["debt_to_equity"] == 0.41
    assert c["div_yield"] == 0.5


def test_empty_page():
    assert tl._parse_blobs("no blobs here") == {}
    assert tl.to_canonical({}) == {}
