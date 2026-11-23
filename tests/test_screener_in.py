"""Offline tests for screener_in.py: header parsing, paginated screen scraping and the CCC helpers.

``requests`` inside the module is replaced by a fake whose pages are canned HTML, so nothing touches
screener.in."""

import types

import pandas as pd
import pytest

import screener_in as si


def row(sym, *cells):
    """A table row; like screener.in, the first cell holds the /company/<SYM>/ link (the name)."""
    tds = [f"<td>{c}</td>" for c in cells]
    if sym and tds:
        tds[0] = f'<td><a href="/company/{sym}/">{cells[0]}</a></td>'
    return f"<tr>{''.join(tds)}</tr>"


def page(rows, header=("Name", "CMP Rs.", "Cash Cycle")):
    ths = "".join(f"<th>{h}</th>" for h in header)
    return f"<table><tr>{ths}</tr>{''.join(rows)}</table>"


class Resp:
    def __init__(self, text="", status=200):
        self.text, self.status_code = text, status


@pytest.fixture
def web(monkeypatch):
    """pages: {page_no: Resp}; unknown pages -> 404. Records every request."""
    state = types.SimpleNamespace(pages={}, calls=[])

    def get(url, headers=None, timeout=None):
        state.calls.append((url, headers, timeout))
        n = int(url.rsplit("page=", 1)[1])
        return state.pages.get(n, Resp("", 404))

    monkeypatch.setattr(si, "requests", types.SimpleNamespace(get=get))
    return state


# ---- _header -------------------------------------------------------------------------------


def test_header_strips_markup_and_whitespace_and_drops_empty_cells():
    html = "<tr><th> S.No. </th><th><span>Cash\n   Cycle</span> <b>Days</b></th><th>  </th><th>P/E</th></tr>"
    assert si._header(html) == ["S.No.", "Cash Cycle Days", "P/E"]


def test_header_is_capped_at_fifteen_names():
    html = "".join(f"<th>h{i}</th>" for i in range(20))
    assert si._header(html) == [f"h{i}" for i in range(15)]


def test_header_with_no_th_is_empty():
    assert si._header("<td>x</td>") == []


# ---- fetch_screen --------------------------------------------------------------------------


def test_fetch_screen_builds_frame_with_symbol_first_and_unescapes_ampersand(web):
    web.pages[1] = Resp(
        page([row("AAA", "A &amp; B Ltd", "10.5", "-3"), row("BBB", "Beta", "20", "4")])
    )
    df = si.fetch_screen("https://x.test/screens/1/foo/", verbose=False)
    assert list(df.columns) == ["Symbol", "Name", "CMP Rs.", "Cash Cycle"]
    assert df["Symbol"].tolist() == ["AAA", "BBB"]
    assert df["Name"].tolist() == ["A & B Ltd", "Beta"]
    assert df["Cash Cycle"].tolist() == ["-3", "4"]


def test_fetch_screen_requests_each_page_with_ua_and_timeout(web):
    web.pages[1] = Resp(page([row("AAA", "n", "1", "2")]))
    web.pages[2] = Resp(page([row("BBB", "n", "1", "2")]))
    si.fetch_screen("https://x.test/s/", verbose=False)
    assert [c[0] for c in web.calls] == [
        "https://x.test/s/?page=1",
        "https://x.test/s/?page=2",
        "https://x.test/s/?page=3",  # 404 ends the walk
    ]
    assert web.calls[0][1] == si._UA and web.calls[0][2] == 25


def test_fetch_screen_stops_on_non_200_status(web):
    web.pages[1] = Resp(page([row("AAA", "n", "1", "2")]))
    web.pages[2] = Resp(page([row("BBB", "n", "1", "2")]), status=500)
    df = si.fetch_screen("https://x.test/s/", verbose=False)
    assert df["Symbol"].tolist() == ["AAA"]


def test_fetch_screen_stops_when_a_page_adds_no_new_symbols(web):
    same = Resp(page([row("AAA", "n", "1", "2")]))
    web.pages[1], web.pages[2], web.pages[3] = same, same, Resp(page([row("ZZZ", "n", "1", "2")]))
    df = si.fetch_screen("https://x.test/s/", verbose=False)
    assert df["Symbol"].tolist() == ["AAA"]  # page 2 repeats AAA -> break before page 3
    assert len(web.calls) == 2


def test_fetch_screen_respects_max_pages(web):
    for n in range(1, 6):
        web.pages[n] = Resp(page([row(f"S{n}", "n", "1", "2")]))
    df = si.fetch_screen("https://x.test/s/", max_pages=3, verbose=False)
    assert df["Symbol"].tolist() == ["S1", "S2", "S3"] and len(web.calls) == 3
    assert si.fetch_screen("https://x.test/s/", max_pages=0, verbose=False).empty


def test_fetch_screen_dedupes_symbols_within_and_across_pages(web):
    web.pages[1] = Resp(page([row("AAA", "first", "1", "2"), row("AAA", "again", "9", "9")]))
    web.pages[2] = Resp(page([row("AAA", "dup", "1", "2"), row("BBB", "n", "1", "2")]))
    df = si.fetch_screen("https://x.test/s/", verbose=False)
    assert df["Symbol"].tolist() == ["AAA", "BBB"] and df.loc[0, "Name"] == "first"


def test_fetch_screen_skips_header_median_and_cellless_rows(web):
    html = page(
        [
            row(None, "median", "1", "2"),  # no /company/ link
            '<tr><th><a href="/company/NOCELLS/">n</a></th></tr>',  # link but no <td> cells
            row("AAA", "n", "1", "2"),
        ]
    )
    web.pages[1] = Resp(html)
    assert si.fetch_screen("https://x.test/s/", verbose=False)["Symbol"].tolist() == ["AAA"]


def test_fetch_screen_symbol_charset_allows_punctuation_but_not_lowercase(web):
    web.pages[1] = Resp(
        page(
            [
                row("M&M", "n", "1", "2"),
                row("BAJAJ-AUTO", "n", "1", "2"),
                row("lower", "n", "1", "2"),
                row("A.B_C", "n", "1", "2"),
            ]
        )
    )
    df = si.fetch_screen("https://x.test/s/", verbose=False)
    assert df["Symbol"].tolist() == ["M&M", "BAJAJ-AUTO", "A.B_C"]


def test_fetch_screen_pads_short_rows_and_names_extra_columns(web):
    web.pages[1] = Resp(
        page(
            [row("AAA", "n", "1"), row("BBB", "n", "1", "2", "3", "4")], header=("Name", "CMP Rs.")
        )
    )
    df = si.fetch_screen("https://x.test/s/", verbose=False)
    assert list(df.columns) == ["Symbol", "Name", "CMP Rs.", "c3", "c4", "c5"]
    assert df.iloc[0].tolist() == ["AAA", "n", "1", "", "", ""]
    assert df.iloc[1].tolist() == ["BBB", "n", "1", "2", "3", "4"]


def test_fetch_screen_truncates_header_wider_than_the_rows(web):
    web.pages[1] = Resp(page([row("AAA", "n")], header=("Name", "CMP Rs.", "Cash Cycle", "Extra")))
    df = si.fetch_screen("https://x.test/s/", verbose=False)
    assert list(df.columns) == ["Symbol", "Name"]


def test_fetch_screen_without_a_header_row_uses_generic_names(web):
    web.pages[1] = Resp("<table>" + row("AAA", "n", "1") + "</table>")
    df = si.fetch_screen("https://x.test/s/", verbose=False)
    assert list(df.columns) == ["Symbol", "c1", "c2"]


def test_fetch_screen_takes_the_header_from_the_first_page_only(web):
    web.pages[1] = Resp(page([row("AAA", "n", "1", "2")]))
    web.pages[2] = Resp(page([row("BBB", "n", "1", "2")], header=("Other", "Names", "Here")))
    df = si.fetch_screen("https://x.test/s/", verbose=False)
    assert list(df.columns) == ["Symbol", "Name", "CMP Rs.", "Cash Cycle"]


def test_fetch_screen_no_rows_gives_an_empty_frame_and_prints_nothing(web, capsys):
    web.pages[1] = Resp(page([]))
    df = si.fetch_screen("https://x.test/s/")
    assert isinstance(df, pd.DataFrame) and df.empty
    assert capsys.readouterr().out == ""
    assert si.fetch_screen("https://x.test/s/", verbose=False).empty  # 404 on page 1 too


def test_fetch_screen_verbose_reports_count_and_screen_slug(web, capsys):
    web.pages[1] = Resp(page([row("AAA", "n", "1", "2"), row("BBB", "n", "1", "2")]))
    si.fetch_screen("https://www.screener.in/screens/9/my-slug/")
    assert capsys.readouterr().out == "  screener.in: 2 stocks from my-slug\n"
    si.fetch_screen("https://www.screener.in/screens/9/my-slug/", verbose=False)
    assert capsys.readouterr().out == ""


# ---- ccc_screen / ccc_map ------------------------------------------------------------------


def screen_frame():
    return pd.DataFrame(
        {
            "Symbol": ["AAA", "BBB"],
            "S.No.": ["1", "2"],
            "Name": ["Alpha", "Beta"],
            "CMP Rs.": ["10", "20"],
            "Mar Cap Rs.Cr.": ["100", "200"],
            "P/E": ["5", "6"],
            "ROCE %": ["11", "12"],
            "Cash Cycle Days": ["-3", "4"],
            "Other": ["x", "y"],
        }
    )


def test_ccc_screen_fetches_the_ccc_url_renames_and_keeps_known_columns(monkeypatch):
    seen = []
    monkeypatch.setattr(si, "fetch_screen", lambda url: seen.append(url) or screen_frame())
    df = si.ccc_screen()
    assert seen == [si.CCC_SCREEN]
    assert list(df.columns) == ["Symbol", "Name", "CMP", "MarCap_Cr", "P/E", "ROCE", "Cash_Cycle"]
    assert df["Cash_Cycle"].tolist() == ["-3", "4"] and df["CMP"].tolist() == ["10", "20"]


def test_ccc_screen_keeps_only_the_columns_that_exist(monkeypatch):
    frame = screen_frame()[["Symbol", "Name", "Cash Cycle Days"]]
    monkeypatch.setattr(si, "fetch_screen", lambda url: frame)
    assert list(si.ccc_screen().columns) == ["Symbol", "Name", "Cash_Cycle"]


def test_ccc_screen_matching_is_case_insensitive_and_prefix_based(monkeypatch):
    frame = pd.DataFrame(
        {"Symbol": ["A"], "cmp rs.": ["1"], "MAR CAP": ["2"], "roce": ["3"], "CASH CYCLE": ["4"]}
    )
    monkeypatch.setattr(si, "fetch_screen", lambda url: frame)
    assert list(si.ccc_screen().columns) == ["Symbol", "CMP", "MarCap_Cr", "ROCE", "Cash_Cycle"]
    # "cmp" must be a prefix: a column merely containing it is not renamed
    frame2 = pd.DataFrame({"Symbol": ["A"], "Prev CMP": ["1"], "Cash Cycle": ["2"]})
    monkeypatch.setattr(si, "fetch_screen", lambda url: frame2)
    assert list(si.ccc_screen().columns) == ["Symbol", "Cash_Cycle"]


def test_ccc_screen_empty_fetch_is_returned_unchanged(monkeypatch):
    empty = pd.DataFrame()
    monkeypatch.setattr(si, "fetch_screen", lambda url: empty)
    assert si.ccc_screen() is empty


def test_ccc_map_maps_symbols_to_numeric_days_with_junk_as_nan(monkeypatch):
    frame = pd.DataFrame({"Symbol": ["AAA", "BBB", "CCC"], "Cash_Cycle": ["-3.5", "4", "n/a"]})
    monkeypatch.setattr(si, "ccc_screen", lambda: frame)
    out = si.ccc_map()
    assert out["AAA"] == -3.5 and out["BBB"] == 4.0 and pd.isna(out["CCC"])
    assert list(out) == ["AAA", "BBB", "CCC"]


def test_ccc_map_is_empty_when_screen_is_empty_or_lacks_the_column(monkeypatch):
    monkeypatch.setattr(si, "ccc_screen", lambda: pd.DataFrame())
    assert si.ccc_map() == {}
    monkeypatch.setattr(si, "ccc_screen", lambda: pd.DataFrame({"Symbol": ["A"], "CMP": ["1"]}))
    assert si.ccc_map() == {}


# ---- __main__ ------------------------------------------------------------------------------


def test_main_prints_the_screen_and_saves_a_parquet_beside_cache_seed(monkeypatch, capsys):
    import runpy
    from pathlib import Path

    fake_requests = types.SimpleNamespace(
        get=lambda url, headers=None, timeout=None: (
            Resp(page([row("AAA", "Alpha", "10", "-3")]))
            if url.endswith("page=1")
            else Resp("", 404)
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "requests", fake_requests)
    saved = []
    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda self, path, index=None: saved.append((Path(path), index, self)),
    )
    runpy.run_path(si.__file__, run_name="__main__")
    out = capsys.readouterr().out
    ((path, index, frame),) = saved
    assert path == Path(si.__file__).parent / "cache_seed" / "india_ccc_screen.parquet"
    assert index is False and frame["Symbol"].tolist() == ["AAA"]
    assert "AAA" in out and out.rstrip().endswith("saved → india_ccc_screen.parquet")


# ---- extra boundary cases ------------------------------------------------------------------


def test_fetch_screen_default_page_cap_is_forty(web):
    for n in range(1, 60):
        web.pages[n] = Resp(page([row(f"S{n}", "n", "1", "2")]))
    df = si.fetch_screen("https://x.test/s/", verbose=False)
    assert len(df) == 40 and len(web.calls) == 40


def test_fetch_screen_a_headerless_first_page_fixes_generic_names_for_good(web):
    web.pages[1] = Resp("<table>" + row("AAA", "n", "1") + "</table>")
    web.pages[2] = Resp(page([row("BBB", "n", "1")], header=("Name", "CMP Rs.")))
    df = si.fetch_screen("https://x.test/s/", verbose=False)
    assert list(df.columns) == ["Symbol", "c1", "c2"]  # page 2's header is not adopted


def test_ccc_screen_does_not_mistake_margin_for_market_cap(monkeypatch):
    frame = pd.DataFrame({"Symbol": ["A"], "Margin %": ["9"], "Cash Cycle": ["2"]})
    monkeypatch.setattr(si, "fetch_screen", lambda url: frame)
    assert list(si.ccc_screen().columns) == ["Symbol", "Cash_Cycle"]
