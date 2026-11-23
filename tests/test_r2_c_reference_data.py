"""Offline tests for reference_data (REF_DATA_CACHE redirected; fake requests)."""

import importlib
import io
import sys
import zipfile

import pandas as pd
import pytest


@pytest.fixture
def rd(tmp_path, monkeypatch):
    monkeypatch.setenv("REF_DATA_CACHE", str(tmp_path / "cache"))
    saved = sys.modules.pop("reference_data", None)
    mod = importlib.import_module("reference_data")
    yield mod
    sys.modules.pop("reference_data", None)
    if saved is not None:
        sys.modules["reference_data"] = saved


class Resp:
    def __init__(self, content=b"", text="", status=200):
        self.content, self.text, self.status = content, text, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(self.status)


def net(monkeypatch, rd, fn):
    calls = []

    def get(url, headers=None, timeout=None):
        calls.append((url, timeout))
        return fn(url)

    monkeypatch.setattr(rd.requests, "get", get)
    return calls


def xlsx_bytes(sheets):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False, header=False)
    return buf.getvalue()


def industry_sheet(extra_rows=None, header="Industry Name"):
    rows = [["Preamble"], ["x"], [None], [header, "Number of firms", "Current PE", "ROE", "Beta"]]
    rows += extra_rows or [
        ["Software (System)", 10, 25.5, 0.3, 1.2],
        ["Bank (Money Center)", 5, 11.0, "n/a", 1.0],
        [None, 1, 1, 1, 1],
    ]
    width = max(len(r) for r in rows)
    return pd.DataFrame([r + [None] * (width - len(r)) for r in rows])


def test_cache_dir_env(rd, tmp_path):
    assert rd.CACHE == tmp_path / "cache" and rd.CACHE.is_dir()
    assert rd.info() == {"cache_dir": str(tmp_path / "cache"), "cached_files": []}
    (rd.CACHE / "b.txt").write_text("x")
    (rd.CACHE / "a.txt").write_text("x")
    assert rd.info()["cached_files"] == ["a.txt", "b.txt"]


def test_cached_get(rd, monkeypatch):
    calls = net(monkeypatch, rd, lambda u: Resp(content=b"abc"))
    assert rd._cached_get("http://x/f", "f.bin") == b"abc"
    assert (rd.CACHE / "f.bin").read_bytes() == b"abc"
    assert rd._cached_get("http://x/f", "f.bin") == b"abc"
    assert calls == [("http://x/f", 45)]  # second call served from cache
    net(monkeypatch, rd, lambda u: Resp(status=404))
    with pytest.raises(RuntimeError):
        rd._cached_get("http://x/g", "g.bin")
    assert not (rd.CACHE / "g.bin").exists()


def test_read_damodaran(rd):
    raw = xlsx_bytes(
        {"Intro": pd.DataFrame([["nothing here"]]), "Industry Averages": industry_sheet()}
    )
    # note: Intro has no header -> skipped, parsed from second sheet
    df = rd._read_damodaran(raw)
    assert list(df["Industry"]) == ["Software (System)", "Bank (Money Center)"]
    assert list(df["_key"]) == ["software (system)", "bank (money center)"]
    assert "Industry Name" not in df.columns
    assert rd._read_damodaran(b"not an excel file") is None
    assert rd._read_damodaran(xlsx_bytes({"S": pd.DataFrame([["a", "b"]])})) is None
    # header case-insensitive / padded
    raw = xlsx_bytes({"S": industry_sheet(header="  industry NAME ")})
    assert len(rd._read_damodaran(raw)) == 2


def test_damodaran_fallback(rd, monkeypatch):
    raw = xlsx_bytes({"S": industry_sheet()})
    seen = []

    def fn(url):
        seen.append(url)
        if "Emerg" in url or "emerg" in url:
            raise RuntimeError("404")
        return Resp(content=raw)

    net(monkeypatch, rd, fn)
    df = rd.damodaran("pe", "emerg")
    assert len(df) == 2
    assert seen[0].endswith("/pedataemerg.xls") and seen[1].endswith("/pedata.xls")
    assert (rd.CACHE / "damodaran_pe_US.xls").exists()
    assert rd.damodaran("nope") is None
    net(monkeypatch, rd, lambda u: (_ for _ in ()).throw(RuntimeError("down")))
    assert rd.damodaran("roe", "") is None
    assert rd.damodaran("roe", "europe") is None
    # unparsable payload -> tries both regions -> None
    net(monkeypatch, rd, lambda u: Resp(content=b"junk"))
    assert rd.damodaran("beta", "japan") is None


def patch_damo(monkeypatch, rd, df):
    monkeypatch.setattr(rd, "damodaran", lambda m, r: df)


def test_industry_metric(rd, monkeypatch):
    raw = xlsx_bytes({"S": industry_sheet()})
    df = rd._read_damodaran(raw)
    patch_damo(monkeypatch, rd, df)
    assert rd.industry_metric("Software (System)") == 25.5
    assert rd.industry_metric("  software (system) ") == 25.5
    assert rd.industry_metric("Software (System)", "roe") == 0.3
    assert pd.isna(rd.industry_metric("Bank (Money Center)", "roe"))  # 'n/a' read as NaN
    assert rd.industry_metric("Software (System)", "beta") == 1.2
    # loose contains match on first 8 chars
    assert rd.industry_metric("Software (System & Application)") == 25.5
    assert rd.industry_metric("Nonexistent Industry") is None
    assert rd.industry_metric("") is None
    assert rd.industry_metric("Software (System)", "pe", column_hint="number") == 10.0
    assert rd.industry_metric("Software (System)", "pe", column_hint="zzz") == 25.5
    assert rd.industry_metric("Software (System)", "unknown_metric") is None
    patch_damo(monkeypatch, rd, None)
    assert rd.industry_metric("Software (System)") is None


def test_industry_metric_margin_hint(rd, monkeypatch):
    sheet = industry_sheet(extra_rows=[["Ind A", 1, 2, 3, 4]])
    sheet.iloc[3, 2] = "Net Margin"
    df = rd._read_damodaran(xlsx_bytes({"S": sheet}))
    patch_damo(monkeypatch, rd, df)
    assert rd.industry_metric("Ind A", "margin") == 2.0


def french_zip(body):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("F-F.csv", body)
    return buf.getvalue()


FF = (
    "This file was created by CMPT\n\n"
    ",Mkt-RF,SMB,HML,RF\n"
    "192607,2.96,-2.56,-2.43,0.22\n"
    "192608,2.64,-1.17,3.82,0.25\n"
    "12345,1,2,3,4\n"
    "192609,bad,1,2,3\n"
    "\n"
    "Annual Factors\n"
    "1927,29.47,-2.5,-4.0,3.1\n"
)


def test_french_factors(rd, monkeypatch):
    net(monkeypatch, rd, lambda u: Resp(content=french_zip(FF)))
    df = rd.french_factors()
    assert list(df.columns) == ["YYYYMM", "Mkt-RF", "SMB", "HML", "RF"]
    # 5-digit row skipped; blank line does not stop; 'Annual Factors' row stops parse
    assert list(df["YYYYMM"]) == ["192607", "192608", "192609"]
    assert df["Mkt-RF"].iloc[0] == 2.96 and df["HML"].iloc[1] == 3.82
    assert pd.isna(df["Mkt-RF"].iloc[2])  # coerced
    assert df["RF"].dtype.kind == "f"


def test_french_factors_failures(rd, monkeypatch):
    net(monkeypatch, rd, lambda u: (_ for _ in ()).throw(RuntimeError("down")))
    assert rd.french_factors() is None
    (rd.CACHE / "FF_3factor.zip").write_bytes(b"not a zip")
    assert rd.french_factors() is None
    (rd.CACHE / "FF_3factor.zip").write_bytes(french_zip("no data rows\nhere\n"))
    assert rd.french_factors() is None


COMPANIES = pd.DataFrame(
    {
        "Company Name": ["Foo Corp", "Bar Ltd", None, "Baz"],
        "Exchange:Ticker": ["NYSE:FOO", "SEHK:700", "X:Y", "NOCOLON"],
        "Industry Group": ["Software", "Banks", "Z", "Misc"],
        "Country": ["United States", "Hong Kong", "X", "Y"],
    }
)


def companies_xlsx(df=COMPANIES, sheet="By company name"):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, sheet_name=sheet, index=False)
    return buf.getvalue()


def test_damodaran_companies(rd, monkeypatch):
    calls = net(monkeypatch, rd, lambda u: Resp(content=companies_xlsx()))
    df = rd.damodaran_companies()
    assert list(df["Company Name"]) == ["Foo Corp", "Bar Ltd", "Baz"]  # null name dropped
    assert list(df["Exchange"]) == ["NYSE", "SEHK", "NOCOLON"]
    assert df["Ticker"].iloc[0] == "FOO" and df["Ticker"].iloc[1] == "700"
    assert pd.isna(df["Ticker"].iloc[2])
    assert (rd.CACHE / "damodaran_companies.parquet").exists()
    n = len(calls)
    again = rd.damodaran_companies()  # parquet cache
    assert len(again) == 3 and len(calls) == n
    rd.damodaran_companies(refresh=True)
    # refresh re-reads the cached xls (no new network)
    assert len(calls) == n


def test_damodaran_companies_failure_and_no_exchange_col(rd, monkeypatch):
    net(monkeypatch, rd, lambda u: Resp(status=500))
    assert rd.damodaran_companies() is None
    net(monkeypatch, rd, lambda u: Resp(content=b"junk"))
    assert rd.damodaran_companies() is None
    plain = COMPANIES.drop(columns=["Exchange:Ticker"])
    net(monkeypatch, rd, lambda u: Resp(content=companies_xlsx(plain)))
    (rd.CACHE / "indname.xls").unlink(missing_ok=True)
    df = rd.damodaran_companies(refresh=True)
    assert "Exchange" not in df.columns and len(df) == 3


def test_company_industry(rd, monkeypatch):
    df = COMPANIES.copy()
    df["Ticker"] = ["FOO", "700", None, None]
    monkeypatch.setattr(rd, "damodaran_companies", lambda: df)
    assert rd.company_industry("foo") == "Software"
    assert rd.company_industry("  FOO ") == "Software"
    assert rd.company_industry("Bar Ltd") == "Banks"  # via company name
    assert rd.company_industry("700") == "Banks"
    assert rd.company_industry("nobody") is None
    monkeypatch.setattr(rd, "damodaran_companies", lambda: None)
    assert rd.company_industry("foo") is None
    monkeypatch.setattr(rd, "damodaran_companies", lambda: df.drop(columns=["Ticker"]))
    assert rd.company_industry("foo corp") == "Software"
    assert rd.company_industry("foo") is None


def test_aqr_links(rd, monkeypatch):
    html = (
        '<a href="/media/a.xlsx">A</a> <a href="https://x.com/b.XLS">B</a> '
        '<a href="/c.pdf">C</a> <a href="/media/a.xlsx">dup</a> <a href=\'/d.xlsx\'>q</a>'
    )
    net(monkeypatch, rd, lambda u: Resp(text=html))
    assert rd.aqr_links() == ["https://www.aqr.com/media/a.xlsx", "https://x.com/b.XLS"]
    net(monkeypatch, rd, lambda u: (_ for _ in ()).throw(RuntimeError("x")))
    assert rd.aqr_links() == []


def test_main_block(rd, monkeypatch, capsys):
    import runpy

    monkeypatch.setattr(
        rd.requests, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
    )
    runpy.run_path(rd.__file__, run_name="__main__")
    out = capsys.readouterr().out
    assert "Damodaran emerging-market PE table:" in out and "Fama-French 3F rows: None" in out
    assert "cache:" in out


def test_industry_metric_text_cell_falls_through(rd, monkeypatch):
    sheet = industry_sheet(extra_rows=[["Ind A", 1, "abc", "xyz", "oops"]])
    df = rd._read_damodaran(xlsx_bytes({"S": sheet}))
    patch_damo(monkeypatch, rd, df)
    assert rd.industry_metric("Ind A", "pe") is None  # every candidate column is text
    assert rd.industry_metric("Ind A", "pe", column_hint="firms") == 1.0


def test_read_damodaran_sheet_probe_failure(rd, monkeypatch):
    raw = xlsx_bytes({"Bad": industry_sheet(), "Good": industry_sheet()})
    real = pd.read_excel
    state = {"n": 0}

    def flaky(*a, **k):
        if k.get("nrows") == 15 and state["n"] == 0:
            state["n"] += 1
            raise ValueError("corrupt sheet")
        return real(*a, **k)

    monkeypatch.setattr(pd, "read_excel", flaky)
    assert len(rd._read_damodaran(raw)) == 2  # first sheet skipped, second parsed
