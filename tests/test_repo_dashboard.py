# Tests for the repo activity dashboard — pure assembly, no network/send.
import repo_dashboard as rd


def test_assemble_returns_subject_text_html():
    subj, text, html = rd.assemble(["IN"])
    assert "Repo Activity Dashboard" in subj
    assert html.startswith("<html>") and "</html>" in html
    # KPI strip + all cards present
    for token in [
        "Governance",
        "Challenges",
        "Market Tape",
        "Stock Lists",
        "Data Coverage",
        "Backlog",
    ]:
        assert token in html
    assert "NOT investment advice" in text


def test_safe_degrades_on_error():
    out = rd._safe(lambda: (_ for _ in ()).throw(RuntimeError("boom")), "fallback msg")
    assert "fallback msg" in out and "boom" in out


def test_stock_lists_uses_combined_fragments(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "COMBINED_DIR", tmp_path)
    (tmp_path / "combined_IN_20260101_0900.html").write_text("<b>OLDER</b>")
    (tmp_path / "combined_IN_20260102_0900.html").write_text("<b>NEWEST</b>")
    html = rd._stock_lists(["IN"])
    assert "NEWEST" in html and "OLDER" not in html
    assert "India" in html


def test_challenges_flags_governance_and_backlog():
    metrics = {"gov_fail": ["P4 Data integrity (integrity manifest)"], "todo": 3}
    out = rd._challenges([], metrics)
    assert "non-conformance" in out.lower()
    assert "3 backlog" in out


def test_challenges_all_green(monkeypatch):
    import pandas as pd

    import market_memory as mm

    # fresh, complete data → no staleness/gaps
    end = __import__("datetime").date.today().isoformat()
    fake = pd.DataFrame([{"market": "IN", "tier": "stm", "symbols": 10, "end": end, "rows": 100}])
    monkeypatch.setattr(mm, "status", lambda markets=None: fake)
    out = rd._challenges(["IN"], {"gov_fail": [], "todo": 0})
    assert "No open challenges" in out
