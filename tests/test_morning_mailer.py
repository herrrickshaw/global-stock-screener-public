# Tests for the morning mailer assembly — pure logic, no network/send.
import morning_mailer as mm


def test_md_to_html_sections_and_table():
    md = "# Title\n\n## Sec\n- item one\n- item two\n\n| A | B |\n|---|---|\n| 1 | 2 |"
    html = mm._md_to_html(md)
    assert "<h2>Title</h2>" in html and "<h3>Sec</h3>" in html
    assert "<ul>" in html and "<li>item one</li>" in html
    assert "<table" in html and "<td>1</td>" in html
    assert "**" not in html  # bold markers stripped


def test_assemble_builds_subject_text_html(monkeypatch):
    monkeypatch.setattr(mm, "_top_buys", lambda m, n=8: ["AAA", "BBB"])
    import daily_brief as db

    monkeypatch.setattr(db, "build_brief", lambda markets: "# Daily Market Brief\nBull: IN")
    subj, text, html = mm.assemble(["IN"])
    assert "Morning Market Brief" in subj
    assert "Top BUY candidates" in text and "AAA, BBB" in text
    assert html.startswith("<html>") and "</html>" in html


def test_send_dry_writes_files(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "RESULTS", tmp_path)
    monkeypatch.setattr(mm, "_top_buys", lambda m, n=8: [])
    import daily_brief as db

    monkeypatch.setattr(db, "build_brief", lambda markets: "# Brief")
    assert mm.send(["IN"], dry=True, verbose=False) is False
    assert (tmp_path / "morning_brief.html").exists()
