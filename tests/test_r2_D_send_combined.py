"""Offline tests for send_mailer.py and combined_mailer.py (no SMTP/IMAP, no network)."""

import runpy
import sys
import types

import pytest

import combined_mailer
import send_mailer


class FakeSMTP:
    instances = []

    def __init__(self, host, port):
        self.host, self.port = host, port
        self.logged = None
        self.sent = None
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def login(self, u, p):
        self.logged = (u, p)

    def sendmail(self, frm, to, body):
        self.sent = (frm, to, body)


class FakeIMAP:
    instances = []

    def __init__(self, host):
        self.host = host
        self.appended = None
        FakeIMAP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def login(self, u, p):
        self.logged = (u, p)

    def append(self, *args):
        self.appended = args


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for k in ("GMAIL_USER", "GMAIL_APP_PASSWORD", "MAIL_TO"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)
    FakeSMTP.instances = []
    FakeIMAP.instances = []
    monkeypatch.setattr(send_mailer.smtplib, "SMTP_SSL", FakeSMTP)
    monkeypatch.setattr(send_mailer.imaplib, "IMAP4_SSL", FakeIMAP)


def test_build_message_parts():
    m = send_mailer._build_message("S", "plain", "<b>h</b>", "a@x", "b@y")
    assert m["Subject"] == "S" and m["From"] == "a@x" and m["To"] == "b@y"
    parts = m.get_payload()
    assert [p.get_content_type() for p in parts] == ["text/plain", "text/html"]


def test_send_no_creds_saves_html(tmp_path, capsys):
    assert send_mailer.send("s", "t", "<p>x</p>") is False
    assert (tmp_path / "brief_today.html").read_text() == "<p>x</p>"
    assert "saved brief_today.html" in capsys.readouterr().out
    assert FakeSMTP.instances == []


def test_send_missing_password_only(monkeypatch):
    monkeypatch.setenv("GMAIL_USER", "u@x")
    assert send_mailer.send("s", "t", "h") is False
    assert FakeSMTP.instances == []


def test_send_ok_defaults_to_self_and_splits_recipients(monkeypatch):
    monkeypatch.setenv("GMAIL_USER", "u@x")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "pw")
    assert send_mailer.send("s", "t", "h") is True
    s = FakeSMTP.instances[0]
    assert (s.host, s.port) == ("smtp.gmail.com", 465)
    assert s.logged == ("u@x", "pw")
    assert s.sent[0] == "u@x" and s.sent[1] == ["u@x"]
    monkeypatch.setenv("MAIL_TO", "a@x, b@y ,c@z")
    send_mailer.send("s", "t", "h")
    assert FakeSMTP.instances[1].sent[1] == ["a@x", "b@y", "c@z"]


def test_gmail_draft_no_creds(tmp_path):
    assert send_mailer.save_as_gmail_draft("s", "t", "<i>h</i>") is False
    assert (tmp_path / "brief_today.html").read_text() == "<i>h</i>"
    assert FakeIMAP.instances == []


def test_gmail_draft_ok(monkeypatch):
    monkeypatch.setenv("GMAIL_USER", "u@x")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "pw")
    assert send_mailer.save_as_gmail_draft("subj", "t", "h") is True
    i = FakeIMAP.instances[0]
    assert i.host == "imap.gmail.com"
    assert i.appended[0] == '"[Gmail]/Drafts"' and i.appended[1] == "\\Draft"
    assert isinstance(i.appended[3], bytes) and b"subj" in i.appended[3]


@pytest.mark.parametrize(
    "argv,expect", [(["--draft"], "draft"), ([], "send"), (["--gmail-draft"], "gdraft")]
)
def test_send_mailer_main(monkeypatch, tmp_path, argv, expect, capsys):
    fake = types.ModuleType("build_mailer")
    fake.build = lambda: ("SUBJ", "txt", "<h/>")
    monkeypatch.setitem(sys.modules, "build_mailer", fake)
    monkeypatch.setattr(sys, "argv", ["send_mailer.py"] + argv)
    monkeypatch.delitem(sys.modules, "send_mailer", raising=False)
    runpy.run_path(str(__import__("pathlib").Path(send_mailer.__file__)), run_name="__main__")
    monkeypatch.setitem(sys.modules, "send_mailer", send_mailer)
    out = capsys.readouterr().out
    if expect == "draft":
        assert "draft saved → brief_today.html (SUBJ)" in out
    else:
        assert "no GMAIL_USER" in out
        assert ("drafting" in out) == (expect == "gdraft")


# ---------------- combined_mailer ----------------


def _install(monkeypatch, name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    monkeypatch.setitem(sys.modules, name, m)
    return m


def _boom():
    raise RuntimeError("x")


def test_daily_ok(monkeypatch):
    _install(monkeypatch, "build_mailer", build=lambda: ("s", "T", "H"))
    assert combined_mailer._daily() == ("T", "H")


def test_daily_fallback_regime(monkeypatch):
    _install(monkeypatch, "build_mailer", build=_boom)
    _install(monkeypatch, "regime_picks", section_html=lambda: "<SEC>")
    t, h = combined_mailer._daily()
    assert "Regime picks" in t and "<SEC>" in h and "Daily Screener Brief" in h


def test_daily_fallback_empty(monkeypatch):
    _install(monkeypatch, "build_mailer", build=_boom)
    _install(monkeypatch, "regime_picks", section_html=_boom)
    assert combined_mailer._daily() == ("", "")


def test_reward_and_overlay(monkeypatch):
    _install(
        monkeypatch, "reward_picks_mailer", section_text=lambda: "rt", section_html=lambda: "rh"
    )
    _install(
        monkeypatch, "reward_overlay_mailer", section_text=lambda: "ot", section_html=lambda: "oh"
    )
    assert combined_mailer._reward_picks() == ("rt", "rh")
    assert combined_mailer._overlay_picks() == ("ot", "oh")
    _install(monkeypatch, "reward_picks_mailer", section_text=_boom, section_html=_boom)
    _install(monkeypatch, "reward_overlay_mailer", section_text=_boom, section_html=_boom)
    assert combined_mailer._reward_picks() == ("", "")
    assert combined_mailer._overlay_picks() == ("", "")


def _patch_all(monkeypatch, reward=("rt", "rh"), overlay=("ot", "oh"), daily=("DT", "DH")):
    monkeypatch.setattr(combined_mailer, "_daily", lambda: daily)
    monkeypatch.setattr(combined_mailer, "_reward_picks", lambda: reward)
    monkeypatch.setattr(combined_mailer, "_overlay_picks", lambda: overlay)
    _install(monkeypatch, "program_summary_mailer", build=lambda: ("ps", "ST", "SH"))


def test_build_stacks_sections(monkeypatch):
    _patch_all(monkeypatch)
    subj, text, html = combined_mailer.build()
    assert "Cross-Market Research" in subj
    assert html.startswith("DHrhoh") and html.endswith("SH")
    assert "CROSS-MARKET RESEARCH" in html
    assert "DT\n\nrt\n\not\n\n===" in text and text.endswith("ST")


def test_build_omits_empty_sections(monkeypatch):
    _patch_all(monkeypatch, reward=("", ""), overlay=("", ""), daily=("", ""))
    _, text, html = combined_mailer.build()
    assert html.startswith("<div") and "\n\nrt" not in text
    assert "Brief + Cross-Market Research" in text
    assert "\n\nrt" not in text and "\n\not" not in text


@pytest.mark.parametrize(
    "send_result,msg", [(True, "sent"), (False, "not sent"), ("err", "send failed")]
)
def test_combined_main(monkeypatch, tmp_path, capsys, send_result, msg):
    _patch_all(monkeypatch)
    if send_result == "err":
        _install(monkeypatch, "send_mailer", send=_boom)
    else:
        _install(monkeypatch, "send_mailer", send=lambda s, t, h: send_result)
    monkeypatch.setattr(sys, "argv", ["combined_mailer.py", "--send"])
    runpy.run_path(combined_mailer.__file__, run_name="__main__")
    out = capsys.readouterr().out
    assert msg in out
    assert (tmp_path / "combined_brief.html").read_text().endswith("SH")


def test_combined_main_no_send(monkeypatch, tmp_path, capsys):
    _patch_all(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["combined_mailer.py"])
    runpy.run_path(combined_mailer.__file__, run_name="__main__")
    assert "html bytes:" in capsys.readouterr().out
