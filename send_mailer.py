#!/usr/bin/env python3
# send_mailer.py
# ==============
# Build the Daily Market Brief and SEND it (or draft it) by email — with ZERO
# Claude/LLM tokens. Pure Python + Gmail SMTP/IMAP, so it can run unattended
# from cron/launchd.
#
# Credentials via environment (never hard-code):
#   GMAIL_USER          your gmail address
#   GMAIL_APP_PASSWORD  a Google "App Password" (Account → Security → App passwords)
#   MAIL_TO             recipient (defaults to GMAIL_USER)
#
#   python3 send_mailer.py                 # build + send (or save .html if no creds)
#   python3 send_mailer.py --draft         # just write brief_today.html, don't send
#   python3 send_mailer.py --gmail-draft   # save into the actual Gmail Drafts folder
#                                            (via IMAP APPEND) -- nothing is sent to
#                                            anyone; falls back to brief_today.html
#                                            if no credentials are configured.
#
# Nothing here calls an LLM: data, screeners, sentiment (VADER) and assembly are
# all local. The only network is market data + the SMTP/IMAP call.

from __future__ import annotations

import imaplib
import os
import smtplib
import sys
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from build_mailer import build


def _build_message(subject: str, text: str, html: str, user: str, to: str) -> MIMEMultipart:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to
    msg.attach(MIMEText(text, "plain"))
    msg.attach(MIMEText(html, "html"))
    return msg


def send(subject: str, text: str, html: str) -> bool:
    user = os.environ.get("GMAIL_USER")
    pw = os.environ.get("GMAIL_APP_PASSWORD")
    to = os.environ.get("MAIL_TO", user)
    if not (user and pw):
        Path("brief_today.html").write_text(html)
        print("  no GMAIL_USER/GMAIL_APP_PASSWORD set — saved brief_today.html instead of sending")
        return False
    msg = _build_message(subject, text, html, user, to)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(user, pw)
        s.sendmail(user, [a.strip() for a in to.split(",")], msg.as_string())
    print(f"  sent '{subject}' → {to}")
    return True


def save_as_gmail_draft(subject: str, text: str, html: str) -> bool:
    """Append the brief as a real draft in the Gmail account's Drafts folder via
    IMAP -- nothing is transmitted to any recipient, it's just saved for the
    user to review/send themselves from Gmail."""
    user = os.environ.get("GMAIL_USER")
    pw = os.environ.get("GMAIL_APP_PASSWORD")
    to = os.environ.get("MAIL_TO", user)
    if not (user and pw):
        Path("brief_today.html").write_text(html)
        print("  no GMAIL_USER/GMAIL_APP_PASSWORD set — saved brief_today.html instead of drafting")
        return False
    msg = _build_message(subject, text, html, user, to)
    with imaplib.IMAP4_SSL("imap.gmail.com") as imap:
        imap.login(user, pw)
        imap.append(
            '"[Gmail]/Drafts"',
            "\\Draft",
            imaplib.Time2Internaldate(time.time()),
            msg.as_bytes(),
        )
    print(f"  draft saved to Gmail Drafts folder: '{subject}'")
    return True


if __name__ == "__main__":
    subject, text, html = build()
    if "--gmail-draft" in sys.argv:
        save_as_gmail_draft(subject, text, html)
    elif "--draft" in sys.argv:
        Path("brief_today.html").write_text(html)
        print(f"  draft saved → brief_today.html ({subject})")
    else:
        send(subject, text, html)
