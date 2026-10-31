"""MCP server: the user's mailbox over IMAP/SMTP (Gmail by default).
Run: python -m personal_ai.connectors.email_server
Env: PAI_EMAIL_USER, PAI_EMAIL_PASS (Gmail: an App Password), optional PAI_IMAP_HOST, PAI_SMTP_HOST, PAI_SMTP_PORT."""
import email
import email.header
import email.utils
import imaplib
import os
import smtplib
from email.message import EmailMessage

from .. import mcp

MAX_BODY = 6000


def _creds() -> tuple[str, str]:
    user, pw = os.environ.get("PAI_EMAIL_USER"), os.environ.get("PAI_EMAIL_PASS")
    if not user or not pw:
        raise RuntimeError("email is not configured: set PAI_EMAIL_USER and PAI_EMAIL_PASS (for Gmail, an App Password)")
    return user, pw


def _imap() -> imaplib.IMAP4_SSL:
    user, pw = _creds()
    m = imaplib.IMAP4_SSL(os.environ.get("PAI_IMAP_HOST", "imap.gmail.com"))
    m.login(user, pw)
    return m


def _decode(value: str | None) -> str:
    if not value:
        return ""
    return "".join(part.decode(enc or "utf8", "replace") if isinstance(part, bytes) else part
                   for part, enc in email.header.decode_header(value))


def _body(msg: email.message.Message) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and "attachment" not in str(part.get("Content-Disposition", "")):
                return part.get_payload(decode=True).decode(part.get_content_charset() or "utf8", "replace")
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                import html, re
                raw = part.get_payload(decode=True).decode(part.get_content_charset() or "utf8", "replace")
                return re.sub(r"\s+", " ", html.unescape(re.sub(r"(?s)<[^>]+>", " ", raw)))
        return ""
    return msg.get_payload(decode=True).decode(msg.get_content_charset() or "utf8", "replace")


def search(query: str, limit: str = "10", folder: str = "INBOX") -> str:
    m = _imap()
    m.select(folder, readonly=True)
    gmail = "gmail" in os.environ.get("PAI_IMAP_HOST", "imap.gmail.com")
    if gmail:
        _, data = m.uid("search", None, "X-GM-RAW", f'"{query}"')
    else:
        _, data = m.uid("search", None, "TEXT", f'"{query}"')
    uids = data[0].split()[-int(limit or 10):][::-1]
    out = []
    for uid in uids:
        _, d = m.uid("fetch", uid, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])")
        hdr = email.message_from_bytes(d[0][1])
        out.append(f"uid={uid.decode()}  {_decode(hdr.get('Date'))[:25]}  from: {_decode(hdr.get('From'))}\n    {_decode(hdr.get('Subject'))}")
    m.logout()
    return "\n".join(out) or "(no matching emails)"


def read(uid: str, folder: str = "INBOX") -> str:
    m = _imap()
    m.select(folder, readonly=True)
    _, d = m.uid("fetch", uid, "(BODY.PEEK[])")
    m.logout()
    if not d or d[0] is None:
        raise ValueError(f"no email with uid {uid}")
    msg = email.message_from_bytes(d[0][1])
    body = _body(msg).strip()
    if len(body) > MAX_BODY:
        body = body[:MAX_BODY] + f"\n...[truncated, {len(body)} chars]"
    return (f"From: {_decode(msg.get('From'))}\nTo: {_decode(msg.get('To'))}\nDate: {_decode(msg.get('Date'))}\n"
            f"Subject: {_decode(msg.get('Subject'))}\n\n{body}")


def send(to: str, subject: str, body: str) -> str:
    user, pw = _creds()
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = user, to, subject
    msg.set_content(body)
    with smtplib.SMTP_SSL(os.environ.get("PAI_SMTP_HOST", "smtp.gmail.com"), int(os.environ.get("PAI_SMTP_PORT", "465"))) as s:
        s.login(user, pw)
        s.send_message(msg)
    return f"sent to {to}: {subject}"


TOOLS = {
    "search": {"fn": search, "args": {"query": "search terms (Gmail search syntax works, e.g. from:x newer_than:7d)", "limit": "max results (default 10)", "folder": "mailbox (default INBOX)"},
               "optional": {"limit": True, "folder": True}, "desc": "Search the mailbox; returns uid, date, sender and subject per email."},
    "read": {"fn": read, "args": {"uid": "email uid from search", "folder": "mailbox (default INBOX)"}, "optional": {"folder": True},
             "desc": "Read one email in full."},
    "send": {"fn": send, "args": {"to": "recipient address", "subject": "subject line", "body": "plain-text body"},
             "desc": "Send an email from the user's account."},
}

if __name__ == "__main__":
    mcp.serve("email", TOOLS)
