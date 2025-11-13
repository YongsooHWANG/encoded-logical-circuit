import mimetypes, smtplib, ssl
from email.message import EmailMessage
from pathlib import Path

def send_mail_with_attachments(smtp_host, smtp_port, user, password,
                               from_addr, to_addrs, subject, body, attachments):
    """
    attachments: 파일 경로 리스트 [str | Path]
    """
    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs) if isinstance(to_addrs, (list, tuple)) else to_addrs
    msg["Subject"] = subject
    msg.set_content(body)

    for fp in attachments:
        fp = Path(fp)
        ctype, encoding = mimetypes.guess_type(fp.name)
        if ctype is None or encoding is not None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)

        with open(fp, "rb") as f:
            msg.add_attachment(f.read(),
                               maintype=maintype,
                               subtype=subtype,
                               filename=fp.name)

    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_host, smtp_port, context=ctx) as s:
        s.login(user, password)
        s.send_message(msg)
