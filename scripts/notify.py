"""Gmail notifier — send mail to yourself via the Gmail API.

Why this exists
===============
Slurm has a built-in --mail-type=ALL but you only get a one-line summary at
job start / end. For a long-running active loop you want richer milestones:
"training round 3 finished, here's the loss curve tail", "rollout done with
SR=0.62", "Phase B finished for episode 4", "BFS expert collected 7 demos for
round 3", and so on. This module is the email plumbing those callsites use.

Setup (one-time, on a machine with a browser)
=============================================
1. Drop your Google OAuth client credentials into the project root as
   `credentials.json` (Desktop OAuth client; create one at
   https://console.cloud.google.com/apis/credentials).
2. Enable the Gmail API for that project:
   https://console.cloud.google.com/apis/library/gmail.googleapis.com
3. Run the OAuth flow once to mint `token.json`:

     python scripts/notify.py --setup

   A browser opens, you authorise the "send mail as you" scope, the token is
   saved to `token.json` next to credentials.json.

4. Copy BOTH `credentials.json` and `token.json` to the HPC project root.
   Tokens auto-refresh; you only repeat the browser step if the refresh
   token gets revoked.

Use as a CLI
============
  python scripts/notify.py --to me@example.com --subject "training done" --body "loss=0.012"
  python scripts/notify.py --to me@example.com --subject "round 3 done" --body-file slurm_logs/round_42.out --tail 200
  python scripts/notify.py --to me@example.com --subject "with attachment" --body "see file" --attach results/x.json

Use as a library
================
  from scripts.notify import send_email, notify_event
  send_email(subject="X", body="Y", to="me@example.com")
  notify_event("TRAIN_DONE", {"loss": 0.012, "epochs": 5000})
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import traceback
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
CRED_PATH = REPO_ROOT / "credentials.json"
TOKEN_PATH = REPO_ROOT / "token.json"

GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

# Default recipient — override via NOTIFY_TO env var or --to flag.
DEFAULT_TO = os.environ.get("NOTIFY_TO", "").strip()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def _load_credentials():
    """Returns google.oauth2.credentials.Credentials, refreshing or running
    the OAuth installed-app flow as needed."""
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as e:
        raise SystemExit(
            "[notify] missing Google libs. Install with:\n"
            "  pip install google-api-python-client google-auth-oauthlib\n"
            f"(import error: {e})"
        )

    creds = None
    if TOKEN_PATH.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), GMAIL_SCOPES)
        except Exception as e:
            print(f"[notify] WARNING: could not load {TOKEN_PATH}: {e}")
            creds = None

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            TOKEN_PATH.write_text(creds.to_json())
            return creds
        except Exception as e:
            print(f"[notify] token refresh failed ({e}); re-running OAuth flow.")

    if not CRED_PATH.exists():
        raise SystemExit(
            f"[notify] {CRED_PATH} not found. Drop your Gmail OAuth client "
            "credentials there (see module docstring)."
        )

    flow = InstalledAppFlow.from_client_secrets_file(str(CRED_PATH), GMAIL_SCOPES)
    creds = flow.run_local_server(port=0)
    TOKEN_PATH.write_text(creds.to_json())
    print(f"[notify] new token saved to {TOKEN_PATH}")
    return creds


def _gmail_service():
    from googleapiclient.discovery import build
    creds = _load_credentials()
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


# ---------------------------------------------------------------------------
# Send
# ---------------------------------------------------------------------------
def _build_message(
    subject: str,
    body: str,
    to: str,
    attachments: Optional[Iterable[str]] = None,
) -> Dict[str, str]:
    msg = EmailMessage()
    msg["To"]      = to
    msg["Subject"] = subject
    msg.set_content(body or "(no body)")

    for path_str in attachments or []:
        path = Path(path_str)
        if not path.exists():
            print(f"[notify] WARNING: attachment {path} not found; skipping.")
            continue
        ctype, encoding = mimetypes.guess_type(str(path))
        if ctype is None or encoding is not None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)
        with open(path, "rb") as f:
            msg.add_attachment(
                f.read(), maintype=maintype, subtype=subtype, filename=path.name,
            )

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")
    return {"raw": raw}


def send_email(
    subject: str,
    body: str,
    to: Optional[str] = None,
    attachments: Optional[Iterable[str]] = None,
    silent: bool = False,
) -> bool:
    """Send one email. Returns True on success, False on any failure (logged).

    Failures NEVER raise — a notifier outage must not crash the active loop.
    """
    recipient = (to or DEFAULT_TO or "").strip()
    if not recipient:
        if not silent:
            print("[notify] no recipient (set NOTIFY_TO or pass --to); skipping send.")
        return False

    try:
        service = _gmail_service()
        message = _build_message(subject, body, recipient, attachments)
        service.users().messages().send(userId="me", body=message).execute()
        if not silent:
            print(f"[notify] sent → {recipient}: {subject!r}")
        return True
    except SystemExit:
        raise
    except Exception as e:
        print(f"[notify] FAILED to send {subject!r}: {e}")
        if not silent:
            traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Convenience: structured event with payload
# ---------------------------------------------------------------------------
def notify_event(
    event: str,
    payload: Optional[Dict[str, Any]] = None,
    *,
    to: Optional[str] = None,
    attachments: Optional[Iterable[str]] = None,
    subject_prefix: str = "[DmN]",
    silent: bool = False,
) -> bool:
    """Send a structured 'event' email.

    event:    short name shown in subject e.g. 'TRAIN_DONE', 'PHASE_C_DONE'
    payload:  dict pretty-printed as JSON in the body
    """
    subject = f"{subject_prefix} {event}"
    if payload:
        body = (
            f"event: {event}\n"
            f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"host: {os.environ.get('HOSTNAME', os.uname().nodename if hasattr(os,'uname') else 'unknown')}\n"
            f"slurm_job_id: {os.environ.get('SLURM_JOB_ID','-')}\n"
            "---\n"
            f"{json.dumps(payload, indent=2, default=str)}\n"
        )
    else:
        body = f"event: {event}\ntimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    return send_email(subject, body, to=to, attachments=attachments, silent=silent)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _tail(path: Path, n_lines: int) -> str:
    """Return the last n_lines of a file. Used to mail .out tails efficiently."""
    if not path.exists():
        return f"(file {path} not found)"
    # Cheap-and-cheerful tail: we only ever ship a few hundred KB.
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-n_lines:]) if lines else "(empty)"


def parse_args():
    p = argparse.ArgumentParser(description="Gmail notifier for the active-loop pipeline.")
    p.add_argument("--setup", action="store_true",
                   help="Run the OAuth flow once to mint token.json next to credentials.json.")
    p.add_argument("--to", type=str, default=None,
                   help=f"Recipient email. Default = $NOTIFY_TO (currently {DEFAULT_TO or 'unset'}).")
    p.add_argument("--subject", type=str, default=None)
    p.add_argument("--body", type=str, default=None)
    p.add_argument("--body-file", type=str, default=None, dest="body_file",
                   help="Read body from this file (e.g. slurm .out).")
    p.add_argument("--tail", type=int, default=0,
                   help="If --body-file is set, send only the last N lines.")
    p.add_argument("--attach", type=str, action="append", default=[],
                   help="Attach a file (repeatable).")
    p.add_argument("--event", type=str, default=None,
                   help="Send as a structured event (sets subject = '[DmN] <event>').")
    p.add_argument("--payload-json", type=str, default=None, dest="payload_json",
                   help="JSON string used as the event payload (with --event).")
    return p.parse_args()


def main():
    args = parse_args()

    if args.setup:
        _ = _load_credentials()
        print(f"[notify] setup OK. token at {TOKEN_PATH}")
        return 0

    if args.event:
        payload = json.loads(args.payload_json) if args.payload_json else None
        ok = notify_event(args.event, payload, to=args.to, attachments=args.attach)
        return 0 if ok else 1

    if not args.subject:
        print("[notify] --subject required (or use --event).")
        return 2

    if args.body_file:
        body = _tail(Path(args.body_file), args.tail) if args.tail > 0 else Path(args.body_file).read_text(errors="replace")
    else:
        body = args.body or ""

    ok = send_email(args.subject, body, to=args.to, attachments=args.attach)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
