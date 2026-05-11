#!/usr/bin/env python3
"""
dgmh-webhook-test — drive iterative Discord webhook tests, scrape
per-iteration metrics from agent.log, emit a markdown table.

See ../README.md for usage.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


_DEFAULT_LOG = Path("~/.hermes/logs/agent.log").expanduser()
_DEFAULT_INTERVAL_S = 35.0

_INBOUND_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) INFO gateway\.run: "
    r"inbound message: platform=discord user=(?P<user>.+?) "
    r"chat=(?P<chat>\d+) msg='(?P<msg>.+)'$"
)
_RESPONSE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) INFO gateway\.run: "
    r"response ready: platform=discord chat=(?P<chat>\d+) time=(?P<time>[\d.]+)s "
    r"api_calls=(?P<api>\d+) response=(?P<resp_len>\d+) chars"
)
_PATINA_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) INFO "
    r"dgmh\.hermes_integration\.humanness_hook: "
    r"\[humanness_hook\] (?:public-mode rewrite|pre-send rewrite) applied "
    r"(?:\(flags=(?P<flags>[^)]+) )?\(?len=(?P<in_len>\d+)→(?P<out_len>\d+)\)?$"
)
_GATE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) INFO "
    r"dgmh\.hermes_integration\.pre_send_gate: \[pre_send_gate\] "
    r"decision=(?P<decision>\w+) in_len=(?P<in_len>\d+) out_len=(?P<out_len>\d+) "
    r"source=(?P<source>\w+) reason=(?P<reason>[^ ]+)$"
)


def _resolve_webhook(args: argparse.Namespace) -> Optional[str]:
    if args.webhook:
        return args.webhook
    if args.webhook_env:
        path = Path(args.webhook_env).expanduser()
        if not path.exists():
            print(f"webhook env file not found: {path}", file=sys.stderr)
            return None
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("webhook=") or line.startswith("WEBHOOK="):
                return line.split("=", 1)[1].strip()
        print(
            f"no `webhook=...` line in {path}",
            file=sys.stderr,
        )
        return None
    return None


def _load_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = []
    if args.prompts:
        prompts.extend(args.prompts)
    if args.prompts_file:
        path = Path(args.prompts_file).expanduser()
        if not path.exists():
            print(f"prompts file not found: {path}", file=sys.stderr)
            return []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    return prompts


def _post_webhook(webhook: str, content: str) -> tuple[bool, str]:
    body = json.dumps({"content": content}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        webhook,
        data=body,
        headers={
            "Content-Type": "application/json",
            # Discord's Cloudflare WAF rejects default Python-urllib UA
            # with `error code: 1010`. Use a DiscordBot-shaped UA per
            # https://discord.com/developers/docs/reference#user-agent.
            "User-Agent": "DiscordBot (https://github.com/devswha/dgmh, 0.1)",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20.0) as resp:
            status = resp.getcode()
            if 200 <= status < 300:
                return True, ""
            return False, f"HTTP {status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTPError {e.code}: {e.read()!r}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _parse_ts(ts_raw: str) -> datetime:
    head, _, _ = ts_raw.partition(",")
    return datetime.strptime(head, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=timezone.utc
    )


def _scan_log(
    log_path: Path,
    started_at: datetime,
    prompts: list[str],
) -> list[dict[str, Any]]:
    """Walk the log, group lines per inbound match. Returns one row per
    prompt that was matched.
    """
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"log read failed ({log_path}): {e}", file=sys.stderr)
        return []

    rows: list[dict[str, Any]] = []
    open_row: Optional[dict[str, Any]] = None

    for line in text.splitlines():
        m_inb = _INBOUND_RE.match(line)
        if m_inb:
            ts = _parse_ts(m_inb.group("ts"))
            if ts < started_at:
                continue
            # close previous row if still open
            if open_row is not None:
                rows.append(open_row)
            open_row = {
                "ts": m_inb.group("ts"),
                "user": m_inb.group("user"),
                "chat": m_inb.group("chat"),
                "prompt": m_inb.group("msg"),
                "response_chars": None,
                "api_calls": None,
                "elapsed_s": None,
                "patina_in": None,
                "patina_out": None,
                "patina_flags": None,
                "gate_decision": None,
                "gate_in_len": None,
                "gate_out_len": None,
                "gate_reason": None,
            }
            continue
        if open_row is None:
            continue

        m_resp = _RESPONSE_RE.match(line)
        if m_resp and m_resp.group("chat") == open_row["chat"]:
            open_row["response_chars"] = int(m_resp.group("resp_len"))
            open_row["api_calls"] = int(m_resp.group("api"))
            open_row["elapsed_s"] = float(m_resp.group("time"))
            continue

        m_pat = _PATINA_RE.match(line)
        if m_pat:
            open_row["patina_in"] = int(m_pat.group("in_len"))
            open_row["patina_out"] = int(m_pat.group("out_len"))
            open_row["patina_flags"] = m_pat.group("flags")
            continue

        m_gate = _GATE_RE.match(line)
        if m_gate:
            # Track only the LAST gate event per row — final outbound.
            open_row["gate_decision"] = m_gate.group("decision")
            open_row["gate_in_len"] = int(m_gate.group("in_len"))
            open_row["gate_out_len"] = int(m_gate.group("out_len"))
            open_row["gate_reason"] = m_gate.group("reason")

    if open_row is not None:
        rows.append(open_row)

    # Reconcile rows to the prompts we sent, in order.
    matched: list[dict[str, Any]] = []
    pending = list(prompts)
    for row in rows:
        if not pending:
            break
        # Discord may shorten very long prompts; do a substring match.
        target = pending[0]
        if row["prompt"] == target or row["prompt"].startswith(target[:40]):
            matched.append(row)
            pending.pop(0)

    return matched


def _emit_table(rows: list[dict[str, Any]], total_sent: int) -> None:
    if not rows:
        print(
            f"no log rows matched (sent {total_sent} prompts).",
            file=sys.stderr,
        )
        return
    print(
        "| # | prompt | response | patina | gate | api | elapsed |"
    )
    print("|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        p = (r["prompt"] or "")[:30].replace("|", "\\|")
        resp = r["response_chars"] if r["response_chars"] is not None else "-"
        pat = (
            f"{r['patina_in']}→{r['patina_out']}"
            if r["patina_in"] is not None
            else "-"
        )
        if r["gate_decision"]:
            gate = f"{r['gate_decision']} ({r['gate_in_len']}→{r['gate_out_len']})"
            if r["gate_reason"] and r["gate_reason"] != "-":
                gate += f" [{r['gate_reason']}]"
        else:
            gate = "-"
        api = r["api_calls"] if r["api_calls"] is not None else "-"
        el = f"{r['elapsed_s']:.1f}s" if r["elapsed_s"] is not None else "-"
        print(f"| {i} | {p} | {resp} | {pat} | {gate} | {api} | {el} |")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Iterative Discord webhook validation for DGM-H flask persona."
        )
    )
    parser.add_argument("--webhook", help="Discord webhook URL")
    parser.add_argument(
        "--webhook-env",
        help="env file containing a `webhook=...` line",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        help="inline prompts (one per arg)",
    )
    parser.add_argument(
        "--prompts-file",
        help="path to a file with one prompt per line",
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=_DEFAULT_INTERVAL_S,
        help=f"seconds between sends (default {_DEFAULT_INTERVAL_S})",
    )
    parser.add_argument(
        "--log-path",
        default=str(_DEFAULT_LOG),
        help=f"agent log (default {_DEFAULT_LOG})",
    )
    parser.add_argument(
        "--output",
        help="optional JSON file to dump per-iteration rows",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress per-send progress lines",
    )
    args = parser.parse_args()

    webhook = _resolve_webhook(args)
    if not webhook:
        print("webhook URL not provided (use --webhook or --webhook-env)", file=sys.stderr)
        return 1

    prompts = _load_prompts(args)
    if not prompts:
        print("no prompts provided (use --prompts or --prompts-file)", file=sys.stderr)
        return 1

    started_at = datetime.now(timezone.utc)
    if not args.quiet:
        print(f"=== START {started_at.isoformat()} sending {len(prompts)} prompts ===")

    failed = 0
    for i, prompt in enumerate(prompts, 1):
        if not args.quiet:
            print(f"  [{i}/{len(prompts)}] {prompt!r}")
        ok, err = _post_webhook(webhook, prompt)
        if not ok:
            failed += 1
            print(f"    POST failed: {err}", file=sys.stderr)
        if i < len(prompts):
            time.sleep(args.interval_s)

    # Tail buffer for the last response.
    if not args.quiet:
        print(f"  waiting {args.interval_s:.0f}s for last response to settle...")
    time.sleep(args.interval_s)

    rows = _scan_log(Path(args.log_path).expanduser(), started_at, prompts)
    _emit_table(rows, total_sent=len(prompts))

    if args.output:
        Path(args.output).expanduser().write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not args.quiet:
            print(f"  wrote {len(rows)} row(s) to {args.output}")

    if failed:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
