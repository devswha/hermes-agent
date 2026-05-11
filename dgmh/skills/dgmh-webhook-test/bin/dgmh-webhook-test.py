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
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional


_DEFAULT_LOG = Path("~/.hermes/logs/agent.log").expanduser()
_DEFAULT_INTERVAL_S = 35.0
_DEFAULT_SCORER_BIN = Path(
    "~/workspace/hermes-agent/venv/bin/python"
).expanduser()
_DISCORD_API = "https://discord.com/api/v10"
_DISCORD_UA = "DiscordBot (https://github.com/devswha/dgmh, 0.1)"

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
_REACTION_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) INFO "
    r"dgmh\.hermes_integration\.reaction_hook: \[reaction_hook\] "
    r"Reaction received: user=(?P<user>\d+) channel=(?P<chan>\d+) "
    r"emoji=(?P<emoji>\S+) msg=(?P<msg>\d+)$"
)
_REACTION_CLASS_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) INFO "
    r"dgmh\.hermes_integration\.reaction_hook: \[reaction_hook\] "
    r"Reaction user=(?P<user>\d+) emoji=(?P<emoji>\S+) "
    r"classification=(?P<class>[\w_]+)"
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


def _resolve_env_value(env_file: Optional[str], keys: tuple[str, ...]) -> Optional[str]:
    """Pull the first matching `KEY=value` line from a dotenv-style file."""
    if not env_file:
        return None
    path = Path(env_file).expanduser()
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for key in keys:
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip()
    return None


def _discord_get(url: str, token: str) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bot {token}",
            "User-Agent": _DISCORD_UA,
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=20.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _resolve_bot_user_id(token: str) -> Optional[str]:
    try:
        me = _discord_get(f"{_DISCORD_API}/users/@me", token)
    except Exception as e:
        print(f"  could not resolve bot user id: {e}", file=sys.stderr)
        return None
    bid = me.get("id")
    return str(bid) if bid else None


def _fetch_channel_messages(
    token: str,
    channel_id: str,
    limit: int = 100,
) -> list[dict]:
    """Fetch up to `limit` most-recent messages from a channel (newest first)."""
    try:
        return _discord_get(
            f"{_DISCORD_API}/channels/{channel_id}/messages?limit={limit}",
            token,
        )
    except Exception as e:
        print(f"  fetch_channel_messages({channel_id}) failed: {e}", file=sys.stderr)
        return []


def _discord_ts_to_dt(ts: str) -> Optional[datetime]:
    """Parse Discord ISO8601 timestamp into aware UTC datetime."""
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts).astimezone(timezone.utc)
    except Exception:
        return None


def _attach_response_text(
    rows: list[dict[str, Any]],
    token: str,
    bot_user_id: Optional[str],
) -> None:
    """For each row, fetch the bot's first reply after the inbound timestamp
    in the same channel and attach `response_text` + `response_msg_id`.
    """
    by_channel: dict[str, list[dict]] = {}
    for row in rows:
        chan = row["chat"]
        if chan not in by_channel:
            by_channel[chan] = _fetch_channel_messages(token, chan, limit=100)

    for row in rows:
        chan = row["chat"]
        msgs = by_channel.get(chan) or []
        inb_ts = _parse_ts(row["ts"])
        # Discord ts are UTC-aware; convert inbound (was naive UTC) — we
        # parse log ts as UTC-aware already (_parse_ts).
        candidates = []
        for m in msgs:
            author = m.get("author") or {}
            aid = str(author.get("id") or "")
            m_ts_raw = m.get("timestamp") or ""
            m_ts = _discord_ts_to_dt(m_ts_raw)
            if m_ts is None or m_ts <= inb_ts:
                continue
            # only count messages authored by the bot, when we know its id
            if bot_user_id and aid != bot_user_id:
                continue
            candidates.append((m_ts, m))
        candidates.sort(key=lambda pair: pair[0])
        if candidates:
            _ts, msg = candidates[0]
            row["response_text"] = msg.get("content") or ""
            row["response_msg_id"] = str(msg.get("id") or "")
        else:
            row.setdefault("response_text", None)
            row.setdefault("response_msg_id", None)


_SCORE_HELPER = Path(__file__).resolve().parent / "_score_helper.py"


def _attach_reactions(
    rows: list[dict[str, Any]],
    log_path: Path,
    started_at: datetime,
) -> None:
    """Scan agent.log for `Reaction received` events, pair them with the
    follow-up `classification=` line, and attach each reaction to the row
    whose response_msg_id matches. Reactions are appended to row.reactions
    as a list of {ts, emoji, user, classification} dicts.
    """
    by_msg: dict[str, list[dict[str, Any]]] = {}
    pending_class: dict[tuple[str, str], dict[str, Any]] = {}

    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  reaction scan failed ({log_path}): {e}", file=sys.stderr)
        return

    for line in text.splitlines():
        m_rcv = _REACTION_RE.match(line)
        if m_rcv:
            ts = _parse_ts(m_rcv.group("ts"))
            if ts < started_at:
                continue
            entry = {
                "ts": m_rcv.group("ts"),
                "user": m_rcv.group("user"),
                "channel": m_rcv.group("chan"),
                "emoji": m_rcv.group("emoji"),
                "classification": None,
            }
            msg_id = m_rcv.group("msg")
            by_msg.setdefault(msg_id, []).append(entry)
            # Stash a key for the upcoming classification line.
            pending_class[(m_rcv.group("user"), m_rcv.group("emoji"))] = entry
            continue
        m_cls = _REACTION_CLASS_RE.match(line)
        if m_cls:
            key = (m_cls.group("user"), m_cls.group("emoji"))
            target = pending_class.pop(key, None)
            if target is not None:
                target["classification"] = m_cls.group("class")

    # Collect every reaction as (ts, channel, msg_id, entry) so we can
    # time-window match per-row when msg_id alone misses (e.g., the
    # operator reacts to the inbound message, not the bot's reply).
    all_events: list[tuple[datetime, str, str, dict[str, Any]]] = []
    for msg_id, entries in by_msg.items():
        for entry in entries:
            ts = _parse_ts(entry["ts"])
            all_events.append((ts, entry["channel"], msg_id, entry))

    # Default window upper bound is 5 minutes, but for fast iteration
    # testing each row's window is clamped at the NEXT row's inbound ts
    # so reactions don't bleed across prompts.
    max_window = timedelta(minutes=5)
    row_starts = [_parse_ts(r["ts"]) for r in rows]

    for idx, row in enumerate(rows):
        rmsg = row.get("response_msg_id")
        chan = row["chat"]
        inb_ts = row_starts[idx]
        upper = inb_ts + max_window
        if idx + 1 < len(rows):
            next_start = row_starts[idx + 1]
            if next_start > inb_ts:
                upper = min(upper, next_start)
        matched: list[dict[str, Any]] = []
        seen: set[str] = set()
        # 1. exact msg_id match on bot's reply (operator 👍/👎/✨ on reply)
        if rmsg and rmsg in by_msg:
            for entry in by_msg[rmsg]:
                key = entry["ts"] + entry["emoji"] + entry["user"]
                if key not in seen:
                    seen.add(key)
                    matched.append(entry)
        # 2. time-window match on this row's channel (bot's auto-ack on
        # the inbound, third-party reactions sharing the window, etc.)
        for ts, ev_chan, _msg_id, entry in all_events:
            if ev_chan != chan:
                continue
            if not (inb_ts <= ts <= upper):
                continue
            key = entry["ts"] + entry["emoji"] + entry["user"]
            if key not in seen:
                seen.add(key)
                matched.append(entry)
        row["reactions"] = matched


def _score_text(text: str, scorer_bin: Path) -> dict[str, Any]:
    """Run score_humanness in the hermes-agent venv subprocess. Returns
    a dict with ai_score, human_likeness, interpretation, or score_error.
    """
    if not text or not text.strip():
        return {"ai_score": None, "human_likeness": None, "score_error": "empty"}
    if not scorer_bin.exists():
        return {
            "ai_score": None,
            "human_likeness": None,
            "score_error": f"scorer python not found at {scorer_bin}",
        }
    if not _SCORE_HELPER.exists():
        return {
            "ai_score": None,
            "human_likeness": None,
            "score_error": f"score helper not found at {_SCORE_HELPER}",
        }
    try:
        proc = subprocess.run(
            [str(scorer_bin), str(_SCORE_HELPER)],
            input=text,
            capture_output=True,
            text=True,
            timeout=60.0,
        )
    except subprocess.TimeoutExpired:
        return {"ai_score": None, "human_likeness": None, "score_error": "timeout"}
    except Exception as e:
        return {
            "ai_score": None,
            "human_likeness": None,
            "score_error": f"{type(e).__name__}: {e}",
        }
    out = (proc.stdout or "").strip().splitlines()
    if not out:
        return {
            "ai_score": None,
            "human_likeness": None,
            "score_error": (
                f"empty stdout (exit={proc.returncode}, "
                f"stderr={proc.stderr.strip()[:160]!r})"
            ),
        }
    try:
        return json.loads(out[-1])
    except Exception as e:
        return {
            "ai_score": None,
            "human_likeness": None,
            "score_error": f"json parse: {e}; stdout_tail={out[-1][:160]!r}",
        }


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


_LOCAL_TZ = datetime.now().astimezone().tzinfo


def _parse_ts(ts_raw: str) -> datetime:
    """agent.log stamps are in the system local timezone (Hermes runs
    with Python's default logging, which renders local). Convert to UTC
    so the result compares correctly with Discord REST timestamps.
    """
    head, _, _ = ts_raw.partition(",")
    naive = datetime.strptime(head, "%Y-%m-%d %H:%M:%S")
    return naive.replace(tzinfo=_LOCAL_TZ).astimezone(timezone.utc)


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
        "| # | prompt | response | patina | gate | api | elapsed | ai_score | reactions |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
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
        if r.get("ai_score") is not None:
            ai = f"{r['ai_score']:.1f}"
        elif r.get("score_error"):
            ai = f"err: {str(r['score_error'])[:18]}"
        else:
            ai = "-"
        reacts = r.get("reactions") or []
        rx = "".join(rr.get("emoji") or "" for rr in reacts) or "-"
        print(
            f"| {i} | {p} | {resp} | {pat} | {gate} | {api} | {el} | {ai} | {rx} |"
        )


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
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="skip patina_judge.score_humanness on each response",
    )
    parser.add_argument(
        "--bot-token",
        help="Discord bot token (overrides --webhook-env DISCORD_TOKEN)",
    )
    parser.add_argument(
        "--scorer-bin",
        default=str(_DEFAULT_SCORER_BIN),
        help=(
            "python interpreter that can import dgmh.patina_judge "
            f"(default: {_DEFAULT_SCORER_BIN})"
        ),
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

    # Attach reactions (US-002). Cheap, always runs — reaction events
    # come from agent.log so this is a pure local scan, no extra REST.
    if rows:
        _attach_reactions(rows, Path(args.log_path).expanduser(), started_at)

    # Score path — fetch actual reply text via Discord REST API then
    # call score_humanness. Skipped when --no-score or token absent.
    if not args.no_score and rows:
        bot_token = args.bot_token or _resolve_env_value(
            args.webhook_env, ("DISCORD_TOKEN", "DISCORD_BOT_TOKEN")
        )
        if not bot_token:
            print(
                "  (no DISCORD_TOKEN — skipping response fetch + scoring)",
                file=sys.stderr,
            )
        else:
            if not args.quiet:
                print("  fetching reply text via Discord REST...")
            bot_user_id = _resolve_bot_user_id(bot_token)
            _attach_response_text(rows, bot_token, bot_user_id)
            scorer_bin = Path(args.scorer_bin).expanduser()
            if not args.quiet:
                print(f"  scoring {len(rows)} responses with {scorer_bin}")
            for r in rows:
                text = r.get("response_text") or ""
                r.update(_score_text(text, scorer_bin))

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
