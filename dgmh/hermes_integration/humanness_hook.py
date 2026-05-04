"""DGM-H humanness hook — wires Discord outbound messages to async patina scoring.

Registered at ``gateway:startup``. Wraps the DiscordAdapter's ``send`` method
to fire fire-and-forget patina scoring on every assistant message. Results are
appended to ``~/.hermes/dgmh/humanness_log.jsonl`` for use as a post-hoc
fitness signal in the DGM-H evolution composite reward.

Skip rules (don't score):
  - Empty content
  - Length < ``DGMH_HUMANNESS_MIN_CHARS`` (default 30) — too short to score reliably
  - Begins with "Error:" or runtime warning glyphs — runtime error frames, not assistant prose
  - DGMH_HUMANNESS_DISABLED env set
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Optional

from dgmh.humanness_log import (
    append_record,
    make_error_record,
    make_success_record,
)

logger = logging.getLogger(__name__)


_DEFAULT_MIN_CHARS = 30
_ERROR_PREFIXES = ("Error:", "⚠️", "❌", "[error]")
_PATCH_FLAG = "_dgmh_humanness_patched"

# Pruning: when an assistant reply scores at or above this AI-likeness
# threshold, the row in state.db.messages is deleted so it does not
# pollute the conversation history that Hermes feeds into the next
# system prompt build. This breaks the self-reinforcing loop where the
# bot mimics its own prior chatgpt-styled replies.
_DEFAULT_PRUNE_THRESHOLD = 15.0
_PRUNE_LOOKBACK_SECONDS = 600.0


# Deterministic pre-check: regex patterns that flag obvious chatgpt-style
# structure before paying for a Codex patina call. When any of these fire,
# prune immediately. Patina runs anyway in parallel for telemetry and to
# catch nuanced AI-tone that the regex cannot see.

# 4+ bullet items at the start of lines (markdown - or * or numbered).
_BULLET_LINE_RE = re.compile(r"^\s*(?:[-*]|\d+\.)\s+\S", re.MULTILINE)
# Bold markdown headers used as labels: **핵심:**, **주제:**, **요약:**, **결론:**
_BOLD_LABEL_RE = re.compile(r"\*\*[^*\n]{1,30}[:：]\s*\*\*")
# Colon-introducing-list: line ending in colon with bullets that follow
_COLON_INTRO_RE = re.compile(
    r"[^\n]+[:：]\s*\n(?:\s*(?:[-*]|\d+\.)\s+\S+\s*\n){3,}",
    re.MULTILINE,
)
# Closing-caveat hedge — start of last paragraph OR start of last sentence.
# Matches `\n` or sentence boundary `.` `!` `?` followed by hedge token,
# anchored to the tail of the response.
_CLOSING_HEDGE_RE = re.compile(
    r"(?:[.!?]\s+|\n\s*)(?:그래도|다만|물론|한편)\s+[^\n]+[.!?]?\s*$"
)


def _structural_pollution_check(content: str) -> tuple[bool, list[str]]:
    """Return (should_prune, list_of_matched_pattern_names).

    Conservative — fires only on patterns the operator has flagged as
    chatgpt-tells in casual chat. Skips when the response is mostly a code
    block (the patterns inside fenced code don't count).
    """
    flags = []

    # Strip fenced code blocks before checking; bullets inside code are fine.
    stripped = re.sub(r"```[\s\S]*?```", "", content)

    bullets = _BULLET_LINE_RE.findall(stripped)
    if len(bullets) >= 4:
        flags.append(f"bullet-list-4plus({len(bullets)})")

    if _BOLD_LABEL_RE.search(stripped):
        flags.append("bold-label-header")

    if _COLON_INTRO_RE.search(stripped):
        flags.append("colon-introducing-list")

    if _CLOSING_HEDGE_RE.search(stripped):
        flags.append("closing-caveat-hedge")

    return (bool(flags), flags)


def _read_soul_hash() -> str:
    """Compute sha256 of the active SOUL.md content (used as generation tag)."""
    soul_path = (
        Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")) / "SOUL.md"
    )
    try:
        content = soul_path.read_text(encoding="utf-8")
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    except Exception:
        return ""


def _should_score(content: str) -> bool:
    if os.environ.get("DGMH_HUMANNESS_DISABLED"):
        return False
    if not content or not content.strip():
        return False
    min_chars = int(os.environ.get("DGMH_HUMANNESS_MIN_CHARS", _DEFAULT_MIN_CHARS))
    if len(content.strip()) < min_chars:
        return False
    stripped = content.lstrip()
    for prefix in _ERROR_PREFIXES:
        if stripped.startswith(prefix):
            return False
    return True


def _prune_polluting_message(content: str, *, ai_score: float) -> int:
    """Delete the polluting assistant row from state.db.messages.

    Matches by exact content + role=assistant + recent timestamp so the
    delete is conservative — same prose in the last few minutes is almost
    certainly the message we just scored. Returns the number of rows
    deleted (0 or 1 in practice).

    Disabled by setting DGMH_PRUNE_DISABLED. Threshold overridden via
    DGMH_PRUNE_AI_THRESHOLD (default 15.0).
    """
    import sqlite3

    if os.environ.get("DGMH_PRUNE_DISABLED"):
        return 0

    threshold = float(
        os.environ.get("DGMH_PRUNE_AI_THRESHOLD", _DEFAULT_PRUNE_THRESHOLD)
    )
    if ai_score < threshold:
        return 0

    db_path = (
        Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")) / "state.db"
    )
    if not db_path.exists():
        logger.info("humanness_hook: state.db not found at %s, skip prune", db_path)
        return 0

    try:
        con = sqlite3.connect(str(db_path), timeout=5.0)
        cur = con.cursor()
        n = cur.execute(
            "DELETE FROM messages "
            "WHERE role = 'assistant' AND content = ? "
            "AND timestamp > strftime('%s','now') - ?",
            (content, _PRUNE_LOOKBACK_SECONDS),
        ).rowcount
        con.commit()
        con.close()
        return n
    except Exception:
        logger.exception("humanness_hook: prune query failed")
        return 0


def _score_in_thread(
    *,
    content: str,
    chat_id: str,
    thread_id: Optional[str],
    message_id: Optional[str],
) -> None:
    """Run patina scoring on a background thread and append the record."""
    soul_hash = _read_soul_hash()
    text_length = len(content)
    pruned_count = 0

    # Step 1: deterministic pre-check. Prune obvious structural pollution
    # without waiting for the slow Codex patina round-trip.
    structural_hit, struct_flags = _structural_pollution_check(content)
    if structural_hit and not os.environ.get("DGMH_PRUNE_DISABLED"):
        # Force prune by passing a synthetic high score above threshold.
        pruned_pre = _prune_polluting_message(content, ai_score=999.0)
        if pruned_pre:
            logger.info(
                "[humanness_hook] structural pre-prune (%s) removed %d row",
                ",".join(struct_flags), pruned_pre,
            )
            pruned_count += pruned_pre

    try:
        from dgmh.patina_judge import score_humanness, PatinaScoreError

        try:
            result = score_humanness(content, lang="ko")
            # Only attempt patina-based prune if the structural pre-check did
            # not already remove the row.
            if pruned_count == 0:
                pruned_count = _prune_polluting_message(
                    content, ai_score=result.ai_score
                )
                if pruned_count:
                    logger.info(
                        "[humanness_hook] pruned %d polluting reply (ai=%.1f >= threshold)",
                        pruned_count, result.ai_score,
                    )
            record = make_success_record(
                chat_id=chat_id,
                thread_id=thread_id,
                message_id=message_id,
                soul_md_hash=soul_hash,
                text_length=text_length,
                ai_score=result.ai_score,
                human_likeness=result.human_likeness,
                sub_scores=result.sub_scores,
                interpretation=result.interpretation,
                elapsed_s=result.elapsed_s,
            )
            record["pruned"] = pruned_count
            record["structural_flags"] = struct_flags
        except PatinaScoreError as exc:
            record = make_error_record(
                chat_id=chat_id,
                thread_id=thread_id,
                message_id=message_id,
                soul_md_hash=soul_hash,
                text_length=text_length,
                error=f"{type(exc).__name__}: {exc}",
            )
    except Exception as exc:  # noqa: BLE001 — never crash the gateway from a side hook
        record = make_error_record(
            chat_id=chat_id,
            thread_id=thread_id,
            message_id=message_id,
            soul_md_hash=soul_hash,
            text_length=text_length,
            error=f"unexpected: {type(exc).__name__}: {exc}",
        )

    try:
        append_record(record)
    except Exception:
        logger.exception("humanness_hook: failed to append record")


def _wrap_send(adapter: Any) -> None:
    """Monkey-patch the adapter's ``send`` to async-score outbound content."""
    if getattr(adapter, _PATCH_FLAG, False):
        logger.info("[humanness_hook] send already patched, skipping")
        return

    original_send = adapter.send

    async def wrapped_send(
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        # Pre-send rewrite: if the outbound content has clear structural
        # pollution AND rewrite is enabled, rewrite it via Codex before
        # actually posting to Discord. The user sees only the cleaned text.
        # This is gated by DGMH_REWRITE_ENABLED to allow disabling in tests
        # or under high latency budgets.
        rewrite_enabled = (
            bool(os.environ.get("DGMH_REWRITE_ENABLED"))
            and not os.environ.get("DGMH_REWRITE_DISABLED")
        )
        if rewrite_enabled and _should_score(content):
            structural_hit, struct_flags = _structural_pollution_check(content)
            if structural_hit:
                try:
                    from dgmh.patina_judge import humanness_rewrite

                    rewritten = await asyncio.to_thread(
                        humanness_rewrite, content, timeout_s=60.0
                    )
                    if rewritten and rewritten != content:
                        # Sanity: don't replace if rewrite still has pollution.
                        rew_hit, _ = _structural_pollution_check(rewritten)
                        if not rew_hit:
                            logger.info(
                                "[humanness_hook] pre-send rewrite applied "
                                "(flags=%s len=%d→%d)",
                                ",".join(struct_flags),
                                len(content),
                                len(rewritten),
                            )
                            content = rewritten
                        else:
                            logger.info(
                                "[humanness_hook] rewrite still polluted (%s); "
                                "keeping original",
                                struct_flags,
                            )
                except Exception:
                    logger.exception(
                        "[humanness_hook] pre-send rewrite failed; using original"
                    )

        result = await original_send(chat_id, content, reply_to=reply_to, metadata=metadata)

        try:
            if not _should_score(content):
                return result

            thread_id = (metadata or {}).get("thread_id")
            message_id = getattr(result, "message_id", None) or (
                result.get("message_id") if isinstance(result, dict) else None
            )

            thread = threading.Thread(
                target=_score_in_thread,
                kwargs={
                    "content": content,
                    "chat_id": str(chat_id),
                    "thread_id": str(thread_id) if thread_id else None,
                    "message_id": str(message_id) if message_id else None,
                },
                daemon=True,
                name="dgmh-humanness-score",
            )
            thread.start()
        except Exception:
            logger.exception("[humanness_hook] failed to dispatch scoring; continuing")

        return result

    adapter.send = wrapped_send  # type: ignore[assignment]
    setattr(adapter, _PATCH_FLAG, True)
    logger.info("[humanness_hook] wrapped DiscordAdapter.send for patina scoring")


async def _wait_and_patch(adapter: Any) -> None:
    """Wait briefly for the adapter to fully connect, then wrap send."""
    try:
        for _ in range(60):
            client = getattr(adapter, "_client", None)
            if client and getattr(client, "is_ready", lambda: False)():
                break
            await asyncio.sleep(0.5)
        _wrap_send(adapter)
    except Exception:
        logger.exception("[humanness_hook] _wait_and_patch failed")


async def handle(event_type: str, context: dict) -> None:
    """Gateway startup hook — find DiscordAdapter and wrap its send method."""
    if event_type != "gateway:startup":
        return

    logger.info("[humanness_hook] gateway:startup — wiring outbound humanness scoring")

    from dgmh.hermes_integration.reaction_hook import _find_discord_adapter

    adapter = _find_discord_adapter(context)
    if adapter is None:
        logger.warning(
            "[humanness_hook] No DiscordAdapter found at startup — humanness "
            "scoring not wired."
        )
        return

    asyncio.create_task(_wait_and_patch(adapter))
    logger.info("[humanness_hook] Scheduled humanness wrap task")
