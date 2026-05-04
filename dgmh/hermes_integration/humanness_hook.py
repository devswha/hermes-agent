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

    try:
        from dgmh.patina_judge import score_humanness, PatinaScoreError

        try:
            result = score_humanness(content, lang="ko")
            pruned_count = _prune_polluting_message(content, ai_score=result.ai_score)
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
