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
import random
import re
import threading
import time
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

# 3+ bullet items at the start of lines (markdown - or * or numbered).
# Lowered from 4+ to 3+ after live-Discord case where a 3-bullet "이 세 범위"
# reply slipped through scoring as ai=2.2 / flags=[] but read as chatgpt-style.
_BULLET_LINE_RE = re.compile(r"^\s*(?:[-*]|\d+\.)\s+\S", re.MULTILINE)
# Bold markdown labels used to head a list:
#   **핵심:**, **주제:**             (colon inside bold)
#   **이 세 범위**야:, **요약**입니다: (colon outside bold + short suffix)
# Either form reads as bot-style label-prefix.
_BOLD_LABEL_RE = re.compile(
    r"\*\*[^*\n]{1,30}\*\*[가-힣ㄱ-ㅎ\sA-Za-z]{0,8}[:：]"
    r"|\*\*[^*\n]{1,30}[:：]\s*\*\*"
)
# ATX-style markdown headers (## ~~, ### ~~) — chatgpt-style tutorial structure
# in Discord casual chat reads as bot-like. One header is acceptable for a
# truly long doc; 2+ in a single reply is the failure mode.
_ATX_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S", re.MULTILINE)
# Chunk continuation markers like "(1/4)", "(2/3)", "(part 2/3)"
_CHUNK_MARKER_RE = re.compile(r"\(\s*(?:part\s*)?\d+\s*/\s*\d+\s*\)\s*$", re.IGNORECASE)
# Colon-introducing-list: a line ending in `:` followed (optionally after
# blank lines) by 3+ bullet/numbered items. Allows blank lines between the
# colon line and the bullets — Discord markdown often inserts them.
_COLON_INTRO_RE = re.compile(
    r"[^\n]+[:：]\s*\n+(?:\s*(?:[-*]|\d+\.)\s+\S[^\n]*\n+){3,}",
    re.MULTILINE,
)
# Closing-caveat hedge — start of last paragraph OR start of last sentence.
# Matches `\n` or sentence boundary `.` `!` `?` followed by hedge token,
# anchored to the tail of the response.
_CLOSING_HEDGE_RE = re.compile(
    r"(?:[.!?]\s+|\n\s*)(?:그래도|다만|물론|한편)\s+[^\n]+[.!?]?\s*$"
)

# Unprompted bio-recitation. The operator already knows the bot is their
# personal assistant on Hermes Agent with DGM-H — repeating it in casual
# replies reads as bot-like self-promotion. Trigger when 2+ of these
# phrases appear together in a single response.
_BIO_RECITATION_TOKENS = (
    "Hermes Agent",
    "DGM-H",
    "self-evolution",
    "self-evolving",
    "self improvement",
    "persona",
    "개인 어시스턴트",
    "예전 flask 프로젝트",
    "flask 프로젝트랑은 무관",
    "flask 프로젝트와는 무관",
    "튜닝되는 skill",
    "튜닝되는 스킬",
    "evolve via DGM-H",
    "feedback-driven self",
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
    if len(bullets) >= 3:
        flags.append(f"bullet-list-3plus({len(bullets)})")

    if _BOLD_LABEL_RE.search(stripped):
        flags.append("bold-label-header")

    atx_headers = _ATX_HEADER_RE.findall(stripped)
    if len(atx_headers) >= 2:
        flags.append(f"atx-headers({len(atx_headers)})")

    if _CHUNK_MARKER_RE.search(stripped):
        flags.append("chunk-marker")

    if _COLON_INTRO_RE.search(stripped):
        flags.append("colon-introducing-list")

    if _CLOSING_HEDGE_RE.search(stripped):
        flags.append("closing-caveat-hedge")

    bio_hits = [t for t in _BIO_RECITATION_TOKENS if t in stripped]
    if len(bio_hits) >= 2:
        flags.append(f"bio-recitation({len(bio_hits)}:{','.join(bio_hits[:3])})")

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


_PRUNE_RECENT_LOOKBACK_S = 60.0


def _prune_polluting_message(
    *,
    message_id: Optional[str] = None,
    ai_score: float,
    lookback_s: float = _PRUNE_RECENT_LOOKBACK_S,
) -> int:
    """Delete the most-recent polluting assistant row from state.db.messages.

    Step 5 (v3): keys by recency rather than content match.

    Hermes' state.db ``messages`` schema is keyed by an internal
    auto-increment ``id``; it does NOT carry a Discord ``message_id``
    column. The previous implementation matched by ``content``, which
    silently broke whenever any mid-flight rewrite stage (humanness
    rewrite, future patina-profile rewrite) mutated the outbound text
    — Hermes core wrote the draft before the wrap, so the post-rewrite
    text never matched any row.

    The fix: delete the single most-recent ``role='assistant'`` row
    inserted within ``lookback_s`` seconds of "now". Because the
    background score thread fires immediately after ``adapter.send``,
    that row is overwhelmingly the polluting reply we just scored.

    ``message_id`` is the Discord message id (passed in for telemetry
    only — it does not appear in the state.db schema, so we cannot
    filter on it; logged for traceability).

    Returns the number of rows deleted (0 or 1 in practice).

    Disabled by setting ``DGMH_PRUNE_DISABLED``. Threshold overridden
    via ``DGMH_PRUNE_AI_THRESHOLD`` (default 15.0).
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
        # Two-step delete-by-id so we only ever remove ONE row even if
        # several assistant messages fall inside the lookback window.
        row = cur.execute(
            "SELECT id FROM messages "
            "WHERE role = 'assistant' "
            "AND timestamp > strftime('%s','now') - ? "
            "ORDER BY timestamp DESC, id DESC LIMIT 1",
            (lookback_s,),
        ).fetchone()
        if row is None:
            con.close()
            return 0
        target_id = row[0]
        n = cur.execute(
            "DELETE FROM messages WHERE id = ?", (target_id,)
        ).rowcount
        con.commit()
        con.close()
        if n and message_id:
            logger.info(
                "humanness_hook: pruned state.db row id=%s for discord msg=%s "
                "(ai=%.1f >= threshold)",
                target_id, message_id, ai_score,
            )
        return n
    except Exception:
        logger.exception("humanness_hook: prune query failed")
        return 0


# ---------------------------------------------------------------------------
# Step 2 (v3): bimodal humanlike latency with operator shortcut
# ---------------------------------------------------------------------------
#
# Real humans don't respond in 200ms. They take longer when the channel is
# warm (recent activity) and longer still when the channel is cold (no
# activity for ages). The bot's instant reply is one of the strongest tells
# in public chat. Bimodal latency closes that gap.
#
# Buckets (gated by DGMH_PUBLIC_HUMAN_MODE=1 AND public channel):
#   active (last_msg < 30s)   → max(3, min(20,  gauss(8,   4)))
#   warm   (last_msg < 600s)  → max(10, min(120, gauss(35,  20)))
#   cold   (otherwise)        → max(60, min(900, gauss(180, 90)))
#
# AC1.1: when the bot is replying TO the operator (author is operator),
# the operator-shortcut bucket fires regardless of activity state:
#   operator → max(3, min(15, gauss(8, 4)))
# This keeps urgent operator pings from waiting up to 15 minutes mid-night.
#
# Operator user-id default: 266436073557590016. Override via
# DGMH_OPERATOR_USER_ID. The wrapped_send call site resolves is_operator
# from metadata["author_id"] when available, falling back to that env id.

_DEFAULT_OPERATOR_USER_ID = "266436073557590016"

# In-memory map: channel_id -> last bot-send unix timestamp. Keeps Step 2's
# bucket lookup honest about when this surface last had bot activity.
_LAST_BOT_SEND_TS: dict[str, float] = {}


def _operator_user_ids() -> set[str]:
    raw = os.environ.get("DGMH_OPERATOR_USER_ID", _DEFAULT_OPERATOR_USER_ID)
    return {x.strip() for x in raw.split(",") if x.strip()}


def _resolve_is_operator(
    metadata: Optional[dict[str, Any]],
    *,
    author_id: Optional[str] = None,
) -> bool:
    """Best-effort check of whether the inbound author is the operator.

    Reads ``metadata["author_id"]`` first (Hermes adapter passes it through
    when available), then falls back to an explicit ``author_id`` arg.
    Returns False on any unexpected shape so a missing/garbled metadata
    blob just demotes the call to the regular bimodal path.
    """
    candidate: Optional[str] = author_id
    if candidate is None and isinstance(metadata, dict):
        candidate = metadata.get("author_id") or metadata.get("user_id")
    if candidate is None:
        return False
    try:
        return str(candidate) in _operator_user_ids()
    except Exception:
        return False


def _seconds_since_last_msg(channel_id: str, *, now: Optional[float] = None) -> float:
    """Seconds since this surface last saw a bot send.

    Returns a large sentinel (1e9) when no prior send is recorded so a
    cold-start surface lands in the cold bucket.
    """
    last = _LAST_BOT_SEND_TS.get(str(channel_id))
    if last is None:
        return 1e9
    return max(0.0, (now if now is not None else time.time()) - last)


def _record_bot_send(channel_id: str, *, ts: Optional[float] = None) -> None:
    _LAST_BOT_SEND_TS[str(channel_id)] = ts if ts is not None else time.time()


def _human_latency_seconds(
    channel_id: str,
    *,
    is_operator: bool,
    rng: Optional[random.Random] = None,
    now: Optional[float] = None,
) -> float:
    """Sample a humanlike pre-send sleep in seconds for ``channel_id``.

    See module-level docstring for bucket definitions. ``rng`` is exposed
    so unit tests can pin the distribution; production callers pass None
    and use the module-default RNG.
    """
    rng_ = rng or random
    if is_operator:
        return max(3.0, min(15.0, rng_.gauss(8.0, 4.0)))
    last_ago = _seconds_since_last_msg(str(channel_id), now=now)
    if last_ago < 30.0:
        return max(3.0, min(20.0, rng_.gauss(8.0, 4.0)))
    if last_ago < 600.0:
        return max(10.0, min(120.0, rng_.gauss(35.0, 20.0)))
    return max(60.0, min(900.0, rng_.gauss(180.0, 90.0)))


def _public_human_mode_enabled() -> bool:
    """Whether DGMH_PUBLIC_HUMAN_MODE is set to a truthy value."""
    val = os.environ.get("DGMH_PUBLIC_HUMAN_MODE", "")
    return val not in ("", "0", "false", "False")


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
        # Step 5 (v3): prune is keyed by recency, not content, so the
        # rewrite stages can mutate ``content`` mid-flight without breaking
        # the prune.
        pruned_pre = _prune_polluting_message(
            message_id=message_id, ai_score=999.0
        )
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
                    message_id=message_id, ai_score=result.ai_score
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

        # Step 2 (v3): bimodal humanlike pre-send sleep for public channels.
        # Gated by DGMH_PUBLIC_HUMAN_MODE=1 AND public channel — with the
        # env unset (default), behavior is byte-identical to the prior
        # adapter.send pipeline so the 1:1 verifier baseline holds.
        try:
            if _public_human_mode_enabled():
                from dgmh.honcho_client import is_public_channel

                if is_public_channel(str(chat_id)):
                    is_op = _resolve_is_operator(metadata)
                    sleep_s = _human_latency_seconds(
                        str(chat_id), is_operator=is_op
                    )
                    logger.info(
                        "[humanness_hook] bimodal sleep %.2fs for chat=%s is_op=%s",
                        sleep_s, chat_id, is_op,
                    )
                    await asyncio.sleep(sleep_s)
        except Exception:
            logger.exception(
                "[humanness_hook] bimodal latency failed; sending without sleep"
            )

        result = await original_send(chat_id, content, reply_to=reply_to, metadata=metadata)

        # Step 2 (v3): record this send timestamp so the next reply on
        # this channel sees the up-to-date last-message-ago bucket.
        try:
            _record_bot_send(str(chat_id))
        except Exception:
            pass

        try:
            if not _should_score(content):
                return result

            thread_id = (metadata or {}).get("thread_id")
            message_id = getattr(result, "message_id", None) or (
                result.get("message_id") if isinstance(result, dict) else None
            )

            # Step 6 (v3): capture context with the resolved channel kind so
            # _score_in_thread sees the right channel_kind ContextVar (used
            # transitively by any honcho-client read it triggers, e.g. via
            # the structural pre-prune path).
            from dgmh.honcho_client import (
                capture_context_with_kind,
                channel_kind_for,
            )

            ctx = capture_context_with_kind(channel_kind_for(str(chat_id)))
            thread = threading.Thread(
                target=ctx.run,
                args=(_score_in_thread,),
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

    # Wrap-order markers (Step 0): expose hook identity + the original
    # callable so dgmh.hermes_integration.wrap_order can walk the chain
    # outer→inner and assert ``["honcho", "humanness"]`` at startup.
    wrapped_send.__dgmh_hook_name__ = "humanness"  # type: ignore[attr-defined]
    wrapped_send.__wrapped__ = original_send  # type: ignore[attr-defined]

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
        # Best-effort wrap-order check; partner hook (honcho) may not have
        # wrapped yet on startup, in which case this logs a warning. The
        # last hook to finish wrapping observes the full chain.
        try:
            from dgmh.hermes_integration.wrap_order import verify_wrap_chain

            verify_wrap_chain(adapter)
        except Exception:
            logger.exception(
                "[humanness_hook] wrap-order verification raised; continuing"
            )
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
