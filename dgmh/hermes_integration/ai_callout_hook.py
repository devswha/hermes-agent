"""DGM-H AI-callout hook — observe other bots' inbound messages and
optionally have the persona comment on AI-shaped replies.

Registered at ``gateway:startup``. Adds an ``on_message`` listener via
``client.add_listener`` so it runs alongside discord.py's primary
``on_message`` handler without competing with it.

Pipeline (per inbound bot message):

  1. Filter: only act on bot-authored messages in
     ``DGMH_AI_CALLOUT_CHANNELS``. Skip self, skip messages that
     reference one of our own replies (we already responded).
  2. Operator-active suppression: if a human spoke on this channel
     within ``DGMH_AI_CALLOUT_OPERATOR_QUIET_S`` seconds, stay silent —
     don't interrupt a live conversation.
  3. Cooldown: at most one callout per
     ``DGMH_AI_CALLOUT_COOLDOWN_S`` window per channel.
  4. Structural pre-check: reuse the cheap regex from humanness_hook
     to short-circuit obvious-non-AI text without paying for Codex.
  5. Score with ``score_humanness``; if ``ai_score >= threshold``,
     dispatch to the configured ``DGMH_AI_CALLOUT_MODE``:
       - ``disabled`` (default): never reach this branch at all.
       - ``dryrun``: append a row to ai_callout_log.jsonl, no send.
       - ``fixed``: random line from the phrasebook.
       - ``agent``: stub (logs + falls back to fixed for now).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


_PATCH_FLAG = "_dgmh_ai_callout_patched"

# Defaults — operator overrides via env.
_DEFAULT_THRESHOLD = 70.0
_DEFAULT_COOLDOWN_S = 600.0
_DEFAULT_OPERATOR_QUIET_S = 60.0
_DEFAULT_MIN_CHARS = 30


def _hermes_home() -> Path:
    raw = os.environ.get("HERMES_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".hermes"


def _log_path() -> Path:
    return _hermes_home() / "dgmh" / "ai_callout_log.jsonl"


def _phrasebook_path() -> Path:
    raw = os.environ.get("DGMH_AI_CALLOUT_PHRASEBOOK")
    if raw:
        return Path(raw).expanduser()
    return Path(__file__).resolve().parent.parent / "prompts" / "ai_callout_lines.txt"


# --- Env resolution ---------------------------------------------------------


def get_mode() -> str:
    mode = os.environ.get("DGMH_AI_CALLOUT_MODE", "disabled").strip().lower()
    if mode not in {"disabled", "dryrun", "fixed", "agent"}:
        return "disabled"
    return mode


def get_callout_channels() -> set[str]:
    raw = os.environ.get("DGMH_AI_CALLOUT_CHANNELS", "")
    return {c.strip() for c in raw.split(",") if c.strip()}


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def get_threshold() -> float:
    return _float_env("DGMH_AI_CALLOUT_THRESHOLD", _DEFAULT_THRESHOLD)


def get_cooldown_s() -> float:
    return _float_env("DGMH_AI_CALLOUT_COOLDOWN_S", _DEFAULT_COOLDOWN_S)


def get_operator_quiet_s() -> float:
    return _float_env("DGMH_AI_CALLOUT_OPERATOR_QUIET_S", _DEFAULT_OPERATOR_QUIET_S)


# --- State (per-process) ----------------------------------------------------


_state_lock = threading.Lock()
_last_callout_at: dict[str, float] = {}        # channel_id → monotonic ts
_last_human_msg_at: dict[str, float] = {}      # channel_id → monotonic ts


def record_human_activity(channel_id: str) -> None:
    """Operator/human spoke on this channel — start the quiet window."""
    with _state_lock:
        _last_human_msg_at[channel_id] = time.monotonic()


def _is_operator_active(channel_id: str, quiet_s: float) -> bool:
    with _state_lock:
        last = _last_human_msg_at.get(channel_id)
    if last is None:
        return False
    return (time.monotonic() - last) < quiet_s


def _is_cooldown_active(channel_id: str, cooldown_s: float) -> bool:
    with _state_lock:
        last = _last_callout_at.get(channel_id)
    if last is None:
        return False
    return (time.monotonic() - last) < cooldown_s


def _mark_callout(channel_id: str) -> None:
    with _state_lock:
        _last_callout_at[channel_id] = time.monotonic()


def _reset_state_for_tests() -> None:
    """Test helper — drop in-process state."""
    with _state_lock:
        _last_callout_at.clear()
        _last_human_msg_at.clear()


# --- Structural pre-check ---------------------------------------------------
# Lightweight regex pre-screen so we don't pay for a Codex round-trip on
# obviously human-shaped messages. Mirrors humanness_hook's pattern.


_BULLET_LINE_RE = re.compile(r"^\s*(?:[-*]|\d+\.)\s+\S", re.MULTILINE)
# Matches both `**Key:**` (colon inside bold — common in Korean) and
# `**Key**:` (colon outside bold). Either is the "label heading a list"
# pattern that's a strong AI-tell.
_BOLD_LABEL_RE = re.compile(
    r"\*\*[^*\n]{1,40}[:：]\s*\*\*|\*\*[^*\n]{1,40}\*\*\s*[:：]"
)


def _has_ai_structural_tells(text: str) -> bool:
    """Cheap deterministic check — True if the text shows obvious
    AI-style structure (multi-bullet lists, bold labels). Drives the
    pre-filter decision: when False, we may still score (the text could
    still be AI-toned without these tells) but the operator can tighten
    via DGMH_AI_CALLOUT_REQUIRE_STRUCTURAL=1.
    """
    if len(_BULLET_LINE_RE.findall(text)) >= 3:
        return True
    if _BOLD_LABEL_RE.search(text):
        return True
    return False


# --- Phrasebook -------------------------------------------------------------


def _load_phrasebook() -> list[str]:
    path = _phrasebook_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except Exception as e:
        logger.warning("[ai_callout_hook] phrasebook read failed: %s", e)
        return []
    lines = [
        ln.strip()
        for ln in raw.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    return lines


def pick_callout_line(rng: Optional[random.Random] = None) -> Optional[str]:
    lines = _load_phrasebook()
    if not lines:
        return None
    chooser = rng or random
    return chooser.choice(lines)


# --- Logging ----------------------------------------------------------------


def _append_log(record: dict) -> None:
    path = _log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("[ai_callout_hook] log append failed: %s", e)


def make_log_record(
    *,
    channel_id: str,
    msg_id: str,
    author_bot_id: str,
    text_length: int,
    ai_score: Optional[float],
    mode: str,
    action: str,
    reason: str = "",
) -> dict:
    """Build a JSONL row. Note: we intentionally do NOT log the scored
    bot text — third-party content stays out of our logs (operator
    decision Q5). Length-only telemetry is enough for calibration.
    """
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "channel_id": channel_id,
        "msg_id": msg_id,
        "author_bot_id": author_bot_id,
        "text_length": text_length,
        "ai_score": ai_score,
        "mode": mode,
        "action": action,
        "reason": reason,
    }


# --- Decision (sync, testable) ---------------------------------------------


def should_observe(
    *,
    is_bot: bool,
    is_self: bool,
    channel_id: str,
    text: str,
    callout_channels: set[str],
    min_chars: int = _DEFAULT_MIN_CHARS,
) -> tuple[bool, str]:
    """Decide if this inbound message is a callout candidate.

    Returns (proceed, reason). ``reason`` is the skip reason when
    ``proceed`` is False (useful for log telemetry).
    """
    if not is_bot:
        return False, "not_bot"
    if is_self:
        return False, "self"
    if channel_id not in callout_channels:
        return False, "channel_not_in_callout_set"
    if not text or len(text) < min_chars:
        return False, "too_short"
    return True, ""


# --- Scoring (off main thread) ---------------------------------------------


def _score_in_thread(text: str) -> Optional[float]:
    """Run patina score_humanness off the event loop. Returns ai_score
    in 0-100 or None on failure.
    """
    try:
        from dgmh.patina_judge import score_humanness, PatinaScoreError
    except Exception as e:
        logger.warning("[ai_callout_hook] patina import failed: %s", e)
        return None
    try:
        result = score_humanness(text, lang="ko")
    except PatinaScoreError as e:
        logger.debug("[ai_callout_hook] score_humanness error: %s", e)
        return None
    except Exception as e:
        logger.debug("[ai_callout_hook] score_humanness unexpected error: %s", e)
        return None
    try:
        return float(result.ai_score)
    except (AttributeError, TypeError, ValueError):
        return None


# --- Action dispatcher ------------------------------------------------------


async def _send_fixed_callout(adapter: Any, channel_id: str) -> str:
    """Pick a phrasebook line and send via adapter. Returns the line or
    empty string if the phrasebook was empty.
    """
    line = pick_callout_line()
    if not line:
        logger.warning("[ai_callout_hook] phrasebook empty — skipping send")
        return ""
    try:
        await adapter.send(channel_id, line)
    except Exception as e:
        logger.warning("[ai_callout_hook] adapter.send failed: %s", e)
        return ""
    return line


# --- Main observation entrypoint -------------------------------------------


async def observe_inbound_bot(
    *,
    adapter: Any,
    channel_id: str,
    msg_id: str,
    author_bot_id: str,
    text: str,
) -> None:
    """Top-level handler called from the on_message listener for messages
    that have ALREADY been confirmed bot-authored, in a callout channel,
    and not from ourselves. Performs cooldown/quiet checks, scores, and
    dispatches by mode.

    Errors are swallowed and logged — this path must never crash the
    Discord event loop.
    """
    mode = get_mode()
    if mode == "disabled":
        return

    quiet_s = get_operator_quiet_s()
    if _is_operator_active(channel_id, quiet_s):
        _append_log(
            make_log_record(
                channel_id=channel_id,
                msg_id=msg_id,
                author_bot_id=author_bot_id,
                text_length=len(text),
                ai_score=None,
                mode=mode,
                action="skipped",
                reason="operator_active",
            )
        )
        return

    cooldown_s = get_cooldown_s()
    if _is_cooldown_active(channel_id, cooldown_s):
        _append_log(
            make_log_record(
                channel_id=channel_id,
                msg_id=msg_id,
                author_bot_id=author_bot_id,
                text_length=len(text),
                ai_score=None,
                mode=mode,
                action="skipped",
                reason="cooldown",
            )
        )
        return

    # Score (off the event loop)
    ai_score = await asyncio.to_thread(_score_in_thread, text)
    if ai_score is None:
        _append_log(
            make_log_record(
                channel_id=channel_id,
                msg_id=msg_id,
                author_bot_id=author_bot_id,
                text_length=len(text),
                ai_score=None,
                mode=mode,
                action="skipped",
                reason="score_failed",
            )
        )
        return

    threshold = get_threshold()
    if ai_score < threshold:
        _append_log(
            make_log_record(
                channel_id=channel_id,
                msg_id=msg_id,
                author_bot_id=author_bot_id,
                text_length=len(text),
                ai_score=ai_score,
                mode=mode,
                action="below_threshold",
            )
        )
        return

    # Score >= threshold — dispatch by mode.
    if mode == "dryrun":
        _append_log(
            make_log_record(
                channel_id=channel_id,
                msg_id=msg_id,
                author_bot_id=author_bot_id,
                text_length=len(text),
                ai_score=ai_score,
                mode=mode,
                action="would_callout",
            )
        )
        _mark_callout(channel_id)
        return

    if mode == "fixed":
        line = await _send_fixed_callout(adapter, channel_id)
        _append_log(
            make_log_record(
                channel_id=channel_id,
                msg_id=msg_id,
                author_bot_id=author_bot_id,
                text_length=len(text),
                ai_score=ai_score,
                mode=mode,
                action="sent" if line else "send_failed",
                reason=line if line else "phrasebook_empty_or_send_error",
            )
        )
        if line:
            _mark_callout(channel_id)
        return

    if mode == "agent":
        # Phase 2: synthetic user-turn injection into the agent's reply
        # pipeline. Not yet implemented — falls back to fixed for safety
        # so the operator gets some signal, and the deferral is visible
        # in the log row.
        logger.info(
            "[ai_callout_hook] agent mode not yet implemented — falling back to fixed"
        )
        line = await _send_fixed_callout(adapter, channel_id)
        _append_log(
            make_log_record(
                channel_id=channel_id,
                msg_id=msg_id,
                author_bot_id=author_bot_id,
                text_length=len(text),
                ai_score=ai_score,
                mode=mode,
                action="agent_fallback_to_fixed" if line else "send_failed",
                reason=line if line else "phrasebook_empty_or_send_error",
            )
        )
        if line:
            _mark_callout(channel_id)
        return


# --- Discord client wiring -------------------------------------------------


def _is_self_referenced(message: Any, self_user_id: Optional[int]) -> bool:
    """True if this message is a reply to one of our own messages —
    don't pile a callout on top of an existing reply chain we're in.
    """
    if self_user_id is None:
        return False
    ref = getattr(message, "reference", None)
    if ref is None:
        return False
    resolved = getattr(ref, "resolved", None)
    if resolved is None:
        return False
    author = getattr(resolved, "author", None)
    if author is None:
        return False
    return getattr(author, "id", None) == self_user_id


def _build_listener(adapter: Any):
    """Return an async on_message listener bound to this adapter."""
    client = getattr(adapter, "_client", None)
    self_user_id_holder = {"id": None}  # resolved lazily once client.user is set

    async def _listener(message: Any) -> None:
        try:
            # Resolve self-user id lazily — client.user is None pre-ready.
            if self_user_id_holder["id"] is None and client is not None:
                user = getattr(client, "user", None)
                if user is not None:
                    self_user_id_holder["id"] = getattr(user, "id", None)
            self_user_id = self_user_id_holder["id"]

            author = getattr(message, "author", None)
            is_bot = bool(getattr(author, "bot", False))
            author_id = getattr(author, "id", None)
            is_self = self_user_id is not None and author_id == self_user_id

            channel = getattr(message, "channel", None)
            channel_id = str(getattr(channel, "id", "") or "")
            text = getattr(message, "content", "") or ""

            # Track human activity for the operator-quiet window regardless of mode.
            if not is_bot and not is_self and channel_id:
                record_human_activity(channel_id)

            mode = get_mode()
            if mode == "disabled":
                return

            callout_channels = get_callout_channels()
            proceed, _reason = should_observe(
                is_bot=is_bot,
                is_self=is_self,
                channel_id=channel_id,
                text=text,
                callout_channels=callout_channels,
            )
            if not proceed:
                return

            if _is_self_referenced(message, self_user_id):
                return

            msg_id = str(getattr(message, "id", "") or "")
            await observe_inbound_bot(
                adapter=adapter,
                channel_id=channel_id,
                msg_id=msg_id,
                author_bot_id=str(author_id or ""),
                text=text,
            )
        except Exception as e:
            logger.error(
                "[ai_callout_hook] listener crashed: %s", e, exc_info=True
            )

    return _listener


def _patch_discord_adapter(adapter: Any) -> None:
    """Attach an on_message listener to the adapter's Discord client.

    Uses add_listener (not @client.event) so we run alongside
    discord.py's primary on_message handler instead of replacing it.
    """
    client = getattr(adapter, "_client", None)
    if client is None:
        logger.warning("[ai_callout_hook] adapter has no _client")
        return
    if getattr(client, _PATCH_FLAG, False):
        logger.debug("[ai_callout_hook] already patched, skipping")
        return

    listener = _build_listener(adapter)
    try:
        client.add_listener(listener, name="on_message")
    except Exception as e:
        logger.error("[ai_callout_hook] add_listener failed: %s", e)
        return

    setattr(client, _PATCH_FLAG, True)
    logger.info("[ai_callout_hook] on_message listener registered")


async def _wait_and_patch(adapter: Any) -> None:
    ready_event = getattr(adapter, "_ready_event", None)
    if ready_event is not None:
        try:
            await asyncio.wait_for(ready_event.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            logger.warning("[ai_callout_hook] ready event timed out")
            return
    _patch_discord_adapter(adapter)


# --- gateway:startup hook handler ------------------------------------------


async def handle(event_type: str, context: dict) -> None:
    """Gateway startup entrypoint registered in HOOK.yaml."""
    if event_type != "gateway:startup":
        return

    logger.info("[ai_callout_hook] gateway:startup — wiring AI callout observer")

    try:
        from dgmh.hermes_integration.reaction_hook import _find_discord_adapter
    except Exception as e:
        logger.warning("[ai_callout_hook] cannot import _find_discord_adapter: %s", e)
        return

    adapter = _find_discord_adapter(context)
    if adapter is None:
        logger.warning(
            "[ai_callout_hook] no DiscordAdapter at startup — hook not wired"
        )
        return

    asyncio.create_task(_wait_and_patch(adapter))
    logger.info("[ai_callout_hook] patch scheduled")
