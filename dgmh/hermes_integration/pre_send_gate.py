"""Pre-send length gate for the flask discord bot.

Calls the ``dgmh-pre-send-gate`` Hermes skill via subprocess. On any failure
(skill not found, timeout, JSON parse error, nonzero exit), falls back to an
in-process simple length cap + last-sentence-boundary truncate so the bot
never gets stuck on this gate.

Output decisions:
- ``send``: pass content through unchanged.
- ``compress``: replace content with the gate's rewritten/truncated version.
- ``silent``: replace content with empty string. The hermes adapter's
  ``base.py:2093`` then auto-skips the send.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


_ALLOWED_DECISIONS = frozenset({"send", "compress", "silent"})
_DEFAULT_PUBLIC_MAX = 200
_DEFAULT_OPERATOR_MAX = 600
_DEFAULT_SKILL_BIN = "~/.hermes/skills/dgmh-pre-send-gate/bin/dgmh-pre-send-gate.py"
_SENTENCE_BOUNDARY_CHARS = (".", "!", "?", "~", "\n")

# Fires the missing-skill warning at most once per process. Tests can reset
# this when they need to re-exercise the warning path.
_seen_skill_warning = False


def _resolve_max_chars(channel_kind: str) -> int:
    if channel_kind == "operator":
        env_val = os.environ.get("DGMH_OPERATOR_MAX_CHARS")
        default = _DEFAULT_OPERATOR_MAX
    else:
        env_val = os.environ.get("DGMH_PUBLIC_MAX_CHARS")
        default = _DEFAULT_PUBLIC_MAX

    if env_val is None:
        return default
    try:
        parsed = int(env_val)
        if parsed <= 0:
            return default
        return parsed
    except (TypeError, ValueError):
        return default


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """Last-sentence-boundary truncate; appends a single ellipsis."""
    if len(text) <= max_chars:
        return text

    head = text[:max_chars]
    best = -1
    for ch in _SENTENCE_BOUNDARY_CHARS:
        idx = head.rfind(ch)
        if idx > best:
            best = idx

    if best <= 0:
        cut = max(0, max_chars - 1)
        return text[:cut] + "…"

    truncated = text[: best + 1].rstrip()
    if len(truncated) < len(text):
        if truncated.endswith("\n"):
            truncated = truncated.rstrip("\n")
        return truncated + "…"
    return truncated


def _fallback_gate(content: str, max_chars: int) -> tuple[str, str]:
    if not content or not content.strip():
        return ("silent", "")
    if len(content) <= max_chars:
        return ("send", content)
    return ("compress", _truncate_at_sentence(content, max_chars))


def _resolve_skill_bin(skill_bin: str | None) -> str:
    if skill_bin:
        return os.path.expanduser(skill_bin)
    env_override = os.environ.get("DGMH_GATE_SKILL_BIN")
    if env_override:
        return os.path.expanduser(env_override)
    return os.path.expanduser(_DEFAULT_SKILL_BIN)


def _warn_missing_skill_once(path: str) -> None:
    global _seen_skill_warning
    if _seen_skill_warning:
        return
    _seen_skill_warning = True
    logger.warning(
        "[pre_send_gate] skill binary missing or not executable at %s; "
        "using in-process fallback for the rest of this process",
        path,
    )


def gate(
    content: str,
    *,
    channel_kind: str = "public",
    max_chars: int | None = None,
    timeout_s: float = 35.0,
    skill_bin: str | None = None,
    pre_rewritten: bool = False,
) -> tuple[str, str]:
    """Run the pre-send length gate.

    Returns ``(decision, content)`` where ``decision`` is ``"send"``,
    ``"compress"``, or ``"silent"``. Caller is responsible for honoring
    the decision (e.g. dropping the send when ``"silent"``).

    ``pre_rewritten=True`` tells the gate skill that ``content`` has
    already been through a patina rewrite (e.g., from humanness_hook's
    pre-send path) so the skill skips its own redundant patina call
    and goes straight to the sentence-boundary truncate when needed.
    """
    cap = max_chars if max_chars is not None else _resolve_max_chars(channel_kind)

    resolved_bin = _resolve_skill_bin(skill_bin)
    bin_path = Path(resolved_bin)

    if not bin_path.exists():
        _warn_missing_skill_once(resolved_bin)
        decision, out = _fallback_gate(content, cap)
        logger.info(
            "[pre_send_gate] decision=%s in_len=%d out_len=%d source=fallback "
            "reason=skill_missing",
            decision, len(content or ""), len(out or ""),
        )
        return decision, out

    request = {
        "content": content,
        "channel_kind": channel_kind,
        "max_chars": cap,
        "pre_rewritten": bool(pre_rewritten),
    }
    payload = json.dumps(request, ensure_ascii=False)

    try:
        result = subprocess.run(
            ["python3", str(bin_path)],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "[pre_send_gate] skill subprocess timed out after %.1fs; "
            "falling back",
            timeout_s,
        )
        decision, out = _fallback_gate(content, cap)
        logger.info(
            "[pre_send_gate] decision=%s in_len=%d out_len=%d source=fallback "
            "reason=timeout",
            decision, len(content or ""), len(out or ""),
        )
        return decision, out
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[pre_send_gate] skill subprocess raised %s: %s; falling back",
            type(exc).__name__, exc,
        )
        decision, out = _fallback_gate(content, cap)
        logger.info(
            "[pre_send_gate] decision=%s in_len=%d out_len=%d source=fallback "
            "reason=subprocess_error",
            decision, len(content or ""), len(out or ""),
        )
        return decision, out

    if result.returncode != 0:
        logger.warning(
            "[pre_send_gate] skill exited %d; stderr=%s; falling back",
            result.returncode, (result.stderr or "")[:200],
        )
        decision, out = _fallback_gate(content, cap)
        logger.info(
            "[pre_send_gate] decision=%s in_len=%d out_len=%d source=fallback "
            "reason=nonzero_exit",
            decision, len(content or ""), len(out or ""),
        )
        return decision, out

    stdout = (result.stdout or "").strip()
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError as exc:
        logger.warning(
            "[pre_send_gate] skill returned non-JSON (%s); falling back",
            exc,
        )
        decision, out = _fallback_gate(content, cap)
        logger.info(
            "[pre_send_gate] decision=%s in_len=%d out_len=%d source=fallback "
            "reason=json_parse_error",
            decision, len(content or ""), len(out or ""),
        )
        return decision, out

    decision = parsed.get("decision")
    out_content = parsed.get("content")

    if decision not in _ALLOWED_DECISIONS or not isinstance(out_content, str):
        logger.warning(
            "[pre_send_gate] skill returned invalid payload "
            "(decision=%r, content_type=%s); falling back",
            decision, type(out_content).__name__,
        )
        decision_fb, out_fb = _fallback_gate(content, cap)
        logger.info(
            "[pre_send_gate] decision=%s in_len=%d out_len=%d source=fallback "
            "reason=invalid_payload",
            decision_fb, len(content or ""), len(out_fb or ""),
        )
        return decision_fb, out_fb

    reason = parsed.get("reason", "-")
    logger.info(
        "[pre_send_gate] decision=%s in_len=%d out_len=%d source=skill reason=%s",
        decision, len(content or ""), len(out_content or ""), reason,
    )
    return decision, out_content


async def gate_async(
    content: str,
    *,
    channel_kind: str = "public",
    max_chars: int | None = None,
    timeout_s: float = 35.0,
    skill_bin: str | None = None,
    pre_rewritten: bool = False,
) -> tuple[str, str]:
    """Async wrapper for :func:`gate` that never blocks the event loop.

    The skill-backed branch may run a subprocess for up to ``timeout_s``. When
    called from Discord's async send pipeline, doing that inline can block the
    gateway heartbeat and force reconnects. Keep the synchronous API for tests
    and non-async callers, but offload gateway use through ``asyncio.to_thread``.
    """
    return await asyncio.to_thread(
        gate,
        content,
        channel_kind=channel_kind,
        max_chars=max_chars,
        timeout_s=timeout_s,
        skill_bin=skill_bin,
        pre_rewritten=pre_rewritten,
    )
