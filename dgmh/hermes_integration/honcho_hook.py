"""DGM-H Honcho hook — mirror Discord messages into Honcho memory.

Registered at ``gateway:startup``. Wraps the DiscordAdapter so that:

  - Inbound operator messages (on_message) → Honcho.add_user_message
  - Outbound bot replies (send) → Honcho.add_bot_message

Both writes are fire-and-forget on background threads — never block the
chat reply path. Honcho's deriver runs background memory extraction
asynchronously, so callers do not wait on it.

The Hermes-native ``memory`` tool and ``state.db`` continue to operate
unchanged. Honcho is an additional persistent memory layer on top.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)


_PATCH_FLAG_SEND = "_dgmh_honcho_send_patched"
_PATCH_FLAG_MSG = "_dgmh_honcho_msg_patched"
_MIN_LENGTH = 4


def _is_disabled() -> bool:
    return bool(os.environ.get("DGMH_HONCHO_DISABLED"))


def _is_relevant_channel(channel_id: str) -> bool:
    """Honcho writes only fire for the configured allowed channels.

    Falls back to True (mirror everywhere) if no allow-list is configured,
    so this matches the rest of the Hermes Discord pipeline.
    """
    raw = os.environ.get("DGMH_HONCHO_ALLOWED_CHANNELS") or os.environ.get(
        "DISCORD_ALLOWED_CHANNELS", ""
    )
    if not raw:
        return True
    allowed = {ch.strip() for ch in raw.split(",") if ch.strip()}
    return channel_id in allowed


def _is_recordable(content: str) -> bool:
    """Skip empty / very short / tool-trace messages so the deriver doesn't
    burn cycles on noise."""
    if not content or len(content.strip()) < _MIN_LENGTH:
        return False
    stripped = content.lstrip()
    for prefix in ("📚", "🌐", "🔎", "🔍", "💻", "📖", "🧠", "📸", "⚡", "📝", "🛠"):
        if stripped.startswith(prefix):
            return False
    return True


def _add_user_async(*, channel_id: str, thread_id: Optional[str], content: str) -> None:
    try:
        from dgmh.honcho_client import add_user_message

        add_user_message(channel_id=channel_id, thread_id=thread_id, content=content)
    except Exception:
        logger.exception("[honcho_hook] add_user_message failed")


def _add_bot_async(*, channel_id: str, thread_id: Optional[str], content: str) -> None:
    try:
        from dgmh.honcho_client import add_bot_message

        add_bot_message(channel_id=channel_id, thread_id=thread_id, content=content)
    except Exception:
        logger.exception("[honcho_hook] add_bot_message failed")


def _wrap_send(adapter: Any) -> None:
    """Wrap DiscordAdapter.send to mirror outbound replies into Honcho."""
    if getattr(adapter, _PATCH_FLAG_SEND, False):
        logger.info("[honcho_hook] send already patched")
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
            if _is_disabled() or not _is_recordable(content):
                return result
            if not _is_relevant_channel(str(chat_id)):
                return result
            thread_id = (metadata or {}).get("thread_id")
            t = threading.Thread(
                target=_add_bot_async,
                kwargs={
                    "channel_id": str(chat_id),
                    "thread_id": str(thread_id) if thread_id else None,
                    "content": content,
                },
                daemon=True,
                name="dgmh-honcho-bot",
            )
            t.start()
        except Exception:
            logger.exception("[honcho_hook] failed to dispatch bot mirror")
        return result

    adapter.send = wrapped_send  # type: ignore[assignment]
    setattr(adapter, _PATCH_FLAG_SEND, True)
    logger.info("[honcho_hook] wrapped DiscordAdapter.send for honcho mirror")


_PATCH_FLAG_ROLE_MENT = "_dgmh_role_mention_promoted"


def _promote_role_mentions(adapter: Any) -> None:
    """Monkey-patch client.on_message so a role mention pointing at any of
    the bot's own roles is promoted to a direct user mention.

    Discord.py exposes message.role_mentions and message.mentions separately.
    Hermes' adapter checks ONLY message.mentions for the require_mention
    gate, so a `<@&role-id>` to a role the bot is in does not currently
    trigger a response. This wrapper appends client.user to message.mentions
    when any role-mention overlaps with the bot's role set, so the existing
    require_mention path naturally accepts the message.
    """
    client = getattr(adapter, "_client", None)
    if client is None:
        logger.warning("[honcho_hook] no _client; skip role-mention promote")
        return
    if getattr(client, _PATCH_FLAG_ROLE_MENT, False):
        logger.info("[honcho_hook] role-mention promote already patched")
        return

    original = getattr(client, "on_message", None)
    if not callable(original):
        logger.warning(
            "[honcho_hook] client.on_message not set; cannot wrap for role mentions"
        )
        return

    async def wrapped_on_message(message):
        try:
            role_mentions = getattr(message, "role_mentions", None) or []
            guild = getattr(message, "guild", None)
            if role_mentions and guild is not None:
                bot_member = getattr(guild, "me", None)
                if bot_member is not None:
                    bot_role_ids = {r.id for r in getattr(bot_member, "roles", [])}
                    if any(r.id in bot_role_ids for r in role_mentions):
                        if client.user not in message.mentions:
                            try:
                                message.mentions.append(client.user)
                                logger.info(
                                    "[honcho_hook] promoted role mention to user "
                                    "mention for message %s", message.id,
                                )
                            except Exception:
                                logger.exception(
                                    "[honcho_hook] could not append client.user "
                                    "to message.mentions"
                                )
        except Exception:
            logger.exception("[honcho_hook] role-mention promote failed")
        await original(message)

    client.on_message = wrapped_on_message  # type: ignore[assignment]
    setattr(client, _PATCH_FLAG_ROLE_MENT, True)
    logger.info("[honcho_hook] wrapped client.on_message for role-mention promotion")


def _wrap_on_message(adapter: Any) -> None:
    """Wrap discord.py's on_message handler to mirror inbound operator messages."""
    client = getattr(adapter, "_client", None)
    if client is None:
        logger.warning("[honcho_hook] no _client on adapter; cannot wrap on_message")
        return
    if getattr(client, _PATCH_FLAG_MSG, False):
        logger.info("[honcho_hook] on_message already patched")
        return

    listeners = getattr(client, "_listeners", None)
    # discord.py keeps registered listeners in _listeners or via dispatch.
    # Easier path: register an additional listener that runs alongside the
    # existing on_message; discord.py supports multiple listeners per event.
    @client.event
    async def on_message_honcho_mirror(message):  # type: ignore[unused-ignore]
        try:
            if _is_disabled():
                return
            # Skip bot's own messages (those are mirrored via wrapped_send) and
            # anything from the bot user.
            if message.author == client.user:
                return
            content = getattr(message, "content", "") or ""
            if not _is_recordable(content):
                return
            channel_id = str(getattr(message.channel, "id", ""))
            if not channel_id or not _is_relevant_channel(channel_id):
                return
            thread_id = None
            parent = getattr(message.channel, "parent", None)
            if parent is not None and hasattr(parent, "id"):
                thread_id = channel_id
                channel_id = str(parent.id)

            t = threading.Thread(
                target=_add_user_async,
                kwargs={
                    "channel_id": channel_id,
                    "thread_id": thread_id,
                    "content": content,
                },
                daemon=True,
                name="dgmh-honcho-user",
            )
            t.start()
        except Exception:
            logger.exception("[honcho_hook] on_message mirror failed")

    setattr(client, _PATCH_FLAG_MSG, True)
    logger.info("[honcho_hook] registered on_message mirror listener")


_PROMPT_AUGMENT_FLAG = "_dgmh_honcho_prompt_augmented"


def _augment_load_soul_md() -> None:
    """Patch agent.prompt_builder.load_soul_md to append Honcho operator snapshot.

    Idempotent — wrap-once. After patching, every agent boot that calls
    load_soul_md gets SOUL.md content followed by a small "Operator
    memory snapshot" block sourced from Honcho's peer model. Empty or
    unavailable snapshot → fall through to plain SOUL.md content.
    """
    try:
        import agent.prompt_builder as pb
    except ImportError:
        logger.warning("[honcho_hook] agent.prompt_builder not importable; skip patch")
        return

    if getattr(pb, _PROMPT_AUGMENT_FLAG, False):
        logger.info("[honcho_hook] load_soul_md already augmented")
        return

    original = getattr(pb, "load_soul_md", None)
    if original is None:
        logger.warning("[honcho_hook] load_soul_md missing; skip patch")
        return

    def _patched_load_soul_md():
        base = original()
        if not base:
            return base
        try:
            from dgmh.honcho_client import get_cached_operator_snapshot

            snap = get_cached_operator_snapshot()
            if snap:
                addendum = (
                    "\n\n## Operator memory snapshot (from Honcho, may be partial)\n\n"
                    "이 블록은 Honcho 가 운영자 과거 대화를 보고 추출한 짧은 요약이야. "
                    "절대값이 아니라 참고 신호 — SOUL.md 본문 룰이 우선이고, 이 블록은 "
                    "tone / 관심사 / 최근 작업 흐름 정도만 거든다.\n\n"
                    f"{snap}\n"
                )
                return base + addendum
        except Exception:
            logger.exception("[honcho_hook] augmenter failed; falling back")
        return base

    pb.load_soul_md = _patched_load_soul_md  # type: ignore[assignment]
    setattr(pb, _PROMPT_AUGMENT_FLAG, True)
    logger.info("[honcho_hook] augmented load_soul_md with operator snapshot")


async def _wait_and_patch(adapter: Any) -> None:
    try:
        for _ in range(60):
            client = getattr(adapter, "_client", None)
            if client and getattr(client, "is_ready", lambda: False)():
                break
            await asyncio.sleep(0.5)
        # Bootstrap the workspace+peers idempotently.
        try:
            from dgmh.honcho_client import bootstrap

            status = bootstrap()
            logger.info("[honcho_hook] bootstrap: %s", status)
        except Exception:
            logger.exception("[honcho_hook] bootstrap failed")

        _wrap_send(adapter)
        _wrap_on_message(adapter)
        _promote_role_mentions(adapter)
        _augment_load_soul_md()
    except Exception:
        logger.exception("[honcho_hook] _wait_and_patch failed")


async def handle(event_type: str, context: dict) -> None:
    """Gateway startup hook entry point."""
    if event_type != "gateway:startup":
        return
    if _is_disabled():
        logger.info("[honcho_hook] DGMH_HONCHO_DISABLED set; skipping")
        return

    logger.info("[honcho_hook] gateway:startup — wiring honcho mirror")

    from dgmh.hermes_integration.reaction_hook import _find_discord_adapter

    adapter = _find_discord_adapter(context)
    if adapter is None:
        logger.warning("[honcho_hook] DiscordAdapter not found; skip")
        return

    asyncio.create_task(_wait_and_patch(adapter))
    logger.info("[honcho_hook] scheduled honcho wrap task")
