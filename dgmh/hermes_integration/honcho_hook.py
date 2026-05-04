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
