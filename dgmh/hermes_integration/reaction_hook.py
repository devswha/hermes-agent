"""
dgmh/hermes_integration/reaction_hook.py — W6: Discord reaction-triggered SOUL.md evolution.

Hook surface investigation findings (2026-05-04):
  - Hermes hook system (gateway/hooks.py) has NO native reaction event type.
    Available events: gateway:startup, session:start/end/reset,
    agent:start/step/end, command:*.
  - Discord intents: Intents.default() is used in DiscordAdapter, which
    includes guild reactions. The 'reactions' intent is part of default().
  - Strategy: register a gateway:startup hook. At startup, iterate
    gateway.adapters (or the global adapter list) to find any DiscordAdapter,
    then monkey-patch on_raw_reaction_add onto its _client.
  - If no DiscordAdapter is found at startup (not yet connected), schedule
    a delayed patch via the adapter's _ready_event.
  - Throttle state is module-level (single process, single-loop design).

Config: ~/.hermes/dgmh/reaction_config.json
  {
    "positive_emoji": ["👍", "✅", "🎉", "💯", "🔥"],
    "negative_emoji": ["👎", "❌", "💀", "🗑️", "😤"]
  }
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure project root importable
# ---------------------------------------------------------------------------

_HERMES_ROOT = Path(__file__).parent.parent.parent
if str(_HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(_HERMES_ROOT))

# ---------------------------------------------------------------------------
# Config paths
# ---------------------------------------------------------------------------


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


def _reaction_config_path() -> Path:
    return _hermes_home() / "dgmh" / "reaction_config.json"


# ---------------------------------------------------------------------------
# Default emoji sets
# ---------------------------------------------------------------------------

_DEFAULT_POSITIVE: frozenset[str] = frozenset(["👍", "✅", "🎉", "💯", "🔥", "⬆️", "❤️"])
_DEFAULT_NEGATIVE: frozenset[str] = frozenset(["👎", "❌", "💀", "🗑️", "😤", "⬇️", "😞"])

# Debounce: ignore new evolution if previous cycle finished within this many seconds
DEBOUNCE_SECONDS: float = 60.0

# Module-level throttle state (reset on each service restart)
_last_evolution_time: float = 0.0


def _load_emoji_sets() -> tuple[frozenset[str], frozenset[str]]:
    """Load positive/negative emoji sets from reaction_config.json.

    Returns (positive_set, negative_set). Falls back to defaults on any error.
    """
    cfg_path = _reaction_config_path()
    if not cfg_path.exists():
        return _DEFAULT_POSITIVE, _DEFAULT_NEGATIVE
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        pos = frozenset(cfg.get("positive_emoji") or []) or _DEFAULT_POSITIVE
        neg = frozenset(cfg.get("negative_emoji") or []) or _DEFAULT_NEGATIVE
        return pos, neg
    except Exception as exc:
        logger.warning("[reaction_hook] Could not load reaction_config.json: %s", exc)
        return _DEFAULT_POSITIVE, _DEFAULT_NEGATIVE


def classify_emoji(emoji_str: str) -> str:
    """Classify an emoji string as 'positive', 'negative', or 'neutral'.

    Args:
        emoji_str: The emoji string (e.g. '👍', '🔥').

    Returns:
        'positive', 'negative', or 'neutral'.
    """
    pos, neg = _load_emoji_sets()
    if emoji_str in pos:
        return "positive"
    if emoji_str in neg:
        return "negative"
    return "neutral"


def is_throttled() -> bool:
    """Return True if we're within DEBOUNCE_SECONDS of the last evolution."""
    global _last_evolution_time
    if _last_evolution_time == 0.0:
        return False
    return (time.monotonic() - _last_evolution_time) < DEBOUNCE_SECONDS


def _mark_evolution_time() -> None:
    """Record that an evolution cycle just completed (for debounce)."""
    global _last_evolution_time
    _last_evolution_time = time.monotonic()


def reset_throttle() -> None:
    """Reset throttle state (for testing)."""
    global _last_evolution_time
    _last_evolution_time = 0.0


# ---------------------------------------------------------------------------
# Operator / channel filtering
# ---------------------------------------------------------------------------


def _get_allowed_user_ids(adapter: Any) -> set[str]:
    """Extract the allowed user IDs from a DiscordAdapter."""
    return set(getattr(adapter, "_allowed_user_ids", set()))


def _get_allowed_channels(adapter: Any) -> set[str]:
    """Extract allowed channel IDs from config (DISCORD_ALLOWED_CHANNELS env or adapter)."""
    raw = os.environ.get("DISCORD_ALLOWED_CHANNELS", "")
    if raw:
        return {ch.strip() for ch in raw.split(",") if ch.strip()}
    return set()


def _get_bot_user_id(adapter: Any) -> int | None:
    """Get the bot's own user ID from the DiscordAdapter's client."""
    client = getattr(adapter, "_client", None)
    if client is None:
        return None
    user = getattr(client, "user", None)
    if user is None:
        return None
    return user.id


def should_process_reaction(
    reactor_user_id: int,
    channel_id: int,
    bot_user_id: int | None,
    allowed_user_ids: set[str],
    allowed_channels: set[str],
) -> bool:
    """Decide whether to process this reaction event.

    Rules:
      1. Ignore reactions from the bot itself.
      2. If allowed_user_ids is set, only accept reactions from those users.
      3. If allowed_channels is set, only accept reactions from those channels.

    Args:
        reactor_user_id: Discord user ID of who reacted.
        channel_id: Discord channel ID where the reaction happened.
        bot_user_id: The bot's own user ID (to ignore self-reactions).
        allowed_user_ids: Set of numeric string IDs of allowed operators.
        allowed_channels: Set of channel IDs (numeric strings) that are allowed.

    Returns:
        True if the reaction should be processed.
    """
    # Rule 1: ignore bot's own reactions
    if bot_user_id is not None and reactor_user_id == bot_user_id:
        return False

    # Rule 2: operator allowlist
    if allowed_user_ids and str(reactor_user_id) not in allowed_user_ids:
        return False

    # Rule 3: channel whitelist
    if allowed_channels and str(channel_id) not in allowed_channels:
        return False

    return True


# ---------------------------------------------------------------------------
# Evolution trigger
# ---------------------------------------------------------------------------


async def _trigger_soul_evolution(classification: str, adapter: Any) -> None:
    """Trigger a SOUL.md evolution cycle on negative reaction.

    Called only for 'negative' classifications that pass throttle check.
    """
    global _last_evolution_time

    if is_throttled():
        logger.info(
            "[reaction_hook] Evolution throttled (last cycle was %.1fs ago)",
            time.monotonic() - _last_evolution_time,
        )
        return

    # Mark throttle immediately to prevent concurrent triggers
    _mark_evolution_time()

    logger.info("[reaction_hook] Negative reaction: triggering SOUL.md evolution")

    try:
        from dgmh.soul_evolution import run_soul_evolution, SoulEvolutionOpts
        opts = SoulEvolutionOpts()
        # Run in executor to avoid blocking the Discord event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_soul_evolution, opts)
    except Exception as exc:
        logger.error("[reaction_hook] Soul evolution failed: %s", exc, exc_info=True)


async def _handle_reaction_event(payload: Any, adapter: Any) -> None:
    """Process a single RawReactionActionEvent payload.

    Args:
        payload: discord.RawReactionActionEvent
        adapter: The DiscordAdapter instance
    """
    # Extract fields from payload (discord.RawReactionActionEvent)
    reactor_user_id: int = payload.user_id
    channel_id: int = payload.channel_id
    message_id: int = payload.message_id
    guild_id: int | None = payload.guild_id

    # Get emoji string
    emoji_obj = payload.emoji
    emoji_str: str = str(emoji_obj)

    bot_user_id = _get_bot_user_id(adapter)
    allowed_user_ids = _get_allowed_user_ids(adapter)
    allowed_channels = _get_allowed_channels(adapter)

    logger.info(
        "[reaction_hook] Reaction received: user=%d channel=%d emoji=%s msg=%d",
        reactor_user_id, channel_id, emoji_str, message_id,
    )

    # Resolve thread channels to their parent channel for the allowed-channels
    # filter — Discord auto-threads have their own channel ID distinct from the
    # parent. We fetch the channel here once to discover parent_id, then use
    # the parent for the channel-allowlist check.
    client = getattr(adapter, "_client", None)
    if client is None:
        return

    try:
        channel = client.get_channel(channel_id)
        if channel is None:
            try:
                channel = await client.fetch_channel(channel_id)
            except Exception:
                logger.info("[reaction_hook] Could not fetch channel %d — skipping", channel_id)
                return
    except Exception as exc:
        logger.info("[reaction_hook] channel lookup error: %s", exc)
        return

    # If this channel is a thread, the effective channel for filtering is the parent.
    effective_channel_id = channel_id
    parent_obj = getattr(channel, "parent", None)
    if parent_obj is not None and hasattr(parent_obj, "id"):
        effective_channel_id = parent_obj.id
        logger.info(
            "[reaction_hook] Thread %d resolved to parent %d for filter",
            channel_id, effective_channel_id,
        )

    # Filter: only process operator's reactions in allowed channels, not bot's own
    if not should_process_reaction(
        reactor_user_id, effective_channel_id, bot_user_id, allowed_user_ids, allowed_channels
    ):
        logger.info(
            "[reaction_hook] Filter rejected reaction: user=%d effective_channel=%d (allowed=%s) bot_user=%s",
            reactor_user_id, effective_channel_id, allowed_channels, bot_user_id,
        )
        return

    try:

        # Fetch the message to check authorship
        try:
            msg = await channel.fetch_message(message_id)
        except Exception as exc:
            logger.debug(
                "[reaction_hook] Could not fetch message %d: %s", message_id, exc
            )
            return

        # Only process reactions on bot's own messages
        if bot_user_id is None or (
            hasattr(msg, "author") and msg.author.id != bot_user_id
        ):
            logger.debug(
                "[reaction_hook] Reaction on non-bot message (author=%s), ignoring",
                getattr(msg.author, "id", "?"),
            )
            return

    except Exception as exc:
        logger.debug("[reaction_hook] Error checking message author: %s", exc)
        return

    # Classify emoji
    classification = classify_emoji(emoji_str)
    logger.info(
        "[reaction_hook] Reaction user=%d emoji=%s classification=%s",
        reactor_user_id,
        emoji_str,
        classification,
    )

    if classification == "positive":
        # Record positive reaction — no evolution triggered
        _record_reaction(classification="positive", emoji=emoji_str)

    elif classification == "negative":
        # Record negative reaction and trigger evolution
        _record_reaction(classification="negative", emoji=emoji_str)
        await _trigger_soul_evolution(classification, adapter)

    else:
        # Neutral — log but no action
        logger.debug(
            "[reaction_hook] Neutral emoji %s — logged, no action", emoji_str
        )


def _record_reaction(classification: str, emoji: str) -> None:
    """Append a reaction event to the dgmh log (non-blocking best-effort)."""
    try:
        log_path = _hermes_home() / "dgmh" / "reaction_events.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        import datetime as _dt

        entry = {
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "classification": classification,
            "emoji": emoji,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.debug("[reaction_hook] Could not log reaction event: %s", exc)


# ---------------------------------------------------------------------------
# Monkey-patch: register on_raw_reaction_add on the Discord client
# ---------------------------------------------------------------------------


def _patch_discord_adapter(adapter: Any) -> None:
    """Inject on_raw_reaction_add onto the DiscordAdapter's _client.

    Called after the adapter is connected (on_ready has fired).
    """
    client = getattr(adapter, "_client", None)
    if client is None:
        logger.warning("[reaction_hook] DiscordAdapter has no _client — cannot patch")
        return

    # Avoid double-patching
    if getattr(client, "_dgmh_reaction_patched", False):
        logger.debug("[reaction_hook] Already patched, skipping")
        return

    adapter_ref = adapter

    @client.event
    async def on_raw_reaction_add(payload):  # type: ignore[no-redef]
        try:
            await _handle_reaction_event(payload, adapter_ref)
        except Exception as exc:
            logger.error(
                "[reaction_hook] Unhandled error in on_raw_reaction_add: %s",
                exc,
                exc_info=True,
            )

    client._dgmh_reaction_patched = True
    logger.info("[reaction_hook] on_raw_reaction_add patched onto Discord client")


async def _wait_and_patch(adapter: Any) -> None:
    """Wait for the adapter to be ready, then patch."""
    ready_event = getattr(adapter, "_ready_event", None)
    if ready_event is not None:
        try:
            await asyncio.wait_for(ready_event.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            logger.warning(
                "[reaction_hook] Timed out waiting for Discord ready event"
            )
            return
    _patch_discord_adapter(adapter)


# ---------------------------------------------------------------------------
# gateway:startup hook handler
# ---------------------------------------------------------------------------


async def handle(event_type: str, context: dict) -> None:
    """Gateway startup hook — find DiscordAdapter and wire reaction handler.

    This is the hook entry point registered in HOOK.yaml for gateway:startup.
    """
    if event_type != "gateway:startup":
        return

    logger.info("[reaction_hook] gateway:startup — wiring Discord reaction handler")

    # Try to find the DiscordAdapter via the gateway module
    adapter = _find_discord_adapter(context)
    if adapter is None:
        logger.warning(
            "[reaction_hook] No DiscordAdapter found at startup — "
            "reaction hook not wired. Is Discord platform enabled?"
        )
        return

    # Schedule the wait+patch as a background task so we don't block startup
    asyncio.create_task(_wait_and_patch(adapter))
    logger.info("[reaction_hook] Scheduled reaction hook patch task")


def _find_discord_adapter(context: dict) -> Any | None:
    """Attempt to locate the active DiscordAdapter from context or registry.

    The gateway:startup context may contain 'adapters' or 'gateway'. Falls
    back to importing gateway.run and inspecting the active adapters list.
    """
    # Try context directly
    adapters = context.get("adapters") or []
    for adapter in adapters:
        if _is_discord_adapter(adapter):
            return adapter

    # Try gateway module-level adapter registry
    try:
        import gateway.run as _gw_run  # noqa: WPS433

        for attr in ("_adapters", "adapters", "_platform_adapters"):
            candidate = getattr(_gw_run, attr, None)
            if candidate:
                if isinstance(candidate, list):
                    for a in candidate:
                        if _is_discord_adapter(a):
                            return a
                elif _is_discord_adapter(candidate):
                    return candidate
    except Exception as exc:
        logger.debug("[reaction_hook] Could not inspect gateway.run: %s", exc)

    # Try platforms module
    try:
        from gateway.platforms import discord as _discord_mod  # noqa: WPS433
        adapter_cls = getattr(_discord_mod, "DiscordAdapter", None)
        if adapter_cls:
            # Scan all live objects of this type (last resort)
            import gc
            for obj in gc.get_objects():
                if isinstance(obj, adapter_cls) and getattr(obj, "_running", False):
                    return obj
    except Exception as exc:
        logger.debug("[reaction_hook] gc scan failed: %s", exc)

    return None


def _is_discord_adapter(obj: Any) -> bool:
    """Check if obj is a DiscordAdapter instance."""
    cls_name = type(obj).__name__
    return cls_name == "DiscordAdapter" or (
        hasattr(obj, "_client") and hasattr(obj, "_ready_event")
        and hasattr(obj, "_allowed_user_ids")
    )
