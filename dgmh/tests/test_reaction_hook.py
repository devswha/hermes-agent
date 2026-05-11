"""
dgmh/tests/test_reaction_hook.py — W6 tests for reaction_hook.py (>=10 tests).

Covers:
  1-3.  Emoji classification: positive, negative, neutral
  4.    Custom emoji config overrides defaults
  5.    Throttle/debounce: is_throttled returns True within DEBOUNCE_SECONDS
  6.    Throttle reset: is_throttled False after reset_throttle()
  7.    Filter: bot's own reaction is ignored
  8.    Filter: non-operator user is rejected when allowlist is non-empty
  9.    Filter: operator in allowlist is accepted
  10.   Filter: wrong channel is rejected when channel whitelist is set
  11.   Filter: right channel is accepted
  12.   Evolution trigger fires exactly once per qualifying negative reaction
  13.   Positive reaction: no evolution triggered
  14.   Neutral reaction: no evolution triggered
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dgmh.hermes_integration.reaction_hook as rh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(
    allowed_user_ids: set | None = None,
    bot_user_id: int = 9999,
) -> MagicMock:
    adapter = MagicMock()
    adapter._allowed_user_ids = allowed_user_ids or set()
    client = MagicMock()
    user = MagicMock()
    user.id = bot_user_id
    client.user = user
    adapter._client = client
    adapter._ready_event = asyncio.Event()
    adapter._running = True
    return adapter


def _make_payload(
    user_id: int,
    channel_id: int,
    message_id: int,
    emoji_str: str,
    guild_id: int | None = 1,
) -> MagicMock:
    payload = MagicMock()
    payload.user_id = user_id
    payload.channel_id = channel_id
    payload.message_id = message_id
    payload.guild_id = guild_id
    emoji = MagicMock()
    emoji.__str__ = MagicMock(return_value=emoji_str)
    payload.emoji = emoji
    return payload


# ---------------------------------------------------------------------------
# 1-3: Emoji classification
# ---------------------------------------------------------------------------


class TestEmojiClassification:
    def setup_method(self):
        # Reset any cached config by patching the config path to non-existent
        pass

    def test_positive_thumbsup(self):
        assert rh.classify_emoji("👍") == "positive"

    def test_negative_thumbsdown(self):
        assert rh.classify_emoji("👎") == "negative"

    def test_neutral_shrug(self):
        assert rh.classify_emoji("🤷") == "neutral"

    def test_positive_checkmark(self):
        assert rh.classify_emoji("✅") == "positive"

    def test_negative_x(self):
        assert rh.classify_emoji("❌") == "negative"

    def test_custom_emoji_config(self, tmp_path):
        """Custom config overrides default emoji sets."""
        cfg_file = tmp_path / "dgmh" / "reaction_config.json"
        cfg_file.parent.mkdir(parents=True)
        cfg_file.write_text(json.dumps({
            "positive_emoji": ["🦄"],
            "negative_emoji": ["💩"],
        }))

        with patch.object(rh, "_reaction_config_path", return_value=cfg_file):
            assert rh.classify_emoji("🦄") == "positive"
            assert rh.classify_emoji("💩") == "negative"
            assert rh.classify_emoji("👍") == "neutral"  # not in custom set


# ---------------------------------------------------------------------------
# 5-6: Throttle / debounce
# ---------------------------------------------------------------------------


class TestThrottle:
    def setup_method(self):
        rh.reset_throttle()

    def test_not_throttled_initially(self):
        assert rh.is_throttled() is False

    def test_throttled_after_mark(self):
        rh._mark_evolution_time()
        assert rh.is_throttled() is True

    def test_not_throttled_after_reset(self):
        rh._mark_evolution_time()
        rh.reset_throttle()
        assert rh.is_throttled() is False

    def test_throttled_within_debounce_seconds(self):
        # Simulate that last evolution was 5 seconds ago (within 60s debounce)
        rh._last_evolution_time = time.monotonic() - 5.0
        assert rh.is_throttled() is True

    def test_not_throttled_after_debounce_expires(self):
        # Simulate last evolution was 120 seconds ago (beyond 60s debounce)
        rh._last_evolution_time = time.monotonic() - 120.0
        assert rh.is_throttled() is False


# ---------------------------------------------------------------------------
# 7-11: Reaction filtering
# ---------------------------------------------------------------------------


class TestReactionFilter:
    def test_bot_own_reaction_ignored(self):
        """Bot's own reactions are always ignored."""
        assert rh.should_process_reaction(
            reactor_user_id=9999,     # same as bot
            channel_id=100,
            bot_user_id=9999,
            allowed_user_ids=set(),   # empty = allow all
            allowed_channels=set(),
        ) is False

    def test_non_operator_rejected_when_allowlist_set(self):
        """If allowlist is non-empty, only listed users pass."""
        assert rh.should_process_reaction(
            reactor_user_id=1111,     # not in allowlist
            channel_id=100,
            bot_user_id=9999,
            allowed_user_ids={"2222"},
            allowed_channels=set(),
        ) is False

    def test_operator_accepted_in_allowlist(self):
        """Operator in allowlist passes."""
        assert rh.should_process_reaction(
            reactor_user_id=2222,
            channel_id=100,
            bot_user_id=9999,
            allowed_user_ids={"2222"},
            allowed_channels=set(),
        ) is True

    def test_empty_allowlist_accepts_all(self):
        """Empty allowlist means all non-bot users are accepted."""
        assert rh.should_process_reaction(
            reactor_user_id=3333,
            channel_id=100,
            bot_user_id=9999,
            allowed_user_ids=set(),
            allowed_channels=set(),
        ) is True

    def test_wrong_channel_rejected(self):
        """Channel not in whitelist is rejected."""
        assert rh.should_process_reaction(
            reactor_user_id=2222,
            channel_id=999,           # not in whitelist
            bot_user_id=9999,
            allowed_user_ids=set(),
            allowed_channels={"100"},
        ) is False

    def test_correct_channel_accepted(self):
        """Channel in whitelist is accepted."""
        assert rh.should_process_reaction(
            reactor_user_id=2222,
            channel_id=100,
            bot_user_id=9999,
            allowed_user_ids=set(),
            allowed_channels={"100"},
        ) is True


# ---------------------------------------------------------------------------
# 12-14: Evolution trigger fires correctly
# ---------------------------------------------------------------------------


class TestEvolutionTrigger:
    def setup_method(self):
        rh.reset_throttle()

    def _run(self, coro):
        # Use a fresh loop per call so Py3.11's deprecation of implicit
        # get_event_loop() across xdist workers doesn't surface as
        # "no current event loop" intermittently.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_negative_reaction_on_bot_message_triggers_evolution(self):
        """Negative reaction on bot's own message triggers exactly one evolution."""
        adapter = _make_adapter(bot_user_id=9999)
        payload = _make_payload(
            user_id=1234,
            channel_id=100,
            message_id=5555,
            emoji_str="👎",
        )

        # Mock channel/message fetch: message is from the bot
        mock_msg = MagicMock()
        mock_msg.author.id = 9999  # bot's ID

        mock_channel = AsyncMock()
        mock_channel.fetch_message = AsyncMock(return_value=mock_msg)
        adapter._client.get_channel = MagicMock(return_value=None)
        adapter._client.fetch_channel = AsyncMock(return_value=mock_channel)

        evolution_calls = []

        async def fake_trigger(classification, adapter, **kwargs):
            evolution_calls.append(classification)

        with patch.object(rh, "_trigger_soul_evolution", side_effect=fake_trigger):
            self._run(rh._handle_reaction_event(payload, adapter))

        assert len(evolution_calls) == 1
        assert evolution_calls[0] == "negative"

    def test_positive_reaction_no_evolution(self):
        """Positive reaction records but does NOT trigger evolution."""
        adapter = _make_adapter(bot_user_id=9999)
        payload = _make_payload(
            user_id=1234,
            channel_id=100,
            message_id=5555,
            emoji_str="👍",
        )

        mock_msg = MagicMock()
        mock_msg.author.id = 9999

        mock_channel = AsyncMock()
        mock_channel.fetch_message = AsyncMock(return_value=mock_msg)
        adapter._client.get_channel = MagicMock(return_value=None)
        adapter._client.fetch_channel = AsyncMock(return_value=mock_channel)

        evolution_calls = []

        async def fake_trigger(classification, adapter, **kwargs):
            evolution_calls.append(classification)

        with patch.object(rh, "_trigger_soul_evolution", side_effect=fake_trigger):
            self._run(rh._handle_reaction_event(payload, adapter))

        assert len(evolution_calls) == 0

    def test_neutral_reaction_no_evolution(self):
        """Neutral emoji does not trigger evolution."""
        adapter = _make_adapter(bot_user_id=9999)
        payload = _make_payload(
            user_id=1234,
            channel_id=100,
            message_id=5555,
            emoji_str="🤷",
        )

        mock_msg = MagicMock()
        mock_msg.author.id = 9999

        mock_channel = AsyncMock()
        mock_channel.fetch_message = AsyncMock(return_value=mock_msg)
        adapter._client.get_channel = MagicMock(return_value=None)
        adapter._client.fetch_channel = AsyncMock(return_value=mock_channel)

        evolution_calls = []

        async def fake_trigger(classification, adapter, **kwargs):
            evolution_calls.append(classification)

        with patch.object(rh, "_trigger_soul_evolution", side_effect=fake_trigger):
            self._run(rh._handle_reaction_event(payload, adapter))

        assert len(evolution_calls) == 0

    def test_bot_own_reaction_not_processed(self):
        """Reactions from the bot itself are filtered before any evolution."""
        adapter = _make_adapter(bot_user_id=9999)
        payload = _make_payload(
            user_id=9999,     # bot's own ID
            channel_id=100,
            message_id=5555,
            emoji_str="👎",
        )

        evolution_calls = []

        async def fake_trigger(classification, adapter, **kwargs):
            evolution_calls.append(classification)

        with patch.object(rh, "_trigger_soul_evolution", side_effect=fake_trigger):
            self._run(rh._handle_reaction_event(payload, adapter))

        assert len(evolution_calls) == 0

    def test_reaction_on_non_bot_message_ignored(self):
        """Reactions on messages NOT from the bot are ignored."""
        adapter = _make_adapter(bot_user_id=9999)
        payload = _make_payload(
            user_id=1234,
            channel_id=100,
            message_id=5555,
            emoji_str="👎",
        )

        mock_msg = MagicMock()
        mock_msg.author.id = 1234  # NOT the bot

        mock_channel = AsyncMock()
        mock_channel.fetch_message = AsyncMock(return_value=mock_msg)
        adapter._client.get_channel = MagicMock(return_value=None)
        adapter._client.fetch_channel = AsyncMock(return_value=mock_channel)

        evolution_calls = []

        async def fake_trigger(classification, adapter, **kwargs):
            evolution_calls.append(classification)

        with patch.object(rh, "_trigger_soul_evolution", side_effect=fake_trigger):
            self._run(rh._handle_reaction_event(payload, adapter))

        assert len(evolution_calls) == 0

    def test_throttle_prevents_second_evolution(self):
        """A second qualifying reaction within DEBOUNCE_SECONDS is throttled."""
        rh._mark_evolution_time()  # simulate recent evolution

        adapter = _make_adapter(bot_user_id=9999)
        payload = _make_payload(
            user_id=1234,
            channel_id=100,
            message_id=5555,
            emoji_str="👎",
        )

        mock_msg = MagicMock()
        mock_msg.author.id = 9999

        mock_channel = AsyncMock()
        mock_channel.fetch_message = AsyncMock(return_value=mock_msg)
        adapter._client.get_channel = MagicMock(return_value=None)
        adapter._client.fetch_channel = AsyncMock(return_value=mock_channel)

        evolution_calls = []

        async def fake_evolution(opts):
            evolution_calls.append(True)

        with patch("dgmh.soul_evolution.run_soul_evolution", side_effect=fake_evolution):
            self._run(rh._trigger_soul_evolution("negative", adapter))

        assert len(evolution_calls) == 0
