"""Tests for Discord bot message filtering (DISCORD_ALLOW_BOTS)."""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.platforms.discord import _promote_raw_user_mention


def _make_author(*, bot: bool = False, is_self: bool = False):
    """Create a mock Discord author."""
    author = MagicMock()
    author.bot = bot
    author.id = 99999 if is_self else 12345
    author.name = "TestBot" if bot else "TestUser"
    author.display_name = author.name
    return author


def _make_message(*, author=None, content="hello", mentions=None, is_dm=False):
    """Create a mock Discord message."""
    msg = MagicMock()
    msg.author = author or _make_author()
    msg.content = content
    msg.attachments = []
    msg.mentions = mentions or []
    if is_dm:
        import discord
        msg.channel = MagicMock(spec=discord.DMChannel)
        msg.channel.id = 111
    else:
        msg.channel = MagicMock()
        msg.channel.id = 222
        msg.channel.name = "test-channel"
        msg.channel.guild = MagicMock()
        msg.channel.guild.name = "TestServer"
        # Make isinstance checks fail for DMChannel and Thread
        type(msg.channel).__name__ = "TextChannel"
    return msg


class TestDiscordBotFilter(unittest.TestCase):
    """Test the DISCORD_ALLOW_BOTS filtering logic."""

    def _run_filter(self, message, allow_bots="none", client_user=None):
        """Simulate the on_message filter logic and return whether message was accepted."""
        # Replicate the exact filter logic from discord.py on_message
        if message.author == client_user:
            return False  # own messages always ignored

        if getattr(message.author, "bot", False):
            allow = allow_bots.lower().strip()
            if allow == "none":
                return False
            elif allow == "mentions":
                if not client_user or client_user not in message.mentions:
                    return False
            # "all" falls through
        
        return True  # message accepted

    def test_own_messages_always_ignored(self):
        """Bot's own messages are always ignored regardless of allow_bots."""
        bot_user = _make_author(is_self=True)
        msg = _make_message(author=bot_user)
        self.assertFalse(self._run_filter(msg, "all", bot_user))

    def test_human_messages_always_accepted(self):
        """Human messages are always accepted regardless of allow_bots."""
        human = _make_author(bot=False)
        msg = _make_message(author=human)
        self.assertTrue(self._run_filter(msg, "none"))
        self.assertTrue(self._run_filter(msg, "mentions"))
        self.assertTrue(self._run_filter(msg, "all"))

    def test_allow_bots_none_rejects_bots(self):
        """With allow_bots=none, all other bot messages are rejected."""
        bot = _make_author(bot=True)
        msg = _make_message(author=bot)
        self.assertFalse(self._run_filter(msg, "none"))

    def test_allow_bots_all_accepts_bots(self):
        """With allow_bots=all, all bot messages are accepted."""
        bot = _make_author(bot=True)
        msg = _make_message(author=bot)
        self.assertTrue(self._run_filter(msg, "all"))

    def test_allow_bots_mentions_rejects_without_mention(self):
        """With allow_bots=mentions, bot messages without @mention are rejected."""
        our_user = _make_author(is_self=True)
        bot = _make_author(bot=True)
        msg = _make_message(author=bot, mentions=[])
        self.assertFalse(self._run_filter(msg, "mentions", our_user))

    def test_allow_bots_mentions_accepts_with_mention(self):
        """With allow_bots=mentions, bot messages with @mention are accepted."""
        our_user = _make_author(is_self=True)
        bot = _make_author(bot=True)
        msg = _make_message(author=bot, mentions=[our_user])
        self.assertTrue(self._run_filter(msg, "mentions", our_user))

    def test_allowed_bot_raw_mention_can_be_promoted(self):
        """Trusted bots may render <@id> while suppressing parsed mentions."""
        our_user = _make_author(is_self=True)
        bot = _make_author(bot=True)
        msg = _make_message(
            author=bot,
            content=f"플라스크~ <@{our_user.id}> 들리니?",
            mentions=[],
        )

        self.assertTrue(_promote_raw_user_mention(msg, our_user))
        self.assertIn(our_user, msg.mentions)

    def test_allowed_bot_raw_nick_mention_can_be_promoted(self):
        """The Discord nickname mention form <@!id> is equivalent."""
        our_user = _make_author(is_self=True)
        bot = _make_author(bot=True)
        msg = _make_message(
            author=bot,
            content=f"플라스크~ <@!{our_user.id}> 들리니?",
            mentions=[],
        )

        self.assertTrue(_promote_raw_user_mention(msg, our_user))
        self.assertIn(our_user, msg.mentions)

    def test_raw_mention_promotion_requires_matching_user_id(self):
        """Do not promote random raw mention text for another user."""
        our_user = _make_author(is_self=True)
        bot = _make_author(bot=True)
        msg = _make_message(
            author=bot,
            content="다른 봇 <@123456> 들리니?",
            mentions=[],
        )

        self.assertFalse(_promote_raw_user_mention(msg, our_user))
        self.assertNotIn(our_user, msg.mentions)

    def test_default_is_none(self):
        """Default behavior (no env var) should be 'none'."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISCORD_ALLOW_BOTS", None)
            default = os.getenv("DISCORD_ALLOW_BOTS", "none")
            self.assertEqual(default, "none")

    def test_case_insensitive(self):
        """Allow_bots value should be case-insensitive."""
        bot = _make_author(bot=True)
        msg = _make_message(author=bot)
        self.assertTrue(self._run_filter(msg, "ALL"))
        self.assertTrue(self._run_filter(msg, "All"))
        self.assertFalse(self._run_filter(msg, "NONE"))
        self.assertFalse(self._run_filter(msg, "None"))


if __name__ == "__main__":
    unittest.main()
