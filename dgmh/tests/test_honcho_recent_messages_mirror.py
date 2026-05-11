"""DGM-H public-channel ambient context — mirror-session merge.

The agent's per-user honcho session only sees that user's turns. The
DGM-H honcho_hook mirrors *every* channel message into a separate
``discord-public-{channel_id}`` session. Without merging the two,
recent_messages reflects only one user's history and the channel-wide
context the persona is supposed to feel never reaches the prompt.

These tests cover the small helper that bridges the two sessions:
``_resolve_public_mirror_channel(session_key)``.

The full ``get_prefetch_context`` path requires a live Honcho client
to mock end-to-end and is exercised by the live gateway behavior.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from plugins.memory.honcho.session import _resolve_public_mirror_channel


_CHANNEL_PUBLIC = "1496735735078715542"
_CHANNEL_OTHER = "1496872245027541062"
_USER_ID = "266436073557590016"


def _agent_key(channel_id: str, user_id: str = _USER_ID) -> str:
    return f"agent-main-discord-group-{channel_id}-{user_id}"


class TestResolvePublicMirrorChannel(unittest.TestCase):
    def test_returns_channel_when_session_matches_configured_public(self) -> None:
        with mock.patch.dict(
            os.environ, {"DGMH_PUBLIC_CHANNELS": _CHANNEL_PUBLIC}
        ):
            got = _resolve_public_mirror_channel(_agent_key(_CHANNEL_PUBLIC))
        self.assertEqual(got, _CHANNEL_PUBLIC)

    def test_returns_none_when_session_belongs_to_non_public_channel(self) -> None:
        with mock.patch.dict(
            os.environ, {"DGMH_PUBLIC_CHANNELS": _CHANNEL_PUBLIC}
        ):
            got = _resolve_public_mirror_channel(_agent_key(_CHANNEL_OTHER))
        self.assertIsNone(got)

    def test_returns_none_when_env_unset(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DGMH_PUBLIC_CHANNELS", None)
            got = _resolve_public_mirror_channel(_agent_key(_CHANNEL_PUBLIC))
        self.assertIsNone(got)

    def test_returns_none_when_env_blank(self) -> None:
        with mock.patch.dict(os.environ, {"DGMH_PUBLIC_CHANNELS": "   "}):
            got = _resolve_public_mirror_channel(_agent_key(_CHANNEL_PUBLIC))
        self.assertIsNone(got)

    def test_returns_none_for_empty_session_key(self) -> None:
        with mock.patch.dict(
            os.environ, {"DGMH_PUBLIC_CHANNELS": _CHANNEL_PUBLIC}
        ):
            got = _resolve_public_mirror_channel("")
        self.assertIsNone(got)

    def test_multiple_channels_picks_first_match(self) -> None:
        env = f"99999,{_CHANNEL_PUBLIC},88888"
        with mock.patch.dict(os.environ, {"DGMH_PUBLIC_CHANNELS": env}):
            got = _resolve_public_mirror_channel(_agent_key(_CHANNEL_PUBLIC))
        self.assertEqual(got, _CHANNEL_PUBLIC)

    def test_ignores_whitespace_around_csv_entries(self) -> None:
        env = f"  {_CHANNEL_PUBLIC}  , 99999 "
        with mock.patch.dict(os.environ, {"DGMH_PUBLIC_CHANNELS": env}):
            got = _resolve_public_mirror_channel(_agent_key(_CHANNEL_PUBLIC))
        self.assertEqual(got, _CHANNEL_PUBLIC)


if __name__ == "__main__":
    unittest.main()
