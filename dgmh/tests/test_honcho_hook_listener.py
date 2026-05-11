"""DGM-H honcho_hook inbound mirror — listener registration regression.

The on_message_honcho_mirror handler must be registered via
``client.add_listener(handler, 'on_message')`` so discord.py dispatches
inbound messages to it alongside the agent's own primary handler.

Earlier code used ``@client.event``, which setattr's the function on the
client under its OWN name (here ``on_message_honcho_mirror``). discord.py
dispatch only looks at attribute names that exactly match an event name
(``on_message``, ``on_ready`` etc.), so the decorator silently disabled
the inbound mirror — user messages never reached Honcho. This test pins
the correct registration path so the bug cannot return.
"""

from __future__ import annotations

import unittest
from unittest import mock

from dgmh.hermes_integration.honcho_hook import _wrap_on_message


class TestOnMessageListenerRegistration(unittest.TestCase):
    def test_registers_via_add_listener_with_on_message_name(self) -> None:
        client = mock.MagicMock()
        client._dgmh_honcho_msg_patched = False
        adapter = mock.MagicMock()
        adapter._client = client

        _wrap_on_message(adapter)

        client.add_listener.assert_called_once()
        args, kwargs = client.add_listener.call_args
        # add_listener(handler, name) — positional or kwarg either way.
        name = kwargs.get("name") if "name" in kwargs else (args[1] if len(args) >= 2 else None)
        self.assertEqual(
            name,
            "on_message",
            msg="listener must bind to the 'on_message' event for discord.py dispatch",
        )

    def test_idempotent_does_not_reregister(self) -> None:
        client = mock.MagicMock()
        client._dgmh_honcho_msg_patched = True  # already patched
        adapter = mock.MagicMock()
        adapter._client = client

        _wrap_on_message(adapter)

        client.add_listener.assert_not_called()

    def test_missing_client_is_warned_not_raised(self) -> None:
        adapter = mock.MagicMock()
        adapter._client = None

        # Should not raise — defensively logs and returns.
        try:
            _wrap_on_message(adapter)
        except Exception as e:
            self.fail(f"_wrap_on_message raised unexpectedly: {e!r}")


if __name__ == "__main__":
    unittest.main()
