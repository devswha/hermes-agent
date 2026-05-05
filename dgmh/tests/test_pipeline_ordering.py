"""Step 0 (v3) — assert humanness/honcho wrap-order on adapter.send.

Both ``dgmh.hermes_integration.humanness_hook._wrap_send`` and
``dgmh.hermes_integration.honcho_hook._wrap_send`` monkey-patch ``adapter.send``.
The pipeline contract is:

    outermost  →  honcho  →  humanness  →  original_send  (innermost)

Honcho must be outermost so its peer-memory mirror sees the post-rewrite
content humanness produces.

These tests install both wrappers on a stub adapter and walk the
``__wrapped__`` chain via ``dgmh.hermes_integration.wrap_order.verify_wrap_chain``.
"""

from __future__ import annotations

import unittest
from typing import Any, Optional

from dgmh.hermes_integration import honcho_hook, humanness_hook
from dgmh.hermes_integration.wrap_order import (
    EXPECTED_CHAIN,
    _walk_chain,
    verify_wrap_chain,
)


class _StubAdapter:
    """Minimal DiscordAdapter stand-in. ``send`` is the original callable."""

    def __init__(self) -> None:
        async def original_send(
            chat_id: str,
            content: str,
            reply_to: Optional[str] = None,
            metadata: Optional[dict[str, Any]] = None,
        ):
            return {"message_id": "m-orig", "chat_id": chat_id, "content": content}

        self.send = original_send  # type: ignore[assignment]


class TestWrapChain(unittest.TestCase):
    def test_humanness_then_honcho_chain_outer_inner(self) -> None:
        """Wrapping humanness first then honcho must yield ``[honcho, humanness]``."""
        adapter = _StubAdapter()
        humanness_hook._wrap_send(adapter)
        honcho_hook._wrap_send(adapter)

        seen = _walk_chain(adapter.send)
        self.assertEqual(seen, EXPECTED_CHAIN)
        self.assertTrue(verify_wrap_chain(adapter))

    def test_humanness_marker_innermost(self) -> None:
        """Innermost wrapper must carry __dgmh_hook_name__ == 'humanness'."""
        adapter = _StubAdapter()
        humanness_hook._wrap_send(adapter)
        self.assertEqual(
            getattr(adapter.send, "__dgmh_hook_name__", None), "humanness"
        )
        # The original send is NOT a dgmh wrapper, so chain length is 1.
        self.assertEqual(_walk_chain(adapter.send), ["humanness"])

    def test_honcho_marker_outermost(self) -> None:
        """Outer wrapper carries 'honcho' and exposes humanness via __wrapped__."""
        adapter = _StubAdapter()
        humanness_hook._wrap_send(adapter)
        honcho_hook._wrap_send(adapter)
        self.assertEqual(
            getattr(adapter.send, "__dgmh_hook_name__", None), "honcho"
        )
        inner = getattr(adapter.send, "__wrapped__", None)
        self.assertIsNotNone(inner)
        self.assertEqual(getattr(inner, "__dgmh_hook_name__", None), "humanness")

    def test_unwrapped_adapter_chain_empty(self) -> None:
        """A bare adapter has no dgmh markers; verification returns False."""
        adapter = _StubAdapter()
        self.assertEqual(_walk_chain(adapter.send), [])
        self.assertFalse(verify_wrap_chain(adapter))

    def test_partial_chain_humanness_only(self) -> None:
        """Only humanness wrapped → partial chain, verify returns False."""
        adapter = _StubAdapter()
        humanness_hook._wrap_send(adapter)
        self.assertEqual(_walk_chain(adapter.send), ["humanness"])
        # Partial: missing honcho on the outside.
        self.assertFalse(verify_wrap_chain(adapter))

    def test_reversed_order_detected_as_broken(self) -> None:
        """If honcho wraps before humanness, the chain is ``[humanness, honcho]``
        which is the wrong order — verify must return False and log ERROR."""
        adapter = _StubAdapter()
        honcho_hook._wrap_send(adapter)
        humanness_hook._wrap_send(adapter)
        seen = _walk_chain(adapter.send)
        self.assertEqual(seen, ["humanness", "honcho"])
        self.assertNotEqual(seen, EXPECTED_CHAIN)
        self.assertFalse(verify_wrap_chain(adapter))


# NOTE: AC14 ("forced rewrite → honcho mirror sees post-rewrite content") is
# an end-to-end pipeline test that belongs with Step 4 (Pipeline ordering),
# not Step 0. Step 0 only fixes the wrap chain identification; the actual
# semantic flow of rewritten text through the chain is wired in Step 4.


if __name__ == "__main__":
    unittest.main()
