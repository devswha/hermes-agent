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
from unittest import mock

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


class TestPostRewriteContentFlowsToHoncho(unittest.TestCase):
    """Step 4 / AC14 (v3) — humanness inner wrap publishes post-rewrite text
    via the ``_post_rewrite_content`` ContextVar; honcho outer wrap reads
    that var and mirrors the rewritten content (not the original draft).

    Tested at the honcho-wrap level rather than the full pipeline so the
    assertion is hermetic — no asyncio.to_thread, no real patina, no real
    state.db. The contract under test is narrow:

        whatever value humanness sets in _post_rewrite_content is what
        honcho's wrap mirrors, modulo the local-content fallback.

    Uses ``asyncio.run`` rather than ``IsolatedAsyncioTestCase`` because
    the latter clears the main thread's event-loop slot on cleanup, which
    breaks downstream tests that call ``asyncio.get_event_loop()``.
    """

    def _run_async(self, coro):
        # Run on a dedicated loop and restore a fresh one afterward so
        # downstream tests that use ``asyncio.get_event_loop()`` don't
        # observe the empty slot that ``asyncio.run`` /
        # IsolatedAsyncioTestCase leave behind in Python 3.10.
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(asyncio.new_event_loop())

    def test_post_rewrite_content_reaches_honcho_mirror(self) -> None:
        self._run_async(self._test_post_rewrite_reaches_honcho_mirror_impl())

    async def _test_post_rewrite_reaches_honcho_mirror_impl(self) -> None:
        from dgmh.honcho_client import (
            _post_rewrite_content,
            set_post_rewrite_content,
        )

        adapter = _StubAdapter()
        # Wrap ONLY honcho — humanness's full pipeline is not under test
        # here; we are exercising honcho's read-from-ContextVar code path.
        honcho_hook._wrap_send(adapter)

        forced_rewritten = "사람이 쓴 짧은 응답인데 30자 넘게 만들어야 게이트를 통과해."
        original_text = "원본 봇풍 텍스트로 시작하는 30자 이상의 긴 ChatGPT 어조 초안."

        captured: dict[str, Any] = {}

        def fake_add_bot_async(*, channel_id, thread_id, content):  # noqa: ANN001
            captured["channel_id"] = channel_id
            captured["content"] = content

        def _thread_factory(target=None, args=None, kwargs=None, **_kw):  # noqa: ANN001
            class _T:
                def __init__(self) -> None:
                    self._target = target
                    self._args = args or ()
                    self._kwargs = kwargs or {}

                def start(self) -> None:
                    self._target(*self._args, **self._kwargs)

            return _T()

        # Manually publish what humanness's inner wrap WOULD publish after
        # its rewrite stage. Honcho's wrap should pick this up regardless
        # of what its caller passed in as ``content``.
        token = set_post_rewrite_content(forced_rewritten)
        try:
            with mock.patch.dict(
                "os.environ",
                {"DGMH_HONCHO_ALLOWED_CHANNELS": "test-channel-1"},
                clear=False,
            ), mock.patch.object(
                honcho_hook, "_add_bot_async", fake_add_bot_async
            ), mock.patch(
                "dgmh.hermes_integration.honcho_hook.threading.Thread"
            ) as fake_thread:
                fake_thread.side_effect = _thread_factory
                await adapter.send("test-channel-1", original_text)
        finally:
            _post_rewrite_content.reset(token)

        self.assertEqual(captured.get("channel_id"), "test-channel-1")
        self.assertEqual(
            captured.get("content"),
            forced_rewritten,
            "honcho mirror must use the post-rewrite content from the "
            "_post_rewrite_content ContextVar, not the local ``content`` "
            "parameter (which still holds the pre-rewrite draft)",
        )

    def test_no_post_rewrite_var_falls_back_to_original_content(self) -> None:
        self._run_async(self._test_fallback_impl())

    async def _test_fallback_impl(self) -> None:
        """When the ContextVar is unset (e.g. 1:1 channel that never went
        through the public-mode rewrite path), honcho mirrors the local
        content unchanged."""
        adapter = _StubAdapter()
        honcho_hook._wrap_send(adapter)

        captured: dict[str, Any] = {}

        def fake_add_bot_async(*, channel_id, thread_id, content):  # noqa: ANN001
            captured["content"] = content

        def _thread_factory(target=None, args=None, kwargs=None, **_kw):  # noqa: ANN001
            class _T:
                def __init__(self) -> None:
                    self._target = target
                    self._args = args or ()
                    self._kwargs = kwargs or {}

                def start(self) -> None:
                    self._target(*self._args, **self._kwargs)

            return _T()

        with mock.patch.dict(
            "os.environ",
            {"DGMH_HONCHO_ALLOWED_CHANNELS": "test-channel-1"},
            clear=False,
        ), mock.patch.object(
            honcho_hook, "_add_bot_async", fake_add_bot_async
        ), mock.patch(
            "dgmh.hermes_integration.honcho_hook.threading.Thread"
        ) as fake_thread:
            fake_thread.side_effect = _thread_factory
            await adapter.send("test-channel-1", "operator 1:1 reply text")

        self.assertEqual(captured.get("content"), "operator 1:1 reply text")

    def test_post_rewrite_var_equal_to_content_no_override(self) -> None:
        self._run_async(self._test_var_equal_impl())

    async def _test_var_equal_impl(self) -> None:
        """If humanness publishes the SAME text as what honcho already
        sees (e.g. no rewrite happened but the var was published as a
        no-op), honcho's override path is a no-op too — mirror still gets
        the same string and no extra log fires."""
        from dgmh.honcho_client import set_post_rewrite_content

        adapter = _StubAdapter()
        honcho_hook._wrap_send(adapter)

        captured: dict[str, Any] = {}

        def fake_add_bot_async(*, channel_id, thread_id, content):  # noqa: ANN001
            captured["content"] = content

        def _thread_factory(target=None, args=None, kwargs=None, **_kw):  # noqa: ANN001
            class _T:
                def __init__(self) -> None:
                    self._target = target
                    self._args = args or ()
                    self._kwargs = kwargs or {}

                def start(self) -> None:
                    self._target(*self._args, **self._kwargs)

            return _T()

        text = "동일한 컨텐츠 — 30자 이상 길어야 _is_recordable 통과."
        from dgmh.honcho_client import _post_rewrite_content

        token = set_post_rewrite_content(text)
        try:
            with mock.patch.dict(
                "os.environ",
                {"DGMH_HONCHO_ALLOWED_CHANNELS": "test-channel-1"},
                clear=False,
            ), mock.patch.object(
                honcho_hook, "_add_bot_async", fake_add_bot_async
            ), mock.patch(
                "dgmh.hermes_integration.honcho_hook.threading.Thread"
            ) as fake_thread:
                fake_thread.side_effect = _thread_factory
                await adapter.send("test-channel-1", text)
        finally:
            _post_rewrite_content.reset(token)

        self.assertEqual(captured.get("content"), text)


if __name__ == "__main__":
    unittest.main()
