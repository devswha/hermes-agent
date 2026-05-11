"""Step 6 (v3) — channel-kind ContextVar isolation + cross-thread propagation.

Three contracts:

1. ``_channel_kind`` defaults to ``"operator"``. ``set_channel_kind`` /
   ``get_channel_kind`` round-trip a kind. Unknown kinds coerce to
   ``"operator"`` so a typo can't mistag operator memory as public.
2. ``capture_context_with_kind`` returns a ``Context`` snapshot whose
   ``_channel_kind`` is the requested kind, WITHOUT mutating the caller's
   ambient context.
3. The captured context propagates the kind across a ``threading.Thread``
   boundary — the worker thread reads the kind set by the dispatcher,
   not the default. This is the bug ``copy_context`` is the fix for.
"""

from __future__ import annotations

import os
import threading
import unittest
from unittest import mock

from dgmh.honcho_client import (
    _channel_kind,
    add_user_message,
    capture_context_with_kind,
    channel_kind_for,
    chat_about_public_persona,
    get_channel_kind,
    is_public_channel,
    session_id_for_channel,
    set_channel_kind,
)


class TestChannelKindContextVar(unittest.TestCase):
    def test_default_is_operator(self) -> None:
        # Ambient ContextVar defaults to "operator" so legacy 1:1 paths
        # that never set the var keep their existing tagging.
        token = _channel_kind.set("operator")
        try:
            self.assertEqual(get_channel_kind(), "operator")
        finally:
            _channel_kind.reset(token)

    def test_set_and_get_roundtrip(self) -> None:
        token = set_channel_kind("public")
        try:
            self.assertEqual(get_channel_kind(), "public")
        finally:
            _channel_kind.reset(token)

    def test_unknown_kind_coerces_to_operator(self) -> None:
        token = set_channel_kind("garbage-typo")
        try:
            self.assertEqual(get_channel_kind(), "operator")
        finally:
            _channel_kind.reset(token)


class TestSessionIdNamespacing(unittest.TestCase):
    def test_operator_kind_uses_legacy_id(self) -> None:
        sid = session_id_for_channel("123", kind="operator")
        self.assertEqual(sid, "discord-123")

    def test_public_kind_namespaced(self) -> None:
        sid = session_id_for_channel("123", kind="public")
        self.assertEqual(sid, "discord-public-123")

    def test_thread_id_appended(self) -> None:
        sid = session_id_for_channel("123", thread_id="t9", kind="public")
        self.assertEqual(sid, "discord-public-123-thread-t9")

    def test_kind_defaults_to_contextvar(self) -> None:
        # When the caller does not pass kind=, the function reads the
        # current ContextVar.
        token = set_channel_kind("public")
        try:
            sid = session_id_for_channel("999")
            self.assertEqual(sid, "discord-public-999")
        finally:
            _channel_kind.reset(token)


class TestPublicChannelLookup(unittest.TestCase):
    def test_no_env_means_no_public(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DGMH_PUBLIC_CHANNELS", None)
            self.assertFalse(is_public_channel("123"))

    def test_env_list_membership(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"DGMH_PUBLIC_CHANNELS": "111, 222 , 333"},
        ):
            self.assertTrue(is_public_channel("111"))
            self.assertTrue(is_public_channel("222"))
            self.assertTrue(is_public_channel("333"))
            self.assertFalse(is_public_channel("444"))

    def test_channel_kind_for(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"DGMH_PUBLIC_CHANNELS": "555"},
        ):
            self.assertEqual(channel_kind_for("555"), "public")
            self.assertEqual(channel_kind_for("444"), "operator")


class TestCaptureContextWithKind(unittest.TestCase):
    def test_returns_isolated_snapshot(self) -> None:
        # Caller stays at default "operator"; captured ctx has "public".
        ctx = capture_context_with_kind("public")
        self.assertEqual(get_channel_kind(), "operator")
        observed = ctx.run(get_channel_kind)
        self.assertEqual(observed, "public")

    def test_unknown_kind_coerces(self) -> None:
        ctx = capture_context_with_kind("nonsense")
        self.assertEqual(ctx.run(get_channel_kind), "operator")


class TestChannelKindPropagatesAcrossThreads(unittest.TestCase):
    """The plan's named test — bare ``threading.Thread`` does NOT propagate
    ContextVar values; ``copy_context()`` does."""

    def test_bare_thread_loses_contextvar(self) -> None:
        """Sanity: without copy_context, the worker sees the default."""
        observed: list[str] = []

        token = set_channel_kind("public")
        try:
            t = threading.Thread(
                target=lambda: observed.append(get_channel_kind()),
                daemon=True,
            )
            t.start()
            t.join(timeout=2.0)
        finally:
            _channel_kind.reset(token)

        self.assertEqual(observed, ["operator"])

    def test_capture_context_propagates_kind_to_thread(self) -> None:
        """With ``capture_context_with_kind`` + ``ctx.run``, the worker reads
        the kind set on the dispatcher."""
        observed: list[str] = []

        ctx = capture_context_with_kind("public")
        t = threading.Thread(
            target=ctx.run,
            args=(lambda: observed.append(get_channel_kind()),),
            daemon=True,
        )
        t.start()
        t.join(timeout=2.0)

        self.assertEqual(observed, ["public"])

    def test_multiple_threads_carry_independent_kinds(self) -> None:
        """Two captured contexts with different kinds dispatch to two
        threads; each worker reads its OWN kind, not the other's."""
        observed_public: list[str] = []
        observed_operator: list[str] = []

        ctx_pub = capture_context_with_kind("public")
        ctx_op = capture_context_with_kind("operator")

        t1 = threading.Thread(
            target=ctx_pub.run,
            args=(lambda: observed_public.append(get_channel_kind()),),
            daemon=True,
        )
        t2 = threading.Thread(
            target=ctx_op.run,
            args=(lambda: observed_operator.append(get_channel_kind()),),
            daemon=True,
        )
        t1.start()
        t2.start()
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)

        self.assertEqual(observed_public, ["public"])
        self.assertEqual(observed_operator, ["operator"])


class _FakePeer:
    def __init__(self, name: str) -> None:
        self.id = name
        self.name = name

    def message(self, content: str, metadata=None):
        return {"peer": self.name, "content": content, "metadata": metadata or {}}

    def chat(self, query: str):
        class _Resp:
            content = f"{query} 공개 채널 말투는 짧고 자연스럽게 유지한다."

        return _Resp()


class _FakeSession:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.messages = []

    def add_messages(self, messages):
        self.messages.extend(messages)


class _FakeHoncho:
    def __init__(self) -> None:
        self.peers: list[str] = []
        self.sessions: dict[str, _FakeSession] = {}

    def peer(self, name: str):
        self.peers.append(name)
        return _FakePeer(name)

    def session(self, session_id: str):
        session = self.sessions.get(session_id)
        if session is None:
            session = _FakeSession(session_id)
            self.sessions[session_id] = session
        return session


class TestPublicHonchoPeerIsolation(unittest.TestCase):
    def test_public_user_messages_use_public_peer_not_operator_peer(self) -> None:
        fake = _FakeHoncho()
        env = {
            "DGMH_HONCHO_PUBLIC_PEER": "public-room",
            "DGMH_HONCHO_OPERATOR_PEER": "operator-private",
            "DGMH_HONCHO_BOT_PEER": "flask",
        }
        with mock.patch.dict(os.environ, env), mock.patch(
            "dgmh.honcho_client.get_client",
            return_value=fake,
        ):
            ok = add_user_message(
                channel_id="1496735735078715542",
                thread_id=None,
                content="정신챙겨봐",
                kind="public",
            )

        self.assertTrue(ok)
        self.assertIn("public-room", fake.peers)
        self.assertNotIn("operator-private", fake.peers)
        session = fake.sessions["discord-public-1496735735078715542"]
        self.assertEqual(session.messages[0]["peer"], "public-room")
        self.assertEqual(session.messages[0]["metadata"]["channel_kind"], "public")

    def test_public_snapshot_chat_uses_public_peer(self) -> None:
        fake = _FakeHoncho()
        env = {
            "DGMH_HONCHO_PUBLIC_PEER": "public-room",
            "DGMH_HONCHO_OPERATOR_PEER": "operator-private",
        }
        with mock.patch.dict(os.environ, env), mock.patch(
            "dgmh.honcho_client.get_client",
            return_value=fake,
        ):
            out = chat_about_public_persona("공개 채널 톤?")

        self.assertIn("공개 채널", out)
        self.assertIn("public-room", fake.peers)
        self.assertNotIn("operator-private", fake.peers)


if __name__ == "__main__":
    unittest.main()
