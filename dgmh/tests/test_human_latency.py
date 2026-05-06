"""Step 2 (v3) — bimodal humanlike latency with operator shortcut.

Buckets gated on activity since last bot send:
  active (< 30s)   → [3, 20]
  warm   (< 600s)  → [5, 45]
  cold   (>= 600s) → [10, 90]

AC1.1: when the inbound author is the operator (user id 266436073557590016
by default), the operator-shortcut bucket fires regardless of activity:
  operator → [3, 15]
"""

from __future__ import annotations

import os
import random
import unittest
from unittest import mock

from dgmh.hermes_integration.humanness_hook import (
    _LAST_BOT_SEND_TS,
    _human_latency_seconds,
    _public_human_mode_enabled,
    _record_bot_send,
    _resolve_is_operator,
    _seconds_since_last_msg,
)


class TestSecondsSinceLastMsg(unittest.TestCase):
    def setUp(self) -> None:
        _LAST_BOT_SEND_TS.clear()

    def test_no_history_returns_huge(self) -> None:
        self.assertGreater(_seconds_since_last_msg("ch-new"), 1e8)

    def test_record_then_query(self) -> None:
        _record_bot_send("ch-1", ts=1000.0)
        # 5 seconds later
        self.assertAlmostEqual(
            _seconds_since_last_msg("ch-1", now=1005.0), 5.0
        )

    def test_clamps_to_non_negative(self) -> None:
        _record_bot_send("ch-1", ts=2000.0)
        # Now is BEFORE the last send (clock skew); should not return negative.
        self.assertEqual(_seconds_since_last_msg("ch-1", now=1500.0), 0.0)


class TestHumanLatencyBuckets(unittest.TestCase):
    """Sample 200 draws per bucket and assert the empirical range."""

    def setUp(self) -> None:
        _LAST_BOT_SEND_TS.clear()
        # Pin RNG so every test is deterministic.
        self.rng = random.Random(20260506)

    def test_active_bucket(self) -> None:
        _record_bot_send("ch-a", ts=1000.0)
        samples = [
            _human_latency_seconds(
                "ch-a", is_operator=False, rng=self.rng, now=1010.0
            )
            for _ in range(200)
        ]
        for s in samples:
            self.assertGreaterEqual(s, 3.0)
            self.assertLessEqual(s, 20.0)

    def test_warm_bucket(self) -> None:
        _record_bot_send("ch-w", ts=1000.0)
        samples = [
            _human_latency_seconds(
                "ch-w", is_operator=False, rng=self.rng, now=1300.0
            )
            for _ in range(200)
        ]
        for s in samples:
            self.assertGreaterEqual(s, 5.0)
            self.assertLessEqual(s, 45.0)

    def test_cold_bucket(self) -> None:
        _record_bot_send("ch-c", ts=1000.0)
        samples = [
            _human_latency_seconds(
                "ch-c", is_operator=False, rng=self.rng, now=2000.0
            )
            for _ in range(200)
        ]
        for s in samples:
            self.assertGreaterEqual(s, 10.0)
            self.assertLessEqual(s, 90.0)

    def test_no_history_uses_cold_bucket(self) -> None:
        # Empty history → seconds_since_last_msg returns 1e9 → cold.
        samples = [
            _human_latency_seconds(
                "ch-fresh", is_operator=False, rng=self.rng
            )
            for _ in range(50)
        ]
        for s in samples:
            self.assertGreaterEqual(s, 10.0)
            self.assertLessEqual(s, 90.0)

    def test_active_bucket_median_below_warm_median(self) -> None:
        """Sanity: bucket means are ordered active < warm < cold."""
        _record_bot_send("ch-act", ts=1000.0)
        _record_bot_send("ch-warm", ts=1000.0)
        _record_bot_send("ch-cold", ts=1000.0)
        n = 500
        active = [
            _human_latency_seconds(
                "ch-act", is_operator=False, rng=self.rng, now=1010.0
            )
            for _ in range(n)
        ]
        warm = [
            _human_latency_seconds(
                "ch-warm", is_operator=False, rng=self.rng, now=1300.0
            )
            for _ in range(n)
        ]
        cold = [
            _human_latency_seconds(
                "ch-cold", is_operator=False, rng=self.rng, now=2000.0
            )
            for _ in range(n)
        ]
        self.assertLess(sum(active) / n, sum(warm) / n)
        self.assertLess(sum(warm) / n, sum(cold) / n)


class TestOperatorShortcut(unittest.TestCase):
    """AC1.1: operator latency stays in [3, 20] regardless of activity."""

    def setUp(self) -> None:
        _LAST_BOT_SEND_TS.clear()
        self.rng = random.Random(20260506)

    def test_operator_shortcut_caps_at_15(self) -> None:
        _record_bot_send("ch-cold", ts=1000.0)
        # Cold surface, but operator → still active-shortcut bucket [3, 15].
        samples = [
            _human_latency_seconds(
                "ch-cold", is_operator=True, rng=self.rng, now=99999.0
            )
            for _ in range(200)
        ]
        for s in samples:
            self.assertGreaterEqual(s, 3.0)
            self.assertLessEqual(s, 15.0)

    def test_operator_shortcut_never_uses_warm_or_cold(self) -> None:
        """Bucket should never produce a value above 15s for operator."""
        _record_bot_send("ch-warm", ts=1000.0)
        for _ in range(500):
            s = _human_latency_seconds(
                "ch-warm", is_operator=True, rng=self.rng, now=1300.0
            )
            self.assertLessEqual(s, 15.0)


class TestResolveIsOperator(unittest.TestCase):
    def setUp(self) -> None:
        # Restore default operator id for each test.
        os.environ.pop("DGMH_OPERATOR_USER_ID", None)

    def test_default_operator_id(self) -> None:
        self.assertTrue(
            _resolve_is_operator({"author_id": "266436073557590016"})
        )

    def test_non_operator_id(self) -> None:
        self.assertFalse(_resolve_is_operator({"author_id": "999111"}))

    def test_user_id_alias(self) -> None:
        self.assertTrue(
            _resolve_is_operator({"user_id": "266436073557590016"})
        )

    def test_explicit_author_id_arg(self) -> None:
        self.assertTrue(
            _resolve_is_operator(None, author_id="266436073557590016")
        )

    def test_no_metadata_returns_false(self) -> None:
        self.assertFalse(_resolve_is_operator(None))
        self.assertFalse(_resolve_is_operator({}))

    def test_env_override_supports_list(self) -> None:
        with mock.patch.dict(
            os.environ, {"DGMH_OPERATOR_USER_ID": "111, 222"}
        ):
            self.assertTrue(_resolve_is_operator({"author_id": "111"}))
            self.assertTrue(_resolve_is_operator({"author_id": "222"}))
            self.assertFalse(_resolve_is_operator({"author_id": "333"}))


class TestPublicHumanModeFlag(unittest.TestCase):
    def test_unset_is_disabled(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DGMH_PUBLIC_HUMAN_MODE", None)
            self.assertFalse(_public_human_mode_enabled())

    def test_set_to_one_is_enabled(self) -> None:
        with mock.patch.dict(
            os.environ, {"DGMH_PUBLIC_HUMAN_MODE": "1"}
        ):
            self.assertTrue(_public_human_mode_enabled())

    def test_set_to_zero_is_disabled(self) -> None:
        with mock.patch.dict(
            os.environ, {"DGMH_PUBLIC_HUMAN_MODE": "0"}
        ):
            self.assertFalse(_public_human_mode_enabled())


if __name__ == "__main__":
    unittest.main()
