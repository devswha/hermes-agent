"""Tests for dgmh.hermes_integration.gate_decisions cache."""

from __future__ import annotations

import unittest

from dgmh.hermes_integration import gate_decisions as gd


class TestRecordAndGet(unittest.TestCase):
    def setUp(self) -> None:
        gd._reset_for_tests()

    def test_get_unknown_returns_none(self) -> None:
        self.assertIsNone(gd.get_decision("nope"))

    def test_record_then_get_roundtrip(self) -> None:
        gd.record_decision("m1", decision="send", in_len=50, out_len=50)
        dec = gd.get_decision("m1")
        assert dec is not None
        self.assertEqual(dec.decision, "send")
        self.assertEqual(dec.in_len, 50)
        self.assertEqual(dec.out_len, 50)
        self.assertFalse(dec.ends_in_ellipsis)
        self.assertFalse(dec.truncated)

    def test_compress_decision_is_truncated(self) -> None:
        gd.record_decision("m1", decision="compress", in_len=1100, out_len=199)
        dec = gd.get_decision("m1")
        assert dec is not None
        self.assertTrue(dec.truncated)

    def test_ellipsis_marker_flags_truncated(self) -> None:
        gd.record_decision(
            "m1",
            decision="send",
            in_len=200,
            out_len=200,
            ends_in_ellipsis=True,
        )
        dec = gd.get_decision("m1")
        assert dec is not None
        self.assertTrue(dec.truncated)

    def test_empty_msg_id_is_ignored(self) -> None:
        gd.record_decision("", decision="send", in_len=1, out_len=1)
        self.assertIsNone(gd.get_decision(""))


class TestEllipsisMarker(unittest.TestCase):
    def test_unicode_ellipsis(self) -> None:
        self.assertTrue(gd._ends_with_ellipsis_marker("hello…"))

    def test_ascii_three_dots(self) -> None:
        self.assertTrue(gd._ends_with_ellipsis_marker("hello..."))

    def test_no_ellipsis(self) -> None:
        self.assertFalse(gd._ends_with_ellipsis_marker("hello."))

    def test_empty(self) -> None:
        self.assertFalse(gd._ends_with_ellipsis_marker(""))


class TestLruEviction(unittest.TestCase):
    def setUp(self) -> None:
        gd._reset_for_tests()

    def test_oldest_entries_evicted_at_cap(self) -> None:
        # Force cap to a small value for the test.
        original_cap = gd._CACHE_MAX
        gd._CACHE_MAX = 5  # type: ignore[assignment]
        try:
            for i in range(7):
                gd.record_decision(
                    f"m{i}", decision="send", in_len=10, out_len=10
                )
            # First two should have been evicted.
            self.assertIsNone(gd.get_decision("m0"))
            self.assertIsNone(gd.get_decision("m1"))
            # Last five remain.
            for i in range(2, 7):
                self.assertIsNotNone(gd.get_decision(f"m{i}"))
        finally:
            gd._CACHE_MAX = original_cap  # type: ignore[assignment]

    def test_repeat_record_moves_to_end(self) -> None:
        gd._CACHE_MAX = 3  # type: ignore[assignment]
        try:
            gd.record_decision("a", decision="send", in_len=1, out_len=1)
            gd.record_decision("b", decision="send", in_len=1, out_len=1)
            gd.record_decision("c", decision="send", in_len=1, out_len=1)
            # touch 'a' — moves to end
            gd.record_decision("a", decision="compress", in_len=99, out_len=10)
            # adding 'd' should evict 'b' (oldest)
            gd.record_decision("d", decision="send", in_len=1, out_len=1)
            self.assertIsNone(gd.get_decision("b"))
            self.assertIsNotNone(gd.get_decision("a"))
            self.assertIsNotNone(gd.get_decision("c"))
            self.assertIsNotNone(gd.get_decision("d"))
        finally:
            gd._CACHE_MAX = 256  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main()
