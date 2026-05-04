"""Tests for dgmh.hermes_integration.humanness_hook gating logic."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from dgmh.hermes_integration.humanness_hook import _should_score


class TestShouldScore(unittest.TestCase):
    def test_long_korean_passes(self) -> None:
        text = "응, 그 방향으로 가. 검증 능력이 더 중요해질 거야. 직접 코드 못 치면 위험하긴 해."
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DGMH_HUMANNESS_DISABLED", None)
            self.assertTrue(_should_score(text))

    def test_empty_skipped(self) -> None:
        self.assertFalse(_should_score(""))
        self.assertFalse(_should_score("   "))
        self.assertFalse(_should_score("\n\t  "))

    def test_short_skipped(self) -> None:
        self.assertFalse(_should_score("응."))
        self.assertFalse(_should_score("ㅇㅋ"))

    def test_error_prefix_skipped(self) -> None:
        self.assertFalse(_should_score("Error: LLM call failed for reasons unknown"))
        self.assertFalse(_should_score("⚠️ Gateway shutting down — interrupted"))
        self.assertFalse(_should_score("[error] something went wrong here please retry"))

    def test_disabled_env_skipped(self) -> None:
        text = "응, 그 방향으로 가. 검증 능력이 더 중요해질 거야. 직접 코드 못 치면 위험."
        with mock.patch.dict(os.environ, {"DGMH_HUMANNESS_DISABLED": "1"}):
            self.assertFalse(_should_score(text))

    def test_min_chars_env_override(self) -> None:
        text = "안녕 어서와"  # 6 chars
        with mock.patch.dict(os.environ, {"DGMH_HUMANNESS_MIN_CHARS": "5"}):
            os.environ.pop("DGMH_HUMANNESS_DISABLED", None)
            self.assertTrue(_should_score(text))
        with mock.patch.dict(os.environ, {"DGMH_HUMANNESS_MIN_CHARS": "20"}):
            os.environ.pop("DGMH_HUMANNESS_DISABLED", None)
            self.assertFalse(_should_score(text))


if __name__ == "__main__":
    unittest.main()
