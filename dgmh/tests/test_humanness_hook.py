"""Tests for dgmh.hermes_integration.humanness_hook gating logic."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from dgmh.hermes_integration.humanness_hook import (
    _prune_polluting_message,
    _should_score,
)


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


class TestPruneLogic(unittest.TestCase):
    """End-to-end test of _prune_polluting_message against a temp state.db."""

    def _make_db(self, tmpdir: str, rows: list[tuple[str, str, str]]) -> str:
        """Build a state.db at HERMES_HOME=tmpdir with given (session, role, content) rows.

        Timestamp is set to 'now' for all rows so the lookback window catches them.
        """
        import sqlite3
        from pathlib import Path

        db = Path(tmpdir) / "state.db"
        con = sqlite3.connect(str(db))
        con.executescript(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, "
            "role TEXT NOT NULL, content TEXT, timestamp REAL NOT NULL);"
        )
        import time

        now = time.time()
        for sid, role, content in rows:
            con.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (sid, role, content, now),
            )
        con.commit()
        con.close()
        return str(db)

    def _count(self, db: str) -> int:
        import sqlite3

        con = sqlite3.connect(db)
        c = con.execute("SELECT count(*) FROM messages").fetchone()[0]
        con.close()
        return c

    def test_high_score_prunes_assistant_row(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(
                tmp,
                [
                    ("s1", "user", "야"),
                    ("s1", "assistant", "5-bullet polluting reply text here"),
                ],
            )
            with mock.patch.dict(os.environ, {"HERMES_HOME": tmp}, clear=False):
                os.environ.pop("DGMH_PRUNE_DISABLED", None)
                n = _prune_polluting_message(
                    "5-bullet polluting reply text here",
                    ai_score=20.0,
                )
            self.assertEqual(n, 1)
            self.assertEqual(self._count(db), 1)

    def test_low_score_does_not_prune(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(
                tmp,
                [
                    ("s1", "assistant", "clean human reply"),
                ],
            )
            with mock.patch.dict(os.environ, {"HERMES_HOME": tmp}, clear=False):
                os.environ.pop("DGMH_PRUNE_DISABLED", None)
                n = _prune_polluting_message(
                    "clean human reply",
                    ai_score=5.0,
                )
            self.assertEqual(n, 0)
            self.assertEqual(self._count(db), 1)

    def test_disabled_env_skips_prune(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(
                tmp,
                [
                    ("s1", "assistant", "polluting reply"),
                ],
            )
            with mock.patch.dict(
                os.environ, {"HERMES_HOME": tmp, "DGMH_PRUNE_DISABLED": "1"}
            ):
                n = _prune_polluting_message(
                    "polluting reply",
                    ai_score=50.0,
                )
            self.assertEqual(n, 0)
            self.assertEqual(self._count(db), 1)

    def test_threshold_env_override(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(
                tmp,
                [
                    ("s1", "assistant", "moderate reply"),
                ],
            )
            with mock.patch.dict(
                os.environ,
                {"HERMES_HOME": tmp, "DGMH_PRUNE_AI_THRESHOLD": "5"},
            ):
                os.environ.pop("DGMH_PRUNE_DISABLED", None)
                n = _prune_polluting_message(
                    "moderate reply",
                    ai_score=8.0,
                )
            self.assertEqual(n, 1)

    def test_user_messages_not_pruned(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(
                tmp,
                [
                    ("s1", "user", "high score user text"),
                ],
            )
            with mock.patch.dict(os.environ, {"HERMES_HOME": tmp}, clear=False):
                os.environ.pop("DGMH_PRUNE_DISABLED", None)
                n = _prune_polluting_message(
                    "high score user text",
                    ai_score=99.0,
                )
            self.assertEqual(n, 0)
            self.assertEqual(self._count(db), 1)


if __name__ == "__main__":
    unittest.main()
