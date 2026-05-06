"""Tests for dgmh.hermes_integration.humanness_hook gating logic."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from dgmh.hermes_integration.humanness_hook import (
    _prune_polluting_message,
    _should_score,
    _structural_pollution_check,
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


class TestStructuralPollutionCheck(unittest.TestCase):
    def test_clean_short_reply(self) -> None:
        hit, flags = _structural_pollution_check("응 그쪽이지. 검증이 더 중요해.")
        self.assertFalse(hit)
        self.assertEqual(flags, [])

    def test_5_bullet_polluting(self) -> None:
        text = (
            "주로 하는 일:\n"
            "- 코드\n- 디버그\n- 테스트\n- Git\n- 문서화\n"
        )
        hit, flags = _structural_pollution_check(text)
        self.assertTrue(hit)
        self.assertTrue(any("bullet-list" in f for f in flags))
        self.assertTrue(any("colon-introducing-list" in f for f in flags))

    def test_bold_label_header(self) -> None:
        hit, flags = _structural_pollution_check("**핵심:** 그건 좀 아닌 듯.")
        self.assertTrue(hit)
        self.assertIn("bold-label-header", flags)

    def test_closing_hedge_single_line(self) -> None:
        hit, flags = _structural_pollution_check("응 가자. 다만 코드는 알아야 해.")
        self.assertTrue(hit)
        self.assertIn("closing-caveat-hedge", flags)

    def test_closing_hedge_multi_paragraph(self) -> None:
        hit, flags = _structural_pollution_check("응 가자.\n그래도 위험해.")
        self.assertTrue(hit)
        self.assertIn("closing-caveat-hedge", flags)

    def test_hedge_in_middle_does_not_trigger(self) -> None:
        hit, flags = _structural_pollution_check(
            "다만 그거랑 별개로 결과는 좋아."
        )
        self.assertFalse(hit)

    def test_2_bullets_ok(self) -> None:
        hit, _ = _structural_pollution_check("두 가지:\n- A\n- B")
        self.assertFalse(hit)

    def test_bullets_inside_code_block_ignored(self) -> None:
        text = "Python 예시:\n```\n- elem1\n- elem2\n- elem3\n- elem4\n```"
        hit, flags = _structural_pollution_check(text)
        self.assertFalse(hit)


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
                    message_id="discord-msg-1",
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
                    message_id="discord-msg-1",
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
                    message_id="discord-msg-1",
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
                    message_id="discord-msg-1",
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
                    message_id="discord-msg-1",
                    ai_score=99.0,
                )
            self.assertEqual(n, 0)
            self.assertEqual(self._count(db), 1)

    def test_prune_robust_to_content_mutation(self) -> None:
        """AC11 (v3): rewrite stage mutates content; prune still hits.

        State.db row was written BEFORE the rewrite, so its content is the
        pre-rewrite draft. The score thread runs with the post-rewrite
        text. The previous content-matching prune missed; the new
        recency-keyed prune still removes the row.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(
                tmp,
                [
                    ("s1", "user", "야"),
                    ("s1", "assistant", "PRE-REWRITE chatgpt-style draft"),
                ],
            )
            # Caller passes the POST-rewrite content, but the implementation
            # ignores content entirely — prune still hits the recent row.
            with mock.patch.dict(os.environ, {"HERMES_HOME": tmp}, clear=False):
                os.environ.pop("DGMH_PRUNE_DISABLED", None)
                n = _prune_polluting_message(
                    message_id="discord-msg-99",
                    ai_score=99.0,
                )
            self.assertEqual(n, 1)
            self.assertEqual(self._count(db), 1)

    def test_prune_only_one_row_with_multiple_recent_assistants(self) -> None:
        """Two recent assistant rows → prune deletes ONLY the most recent
        (rowcount == 1, never accidentally cascading)."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            db = self._make_db(
                tmp,
                [
                    ("s1", "assistant", "older bot reply"),
                    ("s1", "assistant", "newest bot reply"),
                ],
            )
            with mock.patch.dict(os.environ, {"HERMES_HOME": tmp}, clear=False):
                os.environ.pop("DGMH_PRUNE_DISABLED", None)
                n = _prune_polluting_message(
                    message_id="discord-msg-1",
                    ai_score=50.0,
                )
            self.assertEqual(n, 1)
            # Two rows in fixture → one removed → one remaining.
            self.assertEqual(self._count(db), 1)

    def test_ac11_ten_forced_pollutions_all_pruned(self) -> None:
        """AC11 (v3): 10 forced pollutions → rowcount=1 each time.

        Repeats the prune cycle ten times against a fresh-each-iteration db,
        verifying the keyed-by-recency delete consistently removes a single
        row regardless of content.
        """
        import tempfile

        for i in range(10):
            with tempfile.TemporaryDirectory() as tmp:
                db = self._make_db(
                    tmp,
                    [
                        ("s1", "user", f"user msg {i}"),
                        ("s1", "assistant", f"polluting bot reply {i}"),
                    ],
                )
                with mock.patch.dict(
                    os.environ, {"HERMES_HOME": tmp}, clear=False
                ):
                    os.environ.pop("DGMH_PRUNE_DISABLED", None)
                    n = _prune_polluting_message(
                        message_id=f"discord-msg-{i}",
                        ai_score=50.0,
                    )
                self.assertEqual(n, 1, f"iteration {i}: expected rowcount=1")
                self.assertEqual(self._count(db), 1, f"iteration {i}: user row should remain")


if __name__ == "__main__":
    unittest.main()
