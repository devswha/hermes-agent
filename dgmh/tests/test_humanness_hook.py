"""Tests for dgmh.hermes_integration.humanness_hook gating logic."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from dgmh.hermes_integration.humanness_hook import (
    _INLINE_BACKTICK_RE,
    _INLINE_BOLD_RE,
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

    def test_inline_backtick_in_short_reply_flags(self) -> None:
        # Decorating a casual one-liner with `…` is the operator-flagged
        # AI-tone tell — real chat just says the date plainly.
        hit, flags = _structural_pollution_check("오늘은 `2026-05-11 월요일 KST`.")
        self.assertTrue(hit)
        self.assertTrue(any(f.startswith("inline-backtick") for f in flags))

    def test_two_inline_backticks_in_long_reply_flags(self) -> None:
        text = (
            "로드맵은 `CLI → Git → GitHub PR`까지 본 다음 `Docker → compose → 배포`로 "
            "넘어가는 게 자연스러워. 정치는 제도 신뢰가 핵심 같음."
        )
        hit, flags = _structural_pollution_check(text)
        self.assertTrue(hit)
        self.assertTrue(any(f.startswith("inline-backtick") for f in flags))

    def test_single_backtick_in_very_long_reply_passes(self) -> None:
        # In long technical chat (200+ chars), a single inline tick is
        # reasonable — don't false-positive every command name mention.
        text = (
            "어제 시도한 `npm install` 단계에서 멈춘 거 같은데, 의존성 잠금 파일을 다시 "
            "보면 그쪽이 문제일 수도 있어. 일단 빌드까지 한 번 끝까지 돌려보고 "
            "결과로 다시 얘기하는 게 더 빠르겠다. 내가 보기엔 그쪽 흐름이 자연스러워. "
            "당장은 큰 영향 없을 거고 다음 PR에서 같이 가져가면 돼."
        )
        hit, flags = _structural_pollution_check(text)
        self.assertFalse(any(f.startswith("inline-backtick") for f in flags))

    def test_fenced_code_block_does_not_trigger_backtick_flag(self) -> None:
        # Fenced code blocks are stripped before the inline check; only
        # decoration ticks should count, not legitimate multi-line code.
        text = "예시:\n```python\nprint(x)\n```\n쓰면 되긴 해."
        hit, flags = _structural_pollution_check(text)
        self.assertFalse(any(f.startswith("inline-backtick") for f in flags))

    def test_hedge_softener_pileup_flags(self) -> None:
        # Three or more "…같아 / …보여 / …듯" in one response = over-hedging.
        text = (
            "양당 피로감이 크게 느껴지는 것 같아. 대안 세력은 설득력이 약해 보여. "
            "정권 심판보다는 정치 품질 저하가 더 커 보이는 듯해."
        )
        hit, flags = _structural_pollution_check(text)
        self.assertTrue(hit)
        self.assertTrue(any(f.startswith("hedge-softener") for f in flags))

    def test_single_hedge_softener_passes(self) -> None:
        # Casual chat naturally uses one or two "같아"/"보여" — don't punish that.
        hit, flags = _structural_pollution_check("그쪽이 맞는 것 같아.")
        self.assertFalse(any(f.startswith("hedge-softener") for f in flags))


class TestInlineBacktickStrip(unittest.TestCase):
    """The deterministic post-rewrite strip used in wrapped_send.

    Patina is probabilistic and sometimes preserves decorative ticks
    even when prompted to remove them. Strip them via this regex as the
    project's hard guarantee, independent of profile behavior.
    """

    def test_single_pair_stripped(self) -> None:
        out = _INLINE_BACKTICK_RE.sub(r"\1", "오늘은 `2026-05-11 월요일 KST`.")
        self.assertEqual(out, "오늘은 2026-05-11 월요일 KST.")

    def test_multiple_pairs_all_stripped(self) -> None:
        text = (
            "도커는 `honcho-api`, `honcho-deriver`, `honcho-redis`, "
            "`honcho-database`, `flask-postgres` 떠 있어."
        )
        out = _INLINE_BACKTICK_RE.sub(r"\1", text)
        self.assertNotIn("`", out)
        self.assertIn("honcho-api", out)
        self.assertIn("flask-postgres", out)

    def test_noop_when_no_ticks(self) -> None:
        text = "백틱 없는 평범한 문장이야."
        out = _INLINE_BACKTICK_RE.sub(r"\1", text)
        self.assertEqual(out, text)

    def test_does_not_collapse_across_newlines(self) -> None:
        # The regex is line-bounded to keep fenced-block semantics. An
        # opening tick on one line and a closing tick on another must
        # not be treated as a single pair.
        text = "첫 줄 `start\n끝 줄에서` 닫힘"
        out = _INLINE_BACKTICK_RE.sub(r"\1", text)
        # No same-line pair → no substitution happens.
        self.assertEqual(out, text)


class TestInlineBoldDetectionAndStrip(unittest.TestCase):
    """Word-level **bold** patina sometimes leaves on titles/brand names."""

    def test_pollution_flag_fires_on_single_bold(self) -> None:
        hit, flags = _structural_pollution_check("취향 모르면 **삼체** 추천.")
        self.assertTrue(hit)
        self.assertTrue(any(f.startswith("inline-bold") for f in flags))

    def test_pollution_flag_counts_multiple_bolds(self) -> None:
        text = "**삼체** 보고 가볍게는 **원피스 실사**, 세면 **베이비 레인디어**."
        hit, flags = _structural_pollution_check(text)
        self.assertTrue(hit)
        # Three bolds expected
        bold_flags = [f for f in flags if f.startswith("inline-bold")]
        self.assertEqual(bold_flags, ["inline-bold(3)"])

    def test_strip_keeps_content(self) -> None:
        out = _INLINE_BOLD_RE.sub(r"\1", "취향 모르면 **삼체** 추천.")
        self.assertEqual(out, "취향 모르면 삼체 추천.")

    def test_strip_multiple_keeps_all_titles(self) -> None:
        text = "**삼체**랑 **원피스 실사** 둘 다 볼만함"
        out = _INLINE_BOLD_RE.sub(r"\1", text)
        self.assertEqual(out, "삼체랑 원피스 실사 둘 다 볼만함")

    def test_noop_when_no_bold(self) -> None:
        text = "그냥 평범한 문장이야"
        out = _INLINE_BOLD_RE.sub(r"\1", text)
        self.assertEqual(out, text)

    def test_does_not_match_single_asterisks(self) -> None:
        # Single * for italics is a different concern; this regex
        # targets ** specifically to avoid mangling math/symbol uses.
        text = "x*y 곱하기 결과"
        out = _INLINE_BOLD_RE.sub(r"\1", text)
        self.assertEqual(out, text)


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
