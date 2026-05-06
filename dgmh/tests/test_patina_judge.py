"""Tests for dgmh.patina_judge.

Unit tests use a fake patina binary stub; integration tests requiring the real
patina CLI are gated behind DGMH_PATINA_LIVE=1 to keep CI fast and offline.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from unittest import mock

from dgmh.patina_judge import (
    PatinaScore,
    PatinaScoreError,
    _parse_score_output,
    composite_reward,
    humanness_rewrite_with_profile,
    score_humanness,
)


_SAMPLE_OUTPUT = """\
| Category | Weight | Detected | Raw Score | Weighted |
|----------|--------|----------|-----------|----------|
| content | 0.20 | 없음 | 0.0 | 0.0 |
| language | 0.20 | 없음 | 0.0 | 0.0 |
| style | 0.20 | 없음 | 0.0 | 0.0 |
| communication | 0.15 | #19 챗봇 표현 High, #21 아첨하는 말투 High | 66.7 | 10.0 |
| filler | 0.10 | 없음 | 0.0 | 0.0 |
| structure | 0.15 | 없음 | 0.0 | 0.0 |
| **Overall** | | | | **10.0 (±10)** |

Interpretation: 사람다움. 다만 짧은 문장 안에 "좋은 질문이십니다"가 있어 챗봇 신호는 뚜렷합니다.
"""


_HEAVY_AI_OUTPUT = """\
| Category | Weight | Detected | Raw Score | Weighted |
|----------|--------|----------|-----------|----------|
| content | 0.20 | #1 ~적 남발 High, #5 번역체 Medium | 50.0 | 10.0 |
| language | 0.20 | #8 보다 비교 High | 33.3 | 6.7 |
| style | 0.20 | #18 한자어 Medium | 22.2 | 4.4 |
| communication | 0.15 | #19 챗봇 표현 High, #21 아첨 High | 66.7 | 10.0 |
| filler | 0.10 | #31 결론적으로 Medium | 16.7 | 1.7 |
| structure | 0.15 | #25 평행 list Medium | 13.3 | 2.0 |
| **Overall** | | | | **34.8 (±10)** |

Interpretation: AI-like.
"""


class TestParseScoreOutput(unittest.TestCase):
    def test_parse_low_ai_score(self) -> None:
        ai_score, sub_scores, interpretation = _parse_score_output(_SAMPLE_OUTPUT)
        self.assertEqual(ai_score, 10.0)
        self.assertEqual(sub_scores["communication"], 66.7)
        self.assertEqual(sub_scores["content"], 0.0)
        self.assertIn("사람다움", interpretation)

    def test_parse_high_ai_score(self) -> None:
        ai_score, sub_scores, _ = _parse_score_output(_HEAVY_AI_OUTPUT)
        self.assertEqual(ai_score, 34.8)
        self.assertEqual(sub_scores["communication"], 66.7)
        self.assertEqual(sub_scores["content"], 50.0)

    def test_parse_overall_missing_raises(self) -> None:
        with self.assertRaises(PatinaScoreError):
            _parse_score_output("garbage that is not a patina score table")

    def test_parse_excludes_header_rows(self) -> None:
        _, sub_scores, _ = _parse_score_output(_SAMPLE_OUTPUT)
        self.assertNotIn("category", sub_scores)
        self.assertNotIn("overall", sub_scores)


class TestCompositeReward(unittest.TestCase):
    def test_balanced_weights_at_unity(self) -> None:
        r = composite_reward(
            human_likeness=80.0,
            codex_judge_quality=70.0,
            reaction_signal=50.0,
        )
        # 0.50*80 + 0.30*70 + 0.20*50 = 40 + 21 + 10 = 71
        self.assertAlmostEqual(r, 71.0, places=2)

    def test_thumbs_up_pushes_reward(self) -> None:
        baseline = composite_reward(
            human_likeness=70.0, codex_judge_quality=70.0, reaction_signal=50.0
        )
        with_thumbs_up = composite_reward(
            human_likeness=70.0, codex_judge_quality=70.0, reaction_signal=100.0
        )
        self.assertGreater(with_thumbs_up, baseline)
        # 0.20 * 50 = 10 point swing
        self.assertAlmostEqual(with_thumbs_up - baseline, 10.0, places=2)

    def test_thumbs_down_pulls_reward(self) -> None:
        with_thumbs_down = composite_reward(
            human_likeness=70.0, codex_judge_quality=70.0, reaction_signal=0.0
        )
        baseline = composite_reward(
            human_likeness=70.0, codex_judge_quality=70.0, reaction_signal=50.0
        )
        self.assertLess(with_thumbs_down, baseline)
        self.assertAlmostEqual(baseline - with_thumbs_down, 10.0, places=2)

    def test_human_likeness_dominates(self) -> None:
        r_human = composite_reward(
            human_likeness=100.0, codex_judge_quality=0.0, reaction_signal=0.0
        )
        r_judge = composite_reward(
            human_likeness=0.0, codex_judge_quality=100.0, reaction_signal=0.0
        )
        # human weight 0.50 > judge weight 0.30
        self.assertGreater(r_human, r_judge)
        self.assertAlmostEqual(r_human, 50.0, places=2)
        self.assertAlmostEqual(r_judge, 30.0, places=2)


class TestScoreHumannessGuards(unittest.TestCase):
    def test_empty_text_returns_full_humanness(self) -> None:
        result = score_humanness("")
        self.assertEqual(result.ai_score, 0.0)
        self.assertEqual(result.human_likeness, 100.0)
        self.assertEqual(result.interpretation, "empty")

    def test_whitespace_only_returns_full_humanness(self) -> None:
        result = score_humanness("   \n\t  ")
        self.assertEqual(result.human_likeness, 100.0)

    def test_missing_binary_raises(self) -> None:
        with self.assertRaises(PatinaScoreError) as ctx:
            score_humanness(
                "test",
                patina_bin="/nonexistent/path/patina.js",
            )
        self.assertIn("not found", str(ctx.exception))


@unittest.skipUnless(
    os.environ.get("DGMH_PATINA_LIVE") == "1"
    and Path(
        Path.home() / ".hermes" / "skills" / "creative" / "patina" / "bin" / "patina.js"
    ).exists(),
    "live patina test — set DGMH_PATINA_LIVE=1 to run, requires installed patina",
)
class TestScoreHumannessLive(unittest.TestCase):
    def test_clean_korean_scores_low(self) -> None:
        result = score_humanness("응, 그 방향으로 가. 검증 능력이 더 중요해질 거야.")
        self.assertLess(result.ai_score, 30.0)
        self.assertGreater(result.human_likeness, 70.0)

    def test_chatbot_filler_scores_high(self) -> None:
        text = "좋은 질문이십니다! 도움이 되셨으면 좋겠습니다. 궁금한 점이 있으시면 말씀해 주세요."
        result = score_humanness(text)
        self.assertGreater(result.ai_score, 0.0)
        self.assertIn("communication", result.sub_scores)


class TestHumannessRewriteWithProfile(unittest.TestCase):
    """Step P1 (v3) — programmatic ``patina --profile <profile> --backend <backend>``.

    Validates the new public-channel rewrite surface. The existing
    :func:`humanness_rewrite` is intentionally untouched (see Step P1 plan
    note: 1:1 path stays on the old function, public path uses the new).
    """

    def test_empty_text_returns_none(self) -> None:
        self.assertIsNone(humanness_rewrite_with_profile(""))
        self.assertIsNone(humanness_rewrite_with_profile("   \n  "))

    def test_missing_binary_returns_none(self) -> None:
        result = humanness_rewrite_with_profile(
            "테스트 텍스트", patina_bin="/nonexistent/patina.js"
        )
        self.assertIsNone(result)

    def test_subprocess_success_returns_stdout(self) -> None:
        completed = mock.Mock()
        completed.returncode = 0
        completed.stdout = "응 그쪽이지.\n"
        completed.stderr = ""
        with mock.patch(
            "dgmh.patina_judge.subprocess.run", return_value=completed
        ) as run_mock, mock.patch(
            "dgmh.patina_judge.Path.exists", return_value=True
        ):
            out = humanness_rewrite_with_profile(
                "원본 ChatGPT 어조 텍스트", patina_bin="/fake/patina.js"
            )
        self.assertEqual(out, "응 그쪽이지.")

        args = run_mock.call_args[0][0]
        # Sanity: invocation includes --profile social --backend codex-cli
        # (operator-verified flags, see plan v3 Step P1).
        self.assertIn("--profile", args)
        self.assertEqual(args[args.index("--profile") + 1], "social")
        self.assertIn("--backend", args)
        self.assertEqual(args[args.index("--backend") + 1], "codex-cli")
        self.assertIn("--lang", args)
        self.assertEqual(args[args.index("--lang") + 1], "ko")

    def test_subprocess_nonzero_exit_returns_none(self) -> None:
        completed = mock.Mock()
        completed.returncode = 2
        completed.stdout = ""
        completed.stderr = "patina: backend codex-cli unauthenticated"
        with mock.patch(
            "dgmh.patina_judge.subprocess.run", return_value=completed
        ), mock.patch("dgmh.patina_judge.Path.exists", return_value=True):
            self.assertIsNone(
                humanness_rewrite_with_profile(
                    "원본 텍스트", patina_bin="/fake/patina.js"
                )
            )

    def test_subprocess_timeout_returns_none(self) -> None:
        import subprocess as _sp

        with mock.patch(
            "dgmh.patina_judge.subprocess.run",
            side_effect=_sp.TimeoutExpired(cmd=["node"], timeout=30),
        ), mock.patch("dgmh.patina_judge.Path.exists", return_value=True):
            self.assertIsNone(
                humanness_rewrite_with_profile(
                    "원본 텍스트", patina_bin="/fake/patina.js"
                )
            )

    def test_short_stdout_returns_none(self) -> None:
        """An empty / one-character stdout is treated as a non-result."""
        completed = mock.Mock()
        completed.returncode = 0
        completed.stdout = "ㅇ"
        completed.stderr = ""
        with mock.patch(
            "dgmh.patina_judge.subprocess.run", return_value=completed
        ), mock.patch("dgmh.patina_judge.Path.exists", return_value=True):
            self.assertIsNone(
                humanness_rewrite_with_profile(
                    "원본 텍스트", patina_bin="/fake/patina.js"
                )
            )

    def test_passes_text_via_stdin(self) -> None:
        completed = mock.Mock()
        completed.returncode = 0
        completed.stdout = "rewritten"
        completed.stderr = ""
        with mock.patch(
            "dgmh.patina_judge.subprocess.run", return_value=completed
        ) as run_mock, mock.patch(
            "dgmh.patina_judge.Path.exists", return_value=True
        ):
            humanness_rewrite_with_profile(
                "원본 입력 텍스트", patina_bin="/fake/patina.js"
            )
        # subprocess.run was called with input= containing the source.
        call_kwargs = run_mock.call_args.kwargs
        self.assertEqual(call_kwargs["input"], "원본 입력 텍스트")
        self.assertTrue(call_kwargs.get("text", False))

    def test_strips_patina_analyst_preamble(self) -> None:
        """Regression: patina prepends an analyst preamble paragraph
        (e.g. "아직 AI 티 나는 부분: ...") before the actual rewrite. The
        preamble is dropped; only the body is returned.
        """
        preamble = (
            "아직 AI 티 나는 부분: 딱히 없음. 다만 첫 문장의 호흡이 "
            "조금 길어 보일 수 있음."
        )
        body = "응 그쪽이지."
        completed = mock.Mock()
        completed.returncode = 0
        completed.stdout = f"{preamble}\n\n{body}\n"
        completed.stderr = ""
        with mock.patch(
            "dgmh.patina_judge.subprocess.run", return_value=completed
        ), mock.patch("dgmh.patina_judge.Path.exists", return_value=True):
            out = humanness_rewrite_with_profile(
                "원본 텍스트", patina_bin="/fake/patina.js"
            )
        self.assertEqual(out, body)
        self.assertNotIn("AI 티", out or "")
        self.assertNotIn("다만", out or "")

    def test_preserves_multi_paragraph_rewrite_body(self) -> None:
        """No preamble: all paragraph blocks must survive joined by blank lines."""
        body = (
            "그건 좋은 아이디어야.\n\n"
            "다만 자세한 건 생각해봐야 해.\n\n"
            "응, 한번 해보자."
        )
        completed = mock.Mock()
        completed.returncode = 0
        completed.stdout = body + "\n"
        completed.stderr = ""
        with mock.patch(
            "dgmh.patina_judge.subprocess.run", return_value=completed
        ), mock.patch("dgmh.patina_judge.Path.exists", return_value=True):
            out = humanness_rewrite_with_profile(
                "원본 텍스트", patina_bin="/fake/patina.js"
            )
        self.assertIn("그건 좋은 아이디어야.", out or "")
        self.assertIn("다만 자세한 건 생각해봐야 해.", out or "")
        self.assertIn("응, 한번 해보자.", out or "")

    def test_preamble_then_multi_paragraph_body(self) -> None:
        """Preamble dropped; both subsequent body paragraphs preserved."""
        preamble = "아직 AI 티 나는 부분: 없음."
        body_p1 = "현재 버전은 0.9.2야."
        body_p2 = "업데이트는 다음 주에 나올 것 같아."
        completed = mock.Mock()
        completed.returncode = 0
        completed.stdout = f"{preamble}\n\n{body_p1}\n\n{body_p2}\n"
        completed.stderr = ""
        with mock.patch(
            "dgmh.patina_judge.subprocess.run", return_value=completed
        ), mock.patch("dgmh.patina_judge.Path.exists", return_value=True):
            out = humanness_rewrite_with_profile(
                "원본 텍스트", patina_bin="/fake/patina.js"
            )
        self.assertNotIn("AI 티", out or "")
        self.assertIn(body_p1, out or "")
        self.assertIn(body_p2, out or "")


if __name__ == "__main__":
    unittest.main()
