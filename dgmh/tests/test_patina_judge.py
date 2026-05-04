"""Tests for dgmh.patina_judge.

Unit tests use a fake patina binary stub; integration tests requiring the real
patina CLI are gated behind DGMH_PATINA_LIVE=1 to keep CI fast and offline.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from dgmh.patina_judge import (
    PatinaScore,
    PatinaScoreError,
    _parse_score_output,
    composite_reward,
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


if __name__ == "__main__":
    unittest.main()
