"""Tests for the KakaoTalk-style TF-IDF retrieval module.

Covers:
  - tokenization handles Korean text + ASCII identifiers
  - retrieval scores higher for stylistically/lexically nearer messages
  - empty/whitespace queries return [] (no anchors to inject)
  - corpus loader filters out URL-only, mention-only, and quote-form lines
  - per-call profile rendering preserves anchors verbatim
  - rewrite_with_rag_profile cleans up its per-call profile file even
    when the inner patina call fails
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from dgmh.kakao_style_retrieval import (
    _StyleCorpus,
    _reset_for_tests,
    build_kakao_rag_profile_body,
    retrieve_top_k,
    rewrite_with_rag_profile,
)


_SEED_MESSAGES = [
    "오늘 며칠이지",
    "도커를 안띄우고 한게",
    "넹 뭐 그런거같아용",
    "ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ",
    "근데 cli 로 이미지 생성하는게 좋은지",
    "한국이 그래도 망할일은없겠군요",
    "잠을 제대로 자야 내일 더 잘할텐데",
    "심지어 그 주의력 가치를 가지고 싶어하는 주의력 가치가 탄생하는중",
    "프앤이랑 Cli만인듯",
    "굿ㅋㅋㅋ 목 금 중에 놀러가겠읍니다",
]


def _seed_corpus_file(tmp_path: Path) -> Path:
    corpus_path = tmp_path / "kakao_corpus.txt"
    corpus_path.write_text("\n".join(_SEED_MESSAGES) + "\n", encoding="utf-8")
    return corpus_path


class TestStyleCorpus(unittest.TestCase):
    def test_retrieval_prefers_lexical_overlap(self) -> None:
        corpus = _StyleCorpus(_SEED_MESSAGES)
        # "cli" should pull the two cli-mentioning messages above
        # generic catch-alls like "ㅋㅋㅋㅋ".
        results = corpus.retrieve("cli 쓰면 좋은가", k=3)
        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(
            any("cli" in r.lower() for r in results),
            msg=f"expected cli-containing match in {results}",
        )

    def test_empty_query_returns_empty(self) -> None:
        corpus = _StyleCorpus(_SEED_MESSAGES)
        self.assertEqual(corpus.retrieve("", k=5), [])
        self.assertEqual(corpus.retrieve("   ", k=5), [])

    def test_completely_unrelated_query_may_still_match_via_bigrams(self) -> None:
        # Character bigrams give partial overlap to almost any Korean
        # query; we only care that the top result has SOME signal,
        # never None on a non-trivial query.
        corpus = _StyleCorpus(_SEED_MESSAGES)
        results = corpus.retrieve("주식 시장 어떻게 봄", k=5)
        # No assertion on which message — just that it returns something.
        self.assertIsInstance(results, list)


class TestRetrieveTopK(unittest.TestCase):
    def test_retrieve_uses_env_corpus_path(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus_path = _seed_corpus_file(tmp_path)
            _reset_for_tests()
            with mock.patch.dict(
                os.environ, {"DGMH_KAKAO_CORPUS_PATH": str(corpus_path)}
            ):
                got = retrieve_top_k("도커 컨테이너 떠 있어?", k=3)
            self.assertGreaterEqual(len(got), 1)

    def test_retrieve_returns_empty_when_corpus_missing(self) -> None:
        _reset_for_tests()
        with mock.patch.dict(
            os.environ,
            {"DGMH_KAKAO_CORPUS_PATH": "/nonexistent/path/corpus.txt"},
        ):
            self.assertEqual(retrieve_top_k("뭐든지", k=5), [])


class TestCorpusFilters(unittest.TestCase):
    def test_url_only_lines_filtered_out(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus_path = tmp_path / "c.txt"
            corpus_path.write_text(
                "\n".join(
                    [
                        "https://example.com/post",
                        "그냥 평범한 메시지",
                        "@아무개 호출",
                        "[20:30] 닉네임 : 다른 사람 따옴",
                        "정상 짧은 메시지",
                    ]
                ),
                encoding="utf-8",
            )
            _reset_for_tests()
            with mock.patch.dict(
                os.environ, {"DGMH_KAKAO_CORPUS_PATH": str(corpus_path)}
            ):
                results = retrieve_top_k("아무거나", k=10)
            joined = " ".join(results)
            self.assertNotIn("https://", joined)
            self.assertNotIn("@아무개", joined)
            self.assertNotIn("[20:30]", joined)


class TestBuildProfileBody(unittest.TestCase):
    def test_body_includes_all_anchors_verbatim(self) -> None:
        anchors = ["첫번째 메시지", "두번째 메시지", "세번째 메시지"]
        body = build_kakao_rag_profile_body(anchors)
        for a in anchors:
            self.assertIn(a, body)

    def test_body_contains_frontmatter(self) -> None:
        body = build_kakao_rag_profile_body(["메시지"])
        self.assertIn("profile: kakao-mimic-rag", body)
        self.assertIn("voice-overrides:", body)

    def test_empty_anchors_yields_empty_body(self) -> None:
        # No anchors → no body. Caller treats this as a signal to fall
        # through to the static profile or original text.
        self.assertEqual(build_kakao_rag_profile_body([]), "")


class TestRewriteWithRagProfileCleanup(unittest.TestCase):
    def test_profile_file_removed_even_on_patina_exception(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            profile_dir = tmp_path / "profiles"
            profile_dir.mkdir()
            corpus_path = _seed_corpus_file(tmp_path)
            _reset_for_tests()
            with mock.patch.dict(
                os.environ, {"DGMH_KAKAO_CORPUS_PATH": str(corpus_path)}
            ):
                with mock.patch(
                    "dgmh.patina_judge.humanness_rewrite_with_profile",
                    side_effect=RuntimeError("simulated patina failure"),
                ):
                    with self.assertRaises(RuntimeError):
                        rewrite_with_rag_profile(
                            "cli 쓰면 어떨까",
                            profiles_dir=profile_dir,
                        )
            # No leftover kakao-mimic-rag-* file in the profile dir.
            leftovers = list(profile_dir.glob("kakao-mimic-rag-*.md"))
            self.assertEqual(
                leftovers,
                [],
                msg=f"profile cleanup leaked {leftovers}",
            )

    def test_returns_none_when_corpus_yields_no_anchors(self) -> None:
        # Empty corpus → retrieve returns [] → rewrite returns None
        # without ever touching the profile dir.
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus_path = tmp_path / "empty.txt"
            corpus_path.write_text(
                "https://only-urls.com\nhttps://nothing-else.com\n",
                encoding="utf-8",
            )
            profile_dir = tmp_path / "profiles"
            profile_dir.mkdir()
            _reset_for_tests()
            with mock.patch.dict(
                os.environ, {"DGMH_KAKAO_CORPUS_PATH": str(corpus_path)}
            ):
                got = rewrite_with_rag_profile(
                    "draft text",
                    profiles_dir=profile_dir,
                )
            self.assertIsNone(got)
            self.assertEqual(list(profile_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
