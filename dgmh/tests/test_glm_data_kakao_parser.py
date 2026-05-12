"""
dgmh/tests/test_glm_data_kakao_parser.py — Step 3 tests for kakao_parser + medical_groups.

AC2  : Kakao header parsing + preamble redaction (group_name sha256).
AC1b : Medical-domain opt-in (preamble keyword detector).
H5   : 100% line accountability on the smallest corpus file (058_45_466, 4100 lines).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dgmh.glm_data.kakao_parser import (  # noqa: E402
    KakaoParser,
    ParseStats,
    parse_kakao_file,
)
from dgmh.glm_data.medical_groups import (  # noqa: E402
    MEDICAL_KEYWORDS,
    find_medical_files,
    is_medical_group,
)
from dgmh.glm_data.pii_redact import RedactionTrace  # noqa: E402
from dgmh.glm_data.schema import KakaoMsg  # noqa: E402

CORPUS_DIR = Path("/home/devswha/workspace/dgmh")
SMALLEST = CORPUS_DIR / "KakaoTalk_20260512_0058_45_466_group.txt"
MEDICAL_2 = CORPUS_DIR / "KakaoTalk_20260512_0059_34_202_group.txt"
NONMED_FILES = [
    CORPUS_DIR / "KakaoTalk_20260504_2302_52_630_group.txt",  # 어둠의 구인&칭찬방
    CORPUS_DIR / "KakaoTalk_20260512_0058_22_807_group.txt",  # 에르메스단
    CORPUS_DIR / "KakaoTalk_20260512_0059_24_687_group.txt",  # 로아 액기스
    CORPUS_DIR / "KakaoTalk_20260512_0059_44_779_group.txt",  # 그들만의리그
    CORPUS_DIR / "KakaoTalk_20260512_0059_59_086_group.txt",  # 종원, 주씨
]


# --------------------------------------------------------------------------- #
# medical_groups detector
# --------------------------------------------------------------------------- #


def test_medical_detector_flags_058_45_466():
    assert is_medical_group(SMALLEST) is True


def test_medical_detector_flags_059_34_202():
    assert is_medical_group(MEDICAL_2) is True


@pytest.mark.parametrize("path", NONMED_FILES)
def test_medical_detector_does_not_flag_nonmedical(path):
    assert is_medical_group(path) is False


def test_find_medical_files_returns_exactly_the_two_medical_groups():
    found = find_medical_files(CORPUS_DIR)
    assert SMALLEST in found
    assert MEDICAL_2 in found
    for p in NONMED_FILES:
        assert p not in found
    assert len(found) == 2


def test_medical_keywords_include_required_set():
    required = {"의료", "의약", "병원", "약사", "의대", "한의원"}
    assert required.issubset(set(MEDICAL_KEYWORDS))


# Codex review MAJOR-2: every plan-required keyword must be present and must
# actually flag a synthetic group title that contains it.
_NEW_REQUIRED_KEYWORDS = (
    "환자",
    "진료",
    "의사",
    "간호",
    "처방",
    "약물",
    "응급실",
    "클리닉",
    "헬스케어",
    "임상",
)


def test_medical_keywords_include_codex_required_expansion():
    """MAJOR-2: plan v2.1 AC1b keywords must all be in MEDICAL_KEYWORDS."""
    assert set(_NEW_REQUIRED_KEYWORDS).issubset(set(MEDICAL_KEYWORDS))


def _make_kakao_with_title(tmp_path: Path, title: str) -> Path:
    """Synthesize a minimal kakao file with the given group title."""
    p = tmp_path / "synthetic_group.txt"
    p.write_text(
        f"{title} 님과 카카오톡 대화\n"
        "저장한 날짜 : 2026-05-12 12:00:00\n"
        "\n",
        encoding="utf-8",
    )
    return p


@pytest.mark.parametrize("keyword", _NEW_REQUIRED_KEYWORDS)
def test_medical_detector_flags_each_new_keyword(tmp_path, keyword):
    """Synthetic title containing each new keyword must be flagged."""
    title = f"{keyword} 케이스 공유"
    path = _make_kakao_with_title(tmp_path, title)
    assert is_medical_group(path) is True, f"keyword {keyword!r} not detected in title"


def test_medical_detector_does_not_flag_generic_family_group(tmp_path):
    """Negative: a non-medical title must NOT be flagged.

    Guards against an over-broad keyword set that would false-positive on
    everyday Korean group names.
    """
    path = _make_kakao_with_title(tmp_path, "일반 가족톡")
    assert is_medical_group(path) is False


def test_is_medical_group_returns_false_for_missing_path(tmp_path):
    assert is_medical_group(tmp_path / "does_not_exist.txt") is False


def test_find_medical_files_returns_empty_for_missing_dir(tmp_path):
    assert find_medical_files(tmp_path / "no_such_dir") == set()


# --------------------------------------------------------------------------- #
# Parser — 100% line accountability over the smallest corpus file
# --------------------------------------------------------------------------- #


def test_full_line_accountability_058_45_466():
    """Sum of category counters must equal total_lines (≈4100; Python's
    line iterator may include a trailing partial line, so we anchor on
    invariant equality, not on the wc count)."""
    parser = KakaoParser(SMALLEST, name_map={})
    msgs = list(parser)
    stats = parser.stats

    # H5: 100% line accountability — every line lands in exactly one bucket.
    assert stats.classified() == stats.total_lines
    # wc -l reports 4100 newlines; Python iterates 4100 or 4101.
    assert stats.total_lines in (4100, 4101)
    # preamble is 3 lines (group_name + saved_date + blank).
    assert stats.preamble_lines == 3
    # Sanity: many separators and many message headers (file spans months).
    assert stats.separator_lines >= 10
    assert stats.message_header_lines >= 100
    # Yielded message count equals header count.
    assert len(msgs) == stats.message_header_lines


def test_group_name_sha256_recorded_in_trace_058_45_466():
    trace = RedactionTrace()
    parser = KakaoParser(SMALLEST, name_map={}, trace=trace)
    _ = list(parser)
    expected = hashlib.sha256("의료인공지능".encode("utf-8")).hexdigest()[:12]
    assert parser.trace.group_name_sha256 == expected


# --------------------------------------------------------------------------- #
# Parser — message structure
# --------------------------------------------------------------------------- #


def test_parser_yields_kakaomsg_with_redacted_user_handle():
    parser = KakaoParser(SMALLEST, name_map={})
    first = next(iter(parser))
    # speaker_id is the bare redacted handle (no brackets) — "user_NN".
    assert first.speaker_id.startswith("user_")
    assert "[" not in first.speaker_id and "]" not in first.speaker_id
    # msg_id is sequential "msg_NNNNNN".
    assert first.msg_id == "msg_000000"
    # timestamp is ISO 8601 (date_context set on line 3 before first msg).
    assert "T" in first.timestamp
    assert first.timestamp.startswith("2026-")
    # raw_text and redacted_text are distinct fields; both populated.
    assert isinstance(first.raw_text, str)
    assert isinstance(first.redacted_text, str)


def test_parser_name_map_carries_handle_across_messages():
    name_map: dict[str, str] = {}
    parser = KakaoParser(SMALLEST, name_map=name_map)
    _ = list(parser)
    # At least one mapping recorded.
    assert len(name_map) >= 1
    # Every mapping value is a user_NN token.
    for v in name_map.values():
        assert v.startswith("[user_") and v.endswith("]")


def test_parser_separator_sets_date_context_on_subsequent_message():
    parser = KakaoParser(SMALLEST, name_map={})
    msgs = list(parser)
    # The first message's ISO 8601 timestamp must reflect the separator on line 4
    # (`--------------- 2026년 ... ---------------`).
    assert msgs[0].timestamp.startswith("2026-")
    assert "T" in msgs[0].timestamp


def test_parse_kakao_file_is_an_iterator():
    """The convenience function exposed by the module returns an iterator."""
    it = parse_kakao_file(SMALLEST, name_map={})
    first = next(it)
    assert isinstance(first, KakaoMsg)


def test_parser_emits_canonical_schema_kakaomsg_fields():
    """C3 regression: parser output must be `schema.KakaoMsg` so the classifier
    (which reads `.msg_id` and `.redacted_text`) can consume it directly."""
    parser = KakaoParser(SMALLEST, name_map={})
    msgs = list(parser)
    assert len(msgs) >= 2
    for i, m in enumerate(msgs):
        # Canonical schema fields populated.
        assert m.msg_id == f"msg_{i:06d}"
        assert m.speaker_id and "[" not in m.speaker_id
        assert isinstance(m.timestamp, str) and "T" in m.timestamp
        assert isinstance(m.raw_text, str)
        assert isinstance(m.redacted_text, str)
        # No legacy fields linger on the dataclass.
        assert not hasattr(m, "user")
        assert not hasattr(m, "period")
        assert not hasattr(m, "hour")
        assert not hasattr(m, "minute")
        assert not hasattr(m, "text")
        assert not hasattr(m, "date_context")


# --------------------------------------------------------------------------- #
# Parser — synthetic fixture for fine-grained structural assertions
# --------------------------------------------------------------------------- #


@pytest.fixture
def synthetic_kakao(tmp_path: Path) -> Path:
    """Minimal synthetic kakao file exercising every line category."""
    p = tmp_path / "synth.txt"
    p.write_text(
        "테스트방 님과 카카오톡 대화\n"
        "저장한 날짜 : 2026-05-12 12:00:00\n"
        "\n"
        "--------------- 2026년 5월 1일 목요일 ---------------\n"
        "[알파] [오전 9:20] 안녕하세요\n"
        "이건 같은 메시지의 둘째 줄\n"
        "\n"
        "[베타] [오후 1:05] 답장\n"
        "감마님이 들어왔습니다.\n"
        "--------------- 2026년 5월 2일 금요일 ---------------\n"
        "[감마] [오전 10:00] 새 날 새 인사\n",
        encoding="utf-8",
    )
    return p


def test_synthetic_accountability(synthetic_kakao: Path):
    parser = KakaoParser(synthetic_kakao, name_map={})
    msgs = list(parser)
    s = parser.stats

    assert s.total_lines == 11
    assert s.classified() == 11
    assert s.preamble_lines == 3
    assert s.separator_lines == 2
    assert s.message_header_lines == 3
    assert s.event_lines == 1
    # 1 continuation line for alpha's second body line; the blank between
    # alpha and beta is also charged to continuation because pending was open.
    assert s.continuation_lines == 2
    assert s.blank_lines == 0
    assert s.other_lines == 0
    assert len(msgs) == 3


def test_synthetic_continuation_appends_to_prior_message(synthetic_kakao: Path):
    parser = KakaoParser(synthetic_kakao, name_map={})
    msgs = list(parser)
    alpha = msgs[0]
    assert "이건 같은 메시지의 둘째 줄" in alpha.redacted_text
    assert "이건 같은 메시지의 둘째 줄" in alpha.raw_text


def test_synthetic_event_flushes_pending(synthetic_kakao: Path):
    """An event line must close the currently-open message."""
    parser = KakaoParser(synthetic_kakao, name_map={})
    msgs = list(parser)
    beta = msgs[1]
    # Beta's text should NOT contain the event line.
    assert "들어왔습니다" not in beta.redacted_text


def test_synthetic_distinct_handles_get_distinct_user_tokens(synthetic_kakao: Path):
    """알파 / 베타 / 감마 → three distinct ``user_NN`` speaker_ids."""
    name_map: dict[str, str] = {}
    parser = KakaoParser(synthetic_kakao, name_map=name_map)
    msgs = list(parser)
    assert len({m.speaker_id for m in msgs}) == 3
    for m in msgs:
        assert m.speaker_id.startswith("user_")


# --------------------------------------------------------------------------- #
# Parser — MAJOR-3 robust preamble detection
# --------------------------------------------------------------------------- #


def test_parser_preserves_first_message_when_preamble_missing(
    tmp_path: Path, capsys
):
    """MAJOR-3: a file without the recognized preamble shape must keep
    line 0 intact instead of silently dropping it.

    Before the fix the parser dropped the first 3 lines unconditionally, so a
    valid header line at index 0 would be lost on English exports or files
    whose preamble had already been stripped.
    """
    p = tmp_path / "no_preamble.txt"
    p.write_text(
        "--------------- 2026년 5월 1일 목요일 ---------------\n"
        "[알파] [오전 9:20] 첫 메시지\n"
        "[베타] [오후 1:05] 둘째 메시지\n",
        encoding="utf-8",
    )

    parser = KakaoParser(p, name_map={})
    msgs = list(parser)
    captured = capsys.readouterr()

    # Both message-header lines must survive (preamble drop must not eat
    # the separator + first message).
    assert len(msgs) == 2
    assert "첫 메시지" in msgs[0].redacted_text
    assert "둘째 메시지" in msgs[1].redacted_text

    # No preamble lines were dropped.
    assert parser.stats.preamble_lines == 0
    # Operator-visible warning was emitted on stderr.
    assert "non-standard preamble" in captured.err


def test_parser_keeps_standard_preamble_behavior(tmp_path: Path, capsys):
    """Regression guard: when the preamble shape *is* recognized, we still
    drop exactly 3 lines and emit no warning."""
    p = tmp_path / "ok_preamble.txt"
    p.write_text(
        "테스트방 님과 카카오톡 대화\n"
        "저장한 날짜 : 2026-05-12 12:00:00\n"
        "\n"
        "--------------- 2026년 5월 1일 목요일 ---------------\n"
        "[알파] [오전 9:20] 안녕\n",
        encoding="utf-8",
    )
    parser = KakaoParser(p, name_map={})
    msgs = list(parser)
    captured = capsys.readouterr()

    assert len(msgs) == 1
    assert parser.stats.preamble_lines == 3
    assert "non-standard preamble" not in captured.err


def test_parser_recognizes_english_preamble_shape(tmp_path: Path, capsys):
    """English KakaoTalk exports use a different header but the same
    3-line shape. MAJOR-3: detect and drop them too."""
    p = tmp_path / "en_preamble.txt"
    p.write_text(
        "Alice saved KakaoTalk Chats with Bob\n"
        "Date Saved : 2026-05-12 12:00:00\n"
        "\n"
        "--------------- 2026년 5월 1일 목요일 ---------------\n"
        "[알파] [오전 9:20] 안녕\n",
        encoding="utf-8",
    )
    parser = KakaoParser(p, name_map={})
    msgs = list(parser)
    captured = capsys.readouterr()

    assert len(msgs) == 1
    assert parser.stats.preamble_lines == 3
    assert "non-standard preamble" not in captured.err


def test_parser_warns_on_partial_preamble(tmp_path: Path, capsys):
    """If only one of the three shape signals is present, do NOT drop —
    we'd rather over-parse and warn than silently lose data."""
    p = tmp_path / "partial_preamble.txt"
    # Has the KR header but lacks the saved-date and blank.
    p.write_text(
        "테스트방 님과 카카오톡 대화\n"
        "[알파] [오전 9:20] 첫 메시지\n"
        "[베타] [오후 1:05] 둘째 메시지\n",
        encoding="utf-8",
    )
    parser = KakaoParser(p, name_map={})
    msgs = list(parser)
    captured = capsys.readouterr()

    assert parser.stats.preamble_lines == 0
    assert "non-standard preamble" in captured.err
    # The two messages are preserved.
    assert len(msgs) == 2
