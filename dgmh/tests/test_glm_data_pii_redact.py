"""
dgmh/tests/test_glm_data_pii_redact.py — Step 2 adversarial fixture for pii_redact.

Per plan v2.1 §3 AC1/AC1b and §4 Step 2: ≥40 cases covering 6 regex categories
(phone / email / address / birth / bank / kr_name), kakao preamble, multi-segment
bracketed handle stability, name-map carryover across calls, and ≥10 medical
fixtures.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dgmh.glm_data.pii_redact import (  # noqa: E402
    REDACTED_TOKENS,
    RedactionTrace,
    redact_kakao_text,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _run(text: str, name_map: dict | None = None, trace: RedactionTrace | None = None):
    nm = {} if name_map is None else name_map
    out, tr = redact_kakao_text(text, nm, trace=trace)
    return out, tr, nm


# --------------------------------------------------------------------------- #
# 1–10. Medical-domain fixtures (AC1b)
# --------------------------------------------------------------------------- #


def test_med01_bp_and_age_pass_through_unredacted_content():
    """'환자 67세 남성 BP 180/110' — no PII, content survives."""
    out, tr, _ = _run("환자 67세 남성 BP 180/110")
    assert "67세" in out
    assert "BP 180/110" in out
    assert "kakao_group" not in tr.counts


def test_med02_drug_dosage_string_intact():
    out, tr, _ = _run("metformin 500mg bid")
    assert out == "metformin 500mg bid"
    assert tr.counts == {}


def test_med03_hospital_name_not_mis_redacted_as_person_name():
    """'이대목동 응급실 콜드' — 이대목동 is a hospital, not a person."""
    out, tr, _ = _run("이대목동 응급실 콜드")
    assert "이대목동" in out
    assert tr.counts.get("kr_name", 0) == 0


def test_med04_patient_full_name_with_honorific():
    out, tr, _ = _run("김철수 환자분이 입원하셨습니다")
    assert "김철수" not in out
    assert REDACTED_TOKENS["kr_name"] in out
    assert tr.counts.get("kr_name", 0) >= 1


def test_med05_doctor_phone_in_handoff_note():
    out, tr, _ = _run("당직의 010-1234-5678로 연락")
    assert "010-1234-5678" not in out
    assert REDACTED_TOKENS["phone"] in out
    assert tr.counts.get("phone", 0) == 1


def test_med06_lab_result_with_email_to_radiologist():
    out, tr, _ = _run("CT 판독 결과 rad@hospital.kr 로 송부")
    assert "rad@hospital.kr" not in out
    assert REDACTED_TOKENS["email"] in out


def test_med07_dob_in_chart_note():
    out, tr, _ = _run("DOB 1962년 3월 15일, 당뇨 25년")
    assert "1962년 3월 15일" not in out
    assert REDACTED_TOKENS["birth"] in out
    assert "당뇨 25년" in out


def test_med08_prescription_lines_unchanged():
    out, tr, _ = _run("Rx: amlodipine 5mg qd, atorvastatin 20mg qhs")
    assert out == "Rx: amlodipine 5mg qd, atorvastatin 20mg qhs"


def test_med09_korean_address_in_admit_note():
    out, tr, _ = _run("서울특별시 강남구 역삼동 123-45 거주")
    assert "강남구" not in out
    assert REDACTED_TOKENS["address"] in out
    assert tr.counts.get("address", 0) >= 1


def test_med10_medical_jargon_tokens_pass_through():
    """Common medical short forms shouldn't trip the name heuristic."""
    out, tr, _ = _run("EKG NSR, CBC WNL, 환자 stable 합니다")
    assert "EKG" in out
    assert "NSR" in out
    assert tr.counts.get("kr_name", 0) == 0


# --------------------------------------------------------------------------- #
# 11–15. Multi-segment bracketed handle stability (AC1, AC2)
# --------------------------------------------------------------------------- #


def test_handle01_multisegment_payload_is_single_key_no_slash_split():
    out, tr, nm = _run("[행복죤23/CG(3D)/석전연] hello")
    # entire bracketed payload mapped to a single user_NN
    assert "행복죤23" not in out
    assert "석전연" not in out
    assert re.search(r"\[user_\d{2}\] hello", out)
    assert "행복죤23/CG(3D)/석전연" in nm
    assert nm["행복죤23/CG(3D)/석전연"].startswith("[user_")


def test_handle02_simple_two_segment_handle():
    out, tr, nm = _run("[가나/다라] 안녕")
    assert "[가나/다라]" not in out
    assert re.search(r"\[user_\d{2}\] 안녕", out)
    assert "가나/다라" in nm


def test_handle03_alphanumeric_only_handle():
    out, tr, nm = _run("[ABC123] msg")
    assert "[ABC123]" not in out
    assert "ABC123" in nm
    assert tr.counts.get("user_handle", 0) == 1


def test_handle04_kakao_timestamp_brackets_NOT_treated_as_handle():
    out, tr, nm = _run("[령우] [오전 9:20] hi")
    # 령우 -> user_01
    assert "[령우]" not in out
    # 오전 9:20 must survive — it's a kakao timestamp, not a user handle
    assert "[오전 9:20]" in out
    # only one handle entry recorded
    assert "령우" in nm and len(nm) == 1


def test_handle05_pure_digit_brackets_are_NOT_handles():
    """e.g. an enumeration like '[1] 항목' shouldn't be treated as a user handle."""
    out, tr, nm = _run("[1] 항목 [2] 항목")
    assert "[1]" in out
    assert "[2]" in out
    assert tr.counts.get("user_handle", 0) == 0


# --------------------------------------------------------------------------- #
# 16–18. Kakao preamble + group-name redaction (AC2)
# --------------------------------------------------------------------------- #


def test_preamble01_group_name_redacted_and_sha_recorded():
    text = "의료인공지능 님과 카카오톡 대화"
    out, tr, _ = _run(text)
    assert "의료인공지능" not in out
    assert REDACTED_TOKENS["kakao_group"] in out
    assert tr.counts.get("kakao_group", 0) >= 1
    expected = hashlib.sha256("의료인공지능".encode("utf-8")).hexdigest()[:12]
    assert tr.group_name_sha256 == expected


def test_preamble02_first_match_wins_for_digest():
    """If two preamble lines somehow appear, only the first sha is recorded."""
    text = "A1 님과 카카오톡 대화\nB2 님과 카카오톡 대화"
    out, tr, _ = _run(text)
    assert tr.group_name_sha256 == hashlib.sha256("A1".encode("utf-8")).hexdigest()[:12]
    # both counted
    assert tr.counts.get("kakao_group", 0) == 2


def test_preamble03_no_preamble_means_sha_remains_none():
    out, tr, _ = _run("그냥 인사 메시지")
    assert tr.group_name_sha256 is None
    assert "kakao_group" not in tr.counts


# --------------------------------------------------------------------------- #
# 19–24. Per-category property tests — each regex always redacts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw",
    [
        "전화 010-1234-5678 연락",
        "010-9999-0000 입니다",
        "휴대 01098765432 가능",
        "+82-10-1111-2222 international",
        "01012345678",
    ],
)
def test_prop_phone_always_redacted(raw):
    out, tr, _ = _run(raw)
    assert tr.counts.get("phone", 0) >= 1
    # no 10+ digit substring remains
    assert not re.search(r"\d{10}", re.sub(r"\[\w+\]", "", out))


@pytest.mark.parametrize(
    "raw",
    [
        "mail to foo@bar.com please",
        "x.y+z@example.co.kr",
        "abc_123@sub.domain.io test",
        "연락처 user@test.org",
    ],
)
def test_prop_email_always_redacted(raw):
    out, tr, _ = _run(raw)
    assert "@" not in out or out.count("@") < raw.count("@")
    assert tr.counts.get("email", 0) >= 1


@pytest.mark.parametrize(
    "raw",
    [
        "생년월일 1990-05-12",
        "DOB 2001/12/31",
        "1985.07.20 출생",
        "85.07.20 출생",
        "1990년 5월 12일 생",
    ],
)
def test_prop_birth_always_redacted(raw):
    out, tr, _ = _run(raw)
    assert tr.counts.get("birth", 0) >= 1


@pytest.mark.parametrize(
    "raw",
    [
        "계좌 110-234-567890 입금",
        "계좌번호 1002-345-678901",
        "농협 123456-12-345678",
    ],
)
def test_prop_bank_always_redacted(raw):
    out, tr, _ = _run(raw)
    assert tr.counts.get("bank", 0) >= 1


@pytest.mark.parametrize(
    "raw",
    [
        "서울특별시 강남구 역삼동 거주",
        "경기도 성남시 분당구 정자동 178",
        "부산광역시 해운대구",
    ],
)
def test_prop_address_always_redacted(raw):
    out, tr, _ = _run(raw)
    assert tr.counts.get("address", 0) >= 1


@pytest.mark.parametrize(
    "raw",
    [
        "김철수 안녕",
        "이영희 환자분",
        "박민수가 왔다",
    ],
)
def test_prop_kr_name_always_redacted(raw):
    out, tr, _ = _run(raw)
    assert tr.counts.get("kr_name", 0) >= 1


# --------------------------------------------------------------------------- #
# 25–28. Name-map carryover across calls (AC1)
# --------------------------------------------------------------------------- #


def test_carryover01_same_handle_same_token_across_calls():
    nm: dict[str, str] = {}
    out1, _, _ = _run("[행복죤23/CG(3D)/석전연] msg1", name_map=nm)
    out2, _, _ = _run("[행복죤23/CG(3D)/석전연] msg2", name_map=nm)
    token1 = re.search(r"\[user_\d{2}\]", out1).group(0)
    token2 = re.search(r"\[user_\d{2}\]", out2).group(0)
    assert token1 == token2


def test_carryover02_distinct_handles_distinct_tokens():
    nm: dict[str, str] = {}
    out1, _, _ = _run("[가나] hi", name_map=nm)
    out2, _, _ = _run("[다라] hi", name_map=nm)
    t1 = re.search(r"\[user_\d{2}\]", out1).group(0)
    t2 = re.search(r"\[user_\d{2}\]", out2).group(0)
    assert t1 != t2
    assert len(nm) == 2


def test_carryover03_token_indices_are_sequential():
    nm: dict[str, str] = {}
    _run("[a1] [b2] [c3] msg", name_map=nm)
    tokens = sorted(nm.values())
    assert tokens == ["[user_01]", "[user_02]", "[user_03]"]


def test_carryover04_trace_can_be_reused_to_accumulate_counts():
    nm: dict[str, str] = {}
    tr = RedactionTrace()
    _run("010-1111-2222", name_map=nm, trace=tr)
    _run("foo@bar.com", name_map=nm, trace=tr)
    assert tr.counts.get("phone", 0) == 1
    assert tr.counts.get("email", 0) == 1


# --------------------------------------------------------------------------- #
# 29–40. Edge cases, mixed payloads, idempotence
# --------------------------------------------------------------------------- #


def test_edge01_idempotent_redaction_doesnt_re_redact_tokens():
    """Running redact twice should be a no-op after the first pass."""
    nm: dict[str, str] = {}
    once, _, _ = _run("[가나] 010-1234-5678 foo@bar.com", name_map=nm)
    twice, _ = redact_kakao_text(once, nm)
    assert once == twice


def test_edge02_full_kakao_line_combined():
    nm: dict[str, str] = {}
    raw = "[행복죤23/CG(3D)/석전연] [오전 9:20] 010-1111-2222 me@x.kr"
    out, tr, _ = _run(raw, name_map=nm)
    assert "행복죤23" not in out
    assert "010-1111-2222" not in out
    assert "me@x.kr" not in out
    assert "[오전 9:20]" in out
    assert tr.counts.get("phone", 0) == 1
    assert tr.counts.get("email", 0) == 1
    assert tr.counts.get("user_handle", 0) == 1


def test_edge03_existing_redacted_token_not_double_wrapped():
    out, tr, nm = _run("[email] is already redacted")
    assert out == "[email] is already redacted"
    assert tr.counts == {} or "user_handle" not in tr.counts


def test_edge04_existing_user_token_not_re_mapped():
    nm: dict[str, str] = {}
    out, tr, _ = _run("[user_05] previous", name_map=nm)
    assert "[user_05]" in out
    # name_map untouched
    assert nm == {}


def test_edge05_empty_string_returns_empty_trace():
    out, tr, _ = _run("")
    assert out == ""
    assert tr.counts == {}
    assert tr.group_name_sha256 is None


def test_edge06_multiline_mixed_redaction():
    text = (
        "의료인공지능 님과 카카오톡 대화\n"
        "[령우] [오전 9:20] 안녕하세요 010-1234-5678\n"
        "이메일 doc@hospital.kr 로 보내주세요\n"
    )
    out, tr, _ = _run(text)
    assert "의료인공지능" not in out
    assert "010-1234-5678" not in out
    assert "doc@hospital.kr" not in out
    assert "[령우]" not in out
    assert tr.group_name_sha256 is not None
    assert tr.counts.get("phone", 0) == 1
    assert tr.counts.get("email", 0) == 1
    assert tr.counts.get("user_handle", 0) == 1
    assert tr.counts.get("kakao_group", 0) == 1


def test_edge07_phone_not_eaten_by_bank_rule():
    out, tr, _ = _run("010-1234-5678")
    assert tr.counts.get("phone", 0) == 1
    assert tr.counts.get("bank", 0) == 0


def test_edge08_stopword_not_mis_redacted_as_name():
    """Common 3-char Hangul stopwords shouldn't be redacted as names."""
    out, tr, _ = _run("이번에 정말로 안녕히 가세요")
    assert tr.counts.get("kr_name", 0) == 0
    assert "이번에" in out
    assert "정말로" in out


def test_edge09_two_syllable_surname_redacted():
    out, tr, _ = _run("남궁민수 환자")
    assert "남궁민수" not in out
    assert REDACTED_TOKENS["kr_name"] in out


def test_edge10_address_with_road_number():
    out, tr, _ = _run("경기도 성남시 분당구 판교로 123")
    assert "분당구" not in out
    assert tr.counts.get("address", 0) >= 1


def test_edge11_rrn_like_birth_pattern():
    out, tr, _ = _run("생년 90.05.12 확인")
    assert "90.05.12" not in out
    assert tr.counts.get("birth", 0) >= 1


def test_edge12_handle_followed_by_real_message_with_pii():
    nm: dict[str, str] = {}
    out, tr, _ = _run("[김의사] 환자 전화 010-5555-7777", name_map=nm)
    assert "[김의사]" not in out
    assert "010-5555-7777" not in out
    # both fired
    assert tr.counts.get("user_handle", 0) == 1
    assert tr.counts.get("phone", 0) == 1
