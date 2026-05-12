"""
dgmh/glm_data/pii_redact.py — PII redaction for KakaoTalk corpus (Plan v2.1, Step 2).

Single entrypoint:

    redact_kakao_text(raw, name_map, trace=None) -> (redacted, trace)

Layers (applied in order, most-specific first so we don't double-match):

  1. Kakao preamble — `<group_name> 님과 카카오톡 대화` line; group_name redacted in-line
     and recorded as sha256(group_name)[:12] in trace.group_name_sha256.
  2. Email
  3. Phone (010, +82, dashless variants)
  4. RRN-adjacent / birth date (YYYY-MM-DD, YY.MM.DD, YYYY년 M월 D일)
  5. Bank-account-like (3-6 / 2-6 / 2-8 digit groups)
  6. Address (도/시/구/동/로/길/번지)
  7. Multi-segment bracketed handles (`[행복죤23/CG(3D)/석전연]` -> `[user_NN]`).
     The entire bracket payload is the key — NO slash split. Stable across calls via
     the caller-supplied `name_map`.
  8. Korean name heuristic (3-char surname-prefix words + 2/4-char honorific patterns).

Per-category counts and the group_name digest are recorded in `RedactionTrace`.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RedactionTrace:
    """Audit record produced by `redact_kakao_text`.

    counts: per-category replacement counts (only categories that fired appear).
    group_name_sha256: first 12 hex chars of sha256(group_name) when a kakao
        preamble line was redacted; None otherwise.
    """

    counts: dict[str, int] = field(default_factory=dict)
    group_name_sha256: Optional[str] = None

    def bump(self, category: str, n: int = 1) -> None:
        if n <= 0:
            return
        self.counts[category] = self.counts.get(category, 0) + n


# --------------------------------------------------------------------------- #
# Regex layer
# --------------------------------------------------------------------------- #

# Kakao preamble: "<group_name> 님과 카카오톡 대화"
_KAKAO_PREAMBLE_RE = re.compile(r"(.+?)\s*님과 카카오톡 대화")

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# Korean mobile: 010-XXXX-XXXX (dashed/dashless), +82-10-XXXX-XXXX.
# Strict: must START with 01[016789] (after optional +82 / country trunk).
# Non-digit lookarounds prevent eating account-shaped strings like 110-234-567890.
_PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:"
    r"(?:\+?82[\s\-]?)?0?1[016789][\s\-]\d{3,4}[\s\-]\d{4}"  # dashed
    r"|"
    r"(?:\+?82[\s\-]?)?0?1[016789]\d{7,8}"  # dashless (10–11 digits)
    r")"
    r"(?!\d)"
)

# Birth / DOB:
#  - YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD
#  - YY.MM.DD (must be exactly 2 digit yr to avoid colliding with above)
#  - YYYY년 M월 D일
_BIRTH_RE = re.compile(
    r"(?:\b\d{4}[\-./]\d{1,2}[\-./]\d{1,2}\b|"
    r"\b\d{2}\.\d{1,2}\.\d{1,2}\b|"
    r"\b\d{4}년\s?\d{1,2}월\s?\d{1,2}일)"
)

# Bank-account-ish: 3 digit groups joined by dashes (3-6 / 2-6 / 2-8).
# Anchored to avoid eating phone numbers (which match the phone pattern first).
_BANK_RE = re.compile(r"(?<!\d)\d{3,6}-\d{2,6}-\d{2,8}(?!\d)")

# Address: Korean administrative tokens.
# Conservative: requires a 시/도-style head + one-or-more 시/군/구 tails,
# optional 동/읍/면/로/길 and trailing number.
_ADDR_RE = re.compile(
    r"[가-힣]{2,8}(?:특별시|광역시|특별자치시|특별자치도|도)"
    r"(?:\s+[가-힣A-Za-z0-9]{1,12}(?:시|군|구))+"
    r"(?:\s+[가-힣A-Za-z0-9]{1,12}(?:동|읍|면|로|길))?"
    r"(?:\s+\d+(?:번지|-\d+)?)?"
)

# Bracketed user handle. We treat the WHOLE bracket payload as the key, no slash split.
# Kakao header line uses `[name] [오전/오후 H:MM] msg`, so we exclude bracket payloads
# whose content is a timestamp-shaped token (`오전 9:20`, `2026-…`) to avoid corrupting
# the time field.
_HANDLE_RE = re.compile(r"\[([^\[\]\n]+)\]")
_TIMESTAMP_PAYLOAD_RE = re.compile(
    r"^(?:오전|오후)\s*\d{1,2}:\d{2}$|^\d{4}[\-./]\d{1,2}[\-./]\d{1,2}.*$"
)

# Korean surname prefixes (single-syllable, common + a handful of 2-syllable surnames
# folded in below). Used by the inline-name heuristic.
_KR_SURNAMES_1 = set(
    "김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민진지엄채원천방공현함변염여추도소석선설마길연위표명기반라왕금옥육인맹제모"
)
_KR_SURNAMES_2 = {"남궁", "황보", "사공", "선우", "서문", "독고", "제갈", "동방", "어금"}

# Names appear with these honorifics in our corpus.
_HONORIFICS = ("님", "씨", "선생님", "쌤", "환자분", "환자님", "원장님", "교수님")

# Common 3-char Hangul words that start with a surname syllable but are NOT names.
# Keeping this list short — operator review is the safety net.
_NAME_STOPWORDS: set[str] = {
    # 이-prefix
    "이번에", "이렇게", "이거는", "이런건", "이러면", "이상해", "이상한", "이정도", "이게뭐",
    "이대목동",  # hospital name fragment
    # 박-prefix
    "박살나", "박수를",
    # 강-prefix
    "강의실", "강의는", "강조한",
    # 조-prefix
    "조심해", "조용히",
    # 정-prefix
    "정말로", "정확히", "정상적", "정도로", "정리해",
    # 최-prefix
    "최근에", "최고로", "최대한",
    # 김-prefix
    "김밥은", "김치는",
    # 한-prefix
    "한번더", "한참을", "한국어", "한번만", "한국의", "한국인",
    # 안-prefix
    "안녕히", "안되는", "안되요", "안된거",
    # 오-prefix
    "오늘은", "오전에", "오후에", "오랜만", "오히려",
    # 서-prefix
    "서로가", "서울에", "서울의",
    # 손-prefix
    "손이가", "손쉽게",
    # 남-prefix
    "남자가", "남기는", "남기고",
    # 노-prefix
    "노력은", "노력해", "노트북",
    # 도-prefix
    "도와줘", "도와요",
    # generic
    "그러나", "그러면", "그렇게",
}


def _is_kr_surname_prefix(token: str) -> bool:
    if len(token) >= 2 and token[:2] in _KR_SURNAMES_2:
        return True
    if len(token) >= 1 and token[0] in _KR_SURNAMES_1:
        return True
    return False


# Inline Korean-name candidate:
#   - 3 Hangul chars whose first syllable is in the surname set
#   - bounded by non-Hangul (so we don't eat 4+ char words mid-token)
_KR_NAME_3_RE = re.compile(r"(?<![가-힣])[가-힣]{3}(?![가-힣])")
_KR_NAME_2_RE = re.compile(r"(?<![가-힣])[가-힣]{2}(?![가-힣])")
_KR_NAME_4_RE = re.compile(r"(?<![가-힣])[가-힣]{4}(?![가-힣])")

# Korean particles that may attach directly to a name token.
_KR_PARTICLES = (
    "이",
    "가",
    "은",
    "는",
    "을",
    "를",
    "의",
    "에서",
    "에게",
    "께",
    "에",
    "도",
    "만",
    "와",
    "과",
    "한테",
    "으로",
    "로서",
    "로써",
    "로",
    "이나",
    "나",
    "하고",
)
_KR_NAME_3_PARTICLE_RE = re.compile(
    r"(?<![가-힣])([가-힣]{3})(" + "|".join(_KR_PARTICLES) + r")(?![가-힣])"
)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

REDACTED_TOKENS = {
    "email": "[email]",
    "phone": "[phone]",
    "birth": "[date]",
    "bank": "[account]",
    "address": "[address]",
    "kr_name": "[name]",
    "kakao_group": "[group_name]",
}


def redact_kakao_text(
    raw: str,
    name_map: dict[str, str],
    trace: Optional[RedactionTrace] = None,
) -> tuple[str, RedactionTrace]:
    """Apply regex + name-map redaction to a kakao chat string.

    Parameters
    ----------
    raw: str
        Input text (one or more kakao lines).
    name_map: dict[str, str]
        Mutable bracket-payload -> `[user_NN]` mapping. Carried across calls so the
        same handle maps to the same redacted token within a session.
        On entry, may be empty or contain prior assignments from previous calls.
        On exit, augmented with any newly-seen handles.
    trace: RedactionTrace, optional
        Reuse an existing trace to accumulate counts across calls. A new one is
        allocated when omitted.

    Returns
    -------
    tuple[str, RedactionTrace]
        Redacted text and the (possibly newly allocated) trace.
    """
    if trace is None:
        trace = RedactionTrace()

    text = raw

    # 1. Kakao preamble (group name) — apply once per match; record digest.
    def _preamble_sub(m: re.Match[str]) -> str:
        group_name = m.group(1).strip()
        if group_name:
            digest = hashlib.sha256(group_name.encode("utf-8")).hexdigest()[:12]
            # First match wins for the trace field; counts continues to bump.
            if trace.group_name_sha256 is None:
                trace.group_name_sha256 = digest
            trace.bump("kakao_group")
        return f"{REDACTED_TOKENS['kakao_group']} 님과 카카오톡 대화"

    text = _KAKAO_PREAMBLE_RE.sub(_preamble_sub, text)

    # 2. Email
    text, n = _EMAIL_RE.subn(REDACTED_TOKENS["email"], text)
    trace.bump("email", n)

    # 3. Phone
    text, n = _PHONE_RE.subn(REDACTED_TOKENS["phone"], text)
    trace.bump("phone", n)

    # 4. Birth / DOB
    text, n = _BIRTH_RE.subn(REDACTED_TOKENS["birth"], text)
    trace.bump("birth", n)

    # 5. Bank-like
    text, n = _BANK_RE.subn(REDACTED_TOKENS["bank"], text)
    trace.bump("bank", n)

    # 6. Address
    text, n = _ADDR_RE.subn(REDACTED_TOKENS["address"], text)
    trace.bump("address", n)

    # 7. Bracketed handles. Map each unique payload to a stable user_NN.
    def _handle_sub(m: re.Match[str]) -> str:
        payload = m.group(1)
        # Skip kakao timestamp brackets like [오전 9:20] or [2026-…].
        if _TIMESTAMP_PAYLOAD_RE.match(payload.strip()):
            return m.group(0)
        # Skip already-redacted tokens like [user_03], [name], [email] etc.
        if payload.startswith("user_") or f"[{payload}]" in REDACTED_TOKENS.values():
            return m.group(0)
        # Require at least one Hangul or letter — pure-digit brackets are not handles.
        if not re.search(r"[A-Za-z가-힣]", payload):
            return m.group(0)
        token = name_map.get(payload)
        if token is None:
            # Next free user_NN. Count existing user_* tokens in the map.
            existing = sum(1 for v in name_map.values() if v.startswith("[user_"))
            token = f"[user_{existing + 1:02d}]"
            name_map[payload] = token
        trace.bump("user_handle")
        return token

    text = _HANDLE_RE.sub(_handle_sub, text)

    # 8. Inline Korean-name heuristic.
    # Order: 4-char 2-syllable surname, 3-char + particle, standalone 3-char, 2-char + honorific.
    text, n4 = _kr_name_4_pass(text)
    trace.bump("kr_name", n4)
    text, npart = _kr_name_3_particle_pass(text)
    trace.bump("kr_name", npart)
    text, n3 = _kr_name_3_pass(text)
    trace.bump("kr_name", n3)
    text, n2 = _kr_name_2_pass(text)
    trace.bump("kr_name", n2)

    return text, trace


# --------------------------------------------------------------------------- #
# Korean-name passes
# --------------------------------------------------------------------------- #


def _kr_name_3_pass(text: str) -> tuple[str, int]:
    """Replace standalone 3-char Korean tokens whose first syllable is a surname."""

    count = 0

    def _sub(m: re.Match[str]) -> str:
        nonlocal count
        token = m.group(0)
        if token in _NAME_STOPWORDS:
            return token
        if not _is_kr_surname_prefix(token):
            return token
        count += 1
        return REDACTED_TOKENS["kr_name"]

    return _KR_NAME_3_RE.sub(_sub, text), count


def _kr_name_3_particle_pass(text: str) -> tuple[str, int]:
    """Replace 3-char Korean tokens followed by an inflectional particle.

    Catches `박민수가`, `이영희는`, `김의사를`, etc.
    """

    count = 0

    def _sub(m: re.Match[str]) -> str:
        nonlocal count
        token = m.group(1)
        particle = m.group(2)
        if token in _NAME_STOPWORDS:
            return m.group(0)
        if not _is_kr_surname_prefix(token):
            return m.group(0)
        count += 1
        return REDACTED_TOKENS["kr_name"] + particle

    return _KR_NAME_3_PARTICLE_RE.sub(_sub, text), count


def _kr_name_4_pass(text: str) -> tuple[str, int]:
    """Replace 4-char tokens only for 2-syllable surnames (남궁/황보/...)."""

    count = 0

    def _sub(m: re.Match[str]) -> str:
        nonlocal count
        token = m.group(0)
        if token[:2] not in _KR_SURNAMES_2:
            return token
        count += 1
        return REDACTED_TOKENS["kr_name"]

    return _KR_NAME_4_RE.sub(_sub, text), count


def _kr_name_2_pass(text: str) -> tuple[str, int]:
    """Replace 2-char Korean tokens ONLY when immediately followed by an honorific."""

    count = 0
    # Build a single regex: 2 Hangul chars + optional space + honorific.
    honor_alt = "|".join(re.escape(h) for h in _HONORIFICS)
    pattern = re.compile(
        rf"(?<![가-힣])([가-힣]{{2}})(\s*(?:{honor_alt}))(?![가-힣])"
    )

    def _sub(m: re.Match[str]) -> str:
        nonlocal count
        token = m.group(1)
        if not _is_kr_surname_prefix(token):
            return m.group(0)
        if token in _NAME_STOPWORDS:
            return m.group(0)
        count += 1
        return REDACTED_TOKENS["kr_name"] + m.group(2)

    return pattern.sub(_sub, text), count


__all__ = [
    "RedactionTrace",
    "REDACTED_TOKENS",
    "redact_kakao_text",
]
