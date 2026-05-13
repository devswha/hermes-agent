"""DGM-H humanness hook — wires Discord outbound messages to async patina scoring.

Registered at ``gateway:startup``. Wraps the DiscordAdapter's ``send`` method
to fire fire-and-forget patina scoring on every assistant message. Results are
appended to ``~/.hermes/dgmh/humanness_log.jsonl`` for use as a post-hoc
fitness signal in the DGM-H evolution composite reward.

Skip rules (don't score):
  - Empty content
  - Length < ``DGMH_HUMANNESS_MIN_CHARS`` (default 30) — too short to score reliably
  - Begins with "Error:" or runtime warning glyphs — runtime error frames, not assistant prose
  - DGMH_HUMANNESS_DISABLED env set
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

from dgmh.humanness_log import (
    append_record,
    make_error_record,
    make_success_record,
)

logger = logging.getLogger(__name__)


_DEFAULT_MIN_CHARS = 30
_ERROR_PREFIXES = ("Error:", "⚠️", "❌", "[error]")
_PATCH_FLAG = "_dgmh_humanness_patched"

# Pruning: when an assistant reply scores at or above this AI-likeness
# threshold, the row in state.db.messages is deleted so it does not
# pollute the conversation history that Hermes feeds into the next
# system prompt build. This breaks the self-reinforcing loop where the
# bot mimics its own prior chatgpt-styled replies.
_DEFAULT_PRUNE_THRESHOLD = 15.0
_PRUNE_LOOKBACK_SECONDS = 600.0


# Deterministic pre-check: regex patterns that flag obvious chatgpt-style
# structure before paying for a Codex patina call. When any of these fire,
# prune immediately. Patina runs anyway in parallel for telemetry and to
# catch nuanced AI-tone that the regex cannot see.

# 3+ bullet items at the start of lines (markdown - or * or numbered).
# Lowered from 4+ to 3+ after live-Discord case where a 3-bullet "이 세 범위"
# reply slipped through scoring as ai=2.2 / flags=[] but read as chatgpt-style.
_BULLET_LINE_RE = re.compile(r"^\s*(?:[-*]|\d+\.)\s+\S", re.MULTILINE)
# Bold markdown labels used to head a list:
#   **핵심:**, **주제:**             (colon inside bold)
#   **이 세 범위**야:, **요약**입니다: (colon outside bold + short suffix)
# Either form reads as bot-style label-prefix.
_BOLD_LABEL_RE = re.compile(
    r"\*\*[^*\n]{1,30}\*\*[가-힣ㄱ-ㅎ\sA-Za-z]{0,8}[:：]"
    r"|\*\*[^*\n]{1,30}[:：]\s*\*\*"
)
# ATX-style markdown headers (## ~~, ### ~~) — chatgpt-style tutorial structure
# in Discord casual chat reads as bot-like. One header is acceptable for a
# truly long doc; 2+ in a single reply is the failure mode.
_ATX_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S", re.MULTILINE)
# Chunk continuation markers like "(1/4)", "(2/3)", "(part 2/3)"
_CHUNK_MARKER_RE = re.compile(r"\(\s*(?:part\s*)?\d+\s*/\s*\d+\s*\)\s*$", re.IGNORECASE)
# Colon-introducing-list: a line ending in `:` followed (optionally after
# blank lines) by 3+ bullet/numbered items. Allows blank lines between the
# colon line and the bullets — Discord markdown often inserts them.
_COLON_INTRO_RE = re.compile(
    r"[^\n]+[:：]\s*\n+(?:\s*(?:[-*]|\d+\.)\s+\S[^\n]*\n+){3,}",
    re.MULTILINE,
)
# Closing-caveat hedge — start of last paragraph OR start of last sentence.
# Matches `\n` or sentence boundary `.` `!` `?` followed by hedge token,
# anchored to the tail of the response.
_CLOSING_HEDGE_RE = re.compile(
    r"(?:[.!?]\s+|\n\s*)(?:그래도|다만|물론|한편)\s+[^\n]+[.!?]?\s*$"
)
# Inline backtick decoration — AI text often wraps dates, status words, and
# plain phrases in `…` for "formatting polish". Real casual chat doesn't.
# Fenced code blocks are stripped before this runs, so this only catches
# the inline `…` form.
_INLINE_BACKTICK_RE = re.compile(r"`([^`\n]{1,80})`")
# Markdown bold emphasis on casual single-word phrases — \"**삼체**\" /
# \"**원피스 실사**\". Distinct from _BOLD_LABEL_RE (which matches the
# \"**제목:**\" header-style pattern); this catches word-level decoration
# patina sometimes leaves behind when it preserves a quoted title.
_INLINE_BOLD_RE = re.compile(r"\*\*([^*\n]{1,80})\*\*")
_TONE_METADATA_LINE_RE = re.compile(
    r"^\s*(?:tone|tone_source|tone_evidence|tone_confidence)\s*:",
    re.IGNORECASE,
)
_TONE_METADATA_TAIL_RE = re.compile(
    r"\s*---\s*(?:\n\s*)?tone\s*:[\s\S]*$",
    re.IGNORECASE,
)
_PUBLIC_TOOL_PROGRESS_PREFIXES = (
    "📚",
    "💻",
    "🔍",
    "⚙️",
    "🛠️",
    "🧠",
    "📖",
    "🌐",
)
_PUBLIC_RUNTIME_NOISE_PREFIXES = (
    "⚠️",
    "❌",
    "Error:",
    "[error]",
    "지금 하던 작업 잠깐 멈췄어요",
)
_SELF_EVOLUTION_PRESENT_RE = re.compile(
    r"(자체\s*진화|자기\s*발전|진화\s*중|수정\s*중|손보는\s*중|코드[^\n]{0,12}손보)",
    re.IGNORECASE,
)
# Operator-flagged hedge softeners ("…같아", "…보여", "…는 듯", "…는 느낌").
# A single occurrence isn't AI-tone; three or more in one response reads as
# the LLM hedging every clause to stay safe.
_HEDGE_SOFTENER_RE = re.compile(
    r"(?:같아요?|같음|보여요?|보임|보이는데|느낌|느낌이야|느낌임|듯해요?|듯함)(?=[\s.,!?…]|$)"
)
_POLITE_REGISTER_ENDING_RE = re.compile(
    r"(?:요|죠|네요|네여|습니다|습니까|슴다|읍니다|용|여)[.!?…~ㅋㅎㅠㅜ]*(?=$|\s)"
)
_BANMAL_REGISTER_ENDING_RE = re.compile(
    r"(?:야|해|할게|좋아|맞아|몰라|있어|없어|가자|보자|하자|돼|아냐|아니야)"
    r"[.!?…~ㅋㅎㅠㅜ]*(?=$|\s)"
)
_PUBLIC_BANMAL_ENDING_FIXES = (
    (re.compile(r"괜찮아(?P<suffix>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"괜찮아요\g<suffix>"),
    (re.compile(r"아니야(?P<suffix>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"아니에요\g<suffix>"),
    (re.compile(r"아냐(?P<suffix>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"아니에요\g<suffix>"),
    (re.compile(r"좋아(?P<suffix>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"좋아요\g<suffix>"),
    (re.compile(r"맞아(?P<suffix>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"맞아요\g<suffix>"),
    (re.compile(r"몰라(?P<suffix>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"몰라요\g<suffix>"),
    (re.compile(r"있어(?P<suffix>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"있어요\g<suffix>"),
    (re.compile(r"없어(?P<suffix>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"없어요\g<suffix>"),
    (re.compile(r"할게(?P<suffix>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"할게요\g<suffix>"),
    (re.compile(r"갈게(?P<suffix>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"갈게요\g<suffix>"),
    (re.compile(r"볼게(?P<suffix>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"볼게요\g<suffix>"),
    (re.compile(r"해줘(?P<suffix>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"해주세요\g<suffix>"),
    (re.compile(r"돼(?P<suffix>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"돼요\g<suffix>"),
)

# Unprompted bio-recitation. The operator already knows the bot is their
# personal assistant on Hermes Agent with DGM-H — repeating it in casual
# replies reads as bot-like self-promotion. Trigger when 2+ of these
# phrases appear together in a single response.
_BIO_RECITATION_TOKENS = (
    "Hermes Agent",
    "DGM-H",
    "self-evolution",
    "self-evolving",
    "self improvement",
    "persona",
    "개인 어시스턴트",
    "예전 flask 프로젝트",
    "flask 프로젝트랑은 무관",
    "flask 프로젝트와는 무관",
    "튜닝되는 skill",
    "튜닝되는 스킬",
    "evolve via DGM-H",
    "feedback-driven self",
)


def _strip_internal_metadata_blocks(content: str) -> str:
    """Remove patina/tone metadata accidentally appended to user-visible text."""
    text = str(content or "")
    text = _TONE_METADATA_TAIL_RE.sub("", text).rstrip()
    if not text:
        return ""

    kept: list[str] = []
    for line in text.splitlines():
        if _TONE_METADATA_LINE_RE.match(line):
            break
        kept.append(line)
    return "\n".join(kept).rstrip()


def _is_public_runtime_noise(content: str) -> bool:
    """Return True for runtime/status/tool-progress frames not meant for public chat."""
    text = str(content or "").strip()
    if not text:
        return False
    if any(text.startswith(prefix) for prefix in _PUBLIC_RUNTIME_NOISE_PREFIXES):
        return True
    if any(token in text for token in ("Non-retryable error", "BadRequestError", "HTTP 400")):
        return True
    if "skill_view:" in text or "💻 terminal:" in text:
        return True
    first_line = text.splitlines()[0]
    return any(first_line.startswith(prefix) for prefix in _PUBLIC_TOOL_PROGRESS_PREFIXES) and ":" in first_line


def _sanitize_public_outbound(content: str) -> str:
    """Apply deterministic public-room safety cleanup before Discord send."""
    text = _strip_internal_metadata_blocks(content)
    if _is_public_runtime_noise(text):
        return ""
    return text.strip()


def _apply_self_evolution_notice(
    content: str,
    *,
    is_public_channel: bool,
) -> str:
    """Prefix public replies with the self-evolution disclosure when active."""
    text = str(content or "").strip()
    if not is_public_channel or not text:
        return text
    try:
        from dgmh.self_evolution_status import get_self_evolution_notice

        notice = get_self_evolution_notice()
    except Exception:
        logger.exception("[humanness_hook] self-evolution notice lookup failed")
        return text
    if not notice:
        return text
    if notice in text or _SELF_EVOLUTION_PRESENT_RE.search(text):
        return text
    return f"{notice} {text}"


def _normalize_public_haeyo_register(content: str) -> str:
    """Conservatively convert common sentence-final 반말 endings for public chat."""
    text = str(content or "")
    for pattern, replacement in _PUBLIC_BANMAL_ENDING_FIXES:
        text = pattern.sub(replacement, text)
    return text


# operator patch: 해요체 → 반말 mechanical converter (always force 반말)
_FORCE_BANMAL_FIXES = (
    # Longer patterns first to avoid sub-string clash
    (re.compile(r"드릴까요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"드릴까\g<s>"),
    (re.compile(r"드릴게요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"드릴게\g<s>"),
    (re.compile(r"드려요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"드려\g<s>"),
    (re.compile(r"해주세요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"해줘\g<s>"),
    (re.compile(r"잖아요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"잖아\g<s>"),
    (re.compile(r"거든요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"거든\g<s>"),
    (re.compile(r"네요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"네\g<s>"),
    (re.compile(r"습니다(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"어\g<s>"),
    (re.compile(r"ㅂ니다(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"어\g<s>"),
    # Composed-syllable -ㅂ니다 forms (Hangul precomposed; literal regex above misses these)
    (re.compile(r"됩니다(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"돼\g<s>"),
    (re.compile(r"합니다(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"해\g<s>"),
    (re.compile(r"갑니다(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"가\g<s>"),
    (re.compile(r"옵니다(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"와\g<s>"),
    (re.compile(r"봅니다(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"봐\g<s>"),
    (re.compile(r"줍니다(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"줘\g<s>"),
    (re.compile(r"압니다(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"알아\g<s>"),
    (re.compile(r"삽니다(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"사\g<s>"),
    (re.compile(r"씁니다(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"써\g<s>"),
    (re.compile(r"마십니다(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"마셔\g<s>"),
    # -입니다 / -드립니다 (자기소개, 비즈니스 톤)
    (re.compile(r"입니다(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"야\g<s>"),
    (re.compile(r"드립니다(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"드려\g<s>"),
    # 연결어미 -고요 (붙이고요 → 붙이고)
    (re.compile(r"고요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"고\g<s>"),
    # 조사/연결 + 요 끝 (수준으로요 → 수준으로, 거기서요 → 거기서)
    (re.compile(r"으로요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"으로\g<s>"),
    (re.compile(r"에서요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"에서\g<s>"),
    (re.compile(r"부터요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"부터\g<s>"),
    (re.compile(r"까지요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"까지\g<s>"),
    (re.compile(r"한테요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"한테\g<s>"),
    (re.compile(r"라서요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"라서\g<s>"),
    (re.compile(r"아서요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"아서\g<s>"),
    (re.compile(r"어서요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"어서\g<s>"),
    # 음운변화 -서요 forms (precomposed Hangul; literal -아서요/-어서요 misses)
    (re.compile(r"해서요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"해서\g<s>"),
    (re.compile(r"와서요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"와서\g<s>"),
    (re.compile(r"가서요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"가서\g<s>"),
    (re.compile(r"빠서요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"빠서\g<s>"),
    (re.compile(r"는데요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"는데\g<s>"),
    (re.compile(r"은데요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"은데\g<s>"),
    # -하- + -ㄴ데요 (간단한데요, 친한데요)
    (re.compile(r"한데요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"한데\g<s>"),
    (re.compile(r"라구요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"라고\g<s>"),
    # 명령형 -(으)십시오 — composed forms (어간+어 패턴 비문 회피)
    (re.compile(r"가십시오(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"가\g<s>"),
    (re.compile(r"오십시오(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"와\g<s>"),
    (re.compile(r"보십시오(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"봐\g<s>"),
    (re.compile(r"하십시오(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"해\g<s>"),
    (re.compile(r"주십시오(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"줘\g<s>"),
    (re.compile(r"아니에요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"아냐\g<s>"),
    (re.compile(r"예요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"야\g<s>"),
    (re.compile(r"에요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"야\g<s>"),
    (re.compile(r"세요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"어\g<s>"),
    (re.compile(r"어요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"어\g<s>"),
    (re.compile(r"아요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"아\g<s>"),
    (re.compile(r"줘요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"줘\g<s>"),
    (re.compile(r"돼요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"돼\g<s>"),
    (re.compile(r"해요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"해\g<s>"),
    (re.compile(r"가요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"가\g<s>"),
    (re.compile(r"와요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"와\g<s>"),
    (re.compile(r"까요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"까\g<s>"),
    (re.compile(r"죠(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"지\g<s>"),
    # Korean verb conjugation variants
    (re.compile(r"워요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"워\g<s>"),  # 더워요, 추워요
    (re.compile(r"펴요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"펴\g<s>"),
    (re.compile(r"켜요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"켜\g<s>"),
    (re.compile(r"려요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"려\g<s>"),
    (re.compile(r"여요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"여\g<s>"),
    (re.compile(r"게요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"게\g<s>"),  # 할게요, 갈게요, 볼게요
    (re.compile(r"봐요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"봐\g<s>"),
    (re.compile(r"봬요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"봬\g<s>"),
    (re.compile(r"줄게요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"줄게\g<s>"),
    (re.compile(r"줘요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"줘\g<s>"),  # already had but reinforce
    (re.compile(r"지요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"지\g<s>"),
    (re.compile(r"쥐요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"쥐\g<s>"),
    (re.compile(r"라요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"라\g<s>"),  # 골라요 → 골라
    (re.compile(r"러요(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"러\g<s>"),
    (re.compile(r"러죠(?P<s>[.!?…~ㅋㅎㅠㅜ]*)(?=$|\s)"), r"러지\g<s>"),
)


def _force_banmal_register(content: str) -> str:
    """Mechanical 해요체 → 반말 어미 변환 (operator patch, always-on)."""
    text = str(content or "")
    for pattern, replacement in _FORCE_BANMAL_FIXES:
        text = pattern.sub(replacement, text)
    return text


def _structural_pollution_check(content: str) -> tuple[bool, list[str]]:
    """Return (should_prune, list_of_matched_pattern_names).

    Conservative — fires only on patterns the operator has flagged as
    chatgpt-tells in casual chat. Skips when the response is mostly a code
    block (the patterns inside fenced code don't count).
    """
    flags = []

    # Strip fenced code blocks before checking; bullets inside code are fine.
    stripped = re.sub(r"```[\s\S]*?```", "", content)

    bullets = _BULLET_LINE_RE.findall(stripped)
    if len(bullets) >= 3:
        flags.append(f"bullet-list-3plus({len(bullets)})")

    if _BOLD_LABEL_RE.search(stripped):
        flags.append("bold-label-header")

    atx_headers = _ATX_HEADER_RE.findall(stripped)
    if len(atx_headers) >= 2:
        flags.append(f"atx-headers({len(atx_headers)})")

    if _CHUNK_MARKER_RE.search(stripped):
        flags.append("chunk-marker")

    if _COLON_INTRO_RE.search(stripped):
        flags.append("colon-introducing-list")

    if _CLOSING_HEDGE_RE.search(stripped):
        flags.append("closing-caveat-hedge")

    bio_hits = [t for t in _BIO_RECITATION_TOKENS if t in stripped]
    if len(bio_hits) >= 2:
        flags.append(f"bio-recitation({len(bio_hits)}:{','.join(bio_hits[:3])})")

    # Inline backticks in casual chat — AI-tone signal. 2+ anywhere, or
    # 1+ in short responses (<150 chars) where decoration stands out.
    inline_ticks = _INLINE_BACKTICK_RE.findall(stripped)
    if len(inline_ticks) >= 2 or (inline_ticks and len(stripped) < 150):
        flags.append(f"inline-backtick({len(inline_ticks)})")

    # Word-level **bold** decoration (titles, brand names). Distinct from
    # the label-style **xx:** check above. Any occurrence in casual chat
    # is AI-tone.
    inline_bolds = _INLINE_BOLD_RE.findall(stripped)
    if inline_bolds:
        flags.append(f"inline-bold({len(inline_bolds)})")

    # Hedge softener pileup — 3+ "같아/보여/듯" in one response = LLM
    # over-hedging.
    softener_hits = _HEDGE_SOFTENER_RE.findall(stripped)
    if len(softener_hits) >= 3:
        flags.append(f"hedge-softener({len(softener_hits)})")

    if _POLITE_REGISTER_ENDING_RE.search(
        stripped
    ) and _BANMAL_REGISTER_ENDING_RE.search(stripped):
        flags.append("mixed-register")

    return (bool(flags), flags)


def _read_soul_hash() -> str:
    """Compute sha256 of the active SOUL.md content (used as generation tag)."""
    soul_path = (
        Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")) / "SOUL.md"
    )
    try:
        content = soul_path.read_text(encoding="utf-8")
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    except Exception:
        return ""


async def _patina_rewrite_dispatch(
    content: str, *, timeout_s: float = 30.0, register_mode: str = "mirror"
) -> Optional[str]:
    """Pick between dynamic (kakao-mimic-rag) and static patina profiles.

    When ``DGMH_PATINA_PROFILE`` is ``kakao-mimic-rag``, build a per-call
    patina profile from TF-IDF-retrieved corpus anchors. Returns the
    rewritten text on success. The static-profile path is the steady
    state; the RAG path is the new Phase 2 surface that retrieves
    draft-specific voice anchors on every turn.

    Returns ``None`` on any failure path so the caller can decide
    whether to defer to the simpler Codex-direct rewrite.
    """
    from dgmh.patina_judge import humanness_rewrite_with_profile

    profile_env = (os.environ.get("DGMH_PATINA_PROFILE", "") or "").strip()
    if profile_env == "kakao-mimic-rag":
        try:
            from dgmh.kakao_style_retrieval import rewrite_with_rag_profile

            rewritten = await asyncio.to_thread(
                rewrite_with_rag_profile,
                content,
                backend="codex-cli",
                timeout_s=timeout_s,
                register_mode=register_mode,
            )
        except Exception:
            logger.exception(
                "[humanness_hook] RAG profile path raised; "
                "falling back to static kakao-mimic"
            )
            rewritten = None
        if rewritten is not None:
            return rewritten
        # RAG returned None (no anchors / patina error). Fall through
        # to the static kakao-mimic profile so we still get *some*
        # voice mirroring instead of original content.
        return await asyncio.to_thread(
            humanness_rewrite_with_profile,
            content,
            profile="kakao-mimic",
            backend="codex-cli",
            timeout_s=timeout_s,
        )

    return await asyncio.to_thread(
        humanness_rewrite_with_profile,
        content,
        backend="codex-cli",
        timeout_s=timeout_s,
    )


def _should_score(content: str) -> bool:
    if os.environ.get("DGMH_HUMANNESS_DISABLED"):
        return False
    if not content or not content.strip():
        return False
    min_chars = int(os.environ.get("DGMH_HUMANNESS_MIN_CHARS", _DEFAULT_MIN_CHARS))
    if len(content.strip()) < min_chars:
        return False
    stripped = content.lstrip()
    for prefix in _ERROR_PREFIXES:
        if stripped.startswith(prefix):
            return False
    return True


_PRUNE_RECENT_LOOKBACK_S = 60.0


def _prune_polluting_message(
    *,
    message_id: Optional[str] = None,
    ai_score: float,
    lookback_s: float = _PRUNE_RECENT_LOOKBACK_S,
) -> int:
    """Delete the most-recent polluting assistant row from state.db.messages.

    Step 5 (v3): keys by recency rather than content match.

    Hermes' state.db ``messages`` schema is keyed by an internal
    auto-increment ``id``; it does NOT carry a Discord ``message_id``
    column. The previous implementation matched by ``content``, which
    silently broke whenever any mid-flight rewrite stage (humanness
    rewrite, future patina-profile rewrite) mutated the outbound text
    — Hermes core wrote the draft before the wrap, so the post-rewrite
    text never matched any row.

    The fix: delete the single most-recent ``role='assistant'`` row
    inserted within ``lookback_s`` seconds of "now". Because the
    background score thread fires immediately after ``adapter.send``,
    that row is overwhelmingly the polluting reply we just scored.

    ``message_id`` is the Discord message id (passed in for telemetry
    only — it does not appear in the state.db schema, so we cannot
    filter on it; logged for traceability).

    Returns the number of rows deleted (0 or 1 in practice).

    Disabled by setting ``DGMH_PRUNE_DISABLED``. Threshold overridden
    via ``DGMH_PRUNE_AI_THRESHOLD`` (default 15.0).
    """
    import sqlite3

    if os.environ.get("DGMH_PRUNE_DISABLED"):
        return 0

    threshold = float(
        os.environ.get("DGMH_PRUNE_AI_THRESHOLD", _DEFAULT_PRUNE_THRESHOLD)
    )
    if ai_score < threshold:
        return 0

    db_path = (
        Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")) / "state.db"
    )
    if not db_path.exists():
        logger.info("humanness_hook: state.db not found at %s, skip prune", db_path)
        return 0

    try:
        con = sqlite3.connect(str(db_path), timeout=5.0)
        cur = con.cursor()
        # Two-step delete-by-id so we only ever remove ONE row even if
        # several assistant messages fall inside the lookback window.
        row = cur.execute(
            "SELECT id FROM messages "
            "WHERE role = 'assistant' "
            "AND timestamp > strftime('%s','now') - ? "
            "ORDER BY timestamp DESC, id DESC LIMIT 1",
            (lookback_s,),
        ).fetchone()
        if row is None:
            con.close()
            return 0
        target_id = row[0]
        n = cur.execute(
            "DELETE FROM messages WHERE id = ?", (target_id,)
        ).rowcount
        con.commit()
        con.close()
        if n and message_id:
            logger.info(
                "humanness_hook: pruned state.db row id=%s for discord msg=%s "
                "(ai=%.1f >= threshold)",
                target_id, message_id, ai_score,
            )
        return n
    except Exception:
        logger.exception("humanness_hook: prune query failed")
        return 0


# ---------------------------------------------------------------------------
# Step 2 (v3): bimodal humanlike latency with operator shortcut
# ---------------------------------------------------------------------------
#
# Real humans don't respond in 200ms. They take longer when the channel is
# warm (recent activity) and longer still when the channel is cold (no
# activity for ages). The bot's instant reply is one of the strongest tells
# in public chat. Bimodal latency closes that gap.
#
# Buckets (gated by DGMH_PUBLIC_HUMAN_MODE=1 AND public channel):
#   active (last_msg < 30s)   → max(3, min(20,  gauss(8,   4)))
#   warm   (last_msg < 600s)  → max(10, min(120, gauss(35,  20)))
#   cold   (otherwise)        → max(60, min(900, gauss(180, 90)))
#
# AC1.1: when the bot is replying TO the operator (author is operator),
# the operator-shortcut bucket fires regardless of activity state:
#   operator → max(3, min(15, gauss(8, 4)))
# This keeps urgent operator pings from waiting up to 15 minutes mid-night.
#
# Operator user-id default: 266436073557590016. Override via
# DGMH_OPERATOR_USER_ID. The wrapped_send call site resolves is_operator
# from metadata["author_id"] when available, falling back to that env id.

_DEFAULT_OPERATOR_USER_ID = "266436073557590016"

# In-memory map: channel_id -> last bot-send unix timestamp. Keeps Step 2's
# bucket lookup honest about when this surface last had bot activity.
_LAST_BOT_SEND_TS: dict[str, float] = {}


def _operator_user_ids() -> set[str]:
    raw = os.environ.get("DGMH_OPERATOR_USER_ID", _DEFAULT_OPERATOR_USER_ID)
    return {x.strip() for x in raw.split(",") if x.strip()}


def _resolve_is_operator(
    metadata: Optional[dict[str, Any]],
    *,
    author_id: Optional[str] = None,
) -> bool:
    """Best-effort check of whether the inbound author is the operator.

    Reads ``metadata["author_id"]`` first (Hermes adapter passes it through
    when available), then falls back to an explicit ``author_id`` arg.
    Returns False on any unexpected shape so a missing/garbled metadata
    blob just demotes the call to the regular bimodal path.
    """
    candidate: Optional[str] = author_id
    if candidate is None and isinstance(metadata, dict):
        candidate = metadata.get("author_id") or metadata.get("user_id")
    if candidate is None:
        return False
    try:
        return str(candidate) in _operator_user_ids()
    except Exception:
        return False


def _seconds_since_last_msg(channel_id: str, *, now: Optional[float] = None) -> float:
    """Seconds since this surface last saw a bot send.

    Returns a large sentinel (1e9) when no prior send is recorded so a
    cold-start surface lands in the cold bucket.
    """
    last = _LAST_BOT_SEND_TS.get(str(channel_id))
    if last is None:
        return 1e9
    return max(0.0, (now if now is not None else time.time()) - last)


def _record_bot_send(channel_id: str, *, ts: Optional[float] = None) -> None:
    _LAST_BOT_SEND_TS[str(channel_id)] = ts if ts is not None else time.time()


def _human_latency_seconds(
    channel_id: str,
    *,
    is_operator: bool,
    rng: Optional[random.Random] = None,
    now: Optional[float] = None,
) -> float:
    """Sample a humanlike pre-send sleep in seconds for ``channel_id``.

    See module-level docstring for bucket definitions. ``rng`` is exposed
    so unit tests can pin the distribution; production callers pass None
    and use the module-default RNG.
    """
    rng_ = rng or random
    if is_operator:
        return max(3.0, min(15.0, rng_.gauss(8.0, 4.0)))
    last_ago = _seconds_since_last_msg(str(channel_id), now=now)
    if last_ago < 30.0:
        return max(3.0, min(20.0, rng_.gauss(8.0, 4.0)))
    if last_ago < 600.0:
        return max(5.0, min(45.0, rng_.gauss(15.0, 8.0)))
    return max(10.0, min(90.0, rng_.gauss(30.0, 15.0)))


def _public_human_mode_enabled() -> bool:
    """Whether DGMH_PUBLIC_HUMAN_MODE is set to a truthy value."""
    val = os.environ.get("DGMH_PUBLIC_HUMAN_MODE", "")
    return val not in ("", "0", "false", "False")


def _score_in_thread(
    *,
    content: str,
    chat_id: str,
    thread_id: Optional[str],
    message_id: Optional[str],
) -> None:
    """Run patina scoring on a background thread and append the record."""
    soul_hash = _read_soul_hash()
    text_length = len(content)
    pruned_count = 0

    # Step 1: deterministic pre-check. Prune obvious structural pollution
    # without waiting for the slow Codex patina round-trip.
    structural_hit, struct_flags = _structural_pollution_check(content)
    if structural_hit and not os.environ.get("DGMH_PRUNE_DISABLED"):
        # Force prune by passing a synthetic high score above threshold.
        # Step 5 (v3): prune is keyed by recency, not content, so the
        # rewrite stages can mutate ``content`` mid-flight without breaking
        # the prune.
        pruned_pre = _prune_polluting_message(
            message_id=message_id, ai_score=999.0
        )
        if pruned_pre:
            logger.info(
                "[humanness_hook] structural pre-prune (%s) removed %d row",
                ",".join(struct_flags), pruned_pre,
            )
            pruned_count += pruned_pre

    try:
        from dgmh.patina_judge import score_humanness, PatinaScoreError

        try:
            result = score_humanness(content, lang="ko")
            # Only attempt patina-based prune if the structural pre-check did
            # not already remove the row.
            if pruned_count == 0:
                pruned_count = _prune_polluting_message(
                    message_id=message_id, ai_score=result.ai_score
                )
                if pruned_count:
                    logger.info(
                        "[humanness_hook] pruned %d polluting reply (ai=%.1f >= threshold)",
                        pruned_count, result.ai_score,
                    )
            record = make_success_record(
                chat_id=chat_id,
                thread_id=thread_id,
                message_id=message_id,
                soul_md_hash=soul_hash,
                text_length=text_length,
                ai_score=result.ai_score,
                human_likeness=result.human_likeness,
                sub_scores=result.sub_scores,
                interpretation=result.interpretation,
                elapsed_s=result.elapsed_s,
            )
            record["pruned"] = pruned_count
            record["structural_flags"] = struct_flags
        except PatinaScoreError as exc:
            record = make_error_record(
                chat_id=chat_id,
                thread_id=thread_id,
                message_id=message_id,
                soul_md_hash=soul_hash,
                text_length=text_length,
                error=f"{type(exc).__name__}: {exc}",
            )
    except Exception as exc:  # noqa: BLE001 — never crash the gateway from a side hook
        record = make_error_record(
            chat_id=chat_id,
            thread_id=thread_id,
            message_id=message_id,
            soul_md_hash=soul_hash,
            text_length=text_length,
            error=f"unexpected: {type(exc).__name__}: {exc}",
        )

    try:
        append_record(record)
    except Exception:
        logger.exception("humanness_hook: failed to append record")


def _wrap_send(adapter: Any) -> None:
    """Monkey-patch the adapter's ``send`` to async-score outbound content."""
    if getattr(adapter, _PATCH_FLAG, False):
        logger.info("[humanness_hook] send already patched, skipping")
        return

    original_send = adapter.send

    async def wrapped_send(
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        def _suppressed_result(reason: str):
            try:
                from gateway.platforms.base import SendResult

                return SendResult(
                    success=True,
                    raw_response={"suppressed_by": reason},
                )
            except Exception:
                return {"success": True, "suppressed_by": reason}

        # Pre-send rewrite. Step 4 (v3) splits the rewrite path by channel:
        #
        #   public + DGMH_PUBLIC_HUMAN_MODE=1
        #     → humanness_rewrite_with_profile(profile="social",
        #         backend="codex-cli") (Step P1 surface). Falls back to
        #         humanness_rewrite (Codex-direct) on None per D8 in plan.
        #   1:1 (or any non-public surface)
        #     → existing humanness_rewrite, only when DGMH_REWRITE_ENABLED
        #         (preserves AC7 verifier baseline behavior).
        rewrite_enabled = (
            bool(os.environ.get("DGMH_REWRITE_ENABLED"))
            and not os.environ.get("DGMH_REWRITE_DISABLED")
        )

        is_public_mode = False
        is_public_channel_send = False
        try:
            from dgmh.honcho_client import is_public_channel

            is_public_channel_send = is_public_channel(str(chat_id))
            is_public_mode = (
                _public_human_mode_enabled() and is_public_channel_send
            )
        except Exception:
            logger.exception(
                "[humanness_hook] public-mode resolver failed; defaulting off"
            )

        if is_public_channel_send:
            sanitized = _sanitize_public_outbound(content)
            if sanitized != content:
                logger.info(
                    "[humanness_hook] public outbound sanitizer applied "
                    "(len=%d→%d)",
                    len(content or ""),
                    len(sanitized),
                )
                content = sanitized
            if not content.strip():
                try:
                    from dgmh.honcho_client import set_post_rewrite_content

                    set_post_rewrite_content("")
                except Exception:
                    logger.exception(
                        "[humanness_hook] could not publish suppressed public content"
                    )
                return _suppressed_result("dgmh_public_sanitizer")

        # DGM-H W1 dedup flag — true once patina has produced a usable
        # rewrite this turn, so the downstream gate can skip its own
        # redundant patina round-trip.
        _pre_rewritten: bool = False

        if is_public_mode and _should_score(content):
            # Step 4 stage 3 (public path): patina --profile social rewrite.
            # Always attempts a rewrite, structural pollution or not — the
            # social profile is responsible for amplifying voice traits.
            try:
                from dgmh.patina_judge import humanness_rewrite

                # Dispatcher picks RAG profile (kakao-mimic-rag) vs static
                # based on DGMH_PATINA_PROFILE env; both end up invoking
                # patina with --profile <name>.
                rewritten = await _patina_rewrite_dispatch(
                    content,
                    timeout_s=30.0,
                    register_mode="mirror",  # operator patch: stop forcing 해요체
                )
                if rewritten is None:
                    # D8 fallback: Codex-direct prompt as a safety net.
                    rewritten = await asyncio.to_thread(
                        humanness_rewrite, content, timeout_s=60.0
                    )

                if rewritten and rewritten != content:
                    logger.info(
                        "[humanness_hook] public-mode rewrite applied "
                        "(len=%d→%d)",
                        len(content), len(rewritten),
                    )
                    content = rewritten
                    _pre_rewritten = True
            except Exception:
                logger.exception(
                    "[humanness_hook] public-mode rewrite failed; using original"
                )
        elif rewrite_enabled and _should_score(content):
            structural_hit, struct_flags = _structural_pollution_check(content)
            if structural_hit:
                try:
                    from dgmh.patina_judge import humanness_rewrite

                    # Dispatcher honors DGMH_PATINA_PROFILE — including the
                    # Phase 2 kakao-mimic-rag profile that runs per-turn
                    # corpus retrieval before invoking patina.
                    rewritten = await _patina_rewrite_dispatch(
                        content, timeout_s=30.0
                    )
                    if rewritten is None:
                        rewritten = await asyncio.to_thread(
                            humanness_rewrite, content, timeout_s=60.0
                        )
                    if rewritten and rewritten != content:
                        # Sanity: don't replace if rewrite still has pollution.
                        rew_hit, _ = _structural_pollution_check(rewritten)
                        if not rew_hit:
                            logger.info(
                                "[humanness_hook] pre-send rewrite applied "
                                "(flags=%s len=%d→%d)",
                                ",".join(struct_flags),
                                len(content),
                                len(rewritten),
                            )
                            content = rewritten
                            _pre_rewritten = True
                        else:
                            logger.info(
                                "[humanness_hook] rewrite still polluted (%s); "
                                "keeping original",
                                struct_flags,
                            )
                except Exception:
                    logger.exception(
                        "[humanness_hook] pre-send rewrite failed; using original"
                    )

        # Deterministic inline-backtick strip. patina is a probabilistic
        # rewriter — it cannot be the sole line of defense against an AI-
        # tone signal the project has decided is always wrong in casual
        # chat. Decorative `…` ticks are unconditionally stripped here so
        # the outbound surface carries the project's guarantee, not the
        # rewriter's best effort. The regex matches single-backtick pairs
        # on the same line only, so fenced code blocks survive intact.
        if "`" in content:
            scrubbed = _INLINE_BACKTICK_RE.sub(r"\1", content)
            if scrubbed != content:
                logger.info(
                    "[humanness_hook] inline-backtick scrub applied "
                    "(len=%d→%d)",
                    len(content),
                    len(scrubbed),
                )
                content = scrubbed

        # Same guarantee for **bold** word decoration. patina's profile
        # rules ask it to suppress markdown bold but it sometimes leaves
        # quoted titles (e.g., **삼체**) intact. Strip the markers; keep
        # the content.
        if "**" in content:
            scrubbed = _INLINE_BOLD_RE.sub(r"\1", content)
            if scrubbed != content:
                logger.info(
                    "[humanness_hook] inline-bold scrub applied "
                    "(len=%d→%d)",
                    len(content),
                    len(scrubbed),
                )
                content = scrubbed

        # operator patch: force 반말 on every send (1:1 + public), replaces the
        # old public-only haeyo normalizer which forced 해요체.
        normalized = _force_banmal_register(content)
        if normalized != content:
            logger.info(
                "[humanness_hook] force_banmal applied (len=%d→%d)",
                len(content),
                len(normalized),
            )
            content = normalized
        if False and is_public_channel_send:
            normalized = _normalize_public_haeyo_register(content)
            if normalized != content:
                logger.info(
                    "[humanness_hook] public register normalizer applied "
                    "(len=%d→%d)",
                    len(content),
                    len(normalized),
                )
                content = normalized

        # Pre-send length gate (skill-backed, with in-process fallback). Final
        # length enforcement after patina rewrite to honor SOUL.md 1-2 sentence
        # cap. Handles both public and 1:1 paths uniformly.
        _gate_decision: Optional[str] = None
        _gate_in_len: int = len(content)
        _gate_out_len: int = len(content)
        try:
            from dgmh.hermes_integration.pre_send_gate import gate_async as _length_gate
            from dgmh.honcho_client import channel_kind_for as _channel_kind_for

            _kind = _channel_kind_for(str(chat_id))
            _decision, _new_content = await _length_gate(
                content,
                channel_kind=_kind,
                pre_rewritten=_pre_rewritten,
            )
            _gate_decision = _decision
            _gate_out_len = len(_new_content)
            if _decision != "send" and _new_content != content:
                logger.info(
                    "[humanness_hook] length gate applied "
                    "(decision=%s len=%d→%d)",
                    _decision, len(content), len(_new_content),
                )
                content = _new_content
        except Exception:
            logger.exception("[humanness_hook] length gate failed; using rewrite output as-is")

        if is_public_channel_send:
            sanitized = _sanitize_public_outbound(content)
            content = _apply_self_evolution_notice(
                sanitized,
                is_public_channel=True,
            )
            if not content.strip():
                try:
                    from dgmh.honcho_client import set_post_rewrite_content

                    set_post_rewrite_content("")
                except Exception:
                    logger.exception(
                        "[humanness_hook] could not publish suppressed public content"
                    )
                return _suppressed_result("dgmh_public_sanitizer")

        # Step 4 stage 7 (v3): publish the post-rewrite content into the
        # ContextVar so the outer honcho wrapper mirrors the FINAL outbound
        # text rather than the pre-rewrite draft it received as its own
        # ``content`` parameter. Honcho reads with a None default, so when
        # we are NOT in a rewrite path the var stays unset and honcho
        # falls back to its local content.
        try:
            from dgmh.honcho_client import set_post_rewrite_content

            set_post_rewrite_content(content)
        except Exception:
            logger.exception(
                "[humanness_hook] could not publish post-rewrite content"
            )

        # Step 2 (v3): bimodal humanlike pre-send sleep for public channels.
        # Gated by DGMH_PUBLIC_HUMAN_MODE=1 AND public channel — with the
        # env unset (default), behavior is byte-identical to the prior
        # adapter.send pipeline so the 1:1 verifier baseline holds.
        try:
            if _public_human_mode_enabled():
                from dgmh.honcho_client import is_public_channel

                if is_public_channel(str(chat_id)):
                    is_op = _resolve_is_operator(metadata)
                    sleep_s = _human_latency_seconds(
                        str(chat_id), is_operator=is_op
                    )
                    logger.info(
                        "[humanness_hook] bimodal sleep %.2fs for chat=%s is_op=%s",
                        sleep_s, chat_id, is_op,
                    )
                    await asyncio.sleep(sleep_s)
        except Exception:
            logger.exception(
                "[humanness_hook] bimodal latency failed; sending without sleep"
            )

        result = await original_send(chat_id, content, reply_to=reply_to, metadata=metadata)

        # Step 2 (v3): record this send timestamp so the next reply on
        # this channel sees the up-to-date last-message-ago bucket.
        try:
            _record_bot_send(str(chat_id))
        except Exception:
            pass

        # DGM-H W3: cache the gate decision keyed by the outbound message
        # id so reaction_hook can later refine `negative` reactions into
        # `negative_truncated` vs `negative_content` based on whether the
        # send was cut at the cap.
        try:
            if _gate_decision is not None:
                from dgmh.hermes_integration.gate_decisions import (
                    _ends_with_ellipsis_marker,
                    record_decision,
                )

                _msg_id = getattr(result, "message_id", None) or (
                    result.get("message_id") if isinstance(result, dict) else None
                )
                if _msg_id:
                    record_decision(
                        str(_msg_id),
                        decision=_gate_decision,
                        in_len=_gate_in_len,
                        out_len=_gate_out_len,
                        ends_in_ellipsis=_ends_with_ellipsis_marker(content),
                    )
        except Exception:
            logger.exception("[humanness_hook] gate_decisions cache write failed")

        try:
            if not _should_score(content):
                return result

            thread_id = (metadata or {}).get("thread_id")
            message_id = getattr(result, "message_id", None) or (
                result.get("message_id") if isinstance(result, dict) else None
            )

            # Step 6 (v3): capture context with the resolved channel kind so
            # _score_in_thread sees the right channel_kind ContextVar (used
            # transitively by any honcho-client read it triggers, e.g. via
            # the structural pre-prune path).
            from dgmh.honcho_client import (
                capture_context_with_kind,
                channel_kind_for,
            )

            ctx = capture_context_with_kind(channel_kind_for(str(chat_id)))
            thread = threading.Thread(
                target=ctx.run,
                args=(_score_in_thread,),
                kwargs={
                    "content": content,
                    "chat_id": str(chat_id),
                    "thread_id": str(thread_id) if thread_id else None,
                    "message_id": str(message_id) if message_id else None,
                },
                daemon=True,
                name="dgmh-humanness-score",
            )
            thread.start()
        except Exception:
            logger.exception("[humanness_hook] failed to dispatch scoring; continuing")

        return result

    # Wrap-order markers (Step 0): expose hook identity + the original
    # callable so dgmh.hermes_integration.wrap_order can walk the chain
    # outer→inner and assert ``["honcho", "humanness"]`` at startup.
    wrapped_send.__dgmh_hook_name__ = "humanness"  # type: ignore[attr-defined]
    wrapped_send.__wrapped__ = original_send  # type: ignore[attr-defined]

    adapter.send = wrapped_send  # type: ignore[assignment]
    setattr(adapter, _PATCH_FLAG, True)
    logger.info("[humanness_hook] wrapped DiscordAdapter.send for patina scoring")


async def _wait_and_patch(adapter: Any) -> None:
    """Wait briefly for the adapter to fully connect, then wrap send."""
    try:
        for _ in range(60):
            client = getattr(adapter, "_client", None)
            if client and getattr(client, "is_ready", lambda: False)():
                break
            await asyncio.sleep(0.5)
        _wrap_send(adapter)
        # Best-effort wrap-order check; partner hook (honcho) may not have
        # wrapped yet on startup, in which case this logs a warning. The
        # last hook to finish wrapping observes the full chain.
        try:
            from dgmh.hermes_integration.wrap_order import verify_wrap_chain

            verify_wrap_chain(adapter)
        except Exception:
            logger.exception(
                "[humanness_hook] wrap-order verification raised; continuing"
            )
    except Exception:
        logger.exception("[humanness_hook] _wait_and_patch failed")


async def handle(event_type: str, context: dict) -> None:
    """Gateway startup hook — find DiscordAdapter and wrap its send method."""
    if event_type != "gateway:startup":
        return

    logger.info("[humanness_hook] gateway:startup — wiring outbound humanness scoring")

    from dgmh.hermes_integration.reaction_hook import _find_discord_adapter

    adapter = _find_discord_adapter(context)
    if adapter is None:
        logger.warning(
            "[humanness_hook] No DiscordAdapter found at startup — humanness "
            "scoring not wired."
        )
        return

    asyncio.create_task(_wait_and_patch(adapter))
    logger.info("[humanness_hook] Scheduled humanness wrap task")
