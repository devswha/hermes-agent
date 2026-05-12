"""
dgmh/glm_data/kakao_parser.py — KakaoTalk export parser (Plan v2.1 §4 Step 3).

Public API
----------

    parse_kakao_file(path, *, name_map, redactor=redact_kakao_text, trace=None)
        Generator of `KakaoMsg`. Drops the file preamble (group_name header,
        saved-date stamp, blank) after recording the group_name digest to
        `trace.group_name_sha256`. Parses `[user] [오전/오후 H:MM] msg` lines,
        merges continuation lines into the prior message, and recognizes
        `--------------- YYYY년 M월 D일 요일 ---------------` date separators.

    KakaoParser(path, *, name_map, redactor, trace)
        Class form. Iterating produces `KakaoMsg`s; `.stats` exposes a
        `ParseStats` with per-line-category counts that sum to the file's
        total line count (100% line accountability).

Line categories (mutually exclusive — sum equals total line count):
  - preamble           : the dropped header (group_name + saved-date + blank)
  - separator          : `--------------- <date> ---------------` lines
  - message_header     : `[user] [time] msg` lines (start a new KakaoMsg)
  - continuation       : non-header, non-blank lines that belong to the
                         currently-open message (e.g. wrapped URL, body)
  - blank              : empty lines outside any open message
  - event              : `<name>님이 …습니다.` system events
  - other              : anything we couldn't classify (kept for accountability)
"""

from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Callable, Iterator, Optional

from dgmh.glm_data.pii_redact import RedactionTrace, redact_kakao_text
from dgmh.glm_data.schema import KakaoMsg

# --------------------------------------------------------------------------- #
# Constants and regexes
# --------------------------------------------------------------------------- #

# The operator's 7 kakao files all share a 3-line preamble:
#   1. "<group_name> 님과 카카오톡 대화"   (or English: "<name> saved KakaoTalk Chats with ...")
#   2. "저장한 날짜 : YYYY-MM-DD HH:MM:SS"  (or English: "Date Saved : ...")
#   3. blank (whitespace only)
# (Upstream plan documents "5 lines" but the actual format is 3; we record
# whichever count we observe in `ParseStats.preamble_lines`.)
#
# Codex review MAJOR-3 fix: the parser used to drop the first 3 lines
# unconditionally, silently losing data on English exports or already-
# preamble-stripped files. We now check the *shape* before dropping;
# non-matching files are parsed from line 0 with a stderr warning.
_PREAMBLE_LINES = 3

_GROUP_NAME_RE = re.compile(r"^(.+?)\s*님과 카카오톡 대화")

# English KakaoTalk export header (best-effort — Kakao does not publish a stable
# locale spec, so we match the documented substring).
_GROUP_NAME_EN_RE = re.compile(r"saved KakaoTalk Chats with", re.IGNORECASE)

# Saved-date stamp; either Korean (`저장한 날짜 : ...`) or English (`Date Saved : ...`).
_SAVED_DATE_KR_RE = re.compile(r"^저장한 날짜\s*:")
_SAVED_DATE_EN_RE = re.compile(r"^Date Saved\s*:", re.IGNORECASE)


def _looks_like_preamble(buffered: list[str]) -> bool:
    """Return True iff ``buffered`` matches the recognized KakaoTalk preamble.

    Shape (Plan v2.1 §4 Step 3 + codex MAJOR-3):

    - Line 0: ends with " 님과 카카오톡 대화" (KR) **or** contains
      "saved KakaoTalk Chats with" (EN).
    - Line 1: starts with "저장한 날짜 :" (KR) **or** "Date Saved :" (EN).
    - Line 2: whitespace-only (blank).

    A file missing any of these three signals is parsed from index 0 — the
    caller logs a stderr warning so operators notice malformed exports.
    """
    if len(buffered) < 3:
        return False
    line0 = buffered[0].rstrip("\r\n").strip()
    line1 = buffered[1].rstrip("\r\n").strip()
    line2 = buffered[2].rstrip("\r\n")

    line0_ok = bool(_GROUP_NAME_RE.match(line0)) or bool(
        _GROUP_NAME_EN_RE.search(line0)
    )
    line1_ok = bool(_SAVED_DATE_KR_RE.match(line1)) or bool(
        _SAVED_DATE_EN_RE.match(line1)
    )
    line2_ok = line2.strip() == ""
    return line0_ok and line1_ok and line2_ok

# Date separator: `--------------- 2026년 2월 19일 목요일 ---------------`
_SEPARATOR_RE = re.compile(
    r"^-{3,}\s*\d{4}년\s+\d{1,2}월\s+\d{1,2}일\s+\S+요일\s*-{3,}\s*$"
)

# Message header: `[user] [오전 9:20] msg`
_MESSAGE_HEADER_RE = re.compile(
    r"^\[(?P<user>[^\]]+)\]\s+\[(?P<period>오전|오후)\s+(?P<hh>\d{1,2}):(?P<mm>\d{2})\]\s?(?P<msg>.*)$"
)

# Event line: `<name>님이 들어왔습니다.`, `…님을 초대했습니다.`, etc.
# Requires a `님(?:이|을)` token + an event verb ending in 습니다.
_EVENT_RE = re.compile(
    r"^[^\[].*?님(?:이|을)\s*.*?(?:들어왔|나갔|초대했|내보냈|나가셨|들어오셨)습니다\.?\s*$"
)


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #

# Korean date separator, e.g. `--------------- 2026년 2월 19일 목요일 ---------------`.
# Captures the y/m/d so we can render an ISO-8601 timestamp on each yielded msg.
_DATE_PARTS_RE = re.compile(
    r"(?P<year>\d{4})년\s+(?P<month>\d{1,2})월\s+(?P<day>\d{1,2})일"
)


def _compose_timestamp(
    date_context: Optional[str], period: str, hour: int, minute: int
) -> str:
    """Render an ISO-8601 timestamp from a Korean kakao header.

    All seven corpus files place a `--------------- YYYY년 M월 D일 ---------------`
    separator before any message line, so ``date_context`` is set by the time
    we render a header. When the upstream file is malformed (header before
    separator) we drop the date and emit the ISO-8601 time-only form
    ``THH:MM:SS`` so downstream consumers can detect the missing date instead
    of silently inheriting a fake one.
    """
    h = int(hour) % 12
    if period == "오후":
        h += 12
    time_part = f"{h:02d}:{int(minute):02d}:00"
    if date_context:
        m = _DATE_PARTS_RE.search(date_context)
        if m:
            return (
                f"{int(m.group('year')):04d}-"
                f"{int(m.group('month')):02d}-"
                f"{int(m.group('day')):02d}T{time_part}"
            )
    return f"T{time_part}"


def _speaker_id(redacted_user_token: str) -> str:
    """Strip surrounding brackets from a redacted user handle.

    The parser feeds ``[<name>]`` through the PII redactor, which returns
    ``[user_NN]``. Schema's ``speaker_id`` carries the bare ``user_NN`` form so
    downstream consumers (classifier, inventory, anchor_emit) don't carry the
    bracket noise.
    """
    s = redacted_user_token.strip()
    if s.startswith("[") and s.endswith("]"):
        return s[1:-1]
    return s


@dataclass
class ParseStats:
    """Per-line-category counters. Sum must equal `total_lines`."""

    total_lines: int = 0
    preamble_lines: int = 0
    separator_lines: int = 0
    message_header_lines: int = 0
    continuation_lines: int = 0
    blank_lines: int = 0
    event_lines: int = 0
    other_lines: int = 0

    def classified(self) -> int:
        return (
            self.preamble_lines
            + self.separator_lines
            + self.message_header_lines
            + self.continuation_lines
            + self.blank_lines
            + self.event_lines
            + self.other_lines
        )


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


class KakaoParser:
    """Iterable parser exposing `KakaoMsg`s and a `ParseStats` snapshot.

    Designed for streaming use:

        parser = KakaoParser(path, name_map=name_map)
        for msg in parser:
            ...
        parser.stats  # populated by then
    """

    def __init__(
        self,
        path: Path,
        *,
        name_map: dict[str, str],
        redactor: Callable[
            [str, dict[str, str], Optional[RedactionTrace]],
            tuple[str, RedactionTrace],
        ] = redact_kakao_text,
        trace: Optional[RedactionTrace] = None,
    ) -> None:
        self.path = Path(path)
        self.name_map = name_map
        self._redactor = redactor
        self.trace = trace if trace is not None else RedactionTrace()
        self.stats = ParseStats()

    # ------------------------------ iteration ------------------------------ #

    def __iter__(self) -> Iterator[KakaoMsg]:
        pending: Optional[KakaoMsg] = None
        date_context: Optional[str] = None
        next_idx = 0

        with self.path.open("r", encoding="utf-8", errors="replace") as fh:
            # --------------------------------------------------------------- #
            # MAJOR-3: shape-checked preamble.
            #
            # Buffer the first _PREAMBLE_LINES lines and only drop them if
            # they match the recognized KakaoTalk preamble shape. Files
            # exported in English, files whose preamble was already stripped,
            # or otherwise malformed inputs are parsed from line 0 (with a
            # stderr warning) so we don't silently lose data.
            # --------------------------------------------------------------- #
            buffered: list[str] = []
            for _ in range(_PREAMBLE_LINES):
                try:
                    buffered.append(next(fh))
                except StopIteration:
                    break

            if _looks_like_preamble(buffered):
                # Standard preamble: drop the buffered lines, record the
                # group_name digest from line 0 if not already set.
                for idx, raw_line in enumerate(buffered):
                    self.stats.total_lines += 1
                    self.stats.preamble_lines += 1
                    if idx == 0:
                        line0 = raw_line.rstrip("\r\n").strip()
                        m = _GROUP_NAME_RE.match(line0)
                        if m and not self.trace.group_name_sha256:
                            digest = hashlib.sha256(
                                m.group(1).strip().encode("utf-8")
                            ).hexdigest()[:12]
                            self.trace.group_name_sha256 = digest
                rest = fh
            else:
                # Non-standard preamble: log + parse from index 0.
                print(
                    f"[kakao_parser] non-standard preamble at {self.path}: "
                    "drop preamble disabled, lines from index 0 will be parsed",
                    file=sys.stderr,
                )
                rest = chain(buffered, fh)

            for raw_line in rest:
                self.stats.total_lines += 1
                line = raw_line.rstrip("\r\n")

                # 2. Date separator.
                if _SEPARATOR_RE.match(line):
                    if pending is not None:
                        yield pending
                        pending = None
                    date_context = line.strip()
                    self.stats.separator_lines += 1
                    continue

                # 3. Message header.
                mh = _MESSAGE_HEADER_RE.match(line)
                if mh:
                    if pending is not None:
                        yield pending
                    self.stats.message_header_lines += 1
                    raw_user_token = f"[{mh.group('user')}]"
                    raw_body = mh.group("msg")
                    redacted_user_token, _ = self._redactor(
                        raw_user_token, self.name_map, self.trace
                    )
                    redacted_body, _ = self._redactor(
                        raw_body, self.name_map, self.trace
                    )
                    pending = KakaoMsg(
                        msg_id=f"msg_{next_idx:06d}",
                        speaker_id=_speaker_id(redacted_user_token),
                        timestamp=_compose_timestamp(
                            date_context,
                            mh.group("period"),
                            int(mh.group("hh")),
                            int(mh.group("mm")),
                        ),
                        raw_text=raw_body,
                        redacted_text=redacted_body,
                    )
                    next_idx += 1
                    continue

                # 4. Blank line.
                if not line.strip():
                    if pending is not None:
                        pending.raw_text += "\n"
                        pending.redacted_text += "\n"
                        self.stats.continuation_lines += 1
                    else:
                        self.stats.blank_lines += 1
                    continue

                # 5. System event.
                if _EVENT_RE.match(line):
                    if pending is not None:
                        yield pending
                        pending = None
                    self.stats.event_lines += 1
                    continue

                # 6. Continuation of an open message, else orphan.
                if pending is not None:
                    redacted, _ = self._redactor(line, self.name_map, self.trace)
                    pending.raw_text += "\n" + line
                    pending.redacted_text += "\n" + redacted
                    self.stats.continuation_lines += 1
                else:
                    self.stats.other_lines += 1

        if pending is not None:
            yield pending


def parse_kakao_file(
    path: Path,
    *,
    name_map: dict[str, str],
    redactor: Callable[
        [str, dict[str, str], Optional[RedactionTrace]],
        tuple[str, RedactionTrace],
    ] = redact_kakao_text,
    trace: Optional[RedactionTrace] = None,
) -> Iterator[KakaoMsg]:
    """Convenience generator — wraps `KakaoParser` and yields its messages.

    For stats access, instantiate `KakaoParser` directly.
    """
    parser = KakaoParser(
        path, name_map=name_map, redactor=redactor, trace=trace
    )
    yield from parser


__all__ = [
    "KakaoMsg",
    "ParseStats",
    "KakaoParser",
    "parse_kakao_file",
]
