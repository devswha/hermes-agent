"""
dgmh/glm_data/medical_groups.py — medical-domain kakao group detector (Plan v2.1 §4 Step 3).

Public API:

    is_medical_group(kakao_file: Path) -> bool
        Reads the preamble of `kakao_file`, extracts the group name from the
        `<group_name> 님과 카카오톡 대화` line, and returns True iff the group
        name contains any keyword in `MEDICAL_KEYWORDS`.

    find_medical_files(corpus_dir: Path) -> set[Path]
        Convenience: scans `corpus_dir/KakaoTalk_*.txt` and returns the set
        of files for which `is_medical_group` is True.

Default scope of the v1 pipeline EXCLUDES files returned here; the operator
opts in via `--include-medical-groups`.
"""

from __future__ import annotations

import re
from pathlib import Path

MEDICAL_KEYWORDS: tuple[str, ...] = (
    # Original keyword set (Plan v2.1 §4 Step 3, original draft).
    "의료",
    "의약",
    "병원",
    "약사",
    "의대",
    "한의원",
    # Codex review MAJOR-2 expansion (Plan v2.1 AC1b): plan-required keywords
    # missing from the original set. Without these a group titled e.g.
    # "환자 케이스 공유" would not be excluded by default.
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

_GROUP_NAME_RE = re.compile(r"^(.+?)\s*님과 카카오톡 대화")
_PREAMBLE_SCAN_LINES = 5


def _read_group_name(path: Path) -> str | None:
    """Return the group-name token from the kakao preamble, if any.

    Scans the first `_PREAMBLE_SCAN_LINES` lines (defensive: the actual
    preamble is 3 lines for every file in the operator corpus, but the
    upstream plan documents 5).
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= _PREAMBLE_SCAN_LINES:
                    break
                m = _GROUP_NAME_RE.match(line.strip())
                if m:
                    return m.group(1).strip()
    except OSError:
        return None
    return None


def is_medical_group(kakao_file: Path) -> bool:
    """True iff the kakao group name contains a medical keyword."""
    name = _read_group_name(kakao_file)
    if not name:
        return False
    return any(kw in name for kw in MEDICAL_KEYWORDS)


def find_medical_files(corpus_dir: Path) -> set[Path]:
    """Return the set of `KakaoTalk_*.txt` paths flagged as medical."""
    out: set[Path] = set()
    if not corpus_dir.is_dir():
        return out
    for p in sorted(corpus_dir.glob("KakaoTalk_*.txt")):
        if is_medical_group(p):
            out.add(p)
    return out


__all__ = [
    "MEDICAL_KEYWORDS",
    "is_medical_group",
    "find_medical_files",
]
