"""Step 7 (v3) — two-tier persona pinning for SOUL.md evolution.

When the soul-evolution loop produces a candidate SOUL.md, run it through
this module BEFORE atomically replacing the live ``~/.hermes/SOUL.md``.

Two tiers:

  PERSONA-IDENTITY-START..PERSONA-IDENTITY-END  (HARD-PIN)
    The modifier may not mutate this block. If the candidate's IDENTITY
    block differs byte-for-byte from the parent's, force-replace it with
    the parent's block and log the drift event. If the candidate has 0
    or 2+ IDENTITY-START markers, reject the candidate outright (the
    structure of the SOUL is too damaged to safely accept).

  PERSONA-VOICE-START..PERSONA-VOICE-END  (SOFT-PIN)
    The modifier may tune content inside this block, but numeric
    ``target: NN%`` lines must satisfy 55 ≤ N ≤ 85. Any out-of-bounds
    value is auto-corrected to the nearest in-bounds (clamped) and the
    drift event is logged.

Drift log: ``~/.hermes/dgmh/persona_drift.jsonl`` (one JSON per line).
Schema: {"recorded_at", "kind"="identity"|"voice", "event", "before",
         "after", "parent_hash", "child_hash"}.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Marker strings — plan v3 spec.
IDENTITY_START_MARKER = "<!-- PERSONA-IDENTITY-START"
IDENTITY_END_MARKER = "<!-- PERSONA-IDENTITY-END -->"
VOICE_START_MARKER = "<!-- PERSONA-VOICE-START"
VOICE_END_MARKER = "<!-- PERSONA-VOICE-END -->"

# AC10: 55 ≤ N ≤ 85 for any ``target: NN%`` line inside the VOICE block.
VOICE_TARGET_MIN = 55
VOICE_TARGET_MAX = 85

# ``target: NN%`` — captures the integer between ``target:`` and ``%``.
# Spaces around the ``:`` and the digits are flexible; the regex pins
# the suffix to ``%`` so percentage-shaped tunables are matched while
# free-form ``target: someText`` lines aren't.
_TARGET_PERCENT_RE = re.compile(r"target:\s*(\d+)\s*%")


class PersonaPinError(ValueError):
    """Raised when the candidate's persona structure is unrecoverably broken
    (e.g. 0 or 2+ IDENTITY markers) so the candidate must be rejected
    rather than auto-corrected."""


@dataclasses.dataclass
class DriftEvent:
    recorded_at: str
    kind: str  # "identity" or "voice"
    event: str  # "force-replace" / "out-of-bounds" / "marker-missing"
    before: str
    after: str
    parent_hash: str
    child_hash: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _persona_drift_log_path() -> Path:
    home = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
    return home / "dgmh" / "persona_drift.jsonl"


def _append_drift_event(event: DriftEvent, log_path: Optional[Path] = None) -> None:
    target = log_path or _persona_drift_log_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dataclasses.asdict(event), ensure_ascii=False)
    with open(target, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _extract_block(
    text: str, start_marker: str, end_marker: str
) -> Optional[str]:
    """Return the substring from the first occurrence of ``start_marker``
    through the matching ``end_marker`` (inclusive). Returns None if
    either marker is missing."""
    start = text.find(start_marker)
    if start == -1:
        return None
    end = text.find(end_marker, start)
    if end == -1:
        return None
    return text[start : end + len(end_marker)]


def _count_marker(text: str, marker: str) -> int:
    """Count how many times ``marker`` appears in ``text``."""
    return text.count(marker)


def _enforce_identity_block(
    parent: str,
    child: str,
    *,
    parent_hash: str,
    child_hash: str,
    log_path: Optional[Path] = None,
) -> tuple[str, list[DriftEvent]]:
    """Hard-pin: child's IDENTITY block must match parent's exactly.

    Raises PersonaPinError if child has 0 or 2+ IDENTITY-START markers.
    If parent has no IDENTITY block, returns child unchanged.
    """
    drift: list[DriftEvent] = []
    parent_block = _extract_block(parent, IDENTITY_START_MARKER, IDENTITY_END_MARKER)
    if parent_block is None:
        # Parent never had an IDENTITY block — nothing to pin.
        return child, drift

    start_count = _count_marker(child, IDENTITY_START_MARKER)
    end_count = _count_marker(child, IDENTITY_END_MARKER)
    if start_count != 1 or end_count != 1:
        raise PersonaPinError(
            f"PERSONA-IDENTITY markers malformed: start={start_count}, "
            f"end={end_count}; expected exactly 1 of each"
        )

    child_block = _extract_block(child, IDENTITY_START_MARKER, IDENTITY_END_MARKER)
    if child_block is None:
        raise PersonaPinError(
            "PERSONA-IDENTITY-START present without matching END marker"
        )

    if child_block == parent_block:
        return child, drift

    # Force-replace: keep parent's IDENTITY byte-for-byte. Surround with
    # the existing pre/post text from child so the rest of the candidate
    # is preserved.
    revised = child.replace(child_block, parent_block, 1)
    drift.append(
        DriftEvent(
            recorded_at=_now(),
            kind="identity",
            event="force-replace",
            before=child_block,
            after=parent_block,
            parent_hash=parent_hash,
            child_hash=child_hash,
        )
    )
    _append_drift_event(drift[-1], log_path)
    logger.info(
        "[persona_pinning] IDENTITY drift detected; force-replaced with parent"
    )
    return revised, drift


def _clamp_target(n: int) -> int:
    return max(VOICE_TARGET_MIN, min(VOICE_TARGET_MAX, n))


def _enforce_voice_block(
    text: str,
    *,
    parent_hash: str,
    child_hash: str,
    log_path: Optional[Path] = None,
) -> tuple[str, list[DriftEvent]]:
    """Soft-pin: clamp every ``target: NN%`` line inside the VOICE block to
    [55, 85]. Out-of-bounds values are auto-corrected to the nearest
    in-bounds value and logged."""
    drift: list[DriftEvent] = []
    voice_block = _extract_block(text, VOICE_START_MARKER, VOICE_END_MARKER)
    if voice_block is None:
        return text, drift

    revised_block = voice_block

    def _replace(match: re.Match) -> str:
        n = int(match.group(1))
        if VOICE_TARGET_MIN <= n <= VOICE_TARGET_MAX:
            return match.group(0)
        clamped = _clamp_target(n)
        before_str = match.group(0)
        after_str = f"target: {clamped}%"
        drift.append(
            DriftEvent(
                recorded_at=_now(),
                kind="voice",
                event="out-of-bounds",
                before=before_str,
                after=after_str,
                parent_hash=parent_hash,
                child_hash=child_hash,
            )
        )
        return after_str

    revised_block = _TARGET_PERCENT_RE.sub(_replace, revised_block)

    if revised_block == voice_block:
        return text, drift

    # Persist drift events to the log AFTER the rewrite finishes so any
    # log-write failure doesn't corrupt the in-memory state.
    for evt in drift:
        try:
            _append_drift_event(evt, log_path)
        except Exception:
            logger.exception(
                "[persona_pinning] failed to append voice drift event"
            )

    revised = text.replace(voice_block, revised_block, 1)
    logger.info(
        "[persona_pinning] VOICE drift: clamped %d out-of-bounds target line(s)",
        len(drift),
    )
    return revised, drift


def enforce_persona_invariants(
    parent_soul_md: str,
    child_soul_md: str,
    *,
    parent_hash: str = "",
    child_hash: str = "",
    log_path: Optional[Path] = None,
) -> tuple[str, list[DriftEvent]]:
    """Apply both tiers of persona pinning to a candidate SOUL.md.

    Returns the revised candidate text and the full list of drift
    events. Raises ``PersonaPinError`` on structural failure (missing /
    duplicate IDENTITY markers) — caller should reject the candidate.
    """
    revised, identity_events = _enforce_identity_block(
        parent_soul_md,
        child_soul_md,
        parent_hash=parent_hash,
        child_hash=child_hash,
        log_path=log_path,
    )
    revised, voice_events = _enforce_voice_block(
        revised,
        parent_hash=parent_hash,
        child_hash=child_hash,
        log_path=log_path,
    )
    return revised, identity_events + voice_events
