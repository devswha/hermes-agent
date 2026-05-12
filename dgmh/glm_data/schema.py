"""Dataclasses for DGM-H GLM data collection v1.

The schema covers the minimum surface area needed by Step 1 (staging contract)
and the immediate downstream consumers (PII redaction trace, classifier).
Bigger structures (gap rows, patina patches, anchors) live in later steps and
will land alongside their owning modules — keeping this file small keeps the
contract reviewable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

# Baseline 7 tone categories — locked at plan §3 AC5. The classifier may also
# emit ``_new_<slug>`` discovery proposals; we accept those as raw strings and
# validate the ``_new_`` prefix in ``ToneCategory.__post_init__``.
BaselineToneName = Literal[
    "communication",
    "content",
    "filler",
    "language",
    "structure",
    "style",
    "viral-hook",
]

BASELINE_TONE_NAMES: tuple[str, ...] = (
    "communication",
    "content",
    "filler",
    "language",
    "structure",
    "style",
    "viral-hook",
)


@dataclass(frozen=True)
class ToneCategory:
    """A tone-category label paired with the patina file that owns it.

    ``name`` is either one of the seven baseline categories (see
    ``BASELINE_TONE_NAMES``) or a ``_new_<slug>`` discovery proposal. Any other
    string raises ``ValueError`` so we never silently widen the taxonomy.
    """

    name: str
    target_file: str

    def __post_init__(self) -> None:
        if self.name in BASELINE_TONE_NAMES:
            return
        if self.name.startswith("_new_") and len(self.name) > len("_new_"):
            return
        raise ValueError(
            f"ToneCategory.name must be one of {BASELINE_TONE_NAMES} or "
            f"start with '_new_'; got {self.name!r}"
        )

    @property
    def is_new_proposal(self) -> bool:
        return self.name.startswith("_new_")


@dataclass
class KakaoMsg:
    """A single parsed KakaoTalk message line.

    ``redacted_text`` is the only field downstream consumers should serialize;
    ``raw_text`` stays in memory for the redaction trace audit but never
    crosses an I/O boundary.
    """

    msg_id: str
    speaker_id: str
    timestamp: str
    raw_text: str
    redacted_text: str


@dataclass
class CandidateMessage:
    """A redacted message that is a candidate for staging output.

    Carries enough context for the classifier and inventory step without
    leaking PII or raw speaker handles.
    """

    msg_id: str
    redacted_text: str
    speaker_token: str
    source_file: str
    timestamp: str


@dataclass
class RedactionTrace:
    """Per-file PII redaction accounting.

    ``by_category`` counts how many spans were redacted under each PII
    category (``phone``, ``email``, ``korean_name``, ``rrn``, ...). The group
    name is stored only as a SHA-256 prefix so debug logs cannot identify the
    chat room.
    """

    by_category: dict[str, int] = field(default_factory=dict)
    group_name_sha256: str = ""


@dataclass
class Classification:
    """Classifier output for a single Kakao message."""

    msg_id: str
    categories: list[ToneCategory]
    confidence: float
    new_proposal: Optional[dict] = None
