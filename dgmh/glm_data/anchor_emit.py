"""Anchor candidate generator for DGM-H GLM data v1 (Step 10).

Plan reference: ``.omc/plans/dgmh-glm-data-v1.md`` §4 Step 10, AC10, AC15,
V4, V10, D-5, D-8, C2.

The emitter takes a stream of *already-PII-redacted* ``KakaoMsg`` objects
(from Step 3) and writes three staging artifacts under the per-session
``staging/anchor/`` directory:

* ``kakao_anchors_candidate.txt`` — one message per line, UTF-8, no header.
  **Every line passes the five ``_load_corpus`` filters** copied verbatim
  from ``hermes-agent/dgmh/kakao_style_retrieval.py:115-131`` (length ≥ 4,
  no ``@`` prefix, no ``[20`` prefix, no URL, non-empty after strip).
* ``kakao_anchors_candidate.jsonl`` — sidecar metadata for the *emitted*
  lines (``msg_id``, ``category``, ``redaction_trace``,
  ``register_mode_hint``). Review-only; ``kakao_style_retrieval`` does
  **not** load this file.
* ``short_reaction_anchors.jsonl`` — diverted reactions (``ㄷㄷ``, ``ㅋㅋ``)
  with ``len < 4``. Forward-compat for v2; v1's ``_load_corpus`` will not
  consume these.

The canonical corpus at ``~/.hermes/dgmh/kakao_hako_corpus.txt`` is
**never opened, read, or modified** by this module (AC15 / D-8). All
writes route through ``StagingWriter`` which is path-guarded under
``~/.hermes/dgmh/glm_data_v1/staging/<session_id>/``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional

from .schema import KakaoMsg
from .staging_writer import StagingWriter

# Mirror the 5-filter regex from kakao_style_retrieval._load_corpus to avoid
# an import cycle at module load. (kakao_style_retrieval lives one package up
# and pulls in indexing state we don't need here.) The regex is identical;
# a separate test asserts the actual function accepts our output.
_URL_RE = re.compile(r"https?://|www\.")

# Per-message length cap from plan §4 Step 10 "Additional filters" line.
DEFAULT_LENGTH_CAP_CHARS = 200

# Per-run candidate cap from plan §4 Step 10.
DEFAULT_MAX_LINES = 500


# --------------------------------------------------------------- reasons


class DropReason:
    EMPTY = "empty_after_strip"
    AT_PREFIX = "at_prefix"
    HEADER_PREFIX = "header_prefix"
    URL = "contains_url"
    TOO_LONG = "exceeds_length_cap"
    DUPLICATE = "duplicate_normalized"
    SHORT_REACTION = "short_reaction_diverted"
    RUN_CAP = "run_cap_reached"


# ------------------------------------------------------------------ report


@dataclass
class EmitReport:
    """Summary returned by ``AnchorEmitter.emit`` for downstream verification."""

    txt_path: Path
    sidecar_path: Path
    short_reactions_path: Path
    emit_count: int
    short_reaction_count: int
    dropped: dict[str, int] = field(default_factory=dict)


# ------------------------------------------------------------------ helpers


def _passes_five_filters(line: str) -> bool:
    """Replicates ``kakao_style_retrieval._load_corpus`` line-acceptance test.

    Caller must pass the *already-stripped* line. Returns ``True`` iff
    ``_load_corpus`` would accept the line.
    """

    if not line:
        return False
    if _URL_RE.search(line):
        return False
    if line.startswith("@") or line.startswith("[20"):
        return False
    if len(line) < 4:
        return False
    return True


def _normalize_for_dedup(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().casefold())


# ------------------------------------------------------------------ emitter


class AnchorEmitter:
    """Writes anchor candidate files under a per-session staging root.

    The emitter is stateless across calls except for the deduplication set
    accumulated *within* a single ``emit`` invocation. Construct one
    emitter per ``StagingWriter`` (i.e., per session).
    """

    def __init__(
        self,
        writer: StagingWriter,
        *,
        max_lines: int = DEFAULT_MAX_LINES,
        length_cap_chars: int = DEFAULT_LENGTH_CAP_CHARS,
    ) -> None:
        self._writer = writer
        self._max_lines = max_lines
        self._length_cap = length_cap_chars

    # --- file-path conventions ---------------------------------------------------

    TXT_RELPATH = "anchor/kakao_anchors_candidate.txt"
    SIDECAR_RELPATH = "anchor/kakao_anchors_candidate.jsonl"
    SHORT_REACTIONS_RELPATH = "anchor/short_reaction_anchors.jsonl"

    # --- public API --------------------------------------------------------------

    def emit(
        self,
        messages: Iterable[KakaoMsg],
        *,
        classifications: Optional[Mapping[str, "object"]] = None,
        register_mode_hint: str = "mirror",
    ) -> EmitReport:
        """Filter, dedup, cap, and write three anchor files.

        ``classifications`` is a ``{msg_id -> Classification}`` map used to
        decorate the sidecar; missing entries fall back to ``unknown``.
        ``register_mode_hint`` is recorded per row so the review CLI can
        surface ``mirror`` vs ``public_haeyo`` candidates without
        re-classifying.
        """

        emitted_lines: list[str] = []
        sidecar_rows: list[dict] = []
        short_rows: list[dict] = []
        dropped: dict[str, int] = {}
        seen_norms: set[str] = set()

        def _bump(reason: str) -> None:
            dropped[reason] = dropped.get(reason, 0) + 1

        for msg in messages:
            if not isinstance(msg, KakaoMsg):
                raise TypeError(
                    f"AnchorEmitter expects KakaoMsg; got {type(msg).__name__}"
                )
            line = (msg.redacted_text or "").strip()
            if not line:
                _bump(DropReason.EMPTY)
                continue

            # Always divert short reactions FIRST (before any other drop) so the
            # `len < 4` sidecar captures `ㄷㄷ` / `ㅋㅋ` even when they would
            # also fail one of the other filters.
            if len(line) < 4:
                short_rows.append(
                    {
                        "msg_id": msg.msg_id,
                        "speaker_id": msg.speaker_id,
                        "timestamp": msg.timestamp,
                        "text": line,
                        "reason": DropReason.SHORT_REACTION,
                    }
                )
                _bump(DropReason.SHORT_REACTION)
                continue

            if line.startswith("@"):
                _bump(DropReason.AT_PREFIX)
                continue
            if line.startswith("[20"):
                _bump(DropReason.HEADER_PREFIX)
                continue
            if _URL_RE.search(line):
                _bump(DropReason.URL)
                continue
            if len(line) > self._length_cap:
                _bump(DropReason.TOO_LONG)
                continue

            norm = _normalize_for_dedup(line)
            if norm in seen_norms:
                _bump(DropReason.DUPLICATE)
                continue
            seen_norms.add(norm)

            if len(emitted_lines) >= self._max_lines:
                _bump(DropReason.RUN_CAP)
                continue

            # Re-assert the 5-filter on the final line. If this trips, our
            # logic above drifted from kakao_style_retrieval; fail loudly
            # rather than emit a malformed candidate.
            assert _passes_five_filters(line), (
                f"internal: emit candidate {line!r} fails 5-filter contract; "
                "kakao_style_retrieval._load_corpus would reject it."
            )
            emitted_lines.append(line)

            category = "unknown"
            if classifications:
                cls_obj = classifications.get(msg.msg_id)
                if cls_obj is not None and getattr(cls_obj, "categories", None):
                    cats = cls_obj.categories
                    if cats:
                        first = cats[0]
                        category = getattr(first, "name", str(first))
            sidecar_rows.append(
                {
                    "msg_id": msg.msg_id,
                    "speaker_id": msg.speaker_id,
                    "timestamp": msg.timestamp,
                    "text_sha256_12": hashlib.sha256(
                        line.encode("utf-8")
                    ).hexdigest()[:12],
                    "category": category,
                    "register_mode_hint": register_mode_hint,
                    "length": len(line),
                }
            )

        # ----- write the three artifacts atomically per file -----
        txt_payload = "\n".join(emitted_lines) + ("\n" if emitted_lines else "")
        txt_path = self._writer.write_text(self.TXT_RELPATH, txt_payload)
        sidecar_path = self._writer.write_jsonl(self.SIDECAR_RELPATH, sidecar_rows)
        short_path = self._writer.write_jsonl(self.SHORT_REACTIONS_RELPATH, short_rows)

        return EmitReport(
            txt_path=txt_path,
            sidecar_path=sidecar_path,
            short_reactions_path=short_path,
            emit_count=len(emitted_lines),
            short_reaction_count=len(short_rows),
            dropped=dict(dropped),
        )


# ------------------------------------------------------- post-emit verification


def verify_emitted_file_loads(candidate_txt: Path) -> int:
    """Run the *actual* ``kakao_style_retrieval._load_corpus`` on the file.

    Returns the loaded message count. Re-raises whatever ``_load_corpus``
    raises (``RuntimeError`` on empty / unusable corpus). Callers should
    assert ``return == emit_count`` for AC10.
    """

    # Import here to avoid a top-level cycle: kakao_style_retrieval pulls
    # logging and global state we don't want at module load.
    from dgmh.kakao_style_retrieval import _load_corpus  # noqa: WPS433

    corpus = _load_corpus(candidate_txt)
    return len(corpus.messages)


__all__ = [
    "AnchorEmitter",
    "EmitReport",
    "DropReason",
    "verify_emitted_file_loads",
    "DEFAULT_LENGTH_CAP_CHARS",
    "DEFAULT_MAX_LINES",
]


# ``asdict`` is exported indirectly via ``EmitReport`` consumers in the
# review CLI step; re-export the symbol so callers don't need a
# ``dataclasses`` import for the same purpose.
emit_report_asdict = asdict
