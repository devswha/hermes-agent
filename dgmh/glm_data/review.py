"""Review report + approval CLI verbs for DGM-H GLM data v1 (Step 11).

Plan reference: ``.omc/plans/dgmh-glm-data-v1.md`` §4 Step 11, AC12, V9, H4, M3.

This module owns the operator-trust gate between the staging artifacts
(classifications, patina patches, anchor candidates) and the eventual
``apply-dryrun`` step. Four verbs (``approve``, ``reject``, ``defer``,
``status``) plus a report renderer cover all of AC12.

All persistent state lives under the per-session staging root via
``StagingWriter``:

* ``reports/review_report.md`` — human-readable rendering of ≥10 samples.
* ``approvals.jsonl`` — append-only ledger of every verb invocation.
* ``checks/review_warnings.jsonl`` — sub-``min_review_seconds`` approvals
  recorded as warnings (operator-trust: warning, not block — D6 fix).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from .staging_writer import StagingWriter
from .taxonomy_loader import promote as _promote_category

# Default minimum review time per sample (seconds). Operator may override
# via CLI flag; sub-threshold approvals warn but do not block.
DEFAULT_MIN_REVIEW_SECONDS = 60

# Required approve count before ``apply-dryrun`` is allowed (AC12 / V9).
REQUIRED_APPROVALS = 10


# ----------------------------------------------------------------- data shapes


@dataclass
class ReviewSample:
    """One row in ``review_report.md``."""

    sample_id: str
    source: str  # "kakao" | "youtube" | "patina" | "anchor"
    text: str  # already redacted
    classification: str  # category name (baseline or "_new_<slug>")
    redaction_trace: dict = field(default_factory=dict)
    rationale: str = ""
    target_file: str = ""


# --------------------------------------------------------------- error classes


class ReviewError(RuntimeError):
    """Base for review module failures."""


class ApplyDryrunBlocked(ReviewError):
    """Raised when ``apply-dryrun`` is gated by < REQUIRED_APPROVALS."""


class UnknownDecisionError(ReviewError):
    """Raised when a decision string outside the verb set is passed."""


# --------------------------------------------------------------- file paths


REPORT_RELPATH = "reports/review_report.md"
APPROVALS_RELPATH = "approvals.jsonl"
WARNINGS_RELPATH = "checks/review_warnings.jsonl"


# ----------------------------------------------------------------- core class


class ReviewManager:
    """Generate the review report and append approval verbs.

    The manager is stateless across calls except via the files it reads
    and writes. Construct one per ``StagingWriter`` (one per session).
    """

    def __init__(
        self,
        writer: StagingWriter,
        *,
        min_review_seconds: int = DEFAULT_MIN_REVIEW_SECONDS,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._writer = writer
        self._min_review_seconds = min_review_seconds
        self._now = now

    # ------------------------------------------------------------ properties

    @property
    def _root(self) -> Path:
        return self._writer.session_root

    @property
    def report_path(self) -> Path:
        return self._root / REPORT_RELPATH

    @property
    def approvals_path(self) -> Path:
        return self._root / APPROVALS_RELPATH

    @property
    def warnings_path(self) -> Path:
        return self._root / WARNINGS_RELPATH

    # ------------------------------------------------------------ report gen

    def generate_report(self, samples: list[ReviewSample]) -> Path:
        """Render ``review_report.md`` (≥10 samples; less is allowed but flagged).

        The file's mtime is set to ``self._now()`` after write, so
        ``review_duration_s`` reflects a single canonical clock (the same
        ``now`` callable used for entry timestamps). This makes tests fully
        deterministic and keeps the audit trail consistent on disk.
        """

        lines: list[str] = []
        lines.append("# DGM-H GLM Data v1 — Review Report")
        lines.append("")
        lines.append(f"Samples: **{len(samples)}** (required ≥ {REQUIRED_APPROVALS})")
        lines.append("")
        lines.append(
            "Approve via `python -m dgmh.glm_data review --approve <sample_id>`."
        )
        lines.append("")
        for sample in samples:
            lines.append(f"## {sample.sample_id}")
            lines.append("")
            lines.append(f"- **source**: {sample.source}")
            lines.append(f"- **classification**: {sample.classification}")
            lines.append(f"- **target_file**: {sample.target_file or '_n/a_'}")
            redaction_blob = (
                json.dumps(sample.redaction_trace, ensure_ascii=False, sort_keys=True)
                if sample.redaction_trace
                else "{}"
            )
            lines.append(f"- **redaction_trace**: `{redaction_blob}`")
            lines.append("")
            lines.append("**Redacted text**:")
            lines.append("")
            lines.append("```")
            lines.append(sample.text)
            lines.append("```")
            lines.append("")
            if sample.rationale:
                lines.append(f"**Rationale**: {sample.rationale}")
                lines.append("")
        path = self._writer.write_text(REPORT_RELPATH, "\n".join(lines) + "\n")
        import os as _os

        ts = self._now()
        _os.utime(path, (ts, ts))
        return path

    # ------------------------------------------------------------ verb helpers

    def _report_mtime(self) -> Optional[float]:
        if not self.report_path.is_file():
            return None
        return self.report_path.stat().st_mtime

    def _read_jsonl(self, path: Path) -> list[dict]:
        if not path.is_file():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _append_jsonl(self, rel: str, entry: dict) -> None:
        """Append one JSON row to ``rel`` (creates parent dir on first write)."""

        path = self._root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        # Validate the path is under the staging root via StagingWriter's guard
        # — but we need raw append, not rewrite. Sanity-check by resolving.
        resolved = path.resolve()
        resolved.relative_to(self._root)  # raises if escape
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True))
            fh.write("\n")

    def _record(
        self,
        *,
        sample_id: str,
        decision: str,
        reviewer: str,
        notes_or_reason: str,
        sample_classification: Optional[str] = None,
    ) -> dict:
        """Append one entry to ``approvals.jsonl``.

        Warns (does NOT block) when ``review_duration_s < min_review_seconds``
        on an approve decision. If the approved sample carries a ``_new_*``
        classification, also promotes it via ``taxonomy_loader.promote``.
        """

        if decision not in {"approve", "reject", "defer"}:
            raise UnknownDecisionError(
                f"decision must be approve|reject|defer; got {decision!r}"
            )

        ts = self._now()
        mtime = self._report_mtime()
        review_duration_s = (
            None if mtime is None else max(0.0, round(ts - mtime, 3))
        )

        entry = {
            "sample_id": sample_id,
            "decision": decision,
            "ts": ts,
            "reviewer": reviewer,
            "notes_or_reason": notes_or_reason,
            "review_duration_s": review_duration_s,
        }
        self._append_jsonl(APPROVALS_RELPATH, entry)

        # Sub-threshold approval → warning row (audit-traceable, not blocking).
        if (
            decision == "approve"
            and review_duration_s is not None
            and review_duration_s < self._min_review_seconds
        ):
            self._append_jsonl(
                WARNINGS_RELPATH,
                {
                    "sample_id": sample_id,
                    "reviewer": reviewer,
                    "review_duration_s": review_duration_s,
                    "min_review_seconds": self._min_review_seconds,
                    "ts": ts,
                },
            )

        # Promotion hook: approving a _new_* classification promotes the
        # category to the cross-run taxonomy ledger (Step 5b).
        if (
            decision == "approve"
            and sample_classification
            and sample_classification.startswith("_new_")
        ):
            _promote_category(sample_classification, self._root)

        return entry

    # ----------------------------------------------------------------- verbs

    def approve(
        self,
        sample_id: str,
        *,
        reviewer: str = "operator",
        notes: str = "",
        sample_classification: Optional[str] = None,
    ) -> dict:
        return self._record(
            sample_id=sample_id,
            decision="approve",
            reviewer=reviewer,
            notes_or_reason=notes,
            sample_classification=sample_classification,
        )

    def reject(
        self,
        sample_id: str,
        *,
        reviewer: str = "operator",
        reason: str,
    ) -> dict:
        if not reason:
            raise ValueError("reject requires a non-empty reason (AC12)")
        return self._record(
            sample_id=sample_id,
            decision="reject",
            reviewer=reviewer,
            notes_or_reason=reason,
        )

    def defer(
        self,
        sample_id: str,
        *,
        reviewer: str = "operator",
    ) -> dict:
        return self._record(
            sample_id=sample_id,
            decision="defer",
            reviewer=reviewer,
            notes_or_reason="",
        )

    # ---------------------------------------------------------------- status

    def status(self) -> dict:
        """Return counts by decision plus a list of pending samples.

        ``pending`` is computed as ``sample_ids_in_report − decided``; if no
        report has been generated yet, ``pending`` is an empty list.
        """

        entries = self._read_jsonl(self.approvals_path)
        approved = [e for e in entries if e["decision"] == "approve"]
        rejected = [e for e in entries if e["decision"] == "reject"]
        deferred = [e for e in entries if e["decision"] == "defer"]

        decided_ids: set[str] = {e["sample_id"] for e in entries}
        all_ids: list[str] = self._sample_ids_in_report()
        pending: list[str] = [sid for sid in all_ids if sid not in decided_ids]

        return {
            "approved": len(approved),
            "rejected": len(rejected),
            "deferred": len(deferred),
            "required": REQUIRED_APPROVALS,
            "pending": pending,
            "can_apply_dryrun": len(approved) >= REQUIRED_APPROVALS,
        }

    def _sample_ids_in_report(self) -> list[str]:
        if not self.report_path.is_file():
            return []
        ids: list[str] = []
        for line in self.report_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                ids.append(line[3:].strip())
        return ids

    # ------------------------------------------------------------ apply gate

    def check_apply_dryrun(self) -> None:
        """Raise ``ApplyDryrunBlocked`` if approved count < REQUIRED_APPROVALS."""

        st = self.status()
        if not st["can_apply_dryrun"]:
            raise ApplyDryrunBlocked(
                f"apply-dryrun blocked: approved={st['approved']} "
                f"required={REQUIRED_APPROVALS} (AC12)"
            )


__all__ = [
    "ReviewManager",
    "ReviewSample",
    "ApplyDryrunBlocked",
    "UnknownDecisionError",
    "REQUIRED_APPROVALS",
    "DEFAULT_MIN_REVIEW_SECONDS",
    "REPORT_RELPATH",
    "APPROVALS_RELPATH",
    "WARNINGS_RELPATH",
]
