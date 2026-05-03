"""
dgmh/run_log.py — DGM-H per-run persistence (W4 deliverable).

Port note: Python port of playground/dgmh-engine/runLog.ts.
Records one line per run_one_iteration() call into
~/.hermes/dgmh/runs.jsonl (JSON Lines, append-only).

Format: append-only newline-delimited JSON. Each line is one RunRecord.
Schema versioned (schema_version=1) for future migration.

The maturity-check script (dgmh:maturity) reads the file as a streaming
source for the gate metrics described in gen-0003 §Phase 1.6 maturity gate.

Reference: devswha/dgmh @ playground/dgmh-engine/runLog.ts
           hermes-skill-archive-dgmh-plan-v2-addendum.md §CONCERN 4 (W4)
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _dgmh_dir() -> Path:
    """~/.hermes/dgmh/ — DGM-H state home."""
    hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(hermes_home) / "dgmh"


def default_runs_path() -> Path:
    """Default path for runs.jsonl: ~/.hermes/dgmh/runs.jsonl"""
    return _dgmh_dir() / "runs.jsonl"


# ---------------------------------------------------------------------------
# Dataclasses — mirrors runLog.ts interfaces
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RunRecordChild:
    """Accepted child admitted to the archive in one iteration.

    Mirrors runLog.ts::RunRecordChild.
    """

    id: str
    parent_id: str
    score: float
    generation_index: int
    lineage_depth: int


@dataclasses.dataclass
class RunRecordRejection:
    """One rejected candidate from one iteration.

    Mirrors runLog.ts::RunRecordRejection.
    reason: 'modifier-error' | 'critic-error' | 'critic-reject'
    message: truncated to 200 chars for log size control.
    """

    reason: Literal["modifier-error", "critic-error", "critic-reject"]
    parent_id: str
    message: str  # truncated to 200 chars by append_run_record


@dataclasses.dataclass
class RunRecordBiasSummary:
    """Bias summary placeholder until W5.

    Mirrors runLog.ts::RunRecordBiasSummary.
    W4: all-zero counts are acceptable (W5 wires the bias detector).
    """

    inspected_generations: int
    signal_count: int
    # Severity histogram: info / concern / block counts
    severities: dict[str, int] = dataclasses.field(
        default_factory=lambda: {"info": 0, "concern": 0, "block": 0}
    )


@dataclasses.dataclass
class RunRecord:
    """One DGM-H iteration run record.

    Mirrors runLog.ts::RunRecord.
    schema_version=1 for all W4 records.
    """

    schema_version: int
    recorded_at: str  # ISO 8601
    seed: int
    archive_size_pre: int
    archive_size_post: int
    accepted_children: list[RunRecordChild]
    rejections: list[RunRecordRejection]
    bias: RunRecordBiasSummary


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _record_to_dict(record: RunRecord) -> dict[str, Any]:
    """Serialize RunRecord to a JSON-serializable dict."""
    return {
        "schema_version": record.schema_version,
        "recorded_at": record.recorded_at,
        "seed": record.seed,
        "archive_size_pre": record.archive_size_pre,
        "archive_size_post": record.archive_size_post,
        "accepted_children": [
            {
                "id": c.id,
                "parent_id": c.parent_id,
                "score": c.score,
                "generation_index": c.generation_index,
                "lineage_depth": c.lineage_depth,
            }
            for c in record.accepted_children
        ],
        "rejections": [
            {
                "reason": r.reason,
                "parent_id": r.parent_id,
                "message": r.message,
            }
            for r in record.rejections
        ],
        "bias": {
            "inspected_generations": record.bias.inspected_generations,
            "signal_count": record.bias.signal_count,
            "severities": dict(record.bias.severities),
        },
    }


def _dict_to_record(d: dict[str, Any]) -> RunRecord | None:
    """Deserialize a dict to RunRecord. Returns None if schema invalid.

    Defensive parse: skips records with wrong schema_version or missing fields.
    Mirrors runLog.ts::isRunRecord.
    """
    if not isinstance(d, dict):
        return None
    if d.get("schema_version") != 1:
        return None
    try:
        children = [
            RunRecordChild(
                id=c["id"],
                parent_id=c["parent_id"],
                score=float(c["score"]),
                generation_index=int(c["generation_index"]),
                lineage_depth=int(c["lineage_depth"]),
            )
            for c in (d.get("accepted_children") or [])
        ]
        rejections = [
            RunRecordRejection(
                reason=r["reason"],
                parent_id=r["parent_id"],
                message=str(r.get("message", "")),
            )
            for r in (d.get("rejections") or [])
        ]
        bias_raw = d.get("bias") or {}
        severities_raw = bias_raw.get("severities") or {}
        bias = RunRecordBiasSummary(
            inspected_generations=int(bias_raw.get("inspected_generations", 0)),
            signal_count=int(bias_raw.get("signal_count", 0)),
            severities={
                "info": int(severities_raw.get("info", 0)),
                "concern": int(severities_raw.get("concern", 0)),
                "block": int(severities_raw.get("block", 0)),
            },
        )
        return RunRecord(
            schema_version=1,
            recorded_at=str(d["recorded_at"]),
            seed=int(d["seed"]),
            archive_size_pre=int(d["archive_size_pre"]),
            archive_size_post=int(d["archive_size_post"]),
            accepted_children=children,
            rejections=rejections,
            bias=bias,
        )
    except (KeyError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def append_run_record(filepath: str | Path, record: RunRecord) -> None:
    """Atomically append one RunRecord to a JSONL file.

    Creates parent directories if missing (mirrors runLog.ts::appendRunRecord).
    Truncates rejection messages to 200 chars for log size control.
    Uses atomic write: tempfile in same directory + os.replace.

    Args:
        filepath: Path to the JSONL file (typically ~/.hermes/dgmh/runs.jsonl).
        record: RunRecord to append.
    """
    target = Path(filepath)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Truncate rejection messages before serializing
    truncated_rejections = [
        RunRecordRejection(
            reason=r.reason,
            parent_id=r.parent_id,
            message=r.message[:200] if len(r.message) > 200 else r.message,
        )
        for r in record.rejections
    ]
    record = dataclasses.replace(record, rejections=truncated_rejections)

    line = json.dumps(_record_to_dict(record), ensure_ascii=False) + "\n"

    # Atomic append: read existing + write all to tempfile + replace
    # For a pure append (no rewrite of existing), we use "a" mode which is
    # atomic enough for single-writer (POSIX append). Using a tempfile for
    # full atomicity would require rewriting the whole file, which is
    # inappropriate for a potentially large log. Standard append is used here,
    # consistent with runLog.ts::appendFile.
    with open(target, "a", encoding="utf-8") as f:
        f.write(line)

    logger.debug("run_log: appended record seed=%d to %s", record.seed, target)


def read_run_records(filepath: str | Path) -> list[RunRecord]:
    """Read all valid RunRecords from a JSONL file.

    Defensive parse: skips malformed lines and wrong schema_version.
    Returns empty list if file is missing (mirrors runLog.ts::readRunRecords).

    Args:
        filepath: Path to the JSONL file.

    Returns:
        List of valid RunRecord objects.
    """
    target = Path(filepath)
    if not target.exists():
        return []

    try:
        raw = target.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("run_log: could not read %s: %s", target, exc)
        return []

    records: list[RunRecord] = []
    for lineno, line in enumerate(raw.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            d = json.loads(stripped)
        except json.JSONDecodeError as exc:
            logger.debug("run_log: skipping malformed line %d: %s", lineno, exc)
            continue
        record = _dict_to_record(d)
        if record is None:
            logger.debug(
                "run_log: skipping invalid record at line %d (schema mismatch or missing fields)",
                lineno,
            )
            continue
        records.append(record)

    return records


def make_run_record(
    *,
    seed: int,
    archive_size_pre: int,
    archive_size_post: int,
    accepted_children: list[RunRecordChild] | None = None,
    rejections: list[RunRecordRejection] | None = None,
    bias: RunRecordBiasSummary | None = None,
) -> RunRecord:
    """Convenience constructor for RunRecord with defaults.

    Sets schema_version=1 and recorded_at=now automatically.
    """
    return RunRecord(
        schema_version=1,
        recorded_at=datetime.now(timezone.utc).isoformat(),
        seed=seed,
        archive_size_pre=archive_size_pre,
        archive_size_post=archive_size_post,
        accepted_children=accepted_children or [],
        rejections=rejections or [],
        bias=bias or summarize_bias_report(),
    )


def summarize_bias_report(*args: Any, **kwargs: Any) -> RunRecordBiasSummary:
    """Placeholder bias summarizer until W5.

    W5 will wire the real bias detector here. W4 returns a zero-counts struct
    so loop.py can call this without needing the W5 bias module.

    Mirrors runLog.ts::summarizeBiasReport stub behavior for W4.
    """
    return RunRecordBiasSummary(
        inspected_generations=0,
        signal_count=0,
        severities={"info": 0, "concern": 0, "block": 0},
    )
