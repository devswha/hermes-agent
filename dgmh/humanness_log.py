"""Humanness telemetry log for DGM-H post-hoc scoring.

Every Discord assistant message scored by patina is appended to a JSONL log
at ``~/.hermes/dgmh/humanness_log.jsonl``. Evolution can read this log to
compute a rolling humanness fitness score per active SOUL.md generation,
feeding the composite reward signal alongside reactions and Codex-judge.

Schema (one JSON object per line):

    {
      "ts": "2026-05-04T17:30:00+00:00",
      "chat_id": "1496872245027541062",
      "thread_id": "1500574735732445396",
      "message_id": "1500780000000000000",
      "soul_md_hash": "890770f1...",
      "text_length": 142,
      "ai_score": 12.5,
      "human_likeness": 87.5,
      "sub_scores": {"communication": 0.0, ...},
      "interpretation": "human",
      "elapsed_s": 8.4
    }

When scoring fails the entry still gets written with ``"error"`` instead
of score fields, so the evolution loop can detect and handle gaps.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


_WRITE_LOCK = threading.Lock()


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


def humanness_log_path() -> Path:
    """Return the canonical log path; ensure parent exists."""
    path = _hermes_home() / "dgmh" / "humanness_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def append_record(record: dict[str, Any]) -> None:
    """Append a record dict as one JSON line. Thread-safe."""
    payload = json.dumps(record, ensure_ascii=False, sort_keys=False)
    path = humanness_log_path()
    with _WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(payload + "\n")


def make_success_record(
    *,
    chat_id: str,
    thread_id: Optional[str],
    message_id: Optional[str],
    soul_md_hash: str,
    text_length: int,
    ai_score: float,
    human_likeness: float,
    sub_scores: dict[str, float],
    interpretation: str,
    elapsed_s: float,
) -> dict[str, Any]:
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "chat_id": chat_id,
        "thread_id": thread_id,
        "message_id": message_id,
        "soul_md_hash": soul_md_hash,
        "text_length": text_length,
        "ai_score": ai_score,
        "human_likeness": human_likeness,
        "sub_scores": sub_scores,
        "interpretation": interpretation,
        "elapsed_s": elapsed_s,
    }


def make_error_record(
    *,
    chat_id: str,
    thread_id: Optional[str],
    message_id: Optional[str],
    soul_md_hash: str,
    text_length: int,
    error: str,
) -> dict[str, Any]:
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "chat_id": chat_id,
        "thread_id": thread_id,
        "message_id": message_id,
        "soul_md_hash": soul_md_hash,
        "text_length": text_length,
        "error": error,
    }


def read_records(
    *,
    soul_md_hash: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Read records from the log. Optionally filter by soul_md_hash.

    Args:
        soul_md_hash: If set, only records matching this hash are returned.
        limit: If set, return only the most recent N matching records.

    Returns: list of records (newest last).
    """
    path = humanness_log_path()
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("humanness_log: skipped malformed line: %s", line[:80])
                continue
            if soul_md_hash and rec.get("soul_md_hash") != soul_md_hash:
                continue
            records.append(rec)
    if limit and limit > 0:
        records = records[-limit:]
    return records


def mean_humanness(records: Iterable[dict[str, Any]]) -> Optional[float]:
    """Return the arithmetic mean of human_likeness across records, or None."""
    successful = [r["human_likeness"] for r in records if "human_likeness" in r]
    if not successful:
        return None
    return sum(successful) / len(successful)
