"""Taxonomy loader + promotion ledger for DGM-H GLM data v1.

Plan reference: ``.omc/plans/dgmh-glm-data-v1.md`` §4 Step 5b (promotion
ledger) and AC5 (taxonomy conformance).

Two responsibilities live in this small module:

1. ``load(staging_root)`` — return the seven baseline ko-* tone categories
   plus any ``_new_*`` categories the operator has historically approved.
2. ``promote(category_id, staging_root)`` — append one approved ``_new_*``
   category to the staging-root ledger, so the *next* run's classifier sees
   it as a first-class category.

The ledger is append-only YAML at ``<staging_root>/promoted_taxonomy.yaml``.
If the file does not exist yet, ``load`` returns the baseline list only
(J5 fix from round-2 review: bootstrap must not require pre-existing file).
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Iterable

import yaml

# Locked baseline (plan §3 AC5). Order must be preserved for downstream
# prompts / inventory rendering, so we keep this as a tuple-of-truth.
BASELINE_KO_7: list[str] = [
    "communication",
    "content",
    "filler",
    "language",
    "structure",
    "style",
    "viral-hook",
]


_LEDGER_FILENAME = "promoted_taxonomy.yaml"


def _ledger_path(staging_root: Path) -> Path:
    return Path(staging_root) / _LEDGER_FILENAME


def _read_entries(staging_root: Path) -> list[dict]:
    """Read all entries from the promotion ledger.

    Returns an empty list if the file does not exist or is empty. This is
    the J5 bootstrap fix: a fresh operator install has no ledger yet.
    """

    path = _ledger_path(staging_root)
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(
            f"{path} must be a YAML list at the top level; got {type(data)!r}"
        )
    return data


def load(staging_root: Path) -> list[str]:
    """Return ``BASELINE_KO_7`` ∪ promoted ``_new_*`` categories (in order).

    Baseline entries come first; promoted entries follow in the order they
    were appended. Duplicates between baseline and ledger are silently
    deduplicated (baseline wins).
    """

    entries = _read_entries(staging_root)
    promoted: list[str] = []
    seen: set[str] = set(BASELINE_KO_7)
    for entry in entries:
        cat_id = entry.get("category_id") if isinstance(entry, dict) else None
        if not isinstance(cat_id, str):
            continue
        if cat_id in seen:
            continue
        promoted.append(cat_id)
        seen.add(cat_id)
    return list(BASELINE_KO_7) + promoted


def promote(category_id: str, staging_root: Path) -> None:
    """Append ``category_id`` to the promotion ledger (idempotent on dup).

    The ledger is append-only on disk: we never rewrite or reorder existing
    entries. We *do* skip a no-op write if ``category_id`` already exists,
    so callers can promote the same id twice without growing the ledger.
    """

    if not isinstance(category_id, str) or not category_id:
        raise ValueError(f"category_id must be a non-empty string; got {category_id!r}")
    if not category_id.startswith("_new_"):
        raise ValueError(
            "promote() only accepts discovery proposals prefixed with '_new_'; "
            f"got {category_id!r}"
        )

    staging_root = Path(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)

    existing = _read_entries(staging_root)
    for entry in existing:
        if isinstance(entry, dict) and entry.get("category_id") == category_id:
            return  # already promoted; preserve append-only invariant

    new_entry = {
        "category_id": category_id,
        "promoted_at": _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
    }

    path = _ledger_path(staging_root)
    # Append by re-serializing the full list. Atomicity is good enough for
    # operator-local single-process use; a future v2 can swap in fcntl.
    payload: list[dict] = list(existing) + [new_entry]
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)


def iter_promoted(staging_root: Path) -> Iterable[str]:
    """Yield only the promoted ``_new_*`` categories (no baseline)."""

    return (cat for cat in load(staging_root) if cat not in BASELINE_KO_7)
