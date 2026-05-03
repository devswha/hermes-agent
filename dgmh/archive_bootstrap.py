"""
dgmh/archive_bootstrap.py — DGM-H archive bootstrap from Hermes skill registry.

Port note: Archive entry model mirrors playground/dgmh-engine/types.ts::Generation
and archive.ts::archiveAdd semantics. bootstrap_archive() is the concrete Hermes
implementation of v2 §BLOCKER 1 — registers ALL skills (including disabled ones)
with disabled skills flagged selectable=False.

Reference: devswha/dgmh @ 3eddf82a661688f88b4bbf4c55028509d0eac002
  - playground/dgmh-engine/types.ts: Generation, Archive field contracts
  - playground/dgmh-engine/archive.ts: append-only semantics, generationIndex
  - hermes-skill-archive-dgmh-plan-v2-addendum.md §BLOCKER 1 (disabled-skill handling)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Pydantic v2 if available, else plain dataclass
# ---------------------------------------------------------------------------
try:
    from pydantic import BaseModel, Field

    class ArchiveEntry(BaseModel):
        """DGM-H archive entry for a Hermes skill.

        Maps to playground/dgmh-engine/types.ts::Generation plus Hermes-native
        fields (skill_path, selectable).
        """

        # Identity
        skill_id: str = Field(description="<category>/<name> slug")
        name: str
        category: str

        # Lineage (root entries have empty list — mirrors Agent.lineage in TS)
        parent_id: str | None = None
        lineage: list[str] = Field(default_factory=list)

        # Scoring — Codex-judge fills this; bootstrap stub = 0.5 (unevaluated)
        score: float = 0.5
        eval_suite_id: str | None = None

        # Novelty — n_i in §8.2 h_i = 1/(1+n_i)
        compiled_children: int = 0

        # Archive position — monotonically increasing across accepted entries
        generation_index: int = 0

        # Hermes-native
        skill_path: str = Field(description="Absolute path to SKILL.md")
        content_hash: str = Field(description="SHA-256 of SKILL.md content at bootstrap time")
        selectable: bool = True  # False for disabled skills (v2 §BLOCKER 1)

        # Metadata
        admitted_at: str = Field(description="ISO-8601 UTC timestamp")
        schema_version: int = 1

        def to_jsonl_dict(self) -> dict[str, Any]:
            return self.model_dump()

    _PYDANTIC = True

except ImportError:
    import dataclasses

    @dataclasses.dataclass
    class ArchiveEntry:  # type: ignore[no-redef]
        """DGM-H archive entry (dataclass fallback when pydantic unavailable)."""

        skill_id: str
        name: str
        category: str
        parent_id: str | None = None
        lineage: dataclasses.field(default_factory=list) = dataclasses.field(default_factory=list)
        score: float = 0.5
        eval_suite_id: str | None = None
        compiled_children: int = 0
        generation_index: int = 0
        skill_path: str = ""
        content_hash: str = ""
        selectable: bool = True
        admitted_at: str = ""
        schema_version: int = 1

        def to_jsonl_dict(self) -> dict[str, Any]:
            return dataclasses.asdict(self)

    _PYDANTIC = False


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hermes skill registry import (module-level for patchability in tests)
# ---------------------------------------------------------------------------

# Ensure hermes-agent root is on sys.path before importing hermes modules.
_HERMES_ROOT = Path(__file__).parent.parent
if str(_HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(_HERMES_ROOT))

try:
    from tools.skills_tool import _find_all_skills  # noqa: WPS433
except ImportError:  # pragma: no cover — only missing in isolated test envs
    def _find_all_skills(*, skip_disabled: bool = False) -> list[dict]:  # type: ignore[misc]
        return []

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _dgmh_dir() -> Path:
    """~/.hermes/dgmh/ — DGM-H state home."""
    hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(hermes_home) / "dgmh"


def _archive_jsonl_path() -> Path:
    return _dgmh_dir() / "archive.jsonl"


# ---------------------------------------------------------------------------
# Skill path resolution
# ---------------------------------------------------------------------------

def _skill_md_path(category: str, name: str) -> Path | None:
    """Locate SKILL.md for a given category/name under ~/.hermes/skills/."""
    hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    skills_root = Path(hermes_home) / "skills"
    candidate = skills_root / category / name / "SKILL.md"
    if candidate.exists():
        return candidate
    # flat layout (skill directly in category dir)
    candidate2 = skills_root / category / "SKILL.md"
    if candidate2.exists() and category == name:
        return candidate2
    return None


def _skill_id(category: str, name: str) -> str:
    if category:
        return f"{category}/{name}"
    return name


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Disabled-skill detection
# ---------------------------------------------------------------------------

def _get_disabled_skill_names() -> set[str]:
    """Read ~/.hermes/config.yaml disabled lists; returns set of skill names."""
    hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    config_path = Path(hermes_home) / "config.yaml"
    if not config_path.exists():
        return set()
    try:
        import yaml  # type: ignore[import]
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        skills_cfg = cfg.get("skills", {}) or {}
        disabled: set[str] = set(skills_cfg.get("disabled", []) or [])
        # platform_disabled: {platform: [name, ...]} — include all names
        for names in (skills_cfg.get("platform_disabled", {}) or {}).values():
            disabled.update(names or [])
        return disabled
    except Exception as exc:
        logger.warning("Could not read disabled skills from config.yaml: %s", exc)
        return set()


# ---------------------------------------------------------------------------
# Core bootstrap
# ---------------------------------------------------------------------------

def bootstrap_archive(*, persist: bool = True) -> list[ArchiveEntry]:
    """Bootstrap the DGM-H archive from the Hermes skill registry.

    Calls _find_all_skills(skip_disabled=False) to get the deterministic sorted
    list of all skills (disabled skills are returned by passing skip_disabled=False
    but are NOT filtered out by _find_all_skills itself — we detect disabled status
    separately to set selectable=False per v2 §BLOCKER 1).

    Returns list[ArchiveEntry] sorted by (category, name), persisted to
    ~/.hermes/dgmh/archive.jsonl (newline-delimited JSON, schemaVersion=1).

    Note: _find_all_skills() with skip_disabled=False returns skills WITHOUT
    filtering disabled ones — the Hermes source shows that disabled filtering
    only happens when skip_disabled is True (defaults False for the config UI).
    Re-reading the source: _find_all_skills(skip_disabled=False) filters OUT
    disabled skills (the default). We call it with skip_disabled=False and
    then separately check disabled names to register them too.

    Args:
        persist: If True (default), write entries to archive.jsonl.

    Returns:
        Sorted list of ArchiveEntry objects.
    """
    # Get all skills including disabled ones
    # _find_all_skills(skip_disabled=False) returns skills FILTERED of disabled.
    # To also get disabled skills, we call it with skip_disabled=True which
    # returns the full set without filtering, then cross-reference.
    all_skills_raw = _find_all_skills(skip_disabled=True)  # full set
    disabled_names = _get_disabled_skill_names()

    now_iso = datetime.now(timezone.utc).isoformat()
    entries: list[ArchiveEntry] = []

    for idx, skill_meta in enumerate(all_skills_raw):
        name: str = skill_meta["name"]
        category: str = skill_meta.get("category") or ""
        description: str = skill_meta.get("description") or ""

        skill_id = _skill_id(category, name)
        selectable = name not in disabled_names

        # Read SKILL.md for content hash
        skill_md = _skill_md_path(category, name)
        if skill_md and skill_md.exists():
            try:
                content = skill_md.read_text(encoding="utf-8")
                content_hash = _sha256(content)
                skill_path = str(skill_md)
            except Exception as exc:
                logger.warning("Could not read SKILL.md for %s: %s", skill_id, exc)
                content_hash = ""
                skill_path = str(skill_md) if skill_md else ""
        else:
            content_hash = ""
            skill_path = ""

        entry = ArchiveEntry(
            skill_id=skill_id,
            name=name,
            category=category,
            parent_id=None,
            lineage=[],
            score=0.5,  # unevaluated baseline; Codex-judge fills this in W1 scorer
            eval_suite_id=None,
            compiled_children=0,
            generation_index=idx,
            skill_path=skill_path,
            content_hash=content_hash,
            selectable=selectable,
            admitted_at=now_iso,
            schema_version=1,
        )
        entries.append(entry)

    # Sort by (category, name) per plan spec
    entries.sort(key=lambda e: (e.category, e.name))

    # Re-assign generation_index after sort for deterministic ordering
    for i, entry in enumerate(entries):
        if _PYDANTIC:
            # pydantic v2: use model_copy
            entries[i] = entry.model_copy(update={"generation_index": i})
        else:
            entry.generation_index = i

    if persist:
        _persist_archive(entries)

    logger.info(
        "bootstrap_archive: %d entries (%d selectable, %d disabled)",
        len(entries),
        sum(1 for e in entries if e.selectable),
        sum(1 for e in entries if not e.selectable),
    )
    return entries


def _persist_archive(entries: list[ArchiveEntry]) -> None:
    """Write entries to ~/.hermes/dgmh/archive.jsonl (overwrite on bootstrap)."""
    path = _archive_jsonl_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for entry in entries:
        lines.append(json.dumps(entry.to_jsonl_dict(), ensure_ascii=False))

    # Atomic write — tempfile + os.replace (mirrors _atomic_write_text pattern)
    import tempfile

    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".archive.tmp.", suffix="")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    logger.info("Persisted %d archive entries to %s", len(entries), path)


def load_archive(path: Path | None = None) -> list[ArchiveEntry]:
    """Load existing archive entries from JSONL. Returns [] if file missing."""
    target = path or _archive_jsonl_path()
    if not target.exists():
        return []
    entries: list[ArchiveEntry] = []
    with open(target, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if _PYDANTIC:
                    entries.append(ArchiveEntry(**data))
                else:
                    import dataclasses as _dc
                    valid_fields = {f.name for f in _dc.fields(ArchiveEntry)}
                    entries.append(ArchiveEntry(**{k: v for k, v in data.items() if k in valid_fields}))
            except Exception as exc:
                logger.error("Corrupt archive entry at line %d: %s", lineno, exc)
    return entries
