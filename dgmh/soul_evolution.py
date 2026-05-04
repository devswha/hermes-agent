"""
dgmh/soul_evolution.py — W6: SOUL.md as the DGM-H evolution surface.

Why SOUL.md:
  - Single highest-leverage prompt for a 1:1 assistant.
  - Mutating one skill at a time is too slow for 1:1 conversation feedback.
  - Operator-defined SOUL.md is the starting point (gen-0, preserved on first run).

Archive entry shape for SOUL.md domain:
  {
    soul_md_path: str,
    soul_md_content: str,
    soul_md_hash: str,
    recorded_reactions: {pos: int, neg: int},
    score: float,          # pos / (pos + neg + 1)  — smoothed
    generation_index: int,
    lineage: list[str],    # SHA-256 hashes of ancestor soul_md_content
    schema_version: 2,
    admitted_at: str,
    is_gen0: bool,
    status: "active" | "archived"
  }

Score formula: pos / (pos + neg + 1)  (Laplace smoothed; new entries start at 0.5)

On critic-approved candidate:
  - Atomically replace ~/.hermes/SOUL.md (via _atomic_write_text pattern)
  - Archive prior version to ~/.hermes/dgmh/soul_archive/<timestamp>__<hash>.md
  - Append entry to ~/.hermes/dgmh/soul_archive.jsonl

On rejection:
  - Log + write RunRecord, do not touch SOUL.md

CRITICAL: existing SOUL.md is preserved as gen-0 in archive on first run.

Reference: W1 _atomic_write_text (dgmh/loop.py), W4 critic.py
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure project root importable
# ---------------------------------------------------------------------------

_HERMES_ROOT = Path(__file__).parent.parent
if str(_HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(_HERMES_ROOT))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


def _soul_md_path() -> Path:
    return _hermes_home() / "SOUL.md"


def _soul_archive_dir() -> Path:
    return _hermes_home() / "dgmh" / "soul_archive"


def _soul_archive_jsonl() -> Path:
    return _hermes_home() / "dgmh" / "soul_archive.jsonl"


def _soul_runs_jsonl() -> Path:
    return _hermes_home() / "dgmh" / "soul_runs.jsonl"


# ---------------------------------------------------------------------------
# Score formula
# ---------------------------------------------------------------------------


def compute_score(pos: int, neg: int) -> float:
    """Compute smoothed reaction score.

    Formula: pos / (pos + neg + 1)
    New entries with 0 reactions yield 0.5 (0 / (0 + 0 + 1) = 0.0 ... wait —
    spec says 'new entries start at 0.5', so we use a symmetric Laplace:
    (pos + 0.5) / (pos + neg + 1).

    This gives:
      pos=0, neg=0 → 0.5
      pos=1, neg=0 → 1.5/2 = 0.75
      pos=0, neg=1 → 0.5/2 = 0.25
    """
    return (pos + 0.5) / (pos + neg + 1.0)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_FRONTMATTER_NAME_RE = re.compile(
    r"^---\s*\n(.*?)\n---", re.DOTALL
)
_NAME_LINE_RE = re.compile(r"^\s*name\s*:\s*([A-Za-z0-9_-]+)\s*$", re.MULTILINE)


def _derive_soul_skill_id(soul_md_content: str) -> str:
    """Extract `soul/<name>` skill_id from SOUL.md YAML frontmatter.

    Codex modifier expects lineage[-1] == parent_id, where parent_id follows
    the `<category>/<name>` pattern. Codex derives <name> from the YAML
    `name:` field at the top of the markdown, so we must use the same.
    Falls back to "soul/SOUL" only when no frontmatter is present (gen-0
    legacy plain-text SOUL.md).
    """
    fm_match = _FRONTMATTER_NAME_RE.match(soul_md_content)
    if fm_match:
        block = fm_match.group(1)
        name_match = _NAME_LINE_RE.search(block)
        if name_match:
            return f"soul/{name_match.group(1).lower()}"
    return "soul/SOUL"


# ---------------------------------------------------------------------------
# Soul archive entry
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SoulArchiveEntry:
    """One entry in soul_archive.jsonl."""

    soul_md_path: str
    soul_md_content: str
    soul_md_hash: str
    recorded_reactions: dict[str, int]  # {"pos": int, "neg": int}
    score: float
    generation_index: int
    lineage: list[str]  # SHA-256 hashes of ancestor soul_md_content
    admitted_at: str
    is_gen0: bool
    status: Literal["active", "archived"]
    schema_version: int = 2


def _entry_to_dict(entry: SoulArchiveEntry) -> dict[str, Any]:
    return dataclasses.asdict(entry)


def _load_soul_archive(path: Path | None = None) -> list[SoulArchiveEntry]:
    """Load soul_archive.jsonl. Returns [] if file missing."""
    target = path or _soul_archive_jsonl()
    if not target.exists():
        return []
    entries: list[SoulArchiveEntry] = []
    with open(target, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                valid_fields = {f.name for f in dataclasses.fields(SoulArchiveEntry)}
                entries.append(SoulArchiveEntry(
                    **{k: v for k, v in data.items() if k in valid_fields}
                ))
            except Exception as exc:
                logger.error(
                    "soul_evolution: corrupt archive entry at line %d: %s",
                    lineno,
                    exc,
                )
    return entries


def _append_soul_archive_entry(
    entry: SoulArchiveEntry,
    path: Path | None = None,
) -> None:
    """Append one entry to soul_archive.jsonl."""
    target = path or _soul_archive_jsonl()
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_entry_to_dict(entry), ensure_ascii=False) + "\n"
    with open(target, "a", encoding="utf-8") as f:
        f.write(line)


# ---------------------------------------------------------------------------
# Atomic write (mirrors W1 _atomic_write_text)
# ---------------------------------------------------------------------------


def _atomic_write_text(path: Path, content: str) -> None:
    """Write content to path atomically via tempfile + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=".soul.tmp.", suffix=""
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# gen-0 bootstrap
# ---------------------------------------------------------------------------


def _ensure_gen0(
    soul_path: Path | None = None,
    archive_jsonl: Path | None = None,
) -> SoulArchiveEntry:
    """Ensure the operator's original SOUL.md is registered as gen-0.

    Idempotent — skips if a gen-0 entry already exists in soul_archive.jsonl.

    Returns the gen-0 SoulArchiveEntry.
    """
    archive_jsonl = archive_jsonl or _soul_archive_jsonl()
    soul_path = soul_path or _soul_md_path()

    entries = _load_soul_archive(archive_jsonl)
    existing_gen0 = [e for e in entries if e.is_gen0]
    if existing_gen0:
        logger.debug("soul_evolution: gen-0 already registered, skipping")
        return existing_gen0[0]

    # Read current SOUL.md
    if soul_path.exists():
        content = soul_path.read_text(encoding="utf-8")
    else:
        content = ""
        logger.warning(
            "soul_evolution: SOUL.md not found at %s — registering empty gen-0",
            soul_path,
        )

    content_hash = _sha256(content)
    now = datetime.now(timezone.utc).isoformat()

    gen0 = SoulArchiveEntry(
        soul_md_path=str(soul_path),
        soul_md_content=content,
        soul_md_hash=content_hash,
        recorded_reactions={"pos": 0, "neg": 0},
        score=0.5,
        generation_index=0,
        lineage=[],
        admitted_at=now,
        is_gen0=True,
        status="active",
        schema_version=2,
    )

    _append_soul_archive_entry(gen0, archive_jsonl)
    logger.info(
        "soul_evolution: registered gen-0 SOUL.md (hash=%s)", content_hash[:12]
    )
    return gen0


# ---------------------------------------------------------------------------
# SoulEvolutionOpts
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SoulEvolutionOpts:
    """Options for run_soul_evolution()."""

    # Path overrides (default: standard ~/.hermes/ paths)
    soul_md_path: Path | None = None
    soul_archive_jsonl: Path | None = None
    soul_runs_jsonl: Path | None = None

    # Reactions recorded since last evolution (injected for testing)
    pos_reactions: int = 0
    neg_reactions: int = 1  # default: 1 negative triggered this cycle

    # Modifier and critic (injected for testing; None = use real ones)
    modifier: Any = None  # SkillModifier-compatible .modify(parent, archive)
    critic: Any = None    # SkillCritic-compatible .review(child, parent, archive)

    # RNG seed
    seed: int | None = None


# ---------------------------------------------------------------------------
# Soul RunRecord (lightweight, separate from skill RunRecord)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SoulRunRecord:
    """Run record for one soul evolution cycle."""

    schema_version: int
    recorded_at: str
    seed: int
    accepted: bool
    rejection_reason: str
    parent_hash: str
    child_hash: str
    score_before: float
    score_after: float
    pos_reactions: int
    neg_reactions: int


def _append_soul_run_record(record: SoulRunRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dataclasses.asdict(record), ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


# ---------------------------------------------------------------------------
# Modifier and critic adapters for SOUL.md domain
# ---------------------------------------------------------------------------


def _load_humanness_primer() -> str:
    """Load the patina-derived humanness primer if present, else return empty.

    The primer is optional — absence falls back to the stock skill-modifier
    prompt so SOUL evolution still works without it.
    """
    primer_path = (
        Path(__file__).resolve().parent / "prompts" / "humanness-primer.md"
    )
    try:
        return primer_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning(
            "soul_evolution: humanness primer not found at %s — proceeding without it",
            primer_path,
        )
        return ""


def _augment_with_humanness_primer(base_template: str) -> str:
    """Append the humanness primer to a base prompt template, if available.

    The primer is appended as a clearly-marked Reference section. Codex sees
    both the original mutation/critique instructions and the humanness target
    framework in one prompt.
    """
    primer = _load_humanness_primer()
    if not primer.strip():
        return base_template
    separator = (
        "\n\n---\n\n"
        "## Reference: Humanness Primer (SOUL.md domain)\n\n"
        "The following primer comes from the patina humanness skill and "
        "describes the target the SOUL.md mutation should move toward — "
        "less AI-tone in the bot's resulting Discord replies, especially "
        "Korean casual chat. Apply it as evaluation framework when proposing "
        "or critiquing the candidate skill_md.\n\n"
    )
    return base_template + separator + primer


def _build_soul_modifier() -> Any:
    """Build a modifier that generates a new SOUL.md candidate via the critic prompt."""
    from pathlib import Path as _Path
    from dgmh.modifier import make_codex_skill_modifier
    from dgmh.select_parents import Generation

    _prompt_path = _Path(__file__).resolve().parent / "prompts" / "skill-modifier.md"
    _base_template = _prompt_path.read_text(encoding="utf-8")
    _prompt_template = _augment_with_humanness_primer(_base_template)

    class SoulModifier:
        """Wraps SkillModifier for SOUL.md domain."""

        def modify(self, parent: Any, archive: Any) -> Any:
            """Produce a mutated SOUL.md candidate.

            parent.extra['soul_md_content'] contains the current SOUL.md text.
            """
            from dgmh.modifier import SkillCandidate

            parent_content = parent.extra.get("soul_md_content", "")
            parent_hash = parent.extra.get("soul_md_hash", _sha256(parent_content))
            parent_lineage = parent.extra.get("lineage", [])

            # Build a synthetic prompt for SOUL.md mutation
            prompt = _build_soul_modifier_prompt(parent_content, archive)

            # Use SkillModifier for the actual LLM call, adapted for SOUL.md
            try:
                raw_modifier = make_codex_skill_modifier(prompt_template=_prompt_template)
                # Create a synthetic parent Generation for skill modifier
                synth_parent = Generation(
                    id=parent.id,
                    score=parent.score,
                    compiled_children=0,
                    generation_index=parent.generation_index,
                    selectable=True,
                    extra={
                        "skill_md": parent_content,
                        "skill_path": str(_soul_md_path()),
                        "content_hash": parent_hash,
                    },
                )
                result = raw_modifier.modify(synth_parent, archive)
                # Remap skill_md -> soul_md_content
                new_content = result.skill_md
                new_hash = _sha256(new_content)
                new_lineage = list(parent_lineage) + [parent_hash]
                return SkillCandidate(
                    skill_id=_derive_soul_skill_id(new_content),
                    skill_md=new_content,
                    lineage=new_lineage,
                    raw_codex_output=result.raw_codex_output,
                    attempt=result.attempt,
                )
            except Exception as exc:
                from dgmh.modifier import CodexModifierError
                raise CodexModifierError(
                    f"Soul modifier failed: {exc}", exit_code=1
                ) from exc

    return SoulModifier()


def _build_soul_modifier_prompt(parent_content: str, archive: Any) -> str:
    """Build a mutation prompt for SOUL.md."""
    return (
        "You are mutating a bot's system prompt (SOUL.md). "
        "The operator gave a negative reaction, signaling the current SOUL.md needs improvement. "
        "Propose an improved version that better serves the operator's intent.\n\n"
        f"CURRENT SOUL.md:\n{parent_content}\n\n"
        "Produce an improved SOUL.md that addresses the implicit negative feedback."
    )


def _build_soul_critic() -> Any:
    """Build a critic for SOUL.md domain."""
    from pathlib import Path as _Path
    from dgmh.critic import make_codex_skill_critic

    _critic_prompt_path = _Path(__file__).resolve().parent / "prompts" / "skill-critic.md"
    _base_critic_template = _critic_prompt_path.read_text(encoding="utf-8")
    _critic_prompt_template = _augment_with_humanness_primer(_base_critic_template)

    class SoulCritic:
        def __init__(self) -> None:
            self._raw = make_codex_skill_critic(prompt_template=_critic_prompt_template)

        def review(self, child: Any, parent: Any, archive: Any) -> Any:
            """Review a SOUL.md candidate using the critic.

            critic._resolve_skill_content checks `obj.skill_md` first, then
            `obj.skill_path`, then `obj.extra.skill_path` — none of which the
            SOUL parent_gen carries. Surface the SOUL.md content explicitly
            on the parent so the critic sees real text instead of the
            "(skill content unavailable for ...)" placeholder.
            """
            try:
                parent_content = ""
                if isinstance(getattr(parent, "extra", None), dict):
                    parent_content = parent.extra.get("soul_md_content", "")
                if parent_content and not getattr(parent, "skill_md", None):
                    object.__setattr__(parent, "skill_md", parent_content)
            except Exception:
                logger.exception("soul_evolution: failed to surface parent skill_md")
            return self._raw.review(child, parent, archive)

    return SoulCritic()


# ---------------------------------------------------------------------------
# Core evolution function
# ---------------------------------------------------------------------------


def run_soul_evolution(opts: SoulEvolutionOpts) -> bool:
    """Run one SOUL.md evolution cycle triggered by a reaction.

    Steps:
      1. Ensure gen-0 is archived (idempotent).
      2. Load current SOUL.md as parent.
      3. Run modifier to produce candidate.
      4. Run critic gate.
      5a. On approve: atomically replace SOUL.md, archive prior, append entry.
      5b. On reject: log RunRecord, do NOT touch SOUL.md.

    Returns:
        True if the candidate was accepted and SOUL.md updated.
    """
    soul_path = opts.soul_md_path or _soul_md_path()
    archive_jsonl = opts.soul_archive_jsonl or _soul_archive_jsonl()
    runs_jsonl = opts.soul_runs_jsonl or _soul_runs_jsonl()
    seed = opts.seed if opts.seed is not None else int(time.time() * 1000) % (2 ** 31)

    now = datetime.now(timezone.utc).isoformat()
    accepted = False
    rejection_reason = ""
    child_content = ""
    child_hash = ""

    # Step 1: Ensure gen-0 exists
    try:
        gen0 = _ensure_gen0(soul_path, archive_jsonl)
    except Exception as exc:
        logger.error("soul_evolution: gen-0 bootstrap failed: %s", exc)
        return False

    # Load current SOUL.md as parent
    if soul_path.exists():
        parent_content = soul_path.read_text(encoding="utf-8")
    else:
        parent_content = gen0.soul_md_content
        logger.warning("soul_evolution: SOUL.md missing, using gen-0 content")

    parent_hash = _sha256(parent_content)
    score_before = compute_score(opts.pos_reactions, opts.neg_reactions)

    # Load archive for context
    archive_entries = _load_soul_archive(archive_jsonl)
    next_gen_index = (
        max((e.generation_index for e in archive_entries), default=0) + 1
    )

    from dgmh.select_parents import Generation

    parent_lineage = []
    # Find parent's lineage from archive
    for entry in archive_entries:
        if entry.soul_md_hash == parent_hash and entry.status == "active":
            parent_lineage = entry.lineage
            break

    parent_gen = Generation(
        id=_derive_soul_skill_id(parent_content),
        score=score_before,
        compiled_children=0,
        generation_index=next_gen_index - 1,
        selectable=True,
        extra={
            "soul_md_content": parent_content,
            "soul_md_hash": parent_hash,
            "lineage": parent_lineage,
        },
    )

    # Build modifier and critic
    modifier = opts.modifier or _build_soul_modifier()
    critic = opts.critic or _build_soul_critic()

    # Step 3: Modify
    candidate = None
    try:
        # Pass archive entries as Generation list for context
        archive_as_gens = [
            Generation(
                id=f"soul/gen-{e.generation_index}",
                score=e.score,
                compiled_children=0,
                generation_index=e.generation_index,
                selectable=(e.status == "active"),
                extra={
                    "soul_md_content": e.soul_md_content,
                    "soul_md_hash": e.soul_md_hash,
                },
            )
            for e in archive_entries
        ]
        candidate = modifier.modify(parent_gen, archive_as_gens)
        child_content = candidate.skill_md
        child_hash = _sha256(child_content)
    except Exception as exc:
        rejection_reason = f"modifier-error: {exc}"
        logger.warning("soul_evolution: modifier failed: %s", exc)

    # Step 4: Critic gate
    critic_approved = False
    if candidate is not None:
        try:
            review = critic.review(candidate, parent_gen, archive_as_gens)
            if review.verdict == "approve":
                critic_approved = True
            else:
                rejection_reason = f"critic-reject: {review.reason}"
                logger.info(
                    "soul_evolution: critic rejected candidate: %s", review.reason
                )
        except Exception as exc:
            rejection_reason = f"critic-error: {exc}"
            logger.warning("soul_evolution: critic error: %s", exc)

    # Step 5a: Accept
    if critic_approved and child_content:
        try:
            # Archive the prior SOUL.md version
            _archive_prior_soul(
                parent_content,
                parent_hash,
                parent_lineage,
                archive_jsonl,
                next_gen_index - 1,
                opts.pos_reactions,
                opts.neg_reactions,
            )

            # Atomically replace SOUL.md
            _atomic_write_text(soul_path, child_content)
            logger.info(
                "soul_evolution: SOUL.md updated (gen %d, hash=%s)",
                next_gen_index,
                child_hash[:12],
            )

            # Append new archive entry for the accepted child
            child_lineage = list(parent_lineage) + [parent_hash]
            score_after = compute_score(0, 0)  # fresh generation starts at 0.5
            new_entry = SoulArchiveEntry(
                soul_md_path=str(soul_path),
                soul_md_content=child_content,
                soul_md_hash=child_hash,
                recorded_reactions={"pos": 0, "neg": 0},
                score=score_after,
                generation_index=next_gen_index,
                lineage=child_lineage,
                admitted_at=now,
                is_gen0=False,
                status="active",
                schema_version=2,
            )
            _append_soul_archive_entry(new_entry, archive_jsonl)
            accepted = True

        except Exception as exc:
            rejection_reason = f"write-error: {exc}"
            logger.error("soul_evolution: failed to write SOUL.md: %s", exc)
            accepted = False

    # Step 5b / always: Write run record
    record = SoulRunRecord(
        schema_version=2,
        recorded_at=now,
        seed=seed,
        accepted=accepted,
        rejection_reason=rejection_reason,
        parent_hash=parent_hash,
        child_hash=child_hash,
        score_before=score_before,
        score_after=compute_score(0, 0) if accepted else score_before,
        pos_reactions=opts.pos_reactions,
        neg_reactions=opts.neg_reactions,
    )
    try:
        _append_soul_run_record(record, runs_jsonl)
    except Exception as exc:
        logger.error("soul_evolution: could not write run record: %s", exc)

    return accepted


def _archive_prior_soul(
    content: str,
    content_hash: str,
    lineage: list[str],
    archive_jsonl: Path,
    gen_index: int,
    pos: int,
    neg: int,
) -> None:
    """Archive the current (pre-replacement) SOUL.md version to the soul_archive dir."""
    archive_dir = _soul_archive_dir()
    archive_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_file = archive_dir / f"{ts}__{content_hash[:16]}.md"

    # Write the old SOUL.md content to soul_archive/<timestamp>__<hash>.md
    try:
        _atomic_write_text(archive_file, content)
        logger.info("soul_evolution: archived prior SOUL.md to %s", archive_file)
    except Exception as exc:
        logger.error("soul_evolution: could not archive prior SOUL.md: %s", exc)

    # Mark prior active entry as archived in soul_archive.jsonl
    entries = _load_soul_archive(archive_jsonl)
    updated = False
    for entry in entries:
        if entry.soul_md_hash == content_hash and entry.status == "active":
            entry.status = "archived"
            entry.recorded_reactions = {"pos": pos, "neg": neg}
            entry.score = compute_score(pos, neg)
            updated = True

    if updated:
        # Rewrite entire file (small file, atomic)
        target = archive_jsonl
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(_entry_to_dict(e), ensure_ascii=False) for e in entries]
        fd, tmp_path = tempfile.mkstemp(
            dir=str(target.parent), prefix=".soul_archive.tmp.", suffix=""
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
                if lines:
                    f.write("\n")
            os.replace(tmp_path, target)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
