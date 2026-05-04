"""
dgmh/loop.py — DGM-H end-to-end iteration driver (W4 deliverable).

Ties W1 (archive bootstrap), W2 (select_parents), W3 (modifier), and W4
(critic + run_log) into a single iteration covering paper §8.3 lines 4-9.

run_one_iteration() is a single-iteration driver (not a full multi-T loop),
matching the W4 deliverable scope per plan v2 §CONCERN 4. It covers:
  4. SelectParents → parent
  5. acquire per-skill lock for parent
  6. modifier.modify(parent, archive) → SkillCandidate
  6a. critic.review(candidate, parent, archive) → verdict
  6b. on reject → log, do not write
  7-9. on approve → write SKILL.md, append archive entry, bump compiled_children

Always emits one RunRecord to ~/.hermes/dgmh/runs.jsonl per iteration end.
Never crashes on per-iteration errors — catch + log + return IterationResult.

Reference: playground/dgmh-engine/loop.ts::runDgmh (full TS driver)
           hermes-skill-archive-dgmh-plan-v2-addendum.md §CONCERN 4 (W4)
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import random as _random_module
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
import sys
if str(_HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(_HERMES_ROOT))

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------

from dgmh.locks import acquire_skill_lock, SkillLockError
from dgmh.modifier import SkillModifier, SkillCandidate, CodexModifierError, ScopeAttemptsExhaustedError
from dgmh.critic import SkillCritic, CriticReview, CodexCriticError, CriticParseAttemptsExhaustedError
from dgmh.run_log import (
    RunRecord,
    RunRecordChild,
    RunRecordRejection,
    RunRecordBiasSummary,
    append_run_record,
    default_runs_path,
    make_run_record,
    summarize_bias_report,
)
from dgmh.bias_detector import make_heuristic_skill_bias_detector
from dgmh.select_parents import (
    Generation,
    DgmhHyperparams,
    DEFAULT_HYPERPARAMS,
    select_parents,
)

# ---------------------------------------------------------------------------
# Rejection reason type
# ---------------------------------------------------------------------------

IterationRejectionReason = Literal["modifier-error", "critic-error", "critic-reject"]

# ---------------------------------------------------------------------------
# IterationOpts
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class IterationOpts:
    """Options for run_one_iteration().

    Mirrors RunOptions from loop.ts, scoped to a single iteration.
    """

    # Archive of known generations (Generation objects from select_parents.py)
    seed_archive: list[Generation]

    # Modifier (W3) — provides .modify(parent, archive) → SkillCandidate
    modifier: SkillModifier

    # Critic (W4) — provides .review(child, parent, archive) → CriticReview
    # Optional; when omitted the critic gate is skipped (paper-pure §8.3)
    critic: SkillCritic | None = None

    # RNG seed for parent selection (int or None for random)
    seed: int | None = None

    # Parent selection count (paper §8.3 line 4 — default 1)
    parents_per_iter: int = 1

    # Hyperparams for select_parents §8.2
    hyperparams: DgmhHyperparams = dataclasses.field(
        default_factory=lambda: DEFAULT_HYPERPARAMS
    )

    # Path to archive.jsonl for new child admission
    archive_jsonl_path: str | Path | None = None

    # Path to runs.jsonl for run record
    runs_jsonl_path: str | Path | None = None

    # Lock timeout in seconds (passed to acquire_skill_lock)
    lock_timeout_s: float = 30.0

    # Lock retries (passed to acquire_skill_lock)
    lock_retries: int = 3


# ---------------------------------------------------------------------------
# IterationResult
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class IterationResult:
    """Result of one run_one_iteration() call.

    accepted: True if the child was admitted to the archive.
    rejection_reason: set when accepted=False.
    candidate: the SkillCandidate produced by the modifier (None on modifier error).
    run_record: the RunRecord appended to runs.jsonl.
    """

    accepted: bool
    rejection_reason: IterationRejectionReason | None
    rejection_message: str
    candidate: SkillCandidate | None
    parent: Generation | None
    run_record: RunRecord


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _dgmh_dir() -> Path:
    hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(hermes_home) / "dgmh"


def _skills_dir() -> Path:
    hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(hermes_home) / "skills"


def _default_archive_jsonl() -> Path:
    return _dgmh_dir() / "archive.jsonl"


# ---------------------------------------------------------------------------
# Atomic write helper (mirrors _atomic_write_text from skill_manager_tool.py W1)
# ---------------------------------------------------------------------------


def _atomic_write_text(path: Path, content: str) -> None:
    """Write content to path atomically via tempfile + os.replace.

    Mirrors the _atomic_write_text pattern from tools/skill_manager_tool.py.
    Creates parent directories if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".skill.tmp.", suffix="")
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
# Archive JSONL append helper
# ---------------------------------------------------------------------------


def _append_archive_entry(archive_path: Path, entry_dict: dict[str, Any]) -> None:
    """Append one archive entry dict as a JSON line."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry_dict, ensure_ascii=False) + "\n"
    with open(archive_path, "a", encoding="utf-8") as f:
        f.write(line)


# ---------------------------------------------------------------------------
# Monotonic generation index
# ---------------------------------------------------------------------------


def _next_monotonic_index(archive: list[Generation]) -> int:
    """Compute the next monotonic generationIndex for archive admission.

    Port of loop.ts::nextMonotonicIndex — derived from existing entries,
    not from the outer-loop counter. Phase 1.6 rehydration safe.
    """
    if not archive:
        return 0
    return max(g.generation_index for g in archive) + 1


# ---------------------------------------------------------------------------
# Core single-iteration driver
# ---------------------------------------------------------------------------


def run_one_iteration(opts: IterationOpts) -> IterationResult:
    """Run one DGM-H iteration: select → lock → modify → critic → admit.

    Covers paper §8.3 lines 4-9 for the Hermes skill domain:
      4. select_parents(seed_archive, count=1, rng, hp) → parent
      5. acquire per-skill lock for parent
      6. modifier.modify(parent, archive) → SkillCandidate
      6a. critic.review(candidate, parent, archive) → verdict
      6b. on reject → record rejection, do NOT write
      7-9. on approve → write SKILL.md, append archive entry, bump compiled_children

    Always appends one RunRecord to runs.jsonl, even on failure.
    Never raises on per-iteration errors — catch + log + return IterationResult.

    Args:
        opts: IterationOpts with all dependencies injected.

    Returns:
        IterationResult describing what happened.
    """
    # Resolve RNG
    seed = opts.seed if opts.seed is not None else int(time.time() * 1000) % (2 ** 31)
    rng = _random_module.Random(seed)

    archive_size_pre = len(opts.seed_archive)
    accepted_children: list[RunRecordChild] = []
    rejections: list[RunRecordRejection] = []
    candidate: SkillCandidate | None = None
    parent: Generation | None = None
    rejection_reason: IterationRejectionReason | None = None
    rejection_message = ""
    accepted = False

    archive_path = Path(opts.archive_jsonl_path) if opts.archive_jsonl_path else _default_archive_jsonl()
    runs_path = Path(opts.runs_jsonl_path) if opts.runs_jsonl_path else default_runs_path()

    # Make a mutable copy of archive to track in-iteration updates
    archive = list(opts.seed_archive)

    try:
        # Step 4: SelectParents
        parents = select_parents(
            archive,
            opts.parents_per_iter,
            rng,
            opts.hyperparams,
        )
        if not parents:
            rejection_reason = "modifier-error"
            rejection_message = "select_parents returned empty list"
            rejections.append(RunRecordRejection(
                reason="modifier-error",
                parent_id="(none)",
                message=rejection_message,
            ))
            logger.error("loop: select_parents returned empty list")
        else:
            parent = parents[0]
            parent_id = parent.id

            # Parse category/name from skill_id (format: "<cat>/<name>")
            parts = parent_id.split("/", 1)
            category = parts[0] if len(parts) == 2 else ""
            name = parts[1] if len(parts) == 2 else parent_id

            # Step 5: Acquire per-skill lock
            try:
                with acquire_skill_lock(
                    category,
                    name,
                    timeout_s=opts.lock_timeout_s,
                    retries=opts.lock_retries,
                ):
                    # Step 6: modifier.modify(parent, archive) → SkillCandidate
                    try:
                        candidate = opts.modifier.modify(parent, archive)
                    except (CodexModifierError, ScopeAttemptsExhaustedError) as exc:
                        rejection_reason = "modifier-error"
                        rejection_message = str(exc)
                        rejections.append(RunRecordRejection(
                            reason="modifier-error",
                            parent_id=parent_id,
                            message=rejection_message,
                        ))
                        logger.warning(
                            "loop: modifier error for parent %r: %s", parent_id, exc
                        )
                    else:
                        # Step 6a: critic gate (optional)
                        critic_approved = True
                        if opts.critic is not None:
                            try:
                                review: CriticReview = opts.critic.review(
                                    candidate, parent, archive
                                )
                                if review.verdict == "reject":
                                    critic_approved = False
                                    rejection_reason = "critic-reject"
                                    rejection_message = review.reason
                                    rejections.append(RunRecordRejection(
                                        reason="critic-reject",
                                        parent_id=parent_id,
                                        message=review.reason,
                                    ))
                                    logger.info(
                                        "loop: critic rejected parent=%r reason=%r",
                                        parent_id,
                                        review.reason,
                                    )
                            except (CodexCriticError, CriticParseAttemptsExhaustedError) as exc:
                                critic_approved = False
                                rejection_reason = "critic-error"
                                rejection_message = str(exc)
                                rejections.append(RunRecordRejection(
                                    reason="critic-error",
                                    parent_id=parent_id,
                                    message=rejection_message,
                                ))
                                logger.warning(
                                    "loop: critic error for parent %r: %s",
                                    parent_id,
                                    exc,
                                )

                        if critic_approved:
                            # Step 7-9: Admit child to archive
                            # Write SKILL.md to ~/.hermes/skills/<cat>/<name>/SKILL.md
                            skill_path = _skills_dir() / category / name / "SKILL.md"
                            try:
                                _atomic_write_text(skill_path, candidate.skill_md)
                                logger.info(
                                    "loop: wrote SKILL.md to %s", skill_path
                                )
                            except OSError as exc:
                                # File write failure → treat as modifier-error
                                rejection_reason = "modifier-error"
                                rejection_message = f"SKILL.md write failed: {exc}"
                                rejections.append(RunRecordRejection(
                                    reason="modifier-error",
                                    parent_id=parent_id,
                                    message=rejection_message,
                                ))
                                logger.error(
                                    "loop: could not write SKILL.md for %r: %s",
                                    parent_id,
                                    exc,
                                )
                            else:
                                # Allocate monotonic generation_index
                                next_gen_index = _next_monotonic_index(archive)

                                # Bump parent compiled_children in our local copy
                                archive = [
                                    dataclasses.replace(g, compiled_children=g.compiled_children + 1)
                                    if g.id == parent_id else g
                                    for g in archive
                                ]

                                # Build new archive entry
                                child_id = candidate.skill_id
                                child_lineage_depth = len(candidate.lineage)
                                child_score = 0.5  # unevaluated; evaluator wired in W5

                                new_gen = Generation(
                                    id=child_id,
                                    score=child_score,
                                    compiled_children=0,
                                    generation_index=next_gen_index,
                                    selectable=True,
                                    extra={
                                        "skill_path": str(skill_path),
                                        "lineage": candidate.lineage,
                                        "parent_id": parent_id,
                                        "admitted_at": datetime.now(timezone.utc).isoformat(),
                                    },
                                )
                                archive.append(new_gen)

                                # Append to archive.jsonl
                                entry_dict = {
                                    "skill_id": child_id,
                                    "name": name,
                                    "category": category,
                                    "parent_id": parent_id,
                                    "lineage": candidate.lineage,
                                    "score": child_score,
                                    "eval_suite_id": None,
                                    "compiled_children": 0,
                                    "generation_index": next_gen_index,
                                    "skill_path": str(skill_path),
                                    "content_hash": "",
                                    "selectable": True,
                                    "admitted_at": new_gen.extra.get("admitted_at", ""),
                                    "schema_version": 1,
                                }
                                try:
                                    _append_archive_entry(archive_path, entry_dict)
                                except OSError as exc:
                                    logger.error(
                                        "loop: could not append to archive.jsonl: %s", exc
                                    )

                                accepted = True
                                accepted_children.append(RunRecordChild(
                                    id=child_id,
                                    parent_id=parent_id,
                                    score=child_score,
                                    generation_index=next_gen_index,
                                    lineage_depth=child_lineage_depth,
                                ))
                                logger.info(
                                    "loop: admitted child %r generation_index=%d",
                                    child_id,
                                    next_gen_index,
                                )

            except SkillLockError as exc:
                rejection_reason = "modifier-error"
                rejection_message = f"lock error: {exc}"
                rejections.append(RunRecordRejection(
                    reason="modifier-error",
                    parent_id=parent.id if parent else "(none)",
                    message=rejection_message,
                ))
                logger.error(
                    "loop: lock contention for parent %r: %s",
                    parent.id if parent else "(none)",
                    exc,
                )

    except Exception as exc:
        # Catch-all: never crash the iteration
        rejection_reason = rejection_reason or "modifier-error"
        rejection_message = rejection_message or str(exc)
        if not rejections:
            rejections.append(RunRecordRejection(
                reason="modifier-error",
                parent_id=parent.id if parent else "(none)",
                message=str(exc),
            ))
        logger.exception("loop: unexpected error in iteration: %s", exc)

    # Always emit one RunRecord
    archive_size_post = len(archive)

    # Invoke bias detector on post-iteration archive (W5 wiring).
    # Replaces W4 placeholder zero-counts. On concern/block, logs to stderr
    # but does NOT block the iteration (operator-visible signal, not crash).
    _bias_detector = make_heuristic_skill_bias_detector()
    try:
        bias_report = _bias_detector.detect(archive, recent_n=20)
    except Exception as exc:
        logger.error("loop: bias detector failed: %s", exc)
        bias_report = None

    run_record = make_run_record(
        seed=seed,
        archive_size_pre=archive_size_pre,
        archive_size_post=archive_size_post,
        accepted_children=accepted_children,
        rejections=rejections,
        bias=summarize_bias_report(bias_report),
    )

    try:
        append_run_record(runs_path, run_record)
    except Exception as exc:
        logger.error("loop: could not append run record: %s", exc)

    return IterationResult(
        accepted=accepted,
        rejection_reason=rejection_reason,
        rejection_message=rejection_message,
        candidate=candidate,
        parent=parent,
        run_record=run_record,
    )
