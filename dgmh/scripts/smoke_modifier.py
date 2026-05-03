"""
dgmh/scripts/smoke_modifier.py — Smoke test for modifier.py (W3 deliverable).

One-shot driver: loads real archive, picks first selectable entry as parent,
asks real Codex to propose one mutation, prints the SkillCandidate.

NOT run under pytest (no test_ prefix, no conftest registration, __main__ only).
Run via: python -m dgmh.scripts.smoke_modifier

Safety: prints the candidate only — does NOT write to ~/.hermes/skills/.

Usage:
    python -m dgmh.scripts.smoke_modifier [--skill-id <cat>/<name>]

Env:
    DGMH_CODEX_BIN  — override Codex binary path (default: "codex")
    HERMES_HOME     — override ~/.hermes home (default: ~/.hermes)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — ensure project root importable
# ---------------------------------------------------------------------------

_HERMES_ROOT = Path(__file__).parent.parent.parent
if str(_HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(_HERMES_ROOT))

# ---------------------------------------------------------------------------
# Guard: skip when running under pytest
# ---------------------------------------------------------------------------

if "pytest" in sys.modules:
    # This module must not execute under pytest; it invokes real Codex.
    # A pytest import of this file is fine (no code runs at module level
    # other than this guard).
    pass
else:
    pass


def main() -> None:
    """Entry point for smoke test."""
    parser = argparse.ArgumentParser(
        description="DGM-H W3 smoke test: one-shot Codex modifier (print only, no write)"
    )
    parser.add_argument(
        "--skill-id",
        default=None,
        help="<category>/<name> of skill to use as parent (default: first selectable)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("smoke_modifier")

    # 1. Load real archive
    logger.info("Loading archive via bootstrap_archive_for_selection()...")
    from dgmh.archive_bootstrap import bootstrap_archive_for_selection

    generations = bootstrap_archive_for_selection()
    if not generations:
        logger.error("No archive entries found. Run bootstrap_archive() first.")
        sys.exit(1)

    logger.info("Loaded %d generations from archive.", len(generations))

    # 2. Pick parent
    selectable = [g for g in generations if g.selectable]
    if not selectable:
        logger.error("No selectable entries in archive. All skills are disabled?")
        sys.exit(1)

    parent = None
    if args.skill_id:
        for g in selectable:
            if g.id == args.skill_id:
                parent = g
                break
        if parent is None:
            logger.error("--skill-id %r not found in selectable archive entries.", args.skill_id)
            logger.info("Available selectable IDs: %s", [g.id for g in selectable[:10]])
            sys.exit(1)
    else:
        parent = selectable[0]

    logger.info("Using parent: id=%r score=%.3f", parent.id, parent.score)

    # Verify parent has a readable SKILL.md
    skill_path = parent.extra.get("skill_path", "") if isinstance(parent.extra, dict) else ""
    if not skill_path or not Path(skill_path).exists():
        logger.error(
            "Parent %r has no readable SKILL.md at %r. "
            "Archive may be stale or skill was removed.",
            parent.id, skill_path,
        )
        sys.exit(1)

    # 3. Load prompt template
    template_path = Path(__file__).parent.parent / "prompts" / "skill-modifier.md"
    if not template_path.exists():
        logger.error("Prompt template not found: %s", template_path)
        sys.exit(1)

    prompt_template = template_path.read_text(encoding="utf-8")
    logger.info("Loaded prompt template from %s (%d chars)", template_path, len(prompt_template))

    # 4. Build modifier and invoke
    from dgmh.modifier import (
        CodexModifierError,
        ScopeAttemptsExhaustedError,
        make_codex_skill_modifier,
    )

    modifier = make_codex_skill_modifier(
        prompt_template,
        recent_n=5,
        timeout_s=120.0,
        max_scope_attempts=3,
    )

    logger.info("Invoking real Codex modifier for parent %r...", parent.id)
    logger.info("(This calls Codex CLI — ensure OAuth is active.)")

    try:
        candidate = modifier.modify(parent, generations)
    except CodexModifierError as exc:
        logger.error(
            "Codex subprocess failed: %s (exit_code=%s timed_out=%s)",
            exc, exc.exit_code, exc.timed_out,
        )
        sys.exit(1)
    except ScopeAttemptsExhaustedError as exc:
        logger.error(
            "All %d scope attempts exhausted for parent %r: %s",
            exc.attempts, exc.parent_id, exc,
        )
        sys.exit(1)

    # 5. Print candidate (DO NOT write to ~/.hermes/skills/)
    print("\n" + "=" * 70)
    print("SMOKE TEST RESULT — SkillCandidate (print only, NOT written)")
    print("=" * 70)
    print(f"skill_id : {candidate.skill_id}")
    print(f"attempt  : {candidate.attempt}")
    print(f"lineage  : {candidate.lineage}")
    print(f"\n--- skill_md (first 500 chars) ---")
    print(candidate.skill_md[:500])
    if len(candidate.skill_md) > 500:
        print(f"... [{len(candidate.skill_md) - 500} more chars]")
    print("\n--- raw_codex_output (first 200 chars) ---")
    print(candidate.raw_codex_output[:200])
    print("=" * 70)
    print("NOTE: candidate NOT written to ~/.hermes/skills/ (smoke test only)")


if __name__ == "__main__":
    # Guard: refuse to run under pytest import
    if "pytest" in sys.modules:
        print("smoke_modifier: skipped (running under pytest)", file=sys.stderr)
        sys.exit(0)
    main()
