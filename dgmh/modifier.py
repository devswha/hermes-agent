"""
dgmh/modifier.py — DGM-H Skill Modifier (W3 deliverable).

Port note: Python port of playground/dgmh-engine/llmModifier.ts::makeCodexLlmModifier.
Subprocess invocation pattern reuses judge.py::_invoke_codex (mirrors codexSubprocess.ts).
Re-roll semantics and JSON-envelope extraction are identical to the TS source.

Domain change vs. llmModifier.ts:
  - Parent is ArchiveEntry / Generation (not Agent with prompts+agentsConfig)
  - Output envelope: { skill_id, skill_md, lineage } (not { id, prompts, agentsConfig, lineage })
  - Scope check: deny paths outside ~/.hermes/skills/ (tools/, gateway/, agent/, run_agent.py etc.)

Re-roll semantics (mirrors llmModifier.ts gen-0003 §(b) fail-closed):
  - subprocess failure (non-zero exit / timeout / spawn error / empty stdout)
      → raises CodexModifierError immediately (no re-roll; broken pipe)
  - JSON envelope parse failure → re-roll
  - scope violation → re-roll
  - exhausted max_scope_attempts → raises ScopeAttemptsExhaustedError

Reference: devswha/dgmh @ playground/dgmh-engine/llmModifier.ts
           devswha/dgmh @ playground/dgmh-engine/selfModScope.ts (scope guard intent)
           hermes-skill-archive-dgmh-plan-v2-addendum.md §CONCERN 4 (W3 description)
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure project root importable for cross-module imports
# ---------------------------------------------------------------------------

_HERMES_ROOT = Path(__file__).parent.parent
if str(_HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(_HERMES_ROOT))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT_S: float = 60.0
_DEFAULT_RECENT_N: int = 5
_DEFAULT_MAX_SCOPE_ATTEMPTS: int = 3

# Denied path substrings for skill-domain scope guard
# (mirrors selfModScope.ts SELF_MOD_DENIED_PATTERNS, adapted for Hermes)
_SKILL_SCOPE_DENIED_PATTERNS: tuple[str, ...] = (
    "tools/",
    "gateway/",
    "agent/",
    "run_agent.py",
    "hermes_agent.py",
    "mcp_serve.py",
    "batch_runner.py",
    "cli.py",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CodexModifierError(RuntimeError):
    """Raised when a Codex subprocess invocation fails (fail-closed).

    Mirrors CodexJudgeError from judge.py — same fail-closed semantics as
    llmModifier.ts: non-zero exit / timeout / spawn error / empty stdout.
    No re-roll is attempted on CodexModifierError; the pipe is broken.
    """

    def __init__(
        self,
        message: str,
        *,
        timed_out: bool = False,
        exit_code: int | None = None,
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.timed_out = timed_out
        self.exit_code = exit_code
        self.stderr = stderr


class ScopeAttemptsExhaustedError(RuntimeError):
    """Raised after max_scope_attempts re-rolls all fail parse/scope checks.

    Mirrors llmModifier.ts line 113: "N attempts produced no scope-valid child".
    """

    def __init__(self, message: str, *, parent_id: str, attempts: int) -> None:
        super().__init__(message)
        self.parent_id = parent_id
        self.attempts = attempts


# ---------------------------------------------------------------------------
# SkillCandidate dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SkillCandidate:
    """Parsed output of one modifier invocation.

    Mirrors the envelope shape required by the skill-modifier prompt:
      { skill_id: "<cat>/<name>", skill_md: "<full markdown>", lineage: [...] }

    Extra fields for audit:
      raw_codex_output: full stdout from Codex (before parsing)
      attempt: 0-indexed re-roll attempt number that produced this candidate
    """

    skill_id: str               # "<category>/<name>"
    skill_md: str               # full SKILL.md content (mutated)
    lineage: list[str]          # [...parent.lineage, parent.skill_id]
    raw_codex_output: str       # Codex stdout verbatim
    attempt: int                # 0-indexed attempt that succeeded


# ---------------------------------------------------------------------------
# Codex subprocess invocation (reuses judge.py pattern)
# ---------------------------------------------------------------------------


def _invoke_codex(prompt: str, *, timeout_s: float = _DEFAULT_TIMEOUT_S) -> str:
    """Invoke Codex CLI via subprocess, return stdout. Fail-closed.

    Mirrors judge.py::_invoke_codex — identical semantics:
    - stdin receives the prompt
    - timeout → CodexModifierError(timed_out=True)
    - non-zero exit → CodexModifierError
    - empty stdout → CodexModifierError
    - spawn error → CodexModifierError

    Env override: DGMH_CODEX_BIN (default "codex")
    """
    binary = os.environ.get("DGMH_CODEX_BIN", "codex")
    args = [binary, "exec", "-"]

    start = time.monotonic()
    try:
        result = subprocess.run(
            args,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            encoding="utf-8",
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - start
        raise CodexModifierError(
            f"Codex subprocess timed out after {elapsed:.1f}s",
            timed_out=True,
            exit_code=None,
            stderr=str(exc.stderr or ""),
        ) from exc
    except FileNotFoundError as exc:
        raise CodexModifierError(
            f"Codex binary not found: {binary!r} — set DGMH_CODEX_BIN",
            exit_code=None,
            stderr=str(exc),
        ) from exc
    except OSError as exc:
        raise CodexModifierError(
            f"Codex spawn error: {exc}",
            exit_code=None,
            stderr=str(exc),
        ) from exc

    elapsed = time.monotonic() - start

    if result.returncode != 0:
        raise CodexModifierError(
            f"Codex exited with code {result.returncode} after {elapsed:.1f}s",
            exit_code=result.returncode,
            stderr=result.stderr or "",
        )

    if not result.stdout.strip():
        raise CodexModifierError(
            "Codex returned empty stdout",
            exit_code=result.returncode,
            stderr=result.stderr or "",
        )

    return result.stdout


# ---------------------------------------------------------------------------
# JSON envelope extraction (mirrors judge.py::extract_first_json_object)
# ---------------------------------------------------------------------------


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Find and parse the first balanced JSON object in text."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    cleaned = re.sub(r"```[a-z]*\n?", "", candidate).strip()
                    try:
                        return json.loads(cleaned)
                    except json.JSONDecodeError:
                        pass
    return None


# ---------------------------------------------------------------------------
# Envelope validation
# ---------------------------------------------------------------------------


def _validate_envelope(
    raw: dict[str, Any],
    parent_id: str,
    parent_lineage: list[str],
) -> tuple[SkillCandidate | None, str]:
    """Validate parsed JSON envelope against required field constraints.

    Returns (SkillCandidate, "") on success or (None, error_message) on failure.

    Constraints (from skill-modifier.md prompt spec):
      - skill_id: non-empty string matching "<cat>/<name>" (one slash, no leading slash)
      - skill_md: non-empty string
      - lineage: non-empty list[str] that includes parent_id as last element
    """
    skill_id = raw.get("skill_id")
    if not isinstance(skill_id, str) or not skill_id:
        return None, "skill_id missing or not a non-empty string"

    # Must match <cat>/<name> pattern: exactly one slash, both parts non-empty
    parts = skill_id.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None, f"skill_id {skill_id!r} does not match <category>/<name> pattern"

    skill_md = raw.get("skill_md")
    if not isinstance(skill_md, str) or not skill_md.strip():
        return None, "skill_md missing or empty"

    lineage_raw = raw.get("lineage")
    if not isinstance(lineage_raw, list):
        return None, "lineage missing or not a list"
    lineage: list[str] = [s for s in lineage_raw if isinstance(s, str)]
    if not lineage:
        return None, "lineage must be a non-empty list of strings"
    if lineage[-1] != parent_id:
        return None, (
            f"lineage last element {lineage[-1]!r} != parent_id {parent_id!r}; "
            "expected [...parent.lineage, parent.skill_id]"
        )

    return (
        SkillCandidate(
            skill_id=skill_id,
            skill_md=skill_md,
            lineage=lineage,
            raw_codex_output="",  # filled by caller
            attempt=0,            # filled by caller
        ),
        "",
    )


# ---------------------------------------------------------------------------
# Scope guard (port of selfModScope.ts intent for Hermes skill domain)
# ---------------------------------------------------------------------------


def _check_skill_scope(skill_md: str) -> list[str]:
    """Return list of scope violations found in skill_md content.

    Rejects if skill_md mentions any path outside ~/.hermes/skills/ surface.
    Uses substring match on _SKILL_SCOPE_DENIED_PATTERNS (case-sensitive),
    matching the TS selfModScope.ts SELF_MOD_DENIED_PATTERNS approach.

    Returns empty list if scope is clean.
    """
    violations: list[str] = []
    for pattern in _SKILL_SCOPE_DENIED_PATTERNS:
        if pattern in skill_md:
            violations.append(f"skill_md contains denied path pattern {pattern!r}")
    return violations


# ---------------------------------------------------------------------------
# Prompt assembly (mirrors llmModifier.ts::assemblePrompt)
# ---------------------------------------------------------------------------


def _format_archive_summary(archive: list[Any], recent_n: int) -> str:
    """Format archive entries for prompt injection.

    Mirrors llmModifier.ts::formatArchive: newest-first, limited to recent_n.
    Each entry rendered as: - t=<generation_index> id=<id> score=<score> compiled_children=<n>

    Accepts any object with .generation_index, .skill_id (or .id), .score, .compiled_children.
    Falls back to empty-archive marker for empty list.
    """
    if not archive or recent_n <= 0:
        return "(empty archive — this is the first child)"

    # Newest-first: sort by generation_index descending, take recent_n
    def _gen_index(entry: Any) -> int:
        return getattr(entry, "generation_index", 0)

    sorted_entries = sorted(archive, key=_gen_index, reverse=True)
    recent = sorted_entries[:recent_n]

    lines = []
    for entry in recent:
        entry_id = getattr(entry, "skill_id", None) or getattr(entry, "id", "unknown")
        score = getattr(entry, "score", 0.0)
        compiled_children = getattr(entry, "compiled_children", 0)
        gen_index = getattr(entry, "generation_index", 0)
        lines.append(
            f"- t={gen_index} id={entry_id} score={score:.4f} compiled_children={compiled_children}"
        )
    return "\n".join(lines)


def _assemble_prompt(
    template: str,
    parent: Any,
    archive: list[Any],
    recent_n: int,
    attempt: int,
) -> str:
    """Assemble modifier prompt by substituting template placeholders.

    Placeholders (from skill-modifier.md):
      {{PARENT_SKILL_MD}}   — parent's SKILL.md content
      {{ARCHIVE_SUMMARY}}   — recent archive summary (newest-first, limited to recent_n)
      {{ATTEMPT}}           — 0-indexed attempt number (higher = previous attempts failed)
    """
    # Resolve parent skill_md content from the parent object
    # Supports ArchiveEntry (has .skill_path), Generation (has .extra["skill_path"]),
    # or any object with .skill_md directly.
    parent_skill_md = _resolve_parent_skill_md(parent)

    archive_summary = _format_archive_summary(archive, recent_n)

    return (
        template
        .replace("{{PARENT_SKILL_MD}}", parent_skill_md)
        .replace("{{ARCHIVE_SUMMARY}}", archive_summary)
        .replace("{{ATTEMPT}}", str(attempt))
    )


def _resolve_parent_skill_md(parent: Any) -> str:
    """Extract SKILL.md content from a parent entry.

    Priority:
    1. parent.skill_md (SkillCandidate or mock with inline content)
    2. Read from parent.skill_path (ArchiveEntry)
    3. Read from parent.extra["skill_path"] (Generation)
    4. Fallback: repr the parent object (graceful degradation)
    """
    # Direct attribute
    skill_md = getattr(parent, "skill_md", None)
    if isinstance(skill_md, str) and skill_md.strip():
        return skill_md

    # skill_path attribute (ArchiveEntry)
    skill_path = getattr(parent, "skill_path", None)
    if isinstance(skill_path, str) and skill_path:
        try:
            return Path(skill_path).read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("modifier: could not read skill_path %r: %s", skill_path, exc)

    # extra dict (Generation from select_parents.py)
    extra = getattr(parent, "extra", {})
    if isinstance(extra, dict):
        extra_path = extra.get("skill_path")
        if isinstance(extra_path, str) and extra_path:
            try:
                return Path(extra_path).read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning(
                    "modifier: could not read extra.skill_path %r: %s", extra_path, exc
                )

    # Graceful fallback — caller should supply valid parent
    parent_id = getattr(parent, "skill_id", None) or getattr(parent, "id", "unknown")
    logger.warning("modifier: could not resolve SKILL.md content for parent %r; using placeholder", parent_id)
    return f"(parent skill content unavailable for {parent_id!r})"


# ---------------------------------------------------------------------------
# SkillModifier object (returned by make_codex_skill_modifier)
# ---------------------------------------------------------------------------


class SkillModifier:
    """Modifier object returned by make_codex_skill_modifier.

    .modify(parent, archive) → SkillCandidate

    Internal re-roll loop mirrors llmModifier.ts::makeCodexLlmModifier return value.
    """

    def __init__(
        self,
        prompt_template: str,
        *,
        codex_invoker: Any | None = None,
        recent_n: int = _DEFAULT_RECENT_N,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        max_scope_attempts: int = _DEFAULT_MAX_SCOPE_ATTEMPTS,
    ) -> None:
        self._prompt_template = prompt_template
        self._codex_invoker = codex_invoker  # callable(prompt, timeout_s) → str, or None → _invoke_codex
        self._recent_n = recent_n
        self._timeout_s = timeout_s
        self._max_scope_attempts = max_scope_attempts

    def modify(self, parent: Any, archive: list[Any]) -> SkillCandidate:
        """Propose one child skill by invoking Codex and parsing the envelope.

        Pipeline (mirrors llmModifier.ts::modify):
          1. Assemble prompt (substitute placeholders)
          2. Invoke Codex (raises CodexModifierError on subprocess failure — no re-roll)
          3. Extract first balanced JSON object from stdout
          4. Validate envelope shape (skill_id pattern, skill_md non-empty, lineage)
          5. Check skill-domain scope guard
          6. Re-roll on parse/scope failure up to max_scope_attempts
          7. Raise ScopeAttemptsExhaustedError after exhaustion

        Args:
            parent: ArchiveEntry or Generation (or any object with skill_id/id + skill_path)
            archive: List of archive entries for archive summary

        Returns:
            SkillCandidate with validated envelope + raw_codex_output + attempt index

        Raises:
            CodexModifierError: On subprocess failure (fail-closed, no re-roll)
            ScopeAttemptsExhaustedError: After max_scope_attempts all fail parse/scope
        """
        parent_id: str = getattr(parent, "skill_id", None) or getattr(parent, "id", "unknown")
        parent_lineage: list[str] = list(getattr(parent, "lineage", []) or [])

        last_error = ""

        for attempt in range(self._max_scope_attempts):
            prompt = _assemble_prompt(
                self._prompt_template,
                parent,
                archive,
                self._recent_n,
                attempt,
            )

            # Invoke Codex — raises CodexModifierError on failure (no re-roll)
            if self._codex_invoker is not None:
                stdout = self._codex_invoker(prompt, timeout_s=self._timeout_s)
            else:
                stdout = _invoke_codex(prompt, timeout_s=self._timeout_s)

            # Extract JSON envelope
            raw_obj = _extract_first_json_object(stdout)
            if raw_obj is None:
                last_error = "parse: no JSON object found in stdout"
                logger.debug(
                    "modifier: attempt %d — parse failure for parent %r: %s",
                    attempt, parent_id, last_error,
                )
                continue

            # Validate envelope shape
            candidate, err = _validate_envelope(raw_obj, parent_id, parent_lineage)
            if candidate is None:
                last_error = f"parse: {err}"
                logger.debug(
                    "modifier: attempt %d — envelope validation failure for parent %r: %s",
                    attempt, parent_id, last_error,
                )
                continue

            # Scope check
            violations = _check_skill_scope(candidate.skill_md)
            if violations:
                last_error = "scope: " + "; ".join(violations)
                logger.debug(
                    "modifier: attempt %d — scope violation for parent %r: %s",
                    attempt, parent_id, last_error,
                )
                continue

            # Success — fill in audit fields and return
            candidate.raw_codex_output = stdout
            candidate.attempt = attempt
            logger.info(
                "modifier: produced candidate skill_id=%r lineage_depth=%d attempt=%d",
                candidate.skill_id,
                len(candidate.lineage),
                attempt,
            )
            return candidate

        # All attempts exhausted
        raise ScopeAttemptsExhaustedError(
            f"codexSkillModifier: {self._max_scope_attempts} attempts produced no "
            f"scope-valid child for parent {parent_id!r}; last error: {last_error}",
            parent_id=parent_id,
            attempts=self._max_scope_attempts,
        )


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------


def make_codex_skill_modifier(
    prompt_template: str,
    codex_invoker: Any | None = None,
    recent_n: int = _DEFAULT_RECENT_N,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    max_scope_attempts: int = _DEFAULT_MAX_SCOPE_ATTEMPTS,
) -> SkillModifier:
    """Create a SkillModifier for Hermes skill mutations.

    Port of playground/dgmh-engine/llmModifier.ts::makeCodexLlmModifier.

    Args:
        prompt_template: Contents of dgmh/prompts/skill-modifier.md. Loaded by
                         the caller (loop driver) and passed in — not read from
                         disk here so tests can supply a minimal stand-in.
        codex_invoker:   Optional callable(prompt: str, *, timeout_s: float) -> str.
                         Defaults to _invoke_codex (production Codex CLI via subprocess).
                         Tests inject a mock here.
        recent_n:        Newest-first archive rows to include in the prompt summary.
                         Default 5 (mirrors llmModifier.ts recentN default).
        timeout_s:       Per-call subprocess timeout in seconds. Default 60.
        max_scope_attempts: Re-roll attempts on parse / scope rejection. Default 3.
                         Aligned with gen-0003 §(b) maxIterPerGeneration family.

    Returns:
        SkillModifier with .modify(parent, archive) → SkillCandidate

    Raises:
        ValueError: If prompt_template is empty or not a string.
        ValueError: If max_scope_attempts < 1.
    """
    if not prompt_template or not isinstance(prompt_template, str):
        raise ValueError(
            "make_codex_skill_modifier: prompt_template (skill-modifier.md contents) is required"
        )
    if max_scope_attempts < 1:
        raise ValueError("make_codex_skill_modifier: max_scope_attempts must be >= 1")

    return SkillModifier(
        prompt_template,
        codex_invoker=codex_invoker,
        recent_n=recent_n,
        timeout_s=timeout_s,
        max_scope_attempts=max_scope_attempts,
    )
