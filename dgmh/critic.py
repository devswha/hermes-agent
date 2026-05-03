"""
dgmh/critic.py — DGM-H Skill-domain Codex critic gate (W4 deliverable).

Port note: Python port of playground/dgmh-engine/codexCritic.ts::makeCodexCritic.
Subprocess invocation pattern mirrors modifier.py::_invoke_codex (same fail-closed
semantics as codexSubprocess.ts / codexCritic.ts).

Domain change vs. codexCritic.ts:
  - Parent/child are SkillCandidate + ArchiveEntry, not Agent objects.
  - Prompt placeholders: {{PARENT_SKILL_MD}}, {{CHILD_SKILL_MD}},
    {{ARCHIVE_SUMMARY}}, {{ATTEMPT}}
  - Output schema: { "verdict": "approve" | "reject", "reason": "<non-empty>" }

Fail-closed semantics (gen-0003 §(b)):
  - subprocess failure / timeout / spawn error → raises CodexCriticError
  - parse failure after maxParseAttempts re-rolls → raises CriticParseAttemptsExhaustedError
  - loop.py catches both and records as iteration-level "critic-error" rejection

Echo-chamber mitigation (gen-0003 §(a)):
  - distinct prompt template (skill-critic.md) — adversarial reviewer register
  - distinct call (separate subprocess, separate JSON envelope)
  - bias detector (W5) will catch consensus drift in archive

Reference: devswha/dgmh @ playground/dgmh-engine/codexCritic.ts
           dgmh/prompts/skill-critic.md (critic prompt template)
           hermes-skill-archive-dgmh-plan-v2-addendum.md §CONCERN 4 (W4 description)
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
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure project root importable
# ---------------------------------------------------------------------------

_HERMES_ROOT = Path(__file__).parent.parent
if str(_HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(_HERMES_ROOT))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT_S: float = 60.0
_DEFAULT_RECENT_N: int = 5
_DEFAULT_MAX_PARSE_ATTEMPTS: int = 2

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CodexCriticError(RuntimeError):
    """Raised when a Codex subprocess invocation fails (fail-closed).

    Mirrors CodexModifierError from modifier.py — same fail-closed semantics:
    non-zero exit / timeout / spawn error / empty stdout.
    No re-roll is attempted on CodexCriticError; the pipe is broken.
    Loop catches and records as iteration-level 'critic-error'.
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


class CriticParseAttemptsExhaustedError(RuntimeError):
    """Raised after max_parse_attempts re-rolls all produce unparseable output.

    Mirrors codexCritic.ts line 117-119: "N parse attempts failed".
    Loop catches and records as iteration-level 'critic-error'.
    """

    def __init__(self, message: str, *, attempts: int, last_error: str = "") -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


# ---------------------------------------------------------------------------
# CriticReview dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CriticReview:
    """Result of one critic invocation.

    Mirrors codexCritic.ts::CriticReview interface.
    elapsed_ms: wall-clock for the subprocess invocation; useful for budget tuning.
    """

    verdict: Literal["approve", "reject"]
    reason: str
    elapsed_ms: int


# ---------------------------------------------------------------------------
# CodexInvoker type alias
# ---------------------------------------------------------------------------

# callable(prompt: str, *, timeout_s: float) -> str
CodexInvoker = Callable[[str], str]


# ---------------------------------------------------------------------------
# Codex subprocess invocation (mirrors modifier.py::_invoke_codex)
# ---------------------------------------------------------------------------


def _invoke_codex(prompt: str, *, timeout_s: float = _DEFAULT_TIMEOUT_S) -> str:
    """Invoke Codex CLI via subprocess, return stdout. Fail-closed.

    Mirrors modifier.py::_invoke_codex — identical semantics:
    - stdin receives the prompt
    - timeout → CodexCriticError(timed_out=True)
    - non-zero exit → CodexCriticError
    - empty stdout → CodexCriticError
    - spawn error → CodexCriticError

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
        raise CodexCriticError(
            f"Codex subprocess timed out after {elapsed:.1f}s",
            timed_out=True,
            exit_code=None,
            stderr=str(exc.stderr or ""),
        ) from exc
    except FileNotFoundError as exc:
        raise CodexCriticError(
            f"Codex binary not found: {binary!r} — set DGMH_CODEX_BIN",
            exit_code=None,
            stderr=str(exc),
        ) from exc
    except OSError as exc:
        raise CodexCriticError(
            f"Codex spawn error: {exc}",
            exit_code=None,
            stderr=str(exc),
        ) from exc

    elapsed_ms = int((time.monotonic() - start) * 1000)

    if result.returncode != 0:
        raise CodexCriticError(
            f"Codex exited with code {result.returncode} (elapsed {elapsed_ms}ms)",
            exit_code=result.returncode,
            stderr=result.stderr or "",
        )

    if not result.stdout.strip():
        raise CodexCriticError(
            "Codex returned empty stdout",
            exit_code=result.returncode,
            stderr=result.stderr or "",
        )

    return result.stdout


# ---------------------------------------------------------------------------
# JSON envelope extraction (mirrors modifier.py::_extract_first_json_object)
# ---------------------------------------------------------------------------


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Find and parse the first balanced JSON object in text.

    Tolerates preamble prose before the object (mirrors codexCritic.ts::extractFirstJsonObject).
    Also strips markdown fences if the object is inside a fenced block.
    """
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
# Verdict envelope validation
# ---------------------------------------------------------------------------


def _parse_verdict_envelope(stdout: str) -> tuple[CriticReview | None, str]:
    """Parse and validate a critic verdict envelope from Codex stdout.

    Returns (CriticReview, "") on success or (None, error_message) on failure.

    Mirrors codexCritic.ts::parseVerdictEnvelope.
    Tolerates preamble text before the JSON object (fence-block tolerated).
    """
    obj = _extract_first_json_object(stdout)
    if obj is None:
        return None, "no JSON object found in stdout"

    raw_verdict = obj.get("verdict")
    if raw_verdict not in ("approve", "reject"):
        return (
            None,
            f'verdict must be "approve" or "reject"; got {json.dumps(raw_verdict)}',
        )

    reason = obj.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return None, "reason missing or not a non-empty string"

    return (
        CriticReview(
            verdict=raw_verdict,  # type: ignore[arg-type]
            reason=reason,
            elapsed_ms=0,  # filled by caller
        ),
        "",
    )


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def _format_archive_summary(archive: list[Any], recent_n: int) -> str:
    """Format archive entries for prompt injection (newest-first, limited to recent_n).

    Mirrors modifier.py::_format_archive_summary — same format for consistency.
    """
    if not archive or recent_n <= 0:
        return "(empty archive — child is the first proposed mutation)"

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


def _resolve_skill_md(obj: Any) -> str:
    """Extract SKILL.md content from a SkillCandidate or ArchiveEntry.

    Priority:
    1. obj.skill_md (SkillCandidate or any object with inline content)
    2. Read from obj.skill_path (ArchiveEntry)
    3. Read from obj.extra["skill_path"] (Generation)
    4. Fallback: repr the object
    """
    skill_md = getattr(obj, "skill_md", None)
    if isinstance(skill_md, str) and skill_md.strip():
        return skill_md

    skill_path = getattr(obj, "skill_path", None)
    if isinstance(skill_path, str) and skill_path:
        try:
            return Path(skill_path).read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("critic: could not read skill_path %r: %s", skill_path, exc)

    extra = getattr(obj, "extra", {})
    if isinstance(extra, dict):
        extra_path = extra.get("skill_path")
        if isinstance(extra_path, str) and extra_path:
            try:
                return Path(extra_path).read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning(
                    "critic: could not read extra.skill_path %r: %s", extra_path, exc
                )

    obj_id = getattr(obj, "skill_id", None) or getattr(obj, "id", "unknown")
    logger.warning(
        "critic: could not resolve SKILL.md content for %r; using placeholder", obj_id
    )
    return f"(skill content unavailable for {obj_id!r})"


def _assemble_prompt(
    template: str,
    child: Any,
    parent: Any,
    archive: list[Any],
    recent_n: int,
    attempt: int,
) -> str:
    """Assemble critic prompt by substituting template placeholders.

    Placeholders (from skill-critic.md):
      {{PARENT_SKILL_MD}}   — parent's SKILL.md content
      {{CHILD_SKILL_MD}}    — child's SKILL.md content
      {{ARCHIVE_SUMMARY}}   — recent archive summary (newest-first, recent_n)
      {{ATTEMPT}}           — 0-indexed attempt number
    """
    parent_skill_md = _resolve_skill_md(parent)
    child_skill_md = _resolve_skill_md(child)
    archive_summary = _format_archive_summary(archive, recent_n)

    return (
        template
        .replace("{{PARENT_SKILL_MD}}", parent_skill_md)
        .replace("{{CHILD_SKILL_MD}}", child_skill_md)
        .replace("{{ARCHIVE_SUMMARY}}", archive_summary)
        .replace("{{ATTEMPT}}", str(attempt))
    )


# ---------------------------------------------------------------------------
# SkillCritic object
# ---------------------------------------------------------------------------


class SkillCritic:
    """Critic object returned by make_codex_skill_critic.

    .review(child, parent, archive) → CriticReview

    Internal re-roll loop mirrors codexCritic.ts::makeCodexCritic return value.
    """

    def __init__(
        self,
        prompt_template: str,
        *,
        codex_invoker: CodexInvoker | None = None,
        recent_n: int = _DEFAULT_RECENT_N,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        max_parse_attempts: int = _DEFAULT_MAX_PARSE_ATTEMPTS,
    ) -> None:
        self._prompt_template = prompt_template
        self._codex_invoker = codex_invoker
        self._recent_n = recent_n
        self._timeout_s = timeout_s
        self._max_parse_attempts = max_parse_attempts

    def review(
        self,
        child: Any,
        parent: Any,
        archive: list[Any],
    ) -> CriticReview:
        """Invoke critic and return a verdict.

        Pipeline (mirrors codexCritic.ts::review):
          1. Assemble prompt (substitute placeholders)
          2. Invoke Codex — raises CodexCriticError on subprocess failure (no re-roll)
          3. Parse verdict envelope
          4. Re-roll on parse failure up to max_parse_attempts
          5. Raise CriticParseAttemptsExhaustedError after exhaustion

        Args:
            child: SkillCandidate or any object with .skill_md / .skill_path
            parent: ArchiveEntry or Generation with .skill_md / .skill_path
            archive: List of archive entries for context summary

        Returns:
            CriticReview with verdict, reason, elapsed_ms

        Raises:
            CodexCriticError: On subprocess failure (fail-closed, no re-roll)
            CriticParseAttemptsExhaustedError: After max_parse_attempts parse failures
        """
        last_parse_error = ""

        for attempt in range(self._max_parse_attempts):
            prompt = _assemble_prompt(
                self._prompt_template,
                child,
                parent,
                archive,
                self._recent_n,
                attempt,
            )

            start = time.monotonic()
            if self._codex_invoker is not None:
                # Injected invoker — wrap to measure elapsed
                stdout = self._codex_invoker(prompt)
            else:
                stdout = _invoke_codex(prompt, timeout_s=self._timeout_s)
            elapsed_ms = int((time.monotonic() - start) * 1000)

            review, err = _parse_verdict_envelope(stdout)
            if review is not None:
                review = dataclasses.replace(review, elapsed_ms=elapsed_ms)
                logger.info(
                    "critic: verdict=%r reason=%r attempt=%d elapsed_ms=%d",
                    review.verdict,
                    review.reason,
                    attempt,
                    elapsed_ms,
                )
                return review

            last_parse_error = err
            logger.debug(
                "critic: attempt %d — parse failure: %s", attempt, err
            )

        raise CriticParseAttemptsExhaustedError(
            f"codexSkillCritic: {self._max_parse_attempts} parse attempts failed; "
            f"last error: {last_parse_error}",
            attempts=self._max_parse_attempts,
            last_error=last_parse_error,
        )


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------


def make_codex_skill_critic(
    prompt_template: str,
    codex_invoker: CodexInvoker | None = None,
    recent_n: int = _DEFAULT_RECENT_N,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    max_parse_attempts: int = _DEFAULT_MAX_PARSE_ATTEMPTS,
) -> SkillCritic:
    """Create a SkillCritic for Hermes skill-domain critic gate.

    Port of playground/dgmh-engine/codexCritic.ts::makeCodexCritic.

    Args:
        prompt_template: Contents of dgmh/prompts/skill-critic.md. Loaded by
                         the caller (loop driver) and passed in — not read from
                         disk here so tests can supply a minimal stand-in.
        codex_invoker:   Optional callable(prompt: str) -> str.
                         Defaults to _invoke_codex (production Codex CLI).
                         Tests inject a mock here.
                         Note: timeout_s is passed separately; the invoker
                         receives only the prompt string.
        recent_n:        Newest-first archive rows to include in the prompt.
                         Default 5 (mirrors codexCritic.ts recentN default).
        timeout_s:       Per-call subprocess timeout in seconds. Default 60.
        max_parse_attempts: Re-roll attempts on JSON parse failure. Default 2
                         (mirrors codexCritic.ts maxParseAttempts=2; lower than
                         modifier's 3 because the critic output schema is shorter).

    Returns:
        SkillCritic with .review(child, parent, archive) → CriticReview

    Raises:
        ValueError: If prompt_template is empty or not a string.
        ValueError: If max_parse_attempts < 1.
    """
    if not prompt_template or not isinstance(prompt_template, str):
        raise ValueError(
            "make_codex_skill_critic: prompt_template (skill-critic.md contents) is required"
        )
    if max_parse_attempts < 1:
        raise ValueError("make_codex_skill_critic: max_parse_attempts must be >= 1")

    return SkillCritic(
        prompt_template,
        codex_invoker=codex_invoker,
        recent_n=recent_n,
        timeout_s=timeout_s,
        max_parse_attempts=max_parse_attempts,
    )
