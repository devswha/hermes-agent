"""
dgmh/judge.py — Codex-judge scorer for Hermes skills.

Port note: Subprocess invocation pattern mirrors
playground/dgmh-engine/codexSubprocess.ts — fail-closed, timeout, JSON
envelope extraction. Decision to use Codex-judge scorer documented in
hermes-skill-archive-dgmh-plan-v2-addendum.md §BLOCKER 2.

Fail-closed semantics (from codexSubprocess.ts):
  - non-zero exit    → raises CodexJudgeError
  - timeout          → raises CodexJudgeError (timedOut=True)
  - empty stdout     → raises CodexJudgeError
  - spawn error      → raises CodexJudgeError
  - no JSON found    → raises CodexJudgeError
  - empty eval_pack  → returns 0.5 baseline (not an error)

Reference: devswha/dgmh @ 3eddf82a661688f88b4bbf4c55028509d0eac002
  - playground/dgmh-engine/codexSubprocess.ts: subprocess pattern
  - hermes-skill-archive-dgmh-plan-v2-addendum.md §BLOCKER 2: Codex-judge design
  - hermes-skill-archive-dgmh-plan-v2-addendum.md §R-1: calibration requirements
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT_S: float = 180.0
_DEFAULT_CODEX_BIN: str = os.environ.get("DGMH_CODEX_BIN", "codex")


def _resolve_timeout_s(default: float = _DEFAULT_TIMEOUT_S) -> float:
    """Read DGMH_CODEX_TIMEOUT_S env override; fall back to default. Clamps
    to [10.0, 600.0] to avoid pathological config errors."""
    raw = os.environ.get("DGMH_CODEX_TIMEOUT_S")
    if not raw:
        return default
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return default
    return max(10.0, min(600.0, val))

# Score envelope that Codex must return per call
_SCORE_ENVELOPE_PROMPT = """\
You are a skill-quality judge. Given a Hermes skill definition and a task \
prompt, estimate how well the skill would guide an agent to produce a correct \
and useful response to the task.

Return ONLY a JSON object with this exact schema:
{{"score": <float between 0.0 and 1.0>, "reasoning": "<one sentence>"}}

Skill definition:
---
{skill_md_text}
---

Task prompt:
{task_prompt}
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class CodexJudgeError(RuntimeError):
    """Raised when a Codex subprocess invocation fails (fail-closed)."""

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


# ---------------------------------------------------------------------------
# JSON extraction — equivalent to extract_first_json_object in TS callers
# ---------------------------------------------------------------------------

def extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Find and parse the first JSON object in text. Returns None if not found."""
    # Find first '{' and try progressively longer substrings
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
                    # Try stripping markdown code fences
                    cleaned = re.sub(r"```[a-z]*\n?", "", candidate).strip()
                    try:
                        return json.loads(cleaned)
                    except json.JSONDecodeError:
                        pass
    return None


# ---------------------------------------------------------------------------
# Subprocess invocation (mirrors codexSubprocess.ts spawnSubprocess)
# ---------------------------------------------------------------------------

def _invoke_codex(prompt: str, *, timeout_s: float | None = None) -> str:
    """Invoke Codex CLI via subprocess, return stdout. Fail-closed.

    Mirrors codexSubprocess.ts::spawnSubprocess semantics:
    - stdin receives the prompt
    - timeout triggers SIGKILL → CodexJudgeError(timedOut=True)
    - non-zero exit → CodexJudgeError
    - empty stdout → CodexJudgeError
    - spawn error → CodexJudgeError

    Env overrides:
      DGMH_CODEX_BIN — codex binary path (default "codex")
      DGMH_CODEX_TIMEOUT_S — subprocess timeout in seconds (default 180,
                             clamped to [10, 600])
    """
    binary = os.environ.get("DGMH_CODEX_BIN", _DEFAULT_CODEX_BIN)
    args = [binary, "exec", "-"]

    if timeout_s is None:
        timeout_s = _resolve_timeout_s()

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
        raise CodexJudgeError(
            f"Codex subprocess timed out after {elapsed:.1f}s",
            timed_out=True,
            exit_code=None,
            stderr=str(exc.stderr or ""),
        ) from exc
    except FileNotFoundError as exc:
        raise CodexJudgeError(
            f"Codex binary not found: {binary!r} — set DGMH_CODEX_BIN",
            exit_code=None,
            stderr=str(exc),
        ) from exc
    except OSError as exc:
        raise CodexJudgeError(
            f"Codex spawn error: {exc}",
            exit_code=None,
            stderr=str(exc),
        ) from exc

    elapsed = time.monotonic() - start

    if result.returncode != 0:
        raise CodexJudgeError(
            f"Codex exited with code {result.returncode} after {elapsed:.1f}s",
            exit_code=result.returncode,
            stderr=result.stderr or "",
        )

    if not result.stdout.strip():
        raise CodexJudgeError(
            "Codex returned empty stdout",
            exit_code=result.returncode,
            stderr=result.stderr or "",
        )

    return result.stdout


# ---------------------------------------------------------------------------
# Single-call score extraction
# ---------------------------------------------------------------------------

def _score_single(skill_md_text: str, task_prompt: str, *, timeout_s: float) -> float:
    """Run one Codex judge call; return score in [0,1]. Raises CodexJudgeError."""
    prompt = _SCORE_ENVELOPE_PROMPT.format(
        skill_md_text=skill_md_text.strip(),
        task_prompt=task_prompt.strip(),
    )
    stdout = _invoke_codex(prompt, timeout_s=timeout_s)
    obj = extract_first_json_object(stdout)
    if obj is None:
        raise CodexJudgeError(
            f"No JSON object found in Codex output: {stdout[:200]!r}",
            exit_code=0,
            stderr="",
        )
    raw_score = obj.get("score")
    if raw_score is None:
        raise CodexJudgeError(
            f"JSON missing 'score' key: {obj}",
            exit_code=0,
            stderr="",
        )
    try:
        score = float(raw_score)
    except (TypeError, ValueError) as exc:
        raise CodexJudgeError(
            f"'score' is not a number: {raw_score!r}",
            exit_code=0,
            stderr="",
        ) from exc
    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_skill(
    skill_md_text: str,
    eval_prompt_pack: Sequence[str],
    *,
    seed: int | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> float:
    """Score a skill via Codex-judge; returns float in [0, 1].

    Per v2 §BLOCKER 2:
    - eval_prompt_pack is a list of task prompts for this skill's category.
    - Each task is sent to Codex independently; scores averaged.
    - Empty eval_prompt_pack → returns 0.5 baseline (unevaluated).
    - Any Codex subprocess failure raises CodexJudgeError (fail-closed).

    seed is accepted for API compatibility (deterministic prompt ordering)
    but does not alter subprocess behavior since Codex manages its own RNG.

    Args:
        skill_md_text: Full content of SKILL.md for the skill being judged.
        eval_prompt_pack: List of task prompt strings (5-12 per v2 plan).
        seed: Optional integer seed — used to deterministically order prompts
              for reproducibility testing.
        timeout_s: Per-call Codex subprocess timeout in seconds.

    Returns:
        Float in [0.0, 1.0]. Returns 0.5 when eval_prompt_pack is empty.

    Raises:
        CodexJudgeError: On any subprocess failure (fail-closed).
    """
    if not eval_prompt_pack:
        logger.debug("score_skill: empty eval_prompt_pack, returning 0.5 baseline")
        return 0.5

    # Optionally sort prompts deterministically by seed
    prompts = list(eval_prompt_pack)
    if seed is not None:
        import random as _random
        rng = _random.Random(seed)
        rng.shuffle(prompts)

    scores: list[float] = []
    for i, task_prompt in enumerate(prompts):
        try:
            s = _score_single(skill_md_text, task_prompt, timeout_s=timeout_s)
            scores.append(s)
            logger.debug("score_skill: task %d/%d → %.3f", i + 1, len(prompts), s)
        except CodexJudgeError:
            logger.error("score_skill: Codex judge failed on task %d", i + 1, exc_info=True)
            raise

    if not scores:
        return 0.5

    avg = sum(scores) / len(scores)
    logger.debug("score_skill: avg=%.3f over %d tasks", avg, len(scores))
    return avg
