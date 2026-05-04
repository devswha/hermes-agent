"""Patina-based humanness scoring wrapper for DGM-H.

Calls the patina standalone CLI as a subprocess and parses the markdown
``--score`` output into a structured dict. Used to compute the human-likeness
component of the DGM-H composite reward signal.

Patina detects 0-100 "AI-likeness" patterns. We invert that score for fitness:
``human_likeness = 100 - ai_score``. Higher is better.

Usage:

    from dgmh.patina_judge import score_humanness, PatinaScoreError

    result = score_humanness("응 그 방향으로 가.")
    print(result.ai_score, result.human_likeness)

Cost: each call invokes the codex CLI subprocess inside patina; expect 5-30s
per call depending on text length. Plan callers around that latency.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_PATINA_BIN = str(
    Path.home() / ".hermes" / "skills" / "creative" / "patina" / "bin" / "patina.js"
)
_DEFAULT_TIMEOUT_S = 90.0


class PatinaScoreError(RuntimeError):
    """Raised when patina scoring fails (binary missing, timeout, parse error)."""

    def __init__(
        self,
        message: str,
        *,
        exit_code: Optional[int] = None,
        stderr: str = "",
        stdout: str = "",
    ) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = stderr
        self.stdout = stdout


@dataclass
class PatinaScore:
    """Parsed result of patina --score on a chunk of text.

    Attributes:
        ai_score: Overall AI-likeness 0-100 (higher = more AI-like).
        human_likeness: 100 - ai_score (higher = more human-like).
        sub_scores: Per-category raw score (e.g., {"communication": 66.7, ...}).
        interpretation: Patina's text interpretation tag (e.g., "human", "AI-like").
        elapsed_s: Wall-clock seconds the patina invocation took.
        raw_stdout: Original stdout for debugging.
    """

    ai_score: float
    human_likeness: float
    sub_scores: dict[str, float] = field(default_factory=dict)
    interpretation: str = ""
    elapsed_s: float = 0.0
    raw_stdout: str = ""


_OVERALL_RE = re.compile(
    r"\|\s*\*\*Overall\*\*\s*\|.*?\*\*([0-9]+\.?[0-9]*)\s*\(±\d+\)\*\*\s*\|",
    re.IGNORECASE,
)
_CATEGORY_RE = re.compile(
    r"^\|\s*([a-z_-]+)\s*\|\s*[0-9.]+\s*\|\s*[^|]*\|\s*([0-9]+\.?[0-9]*)\s*\|",
    re.MULTILINE,
)
_INTERPRETATION_RE = re.compile(
    r"Interpretation:\s*\*?\*?([^.\n*]+)", re.IGNORECASE
)


def _parse_score_output(stdout: str) -> tuple[float, dict[str, float], str]:
    overall_match = _OVERALL_RE.search(stdout)
    if not overall_match:
        raise PatinaScoreError(
            "could not find Overall score in patina output",
            stdout=stdout,
        )
    ai_score = float(overall_match.group(1))

    sub_scores: dict[str, float] = {}
    for cat_match in _CATEGORY_RE.finditer(stdout):
        category = cat_match.group(1).strip().lower()
        if category in {"category", "overall"}:
            continue
        sub_scores[category] = float(cat_match.group(2))

    interpretation = ""
    interp_match = _INTERPRETATION_RE.search(stdout)
    if interp_match:
        interpretation = interp_match.group(1).strip()

    return ai_score, sub_scores, interpretation


def score_humanness(
    text: str,
    *,
    lang: str = "ko",
    backend: str = "codex-cli",
    patina_bin: Optional[str] = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> PatinaScore:
    """Score the AI-likeness of ``text`` via the patina CLI.

    Args:
        text: The text to score (Korean by default).
        lang: Language code (ko, en, zh, ja). Patina auto-loads matching packs.
        backend: Patina LLM backend. ``codex-cli`` is free via ChatGPT OAuth;
                 ``openai-http`` requires an API key.
        patina_bin: Override path to patina.js. Defaults to the Hermes-managed
                    skill at ``~/.hermes/skills/creative/patina/bin/patina.js``.
        timeout_s: Subprocess timeout in seconds.

    Returns:
        PatinaScore with ai_score (0-100), human_likeness (100 - ai_score),
        per-category sub_scores, and the interpretation tag.

    Raises:
        PatinaScoreError: if the binary is missing, the call times out, or
                          the output cannot be parsed.
    """
    binary = patina_bin or os.environ.get("DGMH_PATINA_BIN", _DEFAULT_PATINA_BIN)
    if not Path(binary).exists():
        raise PatinaScoreError(
            f"patina binary not found at {binary!r}; "
            "set DGMH_PATINA_BIN or install patina under ~/.hermes/skills/creative/patina/"
        )

    if not text or not text.strip():
        return PatinaScore(
            ai_score=0.0,
            human_likeness=100.0,
            sub_scores={},
            interpretation="empty",
            elapsed_s=0.0,
            raw_stdout="",
        )

    args = [
        "node",
        binary,
        "--lang",
        lang,
        "--backend",
        backend,
        "--score",
    ]

    start = time.monotonic()
    try:
        result = subprocess.run(
            args,
            input=text,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            encoding="utf-8",
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - start
        raise PatinaScoreError(
            f"patina subprocess timed out after {elapsed:.1f}s",
            exit_code=None,
            stderr=str(exc.stderr or ""),
        ) from exc
    except FileNotFoundError as exc:
        raise PatinaScoreError(
            "node binary not found on PATH; install Node.js >= 22",
            exit_code=None,
            stderr=str(exc),
        ) from exc
    except OSError as exc:
        raise PatinaScoreError(
            f"patina spawn error: {exc}",
            exit_code=None,
            stderr=str(exc),
        ) from exc

    elapsed = time.monotonic() - start

    if result.returncode != 0:
        raise PatinaScoreError(
            f"patina exited with code {result.returncode} after {elapsed:.1f}s",
            exit_code=result.returncode,
            stderr=result.stderr or "",
            stdout=result.stdout or "",
        )

    ai_score, sub_scores, interpretation = _parse_score_output(result.stdout)
    return PatinaScore(
        ai_score=ai_score,
        human_likeness=100.0 - ai_score,
        sub_scores=sub_scores,
        interpretation=interpretation,
        elapsed_s=elapsed,
        raw_stdout=result.stdout,
    )


def composite_reward(
    *,
    human_likeness: float,
    codex_judge_quality: float,
    reaction_signal: float,
    weight_humanness: float = 0.50,
    weight_judge: float = 0.30,
    weight_reaction: float = 0.20,
) -> float:
    """Combine the three reward components into a single 0-100 score.

    Args:
        human_likeness: 100 - patina_ai_score, in [0, 100].
        codex_judge_quality: Codex-judge score normalized to [0, 100].
        reaction_signal: 100 for thumbs-up, 0 for thumbs-down, 50 for none.
        weight_*: Component weights. Should sum to 1.0.

    Returns:
        Weighted composite in [0, 100]. Caller multiplies by novelty_weight.
    """
    total_weight = weight_humanness + weight_judge + weight_reaction
    if abs(total_weight - 1.0) > 1e-6:
        logger.warning(
            "composite_reward weights sum to %.3f, not 1.0", total_weight
        )
    return (
        weight_humanness * human_likeness
        + weight_judge * codex_judge_quality
        + weight_reaction * reaction_signal
    )
