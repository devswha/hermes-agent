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


_REWRITE_PROMPT_TEMPLATE = """다음은 디스코드 1:1 캐주얼 채팅에서 한 봇의 응답이야. 이 응답에는 ChatGPT 처럼 보이는 구조적 anti-pattern 들이 들어 있어. 같은 의미를 유지하면서 그 anti-pattern 들만 제거해서 다시 써줘.

제거할 것:
- bullet/번호 리스트 4개 이상 평행 구조 (필요하면 1-2개로 줄이거나 prose 로 풀어쓰기)
- bold markdown 라벨 (`**핵심:**`, `**주제:**`, `**요약:**`, `**결론:**`)
- 콜론 introducing list 패턴 (`이렇게 가자:` 뒤에 bullet 들)
- 닫는 caveat hedge (마지막 문단/문장이 `그래도`, `다만`, `물론`, `한편` 으로 시작하면서 본론을 약화시키는 경우)
- 결론 신호어 (`결론적으로`, `요컨대`, `종합하면`)
- 챗봇 filler (`좋은 질문`, `도움이 되셨으면`, `궁금한 점이 있으시면`)
- 학술풍 phrasing (`가능성이 커`, `~로 변모`, `비중이 내려가고`)
- 자기 bio recitation (운영자가 명시적으로 묻지 않는 한 절대 X): `Hermes Agent`, `DGM-H`, `self-evolution`, `persona`, `개인 어시스턴트`, `예전 flask 프로젝트`, `flask 프로젝트랑 무관`, `튜닝되는 skill` 같은 표현 전부 제거. 운영자는 이미 알고 있으니 매번 자기 소개 X. 단순히 이름만 답하고 끝낼 것 (예: `flask 야.`).

유지할 것:
- 모든 사실 / 수치 / 고유명사 / 영어 기술 토큰 (SOUL, DGM-H, gpt-5.5 등)
- 코드 블록 (\`\`\` 펜스 안의 내용은 절대 수정하지 말고 그대로 보존)
- 운영자 어조: 1:1 디스코드 캐주얼 반말 (`~해`, `~야`, `~이지`, `~거든`)
- 응답의 핵심 메시지

출력 규칙:
- 1-4문장의 짧고 흐르는 prose 로 줄여 (캐주얼 reply 면 1-2문장이 이상적)
- 가능하면 120자 이하
- 메타 설명/preamble 없이 다시 쓴 한국어 텍스트만 출력
- 수정 사항 설명 절대 X

원본 응답:
---
{text}
---

다시 쓴 응답 (한국어 텍스트만):"""


_REWRITE_DEFAULT_TIMEOUT_S = 60.0


def humanness_rewrite(
    text: str,
    *,
    timeout_s: float = _REWRITE_DEFAULT_TIMEOUT_S,
    codex_bin: Optional[str] = None,
) -> Optional[str]:
    """Rewrite a Korean Discord reply to strip chatgpt-style anti-patterns.

    Calls the Codex CLI via subprocess with a prompt that names exactly the
    patterns to remove and the operator's voice traits to preserve. Returns
    the rewritten text, or None if Codex fails / times out / produces empty
    output.

    The caller decides what to do with the result (e.g., replace the
    outbound message content). On any failure the caller falls back to the
    original text — never blocking the send path.
    """
    if not text or not text.strip():
        return None

    binary = codex_bin or os.environ.get("DGMH_CODEX_BIN", "codex")
    prompt = _REWRITE_PROMPT_TEMPLATE.format(text=text)

    try:
        result = subprocess.run(
            [binary, "exec", "-"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            encoding="utf-8",
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.warning("humanness_rewrite: codex invocation failed: %s", exc)
        return None

    if result.returncode != 0:
        logger.warning(
            "humanness_rewrite: codex exited %d; stderr=%s",
            result.returncode,
            (result.stderr or "")[:200],
        )
        return None

    out = (result.stdout or "").strip()
    # Codex often echoes a thinking/header preamble before the actual answer.
    # Heuristic: take everything after the last "---" separator if present;
    # otherwise return the whole stdout. Sanity-check by length and content.
    if "\n---\n" in out:
        out = out.rsplit("\n---\n", 1)[-1].strip()
    if not out or len(out) < 5:
        return None
    return out


_REWRITE_PROFILE_DEFAULT_TIMEOUT_S = 30.0


def humanness_rewrite_with_profile(
    text: str,
    *,
    profile: str = "social",
    backend: str = "codex-cli",
    lang: str = "ko",
    timeout_s: float = _REWRITE_PROFILE_DEFAULT_TIMEOUT_S,
    patina_bin: Optional[str] = None,
) -> Optional[str]:
    """Programmatic invocation of ``patina --profile <profile> --backend <backend>``.

    This is the public-channel rewrite surface (Step P1, plan v3). The
    existing :func:`humanness_rewrite` stays unchanged so the 1:1 humanness
    pipeline keeps its current behavior.

    Patina without ``--score`` / ``--audit`` runs in rewrite mode: stdin
    carries the source text, stdout carries the rewritten text. The
    ``social`` profile amplifies casual / fragment / first-person voice,
    which is what we want for the public persona.

    Args:
        text: Source Korean Discord reply.
        profile: Patina profile. Default ``"social"`` for casual public chat.
        backend: Patina LLM backend. Default ``"codex-cli"`` (free via Codex
            ChatGPT OAuth — no API key needed).
        lang: Language code passed to patina. Default ``"ko"``.
        timeout_s: Subprocess timeout in seconds.
        patina_bin: Optional override for the patina.js path.

    Returns:
        The rewritten text on success, or ``None`` on any failure (binary
        missing, timeout, non-zero exit, empty output). On ``None`` the
        caller decides the fallback — typical chain is:
        ``humanness_rewrite_with_profile`` → existing ``humanness_rewrite``
        → original text unchanged.
    """
    if not text or not text.strip():
        return None

    binary = patina_bin or os.environ.get("DGMH_PATINA_BIN", _DEFAULT_PATINA_BIN)
    if not Path(binary).exists():
        logger.warning(
            "humanness_rewrite_with_profile: patina binary missing at %s", binary
        )
        return None

    args = [
        "node",
        binary,
        "--lang",
        lang,
        "--profile",
        profile,
        "--backend",
        backend,
    ]

    try:
        result = subprocess.run(
            args,
            input=text,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            encoding="utf-8",
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.warning(
            "humanness_rewrite_with_profile: patina invocation failed: %s", exc
        )
        return None

    if result.returncode != 0:
        logger.warning(
            "humanness_rewrite_with_profile: patina exited %d; stderr=%s",
            result.returncode,
            (result.stderr or "")[:200],
        )
        return None

    out = (result.stdout or "").strip()
    if not out:
        return None
    # Patina prepends an analyst preamble paragraph (e.g.
    # "아직 AI 티 나는 부분: ...") before the actual rewrite. Drop ONLY the
    # first block if it matches the preamble heuristic; join the rest so
    # legitimate multi-paragraph rewrites are preserved intact.
    blocks = re.split(r"\n\s*\n", out)
    if blocks:
        first = blocks[0].strip()
        is_preamble = bool(
            re.match(r"^(아직|남은)\s*AI\s*티", first)
            or re.search(r"AI\s*티", first[:60])
        )
        if is_preamble and len(blocks) > 1:
            blocks = blocks[1:]
    out = "\n\n".join(b.strip() for b in blocks if b.strip())
    if not out or len(out) < 5:
        return None
    return out


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
