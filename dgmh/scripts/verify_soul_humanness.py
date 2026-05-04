"""SOUL.md humanness verification driver.

Bypasses Discord. For each canonical test prompt, calls ``hermes chat``
locally with the active SOUL.md, captures the assistant response, scores
it via patina, and appends a structured run record to
``~/.hermes/dgmh/verification_runs.jsonl``.

Compares against the previous run (filtered by SOUL.md hash) and prints
a diff: per-prompt AI score change, per-category change, and pass/fail
verdict against the threshold.

Pass criteria (default): mean ai_score across prompts <= 10.

Run with:

    cd /home/devswha/workspace/hermes-agent
    python -m dgmh.scripts.verify_soul_humanness

Env knobs:
    DGMH_VERIFY_THRESHOLD     Mean ai_score threshold (default 10.0)
    DGMH_VERIFY_PROMPT_LIMIT  Limit prompt count (default all)
    DGMH_VERIFY_CHAT_TIMEOUT  hermes chat timeout seconds (default 90)
    DGMH_VERIFY_HERMES_BIN    Override hermes binary path
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dgmh.patina_judge import PatinaScoreError, score_humanness

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical Korean Discord casual prompt set (operator-confirmed 2026-05-04).
# These exercise the SOUL.md surface across casual chat, identity questions,
# topic-pick prompts, technical Q (short + long-prone), opinion questions.
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS = [
    "야",
    "너 누구야?",
    "오늘 뭐할까?",
    "AI 에이전트 어떻게 생각해?",
    "Discord 봇 만드는 법 알려줘",
    "Python dict comprehension 어떻게 써?",
    "GitHub Actions YAML 예시 보여줘",
    "잘 안되네 어떡하지",
    "대화 주제 하나 정해봐",
    "요즘 너 뭐하냐",
]


_DEFAULT_THRESHOLD = 10.0
_DEFAULT_CHAT_TIMEOUT_S = 90.0


@dataclass
class PromptResult:
    prompt: str
    response: str
    ai_score: Optional[float]
    human_likeness: Optional[float]
    sub_scores: dict[str, float] = field(default_factory=dict)
    interpretation: str = ""
    chat_elapsed_s: float = 0.0
    score_elapsed_s: float = 0.0
    error: str = ""


@dataclass
class VerifyRun:
    run_id: str
    started_at: str
    finished_at: str
    soul_md_hash: str
    soul_md_version: str
    threshold: float
    mean_ai_score: float
    mean_human_likeness: float
    pass_: bool
    n_prompts: int
    n_scored: int
    per_category_mean: dict[str, float] = field(default_factory=dict)
    results: list[PromptResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


def _runs_path() -> Path:
    p = _hermes_home() / "dgmh" / "verification_runs.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _read_soul_metadata() -> tuple[str, str]:
    """Return (sha256_hex, version) of the active SOUL.md."""
    soul = _hermes_home() / "SOUL.md"
    try:
        content = soul.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "", ""
    h = hashlib.sha256(content.encode("utf-8")).hexdigest()
    version = ""
    for line in content.splitlines():
        if line.strip().startswith("version:"):
            version = line.split(":", 1)[1].strip()
            break
    return h, version


def _hermes_bin() -> str:
    override = os.environ.get("DGMH_VERIFY_HERMES_BIN")
    if override:
        return override
    repo_bin = Path(__file__).resolve().parents[2] / "venv" / "bin" / "hermes"
    if repo_bin.exists():
        return str(repo_bin)
    return "hermes"


# ---------------------------------------------------------------------------
# Hermes chat invocation
# ---------------------------------------------------------------------------


def _strip_session_header(text: str) -> str:
    """Drop the leading ``session_id: ...`` line that hermes chat prepends."""
    lines = text.splitlines()
    if lines and lines[0].startswith("session_id:"):
        return "\n".join(lines[1:]).strip()
    return text.strip()


def call_hermes_chat(prompt: str, *, timeout_s: float) -> tuple[str, float]:
    binary = _hermes_bin()
    args = [binary, "chat", "-Q", "--source", "tool", "-q", prompt]
    start = time.monotonic()
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        encoding="utf-8",
    )
    elapsed = time.monotonic() - start
    if result.returncode != 0:
        raise RuntimeError(
            f"hermes chat exited {result.returncode}: stderr={result.stderr[:300]}"
        )
    response = _strip_session_header(result.stdout)
    if not response:
        raise RuntimeError("hermes chat returned empty response")
    return response, elapsed


# ---------------------------------------------------------------------------
# Verification flow
# ---------------------------------------------------------------------------


def verify_single(prompt: str, *, chat_timeout_s: float) -> PromptResult:
    try:
        response, chat_elapsed = call_hermes_chat(prompt, timeout_s=chat_timeout_s)
    except (subprocess.TimeoutExpired, RuntimeError) as exc:
        return PromptResult(
            prompt=prompt,
            response="",
            ai_score=None,
            human_likeness=None,
            error=f"chat-error: {type(exc).__name__}: {exc}",
        )

    try:
        score = score_humanness(response, lang="ko")
    except PatinaScoreError as exc:
        return PromptResult(
            prompt=prompt,
            response=response,
            ai_score=None,
            human_likeness=None,
            chat_elapsed_s=chat_elapsed,
            error=f"score-error: {exc}",
        )

    return PromptResult(
        prompt=prompt,
        response=response,
        ai_score=score.ai_score,
        human_likeness=score.human_likeness,
        sub_scores=score.sub_scores,
        interpretation=score.interpretation,
        chat_elapsed_s=chat_elapsed,
        score_elapsed_s=score.elapsed_s,
    )


def run_verification(
    prompts: list[str],
    *,
    threshold: float = _DEFAULT_THRESHOLD,
    chat_timeout_s: float = _DEFAULT_CHAT_TIMEOUT_S,
) -> VerifyRun:
    soul_hash, soul_version = _read_soul_metadata()
    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%S")

    results: list[PromptResult] = []
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] prompt={prompt!r}", flush=True)
        r = verify_single(prompt, chat_timeout_s=chat_timeout_s)
        if r.error:
            print(f"  -> ERROR: {r.error}", flush=True)
        else:
            print(
                f"  -> ai={r.ai_score:.1f} human={r.human_likeness:.1f} "
                f"chat={r.chat_elapsed_s:.1f}s score={r.score_elapsed_s:.1f}s",
                flush=True,
            )
        results.append(r)

    scored = [r for r in results if r.ai_score is not None]
    mean_ai = sum(r.ai_score for r in scored) / len(scored) if scored else 0.0
    mean_human = (
        sum(r.human_likeness for r in scored) / len(scored) if scored else 0.0
    )

    cat_sums: dict[str, list[float]] = {}
    for r in scored:
        for cat, score in r.sub_scores.items():
            cat_sums.setdefault(cat, []).append(score)
    per_cat_mean = {
        cat: sum(vals) / len(vals) for cat, vals in cat_sums.items() if vals
    }

    finished = datetime.now(timezone.utc)
    return VerifyRun(
        run_id=run_id,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        soul_md_hash=soul_hash,
        soul_md_version=soul_version,
        threshold=threshold,
        mean_ai_score=mean_ai,
        mean_human_likeness=mean_human,
        pass_=(len(scored) > 0 and mean_ai <= threshold),
        n_prompts=len(prompts),
        n_scored=len(scored),
        per_category_mean=per_cat_mean,
        results=results,
    )


def append_run(run: VerifyRun) -> None:
    payload = asdict(run)
    payload["pass"] = payload.pop("pass_")
    line = json.dumps(payload, ensure_ascii=False)
    with open(_runs_path(), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def read_runs(*, soul_md_hash: Optional[str] = None) -> list[dict[str, Any]]:
    p = _runs_path()
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if soul_md_hash and rec.get("soul_md_hash") != soul_md_hash:
            continue
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(run: VerifyRun) -> None:
    print()
    print("=" * 70)
    print(
        f"SOUL v{run.soul_md_version}  hash={run.soul_md_hash[:12]}  "
        f"threshold={run.threshold:.1f}"
    )
    print(f"mean ai_score = {run.mean_ai_score:.2f}  "
          f"(human_likeness = {run.mean_human_likeness:.2f})")
    print(f"scored: {run.n_scored}/{run.n_prompts}")
    print(f"VERDICT: {'PASS' if run.pass_ else 'FAIL'}")
    print()
    print("Per-category mean:")
    for cat, score in sorted(
        run.per_category_mean.items(), key=lambda kv: kv[1], reverse=True
    ):
        print(f"  {cat:15s} {score:5.1f}")
    print()
    print("Per-prompt:")
    for r in run.results:
        if r.error:
            print(f"  [err] {r.prompt[:30]:30s} {r.error[:60]}")
            continue
        worst_cat = ""
        if r.sub_scores:
            worst_cat = max(r.sub_scores, key=r.sub_scores.get)
            if r.sub_scores[worst_cat] == 0:
                worst_cat = ""
        print(
            f"  ai={r.ai_score:5.1f}  {r.prompt[:30]:30s}  "
            f"worst_cat={worst_cat or '-'}"
        )
    print("=" * 70)


def print_diff(prev: dict[str, Any], current: VerifyRun) -> None:
    prev_ai = prev.get("mean_ai_score", 0.0)
    cur_ai = current.mean_ai_score
    delta = cur_ai - prev_ai
    sign = "↓" if delta < 0 else ("↑" if delta > 0 else "=")
    print()
    print(
        f"vs previous run (v{prev.get('soul_md_version','?')} "
        f"hash={prev.get('soul_md_hash','')[:12]}):"
    )
    print(f"  mean ai_score: {prev_ai:.2f} → {cur_ai:.2f}  ({sign}{abs(delta):.2f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SOUL.md humanness verifier")
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.environ.get("DGMH_VERIFY_THRESHOLD", _DEFAULT_THRESHOLD)),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.environ.get("DGMH_VERIFY_PROMPT_LIMIT", "0")),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("DGMH_VERIFY_CHAT_TIMEOUT", _DEFAULT_CHAT_TIMEOUT_S)),
    )
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="skip writing to verification_runs.jsonl",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    prompts = DEFAULT_PROMPTS[: args.limit] if args.limit > 0 else DEFAULT_PROMPTS

    soul_hash, soul_version = _read_soul_metadata()
    print(f"SOUL v{soul_version}  hash={soul_hash[:12]}")
    print(f"Running {len(prompts)} prompts (threshold={args.threshold:.1f})")
    print()

    run = run_verification(
        prompts, threshold=args.threshold, chat_timeout_s=args.timeout
    )

    print_report(run)

    prior = [r for r in read_runs() if r.get("soul_md_hash") != soul_hash]
    if prior:
        print_diff(prior[-1], run)

    if not args.no_append:
        append_run(run)
        print(f"\nappended run to {_runs_path()}")

    return 0 if run.pass_ else 1


if __name__ == "__main__":
    sys.exit(main())
