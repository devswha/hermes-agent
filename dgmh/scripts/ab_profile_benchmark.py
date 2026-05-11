"""A/B benchmark: ``social`` vs ``casual-conversation`` patina profiles.

For each canonical Korean Discord prompt:
  1. Generate a fresh response by calling ``hermes chat`` (same path as
     ``verify_soul_humanness``).
  2. Rewrite the response through patina with ``profile=social``; capture
     the rewritten text and its ai_score.
  3. Rewrite the same response through patina with
     ``profile=casual-conversation``; capture rewritten text and ai_score.
  4. Print a comparison table + summary stats (mean ai_score per profile,
     win count, average delta).

Failures (timeout, parse error, missing binary) on either profile mark the
prompt as "skipped" and the benchmark continues. Skip count is reported
separately.

Usage:

    cd /home/devswha/workspace/hermes-agent
    ./venv/bin/python -m dgmh.scripts.ab_profile_benchmark
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from typing import Optional

from dgmh.patina_judge import (
    PatinaScoreError,
    humanness_rewrite_with_profile,
    score_humanness,
)
from dgmh.scripts.verify_soul_humanness import DEFAULT_PROMPTS, call_hermes_chat

logger = logging.getLogger(__name__)


_PROFILE_A = "social"
_PROFILE_B = "casual-conversation"
_REWRITE_TIMEOUT_S = 60.0
_SCORE_TIMEOUT_S = 90.0
_CHAT_TIMEOUT_S = 90.0


@dataclass
class ProfileResult:
    profile: str
    rewritten: Optional[str]
    ai_score: Optional[float]
    elapsed_s: float
    error: str = ""


@dataclass
class PromptComparison:
    prompt: str
    response: str
    chat_elapsed_s: float
    a: ProfileResult
    b: ProfileResult
    chat_error: str = ""

    @property
    def skipped(self) -> bool:
        return (
            bool(self.chat_error)
            or self.a.ai_score is None
            or self.b.ai_score is None
        )

    @property
    def winner(self) -> str:
        if self.skipped:
            return "-"
        assert self.a.ai_score is not None and self.b.ai_score is not None
        if self.a.ai_score < self.b.ai_score:
            return self.a.profile
        if self.b.ai_score < self.a.ai_score:
            return self.b.profile
        return "tie"

    @property
    def delta(self) -> Optional[float]:
        if self.a.ai_score is None or self.b.ai_score is None:
            return None
        # delta = b - a → negative means casual-conversation is lower (better).
        return self.b.ai_score - self.a.ai_score


def _rewrite_and_score(text: str, *, profile: str) -> ProfileResult:
    start = time.monotonic()
    try:
        rewritten = humanness_rewrite_with_profile(
            text,
            profile=profile,
            backend="codex-cli",
            lang="ko",
            timeout_s=_REWRITE_TIMEOUT_S,
        )
    except Exception as exc:  # defensive: function is None-on-failure but be safe.
        return ProfileResult(
            profile=profile,
            rewritten=None,
            ai_score=None,
            elapsed_s=time.monotonic() - start,
            error=f"rewrite-exception: {type(exc).__name__}: {exc}",
        )

    if rewritten is None:
        return ProfileResult(
            profile=profile,
            rewritten=None,
            ai_score=None,
            elapsed_s=time.monotonic() - start,
            error="rewrite-returned-none",
        )

    try:
        score = score_humanness(rewritten, lang="ko", timeout_s=_SCORE_TIMEOUT_S)
    except PatinaScoreError as exc:
        return ProfileResult(
            profile=profile,
            rewritten=rewritten,
            ai_score=None,
            elapsed_s=time.monotonic() - start,
            error=f"score-error: {exc}",
        )

    return ProfileResult(
        profile=profile,
        rewritten=rewritten,
        ai_score=score.ai_score,
        elapsed_s=time.monotonic() - start,
    )


def benchmark_prompt(prompt: str) -> PromptComparison:
    try:
        response, chat_elapsed = call_hermes_chat(prompt, timeout_s=_CHAT_TIMEOUT_S)
    except Exception as exc:
        return PromptComparison(
            prompt=prompt,
            response="",
            chat_elapsed_s=0.0,
            a=ProfileResult(
                profile=_PROFILE_A, rewritten=None, ai_score=None, elapsed_s=0.0,
                error="chat-error",
            ),
            b=ProfileResult(
                profile=_PROFILE_B, rewritten=None, ai_score=None, elapsed_s=0.0,
                error="chat-error",
            ),
            chat_error=f"{type(exc).__name__}: {exc}",
        )

    a = _rewrite_and_score(response, profile=_PROFILE_A)
    b = _rewrite_and_score(response, profile=_PROFILE_B)
    return PromptComparison(
        prompt=prompt,
        response=response,
        chat_elapsed_s=chat_elapsed,
        a=a,
        b=b,
    )


def _truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ").replace("\r", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def print_table(rows: list[PromptComparison]) -> None:
    print()
    print(
        f"{'#':>2}  {'prompt':<22}  {'response':<32}  "
        f"{'social':>7}  {'casual':>7}  {'winner':<19}  {'Δ':>6}"
    )
    print("-" * 110)
    for i, r in enumerate(rows, 1):
        prompt_s = _truncate(r.prompt, 22)
        resp_s = _truncate(r.response, 32) if r.response else "-"
        a_s = f"{r.a.ai_score:6.1f}" if r.a.ai_score is not None else "  err"
        b_s = f"{r.b.ai_score:6.1f}" if r.b.ai_score is not None else "  err"
        delta_s = f"{r.delta:+6.1f}" if r.delta is not None else "    -"
        winner = r.winner
        print(
            f"{i:>2}  {prompt_s:<22}  {resp_s:<32}  "
            f"{a_s:>7}  {b_s:>7}  {winner:<19}  {delta_s:>6}"
        )


def print_summary(rows: list[PromptComparison]) -> None:
    scored = [r for r in rows if not r.skipped]
    skipped = [r for r in rows if r.skipped]

    print()
    print("=" * 70)
    if not scored:
        print("No prompts scored successfully — cannot compute summary.")
        if skipped:
            print(f"skipped: {len(skipped)}/{len(rows)}")
            for r in skipped:
                why = r.chat_error or r.a.error or r.b.error
                print(f"  - {r.prompt[:30]!r}: {why}")
        return

    mean_a = sum(r.a.ai_score for r in scored) / len(scored)  # type: ignore[arg-type]
    mean_b = sum(r.b.ai_score for r in scored) / len(scored)  # type: ignore[arg-type]
    deltas = [r.delta for r in scored if r.delta is not None]
    avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
    b_wins = sum(1 for r in scored if r.winner == _PROFILE_B)
    a_wins = sum(1 for r in scored if r.winner == _PROFILE_A)
    ties = sum(1 for r in scored if r.winner == "tie")

    print(f"mean ai_score ({_PROFILE_A}):              {mean_a:6.2f}")
    print(f"mean ai_score ({_PROFILE_B}):  {mean_b:6.2f}")
    print(
        f"{_PROFILE_B} wins on {b_wins}/{len(scored)} prompts "
        f"({_PROFILE_A} wins {a_wins}, ties {ties}; "
        f"delta avg = {avg_delta:+.2f}, negative = "
        f"{_PROFILE_B} better)"
    )
    if skipped:
        print(f"skipped: {len(skipped)}/{len(rows)}")
        for r in skipped:
            why = r.chat_error or r.a.error or r.b.error
            print(f"  - {r.prompt[:30]!r}: {why}")
    print("=" * 70)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="A/B benchmark patina profiles social vs casual-conversation"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the prompt count (default: all).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    prompts = DEFAULT_PROMPTS[: args.limit] if args.limit > 0 else DEFAULT_PROMPTS
    print(
        f"A/B profile benchmark: {_PROFILE_A} vs {_PROFILE_B}  "
        f"prompts={len(prompts)}"
    )
    print()

    rows: list[PromptComparison] = []
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] prompt={prompt!r}", flush=True)
        cmp_ = benchmark_prompt(prompt)
        if cmp_.chat_error:
            print(f"  -> chat error: {cmp_.chat_error}", flush=True)
        else:
            a_s = (
                f"{cmp_.a.ai_score:.1f}"
                if cmp_.a.ai_score is not None
                else f"err({cmp_.a.error})"
            )
            b_s = (
                f"{cmp_.b.ai_score:.1f}"
                if cmp_.b.ai_score is not None
                else f"err({cmp_.b.error})"
            )
            print(
                f"  -> {_PROFILE_A}={a_s}  {_PROFILE_B}={b_s}  "
                f"chat={cmp_.chat_elapsed_s:.1f}s",
                flush=True,
            )
        rows.append(cmp_)

    print_table(rows)
    print_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
