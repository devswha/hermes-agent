"""Autonomous SOUL.md learning loop driver.

Drives a closed-loop tuning cycle:

  1. Post a user-style prompt to Discord via webhook (visible to operator).
  2. Wait for the bot's reply.
  3. Score the reply via patina + structural pollution check.
  4. If the reply is polluted, post a status note and trigger SOUL evolution
     (negative-reaction equivalent) via run_soul_evolution(neg=1).
  5. Wait for the evolution cycle to finish (modifier + critic).
  6. Move to the next prompt.

Discord-visible: the operator can watch the conversation play out and the
status updates from the driver, while the actual learning trigger goes
through a direct Python call (the bot cannot react to its own messages,
so reactions are not the trigger path here).

Run with:

    cd /home/devswha/workspace/hermes-agent
    ./venv/bin/python -m dgmh.scripts.auto_learn_loop --iterations 5

Env / flags:
    --iterations N     Number of prompt rounds (default 5)
    --threshold X      Pollution threshold for ai_score (default 10)
    --no-trigger       Score and post status but do not call run_soul_evolution
    --webhook-file P   Override webhook credentials file (default /tmp/.dgmh_webhook)
    --channel-id ID    Override Discord channel (default 1496872245027541062)
    --reply-timeout S  How long to wait for bot reply (default 60)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional


# Operator-confirmed casual prompts (pulled from the verifier prompt set).
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
    "넌 역할이 뭐고 어떤 모델이야?",
    "내가 지금 뭐 작업해야 좋을까",
    "어제 한 거 정리해줘",
    "그냥 잡담하자",
    "오늘 점심 추천해",
]


# Stress prompts — designed to provoke ChatGPT-style failure modes:
# - long-form tutorials / list-prone tasks
# - identity / bio-recitation probes
# - both-sides trade-off questions
# - open-ended opinion questions
# - explicit "정리" / "단계별" requests that trigger bullet lists
STRESS_PROMPTS = [
    "Rust 와 Go 뭐가 더 나아? 정리해줘",
    "AI 의 미래 어떻게 생각해? 자세히",
    "Git rebase 사용법 단계별로 알려줘",
    "OAuth 2.0 흐름 설명해줘",
    "Python 학습 로드맵 짜봐",
    "한국 IT 산업 요즘 어때?",
    "Mac 이랑 Linux 중에 개발에 뭐가 좋아?",
    "CI/CD 파이프라인 만드는 법 정리해줘",
    "REST 와 GraphQL 차이 설명해",
    "너 진화하는 시스템이라고 했잖아 자세히 설명해봐",
    "자기소개 좀 해봐",
    "너 무슨 모델이야? 그리고 역할은?",
    "Docker 와 Kubernetes 차이",
    "스타트업 vs 대기업 어디가 나아?",
    "프론트엔드 백엔드 어디 시작이 좋을까",
    "내가 너한테 뭐 시킬 수 있는지 정리해봐",
    "현재 SOUL 진화 흐름 정리해줘",
    "AI 슬롭 이라는게 뭐야 설명해",
    "테스트 작성 잘 하는 법 알려줘",
    "코드 리뷰 어떻게 해야 효과적이야?",
]


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


def _read_webhook(path: str) -> tuple[str, str]:
    raw = Path(path).read_text(encoding="utf-8").strip()
    wid, _, wtoken = raw.partition(":")
    if not wid or not wtoken:
        raise RuntimeError(f"webhook file {path} malformed; expected id:token")
    return wid, wtoken


def _bot_token() -> str:
    env = _hermes_home() / ".env"
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.startswith("DISCORD_BOT_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("DISCORD_BOT_TOKEN not in ~/.hermes/.env")


def _http(method: str, url: str, *, headers: dict, body: Optional[dict] = None, timeout: float = 15) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        text = fh.read().decode("utf-8")
        return json.loads(text) if text else {}


_BOT_MENTION = "<@1496771171151253636>"


def post_user_message(webhook_id: str, webhook_token: str, content: str, *, username: str = "하코") -> dict:
    """Post a webhook message that mentions the bot.

    Hermes' Discord adapter ignores webhook authors (which carry author.bot=True)
    unless DISCORD_ALLOW_BOTS is set to "mentions" or "all". With "mentions"
    we must include an @bot mention or the bot will not respond.
    """
    url = f"https://discord.com/api/v10/webhooks/{webhook_id}/{webhook_token}?wait=true"
    body_content = f"{_BOT_MENTION} {content}"
    return _http(
        "POST",
        url,
        headers={"Content-Type": "application/json", "User-Agent": "DiscordBot (dgmh, 0.1)"},
        body={
            "username": username,
            "content": body_content,
            "allowed_mentions": {"users": ["1496771171151253636"]},
        },
    )


def post_status(webhook_id: str, webhook_token: str, content: str) -> None:
    url = f"https://discord.com/api/v10/webhooks/{webhook_id}/{webhook_token}"
    try:
        _http(
            "POST",
            url,
            headers={"Content-Type": "application/json", "User-Agent": "DiscordBot (dgmh, 0.1)"},
            body={"username": "dgmh-driver", "content": content},
        )
    except Exception as exc:
        print(f"  ! status post failed: {exc}", flush=True)


def fetch_messages_after(channel_id: str, after_id: str, bot_token: str) -> list[dict]:
    url = f"https://discord.com/api/v10/channels/{channel_id}/messages?limit=10&after={after_id}"
    return _http(
        "GET",
        url,
        headers={
            "Authorization": f"Bot {bot_token}",
            "User-Agent": "DiscordBot (dgmh, 0.1)",
        },
    )


# Tool-trace prefix glyphs that Hermes uses for in-progress messages
# (skill loads, browser actions, file reads, terminal calls, memory edits,
# self-interrupt notices). These are NOT the bot's final natural-language
# reply — keep waiting until a non-trace message lands.
_TOOL_TRACE_PREFIXES = (
    "📚",  # skill_view / skill_execute
    "🌐",  # browser_navigate
    "🔎",  # search_files
    "🔍",  # session_search
    "💻",  # terminal
    "📖",  # read_file
    "🧠",  # memory
    "📸",  # browser_snapshot
    "⚡",  # interrupt notice
    "📝",  # write_file
    "🛠",  # tool generic
)


def _is_tool_trace(content: str) -> bool:
    stripped = (content or "").lstrip()
    return any(stripped.startswith(p) for p in _TOOL_TRACE_PREFIXES)


def wait_for_bot_reply(
    channel_id: str,
    after_id: str,
    bot_user_id: str,
    bot_token: str,
    timeout_s: float,
) -> Optional[dict]:
    """Wait for the bot's final natural-language reply.

    Skips tool-trace messages (skill_view, browser_navigate, interrupt
    notices) and treats only regular prose replies as final. The "after_id"
    advances as we see new messages so a long sequence of trace lines
    doesn't cause us to repeat-scan the same payloads.
    """
    start = time.monotonic()
    cursor = after_id
    while time.monotonic() - start < timeout_s:
        msgs = fetch_messages_after(channel_id, cursor, bot_token)
        if msgs:
            # advance cursor to the newest seen message
            cursor = msgs[0]["id"]
        for m in reversed(msgs):
            if m["author"]["id"] != bot_user_id:
                continue
            content = m.get("content", "")
            if not content:
                continue
            if _is_tool_trace(content):
                # tool trace, keep waiting for the actual answer
                continue
            return m
        time.sleep(3)
    return None


def score_reply(text: str) -> tuple[float, list[str]]:
    """Return (ai_score, structural_flags). Uses our scoring + check helpers."""
    from dgmh.patina_judge import score_humanness, PatinaScoreError
    from dgmh.hermes_integration.humanness_hook import _structural_pollution_check

    _, flags = _structural_pollution_check(text)
    try:
        result = score_humanness(text, lang="ko")
        return result.ai_score, flags
    except PatinaScoreError:
        return 0.0, flags


def trigger_evolution() -> tuple[bool, str]:
    """Call dgmh.soul_evolution.run_soul_evolution(neg=1) directly."""
    from dgmh.soul_evolution import run_soul_evolution, SoulEvolutionOpts

    try:
        admitted = run_soul_evolution(SoulEvolutionOpts(pos_reactions=0, neg_reactions=1))
        return admitted, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def read_soul_meta() -> tuple[str, str]:
    soul = _hermes_home() / "SOUL.md"
    content = soul.read_text(encoding="utf-8")
    import hashlib

    h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    version = ""
    for line in content.splitlines():
        if line.strip().startswith("version:"):
            version = line.split(":", 1)[1].strip()
            break
    return h, version


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--no-trigger", action="store_true")
    parser.add_argument("--webhook-file", default="/tmp/.dgmh_webhook")
    parser.add_argument("--channel-id", default="1496872245027541062")
    parser.add_argument("--reply-timeout", type=float, default=60.0)
    parser.add_argument(
        "--prompt-set",
        choices=("default", "stress", "all"),
        default="default",
    )
    args = parser.parse_args(argv)

    if args.prompt_set == "stress":
        prompts = STRESS_PROMPTS
    elif args.prompt_set == "all":
        prompts = DEFAULT_PROMPTS + STRESS_PROMPTS
    else:
        prompts = DEFAULT_PROMPTS

    wid, wtoken = _read_webhook(args.webhook_file)
    bot_token = _bot_token()
    bot_user_id = "1496771171151253636"  # flask bot user id
    iterations = max(1, args.iterations)

    h, ver = read_soul_meta()
    post_status(wid, wtoken, f"🚀 dgmh auto-learn 시작 — SOUL v{ver} ({h}), threshold={args.threshold}")
    print(f"start: SOUL v{ver} ({h}) threshold={args.threshold} iters={iterations}", flush=True)

    polluted_count = 0
    triggered_count = 0

    for i in range(iterations):
        prompt = prompts[i % len(prompts)]
        print(f"\n[{i+1}/{iterations}] prompt: {prompt}", flush=True)

        # Snapshot last message id BEFORE posting so we can scan strictly newer.
        latest_resp = _http(
            "GET",
            f"https://discord.com/api/v10/channels/{args.channel_id}/messages?limit=1",
            headers={"Authorization": f"Bot {bot_token}", "User-Agent": "DiscordBot (dgmh, 0.1)"},
        )
        anchor_id = latest_resp[0]["id"] if latest_resp else "0"

        # Post user-style prompt via webhook
        try:
            post_user_message(wid, wtoken, prompt)
        except Exception as exc:
            print(f"  ! webhook post failed: {exc}", flush=True)
            time.sleep(5)
            continue

        # Wait for bot reply
        reply = wait_for_bot_reply(args.channel_id, anchor_id, bot_user_id, bot_token, args.reply_timeout)
        if not reply:
            print("  ! no reply within timeout", flush=True)
            continue

        text = reply.get("content", "")
        print(f"  reply: {text[:140]}", flush=True)

        ai, flags = score_reply(text)
        polluted = ai >= args.threshold or bool(flags)
        if polluted:
            polluted_count += 1
            print(f"  POLLUTED ai={ai:.1f} flags={flags}", flush=True)
            post_status(
                wid, wtoken,
                f"⚠️ pollution detected: ai={ai:.1f} flags={','.join(flags) or '-'} → 진화 trigger",
            )
            if not args.no_trigger:
                accepted, err = trigger_evolution()
                triggered_count += 1
                if accepted:
                    nh, nver = read_soul_meta()
                    post_status(wid, wtoken, f"✅ SOUL.md 진화 OK → v{nver} ({nh})")
                    print(f"  evolution accepted: v{nver} ({nh})", flush=True)
                else:
                    msg = f"❌ 진화 reject{(': ' + err) if err else ''}"
                    post_status(wid, wtoken, msg)
                    print(f"  evolution rejected: {err}", flush=True)
                # cool off so debounce + state.db doesn't pile up
                time.sleep(15)
        else:
            print(f"  CLEAN ai={ai:.1f}", flush=True)

        # cool-off between prompts so the bot can finish any in-flight tool
        # work before we send the next one — avoids self-interrupt loops
        time.sleep(30)

    fh, fver = read_soul_meta()
    post_status(
        wid, wtoken,
        f"🏁 auto-learn 종료 — SOUL v{fver} ({fh}), polluted={polluted_count}/{iterations}, "
        f"triggered={triggered_count}",
    )
    print(
        f"\nfinal: SOUL v{fver} ({fh}) polluted={polluted_count} triggered={triggered_count}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
