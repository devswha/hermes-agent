"""Regenerate the patina ``kakao-mimic`` profile from a KakaoTalk corpus.

Voice anchors are sampled from the operator's real chat history stored at
``~/.hermes/dgmh/kakao_hako_corpus.txt`` (one clean message per line, no
media stubs, no system rows). Output mirrors into
``dgmh/patina_profiles/kakao-mimic.md`` for version control and into the
live patina install at
``~/.hermes/skills/creative/patina/profiles/kakao-mimic.md``.

Re-run this whenever:
  - new chat data is added to the corpus
  - the seed / sampling strategy changes
  - the profile body template changes

Usage:
    python3 dgmh/patina_profiles/build_kakao_mimic.py
"""

from __future__ import annotations

import random
import re
from pathlib import Path

CORPUS_PATH = Path.home() / ".hermes" / "dgmh" / "kakao_hako_corpus.txt"
REPO_PROFILE = (
    Path(__file__).resolve().parent / "kakao-mimic.md"
)
LIVE_PROFILE = (
    Path.home()
    / ".hermes"
    / "skills"
    / "creative"
    / "patina"
    / "profiles"
    / "kakao-mimic.md"
)

_URL_RE = re.compile(r"https?://|www\.")
_HANGUL_CHARS = set("ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ가나다라마바사아자차카타파하")


def _is_anchor_quality(msg: str) -> bool:
    if _URL_RE.search(msg):
        return False
    if msg.startswith("@") or msg.startswith("[20"):
        return False
    if "share.google" in msg or "GeekNews" in msg:
        return False
    if all(c.isascii() for c in msg):
        return False
    latin = sum(1 for c in msg if c.isascii() and c.isalpha())
    if latin > len(msg) * 0.5:
        return False
    return True


def _sample_anchors(corpus: list[str], seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    clean = [m for m in corpus if _is_anchor_quality(m)]
    buckets = {
        "short": [m for m in clean if len(m) < 10],
        "mid": [m for m in clean if 10 <= len(m) < 30],
        "long": [m for m in clean if 30 <= len(m) < 60],
        "xlong": [m for m in clean if 60 <= len(m) < 140],
    }
    targets = {"short": 12, "mid": 22, "long": 13, "xlong": 3}
    picks: list[str] = []
    for bucket, count in targets.items():
        pool = buckets[bucket]
        if not pool:
            continue
        picks.extend(rng.sample(pool, min(count, len(pool))))
    rng.shuffle(picks)
    return picks


def _build_profile_md(anchors: list[str]) -> str:
    body = [
        "---",
        "profile: kakao-mimic",
        "name: 카카오톡 본인 문체 미러 프로필",
        "version: 1.0.0",
        "scope: 운영자(devswha) 카카오톡 그룹 대화 톤을 1:1 채팅 응답에 미러링",
        "voice-overrides:",
        "  first-person: amplify",
        "  opinions: amplify",
        "  rhythm-variation: amplify",
        "  humor: allow",
        "  messiness: amplify",
        "  concrete-emotions: amplify",
        "  reader-address: amplify",
        "  hedge-tone: amplify",
        "pattern-overrides:",
        "  ko:",
        "    8: amplify",
        "    18: amplify",
        "    14: suppress",
        "    19: reduce",
        "  en:",
        "    8: amplify",
        "    7: amplify",
        "    14: suppress",
        "---",
        "",
        "# 카카오톡 본인 문체 미러 프로필 (`kakao-mimic`)",
        "",
        "운영자의 실제 카카오톡 그룹 대화에서 추출한 voice anchor를 in-context로 사용한다.",
        f"아래 “Reference voice anchors” 섹션의 {len(anchors)}개 메시지는 운영자가 친한 사람들과 주고받은",
        "**실제 메시지**다. 어미·문장 길이·필러 사용 패턴·오타 허용도·인터넷 슬랭 빈도를",
        "그대로 미러링한다. casual-conversation 프로필보다 한 단계 더 “쓰는 글” 보다는 “타이핑하는 채팅”에 가깝다.",
        "",
        "## Voice 가이드라인",
        "",
        "### 1. 문장 길이 분포",
        "",
        "- 압도적 다수는 30자 미만의 짧은 단문. 30-60자 중문 일부, 60자+ 장문은 드물게.",
        "- 끝맺지 않는 문장도 자연스럽다 — 예: \"근데 저걸로 실직하면\", \"흐름에 타야지\".",
        "- 한 호흡으로 두세 짧은 문장을 줄바꿈 없이 이어붙이지 말 것.",
        "",
        "### 2. 어미·종결",
        "",
        "- \"요\"/\"죠\"/\"어요\" 위주, 평어 \"다\"는 제한적.",
        "- 변형 어미 자유롭게: \"용\", \"여\", \"거든요\", \"임\", \"임다\", \"슴다\", \"겠읍니다\", \"읍니다\", \"임까\".",
        "- \"~데\", \"~데요\" 미완 종결 OK — \"근데\", \"~인데\" 로 끝나도 자연스럽다.",
        "",
        "### 3. 필러·인터넷 슬랭",
        "",
        "- `ㅋㅋ`, `ㅋㅋㅋㅋ`, `ㅎㅎ`, `ㄷㄷ`, `ㅇㅇ`, `ㅇㅇ..`, `굿ㅋㅋㅋ`, `..`, `~~~` 자유롭게.",
        "- 의성어·의태어 OK: \"퍄퍄\", \"끼룩끼룩낄낄낄\", \"오우\", \"에이\", \"헐\".",
        "- 영문 약어·단어 한국어 사이 혼용: `cli`, `omc`, `ai`, `api`, `notion`. 대소문자 일관성 없어도 자연스럽다.",
        "",
        "### 4. 의도적 오타·구어 변형 허용",
        "",
        "- \"쪼아요\" (좋아요), \"괜찮슴다\" (괜찮습니다), \"아임까\" (입니까), \"아닉ㅆ지\" (아니겠지),",
        "  \"되엇네\" (됐네), \"넘\" (너무), \"걸로\", \"거의\" 줄임형 등 입력 속도에서 나온 오타·축약은 그대로 둔다.",
        "- 격식체 정확 표기보다 입말 리듬이 우선.",
        "",
        "### 5. 백틱·마크다운 금지",
        "",
        "- 인라인 백틱(`...`)으로 단어·날짜·키워드를 감싸지 않는다. **굵게**·*기울임*·`# 헤더`도 안 쓴다.",
        "- 코드는 직접 적거나 그냥 풀어서 말한다 — `npm install` 이 아니라 `npm install`, 또는 \"npm install 하고\".",
        "",
        "### 6. hedging은 한 번만",
        "",
        "- \"같아요\"/\"보여요\"/\"~듯\" 같은 완충은 한 응답에 한 번 정도. 세 번 이상 누적되면 LLM 톤.",
        "- 의견은 1인칭으로 정확히 한쪽으로: \"난 ~쪽이야\", \"전 그게 더 나아요\" 식.",
        "",
        "## Reference voice anchors",
        "",
        f"다음 {len(anchors)}개 메시지는 운영자의 실제 카톡 그룹 대화에서 그대로 가져온 voice anchor다.",
        "rewrite 시 이 메시지들의 톤·길이·어미·필러 사용 패턴을 미러링한다.",
        "",
        "```",
    ]
    body.extend(anchors)
    body.extend(
        [
            "```",
            "",
            "## 다른 프로필과의 차이",
            "",
            "| | casual-conversation | kakao-mimic |",
            "|---|---|---|",
            "| 출처 | 일반 친한 대화체 규칙 | **본인 카톡 실제 메시지** |",
            "| 어미 다양성 | 요/죠 위주 | **요/죠/임/슴다/거든요/읍니다 자유** |",
            "| 오타 허용 | 암시적 OK | **명시적 OK + 예시 다수** |",
            "| 인터넷 슬랭 | hedge 어휘 위주 | **ㅋㅋ/ㅇㅇ/ㄷㄷ/퍄퍄 등 적극** |",
            "| 백틱·강조 | 일반적으로 회피 | **명시적 금지** |",
            "",
            "## 사용",
            "",
            "```bash",
            "node patina.js --lang ko --profile kakao-mimic --backend codex-cli < input.txt",
            "```",
            "",
            "DGM-H humanness_hook에서는 `DGMH_PATINA_PROFILE=kakao-mimic` 환경변수로 활성화한다.",
        ]
    )
    return "\n".join(body) + "\n"


def main() -> int:
    if not CORPUS_PATH.exists():
        print(f"corpus not found at {CORPUS_PATH}")
        return 1
    corpus = [
        line.rstrip("\n")
        for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(corpus) < 100:
        print(f"corpus too small ({len(corpus)} lines); aborting")
        return 1
    anchors = _sample_anchors(corpus)
    if len(anchors) < 30:
        print(f"only sampled {len(anchors)} anchors; aborting")
        return 1
    md = _build_profile_md(anchors)
    REPO_PROFILE.write_text(md, encoding="utf-8")
    LIVE_PROFILE.write_text(md, encoding="utf-8")
    print(f"wrote {len(anchors)} anchors to:")
    print(f"  {REPO_PROFILE}")
    print(f"  {LIVE_PROFILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
