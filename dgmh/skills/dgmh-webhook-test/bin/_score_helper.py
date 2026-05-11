"""Tiny stdin → JSON wrapper around dgmh.patina_judge.score_humanness.

Spawned as a subprocess by dgmh-webhook-test.py via the hermes-agent
venv so the parent script can stay on system Python.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


_HERMES_ROOT = Path(
    os.environ.get("DGMH_HERMES_ROOT")
    or "~/workspace/hermes-agent"
).expanduser()
if _HERMES_ROOT.exists() and str(_HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(_HERMES_ROOT))


def main() -> int:
    text = sys.stdin.read()
    try:
        from dgmh.patina_judge import PatinaScoreError, score_humanness
    except Exception as e:
        print(
            json.dumps(
                {
                    "ai_score": None,
                    "human_likeness": None,
                    "score_error": f"import failed: {type(e).__name__}: {e}",
                },
                ensure_ascii=False,
            )
        )
        return 0

    try:
        result = score_humanness(text, lang="ko")
    except PatinaScoreError as e:
        print(
            json.dumps(
                {
                    "ai_score": None,
                    "human_likeness": None,
                    "score_error": f"PatinaScoreError: {e}",
                },
                ensure_ascii=False,
            )
        )
        return 0
    except Exception as e:
        print(
            json.dumps(
                {
                    "ai_score": None,
                    "human_likeness": None,
                    "score_error": f"{type(e).__name__}: {e}",
                },
                ensure_ascii=False,
            )
        )
        return 0

    print(
        json.dumps(
            {
                "ai_score": result.ai_score,
                "human_likeness": result.human_likeness,
                "interpretation": getattr(result, "interpretation", None),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
