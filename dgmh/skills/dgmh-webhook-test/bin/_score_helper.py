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


def _emit(payload: dict) -> int:
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def _emit_failure(reason: str) -> int:
    return _emit(
        {"ai_score": None, "human_likeness": None, "score_error": reason}
    )


def main() -> int:
    text = sys.stdin.read()
    try:
        from dgmh.patina_judge import PatinaScoreError, score_humanness
    except Exception as e:
        return _emit_failure(f"import failed: {type(e).__name__}: {e}")

    try:
        result = score_humanness(text, lang="ko")
    except PatinaScoreError as e:
        return _emit_failure(f"PatinaScoreError: {e}")
    except Exception as e:
        return _emit_failure(f"{type(e).__name__}: {e}")

    return _emit(
        {
            "ai_score": result.ai_score,
            "human_likeness": result.human_likeness,
            "interpretation": getattr(result, "interpretation", None),
        }
    )


if __name__ == "__main__":
    sys.exit(main())
