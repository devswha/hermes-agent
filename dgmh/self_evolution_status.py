"""Runtime marker for flask/DGM-H self-development status.

The gateway needs a cheap, process-independent way to know when the bot is
currently mutating its own behaviour (SOUL.md / skill/code evolution) so public
Discord replies can disclose that state instead of looking randomly unstable.
The marker is intentionally file/env based: evolution may run in a worker
thread, a separate process, or via an operator-side coding session.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_NOTICE = "지금 제 코드/말투를 손보는 중이라 답이 잠깐 불안정할 수 있어요."
DEFAULT_TTL_SECONDS = 60 * 60


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y", "t"}
    return bool(value)


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


def status_path() -> Path:
    raw = os.environ.get("DGMH_SELF_EVOLUTION_STATUS_PATH")
    if raw:
        return Path(raw).expanduser()
    return _hermes_home() / "dgmh" / "self_evolution_status.json"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=".self-evolution.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _ttl_seconds(ttl_s: int | float | None) -> float:
    if ttl_s is not None:
        return max(1.0, float(ttl_s))
    raw = os.environ.get("DGMH_SELF_EVOLUTION_TTL_SECONDS")
    try:
        parsed = float(raw) if raw is not None else DEFAULT_TTL_SECONDS
    except (TypeError, ValueError):
        parsed = DEFAULT_TTL_SECONDS
    return max(1.0, parsed)


def mark_self_evolution_active(
    *,
    reason: str = "self-evolution",
    source: str = "dgmh",
    message: str | None = None,
    ttl_s: int | float | None = None,
) -> None:
    """Create/update the active marker.

    ``ttl_s`` prevents a crashed evolution worker from leaving the bot in a
    permanent "I'm evolving" state. Long-running flows can refresh the marker.
    """
    now = time.time()
    ttl = _ttl_seconds(ttl_s)
    payload = {
        "active": True,
        "source": source,
        "reason": reason,
        "message": message or os.environ.get("DGMH_SELF_EVOLUTION_NOTICE") or DEFAULT_NOTICE,
        "started_at": _iso_now(),
        "expires_at_epoch": now + ttl,
        "ttl_seconds": ttl,
        "schema_version": 1,
    }
    _atomic_write_json(status_path(), payload)


def clear_self_evolution_active(*, source: str | None = None) -> None:
    """Clear the marker if present.

    When ``source`` is provided, only clears markers written by that source so
    a short SOUL.md cycle cannot erase a separate operator/code-edit marker.
    """
    path = status_path()
    if source is not None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception:
            payload = {}
        if isinstance(payload, dict) and payload.get("source") not in (None, source):
            return
    try:
        path.unlink()
    except FileNotFoundError:
        return


def load_self_evolution_status(now_epoch: float | None = None) -> dict[str, Any]:
    """Return the marker payload with stale markers normalized to inactive."""
    path = status_path()
    if not path.exists():
        return {"active": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"active": False, "error": "invalid_status_payload"}
    if not isinstance(payload, dict) or not payload.get("active"):
        return {"active": False}

    expires_at = payload.get("expires_at_epoch")
    try:
        expired = expires_at is not None and float(expires_at) <= (now_epoch or time.time())
    except (TypeError, ValueError):
        expired = False
    if expired:
        return {**payload, "active": False, "expired": True}
    return payload


def get_self_evolution_notice(now_epoch: float | None = None) -> str | None:
    """Return the public disclosure notice when self-evolution is active."""
    env_notice = os.environ.get("DGMH_SELF_EVOLUTION_NOTICE") or DEFAULT_NOTICE
    if _truthy(os.environ.get("DGMH_SELF_EVOLVING")):
        return env_notice

    status = load_self_evolution_status(now_epoch=now_epoch)
    if not status.get("active"):
        return None
    message = status.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    return env_notice
