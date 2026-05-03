"""
dgmh/locks.py — Per-skill flock-based concurrency safety for DGM-H mutations.

Port note: Addresses v2 §R-3 (lock contention scope). Extends the DGM-H
write lock to per-skill directories using fcntl.flock(LOCK_EX | LOCK_NB)
against ~/.hermes/dgmh/locks/<cat>__<name>.lock. Acquired before any modifier
emits a candidate child for a skill; released after archive admission or
rejection.

The lock also applies to Hermes' native review path when
dgmh.gate_native_review=true: suggestions targeting a locked skill stay queued
until the DGM-H iteration releases the lock (v2 §R-3).

Reference: devswha/dgmh @ 3eddf82a661688f88b4bbf4c55028509d0eac002
  - hermes-skill-archive-dgmh-plan-v2-addendum.md §R-3: per-skill lock scope
  - hermes-skill-archive-dgmh-plan-v2-addendum.md §CONCERN 6: updated acceptance metric §6
  - tools/skill_manager_tool.py::_atomic_write_text (line 290): single-writer OS rename
"""

from __future__ import annotations

import fcntl
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class SkillLockError(RuntimeError):
    """Raised when a skill lock cannot be acquired within timeout + retries."""

    def __init__(self, message: str, *, category: str, name: str) -> None:
        super().__init__(message)
        self.category = category
        self.name = name


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _locks_dir() -> Path:
    """~/.hermes/dgmh/locks/ — lock file directory."""
    hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(hermes_home) / "dgmh" / "locks"


def _lock_file_path(category: str, name: str) -> Path:
    """Return path to lock file for <category>/<name> skill.

    Uses <cat>__<name>.lock convention (double underscore separator) to
    avoid ambiguity with nested paths.
    """
    safe_cat = category.replace("/", "_").replace(" ", "_") if category else "root"
    safe_name = name.replace("/", "_").replace(" ", "_")
    filename = f"{safe_cat}__{safe_name}.lock"
    return _locks_dir() / filename


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

@contextmanager
def acquire_skill_lock(
    category: str,
    name: str,
    *,
    timeout_s: float = 30.0,
    retries: int = 3,
) -> Generator[Path, None, None]:
    """Acquire an exclusive flock for a skill directory mutation.

    Uses fcntl.flock(LOCK_EX | LOCK_NB) with retry loop. On timeout
    exhaustion logs 'archive-lock-error' and raises SkillLockError.

    Per v2 §R-3:
    - Lock is per-skill: ~/.hermes/dgmh/locks/<cat>__<name>.lock
    - default timeout_s=30, default retries=3
    - Logs 'archive-lock-error' on exhaustion

    Args:
        category: Skill category string.
        name: Skill name string.
        timeout_s: Per-attempt timeout in seconds before retrying.
        retries: Number of acquisition attempts before giving up.

    Yields:
        Path to the lock file (for test introspection).

    Raises:
        SkillLockError: After all retries are exhausted.
    """
    lock_path = _lock_file_path(category, name)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    fd: int | None = None
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        deadline = time.monotonic() + timeout_s
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY, 0o600)
            # Busy-wait with LOCK_NB until deadline
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    # Acquired
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)

            logger.debug(
                "acquire_skill_lock: acquired %s/%s on attempt %d",
                category,
                name,
                attempt,
            )
            try:
                yield lock_path
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                try:
                    os.close(fd)
                except OSError:
                    pass
                fd = None
            return  # success — exit generator

        except BlockingIOError as exc:
            last_exc = exc
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
                fd = None
            logger.warning(
                "acquire_skill_lock: attempt %d/%d timed out for %s/%s",
                attempt,
                retries,
                category,
                name,
            )
        except Exception as exc:
            last_exc = exc
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
                fd = None
            raise

    # All retries exhausted
    logger.error(
        "archive-lock-error: could not acquire lock for %s/%s after %d retries (timeout_s=%.1f): %s",
        category,
        name,
        retries,
        timeout_s,
        last_exc,
    )
    raise SkillLockError(
        f"Could not acquire lock for {category!r}/{name!r} after {retries} retries",
        category=category,
        name=name,
    )
