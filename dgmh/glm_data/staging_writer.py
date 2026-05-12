"""Whitelist (positive-permit) writer for DGM-H GLM data staging.

This module is AC13 *primary* defense for the plan in
``.omc/plans/dgmh-glm-data-v1.md``: every byte written by the v1 pipeline
must land under ``~/.hermes/dgmh/glm_data_v1/staging/<session_id>/``.

The writer refuses everything else — production telemetry files, repo
working trees, ``/etc``, ``/tmp``, symlinked back-doors. Tests cover the
nine production paths under ``~/.hermes/dgmh/``, the canonical SOUL.md,
50 random paths, the symlink-escape trick, and the happy path.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable, Mapping, Union

from . import STAGING_ROOT_RELATIVE

# Allow letters, digits, hyphen, underscore. Forbid path separators and ".."
# tokens so a malicious caller cannot build a session_id like "../foo".
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


class PathGuardError(RuntimeError):
    """Raised when ``StagingWriter`` is asked to write outside its session root."""


JsonRow = Mapping[str, object]
PathLike = Union[str, os.PathLike[str]]


class StagingWriter:
    """Path-guarded file writer scoped to a single staging session.

    Every public write method resolves the target with ``Path.resolve()``
    (which canonicalizes symlinks and ``..`` segments) and asserts the
    result is *inside* the session root. Anything else raises
    ``PathGuardError`` before a single byte is written.

    v1 single-user TOCTOU assumption (codex review MAJOR-4)
    -------------------------------------------------------
    ``_guard()`` resolves the target path, then ``write_text`` / ``write_bytes``
    / ``write_jsonl`` perform the actual write moments later. Between those
    two operations a sufficiently privileged adversary could swap an
    intermediate directory component for a symlink that escapes the staging
    root (a classic time-of-check / time-of-use race).

    v1 assumes **no concurrent malicious symlink swap** between the guard and
    the write — the pipeline runs as a single operator on their own laptop
    with no concurrent adversary. Multi-user or concurrent-adversary scenarios
    require ``open(O_NOFOLLOW)`` plus an atomic temp-then-rename pattern,
    which is deferred to v2 (see plan v2.1 §5 R7 mitigation note).
    """

    def __init__(self, session_id: str, *, home: Path | None = None) -> None:
        if not _SESSION_ID_RE.match(session_id):
            raise ValueError(
                "session_id must match [A-Za-z0-9][A-Za-z0-9_-]{0,127}; "
                f"got {session_id!r}"
            )
        self.session_id = session_id
        home_path = (home or Path.home()).resolve()
        # Compute the session root without requiring it to exist on disk yet.
        # We resolve(strict=False) so symlinks/.. are still canonicalized.
        self._session_root: Path = (
            home_path / STAGING_ROOT_RELATIVE / session_id
        ).resolve()

    # ------------------------------------------------------------------ API

    @property
    def session_root(self) -> Path:
        """Resolved absolute path of this writer's session root."""

        return self._session_root

    def write_text(self, path: PathLike, content: str) -> Path:
        """Write a UTF-8 text file. Returns the resolved target path."""

        target = self._guard(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def write_bytes(self, path: PathLike, data: bytes) -> Path:
        """Write raw bytes. Returns the resolved target path."""

        target = self._guard(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target

    def write_jsonl(self, path: PathLike, rows: Iterable[JsonRow]) -> Path:
        """Write an iterable of JSON-serializable mappings, one per line."""

        target = self._guard(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                fh.write("\n")
        return target

    # -------------------------------------------------------------- Internals

    def _guard(self, path: PathLike) -> Path:
        """Resolve ``path`` and assert it lives strictly under the session root.

        The path may be absolute (then it must literally be inside the
        session root) or relative (then it is joined to the session root).
        Either way ``resolve()`` canonicalizes symlinks and ``..`` so we get
        a single comparison point.
        """

        raw = Path(os.fspath(path))
        candidate = raw if raw.is_absolute() else (self._session_root / raw)
        resolved = candidate.resolve()

        root = self._session_root
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise PathGuardError(
                f"refusing to write outside staging root: target={resolved} "
                f"root={root} (input={path!r})"
            ) from exc

        if resolved == root:
            raise PathGuardError(
                f"refusing to write to the session root itself: {resolved}"
            )

        return resolved
