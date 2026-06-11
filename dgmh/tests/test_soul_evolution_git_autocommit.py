"""Tests for _git_autocommit_soul_history — runtime-home git snapshots.

The hook must be strictly best-effort:
  1. No .git in the runtime home → no-op, returns False.
  2. A git repo WITHOUT the default-deny (allowlist) .gitignore → refuse:
     secrets must be structurally untrackable before `git add -A` runs.
  3. Proper allowlist repo → exactly one commit, secrets never tracked.
  4. Nothing changed since last snapshot → no empty commit.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dgmh.soul_evolution import _git_autocommit_soul_history

ALLOWLIST_GITIGNORE = """\
*
!.gitignore
!SOUL.md
!dgmh/
dgmh/*
!dgmh/soul_archive.jsonl
!dgmh/persona_drift.jsonl
!dgmh/soul_archive/
!dgmh/soul_archive/**
"""


def _git(home: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(home), capture_output=True, text=True
    )


def _write_runtime_home(home: Path) -> None:
    (home / "SOUL.md").write_text("soul gen N\n", encoding="utf-8")
    (home / "auth.json").write_text('{"token": "SECRET"}', encoding="utf-8")
    (home / ".env").write_text("DISCORD_TOKEN=SECRET\n", encoding="utf-8")
    (home / "config.yaml").write_text("api_key: SECRET\n", encoding="utf-8")
    archive = home / "dgmh" / "soul_archive"
    archive.mkdir(parents=True)
    (archive / "20260101T000000Z__abc.md").write_text("old", encoding="utf-8")


def _commit_count(home: Path) -> int:
    result = _git(home, "rev-list", "--count", "HEAD")
    return int(result.stdout.strip()) if result.returncode == 0 else 0


def test_no_git_repo_is_noop(tmp_path):
    _write_runtime_home(tmp_path)
    assert _git_autocommit_soul_history(tmp_path, "gen 1 accepted") is False
    assert not (tmp_path / ".git").exists()


def test_refuses_without_default_deny_gitignore(tmp_path):
    _write_runtime_home(tmp_path)
    _git(tmp_path, "init", "-q")
    # No .gitignore at all: unknown files are not ignored → must refuse
    # before anything is staged.
    assert _git_autocommit_soul_history(tmp_path, "gen 1 accepted") is False
    assert _commit_count(tmp_path) == 0
    assert _git(tmp_path, "ls-files").stdout.strip() == ""


def test_refuses_when_a_secret_is_unignored(tmp_path):
    _write_runtime_home(tmp_path)
    _git(tmp_path, "init", "-q")
    # Allowlist that mistakenly re-includes a secret file.
    (tmp_path / ".gitignore").write_text(
        ALLOWLIST_GITIGNORE + "!auth.json\n", encoding="utf-8"
    )
    assert _git_autocommit_soul_history(tmp_path, "gen 1 accepted") is False
    assert _commit_count(tmp_path) == 0


def test_commits_allowlisted_files_only(tmp_path):
    _write_runtime_home(tmp_path)
    _git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text(ALLOWLIST_GITIGNORE, encoding="utf-8")

    assert (
        _git_autocommit_soul_history(tmp_path, "gen 7 accepted (hash abc123)")
        is True
    )
    assert _commit_count(tmp_path) == 1

    message = _git(tmp_path, "log", "--format=%s", "-1").stdout.strip()
    assert message == "chore(soul): gen 7 accepted (hash abc123)"

    tracked = set(_git(tmp_path, "ls-files").stdout.split())
    assert "SOUL.md" in tracked
    assert "dgmh/soul_archive/20260101T000000Z__abc.md" in tracked
    assert "auth.json" not in tracked
    assert ".env" not in tracked
    assert "config.yaml" not in tracked


def test_noop_when_nothing_changed(tmp_path):
    _write_runtime_home(tmp_path)
    _git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text(ALLOWLIST_GITIGNORE, encoding="utf-8")

    assert _git_autocommit_soul_history(tmp_path, "gen 7 accepted") is True
    assert _git_autocommit_soul_history(tmp_path, "gen 7 accepted") is False
    assert _commit_count(tmp_path) == 1

    # A subsequent SOUL.md change snapshots again.
    (tmp_path / "SOUL.md").write_text("soul gen N+1\n", encoding="utf-8")
    assert _git_autocommit_soul_history(tmp_path, "gen 8 accepted") is True
    assert _commit_count(tmp_path) == 2
