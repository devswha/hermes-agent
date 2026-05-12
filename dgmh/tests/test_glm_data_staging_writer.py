"""Tests for ``dgmh.glm_data.staging_writer.StagingWriter``.

Plan reference: ``.omc/plans/dgmh-glm-data-v1.md`` §3 AC13 (primary defense)
and §4 Step 1. This file is intentionally large because it is the unit-level
proof that no future glm_data subcommand can scribble over production
telemetry files in ``~/.hermes/dgmh/``.
"""

from __future__ import annotations

import os
import random
import string
from pathlib import Path

import pytest

from dgmh.glm_data import STAGING_ROOT_RELATIVE
from dgmh.glm_data.staging_writer import PathGuardError, StagingWriter


# --------------------------------------------------------------------- fixtures


@pytest.fixture()
def fake_home(tmp_path: Path) -> Path:
    """A synthetic ``$HOME`` so tests never touch the operator's real .hermes/."""

    home = tmp_path / "home"
    (home / ".hermes" / "dgmh").mkdir(parents=True)
    return home


@pytest.fixture()
def writer(fake_home: Path) -> StagingWriter:
    return StagingWriter("sess-001", home=fake_home)


# Nine production files that already exist under ~/.hermes/dgmh/ plus the
# soul_archive/ directory and the canonical SOUL.md at the .hermes root.
# These are the explicit deny targets called out in the task brief.
PRODUCTION_TARGETS = (
    ".hermes/dgmh/kakao_hako_corpus.txt",
    ".hermes/dgmh/humanness_log.jsonl",
    ".hermes/dgmh/soul_archive.jsonl",
    ".hermes/dgmh/verification_runs.jsonl",
    ".hermes/dgmh/reaction_events.jsonl",
    ".hermes/dgmh/persona_drift.jsonl",
    ".hermes/dgmh/ai_callout_log.jsonl",
    ".hermes/dgmh/soul_archive/run-2026-05-12.jsonl",
    ".hermes/dgmh/soul_runs.jsonl",
    ".hermes/SOUL.md",
)


# -------------------------------------------------------------------- denylist


@pytest.mark.parametrize("rel", PRODUCTION_TARGETS)
def test_rejects_production_telemetry_files(
    writer: StagingWriter, fake_home: Path, rel: str
) -> None:
    """All 9 production paths + SOUL.md raise PathGuardError, no bytes written."""

    target = fake_home / rel
    # Sanity: target is *not* under the session root.
    assert STAGING_ROOT_RELATIVE.parts[0] not in target.relative_to(fake_home).parts \
        or "glm_data_v1" not in target.parts

    before = target.exists()
    with pytest.raises(PathGuardError):
        writer.write_text(target, "leaked!")
    # The writer must not have created the file as a side-effect.
    assert target.exists() == before


def test_rejects_repo_working_tree_paths(
    writer: StagingWriter, tmp_path: Path
) -> None:
    """Paths under a sibling git repo tree are rejected."""

    fake_repo = tmp_path / "fake_repo" / ".git" / "config"
    with pytest.raises(PathGuardError):
        writer.write_text(fake_repo, "x")


# ------------------------------------------------------------- property (50x)


def _random_outside_path(rng: random.Random, fake_home: Path) -> Path:
    """Synthesize a random path that must NOT be under the staging root."""

    pool = string.ascii_lowercase + string.digits + "/"
    while True:
        depth = rng.randint(1, 5)
        parts = [
            "".join(rng.choices(string.ascii_lowercase + string.digits, k=rng.randint(1, 8)))
            for _ in range(depth)
        ]
        # Bias toward dangerous prefixes so the property test exercises real
        # adversarial shapes (production telemetry, /etc, /tmp, /var, repos).
        prefix = rng.choice([
            fake_home / ".hermes" / "dgmh",
            fake_home / ".hermes",
            fake_home,
            Path("/tmp"),
            Path("/etc"),
            Path("/var/log"),
            fake_home / "workspace" / "dgmh",
            fake_home / "workspace" / "hermes-agent" / "dgmh",
            fake_home / ".hermes" / "dgmh" / "glm_data_v1",  # near-miss
            fake_home / ".hermes" / "dgmh" / "glm_data_v1" / "staging",  # near-miss
        ])
        candidate = prefix.joinpath(*parts)
        # Filter: ensure the candidate is genuinely *not* inside the session root
        # so the test asserts the guard, not the candidate generator.
        session_root = (
            fake_home / STAGING_ROOT_RELATIVE / "sess-001"
        ).resolve()
        try:
            candidate.resolve().relative_to(session_root)
        except ValueError:
            return candidate
        # Otherwise loop and try again.
        _ = pool  # keep linter quiet


def test_property_50_random_paths_rejected(
    writer: StagingWriter, fake_home: Path
) -> None:
    rng = random.Random(0xDA7A)  # deterministic seed for reproducibility
    for _ in range(50):
        bogus = _random_outside_path(rng, fake_home)
        with pytest.raises(PathGuardError):
            writer.write_text(bogus, "nope")


# ------------------------------------------------------------------ happy path


def test_write_text_inside_session_root(
    writer: StagingWriter, fake_home: Path
) -> None:
    target = writer.write_text("inventory/inventory.md", "# inventory\n")
    expected_root = (
        fake_home / STAGING_ROOT_RELATIVE / "sess-001"
    ).resolve()
    assert target.is_file()
    assert target.read_text(encoding="utf-8") == "# inventory\n"
    target.relative_to(expected_root)  # raises if escape


def test_write_bytes_inside_session_root(writer: StagingWriter) -> None:
    target = writer.write_bytes("checks/snapshot.bin", b"\x00\x01\x02")
    assert target.read_bytes() == b"\x00\x01\x02"


def test_write_jsonl_inside_session_root(writer: StagingWriter) -> None:
    rows = [{"msg_id": "m1", "ok": True}, {"msg_id": "m2", "ok": False}]
    target = writer.write_jsonl("approvals.jsonl", rows)
    contents = target.read_text(encoding="utf-8").splitlines()
    assert len(contents) == 2
    assert contents[0].startswith("{")
    # Each line independently parses as JSON.
    import json as _json

    parsed = [_json.loads(line) for line in contents]
    assert parsed == rows


def test_absolute_path_inside_root_accepted(
    writer: StagingWriter, fake_home: Path
) -> None:
    abs_target = (
        fake_home / STAGING_ROOT_RELATIVE / "sess-001" / "anchor" / "x.txt"
    )
    writer.write_text(abs_target, "ok")
    assert abs_target.read_text(encoding="utf-8") == "ok"


def test_refuses_to_write_session_root_itself(writer: StagingWriter) -> None:
    with pytest.raises(PathGuardError):
        writer.write_text(".", "x")


# --------------------------------------------------------------------- escapes


def test_relative_dotdot_escape_blocked(
    writer: StagingWriter, fake_home: Path
) -> None:
    """A relative path with .. that escapes the session root is rejected."""

    with pytest.raises(PathGuardError):
        writer.write_text("../sess-002/foo.txt", "x")
    with pytest.raises(PathGuardError):
        writer.write_text("../../../etc/passwd", "x")


def test_symlink_escape_blocked(
    writer: StagingWriter, fake_home: Path, tmp_path: Path
) -> None:
    """A symlink inside the session root that points outside is rejected.

    This is the key test: ``Path.resolve()`` must canonicalize the symlink so
    the guard sees the *real* target, not the symlink path.
    """

    session_root = (fake_home / STAGING_ROOT_RELATIVE / "sess-001").resolve()
    session_root.mkdir(parents=True, exist_ok=True)
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()

    # Symlink: session_root/escape -> tmp_path/outside
    link = session_root / "escape"
    os.symlink(outside_dir, link)

    # Writing to escape/file.txt would land in outside/file.txt — must reject.
    with pytest.raises(PathGuardError):
        writer.write_text("escape/file.txt", "leaked")
    assert not (outside_dir / "file.txt").exists()


def test_symlink_inside_root_allowed(
    writer: StagingWriter, fake_home: Path
) -> None:
    """A symlink that stays *inside* the session root must still work."""

    session_root = (fake_home / STAGING_ROOT_RELATIVE / "sess-001").resolve()
    session_root.mkdir(parents=True, exist_ok=True)
    real_dir = session_root / "real"
    real_dir.mkdir()
    link = session_root / "alias"
    os.symlink(real_dir, link)

    writer.write_text("alias/x.txt", "inside-link")
    assert (real_dir / "x.txt").read_text(encoding="utf-8") == "inside-link"


# -------------------------------------------------------------- session_id sanity


def test_session_id_must_be_safe() -> None:
    with pytest.raises(ValueError):
        StagingWriter("../escape")
    with pytest.raises(ValueError):
        StagingWriter("a/b")
    with pytest.raises(ValueError):
        StagingWriter("")
    with pytest.raises(ValueError):
        StagingWriter("-leading-hyphen-not-allowed")


def test_session_id_accepts_typical_ids() -> None:
    StagingWriter("sess-001")
    StagingWriter("run_2026_05_12T15_00_00")
    StagingWriter("a")
