"""
dgmh/tests/test_bootstrap.py — Tests for archive_bootstrap.py.

Verifies:
- bootstrap_archive() returns N entries (one per skill)
- entries are deterministic across reseeds (same order, same content_hash)
- disabled skills registered with selectable=False
- generation_index is sorted 0..N-1 after (category, name) sort
- archive.jsonl is written atomically and re-loadable

Reference: playground/dgmh-engine/archive.ts, types.ts
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root importable
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers / fake skill registry
# ---------------------------------------------------------------------------

FAKE_SKILLS = [
    {"name": "skill-beta", "description": "Beta skill", "category": "cat-a"},
    {"name": "skill-alpha", "description": "Alpha skill", "category": "cat-a"},
    {"name": "skill-gamma", "description": "Gamma skill", "category": "cat-b"},
    {"name": "skill-delta", "description": "Delta skill (disabled)", "category": "cat-b"},
]

DISABLED_NAMES = {"skill-delta"}


def _make_fake_skills_dir(tmp_path: Path) -> None:
    """Create minimal SKILL.md files for fake skills."""
    for s in FAKE_SKILLS:
        cat, name = s["category"], s["name"]
        skill_dir = tmp_path / "skills" / cat / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {s['description']}\n---\n# {name}\nContent.",
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBootstrapArchive:
    def test_returns_entries_for_all_skills(self, tmp_path):
        """N skills → N entries."""
        _make_fake_skills_dir(tmp_path)

        with (
            patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}),
            patch("dgmh.archive_bootstrap._find_all_skills", return_value=FAKE_SKILLS),
            patch("dgmh.archive_bootstrap._get_disabled_skill_names", return_value=DISABLED_NAMES),
        ):
            from dgmh.archive_bootstrap import bootstrap_archive

            entries = bootstrap_archive(persist=True)

        assert len(entries) == len(FAKE_SKILLS)

    def test_sorted_by_category_then_name(self, tmp_path):
        """Entries must be sorted (category, name)."""
        _make_fake_skills_dir(tmp_path)

        with (
            patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}),
            patch("dgmh.archive_bootstrap._find_all_skills", return_value=FAKE_SKILLS),
            patch("dgmh.archive_bootstrap._get_disabled_skill_names", return_value=set()),
        ):
            from dgmh.archive_bootstrap import bootstrap_archive

            entries = bootstrap_archive(persist=False)

        keys = [(e.category, e.name) for e in entries]
        assert keys == sorted(keys), f"Expected sorted, got {keys}"

    def test_generation_index_sequential(self, tmp_path):
        """generation_index must be 0..N-1 after sort."""
        _make_fake_skills_dir(tmp_path)

        with (
            patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}),
            patch("dgmh.archive_bootstrap._find_all_skills", return_value=FAKE_SKILLS),
            patch("dgmh.archive_bootstrap._get_disabled_skill_names", return_value=set()),
        ):
            from dgmh.archive_bootstrap import bootstrap_archive

            entries = bootstrap_archive(persist=False)

        indices = [e.generation_index for e in entries]
        assert indices == list(range(len(entries))), f"Indices not sequential: {indices}"

    def test_disabled_skills_have_selectable_false(self, tmp_path):
        """Disabled skills must be registered with selectable=False."""
        _make_fake_skills_dir(tmp_path)

        with (
            patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}),
            patch("dgmh.archive_bootstrap._find_all_skills", return_value=FAKE_SKILLS),
            patch("dgmh.archive_bootstrap._get_disabled_skill_names", return_value=DISABLED_NAMES),
        ):
            from dgmh.archive_bootstrap import bootstrap_archive

            entries = bootstrap_archive(persist=False)

        disabled_entries = [e for e in entries if e.name in DISABLED_NAMES]
        assert len(disabled_entries) == len(DISABLED_NAMES), "All disabled skills must appear"
        for e in disabled_entries:
            assert e.selectable is False, f"{e.name} should be selectable=False"

        enabled_entries = [e for e in entries if e.name not in DISABLED_NAMES]
        for e in enabled_entries:
            assert e.selectable is True, f"{e.name} should be selectable=True"

    def test_deterministic_across_calls(self, tmp_path):
        """Two bootstrap calls with same skill set produce identical skill_ids + hashes."""
        _make_fake_skills_dir(tmp_path)

        def _run():
            with (
                patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}),
                patch("dgmh.archive_bootstrap._find_all_skills", return_value=FAKE_SKILLS),
                patch("dgmh.archive_bootstrap._get_disabled_skill_names", return_value=set()),
            ):
                from dgmh.archive_bootstrap import bootstrap_archive
                return bootstrap_archive(persist=False)

        entries_a = _run()
        entries_b = _run()

        assert [e.skill_id for e in entries_a] == [e.skill_id for e in entries_b]
        assert [e.content_hash for e in entries_a] == [e.content_hash for e in entries_b]

    def test_persists_jsonl_and_reloads(self, tmp_path):
        """archive.jsonl is written and re-loadable with correct entry count."""
        _make_fake_skills_dir(tmp_path)

        with (
            patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}),
            patch("dgmh.archive_bootstrap._find_all_skills", return_value=FAKE_SKILLS),
            patch("dgmh.archive_bootstrap._get_disabled_skill_names", return_value=set()),
        ):
            from dgmh.archive_bootstrap import bootstrap_archive, load_archive

            entries = bootstrap_archive(persist=True)
            archive_path = tmp_path / "dgmh" / "archive.jsonl"
            assert archive_path.exists(), "archive.jsonl should be created"

            loaded = load_archive(archive_path)

        assert len(loaded) == len(entries)
        assert [e.skill_id for e in loaded] == [e.skill_id for e in entries]

    def test_schema_version_is_1(self, tmp_path):
        """All entries must have schema_version=1."""
        _make_fake_skills_dir(tmp_path)

        with (
            patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}),
            patch("dgmh.archive_bootstrap._find_all_skills", return_value=FAKE_SKILLS),
            patch("dgmh.archive_bootstrap._get_disabled_skill_names", return_value=set()),
        ):
            from dgmh.archive_bootstrap import bootstrap_archive

            entries = bootstrap_archive(persist=False)

        for e in entries:
            assert e.schema_version == 1

    def test_content_hash_is_sha256(self, tmp_path):
        """content_hash should be a 64-char hex string (SHA-256) when SKILL.md readable."""
        _make_fake_skills_dir(tmp_path)

        with (
            patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}),
            patch("dgmh.archive_bootstrap._find_all_skills", return_value=FAKE_SKILLS),
            patch("dgmh.archive_bootstrap._get_disabled_skill_names", return_value=set()),
        ):
            from dgmh.archive_bootstrap import bootstrap_archive

            entries = bootstrap_archive(persist=False)

        for e in entries:
            if e.content_hash:
                assert len(e.content_hash) == 64, f"Expected SHA-256 hex for {e.skill_id}"
                assert all(c in "0123456789abcdef" for c in e.content_hash)

    def test_score_is_baseline_float(self, tmp_path):
        """Bootstrap score should be 0.5 (unevaluated baseline)."""
        _make_fake_skills_dir(tmp_path)

        with (
            patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}),
            patch("dgmh.archive_bootstrap._find_all_skills", return_value=FAKE_SKILLS),
            patch("dgmh.archive_bootstrap._get_disabled_skill_names", return_value=set()),
        ):
            from dgmh.archive_bootstrap import bootstrap_archive

            entries = bootstrap_archive(persist=False)

        for e in entries:
            assert isinstance(e.score, float)
            assert 0.0 <= e.score <= 1.0
