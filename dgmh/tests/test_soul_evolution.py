"""
dgmh/tests/test_soul_evolution.py — W6 tests for soul_evolution.py (>=8 tests).

Covers:
  1. gen-0 preservation: existing SOUL.md preserved as gen-0 on first run
  2. gen-0 idempotent: calling _ensure_gen0 twice only writes one entry
  3. Score formula: compute_score(0,0)=0.5, pos-heavy > 0.5, neg-heavy < 0.5
  4. Atomic SOUL.md replace on critic approve
  5. SOUL.md NOT touched on critic reject
  6. SOUL.md NOT touched on modifier error
  7. Lineage preserved across two evolutions
  8. RunRecord written on both accept and reject paths
  9. soul_archive.jsonl appended on accept
  10. Prior SOUL.md archived to soul_archive/ directory on accept
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dgmh.soul_evolution import (
    SoulEvolutionOpts,
    SoulArchiveEntry,
    _ensure_gen0,
    _load_soul_archive,
    _append_soul_archive_entry,
    compute_score,
    run_soul_evolution,
    _soul_archive_dir,
)
from dgmh.critic import CriticReview
from dgmh.modifier import SkillCandidate


# ---------------------------------------------------------------------------
# Fake modifier and critic
# ---------------------------------------------------------------------------


class FakeModifier:
    def __init__(self, new_content: str = "# Evolved SOUL\nImproved."):
        self._content = new_content
        self.call_count = 0

    def modify(self, parent: Any, archive: Any) -> SkillCandidate:
        self.call_count += 1
        return SkillCandidate(
            skill_id="soul/SOUL",
            skill_md=self._content,
            lineage=["deadbeef"] * 1,
            raw_codex_output=json.dumps({
                "skill_id": "soul/SOUL",
                "skill_md": self._content,
                "lineage": ["deadbeef"],
            }),
            attempt=0,
        )


class FakeCritic:
    def __init__(self, verdict: str = "approve", reason: str = "Good."):
        self._verdict = verdict
        self._reason = reason
        self.call_count = 0

    def review(self, child: Any, parent: Any, archive: Any) -> CriticReview:
        self.call_count += 1
        return CriticReview(verdict=self._verdict, reason=self._reason, elapsed_ms=1)


class ErrorModifier:
    def modify(self, parent: Any, archive: Any) -> SkillCandidate:
        from dgmh.modifier import CodexModifierError
        raise CodexModifierError("modifier broken", exit_code=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_opts(
    tmp_path: Path,
    modifier=None,
    critic=None,
    pos: int = 0,
    neg: int = 1,
    soul_content: str = "# Original SOUL\nHello.",
) -> tuple[Path, SoulEvolutionOpts]:
    soul_path = tmp_path / "SOUL.md"
    soul_path.write_text(soul_content, encoding="utf-8")
    archive_jsonl = tmp_path / "soul_archive.jsonl"
    runs_jsonl = tmp_path / "soul_runs.jsonl"

    opts = SoulEvolutionOpts(
        soul_md_path=soul_path,
        soul_archive_jsonl=archive_jsonl,
        soul_runs_jsonl=runs_jsonl,
        pos_reactions=pos,
        neg_reactions=neg,
        modifier=modifier or FakeModifier(),
        critic=critic or FakeCritic(),
        seed=42,
    )
    return soul_path, opts


# ---------------------------------------------------------------------------
# 1. gen-0 preservation
# ---------------------------------------------------------------------------


class TestGen0Preservation:
    def test_gen0_created_from_existing_soul_md(self, tmp_path):
        """Existing SOUL.md content is preserved as gen-0 on first _ensure_gen0."""
        soul_path = tmp_path / "SOUL.md"
        soul_path.write_text("# Operator Soul\nOriginal.", encoding="utf-8")
        archive_jsonl = tmp_path / "soul_archive.jsonl"

        gen0 = _ensure_gen0(soul_path, archive_jsonl)

        assert gen0.is_gen0 is True
        assert gen0.generation_index == 0
        assert gen0.soul_md_content == "# Operator Soul\nOriginal."
        assert gen0.status == "active"
        assert gen0.lineage == []

        # Verify persisted
        entries = _load_soul_archive(archive_jsonl)
        assert len(entries) == 1
        assert entries[0].is_gen0 is True

    def test_gen0_idempotent(self, tmp_path):
        """Calling _ensure_gen0 twice does not create duplicate entries."""
        soul_path = tmp_path / "SOUL.md"
        soul_path.write_text("# Soul", encoding="utf-8")
        archive_jsonl = tmp_path / "soul_archive.jsonl"

        _ensure_gen0(soul_path, archive_jsonl)
        _ensure_gen0(soul_path, archive_jsonl)

        entries = _load_soul_archive(archive_jsonl)
        gen0_entries = [e for e in entries if e.is_gen0]
        assert len(gen0_entries) == 1

    def test_gen0_preserves_original_content(self, tmp_path):
        """gen-0 content must exactly match what was in SOUL.md at first run."""
        original = "# EOS\nYou are EOS.\n\nHelp with gameworld."
        soul_path = tmp_path / "SOUL.md"
        soul_path.write_text(original, encoding="utf-8")
        archive_jsonl = tmp_path / "soul_archive.jsonl"

        gen0 = _ensure_gen0(soul_path, archive_jsonl)
        assert gen0.soul_md_content == original


# ---------------------------------------------------------------------------
# 3. Score formula
# ---------------------------------------------------------------------------


class TestScoreFormula:
    def test_zero_reactions_is_half(self):
        assert compute_score(0, 0) == pytest.approx(0.5)

    def test_positive_only_above_half(self):
        assert compute_score(5, 0) > 0.5

    def test_negative_only_below_half(self):
        assert compute_score(0, 5) < 0.5

    def test_equal_reactions_near_half(self):
        # pos=3, neg=3: (3+0.5)/(3+3+1) = 3.5/7 = 0.5
        assert compute_score(3, 3) == pytest.approx(0.5)

    def test_all_positive_approaches_one(self):
        score = compute_score(1000, 0)
        assert score > 0.99


# ---------------------------------------------------------------------------
# 4. Atomic SOUL.md replace on approve
# ---------------------------------------------------------------------------


class TestAtomicReplace:
    def test_soul_md_replaced_on_approve(self, tmp_path):
        """On critic approve, SOUL.md is updated with candidate content."""
        soul_path, opts = _make_opts(
            tmp_path,
            modifier=FakeModifier("# New Soul\nBetter."),
            critic=FakeCritic("approve"),
        )

        accepted = run_soul_evolution(opts)

        assert accepted is True
        assert soul_path.read_text(encoding="utf-8") == "# New Soul\nBetter."

    def test_soul_md_unchanged_on_reject(self, tmp_path):
        """On critic reject, SOUL.md is NOT touched."""
        original = "# Original Soul\nUnchanged."
        soul_path, opts = _make_opts(
            tmp_path,
            soul_content=original,
            modifier=FakeModifier("# New Soul\nRejected."),
            critic=FakeCritic("reject", "Too much drift."),
        )

        accepted = run_soul_evolution(opts)

        assert accepted is False
        assert soul_path.read_text(encoding="utf-8") == original

    def test_soul_md_unchanged_on_modifier_error(self, tmp_path):
        """On modifier error, SOUL.md is NOT touched."""
        original = "# Original Soul\nStill here."
        soul_path, opts = _make_opts(
            tmp_path,
            soul_content=original,
            modifier=ErrorModifier(),
            critic=FakeCritic("approve"),
        )

        accepted = run_soul_evolution(opts)

        assert accepted is False
        assert soul_path.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# 7. Lineage preserved across evolutions
# ---------------------------------------------------------------------------


class TestLineagePreservation:
    def test_lineage_grows_across_two_evolutions(self, tmp_path):
        """After two accepted evolutions, the second entry's lineage has 2 hashes."""
        import hashlib

        soul_path, opts1 = _make_opts(
            tmp_path,
            soul_content="# Gen0\nOriginal.",
            modifier=FakeModifier("# Gen1\nFirst mutation."),
            critic=FakeCritic("approve"),
        )

        # First evolution
        accepted1 = run_soul_evolution(opts1)
        assert accepted1 is True

        # Second evolution — re-use same paths, content now updated
        opts2 = SoulEvolutionOpts(
            soul_md_path=soul_path,
            soul_archive_jsonl=opts1.soul_archive_jsonl,
            soul_runs_jsonl=opts1.soul_runs_jsonl,
            pos_reactions=0,
            neg_reactions=1,
            modifier=FakeModifier("# Gen2\nSecond mutation."),
            critic=FakeCritic("approve"),
            seed=99,
        )
        accepted2 = run_soul_evolution(opts2)
        assert accepted2 is True

        # Check archive: last active entry should have lineage of depth 2
        entries = _load_soul_archive(opts1.soul_archive_jsonl)
        active = [e for e in entries if e.status == "active"]
        assert len(active) == 1
        assert len(active[0].lineage) == 2


# ---------------------------------------------------------------------------
# 8. RunRecord written on both paths
# ---------------------------------------------------------------------------


class TestRunRecordWrite:
    def test_run_record_written_on_accept(self, tmp_path):
        soul_path, opts = _make_opts(
            tmp_path,
            modifier=FakeModifier(),
            critic=FakeCritic("approve"),
        )
        run_soul_evolution(opts)
        assert opts.soul_runs_jsonl.exists()
        lines = [l for l in opts.soul_runs_jsonl.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["accepted"] is True

    def test_run_record_written_on_reject(self, tmp_path):
        soul_path, opts = _make_opts(
            tmp_path,
            modifier=FakeModifier(),
            critic=FakeCritic("reject", "No good."),
        )
        run_soul_evolution(opts)
        assert opts.soul_runs_jsonl.exists()
        lines = [l for l in opts.soul_runs_jsonl.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["accepted"] is False
        assert "critic-reject" in record["rejection_reason"]


# ---------------------------------------------------------------------------
# 9. soul_archive.jsonl appended on accept
# ---------------------------------------------------------------------------


class TestArchiveAppend:
    def test_archive_appended_on_accept(self, tmp_path):
        soul_path, opts = _make_opts(
            tmp_path,
            modifier=FakeModifier("# New\nContent."),
            critic=FakeCritic("approve"),
        )
        run_soul_evolution(opts)

        entries = _load_soul_archive(opts.soul_archive_jsonl)
        # gen-0 + new active entry
        assert len(entries) >= 2
        active = [e for e in entries if e.status == "active"]
        assert len(active) == 1
        assert active[0].soul_md_content == "# New\nContent."

    def test_archive_not_appended_on_reject(self, tmp_path):
        soul_path, opts = _make_opts(
            tmp_path,
            modifier=FakeModifier("# Rejected."),
            critic=FakeCritic("reject"),
        )
        run_soul_evolution(opts)

        entries = _load_soul_archive(opts.soul_archive_jsonl)
        # Only gen-0 should be present
        active = [e for e in entries if e.status == "active"]
        assert len(active) == 1
        assert active[0].is_gen0 is True


# ---------------------------------------------------------------------------
# 10. Prior SOUL.md archived to soul_archive/ directory on accept
# ---------------------------------------------------------------------------


class TestPhysicalArchive:
    def test_prior_soul_md_archived_on_accept(self, tmp_path):
        """After accept, old SOUL.md content is written to soul_archive/ dir."""
        original = "# Original\nOld content."
        soul_path, opts = _make_opts(
            tmp_path,
            soul_content=original,
            modifier=FakeModifier("# New\nNew content."),
            critic=FakeCritic("approve"),
        )

        # Override archive dir to use tmp_path
        archive_dir = tmp_path / "soul_archive"

        with patch("dgmh.soul_evolution._soul_archive_dir", return_value=archive_dir):
            run_soul_evolution(opts)

        # soul_archive/ should contain at least one .md file with the old content
        md_files = list(archive_dir.glob("*.md")) if archive_dir.exists() else []
        assert len(md_files) >= 1
        found = any(original in f.read_text(encoding="utf-8") for f in md_files)
        assert found, "Old SOUL.md content not found in soul_archive/"
