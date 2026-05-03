"""
dgmh/tests/test_loop.py — Tests for loop.py (W4 deliverable).

Covers per W4 spec (>=6 tests):
1.  End-to-end with mock modifier+critic: full approve path
2.  All-approve case: child admitted to archive, SKILL.md written
3.  Critic-reject case: rejection logged, no skill write, no archive admission
4.  Modifier-error case: rejection logged, no skill write
5.  Monotonic generation_index allocation
6.  Lock contention test (fake SkillLockError)
7.  Idempotent run record write: RunRecord always appended
8.  Critic-error (subprocess failure) case: logs critic-error rejection
9.  No critic gate: when critic=None, approve proceeds without critic

Reference: playground/dgmh-engine/loop.ts::runDgmh (TS driver)
           dgmh/loop.py (implementation under test)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dgmh.critic import CriticReview, CodexCriticError, CriticParseAttemptsExhaustedError
from dgmh.locks import SkillLockError
from dgmh.modifier import SkillCandidate, CodexModifierError, ScopeAttemptsExhaustedError
from dgmh.run_log import read_run_records
from dgmh.select_parents import Generation
from dgmh.loop import (
    IterationOpts,
    IterationResult,
    _next_monotonic_index,
    run_one_iteration,
)


# ---------------------------------------------------------------------------
# Fake modifier and critic objects
# ---------------------------------------------------------------------------


class FakeModifier:
    """Mock modifier — returns a scripted SkillCandidate or raises."""

    def __init__(self, response: SkillCandidate | Exception) -> None:
        self._response = response
        self.call_count = 0

    def modify(self, parent: Any, archive: Any) -> SkillCandidate:
        self.call_count += 1
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class FakeCritic:
    """Mock critic — returns a scripted CriticReview or raises."""

    def __init__(self, response: CriticReview | Exception) -> None:
        self._response = response
        self.call_count = 0

    def review(self, child: Any, parent: Any, archive: Any) -> CriticReview:
        self.call_count += 1
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PARENT_MD = "# Parent\nOriginal content."
SAMPLE_CHILD_MD = "# Child\nImproved content."


def _make_archive(n: int = 3) -> list[Generation]:
    return [
        Generation(
            id=f"coding/skill-{i}",
            score=0.5 + i * 0.1,
            compiled_children=0,
            generation_index=i,
            selectable=True,
            extra={"skill_md": SAMPLE_PARENT_MD},
        )
        for i in range(n)
    ]


def _make_candidate(
    skill_id: str = "coding/skill-child",
    parent_id: str = "coding/skill-0",
) -> SkillCandidate:
    return SkillCandidate(
        skill_id=skill_id,
        skill_md=SAMPLE_CHILD_MD,
        lineage=[parent_id],
        raw_codex_output=json.dumps({"skill_id": skill_id, "skill_md": SAMPLE_CHILD_MD, "lineage": [parent_id]}),
        attempt=0,
    )


def _approve_review(reason: str = "Looks good.") -> CriticReview:
    return CriticReview(verdict="approve", reason=reason, elapsed_ms=10)


def _reject_review(reason: str = "Cargo-cult.") -> CriticReview:
    return CriticReview(verdict="reject", reason=reason, elapsed_ms=10)


# ---------------------------------------------------------------------------
# 1. End-to-end: mock modifier + critic, approve path
# ---------------------------------------------------------------------------

class TestEndToEndApprove:
    def test_full_approve_path(self, tmp_path):
        archive = _make_archive(3)
        candidate = _make_candidate(
            skill_id="coding/skill-child",
            parent_id=archive[0].id,  # selector will pick some parent
        )
        # Make candidate match any parent by adapting after selection
        modifier = FakeModifier(candidate)
        critic = FakeCritic(_approve_review("Clean improvement."))

        opts = IterationOpts(
            seed_archive=archive,
            modifier=modifier,
            critic=critic,
            seed=42,
            archive_jsonl_path=tmp_path / "archive.jsonl",
            runs_jsonl_path=tmp_path / "runs.jsonl",
        )

        # Patch lock to avoid filesystem lock operations, and _atomic_write_text
        with patch("dgmh.loop.acquire_skill_lock") as mock_lock, \
             patch("dgmh.loop._atomic_write_text") as mock_write:
            mock_lock.return_value.__enter__ = MagicMock(return_value=None)
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)

            result = run_one_iteration(opts)

        assert isinstance(result, IterationResult)
        assert result.accepted is True
        assert result.rejection_reason is None
        assert result.candidate is not None
        mock_write.assert_called_once()


# ---------------------------------------------------------------------------
# 2. All-approve: SKILL.md written, archive updated, RunRecord recorded
# ---------------------------------------------------------------------------

class TestAllApprove:
    def test_skill_md_written_on_approve(self, tmp_path):
        archive = _make_archive(2)
        candidate = _make_candidate(parent_id=archive[0].id)
        modifier = FakeModifier(candidate)
        critic = FakeCritic(_approve_review())

        opts = IterationOpts(
            seed_archive=archive,
            modifier=modifier,
            critic=critic,
            seed=1,
            archive_jsonl_path=tmp_path / "archive.jsonl",
            runs_jsonl_path=tmp_path / "runs.jsonl",
        )

        written_content = {}

        def fake_write(path, content):
            written_content["path"] = path
            written_content["content"] = content

        with patch("dgmh.loop.acquire_skill_lock") as mock_lock, \
             patch("dgmh.loop._atomic_write_text", side_effect=fake_write):
            mock_lock.return_value.__enter__ = MagicMock(return_value=None)
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)
            result = run_one_iteration(opts)

        assert result.accepted is True
        assert SAMPLE_CHILD_MD in written_content.get("content", "")

    def test_run_record_appended_on_approve(self, tmp_path):
        archive = _make_archive(2)
        candidate = _make_candidate(parent_id=archive[0].id)
        modifier = FakeModifier(candidate)
        critic = FakeCritic(_approve_review())

        runs_file = tmp_path / "runs.jsonl"
        opts = IterationOpts(
            seed_archive=archive,
            modifier=modifier,
            critic=critic,
            seed=2,
            archive_jsonl_path=tmp_path / "archive.jsonl",
            runs_jsonl_path=runs_file,
        )

        with patch("dgmh.loop.acquire_skill_lock") as mock_lock, \
             patch("dgmh.loop._atomic_write_text"):
            mock_lock.return_value.__enter__ = MagicMock(return_value=None)
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)
            result = run_one_iteration(opts)

        records = read_run_records(runs_file)
        assert len(records) == 1
        assert records[0].seed == 2
        assert len(records[0].accepted_children) == 1
        assert records[0].accepted_children[0].id == "coding/skill-child"

    def test_archive_jsonl_appended_on_approve(self, tmp_path):
        archive = _make_archive(2)
        candidate = _make_candidate(parent_id=archive[0].id)
        modifier = FakeModifier(candidate)
        critic = FakeCritic(_approve_review())

        archive_file = tmp_path / "archive.jsonl"
        opts = IterationOpts(
            seed_archive=archive,
            modifier=modifier,
            critic=critic,
            seed=3,
            archive_jsonl_path=archive_file,
            runs_jsonl_path=tmp_path / "runs.jsonl",
        )

        with patch("dgmh.loop.acquire_skill_lock") as mock_lock, \
             patch("dgmh.loop._atomic_write_text"):
            mock_lock.return_value.__enter__ = MagicMock(return_value=None)
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)
            result = run_one_iteration(opts)

        assert result.accepted is True
        assert archive_file.exists()
        lines = [l for l in archive_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["skill_id"] == "coding/skill-child"


# ---------------------------------------------------------------------------
# 3. Critic-reject: rejection logged, no skill write, no archive admission
# ---------------------------------------------------------------------------

class TestCriticReject:
    def test_critic_reject_no_skill_write(self, tmp_path):
        archive = _make_archive(2)
        candidate = _make_candidate(parent_id=archive[0].id)
        modifier = FakeModifier(candidate)
        critic = FakeCritic(_reject_review("Scope leak detected."))

        opts = IterationOpts(
            seed_archive=archive,
            modifier=modifier,
            critic=critic,
            seed=10,
            archive_jsonl_path=tmp_path / "archive.jsonl",
            runs_jsonl_path=tmp_path / "runs.jsonl",
        )

        with patch("dgmh.loop.acquire_skill_lock") as mock_lock, \
             patch("dgmh.loop._atomic_write_text") as mock_write:
            mock_lock.return_value.__enter__ = MagicMock(return_value=None)
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)
            result = run_one_iteration(opts)

        assert result.accepted is False
        assert result.rejection_reason == "critic-reject"
        assert "Scope leak" in result.rejection_message
        mock_write.assert_not_called()

    def test_critic_reject_logged_in_run_record(self, tmp_path):
        archive = _make_archive(2)
        candidate = _make_candidate(parent_id=archive[0].id)
        modifier = FakeModifier(candidate)
        critic = FakeCritic(_reject_review("Empty mutation."))

        runs_file = tmp_path / "runs.jsonl"
        opts = IterationOpts(
            seed_archive=archive,
            modifier=modifier,
            critic=critic,
            seed=11,
            archive_jsonl_path=tmp_path / "archive.jsonl",
            runs_jsonl_path=runs_file,
        )

        with patch("dgmh.loop.acquire_skill_lock") as mock_lock, \
             patch("dgmh.loop._atomic_write_text"):
            mock_lock.return_value.__enter__ = MagicMock(return_value=None)
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)
            run_one_iteration(opts)

        records = read_run_records(runs_file)
        assert len(records) == 1
        assert len(records[0].rejections) == 1
        assert records[0].rejections[0].reason == "critic-reject"
        assert len(records[0].accepted_children) == 0

    def test_critic_reject_no_archive_admission(self, tmp_path):
        archive = _make_archive(2)
        candidate = _make_candidate(parent_id=archive[0].id)
        modifier = FakeModifier(candidate)
        critic = FakeCritic(_reject_review("Structural regression."))

        archive_file = tmp_path / "archive.jsonl"
        opts = IterationOpts(
            seed_archive=archive,
            modifier=modifier,
            critic=critic,
            seed=12,
            archive_jsonl_path=archive_file,
            runs_jsonl_path=tmp_path / "runs.jsonl",
        )

        with patch("dgmh.loop.acquire_skill_lock") as mock_lock, \
             patch("dgmh.loop._atomic_write_text"):
            mock_lock.return_value.__enter__ = MagicMock(return_value=None)
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)
            result = run_one_iteration(opts)

        # archive.jsonl must not be created/written
        assert not archive_file.exists() or archive_file.read_text().strip() == ""


# ---------------------------------------------------------------------------
# 4. Modifier-error case: rejection logged, no skill write
# ---------------------------------------------------------------------------

class TestModifierError:
    def test_modifier_error_no_skill_write(self, tmp_path):
        archive = _make_archive(2)
        modifier = FakeModifier(CodexModifierError("subprocess failed", exit_code=1))
        critic = FakeCritic(_approve_review())

        opts = IterationOpts(
            seed_archive=archive,
            modifier=modifier,
            critic=critic,
            seed=20,
            archive_jsonl_path=tmp_path / "archive.jsonl",
            runs_jsonl_path=tmp_path / "runs.jsonl",
        )

        with patch("dgmh.loop.acquire_skill_lock") as mock_lock, \
             patch("dgmh.loop._atomic_write_text") as mock_write:
            mock_lock.return_value.__enter__ = MagicMock(return_value=None)
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)
            result = run_one_iteration(opts)

        assert result.accepted is False
        assert result.rejection_reason == "modifier-error"
        mock_write.assert_not_called()

    def test_modifier_error_logged_in_run_record(self, tmp_path):
        archive = _make_archive(2)
        modifier = FakeModifier(ScopeAttemptsExhaustedError(
            "all attempts failed", parent_id="coding/skill-0", attempts=3
        ))

        runs_file = tmp_path / "runs.jsonl"
        opts = IterationOpts(
            seed_archive=archive,
            modifier=modifier,
            critic=None,
            seed=21,
            archive_jsonl_path=tmp_path / "archive.jsonl",
            runs_jsonl_path=runs_file,
        )

        with patch("dgmh.loop.acquire_skill_lock") as mock_lock, \
             patch("dgmh.loop._atomic_write_text"):
            mock_lock.return_value.__enter__ = MagicMock(return_value=None)
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)
            run_one_iteration(opts)

        records = read_run_records(runs_file)
        assert len(records) == 1
        assert records[0].rejections[0].reason == "modifier-error"


# ---------------------------------------------------------------------------
# 5. Monotonic generation_index allocation
# ---------------------------------------------------------------------------

class TestMonotonicGenerationIndex:
    def test_next_monotonic_index_empty(self):
        assert _next_monotonic_index([]) == 0

    def test_next_monotonic_index_basic(self):
        archive = _make_archive(3)  # indices 0, 1, 2
        assert _next_monotonic_index(archive) == 3

    def test_next_monotonic_index_gaps(self):
        archive = [
            Generation("a", 0.5, generation_index=0),
            Generation("b", 0.6, generation_index=5),
            Generation("c", 0.7, generation_index=3),
        ]
        assert _next_monotonic_index(archive) == 6

    def test_admitted_child_gets_next_index(self, tmp_path):
        archive = _make_archive(3)  # max index = 2
        candidate = _make_candidate(parent_id=archive[0].id)
        modifier = FakeModifier(candidate)
        critic = FakeCritic(_approve_review())

        runs_file = tmp_path / "runs.jsonl"
        opts = IterationOpts(
            seed_archive=archive,
            modifier=modifier,
            critic=critic,
            seed=30,
            archive_jsonl_path=tmp_path / "archive.jsonl",
            runs_jsonl_path=runs_file,
        )

        with patch("dgmh.loop.acquire_skill_lock") as mock_lock, \
             patch("dgmh.loop._atomic_write_text"):
            mock_lock.return_value.__enter__ = MagicMock(return_value=None)
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)
            result = run_one_iteration(opts)

        assert result.accepted is True
        records = read_run_records(runs_file)
        # Child generation_index must be > max existing (2)
        child_gen_idx = records[0].accepted_children[0].generation_index
        assert child_gen_idx == 3


# ---------------------------------------------------------------------------
# 6. Lock contention test (fake SkillLockError)
# ---------------------------------------------------------------------------

class TestLockContention:
    def test_skill_lock_error_treated_as_modifier_error(self, tmp_path):
        archive = _make_archive(2)
        modifier = FakeModifier(_make_candidate())
        critic = FakeCritic(_approve_review())

        runs_file = tmp_path / "runs.jsonl"
        opts = IterationOpts(
            seed_archive=archive,
            modifier=modifier,
            critic=critic,
            seed=40,
            archive_jsonl_path=tmp_path / "archive.jsonl",
            runs_jsonl_path=runs_file,
        )

        def _raise_lock(*args, **kwargs):
            class _Ctx:
                def __enter__(self):
                    raise SkillLockError("timeout", category="coding", name="skill-0")
                def __exit__(self, *a):
                    return False
            return _Ctx()

        with patch("dgmh.loop.acquire_skill_lock", side_effect=_raise_lock):
            result = run_one_iteration(opts)

        assert result.accepted is False
        assert result.rejection_reason == "modifier-error"
        assert "lock" in result.rejection_message.lower()

        records = read_run_records(runs_file)
        assert len(records) == 1
        assert records[0].rejections[0].reason == "modifier-error"


# ---------------------------------------------------------------------------
# 7. Idempotent run record write (RunRecord always appended)
# ---------------------------------------------------------------------------

class TestIdempotentRunRecordWrite:
    def test_run_record_always_appended_on_reject(self, tmp_path):
        archive = _make_archive(2)
        modifier = FakeModifier(CodexModifierError("broken"))
        runs_file = tmp_path / "runs.jsonl"

        opts = IterationOpts(
            seed_archive=archive,
            modifier=modifier,
            seed=50,
            archive_jsonl_path=tmp_path / "archive.jsonl",
            runs_jsonl_path=runs_file,
        )

        with patch("dgmh.loop.acquire_skill_lock") as mock_lock:
            mock_lock.return_value.__enter__ = MagicMock(return_value=None)
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)
            result = run_one_iteration(opts)

        assert not result.accepted
        records = read_run_records(runs_file)
        assert len(records) == 1  # always one record per iteration

    def test_run_record_always_appended_on_approve(self, tmp_path):
        archive = _make_archive(2)
        candidate = _make_candidate(parent_id=archive[0].id)
        modifier = FakeModifier(candidate)
        runs_file = tmp_path / "runs.jsonl"

        opts = IterationOpts(
            seed_archive=archive,
            modifier=modifier,
            seed=51,
            archive_jsonl_path=tmp_path / "archive.jsonl",
            runs_jsonl_path=runs_file,
        )

        with patch("dgmh.loop.acquire_skill_lock") as mock_lock, \
             patch("dgmh.loop._atomic_write_text"):
            mock_lock.return_value.__enter__ = MagicMock(return_value=None)
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)
            result = run_one_iteration(opts)

        records = read_run_records(runs_file)
        assert len(records) == 1


# ---------------------------------------------------------------------------
# 8. Critic-error case (subprocess failure)
# ---------------------------------------------------------------------------

class TestCriticError:
    def test_critic_subprocess_error_logged_as_critic_error(self, tmp_path):
        archive = _make_archive(2)
        candidate = _make_candidate(parent_id=archive[0].id)
        modifier = FakeModifier(candidate)
        critic = FakeCritic(CodexCriticError("critic subprocess failed", exit_code=1))

        runs_file = tmp_path / "runs.jsonl"
        opts = IterationOpts(
            seed_archive=archive,
            modifier=modifier,
            critic=critic,
            seed=60,
            archive_jsonl_path=tmp_path / "archive.jsonl",
            runs_jsonl_path=runs_file,
        )

        with patch("dgmh.loop.acquire_skill_lock") as mock_lock, \
             patch("dgmh.loop._atomic_write_text") as mock_write:
            mock_lock.return_value.__enter__ = MagicMock(return_value=None)
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)
            result = run_one_iteration(opts)

        assert result.accepted is False
        assert result.rejection_reason == "critic-error"
        mock_write.assert_not_called()

        records = read_run_records(runs_file)
        assert records[0].rejections[0].reason == "critic-error"

    def test_critic_parse_exhausted_logged_as_critic_error(self, tmp_path):
        archive = _make_archive(2)
        candidate = _make_candidate(parent_id=archive[0].id)
        modifier = FakeModifier(candidate)
        critic = FakeCritic(CriticParseAttemptsExhaustedError(
            "parse exhausted", attempts=2
        ))

        runs_file = tmp_path / "runs.jsonl"
        opts = IterationOpts(
            seed_archive=archive,
            modifier=modifier,
            critic=critic,
            seed=61,
            archive_jsonl_path=tmp_path / "archive.jsonl",
            runs_jsonl_path=runs_file,
        )

        with patch("dgmh.loop.acquire_skill_lock") as mock_lock, \
             patch("dgmh.loop._atomic_write_text"):
            mock_lock.return_value.__enter__ = MagicMock(return_value=None)
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)
            result = run_one_iteration(opts)

        assert result.rejection_reason == "critic-error"


# ---------------------------------------------------------------------------
# 9. No critic gate
# ---------------------------------------------------------------------------

class TestNoCriticGate:
    def test_no_critic_approves_without_gate(self, tmp_path):
        archive = _make_archive(2)
        candidate = _make_candidate(parent_id=archive[0].id)
        modifier = FakeModifier(candidate)

        opts = IterationOpts(
            seed_archive=archive,
            modifier=modifier,
            critic=None,  # no critic
            seed=70,
            archive_jsonl_path=tmp_path / "archive.jsonl",
            runs_jsonl_path=tmp_path / "runs.jsonl",
        )

        with patch("dgmh.loop.acquire_skill_lock") as mock_lock, \
             patch("dgmh.loop._atomic_write_text"):
            mock_lock.return_value.__enter__ = MagicMock(return_value=None)
            mock_lock.return_value.__exit__ = MagicMock(return_value=False)
            result = run_one_iteration(opts)

        assert result.accepted is True
        assert result.rejection_reason is None
