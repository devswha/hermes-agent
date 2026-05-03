"""
dgmh/tests/test_run_log.py — Tests for run_log.py (W4 deliverable).

Covers per W4 spec (>=5 tests):
1. append + read round-trip
2. schema_version filter (wrong version skipped)
3. malformed-line skip
4. mkdir parents on missing dir
5. summarize_bias_report placeholder returns zero-counts struct

Reference: playground/dgmh-engine/runLog.ts (TS source)
           dgmh/run_log.py (implementation under test)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dgmh.run_log import (
    RunRecord,
    RunRecordBiasSummary,
    RunRecordChild,
    RunRecordRejection,
    append_run_record,
    make_run_record,
    read_run_records,
    summarize_bias_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_record(seed: int = 42, accepted: bool = True) -> RunRecord:
    children = []
    rejections = []
    if accepted:
        children = [
            RunRecordChild(
                id="coding/test-skill",
                parent_id="coding/parent-skill",
                score=0.72,
                generation_index=5,
                lineage_depth=2,
            )
        ]
    else:
        rejections = [
            RunRecordRejection(
                reason="critic-reject",
                parent_id="coding/parent-skill",
                message="Cargo-cult mutation.",
            )
        ]
    return make_run_record(
        seed=seed,
        archive_size_pre=10,
        archive_size_post=11 if accepted else 10,
        accepted_children=children,
        rejections=rejections,
    )


# ---------------------------------------------------------------------------
# 1. append + read round-trip
# ---------------------------------------------------------------------------

class TestAppendReadRoundTrip:
    def test_single_record_round_trip(self, tmp_path):
        runs_file = tmp_path / "runs.jsonl"
        record = _sample_record(seed=1, accepted=True)

        append_run_record(runs_file, record)
        loaded = read_run_records(runs_file)

        assert len(loaded) == 1
        r = loaded[0]
        assert r.schema_version == 1
        assert r.seed == 1
        assert r.archive_size_pre == 10
        assert r.archive_size_post == 11
        assert len(r.accepted_children) == 1
        assert r.accepted_children[0].id == "coding/test-skill"
        assert r.accepted_children[0].score == pytest.approx(0.72)
        assert r.accepted_children[0].generation_index == 5
        assert r.accepted_children[0].lineage_depth == 2
        assert len(r.rejections) == 0

    def test_multiple_records_round_trip(self, tmp_path):
        runs_file = tmp_path / "runs.jsonl"
        records = [_sample_record(seed=i, accepted=(i % 2 == 0)) for i in range(3)]

        for rec in records:
            append_run_record(runs_file, rec)

        loaded = read_run_records(runs_file)
        assert len(loaded) == 3
        assert loaded[0].seed == 0
        assert loaded[1].seed == 1
        assert loaded[2].seed == 2

    def test_rejection_record_round_trip(self, tmp_path):
        runs_file = tmp_path / "runs.jsonl"
        record = _sample_record(seed=99, accepted=False)

        append_run_record(runs_file, record)
        loaded = read_run_records(runs_file)

        assert len(loaded) == 1
        r = loaded[0]
        assert len(r.rejections) == 1
        rej = r.rejections[0]
        assert rej.reason == "critic-reject"
        assert rej.parent_id == "coding/parent-skill"
        assert "Cargo-cult" in rej.message

    def test_rejection_message_truncated_to_200(self, tmp_path):
        runs_file = tmp_path / "runs.jsonl"
        long_message = "x" * 500
        record = make_run_record(
            seed=7,
            archive_size_pre=5,
            archive_size_post=5,
            rejections=[RunRecordRejection(
                reason="critic-error",
                parent_id="coding/skill",
                message=long_message,
            )],
        )

        append_run_record(runs_file, record)
        loaded = read_run_records(runs_file)

        assert len(loaded[0].rejections[0].message) <= 200


# ---------------------------------------------------------------------------
# 2. schema_version filter
# ---------------------------------------------------------------------------

class TestSchemaVersionFilter:
    def test_wrong_schema_version_skipped(self, tmp_path):
        runs_file = tmp_path / "runs.jsonl"

        # Write a valid record first
        valid = _sample_record(seed=1)
        append_run_record(runs_file, valid)

        # Manually append a record with wrong schema_version
        bad = {
            "schema_version": 99,
            "recorded_at": "2024-01-01T00:00:00+00:00",
            "seed": 2,
            "archive_size_pre": 5,
            "archive_size_post": 5,
            "accepted_children": [],
            "rejections": [],
            "bias": {"inspected_generations": 0, "signal_count": 0, "severities": {}},
        }
        with open(runs_file, "a") as f:
            f.write(json.dumps(bad) + "\n")

        loaded = read_run_records(runs_file)
        # Only the valid schema_version=1 record should be returned
        assert len(loaded) == 1
        assert loaded[0].seed == 1


# ---------------------------------------------------------------------------
# 3. malformed-line skip
# ---------------------------------------------------------------------------

class TestMalformedLineSkip:
    def test_malformed_json_line_skipped(self, tmp_path):
        runs_file = tmp_path / "runs.jsonl"

        # Write valid record
        valid = _sample_record(seed=10)
        append_run_record(runs_file, valid)

        # Inject a malformed line in the middle
        with open(runs_file, "a") as f:
            f.write("this is not json at all\n")

        # Write another valid record
        valid2 = _sample_record(seed=11)
        append_run_record(runs_file, valid2)

        loaded = read_run_records(runs_file)
        assert len(loaded) == 2
        assert loaded[0].seed == 10
        assert loaded[1].seed == 11

    def test_empty_lines_skipped(self, tmp_path):
        runs_file = tmp_path / "runs.jsonl"
        valid = _sample_record(seed=5)
        append_run_record(runs_file, valid)

        # Inject blank lines
        with open(runs_file, "a") as f:
            f.write("\n\n\n")

        loaded = read_run_records(runs_file)
        assert len(loaded) == 1

    def test_missing_file_returns_empty(self, tmp_path):
        missing = tmp_path / "nonexistent" / "runs.jsonl"
        loaded = read_run_records(missing)
        assert loaded == []


# ---------------------------------------------------------------------------
# 4. mkdir parents on missing dir
# ---------------------------------------------------------------------------

class TestMkdirParents:
    def test_missing_parent_dirs_created(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "runs.jsonl"
        assert not deep_path.parent.exists()

        record = _sample_record(seed=42)
        append_run_record(deep_path, record)

        assert deep_path.exists()
        loaded = read_run_records(deep_path)
        assert len(loaded) == 1

    def test_existing_dir_does_not_fail(self, tmp_path):
        runs_file = tmp_path / "runs.jsonl"
        record = _sample_record(seed=1)

        # Call twice — second call should not fail on existing dir
        append_run_record(runs_file, record)
        append_run_record(runs_file, _sample_record(seed=2))

        loaded = read_run_records(runs_file)
        assert len(loaded) == 2


# ---------------------------------------------------------------------------
# 5. summarize_bias_report placeholder
# ---------------------------------------------------------------------------

class TestSummarizeBiasReportPlaceholder:
    def test_returns_zero_counts(self):
        summary = summarize_bias_report()
        assert isinstance(summary, RunRecordBiasSummary)
        assert summary.inspected_generations == 0
        assert summary.signal_count == 0

    def test_severities_are_zero(self):
        summary = summarize_bias_report()
        assert summary.severities["info"] == 0
        assert summary.severities["concern"] == 0
        assert summary.severities["block"] == 0

    def test_accepts_arbitrary_args_gracefully(self):
        """Placeholder must accept any args for forward-compat with W5 API."""
        summary = summarize_bias_report("anything", key="value")
        assert isinstance(summary, RunRecordBiasSummary)

    def test_bias_persists_in_run_record(self, tmp_path):
        runs_file = tmp_path / "runs.jsonl"
        record = make_run_record(
            seed=1,
            archive_size_pre=3,
            archive_size_post=3,
            bias=RunRecordBiasSummary(
                inspected_generations=10,
                signal_count=2,
                severities={"info": 1, "concern": 1, "block": 0},
            ),
        )
        append_run_record(runs_file, record)
        loaded = read_run_records(runs_file)

        b = loaded[0].bias
        assert b.inspected_generations == 10
        assert b.signal_count == 2
        assert b.severities["concern"] == 1
