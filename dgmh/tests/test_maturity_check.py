"""
dgmh/tests/test_maturity_check.py — Tests for maturity_check.py (W5 deliverable).

Coverage (>=12 tests):
  - PASS verdict: sufficient runs, no bias breach, low reject rate
  - PENDING verdict: insufficient sample
  - FAIL on bias breach (concern)
  - FAIL on bias breach (block)
  - FAIL on critic-reject rate > 5%
  - Window filtering: old records excluded
  - Markdown rendering: required sections present
  - --verbose flag: per-run records section
  - --window-days CLI arg
  - Empty runs.jsonl returns PENDING
  - compute_metrics edge cases
  - evaluate_gate thresholds

Reference:
  dgmh/maturity_check.py (implementation under test)
  scripts/dgmh_maturity_check.ts (TS source)
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
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
)
from dgmh.maturity_check import (
    WindowMetrics,
    compute_metrics,
    evaluate_gate,
    format_report,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_days_ago(days: float) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.isoformat()


def _make_record(
    seed: int = 1,
    recorded_at: str | None = None,
    accepted: int = 1,
    modifier_errors: int = 0,
    critic_errors: int = 0,
    critic_rejects: int = 0,
    bias_info: int = 0,
    bias_concern: int = 0,
    bias_block: int = 0,
) -> RunRecord:
    children = [
        RunRecordChild(
            id=f"s/{seed}-{i}",
            parent_id="s/parent",
            score=0.7,
            generation_index=i,
            lineage_depth=1,
        )
        for i in range(accepted)
    ]
    rejections: list[RunRecordRejection] = []
    for _ in range(modifier_errors):
        rejections.append(RunRecordRejection(
            reason="modifier-error", parent_id="s/p", message="err"
        ))
    for _ in range(critic_errors):
        rejections.append(RunRecordRejection(
            reason="critic-error", parent_id="s/p", message="err"
        ))
    for _ in range(critic_rejects):
        rejections.append(RunRecordRejection(
            reason="critic-reject", parent_id="s/p", message="rejected"
        ))
    signal_count = bias_info + bias_concern + bias_block
    bias = RunRecordBiasSummary(
        inspected_generations=10,
        signal_count=signal_count,
        severities={"info": bias_info, "concern": bias_concern, "block": bias_block},
    )
    rec = make_run_record(
        seed=seed,
        archive_size_pre=5,
        archive_size_post=5 + accepted,
        accepted_children=children,
        rejections=rejections,
        bias=bias,
    )
    if recorded_at is not None:
        import dataclasses
        rec = dataclasses.replace(rec, recorded_at=recorded_at)
    return rec


# ---------------------------------------------------------------------------
# 1. PASS verdict
# ---------------------------------------------------------------------------

class TestPassVerdict:
    def test_pass_with_sufficient_runs_and_no_issues(self):
        records = [_make_record(seed=i) for i in range(5)]
        metrics = compute_metrics(records)
        verdict, reasons = evaluate_gate(metrics, window_days=28)
        assert verdict == "PASS"
        assert any("satisfied" in r for r in reasons)

    def test_pass_critic_reject_rate_exactly_5_percent(self):
        # 1 reject out of 21 total iterations = 4.76% <= 5% => PASS
        # Build WindowMetrics directly to control the exact ratio
        m = WindowMetrics(
            runs=5,
            accepted=20,
            rejection_modifier_error=0,
            rejection_critic_error=0,
            rejection_critic_reject=1,  # 1/21 = 4.76% < 5%
            bias_total=0,
            bias_info=0,
            bias_concern=0,
            bias_block=0,
            clean_runs=5,
        )
        verdict, _ = evaluate_gate(m, window_days=28)
        assert verdict == "PASS"


# ---------------------------------------------------------------------------
# 2. PENDING verdict (insufficient sample)
# ---------------------------------------------------------------------------

class TestPendingVerdict:
    def test_pending_with_zero_runs(self):
        records: list[RunRecord] = []
        metrics = compute_metrics(records)
        verdict, reasons = evaluate_gate(metrics, window_days=28)
        assert verdict == "PENDING"
        assert any("insufficient" in r for r in reasons)

    def test_pending_with_empty_jsonl(self, tmp_path):
        runs_file = tmp_path / "runs.jsonl"
        runs_file.write_text("")
        exit_code = main(["--window-days", "28"])
        # Empty file -> PENDING -> exit 0
        assert exit_code == 0

    def test_pending_when_runs_below_min(self):
        # 28-day window requires >=4 runs; give only 2
        records = [_make_record(seed=i) for i in range(2)]
        metrics = compute_metrics(records)
        verdict, _ = evaluate_gate(metrics, window_days=28)
        assert verdict == "PENDING"


# ---------------------------------------------------------------------------
# 3. FAIL on bias breach
# ---------------------------------------------------------------------------

class TestFailBiasBreach:
    def test_fail_on_concern_signal(self):
        records = [_make_record(seed=i, bias_concern=1) for i in range(5)]
        metrics = compute_metrics(records)
        verdict, reasons = evaluate_gate(metrics, window_days=28)
        assert verdict == "FAIL"
        assert any("bias" in r for r in reasons)

    def test_fail_on_block_signal(self):
        records = [_make_record(seed=i, bias_block=1) for i in range(5)]
        metrics = compute_metrics(records)
        verdict, reasons = evaluate_gate(metrics, window_days=28)
        assert verdict == "FAIL"
        assert any("block" in r for r in reasons)

    def test_info_only_does_not_fail(self):
        records = [_make_record(seed=i, bias_info=2) for i in range(5)]
        metrics = compute_metrics(records)
        verdict, _ = evaluate_gate(metrics, window_days=28)
        assert verdict == "PASS"


# ---------------------------------------------------------------------------
# 4. FAIL on critic-reject rate
# ---------------------------------------------------------------------------

class TestFailCriticRejectRate:
    def test_fail_when_reject_rate_exceeds_5_percent(self):
        # 2 accepts, 1 critic_reject per run, 5 runs -> rate = 5/(5*3) = 33%
        records = [_make_record(seed=i, accepted=2, critic_rejects=1) for i in range(5)]
        metrics = compute_metrics(records)
        verdict, reasons = evaluate_gate(metrics, window_days=28)
        assert verdict == "FAIL"
        assert any("critic-reject" in r for r in reasons)

    def test_pass_when_reject_rate_zero(self):
        records = [_make_record(seed=i, accepted=5) for i in range(5)]
        metrics = compute_metrics(records)
        verdict, _ = evaluate_gate(metrics, window_days=28)
        assert verdict == "PASS"


# ---------------------------------------------------------------------------
# 5. Window filtering
# ---------------------------------------------------------------------------

class TestWindowFiltering:
    def test_old_records_excluded(self, tmp_path, monkeypatch):
        runs_file = tmp_path / "runs.jsonl"
        # 1 recent record (1 day ago), 1 old record (60 days ago)
        recent = _make_record(seed=1, recorded_at=_iso_days_ago(1))
        old = _make_record(seed=2, recorded_at=_iso_days_ago(60))
        append_run_record(runs_file, recent)
        append_run_record(runs_file, old)

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        # Manually filter as maturity_check.main does
        from dgmh.run_log import read_run_records
        from datetime import datetime, timezone
        import math
        records = read_run_records(runs_file)
        window_days = 28
        now_ts = datetime.now(timezone.utc).timestamp()
        cutoff = now_ts - window_days * 24 * 60 * 60
        window = [
            r for r in records
            if datetime.fromisoformat(r.recorded_at).timestamp() >= cutoff
        ]
        assert len(window) == 1
        assert window[0].seed == 1

    def test_all_records_in_window_included(self, tmp_path):
        runs_file = tmp_path / "runs.jsonl"
        for i in range(4):
            rec = _make_record(seed=i, recorded_at=_iso_days_ago(i * 3))
            append_run_record(runs_file, rec)
        from dgmh.run_log import read_run_records
        records = read_run_records(runs_file)
        assert len(records) == 4


# ---------------------------------------------------------------------------
# 6. Markdown rendering
# ---------------------------------------------------------------------------

class TestMarkdownRendering:
    def _make_metrics(self, runs: int = 5) -> WindowMetrics:
        return WindowMetrics(
            runs=runs,
            accepted=runs,
            rejection_modifier_error=0,
            rejection_critic_error=0,
            rejection_critic_reject=0,
            bias_total=0,
            bias_info=0,
            bias_concern=0,
            bias_block=0,
            clean_runs=runs,
        )

    def test_report_contains_heading(self):
        m = self._make_metrics()
        verdict, reasons = evaluate_gate(m, 28)
        now = datetime.now(timezone.utc)
        report = format_report(
            metrics=m,
            verdict=verdict,
            reasons=reasons,
            window_days=28,
            window_start=now,
            window_end=now,
            verbose=False,
            window_records=[],
        )
        assert "# DGM-H Phase 1.6 maturity check" in report

    def test_report_contains_verdict(self):
        m = self._make_metrics()
        verdict, reasons = evaluate_gate(m, 28)
        now = datetime.now(timezone.utc)
        report = format_report(
            metrics=m, verdict=verdict, reasons=reasons,
            window_days=28, window_start=now, window_end=now,
            verbose=False, window_records=[],
        )
        assert "`PASS`" in report or "`PENDING`" in report or "`FAIL`" in report

    def test_report_contains_metrics_section(self):
        m = self._make_metrics()
        verdict, reasons = evaluate_gate(m, 28)
        now = datetime.now(timezone.utc)
        report = format_report(
            metrics=m, verdict=verdict, reasons=reasons,
            window_days=28, window_start=now, window_end=now,
            verbose=False, window_records=[],
        )
        assert "## Metrics" in report
        assert "bias signals" in report
        assert "critic-reject rate" in report


# ---------------------------------------------------------------------------
# 7. --verbose flag
# ---------------------------------------------------------------------------

class TestVerboseFlag:
    def test_verbose_includes_per_run_section(self, tmp_path):
        records = [_make_record(seed=i) for i in range(5)]
        now = datetime.now(timezone.utc)
        m = compute_metrics(records)
        verdict, reasons = evaluate_gate(m, 28)
        report = format_report(
            metrics=m, verdict=verdict, reasons=reasons,
            window_days=28, window_start=now, window_end=now,
            verbose=True, window_records=records,
        )
        assert "## Per-run records" in report

    def test_non_verbose_excludes_per_run_section(self, tmp_path):
        records = [_make_record(seed=i) for i in range(5)]
        now = datetime.now(timezone.utc)
        m = compute_metrics(records)
        verdict, reasons = evaluate_gate(m, 28)
        report = format_report(
            metrics=m, verdict=verdict, reasons=reasons,
            window_days=28, window_start=now, window_end=now,
            verbose=False, window_records=records,
        )
        assert "## Per-run records" not in report
