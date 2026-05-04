"""
dgmh/maturity_check.py — Phase 1.6 maturity-gate metrics (W5 deliverable).

Python port of scripts/dgmh_maturity_check.ts.

Reads ~/.hermes/dgmh/runs.jsonl via run_log.read_run_records(), filters to a
rolling window (default 28 days), computes gate metrics, and emits a markdown
report to stdout.

Verdict thresholds (mirrors dgmh_maturity_check.ts::evaluateGate):
  PENDING: runs < max(1, floor(window_days / 7))  — insufficient sample
  FAIL:    bias.concern > 0 OR bias.block > 0      — bias breach
  FAIL:    critic_reject_rate > 0.05               — too many rejections
  PASS:    all thresholds satisfied

Exit codes:
  0 — PASS or PENDING
  1 — FAIL

CLI:
  python -m dgmh.maturity_check [--window-days N] [--verbose]

Reference:
  scripts/dgmh_maturity_check.ts (TS source)
  dgmh/run_log.py               (read_run_records)
  hermes-skill-archive-dgmh-plan-v2-addendum.md §Updated runtime acceptance metrics §7
"""

from __future__ import annotations

import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_HERMES_ROOT = Path(__file__).parent.parent
if str(_HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(_HERMES_ROOT))

from dgmh.run_log import RunRecord, read_run_records

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SECONDS_PER_DAY = 24 * 60 * 60
_DEFAULT_WINDOW_DAYS = 28


def _default_runs_path() -> Path:
    hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(hermes_home) / "dgmh" / "runs.jsonl"


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

class WindowMetrics:
    """Computed metrics over a filtered window of RunRecords."""

    def __init__(
        self,
        runs: int,
        accepted: int,
        rejection_modifier_error: int,
        rejection_critic_error: int,
        rejection_critic_reject: int,
        bias_total: int,
        bias_info: int,
        bias_concern: int,
        bias_block: int,
        clean_runs: int,
    ) -> None:
        self.runs = runs
        self.accepted = accepted
        self.rejection_modifier_error = rejection_modifier_error
        self.rejection_critic_error = rejection_critic_error
        self.rejection_critic_reject = rejection_critic_reject
        self.bias_total = bias_total
        self.bias_info = bias_info
        self.bias_concern = bias_concern
        self.bias_block = bias_block
        self.clean_runs = clean_runs

        total_iterations = accepted + rejection_modifier_error + rejection_critic_error + rejection_critic_reject
        self.critic_reject_rate = (
            rejection_critic_reject / total_iterations if total_iterations > 0 else 0.0
        )
        self.clean_run_rate = clean_runs / runs if runs > 0 else 0.0


def compute_metrics(records: list[RunRecord]) -> WindowMetrics:
    """Compute gate metrics from a filtered list of RunRecords.

    Mirrors dgmh_maturity_check.ts::computeMetrics.
    """
    accepted = 0
    rej_modifier = 0
    rej_critic_error = 0
    rej_critic_reject = 0
    bias_total = 0
    bias_info = 0
    bias_concern = 0
    bias_block = 0
    clean_runs = 0

    for r in records:
        accepted += len(r.accepted_children)
        for x in r.rejections:
            if x.reason == "modifier-error":
                rej_modifier += 1
            elif x.reason == "critic-error":
                rej_critic_error += 1
            elif x.reason == "critic-reject":
                rej_critic_reject += 1

        bias_total += r.bias.signal_count
        bias_info += r.bias.severities.get("info", 0)
        bias_concern += r.bias.severities.get("concern", 0)
        bias_block += r.bias.severities.get("block", 0)

        if r.bias.signal_count == 0:
            clean_runs += 1

    return WindowMetrics(
        runs=len(records),
        accepted=accepted,
        rejection_modifier_error=rej_modifier,
        rejection_critic_error=rej_critic_error,
        rejection_critic_reject=rej_critic_reject,
        bias_total=bias_total,
        bias_info=bias_info,
        bias_concern=bias_concern,
        bias_block=bias_block,
        clean_runs=clean_runs,
    )


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------

GateVerdict = Literal["PASS", "PENDING", "FAIL"]


def evaluate_gate(
    metrics: WindowMetrics,
    window_days: int,
) -> tuple[GateVerdict, list[str]]:
    """Evaluate gate thresholds and return (verdict, reasons).

    Mirrors dgmh_maturity_check.ts::evaluateGate.

    Thresholds (same as TS):
      1. sample_size: >= window_days/7 runs (else PENDING)
      2. bias: zero concern + zero block severities (FAIL if breached)
      3. critic-reject rate: <= 5% (FAIL if breached)
    """
    min_runs = max(1, math.floor(window_days / 7))
    if metrics.runs < min_runs:
        reason = (
            f"insufficient sample: {metrics.runs} runs < {min_runs} "
            f"expected over {window_days} days"
        )
        return "PENDING", [reason]

    if metrics.bias_concern > 0 or metrics.bias_block > 0:
        reason = (
            f"bias signals breached: {metrics.bias_concern} concern, "
            f"{metrics.bias_block} block"
        )
        return "FAIL", [reason]

    if metrics.critic_reject_rate > 0.05:
        reason = (
            f"critic-reject rate {metrics.critic_reject_rate * 100:.2f}% > 5% threshold"
        )
        return "FAIL", [reason]

    return "PASS", ["all thresholds satisfied"]


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(
    metrics: WindowMetrics,
    verdict: GateVerdict,
    reasons: list[str],
    window_days: int,
    window_start: datetime,
    window_end: datetime,
    verbose: bool,
    window_records: list[RunRecord],
) -> str:
    """Format a markdown maturity report.

    Mirrors dgmh_maturity_check.ts::formatReport.
    """
    lines = [
        "# DGM-H Phase 1.6 maturity check",
        "",
        (
            f"**Window**: {window_days} days "
            f"({window_start.isoformat()} -> {window_end.isoformat()})"
        ),
        f"**Verdict**: `{verdict}`",
        "",
        "## Reasons",
        *[f"- {r}" for r in reasons],
        "",
        "## Metrics",
        f"- runs: {metrics.runs}",
        f"- accepted children: {metrics.accepted}",
        (
            f"- iteration rejections: "
            f"modifier-error={metrics.rejection_modifier_error} "
            f"critic-error={metrics.rejection_critic_error} "
            f"critic-reject={metrics.rejection_critic_reject}"
        ),
        (
            f"- bias signals: total={metrics.bias_total} "
            f"info={metrics.bias_info} "
            f"concern={metrics.bias_concern} "
            f"block={metrics.bias_block}"
        ),
        (
            f"- clean runs (0 signals): {metrics.clean_runs} / {metrics.runs} "
            f"({metrics.clean_run_rate * 100:.1f}%)"
        ),
        f"- critic-reject rate: {metrics.critic_reject_rate * 100:.2f}%",
    ]

    if verbose and window_records:
        lines.append("")
        lines.append("## Per-run records")
        for r in window_records:
            lines.append(
                f"- {r.recorded_at} seed={r.seed} "
                f"accepted={len(r.accepted_children)} "
                f"rejections={len(r.rejections)} "
                f"biasSignals={r.bias.signal_count}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> tuple[int, bool]:
    """Parse CLI arguments. Returns (window_days, verbose)."""
    window_days = _DEFAULT_WINDOW_DAYS
    verbose = False
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--window-days":
            i += 1
            if i >= len(argv):
                print("--window-days requires a value", file=sys.stderr)
                sys.exit(2)
            try:
                days = int(argv[i])
                if days <= 0:
                    raise ValueError("must be positive")
                window_days = days
            except ValueError as exc:
                print(f"--window-days value invalid: {argv[i]}: {exc}", file=sys.stderr)
                sys.exit(2)
        elif arg in ("--verbose", "-v"):
            verbose = True
        elif arg in ("--help", "-h"):
            print(
                "\n".join([
                    "dgmh/maturity_check.py — Phase 1.6 gate metrics",
                    "",
                    "Usage: python -m dgmh.maturity_check [--window-days N] [--verbose]",
                    "",
                    "  --window-days N    Rolling window in days (default 28).",
                    "  --verbose          Print per-run details inside the window.",
                ])
            )
            sys.exit(0)
        else:
            print(f"unknown arg: {arg}", file=sys.stderr)
            sys.exit(2)
        i += 1
    return window_days, verbose


def main(argv: list[str] | None = None) -> int:
    """Main entry point. Returns exit code (0=PASS/PENDING, 1=FAIL)."""
    if argv is None:
        argv = sys.argv[1:]

    window_days, verbose = _parse_args(list(argv))

    runs_path = _default_runs_path()
    records = read_run_records(runs_path)

    now_dt = datetime.now(timezone.utc)
    now_ts = now_dt.timestamp()
    cutoff_ts = now_ts - window_days * _SECONDS_PER_DAY

    window_records: list[RunRecord] = []
    for r in records:
        try:
            t = datetime.fromisoformat(r.recorded_at).timestamp()
        except (ValueError, OverflowError):
            continue
        if t >= cutoff_ts:
            window_records.append(r)

    metrics = compute_metrics(window_records)
    verdict, reasons = evaluate_gate(metrics, window_days)

    window_start = datetime.fromtimestamp(cutoff_ts, tz=timezone.utc)
    report = format_report(
        metrics=metrics,
        verdict=verdict,
        reasons=reasons,
        window_days=window_days,
        window_start=window_start,
        window_end=now_dt,
        verbose=verbose,
        window_records=window_records,
    )

    print(report)
    return 1 if verdict == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
