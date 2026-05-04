"""
dgmh/tests/test_bias_detector.py — Tests for bias_detector.py (W5 deliverable).

Coverage (>=18 tests):
  - failure-run: fires under controlled fixture, does not fire on healthy archive
  - failure-run: severity concern vs block based on run length
  - score-plateau: fires / does not fire
  - lineage-convergence: fires / does not fire
  - prompt-bloat (new): fires / does not fire / score-gain suppresses
  - severity classification
  - envelope-shape output (Codex BiasReport JSON)
  - make_heuristic_skill_bias_detector factory + custom opts
  - empty archive
  - archive smaller than window

Reference:
  dgmh/bias_detector.py (implementation under test)
  playground/sandbox/biasDetector.ts (original TS source)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dgmh.bias_detector import (
    BiasReport,
    BiasSignal,
    HeuristicBiasDetectorOpts,
    _Generation,
    _coerce_generation,
    _detect_failure_run,
    _detect_lineage_convergence,
    _detect_plateau,
    _detect_prompt_bloat,
    make_heuristic_skill_bias_detector,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_gen(
    idx: int,
    score: float,
    lineage: list[str] | None = None,
    skill_md: str = "",
) -> _Generation:
    return _Generation(
        generation_index=idx,
        score=score,
        lineage=lineage or [],
        skill_md=skill_md,
    )


def _healthy_archive(n: int = 10) -> list[_Generation]:
    """Archive with steadily improving scores and diverse lineage."""
    return [
        _make_gen(i, score=0.5 + i * 0.04, lineage=[f"anc-{i % 3}"])
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 1. failure-run signal
# ---------------------------------------------------------------------------

class TestFailureRun:
    def test_fires_on_consecutive_low_scores(self):
        gens = [
            _make_gen(0, 0.8),
            _make_gen(1, 0.1),
            _make_gen(2, 0.1),
            _make_gen(3, 0.1),
        ]
        sig = _detect_failure_run(gens, threshold=0.3, min_window=3)
        assert sig is not None
        assert sig.id == "failure-run"
        assert sig.severity in ("concern", "block")
        assert 1 in sig.evidence_generations
        assert 3 in sig.evidence_generations

    def test_does_not_fire_below_min_window(self):
        gens = [
            _make_gen(0, 0.1),
            _make_gen(1, 0.1),
        ]
        # run of 2, min_window=3 => should NOT fire
        sig = _detect_failure_run(gens, threshold=0.3, min_window=3)
        assert sig is None

    def test_does_not_fire_on_healthy_archive(self):
        gens = _healthy_archive(10)
        sig = _detect_failure_run(gens, threshold=0.3, min_window=3)
        assert sig is None

    def test_severity_concern_at_exactly_min_window(self):
        gens = [_make_gen(i, 0.1) for i in range(3)]
        sig = _detect_failure_run(gens, threshold=0.3, min_window=3)
        assert sig is not None
        assert sig.severity == "concern"

    def test_severity_block_at_double_min_window(self):
        # run_len = 6 >= 2 * min_window(3) => block
        gens = [_make_gen(i, 0.1) for i in range(6)]
        sig = _detect_failure_run(gens, threshold=0.3, min_window=3)
        assert sig is not None
        assert sig.severity == "block"

    def test_evidence_generations_are_correct_indices(self):
        gens = [
            _make_gen(10, 0.8),
            _make_gen(11, 0.2),
            _make_gen(12, 0.2),
            _make_gen(13, 0.2),
        ]
        sig = _detect_failure_run(gens, threshold=0.3, min_window=3)
        assert sig is not None
        assert sig.evidence_generations == [11, 12, 13]


# ---------------------------------------------------------------------------
# 2. score-plateau signal
# ---------------------------------------------------------------------------

class TestScorePlateau:
    def test_fires_when_range_below_epsilon(self):
        # All scores within 0.02 range
        gens = [_make_gen(i, 0.6 + i * 0.005) for i in range(5)]
        sig = _detect_plateau(gens, window=5, epsilon=0.05)
        assert sig is not None
        assert sig.id == "score-plateau"
        assert sig.severity == "concern"

    def test_does_not_fire_when_range_above_epsilon(self):
        gens = [_make_gen(i, 0.4 + i * 0.05) for i in range(5)]
        sig = _detect_plateau(gens, window=5, epsilon=0.05)
        assert sig is None

    def test_does_not_fire_below_window_size(self):
        gens = [_make_gen(i, 0.5) for i in range(4)]
        sig = _detect_plateau(gens, window=5, epsilon=0.05)
        assert sig is None

    def test_evidence_contains_all_window_generations(self):
        gens = [_make_gen(i, 0.5) for i in range(5)]
        sig = _detect_plateau(gens, window=5, epsilon=0.05)
        assert sig is not None
        assert sig.evidence_generations == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# 3. lineage-convergence signal
# ---------------------------------------------------------------------------

class TestLineageConvergence:
    def test_fires_when_ancestor_dominates(self):
        # 4 of 5 gens share ancestor "root-0"
        gens = [
            _make_gen(0, 0.5, lineage=["root-0"]),
            _make_gen(1, 0.5, lineage=["root-0"]),
            _make_gen(2, 0.5, lineage=["root-0"]),
            _make_gen(3, 0.5, lineage=["root-0"]),
            _make_gen(4, 0.5, lineage=["root-1"]),
        ]
        sig = _detect_lineage_convergence(gens, threshold=0.7, window=5)
        assert sig is not None
        assert sig.id == "lineage-convergence"
        assert sig.severity == "info"
        assert "root-0" in sig.description

    def test_does_not_fire_with_diverse_lineage(self):
        gens = [_make_gen(i, 0.5, lineage=[f"root-{i}"]) for i in range(5)]
        sig = _detect_lineage_convergence(gens, threshold=0.7, window=5)
        assert sig is None

    def test_does_not_fire_below_window(self):
        gens = [_make_gen(i, 0.5, lineage=["root-0"]) for i in range(4)]
        sig = _detect_lineage_convergence(gens, threshold=0.7, window=5)
        assert sig is None

    def test_evidence_contains_matching_generations(self):
        gens = [
            _make_gen(10, 0.5, lineage=["A"]),
            _make_gen(11, 0.5, lineage=["A"]),
            _make_gen(12, 0.5, lineage=["A"]),
            _make_gen(13, 0.5, lineage=["A"]),
            _make_gen(14, 0.5, lineage=["B"]),
        ]
        sig = _detect_lineage_convergence(gens, threshold=0.7, window=5)
        assert sig is not None
        assert 14 not in sig.evidence_generations
        assert all(idx in [10, 11, 12, 13] for idx in sig.evidence_generations)


# ---------------------------------------------------------------------------
# 4. prompt-bloat signal (new per plan v2 §CONCERN 5)
# ---------------------------------------------------------------------------

class TestPromptBloat:
    def test_fires_on_monotonic_growth_without_score_gain(self):
        # chars grow monotonically, score flat
        gens = [
            _make_gen(i, 0.5, skill_md="A" * (100 + i * 20))
            for i in range(5)
        ]
        # first=100, last=180 -> 80% growth >= 50% threshold; score flat
        sig = _detect_prompt_bloat(gens, window=5, growth_rate=0.5)
        assert sig is not None
        assert sig.id == "prompt-bloat"
        assert sig.severity == "concern"

    def test_does_not_fire_when_score_improves(self):
        gens = [
            _make_gen(i, score=0.5 + i * 0.1, skill_md="A" * (100 + i * 50))
            for i in range(5)
        ]
        # score improves => bloat may be justified
        sig = _detect_prompt_bloat(gens, window=5, growth_rate=0.5)
        assert sig is None

    def test_does_not_fire_on_non_monotonic_growth(self):
        gens = [
            _make_gen(0, 0.5, skill_md="A" * 100),
            _make_gen(1, 0.5, skill_md="A" * 200),
            _make_gen(2, 0.5, skill_md="A" * 150),  # shrinks
            _make_gen(3, 0.5, skill_md="A" * 300),
            _make_gen(4, 0.5, skill_md="A" * 400),
        ]
        sig = _detect_prompt_bloat(gens, window=5, growth_rate=0.5)
        assert sig is None

    def test_does_not_fire_below_growth_threshold(self):
        gens = [
            _make_gen(i, 0.5, skill_md="A" * (100 + i * 5))
            for i in range(5)
        ]
        # first=100, last=120 -> 20% growth < 50% threshold
        sig = _detect_prompt_bloat(gens, window=5, growth_rate=0.5)
        assert sig is None

    def test_does_not_fire_below_window_size(self):
        gens = [
            _make_gen(i, 0.5, skill_md="A" * (100 + i * 50))
            for i in range(4)
        ]
        sig = _detect_prompt_bloat(gens, window=5, growth_rate=0.5)
        assert sig is None

    def test_evidence_contains_all_window_generations(self):
        gens = [
            _make_gen(i, 0.5, skill_md="A" * (100 + i * 30))
            for i in range(5)
        ]
        sig = _detect_prompt_bloat(gens, window=5, growth_rate=0.5)
        assert sig is not None
        assert sig.evidence_generations == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# 5. Severity classification
# ---------------------------------------------------------------------------

class TestSeverityClassification:
    def test_failure_run_concern_severity(self):
        gens = [_make_gen(i, 0.1) for i in range(3)]
        sig = _detect_failure_run(gens, threshold=0.3, min_window=3)
        assert sig is not None
        assert sig.severity == "concern"

    def test_failure_run_block_severity(self):
        gens = [_make_gen(i, 0.1) for i in range(6)]
        sig = _detect_failure_run(gens, threshold=0.3, min_window=3)
        assert sig is not None
        assert sig.severity == "block"

    def test_plateau_severity_is_concern(self):
        gens = [_make_gen(i, 0.5) for i in range(5)]
        sig = _detect_plateau(gens, window=5, epsilon=0.05)
        assert sig is not None
        assert sig.severity == "concern"

    def test_lineage_convergence_severity_is_info(self):
        gens = [_make_gen(i, 0.5, lineage=["root-0"]) for i in range(5)]
        sig = _detect_lineage_convergence(gens, threshold=0.7, window=5)
        assert sig is not None
        assert sig.severity == "info"


# ---------------------------------------------------------------------------
# 6. Envelope-shape output
# ---------------------------------------------------------------------------

class TestEnvelopeShape:
    def test_envelope_json_is_valid_json(self):
        detector = make_heuristic_skill_bias_detector()
        gens = [_make_gen(i, 0.1) for i in range(6)]
        report = detector.detect(gens)
        parsed = json.loads(report.envelope_json)
        assert "signals" in parsed

    def test_envelope_has_correct_signal_fields(self):
        detector = make_heuristic_skill_bias_detector()
        gens = [_make_gen(i, 0.1) for i in range(6)]
        report = detector.detect(gens)
        parsed = json.loads(report.envelope_json)
        assert len(parsed["signals"]) >= 1
        sig = parsed["signals"][0]
        assert "id" in sig
        assert "description" in sig
        assert "severity" in sig
        assert "evidenceGenerations" in sig

    def test_str_returns_envelope_json(self):
        detector = make_heuristic_skill_bias_detector()
        gens = _healthy_archive(5)
        report = detector.detect(gens)
        assert str(report) == report.envelope_json

    def test_empty_archive_returns_empty_signals(self):
        detector = make_heuristic_skill_bias_detector()
        report = detector.detect([])
        parsed = json.loads(report.envelope_json)
        assert parsed["signals"] == []
        assert report.inspected_generations == 0

    def test_archive_smaller_than_window_partial_result(self):
        # archive of 3, window=5 -> plateau/convergence won't fire; failure-run might
        detector = make_heuristic_skill_bias_detector()
        gens = [_make_gen(i, 0.8) for i in range(3)]
        report = detector.detect(gens)
        assert isinstance(report, BiasReport)
        # No exceptions
        assert report.inspected_generations == 3

    def test_severity_counts_method(self):
        detector = make_heuristic_skill_bias_detector()
        # Trigger failure-run (block) + plateau (concern)
        gens = [_make_gen(i, 0.1) for i in range(6)]
        report = detector.detect(gens)
        counts = report.severity_counts()
        assert "info" in counts
        assert "concern" in counts
        assert "block" in counts


# ---------------------------------------------------------------------------
# 7. Factory and custom opts
# ---------------------------------------------------------------------------

class TestFactory:
    def test_default_opts_fire_on_low_scores(self):
        detector = make_heuristic_skill_bias_detector()
        # Default failure_run_threshold=0.3, window=3
        gens = [_make_gen(i, 0.1) for i in range(3)]
        report = detector.detect(gens)
        ids = [s.id for s in report.signals]
        assert "failure-run" in ids

    def test_custom_opts_respected(self):
        opts = HeuristicBiasDetectorOpts(
            failure_run_threshold=0.9,  # very high; almost everything fires
            failure_run_window=2,
        )
        detector = make_heuristic_skill_bias_detector(opts)
        gens = [_make_gen(i, 0.5) for i in range(5)]
        report = detector.detect(gens)
        # 0.5 < 0.9, run of 5 >= window of 2 => failure-run fires
        ids = [s.id for s in report.signals]
        assert "failure-run" in ids

    def test_context_note_does_not_change_signals(self):
        detector = make_heuristic_skill_bias_detector()
        gens = _healthy_archive(5)
        r1 = detector.detect(gens, context_note=None)
        r2 = detector.detect(gens, context_note="test-context")
        assert len(r1.signals) == len(r2.signals)
