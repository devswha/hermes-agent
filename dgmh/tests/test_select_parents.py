"""
dgmh/tests/test_select_parents.py — Tests for select_parents.py (W2 deliverable).

Covers per W2 spec:
1. Math parity (ALGORITHM.md §8.2 formula correctness, 1e-9 tolerance)
2. Edge cases (empty archive, single entry, all-zero scores)
3. Determinism (same seed → same sequence, different seed → different)
4. Distributional sampling (KS D < 0.1 over 1000 trials)
5. Selection log (one call → one JSONL line; multiple calls → append)

Port note: tests exercise Python port of selectParents.ts against the same
formulas described in ALGORITHM.md §8.2. RNG distributional parity tested
(not bit-exact) per hermes-skill-archive-dgmh-plan-v2-addendum.md §CONCERN 6.

Reference: devswha/dgmh playground/dgmh-engine/selectParents.ts
           hermes-skill-archive-dgmh-plan-v2-addendum.md §CONCERN 6
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root importable
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dgmh.select_parents import (
    DEFAULT_HYPERPARAMS,
    DgmhHyperparams,
    Generation,
    archive_top_midpoint,
    combined_weight,
    novelty_weight,
    sample_parent,
    select_parents,
    sigmoid_weight,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_archive(scores: list[float], compiled_children: list[int] | None = None) -> list[Generation]:
    """Build a minimal archive for testing."""
    if compiled_children is None:
        compiled_children = [0] * len(scores)
    return [
        Generation(
            id=f"gen-{i}",
            score=scores[i],
            compiled_children=compiled_children[i],
            generation_index=i,
            selectable=True,
        )
        for i in range(len(scores))
    ]


# ---------------------------------------------------------------------------
# 1. Math parity — ALGORITHM.md §8.2 formula correctness
# ---------------------------------------------------------------------------

class TestSigmoidWeight:
    """Port note: selectParents.ts::sigmoidWeight lines 22-28."""

    def test_midpoint_returns_half(self):
        """sigmoid_weight(0.5, 0.5, 10) == 0.5 exactly (symmetric point)."""
        result = sigmoid_weight(0.5, 0.5, 10.0)
        assert abs(result - 0.5) < 1e-9, f"Expected 0.5, got {result}"

    def test_high_score_near_one(self):
        """sigmoid_weight(1.0, 0.5, 10) ≈ 0.9933 (exp(-5) ≈ 0.00674)."""
        expected = 1.0 / (1.0 + math.exp(-10.0 * (1.0 - 0.5)))
        result = sigmoid_weight(1.0, 0.5, 10.0)
        assert abs(result - expected) < 1e-9, f"Expected {expected}, got {result}"
        assert abs(result - 0.9933071490757153) < 1e-9

    def test_low_score_near_zero(self):
        """sigmoid_weight(0.0, 0.5, 10) ≈ 0.0067 (exp(5) large denom)."""
        expected = 1.0 / (1.0 + math.exp(-10.0 * (0.0 - 0.5)))
        result = sigmoid_weight(0.0, 0.5, 10.0)
        assert abs(result - expected) < 1e-9, f"Expected {expected}, got {result}"
        assert abs(result - 0.006692850924284856) < 1e-9

    def test_default_lambda(self):
        """sigmoid_weight uses lambda=10 by default (DEFAULT_HYPERPARAMS)."""
        result_default = sigmoid_weight(0.7, 0.5)
        result_explicit = sigmoid_weight(0.7, 0.5, 10.0)
        assert abs(result_default - result_explicit) < 1e-15

    def test_formula_matches_paper(self):
        """Verify formula 1/(1+exp(-lambda*(alpha_i - alpha_mid))) for several points."""
        for alpha_i, alpha_mid, lam in [
            (0.3, 0.6, 5.0),
            (0.8, 0.2, 20.0),
            (0.5, 0.5, 1.0),
            (0.0, 1.0, 10.0),
        ]:
            expected = 1.0 / (1.0 + math.exp(-lam * (alpha_i - alpha_mid)))
            result = sigmoid_weight(alpha_i, alpha_mid, lam)
            assert abs(result - expected) < 1e-12, (
                f"Mismatch for alpha_i={alpha_i}, alpha_mid={alpha_mid}, lam={lam}: "
                f"expected={expected}, got={result}"
            )


class TestNoveltyWeight:
    """Port note: selectParents.ts::noveltyWeight lines 30-32."""

    def test_zero_children_is_one(self):
        """novelty_weight(0) == 1.0 (no compiled children → max novelty)."""
        assert novelty_weight(0) == 1.0

    def test_one_child_is_half(self):
        """novelty_weight(1) == 0.5."""
        assert novelty_weight(1) == 0.5

    def test_nine_children(self):
        """novelty_weight(9) == 0.1."""
        assert abs(novelty_weight(9) - 0.1) < 1e-15

    def test_formula_is_1_over_1_plus_n(self):
        """Verify formula 1/(1+n_i) for several n values."""
        for n in [0, 1, 2, 5, 10, 100]:
            expected = 1.0 / (1.0 + n)
            result = novelty_weight(n)
            assert abs(result - expected) < 1e-15, f"n={n}: expected={expected}, got={result}"


class TestCombinedWeight:
    """Port note: selectParents.ts::combinedWeight lines 34-41."""

    def test_product_of_sigmoid_and_novelty(self):
        """combined_weight(s, h) == s * h."""
        for s, h in [(0.5, 1.0), (0.9933, 0.5), (0.0067, 0.1), (1.0, 1.0), (0.0, 0.0)]:
            assert abs(combined_weight(s, h) - s * h) < 1e-15

    def test_zero_sigmoid_gives_zero(self):
        assert combined_weight(0.0, 1.0) == 0.0

    def test_zero_novelty_gives_zero(self):
        assert combined_weight(1.0, 0.0) == 0.0


class TestArchiveTopMidpoint:
    """Port note: selectParents.ts::archiveTopMidpoint lines 43-52."""

    def test_empty_returns_zero(self):
        """Empty archive → 0.0 (matches TS: 'if (archive.generations.length === 0) return 0')."""
        assert archive_top_midpoint([]) == 0.0

    def test_top3_of_5(self):
        """5 entries, top_m=3: picks top 3 by score and averages."""
        archive = _make_archive([0.1, 0.9, 0.3, 0.7, 0.5])
        result = archive_top_midpoint(archive, top_m=3)
        # Top 3 scores: 0.9, 0.7, 0.5 → mean = 0.7
        assert abs(result - (0.9 + 0.7 + 0.5) / 3) < 1e-12

    def test_fewer_than_top_m_graceful(self):
        """2 entries, top_m=3: returns mean of available 2 (no crash)."""
        archive = _make_archive([0.6, 0.4])
        result = archive_top_midpoint(archive, top_m=3)
        assert abs(result - (0.6 + 0.4) / 2) < 1e-12

    def test_single_entry(self):
        """1 entry: alpha_mid = that entry's score."""
        archive = _make_archive([0.75])
        result = archive_top_midpoint(archive, top_m=3)
        assert abs(result - 0.75) < 1e-12

    def test_default_top_m(self):
        """Default top_m=3 from DEFAULT_HYPERPARAMS."""
        archive = _make_archive([0.2, 0.8, 0.6, 0.4, 0.9])
        result_default = archive_top_midpoint(archive)
        result_explicit = archive_top_midpoint(archive, top_m=3)
        assert abs(result_default - result_explicit) < 1e-15


# ---------------------------------------------------------------------------
# 2. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_archive_raises(self):
        """Empty archive → select_parents raises ValueError with clear message."""
        rng = random.Random(42)
        with pytest.raises(ValueError, match="empty"):
            select_parents([], 1, rng)

    def test_empty_archive_sample_raises(self):
        """sample_parent on empty archive raises ValueError."""
        rng = random.Random(42)
        with pytest.raises(ValueError, match="empty"):
            sample_parent([], rng)

    def test_single_entry_always_selected(self):
        """Single entry archive → that entry always selected regardless of RNG."""
        archive = _make_archive([0.5])
        for seed in [0, 1, 42, 999, 12345]:
            rng = random.Random(seed)
            result = select_parents(archive, 1, rng)
            assert len(result) == 1
            assert result[0].id == "gen-0"

    def test_all_zero_scores_no_crash(self):
        """All-zero scores → doesn't crash, returns valid entry from archive."""
        archive = _make_archive([0.0, 0.0, 0.0])
        rng = random.Random(42)
        result = select_parents(archive, 3, rng)
        assert len(result) == 3
        valid_ids = {g.id for g in archive}
        for g in result:
            assert g.id in valid_ids

    def test_select_count_respected(self):
        """select_parents returns exactly `count` entries."""
        archive = _make_archive([0.1, 0.5, 0.9])
        rng = random.Random(7)
        for count in [1, 3, 5, 10]:
            result = select_parents(archive, count, rng)
            assert len(result) == count, f"Expected {count} results, got {len(result)}"

    def test_non_selectable_excluded_from_candidates(self, tmp_path):
        """Entries with selectable=False must never be selected when selectable ones exist."""
        archive = [
            Generation(id="disabled-0", score=1.0, compiled_children=0, selectable=False),
            Generation(id="enabled-1", score=0.1, compiled_children=0, selectable=True),
            Generation(id="enabled-2", score=0.2, compiled_children=0, selectable=True),
        ]
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            rng = random.Random(42)
            result = select_parents(archive, 50, rng)
        selected_ids = {g.id for g in result}
        assert "disabled-0" not in selected_ids, "Non-selectable entry should never be selected"


# ---------------------------------------------------------------------------
# 3. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_sequence(self, tmp_path):
        """Same seed + same archive → same selected sequence."""
        archive = _make_archive([0.1, 0.3, 0.5, 0.7, 0.9])
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            rng_a = random.Random(42)
            result_a = select_parents(archive, 5, rng_a)

            rng_b = random.Random(42)
            result_b = select_parents(archive, 5, rng_b)

        assert [g.id for g in result_a] == [g.id for g in result_b]

    def test_different_seed_different_sequence(self, tmp_path):
        """Different seeds → different sequences (with high probability over 10 draws)."""
        archive = _make_archive([0.1, 0.3, 0.5, 0.7, 0.9])
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            rng_a = random.Random(1)
            result_a = select_parents(archive, 10, rng_a)

            rng_b = random.Random(9999)
            result_b = select_parents(archive, 10, rng_b)

        # Very unlikely to be identical over 10 draws from 5-entry archive
        assert [g.id for g in result_a] != [g.id for g in result_b], (
            "Different seeds produced identical sequences (astronomically unlikely)"
        )

    def test_sample_parent_deterministic(self, tmp_path):
        """sample_parent with same seed produces same parent."""
        archive = _make_archive([0.2, 0.5, 0.8])
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            rng_a = random.Random(77)
            p_a = sample_parent(archive, rng_a)

            rng_b = random.Random(77)
            p_b = sample_parent(archive, rng_b)

        assert p_a.id == p_b.id


# ---------------------------------------------------------------------------
# 4. Distributional sampling test (KS statistic D < 0.1)
# ---------------------------------------------------------------------------

class TestDistributionalSampling:
    """
    Port note: v2 §CONCERN 6 — distributional parity, not bit-exact.
    KS D < 0.1 (looser than plan's 0.05; 1000 trials is small).

    Archive: 5 entries, scores [0.1, 0.3, 0.5, 0.7, 0.9], compiled_children=0.
    Expected weights: sigmoid(score, alpha_mid) * novelty(0) for each.
    alpha_mid = mean of top-3 = (0.9 + 0.7 + 0.5) / 3 = 0.7.
    """

    N_TRIALS = 1000
    KS_THRESHOLD = 0.1

    def _expected_probs(self) -> list[float]:
        """Compute expected selection probabilities from formula."""
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        alpha_mid = (0.9 + 0.7 + 0.5) / 3  # = 0.7
        lam = 10.0
        weights = [
            sigmoid_weight(s, alpha_mid, lam) * novelty_weight(0)
            for s in scores
        ]
        total = sum(weights)
        return [w / total for w in weights]

    def test_ks_statistic_below_threshold(self, tmp_path):
        """1000-trial frequency histogram KS D < 0.1 vs expected sigmoid×novelty weights."""
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        archive = _make_archive(scores, compiled_children=[0] * 5)
        expected_probs = self._expected_probs()
        n = len(archive)

        counts = [0] * n
        id_to_idx = {g.id: i for i, g in enumerate(archive)}

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            rng = random.Random(42)
            for _ in range(self.N_TRIALS):
                parent = sample_parent(archive, rng)
                counts[id_to_idx[parent.id]] += 1

        observed_probs = [c / self.N_TRIALS for c in counts]

        # Kolmogorov-Smirnov statistic D = max|F_obs(x) - F_exp(x)| over CDF
        # Build CDFs from sorted probabilities
        sorted_indices = sorted(range(n), key=lambda i: expected_probs[i])
        cdf_obs = 0.0
        cdf_exp = 0.0
        d_max = 0.0
        for idx in sorted_indices:
            cdf_obs += observed_probs[idx]
            cdf_exp += expected_probs[idx]
            d_max = max(d_max, abs(cdf_obs - cdf_exp))

        assert d_max < self.KS_THRESHOLD, (
            f"KS statistic D={d_max:.4f} >= threshold {self.KS_THRESHOLD}. "
            f"Observed: {observed_probs}, Expected: {[f'{p:.4f}' for p in expected_probs]}"
        )

    def test_higher_score_selected_more_often(self, tmp_path):
        """Higher-scored entries should be selected more frequently than lower ones."""
        archive = _make_archive([0.1, 0.9], compiled_children=[0, 0])
        counts = {"gen-0": 0, "gen-1": 0}

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            rng = random.Random(0)
            for _ in range(500):
                p = sample_parent(archive, rng)
                counts[p.id] += 1

        assert counts["gen-1"] > counts["gen-0"], (
            f"Higher-scored entry selected less: high={counts['gen-1']}, low={counts['gen-0']}"
        )


# ---------------------------------------------------------------------------
# 5. Selection log tests
# ---------------------------------------------------------------------------

class TestSelectionLog:
    def test_one_call_writes_one_line(self, tmp_path):
        """One select_parents call writes exactly one JSONL line."""
        archive = _make_archive([0.3, 0.7])
        log_path = tmp_path / "dgmh" / "logs" / "select_parents.jsonl"

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            rng = random.Random(1)
            select_parents(archive, 1, rng)

        assert log_path.exists(), "Log file should be created"
        lines = [l for l in log_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 1, f"Expected 1 log line, got {len(lines)}"

    def test_log_has_all_required_fields(self, tmp_path):
        """Log entry must contain all required fields per W2 spec."""
        required_fields = {
            "timestamp", "candidate_ids", "scores", "alpha_mid",
            "sigmoid_weights", "novelty_bonuses", "combined_weights",
            "total_weight", "rng_draws", "selected_ids",
        }
        archive = _make_archive([0.4, 0.6, 0.8])

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            rng = random.Random(5)
            select_parents(archive, 2, rng)

        log_path = tmp_path / "dgmh" / "logs" / "select_parents.jsonl"
        entry = json.loads(log_path.read_text().strip())
        missing = required_fields - set(entry.keys())
        assert not missing, f"Log entry missing fields: {missing}"

    def test_log_selected_ids_match_returned(self, tmp_path):
        """selected_ids in log must match the returned Generation ids."""
        archive = _make_archive([0.2, 0.5, 0.8])

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            rng = random.Random(13)
            result = select_parents(archive, 3, rng)

        log_path = tmp_path / "dgmh" / "logs" / "select_parents.jsonl"
        entry = json.loads(log_path.read_text().strip())
        assert entry["selected_ids"] == [g.id for g in result]

    def test_multiple_calls_append_not_overwrite(self, tmp_path):
        """Multiple calls must append to log, never overwrite."""
        archive = _make_archive([0.3, 0.7])

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            for i in range(5):
                rng = random.Random(i)
                select_parents(archive, 1, rng)

        log_path = tmp_path / "dgmh" / "logs" / "select_parents.jsonl"
        lines = [l for l in log_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 5, f"Expected 5 log lines after 5 calls, got {len(lines)}"
        # Each line must be valid JSON
        for i, line in enumerate(lines):
            json.loads(line)  # raises if invalid

    def test_log_candidate_ids_and_scores_match_archive(self, tmp_path):
        """candidate_ids and scores in log must match the selectable archive entries."""
        archive = _make_archive([0.1, 0.5, 0.9])

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            rng = random.Random(99)
            select_parents(archive, 1, rng)

        log_path = tmp_path / "dgmh" / "logs" / "select_parents.jsonl"
        entry = json.loads(log_path.read_text().strip())
        assert entry["candidate_ids"] == [g.id for g in archive]
        assert entry["scores"] == [g.score for g in archive]

    def test_log_alpha_mid_is_correct(self, tmp_path):
        """alpha_mid in log matches expected top-3 mean."""
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        archive = _make_archive(scores)
        expected_alpha_mid = (0.9 + 0.7 + 0.5) / 3  # 0.7

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            rng = random.Random(7)
            select_parents(archive, 1, rng)

        log_path = tmp_path / "dgmh" / "logs" / "select_parents.jsonl"
        entry = json.loads(log_path.read_text().strip())
        assert abs(entry["alpha_mid"] - expected_alpha_mid) < 1e-9

    def test_log_rng_draws_count_matches_count(self, tmp_path):
        """rng_draws list length must equal the `count` argument."""
        archive = _make_archive([0.4, 0.6])

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            rng = random.Random(3)
            select_parents(archive, 4, rng)

        log_path = tmp_path / "dgmh" / "logs" / "select_parents.jsonl"
        entry = json.loads(log_path.read_text().strip())
        assert len(entry["rng_draws"]) == 4
