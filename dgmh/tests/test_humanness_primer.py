"""Tests for the SOUL evolution humanness primer injection."""

from __future__ import annotations

import unittest

from dgmh.soul_evolution import (
    _augment_with_humanness_primer,
    _load_humanness_primer,
)


class TestPrimerLoad(unittest.TestCase):
    def test_primer_loads_non_empty(self) -> None:
        primer = _load_humanness_primer()
        self.assertGreater(len(primer), 1000)

    def test_primer_mentions_korean_patterns(self) -> None:
        primer = _load_humanness_primer()
        self.assertIn("번역체", primer)
        self.assertIn("결론적으로", primer)

    def test_primer_includes_voice_principles(self) -> None:
        primer = _load_humanness_primer()
        self.assertIn("Voice injection", primer)
        self.assertIn("Take a position", primer)

    def test_primer_includes_operator_voice_corpus(self) -> None:
        primer = _load_humanness_primer()
        self.assertIn("Operator voice ground truth", primer)
        self.assertIn("devswha", primer)
        # at least one verbatim sample present
        self.assertIn("이모지 개띠껍네", primer)


class TestPrimerAugmentation(unittest.TestCase):
    def test_augment_appends_primer_section(self) -> None:
        base = "BASE_PROMPT_TEMPLATE_TEXT"
        augmented = _augment_with_humanness_primer(base)
        self.assertIn(base, augmented)
        self.assertIn("Humanness Primer", augmented)
        self.assertGreater(len(augmented), len(base) + 1000)

    def test_augment_preserves_base_at_top(self) -> None:
        base = "ORIGINAL_INSTRUCTIONS"
        augmented = _augment_with_humanness_primer(base)
        self.assertTrue(augmented.startswith(base))

    def test_augment_with_empty_base(self) -> None:
        augmented = _augment_with_humanness_primer("")
        self.assertIn("Humanness Primer", augmented)


if __name__ == "__main__":
    unittest.main()
