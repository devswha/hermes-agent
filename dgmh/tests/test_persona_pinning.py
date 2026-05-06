"""Step 7 / AC10 (v3) — two-tier persona pinning.

  PERSONA-IDENTITY block is invariant. Any candidate diff against the
  parent's IDENTITY is force-reverted; missing / duplicate markers
  reject the candidate.

  PERSONA-VOICE numeric ``target: NN%`` lines must satisfy 55 ≤ N ≤ 85.
  Out-of-bounds values are auto-clamped to the nearest in-bounds value.

Drift events go to ``~/.hermes/dgmh/persona_drift.jsonl``. The tests
redirect that path to a tempdir so they don't pollute the operator's
real drift log.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dgmh.persona_pinning import (
    IDENTITY_END_MARKER,
    IDENTITY_START_MARKER,
    PersonaPinError,
    VOICE_END_MARKER,
    VOICE_START_MARKER,
    _clamp_target,
    enforce_persona_invariants,
)


_PARENT_IDENTITY = (
    f"{IDENTITY_START_MARKER}: HARD-PIN, modifier may not mutate -->\n"
    "- Display name: flask\n"
    "- Backstory: 20대 후반, 개발 보조 일하는 동네 친구\n"
    f"{IDENTITY_END_MARKER}"
)
_PARENT_VOICE = (
    f"{VOICE_START_MARKER}: SOFT-PIN, modifier may tune within bounds -->\n"
    "- Register: 해요체 with strangers, 반말 mirroring asker\n"
    "- Casual feature target: target: 65%\n"
    "- Knowledge dampening: target: 70%\n"
    f"{VOICE_END_MARKER}"
)


def _build_soul(identity: str = _PARENT_IDENTITY, voice: str = _PARENT_VOICE) -> str:
    return f"---\nname: flask\n---\n\n{identity}\n\n{voice}\n\nbody text\n"


class TestIdentityHardPin(unittest.TestCase):
    def test_identical_block_no_drift(self) -> None:
        parent = _build_soul()
        child = _build_soul()
        revised, events = enforce_persona_invariants(
            parent,
            child,
            log_path=Path(tempfile.mkstemp()[1]),
        )
        self.assertEqual(revised, child)
        self.assertEqual(events, [])

    def test_identity_drift_force_reverted(self) -> None:
        parent = _build_soul()
        # Modify the IDENTITY block — this should be force-replaced.
        modified_identity = (
            f"{IDENTITY_START_MARKER}: HARD-PIN, modifier may not mutate -->\n"
            "- Display name: NOT-FLASK-INJECTED\n"
            "- Backstory: 다른 backstory\n"
            f"{IDENTITY_END_MARKER}"
        )
        child = _build_soul(identity=modified_identity)
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "drift.jsonl"
            revised, events = enforce_persona_invariants(
                parent,
                child,
                parent_hash="abc",
                child_hash="def",
                log_path=log,
            )
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].kind, "identity")
            self.assertEqual(events[0].event, "force-replace")
            self.assertIn("flask", revised)
            self.assertNotIn("NOT-FLASK-INJECTED", revised)
            # Drift was logged.
            self.assertTrue(log.exists())
            line = log.read_text(encoding="utf-8").strip().splitlines()[0]
            payload = json.loads(line)
            self.assertEqual(payload["kind"], "identity")
            self.assertEqual(payload["parent_hash"], "abc")
            self.assertEqual(payload["child_hash"], "def")

    def test_missing_identity_marker_rejects(self) -> None:
        parent = _build_soul()
        # Strip identity markers entirely.
        child = _build_soul(identity="")
        with self.assertRaises(PersonaPinError):
            enforce_persona_invariants(parent, child)

    def test_duplicate_identity_marker_rejects(self) -> None:
        parent = _build_soul()
        child = _build_soul() + "\n\n" + _PARENT_IDENTITY  # second copy
        with self.assertRaises(PersonaPinError):
            enforce_persona_invariants(parent, child)

    def test_parent_without_identity_block_passes_through(self) -> None:
        # Legacy parent — no IDENTITY block to enforce.
        parent = "---\nname: flask\n---\n\nbody text\n"
        child = parent
        revised, events = enforce_persona_invariants(parent, child)
        self.assertEqual(revised, child)
        self.assertEqual(events, [])


class TestVoiceSoftPin(unittest.TestCase):
    def test_in_bounds_targets_unchanged(self) -> None:
        parent = _build_soul()
        child = _build_soul()
        revised, events = enforce_persona_invariants(parent, child)
        self.assertEqual(revised, child)
        self.assertEqual(events, [])

    def test_below_min_clamps_to_55(self) -> None:
        parent = _build_soul()
        below_voice = _PARENT_VOICE.replace("target: 65%", "target: 30%")
        child = _build_soul(voice=below_voice)
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "drift.jsonl"
            revised, events = enforce_persona_invariants(
                parent, child, parent_hash="p", child_hash="c", log_path=log
            )
        self.assertIn("target: 55%", revised)
        self.assertNotIn("target: 30%", revised)
        out_of_bounds_events = [e for e in events if e.kind == "voice"]
        self.assertEqual(len(out_of_bounds_events), 1)
        self.assertEqual(out_of_bounds_events[0].event, "out-of-bounds")
        self.assertIn("30%", out_of_bounds_events[0].before)
        self.assertIn("55%", out_of_bounds_events[0].after)

    def test_above_max_clamps_to_85(self) -> None:
        parent = _build_soul()
        above_voice = _PARENT_VOICE.replace("target: 70%", "target: 95%")
        child = _build_soul(voice=above_voice)
        revised, events = enforce_persona_invariants(parent, child)
        self.assertIn("target: 85%", revised)
        self.assertNotIn("target: 95%", revised)
        self.assertEqual(len(events), 1)

    def test_multiple_out_of_bounds_clamped_independently(self) -> None:
        parent = _build_soul()
        bad_voice = _PARENT_VOICE.replace(
            "target: 65%", "target: 10%"
        ).replace("target: 70%", "target: 99%")
        child = _build_soul(voice=bad_voice)
        revised, events = enforce_persona_invariants(parent, child)
        self.assertIn("target: 55%", revised)
        self.assertIn("target: 85%", revised)
        self.assertNotIn("target: 10%", revised)
        self.assertNotIn("target: 99%", revised)
        self.assertEqual(len(events), 2)

    def test_clamp_helper_bounds(self) -> None:
        self.assertEqual(_clamp_target(40), 55)
        self.assertEqual(_clamp_target(55), 55)
        self.assertEqual(_clamp_target(70), 70)
        self.assertEqual(_clamp_target(85), 85)
        self.assertEqual(_clamp_target(99), 85)


class TestAC10FiveCycleIdentityZeroDiff(unittest.TestCase):
    """AC10 (a): 5 evolution cycles with adversarial modifier diffs →
    identity-block diff = 0 each time, drift always logged."""

    def test_five_cycles_identity_invariant(self) -> None:
        parent = _build_soul()
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "drift.jsonl"
            for i in range(5):
                # Adversarial: mutate the IDENTITY block differently each cycle.
                bad_identity = (
                    f"{IDENTITY_START_MARKER}: HARD-PIN, modifier may not mutate -->\n"
                    f"- Display name: hijacked-{i}\n"
                    f"{IDENTITY_END_MARKER}"
                )
                child = _build_soul(identity=bad_identity)
                revised, events = enforce_persona_invariants(
                    parent,
                    child,
                    parent_hash=f"p{i}",
                    child_hash=f"c{i}",
                    log_path=log,
                )
                # IDENTITY block in revised candidate matches parent's.
                self.assertIn("Display name: flask", revised)
                self.assertNotIn(f"hijacked-{i}", revised)
                self.assertTrue(any(e.kind == "identity" for e in events))

            # Drift log has 5 identity events.
            lines = log.read_text(encoding="utf-8").strip().splitlines()
            identity_lines = [
                json.loads(l) for l in lines if json.loads(l)["kind"] == "identity"
            ]
            self.assertEqual(len(identity_lines), 5)


class TestAC10VoiceBoundsAcrossCycles(unittest.TestCase):
    """AC10 (b): voice block target NN% lines auto-corrected to in-bounds."""

    def test_random_out_of_bounds_clamped_consistently(self) -> None:
        parent = _build_soul()
        # Span the range from below-min to above-max plus in-bounds.
        cases = [
            ("target: 65%", "target: 65%"),  # in bounds → unchanged
            ("target: 65%", "target: 70%"),  # in bounds → unchanged
            ("target: 65%", "target: 30%"),  # below → clamp to 55
            ("target: 65%", "target: 100%"),  # above → clamp to 85
        ]
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "drift.jsonl"
            for orig, swapped in cases:
                child_voice = _PARENT_VOICE.replace(orig, swapped, 1)
                child = _build_soul(voice=child_voice)
                revised, events = enforce_persona_invariants(
                    parent, child, log_path=log
                )
                # Every numeric target NN% remaining must be in [55, 85].
                import re

                for n_str in re.findall(r"target:\s*(\d+)\s*%", revised):
                    n = int(n_str)
                    self.assertGreaterEqual(n, 55)
                    self.assertLessEqual(n, 85)


if __name__ == "__main__":
    unittest.main()
