"""Guard the SOUL.md sections that the DGM-H Mission depends on.

The Mission block, Knowledge confidence section, and KB v0 section
each landed deliberately as part of the "AI that looks human"
mission strengthening. If any gets removed (operator edit accident
or a future critic miss), the persona regresses toward fresh-LLM
behavior and the mission load-bearing rules disappear.

This test reads the live ~/.hermes/SOUL.md so the regression check
runs against the actual file the gateway loads each turn. It
gracefully skips when SOUL.md is absent (CI / fresh checkouts
without an operator home directory).
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path


_MISSION_HEAD = "# Mission"
_KNOWLEDGE_CONFIDENCE_HEAD = "# Knowledge confidence (HARD"
_KB_HEAD = "## Flask's persistent context (KB v0)"


def _soul_md_path() -> Path:
    raw = os.environ.get("HERMES_HOME") or "~/.hermes"
    return Path(raw).expanduser() / "SOUL.md"


def _read_or_skip(test: unittest.TestCase) -> str:
    path = _soul_md_path()
    if not path.exists():
        test.skipTest(f"SOUL.md not present at {path}")
    return path.read_text(encoding="utf-8")


class TestSoulMdIntegrity(unittest.TestCase):
    """Live SOUL.md must retain the Mission / Knowledge confidence / KB sections."""

    def test_mission_block_present(self) -> None:
        text = _read_or_skip(self)
        matches = [
            line
            for line in text.splitlines()
            if line.strip().startswith(_MISSION_HEAD)
        ]
        self.assertEqual(
            len(matches),
            1,
            msg=f"expected exactly one `{_MISSION_HEAD}` header; found {len(matches)}",
        )

    def test_knowledge_confidence_block_present(self) -> None:
        text = _read_or_skip(self)
        self.assertIn(
            _KNOWLEDGE_CONFIDENCE_HEAD,
            text,
            msg=(
                "Knowledge confidence section missing — Mission's "
                "hallucination guard is gone"
            ),
        )

    def test_kb_v0_section_present(self) -> None:
        text = _read_or_skip(self)
        self.assertIn(
            _KB_HEAD,
            text,
            msg=(
                "KB v0 section missing — persona persistent context "
                "(opinions / experiences / preferences / don't-know zones) "
                "is gone"
            ),
        )

    def test_mission_outranking_statement_intact(self) -> None:
        """The Mission block must keep its 'outranks every other HARD'
        line; weakening it means subsequent rules can override the
        single optimization target.
        """
        text = _read_or_skip(self)
        if _MISSION_HEAD not in text:
            self.skipTest("Mission header absent (covered by sibling test)")
        # Allow some minor rewording but require the outranking semantics.
        keywords_outranks = ("outrank", "우선")
        self.assertTrue(
            any(k in text for k in keywords_outranks),
            msg="Mission block lost its outranking statement",
        )


class TestSoulMdSkipBehavior(unittest.TestCase):
    """Verify graceful skip when SOUL.md is missing (CI / fresh env)."""

    def test_skip_when_soul_md_missing(self) -> None:
        # Point HERMES_HOME at a tmp dir that has no SOUL.md.
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            os.environ["HERMES_HOME"] = tmp
            try:
                with self.assertRaises(unittest.SkipTest):
                    _read_or_skip(self)
            finally:
                os.environ.pop("HERMES_HOME", None)


class TestSoulMdDeletionDetection(unittest.TestCase):
    """Meta-test — confirm the integrity checks would actually fail
    if a section were deleted, not just silently pass.
    """

    def test_missing_mission_would_fail(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            (Path(tmp) / "SOUL.md").write_text(
                "---\nname: flask\n---\n\nYou are flask.\n", encoding="utf-8"
            )
            os.environ["HERMES_HOME"] = tmp
            try:
                text = _read_or_skip(self)
                matches = [
                    line
                    for line in text.splitlines()
                    if line.strip().startswith(_MISSION_HEAD)
                ]
                self.assertEqual(
                    len(matches),
                    0,
                    msg="seed file should not contain the Mission header",
                )
            finally:
                os.environ.pop("HERMES_HOME", None)


if __name__ == "__main__":
    unittest.main()
