"""End-to-end tests for the dgmh-pre-send-gate skill binary itself.

These spawn the actual skill script via subprocess. Patina-calling tests are
skipped when the patina binary isn't available; everything else uses
``patina_bin="/nonexistent"`` to keep tests hermetic.
"""

from __future__ import annotations

import json
import os
import subprocess
import unittest
from pathlib import Path


_SKILL_BIN = Path(
    os.path.expanduser("~/.hermes/skills/dgmh-pre-send-gate/bin/dgmh-pre-send-gate.py")
)
_PATINA_BIN = Path(
    os.path.expanduser("~/.hermes/skills/creative/patina/bin/patina.js")
)


def _run_skill(payload: dict, timeout: float = 10.0) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["python3", str(_SKILL_BIN)],
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


@unittest.skipUnless(_SKILL_BIN.exists(), f"skill binary not present at {_SKILL_BIN}")
class TestSkillBinary(unittest.TestCase):
    def test_empty_input(self) -> None:
        rc, stdout, _stderr = _run_skill(
            {"content": "", "channel_kind": "public", "max_chars": 200}
        )
        self.assertEqual(rc, 0)
        parsed = json.loads(stdout.strip())
        self.assertEqual(parsed["decision"], "silent")
        self.assertEqual(parsed["content"], "")

    def test_below_cap(self) -> None:
        rc, stdout, _stderr = _run_skill(
            {"content": "hello", "channel_kind": "public", "max_chars": 200}
        )
        self.assertEqual(rc, 0)
        parsed = json.loads(stdout.strip())
        self.assertEqual(parsed["decision"], "send")
        self.assertEqual(parsed["content"], "hello")

    def test_at_cap_boundary(self) -> None:
        content = "x" * 200
        rc, stdout, _stderr = _run_skill(
            {"content": content, "channel_kind": "public", "max_chars": 200}
        )
        self.assertEqual(rc, 0)
        parsed = json.loads(stdout.strip())
        self.assertEqual(parsed["decision"], "send")
        self.assertEqual(parsed["content"], content)

    def test_over_cap_with_nonexistent_patina_truncates(self) -> None:
        # Force the no-patina branch via a nonexistent patina_bin.
        text = (
            "first sentence here. second sentence here. "
            + "padding word " * 80
        )
        rc, stdout, _stderr = _run_skill(
            {
                "content": text,
                "channel_kind": "public",
                "max_chars": 50,
                "patina_bin": "/nonexistent/patina.js",
            }
        )
        self.assertEqual(rc, 0)
        parsed = json.loads(stdout.strip())
        self.assertEqual(parsed["decision"], "compress")
        self.assertLessEqual(len(parsed["content"]), 50)
        self.assertTrue(parsed["content"].endswith("…"))

    def test_invalid_json_input_returns_safe_payload(self) -> None:
        # Pass garbage on stdin; skill should emit a JSON object with a benign
        # decision and exit 0.
        proc = subprocess.run(
            ["python3", str(_SKILL_BIN)],
            input="not json at all",
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        self.assertEqual(proc.returncode, 0)
        parsed = json.loads(proc.stdout.strip())
        self.assertIn("decision", parsed)
        self.assertEqual(parsed["decision"], "send")
        self.assertEqual(parsed.get("reason"), "json_parse_error")


if __name__ == "__main__":
    unittest.main()
