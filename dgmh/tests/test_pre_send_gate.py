"""Tests for dgmh.hermes_integration.pre_send_gate (thin layer)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import unittest
from unittest import mock

from dgmh.hermes_integration import pre_send_gate as gate_mod
from dgmh.hermes_integration.pre_send_gate import (
    _resolve_max_chars,
    _truncate_at_sentence,
    gate,
)


# Use a real-but-existing path so the skill-present branch runs in tests that
# mock subprocess.run. The actual binary at this path is irrelevant when
# subprocess is mocked. We use the real skill location to avoid relying on any
# write side-effects.
_REAL_SKILL_BIN = os.path.expanduser(
    "~/.hermes/skills/dgmh-pre-send-gate/bin/dgmh-pre-send-gate.py"
)
_NONEXISTENT_BIN = "/tmp/__definitely_not_a_real_path__/nope.py"


def _reset_skill_warning_flag() -> None:
    gate_mod._seen_skill_warning = False


class TestResolveMaxChars(unittest.TestCase):
    def setUp(self) -> None:
        for k in ("DGMH_PUBLIC_MAX_CHARS", "DGMH_OPERATOR_MAX_CHARS"):
            os.environ.pop(k, None)

    def test_default_public(self) -> None:
        self.assertEqual(_resolve_max_chars("public"), 200)

    def test_default_operator(self) -> None:
        self.assertEqual(_resolve_max_chars("operator"), 600)

    def test_public_env_override(self) -> None:
        with mock.patch.dict(os.environ, {"DGMH_PUBLIC_MAX_CHARS": "100"}):
            self.assertEqual(_resolve_max_chars("public"), 100)

    def test_operator_env_override(self) -> None:
        with mock.patch.dict(os.environ, {"DGMH_OPERATOR_MAX_CHARS": "300"}):
            self.assertEqual(_resolve_max_chars("operator"), 300)

    def test_invalid_env_falls_back(self) -> None:
        with mock.patch.dict(os.environ, {"DGMH_PUBLIC_MAX_CHARS": "garbage"}):
            self.assertEqual(_resolve_max_chars("public"), 200)


class TestTruncateAtSentence(unittest.TestCase):
    def test_under_cap_unchanged(self) -> None:
        self.assertEqual(_truncate_at_sentence("hello", 100), "hello")

    def test_period_boundary(self) -> None:
        text = "first part here. and a much longer second part beyond the cap"
        out = _truncate_at_sentence(text, 30)
        self.assertTrue(out.endswith("…"))
        self.assertIn("first part here.", out)

    def test_no_boundary_hard_truncate(self) -> None:
        text = "abcdefghijklmnopqrstuvwxyz"
        out = _truncate_at_sentence(text, 10)
        self.assertEqual(out, "abcdefghi…")


class TestGateSkillPath(unittest.TestCase):
    def setUp(self) -> None:
        _reset_skill_warning_flag()
        for k in ("DGMH_PUBLIC_MAX_CHARS", "DGMH_OPERATOR_MAX_CHARS", "DGMH_GATE_SKILL_BIN"):
            os.environ.pop(k, None)

    def test_skill_subprocess_success(self) -> None:
        # Long input so the skill path is exercised even if a fallback existed.
        content = "x" * 500
        fake_result = mock.Mock(
            returncode=0,
            stdout=json.dumps({"decision": "compress", "content": "short"}),
            stderr="",
        )
        with mock.patch.object(gate_mod.subprocess, "run", return_value=fake_result):
            decision, out = gate(content, channel_kind="public", skill_bin=_REAL_SKILL_BIN)
        self.assertEqual(decision, "compress")
        self.assertEqual(out, "short")

    def test_skill_subprocess_timeout(self) -> None:
        content = "long. " * 80  # exceeds 200
        with mock.patch.object(
            gate_mod.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd="python3", timeout=35.0),
        ):
            decision, out = gate(content, channel_kind="public", skill_bin=_REAL_SKILL_BIN)
        # Fallback truncates → compress.
        self.assertEqual(decision, "compress")
        self.assertTrue(out.endswith("…"))
        self.assertLessEqual(len(out), 200)

    def test_skill_subprocess_nonzero_exit(self) -> None:
        content = "long. " * 80
        fake_result = mock.Mock(returncode=1, stdout="", stderr="boom")
        with mock.patch.object(gate_mod.subprocess, "run", return_value=fake_result):
            decision, out = gate(content, channel_kind="public", skill_bin=_REAL_SKILL_BIN)
        self.assertEqual(decision, "compress")
        self.assertLessEqual(len(out), 200)

    def test_skill_missing_falls_back_and_warns_once(self) -> None:
        _reset_skill_warning_flag()
        content = "long. " * 80
        with self.assertLogs("dgmh.hermes_integration.pre_send_gate", level=logging.WARNING) as cap:
            decision1, out1 = gate(content, channel_kind="public", skill_bin=_NONEXISTENT_BIN)
            decision2, out2 = gate(content, channel_kind="public", skill_bin=_NONEXISTENT_BIN)
        self.assertEqual(decision1, "compress")
        self.assertEqual(decision2, "compress")
        warn_lines = [r for r in cap.records if r.levelno == logging.WARNING and "skill binary missing" in r.getMessage()]
        self.assertEqual(len(warn_lines), 1, "missing-skill warning should fire exactly once per process")

    def test_skill_json_malformed(self) -> None:
        content = "long. " * 80
        fake_result = mock.Mock(returncode=0, stdout="not json", stderr="")
        with mock.patch.object(gate_mod.subprocess, "run", return_value=fake_result):
            decision, out = gate(content, channel_kind="public", skill_bin=_REAL_SKILL_BIN)
        self.assertEqual(decision, "compress")
        self.assertLessEqual(len(out), 200)

    def test_skill_invalid_decision(self) -> None:
        content = "long. " * 80
        fake_result = mock.Mock(
            returncode=0,
            stdout=json.dumps({"decision": "explode", "content": "x"}),
            stderr="",
        )
        with mock.patch.object(gate_mod.subprocess, "run", return_value=fake_result):
            decision, out = gate(content, channel_kind="public", skill_bin=_REAL_SKILL_BIN)
        self.assertEqual(decision, "compress")
        self.assertLessEqual(len(out), 200)


class TestGateNoSkillNeeded(unittest.TestCase):
    """Cases where input is short enough that no skill call should happen."""

    def setUp(self) -> None:
        _reset_skill_warning_flag()
        for k in ("DGMH_PUBLIC_MAX_CHARS", "DGMH_OPERATOR_MAX_CHARS", "DGMH_GATE_SKILL_BIN"):
            os.environ.pop(k, None)

    def test_empty_input(self) -> None:
        # Skill missing path: empty input → silent without invoking subprocess.
        with mock.patch.object(gate_mod.subprocess, "run") as mock_run:
            decision, out = gate("", channel_kind="public", skill_bin=_NONEXISTENT_BIN)
        self.assertEqual(decision, "silent")
        self.assertEqual(out, "")
        # When skill is missing, fallback runs in-process and subprocess.run is not invoked.
        mock_run.assert_not_called()

    def test_below_cap(self) -> None:
        content = "x" * 100
        # Under cap with skill present: skill should report send unchanged.
        fake_result = mock.Mock(
            returncode=0,
            stdout=json.dumps({"decision": "send", "content": content}),
            stderr="",
        )
        with mock.patch.object(gate_mod.subprocess, "run", return_value=fake_result):
            decision, out = gate(content, channel_kind="public", skill_bin=_REAL_SKILL_BIN)
        self.assertEqual(decision, "send")
        self.assertEqual(out, content)

    def test_at_cap_boundary(self) -> None:
        content = "x" * 200
        fake_result = mock.Mock(
            returncode=0,
            stdout=json.dumps({"decision": "send", "content": content}),
            stderr="",
        )
        with mock.patch.object(gate_mod.subprocess, "run", return_value=fake_result):
            decision, out = gate(content, channel_kind="public", skill_bin=_REAL_SKILL_BIN)
        self.assertEqual(decision, "send")
        self.assertEqual(len(out), 200)


class TestFallbackTruncate(unittest.TestCase):
    def test_sentence_boundary(self) -> None:
        # Two sentences; second past 200. Fallback truncates after first period.
        sentence_a = "이건 첫 번째 문장이야. "  # ~14 chars
        # Pad to ensure first sentence exists, then add long second sentence.
        first = "이건 짧은 첫 문장이야. "
        rest = "두 번째 문장은 매우 길어서 200자 캡을 한참 넘어가게 만들어. " * 15
        text = first + rest
        self.assertGreater(len(text), 200)
        # No skill: fallback engages.
        decision, out = gate(text, channel_kind="public", skill_bin=_NONEXISTENT_BIN)
        self.assertEqual(decision, "compress")
        self.assertTrue(out.endswith("…"))
        self.assertIn(".", out)
        self.assertLessEqual(len(out), 200)


class TestEnvOverrides(unittest.TestCase):
    def setUp(self) -> None:
        _reset_skill_warning_flag()
        for k in ("DGMH_PUBLIC_MAX_CHARS", "DGMH_OPERATOR_MAX_CHARS"):
            os.environ.pop(k, None)

    def test_public_env_override_triggers(self) -> None:
        # Cap to 100; 150-char input must compress.
        content = "x" * 150
        with mock.patch.dict(os.environ, {"DGMH_PUBLIC_MAX_CHARS": "100"}):
            decision, out = gate(content, channel_kind="public", skill_bin=_NONEXISTENT_BIN)
        self.assertEqual(decision, "compress")
        self.assertLessEqual(len(out), 100)

    def test_operator_default_600(self) -> None:
        # 500-char input under operator cap → send unchanged via fallback.
        content = "x" * 500
        decision, out = gate(content, channel_kind="operator", skill_bin=_NONEXISTENT_BIN)
        self.assertEqual(decision, "send")
        self.assertEqual(out, content)


if __name__ == "__main__":
    unittest.main()
