"""
dgmh/tests/test_judge.py — Tests for judge.py Codex-judge scorer.

Verifies:
- score_skill returns float in [0, 1]
- empty eval_prompt_pack returns 0.5 baseline (not an error)
- subprocess failure raises CodexJudgeError (fail-closed)
- timeout raises CodexJudgeError with timedOut=True
- non-zero exit raises CodexJudgeError
- empty stdout raises CodexJudgeError
- JSON without 'score' raises CodexJudgeError
- extract_first_json_object handles markdown fences and nested text

Reference: playground/dgmh-engine/codexSubprocess.ts (fail-closed semantics)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dgmh.judge import (
    CodexJudgeError,
    extract_first_json_object,
    score_skill,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_SKILL_MD = """\
---
name: test-skill
description: A test skill.
---
# Test Skill
This skill helps with testing.
"""

EVAL_PACK = ["Does this skill help test something? Rate 0-1."]


def _make_completed_process(stdout: str, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    r = subprocess.CompletedProcess(args=["codex", "exec", "-"], returncode=returncode)
    r.stdout = stdout
    r.stderr = stderr
    return r


# ---------------------------------------------------------------------------
# extract_first_json_object
# ---------------------------------------------------------------------------

class TestExtractFirstJsonObject:
    def test_plain_json(self):
        obj = extract_first_json_object('{"score": 0.8, "reasoning": "good"}')
        assert obj == {"score": 0.8, "reasoning": "good"}

    def test_json_embedded_in_text(self):
        obj = extract_first_json_object('Here is the result: {"score": 0.5, "reasoning": "ok"} done.')
        assert obj is not None
        assert obj["score"] == 0.5

    def test_json_in_code_fence(self):
        text = '```json\n{"score": 0.9, "reasoning": "excellent"}\n```'
        obj = extract_first_json_object(text)
        assert obj is not None
        assert obj["score"] == 0.9

    def test_no_json_returns_none(self):
        obj = extract_first_json_object("No JSON here at all.")
        assert obj is None

    def test_malformed_json_returns_none(self):
        obj = extract_first_json_object("{score: 0.8}")  # invalid JSON
        assert obj is None


# ---------------------------------------------------------------------------
# score_skill — baseline
# ---------------------------------------------------------------------------

class TestScoreSkillBaseline:
    def test_empty_pack_returns_half(self):
        """Empty eval_prompt_pack → 0.5 (no Codex call)."""
        result = score_skill(SAMPLE_SKILL_MD, [], seed=42)
        assert result == 0.5

    def test_none_pack_treated_as_empty(self):
        """Passing [] explicitly returns 0.5."""
        result = score_skill(SAMPLE_SKILL_MD, [])
        assert result == 0.5


# ---------------------------------------------------------------------------
# score_skill — success path (subprocess mocked)
# ---------------------------------------------------------------------------

class TestScoreSkillSuccess:
    def test_returns_float_in_range(self):
        """Single task → Codex returns 0.75 → score_skill returns 0.75."""
        stdout = json.dumps({"score": 0.75, "reasoning": "good skill"})
        with patch("subprocess.run", return_value=_make_completed_process(stdout)):
            result = score_skill(SAMPLE_SKILL_MD, EVAL_PACK, seed=0)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        assert abs(result - 0.75) < 1e-9

    def test_averages_multiple_tasks(self):
        """Multiple tasks → scores averaged."""
        responses = [
            json.dumps({"score": 0.6, "reasoning": "ok"}),
            json.dumps({"score": 0.8, "reasoning": "nice"}),
        ]
        call_count = 0

        def _fake_run(*args, **kwargs):
            nonlocal call_count
            r = _make_completed_process(responses[call_count % len(responses)])
            call_count += 1
            return r

        with patch("subprocess.run", side_effect=_fake_run):
            result = score_skill(SAMPLE_SKILL_MD, ["task1", "task2"], seed=0)

        assert abs(result - 0.7) < 1e-9

    def test_score_clamped_above_1(self):
        """Codex returning score > 1.0 is clamped to 1.0."""
        stdout = json.dumps({"score": 1.5, "reasoning": "over"})
        with patch("subprocess.run", return_value=_make_completed_process(stdout)):
            result = score_skill(SAMPLE_SKILL_MD, EVAL_PACK)
        assert result == 1.0

    def test_score_clamped_below_0(self):
        """Codex returning score < 0.0 is clamped to 0.0."""
        stdout = json.dumps({"score": -0.3, "reasoning": "under"})
        with patch("subprocess.run", return_value=_make_completed_process(stdout)):
            result = score_skill(SAMPLE_SKILL_MD, EVAL_PACK)
        assert result == 0.0


# ---------------------------------------------------------------------------
# score_skill — fail-closed paths
# ---------------------------------------------------------------------------

class TestScoreSkillFailClosed:
    def test_nonzero_exit_raises(self):
        """Non-zero subprocess exit → CodexJudgeError (fail-closed)."""
        with patch(
            "subprocess.run",
            return_value=_make_completed_process("", returncode=1, stderr="error"),
        ):
            with pytest.raises(CodexJudgeError) as exc_info:
                score_skill(SAMPLE_SKILL_MD, EVAL_PACK)
        assert exc_info.value.exit_code == 1

    def test_empty_stdout_raises(self):
        """Empty stdout → CodexJudgeError."""
        with patch("subprocess.run", return_value=_make_completed_process("")):
            with pytest.raises(CodexJudgeError):
                score_skill(SAMPLE_SKILL_MD, EVAL_PACK)

    def test_no_json_in_output_raises(self):
        """Stdout with no parseable JSON → CodexJudgeError."""
        with patch(
            "subprocess.run",
            return_value=_make_completed_process("I cannot score this skill."),
        ):
            with pytest.raises(CodexJudgeError):
                score_skill(SAMPLE_SKILL_MD, EVAL_PACK)

    def test_missing_score_key_raises(self):
        """JSON without 'score' key → CodexJudgeError."""
        stdout = json.dumps({"verdict": "good", "reasoning": "fine"})
        with patch("subprocess.run", return_value=_make_completed_process(stdout)):
            with pytest.raises(CodexJudgeError):
                score_skill(SAMPLE_SKILL_MD, EVAL_PACK)

    def test_timeout_raises_with_flag(self):
        """Timeout → CodexJudgeError with timed_out=True."""
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["codex"], timeout=60),
        ):
            with pytest.raises(CodexJudgeError) as exc_info:
                score_skill(SAMPLE_SKILL_MD, EVAL_PACK)
        assert exc_info.value.timed_out is True

    def test_file_not_found_raises(self):
        """Missing Codex binary → CodexJudgeError."""
        with patch("subprocess.run", side_effect=FileNotFoundError("codex not found")):
            with pytest.raises(CodexJudgeError):
                score_skill(SAMPLE_SKILL_MD, EVAL_PACK)

    def test_os_error_raises(self):
        """Generic OSError → CodexJudgeError."""
        with patch("subprocess.run", side_effect=OSError("spawn error")):
            with pytest.raises(CodexJudgeError):
                score_skill(SAMPLE_SKILL_MD, EVAL_PACK)

    def test_score_not_numeric_raises(self):
        """score field not a number → CodexJudgeError."""
        stdout = json.dumps({"score": "high", "reasoning": "..."})
        with patch("subprocess.run", return_value=_make_completed_process(stdout)):
            with pytest.raises(CodexJudgeError):
                score_skill(SAMPLE_SKILL_MD, EVAL_PACK)
