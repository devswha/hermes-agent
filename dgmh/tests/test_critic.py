"""
dgmh/tests/test_critic.py — Tests for critic.py (W4 deliverable).

Covers per W4 spec (>=10 tests):
1.  Happy approve: scripted Codex output → CriticReview(verdict="approve")
2.  Happy reject: scripted Codex output → CriticReview(verdict="reject")
3.  Parse fail re-roll: first attempt bad JSON, second returns valid verdict
4.  Subprocess fail-closed: CodexCriticError raised immediately, no re-roll
5.  Timeout: CodexCriticError with timed_out=True
6.  Fence-block tolerated preamble: prose before JSON object is ignored
7.  Missing reason rejected: parse returns (None, error) on empty reason
8.  Invalid verdict rejected: parse returns (None, error) for unknown verdict
9.  Parse exhaustion exception: CriticParseAttemptsExhaustedError after max
10. Prompt assembly: all four placeholders substituted
11. Factory validation: empty template / max_parse_attempts < 1 raise ValueError
12. elapsed_ms is populated in happy path
13. Approve with real archive summary

Reference: playground/dgmh-engine/codexCritic.ts (TS source)
           dgmh/critic.py (implementation under test)
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dgmh.critic import (
    CodexCriticError,
    CriticParseAttemptsExhaustedError,
    CriticReview,
    SkillCritic,
    _assemble_prompt,
    _extract_first_json_object,
    _format_archive_summary,
    _parse_verdict_envelope,
    make_codex_skill_critic,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_TEMPLATE = """\
PARENT:
{{PARENT_SKILL_MD}}
CHILD:
{{CHILD_SKILL_MD}}
ARCHIVE:
{{ARCHIVE_SUMMARY}}
ATTEMPT:
{{ATTEMPT}}
"""

SAMPLE_PARENT_MD = """\
---
name: test-skill
description: A test skill.
version: "1.0"
---
# Test Skill
Original instructions.
"""

SAMPLE_CHILD_MD = """\
---
name: test-skill
description: An improved test skill.
version: "1.1"
---
# Test Skill (improved)
Better instructions.
"""


@dataclass
class FakeSkill:
    """Minimal object with skill_md for tests."""
    skill_md: str = SAMPLE_CHILD_MD
    skill_id: str = "coding/test-skill"
    generation_index: int = 0
    score: float = 0.5
    compiled_children: int = 0


@dataclass
class FakeArchiveEntry:
    skill_id: str
    score: float
    compiled_children: int = 0
    generation_index: int = 0


def _approve_stdout(reason: str = "Child improves instructions.") -> str:
    return json.dumps({"verdict": "approve", "reason": reason})


def _reject_stdout(reason: str = "Cargo-cult mutation — no semantic change.") -> str:
    return json.dumps({"verdict": "reject", "reason": reason})


# ---------------------------------------------------------------------------
# Mock CodexInvoker
# ---------------------------------------------------------------------------


class ScriptedInvoker:
    """Returns pre-scripted responses in sequence. Raises on exhaustion."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = list(responses)
        self.call_count = 0
        self.prompts_received: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        self.prompts_received.append(prompt)
        if not self._responses:
            raise CodexCriticError("ScriptedInvoker: queue exhausted")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


# ---------------------------------------------------------------------------
# 1. Happy approve
# ---------------------------------------------------------------------------

class TestHappyApprove:
    def test_approve_verdict_returned(self):
        parent = FakeSkill(skill_md=SAMPLE_PARENT_MD)
        child = FakeSkill(skill_md=SAMPLE_CHILD_MD)
        invoker = ScriptedInvoker([_approve_stdout("Child adds clearer guidelines.")])
        critic = make_codex_skill_critic(MINIMAL_TEMPLATE, codex_invoker=invoker)

        review = critic.review(child, parent, [])

        assert isinstance(review, CriticReview)
        assert review.verdict == "approve"
        assert review.reason == "Child adds clearer guidelines."
        assert invoker.call_count == 1


# ---------------------------------------------------------------------------
# 2. Happy reject
# ---------------------------------------------------------------------------

class TestHappyReject:
    def test_reject_verdict_returned(self):
        parent = FakeSkill(skill_md=SAMPLE_PARENT_MD)
        child = FakeSkill(skill_md=SAMPLE_CHILD_MD)
        invoker = ScriptedInvoker([_reject_stdout("Scope leak: mentions tools/ path.")])
        critic = make_codex_skill_critic(MINIMAL_TEMPLATE, codex_invoker=invoker)

        review = critic.review(child, parent, [])

        assert review.verdict == "reject"
        assert "tools/" in review.reason
        assert invoker.call_count == 1


# ---------------------------------------------------------------------------
# 3. Parse fail re-roll → success
# ---------------------------------------------------------------------------

class TestParseFailReroll:
    def test_reroll_on_bad_json_then_success(self):
        parent = FakeSkill(skill_md=SAMPLE_PARENT_MD)
        child = FakeSkill(skill_md=SAMPLE_CHILD_MD)
        invoker = ScriptedInvoker([
            "This is not JSON at all.",
            _approve_stdout("Second attempt succeeds."),
        ])
        critic = make_codex_skill_critic(
            MINIMAL_TEMPLATE, codex_invoker=invoker, max_parse_attempts=2
        )

        review = critic.review(child, parent, [])

        assert review.verdict == "approve"
        assert invoker.call_count == 2

    def test_reroll_on_invalid_envelope_then_success(self):
        """First response has JSON but wrong schema, second is valid."""
        parent = FakeSkill(skill_md=SAMPLE_PARENT_MD)
        child = FakeSkill(skill_md=SAMPLE_CHILD_MD)
        invoker = ScriptedInvoker([
            json.dumps({"result": "ok"}),  # missing verdict + reason
            _approve_stdout("Valid on second attempt."),
        ])
        critic = make_codex_skill_critic(
            MINIMAL_TEMPLATE, codex_invoker=invoker, max_parse_attempts=2
        )

        review = critic.review(child, parent, [])

        assert review.verdict == "approve"
        assert invoker.call_count == 2


# ---------------------------------------------------------------------------
# 4. Subprocess fail-closed
# ---------------------------------------------------------------------------

class TestSubprocessFailClosed:
    def test_codex_error_raises_immediately(self):
        parent = FakeSkill(skill_md=SAMPLE_PARENT_MD)
        child = FakeSkill(skill_md=SAMPLE_CHILD_MD)
        invoker = ScriptedInvoker([
            CodexCriticError("exit 1", exit_code=1, stderr="error"),
        ])
        critic = make_codex_skill_critic(
            MINIMAL_TEMPLATE, codex_invoker=invoker, max_parse_attempts=2
        )

        with pytest.raises(CodexCriticError) as exc_info:
            critic.review(child, parent, [])

        assert exc_info.value.exit_code == 1
        assert invoker.call_count == 1  # no re-roll on subprocess failure

    def test_subprocess_failure_via_patch(self):
        """subprocess.run non-zero exit raises CodexCriticError (integration path)."""
        parent = FakeSkill(skill_md=SAMPLE_PARENT_MD)
        child = FakeSkill(skill_md=SAMPLE_CHILD_MD)
        critic = make_codex_skill_critic(MINIMAL_TEMPLATE, codex_invoker=None)

        fake_result = subprocess.CompletedProcess(
            args=["codex", "exec", "-"], returncode=1
        )
        fake_result.stdout = ""
        fake_result.stderr = "codex error"

        with patch("dgmh.critic.subprocess.run", return_value=fake_result):
            with pytest.raises(CodexCriticError) as exc_info:
                critic.review(child, parent, [])

        assert exc_info.value.exit_code == 1


# ---------------------------------------------------------------------------
# 5. Timeout
# ---------------------------------------------------------------------------

class TestTimeout:
    def test_timeout_raises_codex_critic_error(self):
        parent = FakeSkill(skill_md=SAMPLE_PARENT_MD)
        child = FakeSkill(skill_md=SAMPLE_CHILD_MD)
        invoker = ScriptedInvoker([
            CodexCriticError("timed out", timed_out=True, exit_code=None),
        ])
        critic = make_codex_skill_critic(MINIMAL_TEMPLATE, codex_invoker=invoker)

        with pytest.raises(CodexCriticError) as exc_info:
            critic.review(child, parent, [])

        assert exc_info.value.timed_out is True
        assert invoker.call_count == 1

    def test_timeout_via_patch(self):
        """subprocess.TimeoutExpired → CodexCriticError timed_out=True."""
        parent = FakeSkill(skill_md=SAMPLE_PARENT_MD)
        child = FakeSkill(skill_md=SAMPLE_CHILD_MD)
        critic = make_codex_skill_critic(MINIMAL_TEMPLATE, codex_invoker=None)

        with patch(
            "dgmh.critic.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["codex"], timeout=60),
        ):
            with pytest.raises(CodexCriticError) as exc_info:
                critic.review(child, parent, [])

        assert exc_info.value.timed_out is True


# ---------------------------------------------------------------------------
# 6. Fence-block tolerated preamble
# ---------------------------------------------------------------------------

class TestFenceBlockPreamble:
    def test_prose_before_json_is_tolerated(self):
        """Preamble prose before the JSON object does not break parsing."""
        stdout = "Here is my verdict:\n\n" + _approve_stdout("Clean mutation.")
        review, err = _parse_verdict_envelope(stdout)
        assert review is not None
        assert review.verdict == "approve"

    def test_json_in_fenced_block_is_tolerated(self):
        """JSON inside ```json fence is extracted correctly."""
        stdout = "```json\n" + _approve_stdout("Inside fence.") + "\n```"
        # _extract_first_json_object finds the first { ... }
        obj = _extract_first_json_object(stdout)
        assert obj is not None
        assert obj["verdict"] == "approve"


# ---------------------------------------------------------------------------
# 7. Missing reason rejected
# ---------------------------------------------------------------------------

class TestMissingReason:
    def test_empty_reason_causes_parse_failure(self):
        stdout = json.dumps({"verdict": "approve", "reason": ""})
        review, err = _parse_verdict_envelope(stdout)
        assert review is None
        assert "reason" in err

    def test_missing_reason_field_causes_parse_failure(self):
        stdout = json.dumps({"verdict": "approve"})
        review, err = _parse_verdict_envelope(stdout)
        assert review is None
        assert "reason" in err


# ---------------------------------------------------------------------------
# 8. Invalid verdict rejected
# ---------------------------------------------------------------------------

class TestInvalidVerdict:
    def test_unknown_verdict_causes_parse_failure(self):
        stdout = json.dumps({"verdict": "maybe", "reason": "unsure"})
        review, err = _parse_verdict_envelope(stdout)
        assert review is None
        assert "verdict" in err

    def test_null_verdict_causes_parse_failure(self):
        stdout = json.dumps({"verdict": None, "reason": "some reason"})
        review, err = _parse_verdict_envelope(stdout)
        assert review is None


# ---------------------------------------------------------------------------
# 9. Parse exhaustion exception
# ---------------------------------------------------------------------------

class TestParseExhaustion:
    def test_exhaustion_raises_after_max_attempts(self):
        parent = FakeSkill(skill_md=SAMPLE_PARENT_MD)
        child = FakeSkill(skill_md=SAMPLE_CHILD_MD)
        invoker = ScriptedInvoker(["not json", "also not json"])
        critic = make_codex_skill_critic(
            MINIMAL_TEMPLATE, codex_invoker=invoker, max_parse_attempts=2
        )

        with pytest.raises(CriticParseAttemptsExhaustedError) as exc_info:
            critic.review(child, parent, [])

        assert exc_info.value.attempts == 2
        assert invoker.call_count == 2

    def test_max_parse_attempts_one_exhausts_immediately(self):
        parent = FakeSkill(skill_md=SAMPLE_PARENT_MD)
        child = FakeSkill(skill_md=SAMPLE_CHILD_MD)
        invoker = ScriptedInvoker(["not json"])
        critic = make_codex_skill_critic(
            MINIMAL_TEMPLATE, codex_invoker=invoker, max_parse_attempts=1
        )

        with pytest.raises(CriticParseAttemptsExhaustedError) as exc_info:
            critic.review(child, parent, [])

        assert exc_info.value.attempts == 1
        assert invoker.call_count == 1


# ---------------------------------------------------------------------------
# 10. Prompt assembly
# ---------------------------------------------------------------------------

class TestPromptAssembly:
    def test_all_placeholders_substituted(self):
        parent = FakeSkill(skill_md=SAMPLE_PARENT_MD)
        child = FakeSkill(skill_md=SAMPLE_CHILD_MD)
        prompt = _assemble_prompt(MINIMAL_TEMPLATE, child, parent, [], 5, 0)
        assert "{{PARENT_SKILL_MD}}" not in prompt
        assert "{{CHILD_SKILL_MD}}" not in prompt
        assert "{{ARCHIVE_SUMMARY}}" not in prompt
        assert "{{ATTEMPT}}" not in prompt

    def test_parent_skill_md_in_prompt(self):
        parent = FakeSkill(skill_md=SAMPLE_PARENT_MD)
        child = FakeSkill(skill_md=SAMPLE_CHILD_MD)
        prompt = _assemble_prompt(MINIMAL_TEMPLATE, child, parent, [], 5, 0)
        assert SAMPLE_PARENT_MD in prompt

    def test_child_skill_md_in_prompt(self):
        parent = FakeSkill(skill_md=SAMPLE_PARENT_MD)
        child = FakeSkill(skill_md=SAMPLE_CHILD_MD)
        prompt = _assemble_prompt(MINIMAL_TEMPLATE, child, parent, [], 5, 0)
        assert SAMPLE_CHILD_MD in prompt

    def test_attempt_index_in_prompt(self):
        parent = FakeSkill(skill_md=SAMPLE_PARENT_MD)
        child = FakeSkill(skill_md=SAMPLE_CHILD_MD)
        prompt = _assemble_prompt(MINIMAL_TEMPLATE, child, parent, [], 5, 3)
        assert "3" in prompt

    def test_archive_summary_in_prompt(self):
        parent = FakeSkill(skill_md=SAMPLE_PARENT_MD)
        child = FakeSkill(skill_md=SAMPLE_CHILD_MD)
        archive = [FakeArchiveEntry("coding/skill-a", 0.8, generation_index=1)]
        prompt = _assemble_prompt(MINIMAL_TEMPLATE, child, parent, archive, 5, 0)
        assert "coding/skill-a" in prompt


# ---------------------------------------------------------------------------
# 11. Factory validation
# ---------------------------------------------------------------------------

class TestFactoryValidation:
    def test_empty_template_raises(self):
        with pytest.raises(ValueError, match="prompt_template"):
            make_codex_skill_critic("")

    def test_none_template_raises(self):
        with pytest.raises((ValueError, TypeError)):
            make_codex_skill_critic(None)  # type: ignore[arg-type]

    def test_max_parse_attempts_zero_raises(self):
        with pytest.raises(ValueError, match="max_parse_attempts"):
            make_codex_skill_critic(MINIMAL_TEMPLATE, max_parse_attempts=0)

    def test_returns_skill_critic_instance(self):
        critic = make_codex_skill_critic(MINIMAL_TEMPLATE)
        assert isinstance(critic, SkillCritic)


# ---------------------------------------------------------------------------
# 12. elapsed_ms populated
# ---------------------------------------------------------------------------

class TestElapsedMs:
    def test_elapsed_ms_is_non_negative(self):
        parent = FakeSkill(skill_md=SAMPLE_PARENT_MD)
        child = FakeSkill(skill_md=SAMPLE_CHILD_MD)
        invoker = ScriptedInvoker([_approve_stdout("Quick response.")])
        critic = make_codex_skill_critic(MINIMAL_TEMPLATE, codex_invoker=invoker)

        review = critic.review(child, parent, [])

        assert isinstance(review.elapsed_ms, int)
        assert review.elapsed_ms >= 0


# ---------------------------------------------------------------------------
# 13. Approve with real archive summary
# ---------------------------------------------------------------------------

class TestApproveWithArchive:
    def test_archive_entries_appear_in_prompt(self):
        archive = [
            FakeArchiveEntry("coding/skill-a", 0.9, generation_index=2),
            FakeArchiveEntry("coding/skill-b", 0.7, generation_index=1),
            FakeArchiveEntry("coding/skill-c", 0.5, generation_index=0),
        ]
        summary = _format_archive_summary(archive, recent_n=5)
        assert "coding/skill-a" in summary
        assert "coding/skill-b" in summary
        assert "coding/skill-c" in summary

    def test_recent_n_limits_archive_in_summary(self):
        archive = [
            FakeArchiveEntry(f"coding/skill-{i}", 0.5, generation_index=i)
            for i in range(6)
        ]
        summary = _format_archive_summary(archive, recent_n=2)
        # newest-first: indices 5 and 4 included, 0-3 not
        assert "coding/skill-5" in summary
        assert "coding/skill-4" in summary
        assert "coding/skill-0" not in summary

    def test_approve_with_archive_context(self):
        parent = FakeSkill(skill_md=SAMPLE_PARENT_MD)
        child = FakeSkill(skill_md=SAMPLE_CHILD_MD)
        archive = [FakeArchiveEntry("coding/skill-a", 0.8, generation_index=0)]
        invoker = ScriptedInvoker([_approve_stdout("Improvement confirmed by archive.")])
        critic = make_codex_skill_critic(MINIMAL_TEMPLATE, codex_invoker=invoker)

        review = critic.review(child, parent, archive)

        assert review.verdict == "approve"
        # Verify archive appeared in prompt
        assert "coding/skill-a" in invoker.prompts_received[0]
