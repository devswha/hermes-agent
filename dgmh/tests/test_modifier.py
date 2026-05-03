"""
dgmh/tests/test_modifier.py — Tests for modifier.py (W3 deliverable).

Covers per W3 spec:
1. Happy path: scripted Codex output → SkillCandidate with correct fields
2. JSON parse fail → re-roll → success on later attempt
3. Scope violation (output mentions tools/skill_manager_tool.py) → re-roll → success on clean output
4. Subprocess failure (non-zero exit) → CodexModifierError raised, NO re-roll
5. Timeout → CodexModifierError raised
6. Persistent scope violation → ScopeAttemptsExhaustedError after max_scope_attempts exhaustion
7. Prompt assembly substitutes {{PARENT_SKILL_MD}}, {{ARCHIVE_SUMMARY}}, {{ATTEMPT}}
8. recent_n window respected (newest-first, limited)
9. Empty archive → prompt contains archive-summary placeholder marker

Reference: playground/dgmh-engine/llmModifier.ts (re-roll semantics)
           playground/dgmh-engine/selfModScope.ts (scope guard intent)
           dgmh/modifier.py (implementation under test)
"""

from __future__ import annotations

import dataclasses
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

from dgmh.modifier import (
    CodexModifierError,
    ScopeAttemptsExhaustedError,
    SkillCandidate,
    SkillModifier,
    _assemble_prompt,
    _check_skill_scope,
    _extract_first_json_object,
    _format_archive_summary,
    _validate_envelope,
    make_codex_skill_modifier,
)


# ---------------------------------------------------------------------------
# Minimal test fixtures
# ---------------------------------------------------------------------------

MINIMAL_TEMPLATE = """\
PARENT:
{{PARENT_SKILL_MD}}
ARCHIVE:
{{ARCHIVE_SUMMARY}}
ATTEMPT:
{{ATTEMPT}}
"""

SAMPLE_SKILL_MD = """\
---
name: test-skill
description: A test skill for testing.
version: "1.0"
tags: [testing]
---
# Test Skill

This skill helps write unit tests.

## Guidelines
- Be precise
- Cover edge cases
"""

SAMPLE_CHILD_SKILL_MD = """\
---
name: test-skill
description: An improved test skill for testing.
version: "1.1"
tags: [testing, quality]
---
# Test Skill (improved)

This skill helps write comprehensive unit tests.

## Guidelines
- Be precise and thorough
- Cover edge cases and happy paths
- Use parameterized tests where appropriate
"""


@dataclass
class FakeParent:
    """Minimal parent object for tests — provides skill_id, skill_md, lineage."""
    skill_id: str = "coding/test-skill"
    skill_md: str = SAMPLE_SKILL_MD
    lineage: list[str] = field(default_factory=list)
    score: float = 0.5
    compiled_children: int = 0
    generation_index: int = 0


@dataclass
class FakeArchiveEntry:
    """Minimal archive entry for archive summary tests."""
    skill_id: str
    score: float
    compiled_children: int = 0
    generation_index: int = 0
    skill_md: str = ""


def _make_valid_envelope(
    skill_id: str = "coding/test-skill",
    lineage: list[str] | None = None,
    skill_md: str = SAMPLE_CHILD_SKILL_MD,
) -> dict[str, Any]:
    """Build a valid JSON envelope dict."""
    if lineage is None:
        lineage = ["coding/test-skill"]
    return {"skill_id": skill_id, "skill_md": skill_md, "lineage": lineage}


def _stdout_from(obj: dict[str, Any]) -> str:
    """Serialize a dict as JSON string (simulates Codex stdout)."""
    return json.dumps(obj)


# ---------------------------------------------------------------------------
# Mock CodexInvoker — injectable via codex_invoker= param
# ---------------------------------------------------------------------------


class ScriptedInvoker:
    """Invoke mock that returns pre-scripted responses in sequence.

    Each call pops the next response from the queue.
    If queue is exhausted, raises CodexModifierError (subprocess failure).
    """

    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = list(responses)
        self.call_count = 0
        self.prompts_received: list[str] = []

    def __call__(self, prompt: str, *, timeout_s: float = 60.0) -> str:
        self.call_count += 1
        self.prompts_received.append(prompt)
        if not self._responses:
            raise CodexModifierError("ScriptedInvoker: queue exhausted")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_returns_skill_candidate(self):
        """Happy path: valid Codex output → SkillCandidate with correct fields."""
        parent = FakeParent(skill_id="coding/test-skill", lineage=[])
        envelope = _make_valid_envelope(
            skill_id="coding/test-skill",
            lineage=["coding/test-skill"],
        )
        invoker = ScriptedInvoker([_stdout_from(envelope)])
        modifier = make_codex_skill_modifier(
            MINIMAL_TEMPLATE, codex_invoker=invoker, max_scope_attempts=3
        )

        candidate = modifier.modify(parent, [])

        assert isinstance(candidate, SkillCandidate)
        assert candidate.skill_id == "coding/test-skill"
        assert candidate.skill_md == SAMPLE_CHILD_SKILL_MD
        assert candidate.lineage == ["coding/test-skill"]
        assert candidate.attempt == 0
        assert candidate.raw_codex_output != ""

    def test_lineage_includes_parent_id(self):
        """lineage must end with parent.skill_id."""
        parent = FakeParent(skill_id="research/summarize", lineage=["root/base"])
        envelope = {
            "skill_id": "research/summarize",
            "skill_md": SAMPLE_CHILD_SKILL_MD,
            "lineage": ["root/base", "research/summarize"],
        }
        invoker = ScriptedInvoker([_stdout_from(envelope)])
        modifier = make_codex_skill_modifier(MINIMAL_TEMPLATE, codex_invoker=invoker)

        candidate = modifier.modify(parent, [])

        assert candidate.lineage[-1] == "research/summarize"
        assert candidate.lineage == ["root/base", "research/summarize"]

    def test_skill_id_has_cat_slash_name(self):
        """skill_id must be <cat>/<name> format."""
        parent = FakeParent(skill_id="coding/test-skill", lineage=[])
        envelope = _make_valid_envelope(skill_id="coding/test-skill", lineage=["coding/test-skill"])
        invoker = ScriptedInvoker([_stdout_from(envelope)])
        modifier = make_codex_skill_modifier(MINIMAL_TEMPLATE, codex_invoker=invoker)

        candidate = modifier.modify(parent, [])

        parts = candidate.skill_id.split("/")
        assert len(parts) == 2
        assert all(parts)


# ---------------------------------------------------------------------------
# 2. JSON parse fail → re-roll → success
# ---------------------------------------------------------------------------

class TestJsonParseFailReroll:
    def test_reroll_on_parse_failure(self):
        """First attempt returns unparseable text; second returns valid envelope."""
        parent = FakeParent(skill_id="coding/test-skill", lineage=[])
        envelope = _make_valid_envelope(lineage=["coding/test-skill"])
        invoker = ScriptedInvoker([
            "This is not JSON at all — prose response.",
            _stdout_from(envelope),
        ])
        modifier = make_codex_skill_modifier(
            MINIMAL_TEMPLATE, codex_invoker=invoker, max_scope_attempts=3
        )

        candidate = modifier.modify(parent, [])

        assert candidate.skill_id == "coding/test-skill"
        assert invoker.call_count == 2
        assert candidate.attempt == 1  # 0-indexed, succeeded on attempt 1

    def test_reroll_on_invalid_envelope(self):
        """First attempt returns JSON but missing required skill_id field."""
        parent = FakeParent(skill_id="coding/test-skill", lineage=[])
        bad_envelope = {"result": "some output", "lineage": ["coding/test-skill"]}
        good_envelope = _make_valid_envelope(lineage=["coding/test-skill"])
        invoker = ScriptedInvoker([
            _stdout_from(bad_envelope),
            _stdout_from(good_envelope),
        ])
        modifier = make_codex_skill_modifier(
            MINIMAL_TEMPLATE, codex_invoker=invoker, max_scope_attempts=3
        )

        candidate = modifier.modify(parent, [])

        assert candidate.skill_id == "coding/test-skill"
        assert invoker.call_count == 2


# ---------------------------------------------------------------------------
# 3. Scope violation → re-roll → success
# ---------------------------------------------------------------------------

class TestScopeViolationReroll:
    def test_reroll_on_scope_violation_then_success(self):
        """Scope violation (tools/ mention) → re-roll → clean output succeeds."""
        parent = FakeParent(skill_id="coding/test-skill", lineage=[])
        dirty_skill_md = SAMPLE_CHILD_SKILL_MD + "\n\nSee also: tools/skill_manager_tool.py"
        dirty_envelope = _make_valid_envelope(
            skill_md=dirty_skill_md,
            lineage=["coding/test-skill"],
        )
        clean_envelope = _make_valid_envelope(lineage=["coding/test-skill"])
        invoker = ScriptedInvoker([
            _stdout_from(dirty_envelope),
            _stdout_from(clean_envelope),
        ])
        modifier = make_codex_skill_modifier(
            MINIMAL_TEMPLATE, codex_invoker=invoker, max_scope_attempts=3
        )

        candidate = modifier.modify(parent, [])

        assert "tools/" not in candidate.skill_md
        assert invoker.call_count == 2
        assert candidate.attempt == 1

    def test_multiple_scope_patterns_detected(self):
        """gateway/ and agent/ mentions both trigger scope rejection."""
        violations = _check_skill_scope("This mentions gateway/ and agent/ paths")
        assert any("gateway/" in v for v in violations)
        assert any("agent/" in v for v in violations)

    def test_clean_skill_md_passes_scope(self):
        """Normal SKILL.md content with no denied paths passes scope check."""
        violations = _check_skill_scope(SAMPLE_CHILD_SKILL_MD)
        assert violations == []


# ---------------------------------------------------------------------------
# 4. Subprocess failure → CodexModifierError, NO re-roll
# ---------------------------------------------------------------------------

class TestSubprocessFailure:
    def test_nonzero_exit_raises_no_reroll(self):
        """Non-zero subprocess exit → CodexModifierError raised; no re-roll."""
        parent = FakeParent(skill_id="coding/test-skill", lineage=[])
        invoker = ScriptedInvoker([
            CodexModifierError("exit 1", exit_code=1, stderr="error"),
        ])
        modifier = make_codex_skill_modifier(
            MINIMAL_TEMPLATE, codex_invoker=invoker, max_scope_attempts=3
        )

        with pytest.raises(CodexModifierError) as exc_info:
            modifier.modify(parent, [])

        assert exc_info.value.exit_code == 1
        # Only 1 call: no re-roll on subprocess failure
        assert invoker.call_count == 1

    def test_subprocess_failure_via_patch(self):
        """subprocess.run non-zero exit raises CodexModifierError (integration path)."""
        parent = FakeParent(skill_id="coding/test-skill", lineage=[])
        # Use None codex_invoker to go through _invoke_codex
        modifier = make_codex_skill_modifier(
            MINIMAL_TEMPLATE, codex_invoker=None, max_scope_attempts=3
        )

        fake_result = subprocess.CompletedProcess(
            args=["codex", "exec", "-"], returncode=1
        )
        fake_result.stdout = ""
        fake_result.stderr = "error output"

        with patch("dgmh.modifier.subprocess.run", return_value=fake_result):
            with pytest.raises(CodexModifierError) as exc_info:
                modifier.modify(parent, [])

        assert exc_info.value.exit_code == 1


# ---------------------------------------------------------------------------
# 5. Timeout → CodexModifierError
# ---------------------------------------------------------------------------

class TestTimeout:
    def test_timeout_raises_codex_modifier_error(self):
        """Timeout → CodexModifierError with timed_out=True."""
        parent = FakeParent(skill_id="coding/test-skill", lineage=[])
        invoker = ScriptedInvoker([
            CodexModifierError("timed out", timed_out=True, exit_code=None),
        ])
        modifier = make_codex_skill_modifier(
            MINIMAL_TEMPLATE, codex_invoker=invoker, max_scope_attempts=3
        )

        with pytest.raises(CodexModifierError) as exc_info:
            modifier.modify(parent, [])

        assert exc_info.value.timed_out is True
        assert invoker.call_count == 1  # no re-roll

    def test_timeout_via_patch(self):
        """subprocess.TimeoutExpired → CodexModifierError timed_out=True (integration path)."""
        parent = FakeParent(skill_id="coding/test-skill", lineage=[])
        modifier = make_codex_skill_modifier(
            MINIMAL_TEMPLATE, codex_invoker=None, max_scope_attempts=3
        )

        with patch(
            "dgmh.modifier.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["codex"], timeout=60),
        ):
            with pytest.raises(CodexModifierError) as exc_info:
                modifier.modify(parent, [])

        assert exc_info.value.timed_out is True


# ---------------------------------------------------------------------------
# 6. Persistent scope violation → ScopeAttemptsExhaustedError
# ---------------------------------------------------------------------------

class TestScopeAttemptsExhausted:
    def test_exhausted_after_max_attempts(self):
        """All attempts produce scope violations → ScopeAttemptsExhaustedError."""
        parent = FakeParent(skill_id="coding/test-skill", lineage=[])
        dirty_skill_md = SAMPLE_CHILD_SKILL_MD + "\nSee: tools/skill_manager_tool.py"
        dirty_envelope = _make_valid_envelope(
            skill_md=dirty_skill_md,
            lineage=["coding/test-skill"],
        )
        # All 3 attempts return scope-violating output
        invoker = ScriptedInvoker([
            _stdout_from(dirty_envelope),
            _stdout_from(dirty_envelope),
            _stdout_from(dirty_envelope),
        ])
        modifier = make_codex_skill_modifier(
            MINIMAL_TEMPLATE, codex_invoker=invoker, max_scope_attempts=3
        )

        with pytest.raises(ScopeAttemptsExhaustedError) as exc_info:
            modifier.modify(parent, [])

        assert exc_info.value.attempts == 3
        assert exc_info.value.parent_id == "coding/test-skill"
        assert invoker.call_count == 3

    def test_exhausted_on_persistent_parse_failure(self):
        """All attempts return unparseable output → ScopeAttemptsExhaustedError."""
        parent = FakeParent(skill_id="coding/test-skill", lineage=[])
        invoker = ScriptedInvoker(["not json", "also not json", "still not json"])
        modifier = make_codex_skill_modifier(
            MINIMAL_TEMPLATE, codex_invoker=invoker, max_scope_attempts=3
        )

        with pytest.raises(ScopeAttemptsExhaustedError) as exc_info:
            modifier.modify(parent, [])

        assert exc_info.value.attempts == 3

    def test_custom_max_scope_attempts_respected(self):
        """max_scope_attempts=2 exhausts after exactly 2 attempts."""
        parent = FakeParent(skill_id="coding/test-skill", lineage=[])
        invoker = ScriptedInvoker(["not json", "not json"])
        modifier = make_codex_skill_modifier(
            MINIMAL_TEMPLATE, codex_invoker=invoker, max_scope_attempts=2
        )

        with pytest.raises(ScopeAttemptsExhaustedError):
            modifier.modify(parent, [])

        assert invoker.call_count == 2


# ---------------------------------------------------------------------------
# 7. Prompt assembly substitution
# ---------------------------------------------------------------------------

class TestPromptAssembly:
    def test_parent_skill_md_substituted(self):
        """{{PARENT_SKILL_MD}} is replaced with parent's SKILL.md content."""
        parent = FakeParent(skill_id="coding/test-skill", skill_md=SAMPLE_SKILL_MD)
        prompt = _assemble_prompt(MINIMAL_TEMPLATE, parent, [], 5, 0)
        assert "{{PARENT_SKILL_MD}}" not in prompt
        assert SAMPLE_SKILL_MD in prompt

    def test_archive_summary_substituted(self):
        """{{ARCHIVE_SUMMARY}} is replaced with formatted archive text."""
        parent = FakeParent(skill_id="coding/test-skill")
        archive = [
            FakeArchiveEntry(skill_id="coding/a", score=0.8, generation_index=1),
            FakeArchiveEntry(skill_id="coding/b", score=0.6, generation_index=0),
        ]
        prompt = _assemble_prompt(MINIMAL_TEMPLATE, parent, archive, 5, 0)
        assert "{{ARCHIVE_SUMMARY}}" not in prompt
        assert "coding/a" in prompt
        assert "coding/b" in prompt

    def test_attempt_substituted(self):
        """{{ATTEMPT}} is replaced with the attempt index string."""
        parent = FakeParent(skill_id="coding/test-skill")
        prompt = _assemble_prompt(MINIMAL_TEMPLATE, parent, [], 5, 2)
        assert "{{ATTEMPT}}" not in prompt
        assert "2" in prompt

    def test_all_placeholders_replaced(self):
        """All three placeholders are replaced in a single call."""
        parent = FakeParent(skill_id="coding/test-skill")
        prompt = _assemble_prompt(MINIMAL_TEMPLATE, parent, [], 5, 0)
        assert "{{PARENT_SKILL_MD}}" not in prompt
        assert "{{ARCHIVE_SUMMARY}}" not in prompt
        assert "{{ATTEMPT}}" not in prompt


# ---------------------------------------------------------------------------
# 8. recent_n window respected
# ---------------------------------------------------------------------------

class TestRecentN:
    def test_recent_n_limits_archive_entries(self):
        """recent_n=2 includes only 2 newest entries in archive summary."""
        archive = [
            FakeArchiveEntry(skill_id=f"coding/skill-{i}", score=0.5, generation_index=i)
            for i in range(5)
        ]
        summary = _format_archive_summary(archive, recent_n=2)
        # newest-first: indices 4 and 3 should appear, 0,1,2 should not
        assert "coding/skill-4" in summary
        assert "coding/skill-3" in summary
        assert "coding/skill-0" not in summary
        assert "coding/skill-1" not in summary
        assert "coding/skill-2" not in summary

    def test_recent_n_newest_first(self):
        """Archive summary entries are ordered newest-first."""
        archive = [
            FakeArchiveEntry(skill_id="coding/old", score=0.3, generation_index=0),
            FakeArchiveEntry(skill_id="coding/new", score=0.7, generation_index=5),
        ]
        summary = _format_archive_summary(archive, recent_n=5)
        idx_new = summary.index("coding/new")
        idx_old = summary.index("coding/old")
        assert idx_new < idx_old, "Newer entry should appear before older entry"

    def test_recent_n_zero_returns_empty_marker(self):
        """recent_n=0 returns empty-archive marker regardless of archive content."""
        archive = [FakeArchiveEntry(skill_id="coding/x", score=0.5, generation_index=0)]
        summary = _format_archive_summary(archive, recent_n=0)
        assert "empty archive" in summary.lower() or "(empty" in summary


# ---------------------------------------------------------------------------
# 9. Empty archive → placeholder marker in prompt
# ---------------------------------------------------------------------------

class TestEmptyArchive:
    def test_empty_archive_produces_placeholder(self):
        """Empty archive list → archive summary contains empty-archive marker."""
        summary = _format_archive_summary([], recent_n=5)
        assert "empty archive" in summary.lower() or "(empty" in summary

    def test_empty_archive_still_produces_prompt(self):
        """Even with empty archive, prompt is assembled without errors."""
        parent = FakeParent(skill_id="coding/test-skill")
        prompt = _assemble_prompt(MINIMAL_TEMPLATE, parent, [], 5, 0)
        assert "{{ARCHIVE_SUMMARY}}" not in prompt
        assert len(prompt) > 0

    def test_empty_archive_modifier_succeeds(self):
        """Modifier with empty archive still produces a valid SkillCandidate."""
        parent = FakeParent(skill_id="coding/test-skill", lineage=[])
        envelope = _make_valid_envelope(lineage=["coding/test-skill"])
        invoker = ScriptedInvoker([_stdout_from(envelope)])
        modifier = make_codex_skill_modifier(
            MINIMAL_TEMPLATE, codex_invoker=invoker
        )

        candidate = modifier.modify(parent, [])
        assert candidate.skill_id == "coding/test-skill"


# ---------------------------------------------------------------------------
# Validate envelope — unit tests
# ---------------------------------------------------------------------------

class TestValidateEnvelope:
    def test_valid_envelope_accepted(self):
        raw = {"skill_id": "coding/test", "skill_md": SAMPLE_CHILD_SKILL_MD, "lineage": ["coding/test"]}
        candidate, err = _validate_envelope(raw, "coding/test", [])
        assert candidate is not None
        assert err == ""

    def test_missing_skill_id_rejected(self):
        raw = {"skill_md": SAMPLE_CHILD_SKILL_MD, "lineage": ["coding/test"]}
        candidate, err = _validate_envelope(raw, "coding/test", [])
        assert candidate is None
        assert "skill_id" in err

    def test_skill_id_no_slash_rejected(self):
        raw = {"skill_id": "noslash", "skill_md": SAMPLE_CHILD_SKILL_MD, "lineage": ["noslash"]}
        candidate, err = _validate_envelope(raw, "noslash", [])
        assert candidate is None
        assert "pattern" in err

    def test_empty_skill_md_rejected(self):
        raw = {"skill_id": "coding/test", "skill_md": "", "lineage": ["coding/test"]}
        candidate, err = _validate_envelope(raw, "coding/test", [])
        assert candidate is None
        assert "skill_md" in err

    def test_lineage_not_ending_with_parent_rejected(self):
        raw = {"skill_id": "coding/test", "skill_md": SAMPLE_CHILD_SKILL_MD, "lineage": ["other/skill"]}
        candidate, err = _validate_envelope(raw, "coding/test", [])
        assert candidate is None
        assert "lineage" in err or "parent_id" in err

    def test_empty_lineage_rejected(self):
        raw = {"skill_id": "coding/test", "skill_md": SAMPLE_CHILD_SKILL_MD, "lineage": []}
        candidate, err = _validate_envelope(raw, "coding/test", [])
        assert candidate is None


# ---------------------------------------------------------------------------
# make_codex_skill_modifier — factory validation
# ---------------------------------------------------------------------------

class TestFactory:
    def test_empty_template_raises(self):
        with pytest.raises(ValueError, match="prompt_template"):
            make_codex_skill_modifier("")

    def test_none_template_raises(self):
        with pytest.raises((ValueError, TypeError)):
            make_codex_skill_modifier(None)  # type: ignore[arg-type]

    def test_max_scope_attempts_zero_raises(self):
        with pytest.raises(ValueError, match="max_scope_attempts"):
            make_codex_skill_modifier(MINIMAL_TEMPLATE, max_scope_attempts=0)

    def test_returns_skill_modifier_instance(self):
        modifier = make_codex_skill_modifier(MINIMAL_TEMPLATE)
        assert isinstance(modifier, SkillModifier)
