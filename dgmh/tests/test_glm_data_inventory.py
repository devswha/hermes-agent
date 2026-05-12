"""Tests for ``dgmh.glm_data.inventory``.

Covers plan §4 Step 6 + V7:

- Deterministic markdown output given fixed classifier output.
- Categories rendered in taxonomy order; missing categories get the
  ``_(no coverage)_`` marker.
- Redacted exemplars are sourced from ``KakaoMsg.redacted_text`` only —
  raw PII strings injected into ``raw_text`` must never leak into the
  rendered markdown.
- ``_new_*`` proposals appear with rationale and target file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dgmh.glm_data.classifier import ClassifyRunResult, NewProposal  # noqa: E402
from dgmh.glm_data.inventory import (  # noqa: E402
    NO_COVERAGE_MARKER,
    generate_inventory,
)
from dgmh.glm_data.schema import Classification, KakaoMsg, ToneCategory  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_msg(msg_id: str, *, redacted: str, raw: str | None = None) -> KakaoMsg:
    return KakaoMsg(
        msg_id=msg_id,
        speaker_id="user_01",
        timestamp="2026-05-12T10:00:00+09:00",
        raw_text=raw if raw is not None else redacted,
        redacted_text=redacted,
    )


def _mk_classification(
    msg_id: str,
    category_names: list[str],
    *,
    confidence: float = 0.8,
) -> Classification:
    return Classification(
        msg_id=msg_id,
        categories=[
            ToneCategory(name=name, target_file=f"patterns/ko-{name}.md" if not name.startswith("_new_") else "patterns/ko-content.md")
            for name in category_names
        ],
        confidence=confidence,
    )


BASELINE_7 = [
    "communication",
    "content",
    "filler",
    "language",
    "structure",
    "style",
    "viral-hook",
]


# ---------------------------------------------------------------------------
# AC3: deterministic markdown output
# ---------------------------------------------------------------------------


def test_inventory_deterministic_given_fixed_input():
    messages = [
        _mk_msg("m1", redacted="안녕하세요"),
        _mk_msg("m2", redacted="ㅋㅋㅋ"),
        _mk_msg("m3", redacted="내일 만나요"),
    ]
    classifications = [
        _mk_classification("m1", ["communication"]),
        _mk_classification("m2", ["filler", "communication"]),
        _mk_classification("m3", ["structure"]),
    ]
    result = ClassifyRunResult(
        classifications=classifications,
        new_proposals=[],
        n_calls=1,
        n_messages=3,
        taxonomy=list(BASELINE_7),
    )

    md1 = generate_inventory(result, messages)
    md2 = generate_inventory(result, messages)

    assert md1 == md2
    # Each baseline category appears in the summary table, in taxonomy order.
    for name in BASELINE_7:
        assert f"`{name}`" in md1


def test_inventory_renders_no_coverage_marker_for_empty_categories():
    messages = [_mk_msg("m1", redacted="hi")]
    classifications = [_mk_classification("m1", ["communication"])]
    result = ClassifyRunResult(
        classifications=classifications,
        new_proposals=[],
        n_calls=1,
        n_messages=1,
        taxonomy=list(BASELINE_7),
    )
    md = generate_inventory(result, messages)
    # `viral-hook` has no hits — must show the no-coverage marker in its
    # examples block.
    viral_block = md.split("### `viral-hook`", 1)[1].split("###", 1)[0]
    assert NO_COVERAGE_MARKER in viral_block


def test_inventory_summary_table_counts_match_classifier_output():
    """``count`` for category X equals the number of classifications that
    listed X among their categories (deduplicated per message)."""

    messages = [_mk_msg(f"m{i}", redacted=f"text-{i}") for i in range(5)]
    classifications = [
        _mk_classification("m0", ["communication"]),
        _mk_classification("m1", ["communication"]),
        _mk_classification("m2", ["filler", "communication"]),  # 'communication' counted once for m2
        _mk_classification("m3", ["filler"]),
        _mk_classification("m4", ["structure"]),
    ]
    result = ClassifyRunResult(
        classifications=classifications,
        new_proposals=[],
        n_calls=1,
        n_messages=5,
        taxonomy=list(BASELINE_7),
    )
    md = generate_inventory(result, messages)

    # `communication` has 3 hits, `filler` has 2, `structure` has 1.
    assert "| `communication` | 3 | 60.00% |" in md
    assert "| `filler` | 2 | 40.00% |" in md
    assert "| `structure` | 1 | 20.00% |" in md


# ---------------------------------------------------------------------------
# V7: PII / redaction safety
# ---------------------------------------------------------------------------


def test_inventory_uses_only_redacted_text_never_raw():
    """A KakaoMsg whose ``raw_text`` contains a phone number must never leak;
    the rendered markdown must contain only the ``redacted_text``."""

    raw_with_pii = "안녕 홍길동 010-1234-5678 입니다"
    redacted = "안녕 [REDACTED_NAME] [REDACTED_PHONE] 입니다"
    messages = [_mk_msg("m1", redacted=redacted, raw=raw_with_pii)]
    classifications = [_mk_classification("m1", ["communication"])]
    result = ClassifyRunResult(
        classifications=classifications,
        new_proposals=[],
        n_calls=1,
        n_messages=1,
        taxonomy=list(BASELINE_7),
    )
    md = generate_inventory(result, messages)

    assert redacted in md
    assert "010-1234-5678" not in md
    assert "홍길동" not in md


def test_inventory_caps_examples_per_category():
    """Default 3 examples per category — extras are not rendered."""

    messages = [_mk_msg(f"m{i}", redacted=f"hello-{i}") for i in range(10)]
    classifications = [_mk_classification(f"m{i}", ["communication"]) for i in range(10)]
    result = ClassifyRunResult(
        classifications=classifications,
        new_proposals=[],
        n_calls=1,
        n_messages=10,
        taxonomy=list(BASELINE_7),
    )
    md = generate_inventory(result, messages)

    comm_block = md.split("### `communication`", 1)[1].split("###", 1)[0]
    assert "hello-0" in comm_block
    assert "hello-1" in comm_block
    assert "hello-2" in comm_block
    # 4th and later examples must not appear in the section.
    assert "hello-3" not in comm_block
    assert "hello-9" not in comm_block


# ---------------------------------------------------------------------------
# New proposal section
# ---------------------------------------------------------------------------


def test_inventory_renders_new_proposal_section():
    messages = [
        _mk_msg("m1", redacted="치맥 콜?"),
        _mk_msg("m2", redacted="ㄱㄱ 치맥"),
        _mk_msg("m3", redacted="오늘 치맥 ㄱ?"),
    ]
    classifications = [_mk_classification(f"m{i}", ["communication"]) for i in range(1, 4)]
    proposal = NewProposal(
        name="_new_chimaek_register",
        exemplar_msg_ids=["m1", "m2", "m3"],
        candidate_target_file="patterns/ko-communication.md",
        rationale="casual food-and-drink banter not covered by baseline",
    )
    result = ClassifyRunResult(
        classifications=classifications,
        new_proposals=[proposal],
        n_calls=1,
        n_messages=3,
        taxonomy=BASELINE_7 + ["_new_chimaek_register"],
    )
    md = generate_inventory(result, messages)

    assert "### `_new_chimaek_register`" in md
    assert "patterns/ko-communication.md" in md
    assert "casual food-and-drink banter not covered by baseline" in md
    # Exemplar bodies present in proposals section.
    proposals_section = md.split("## New category proposals", 1)[1]
    assert "치맥 콜?" in proposals_section
    assert "ㄱㄱ 치맥" in proposals_section


def test_inventory_handles_empty_new_proposals_gracefully():
    messages = [_mk_msg("m1", redacted="hi")]
    classifications = [_mk_classification("m1", ["communication"])]
    result = ClassifyRunResult(
        classifications=classifications,
        new_proposals=[],
        n_calls=1,
        n_messages=1,
        taxonomy=list(BASELINE_7),
    )
    md = generate_inventory(result, messages)
    assert "_No new category proposals this run._" in md


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_inventory_handles_zero_messages():
    result = ClassifyRunResult(
        classifications=[],
        new_proposals=[],
        n_calls=0,
        n_messages=0,
        taxonomy=list(BASELINE_7),
    )
    md = generate_inventory(result, [])
    # No crash; every baseline category appears with count=0.
    for name in BASELINE_7:
        assert f"| `{name}` | 0 | 0.00% |" in md


def test_inventory_promoted_categories_appear_after_baseline():
    """Promoted ``_new_*`` rows render after baseline rows in summary order."""

    messages = [_mk_msg("m1", redacted="hi"), _mk_msg("m2", redacted="hello")]
    classifications = [
        _mk_classification("m1", ["communication"]),
        _mk_classification("m2", ["_new_promoted"]),
    ]
    result = ClassifyRunResult(
        classifications=classifications,
        new_proposals=[],
        n_calls=1,
        n_messages=2,
        taxonomy=BASELINE_7 + ["_new_promoted"],
    )
    md = generate_inventory(result, messages)
    # Scope to the summary table: every category is listed there exactly
    # once, in taxonomy order.
    table_block = md.split("## Summary", 1)[1].split("##", 1)[0]
    viral_idx = table_block.find("`viral-hook`")
    promoted_idx = table_block.find("`_new_promoted`")
    assert 0 < viral_idx < promoted_idx


def test_inventory_examples_first_seen_order():
    """Exemplar ids per category appear in corpus order, not classification order."""

    messages = [
        _mk_msg("alpha", redacted="A"),
        _mk_msg("bravo", redacted="B"),
        _mk_msg("charlie", redacted="C"),
        _mk_msg("delta", redacted="D"),
    ]
    # Classify in scrambled order.
    classifications = [
        _mk_classification("delta", ["communication"]),
        _mk_classification("bravo", ["communication"]),
        _mk_classification("alpha", ["communication"]),
        _mk_classification("charlie", ["communication"]),
    ]
    result = ClassifyRunResult(
        classifications=classifications,
        new_proposals=[],
        n_calls=1,
        n_messages=4,
        taxonomy=list(BASELINE_7),
    )
    md = generate_inventory(result, messages)
    comm_block = md.split("### `communication`", 1)[1].split("###", 1)[0]
    # First three exemplars by classification iteration order (delta, bravo, alpha).
    first = comm_block.find("delta")
    second = comm_block.find("bravo")
    third = comm_block.find("alpha")
    assert 0 < first < second < third
    # 4th classification ('charlie') is past the cap and excluded.
    assert "charlie" not in comm_block
