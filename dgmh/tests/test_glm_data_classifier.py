"""Tests for ``dgmh.glm_data.classifier``.

Covers plan §3 AC5 (taxonomy conformance: baseline + ``_new_*`` discovery)
and plan §4 Step 5 acceptance:

- Mocked GLM → deterministic classification across batches
- ``_new_*`` cap ≤ 5/run + ≥3 exemplars per proposal (R5)
- Per-message no-silent-drop: every input msg_id gets ≥1 category
- Schema-validation failures surface as ``GLMResponseSchemaError``
- Unknown categories that are neither baseline nor ``_new_*`` raise
  :class:`TaxonomyViolationError`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dgmh.glm_data.classifier import (  # noqa: E402
    Classifier,
    MAX_NEW_PROPOSALS_PER_RUN,
    MIN_EXEMPLARS_PER_NEW_PROPOSAL,
    TaxonomyViolationError,
)
from dgmh.glm_data.glm_client import GLMResponseSchemaError  # noqa: E402
from dgmh.glm_data.schema import KakaoMsg, ToneCategory  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeClient:
    """Stand-in for :class:`GLMClient` that returns scripted envelopes."""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, messages, *, model, response_format=None, max_retries=None, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "response_format": response_format,
                "max_retries": max_retries,
            }
        )
        if not self._responses:
            raise AssertionError("no scripted responses left")
        return self._responses.pop(0)


def _envelope(content_dict: dict) -> dict:
    """Wrap classifier content in a chat-completions envelope."""

    return {
        "id": "chatcmpl-test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": json.dumps(content_dict)},
                "finish_reason": "stop",
            }
        ],
    }


def _mk(msg_id: str, text: str = "안녕") -> KakaoMsg:
    return KakaoMsg(
        msg_id=msg_id,
        speaker_id="user_01",
        timestamp="2026-05-12T10:00:00+09:00",
        raw_text=text,
        redacted_text=text,
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
# AC5: deterministic classification across baseline
# ---------------------------------------------------------------------------


def test_classifies_with_baseline_taxonomy_deterministic():
    messages = [_mk("m1", "ㅋㅋㅋ"), _mk("m2", "내일 보자")]
    response = _envelope(
        {
            "classifications": [
                {"msg_id": "m1", "categories": ["ko-communication"], "confidence": 0.9},
                {"msg_id": "m2", "categories": ["structure", "language"], "confidence": 0.7},
            ]
        }
    )
    client = _FakeClient([response])
    classifier = Classifier(client, taxonomy=BASELINE_7, batch_size=20)

    result = classifier.classify(messages)

    assert result.n_calls == 1
    assert result.n_messages == 2
    assert [c.msg_id for c in result.classifications] == ["m1", "m2"]
    # ``ko-`` prefix is stripped to the bare baseline form.
    cats_m1 = [tc.name for tc in result.classifications[0].categories]
    cats_m2 = [tc.name for tc in result.classifications[1].categories]
    assert cats_m1 == ["communication"]
    assert cats_m2 == ["structure", "language"]
    # Target files default to ``patterns/ko-{name}.md`` for baseline.
    assert result.classifications[0].categories[0].target_file == "patterns/ko-communication.md"
    assert result.classifications[0].confidence == pytest.approx(0.9)


def test_classifier_passes_response_format_json_object():
    msgs = [_mk("m1")]
    client = _FakeClient(
        [
            _envelope(
                {"classifications": [{"msg_id": "m1", "categories": ["filler"], "confidence": 0.5}]}
            )
        ]
    )
    Classifier(client, taxonomy=BASELINE_7, batch_size=20).classify(msgs)
    assert client.calls[0]["response_format"] == {"type": "json_object"}


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def test_batches_messages_in_chunks_of_n():
    msgs = [_mk(f"m{i:02d}") for i in range(25)]
    # Two batches with batch_size=20: [m00..m19] then [m20..m24]
    response_a = _envelope(
        {
            "classifications": [
                {"msg_id": f"m{i:02d}", "categories": ["filler"], "confidence": 0.5}
                for i in range(20)
            ]
        }
    )
    response_b = _envelope(
        {
            "classifications": [
                {"msg_id": f"m{i:02d}", "categories": ["filler"], "confidence": 0.5}
                for i in range(20, 25)
            ]
        }
    )
    client = _FakeClient([response_a, response_b])
    result = Classifier(client, taxonomy=BASELINE_7, batch_size=20).classify(msgs)
    assert result.n_calls == 2
    assert len(result.classifications) == 25


# ---------------------------------------------------------------------------
# Plan §4 Step 5 acceptance: no silent drop
# ---------------------------------------------------------------------------


def test_missing_message_gets_filler_placeholder():
    """Plan §4 Step 5 acceptance: every msg gets ≥1 category, never silent drop."""

    msgs = [_mk("m1"), _mk("m2")]
    # GLM forgets m2 entirely.
    response = _envelope(
        {"classifications": [{"msg_id": "m1", "categories": ["communication"], "confidence": 0.9}]}
    )
    client = _FakeClient([response])
    result = Classifier(client, taxonomy=BASELINE_7).classify(msgs)
    by_id = {c.msg_id: c for c in result.classifications}
    assert set(by_id) == {"m1", "m2"}
    assert by_id["m2"].categories == [
        ToneCategory(name="filler", target_file="patterns/ko-filler.md")
    ]
    assert by_id["m2"].confidence == 0.0


def test_empty_categories_synthesises_filler():
    """A msg with ``categories: []`` gets a synthesised filler entry."""

    msgs = [_mk("m1")]
    response = _envelope(
        {"classifications": [{"msg_id": "m1", "categories": [], "confidence": 0.4}]}
    )
    client = _FakeClient([response])
    result = Classifier(client, taxonomy=BASELINE_7).classify(msgs)
    assert result.classifications[0].categories == [
        ToneCategory(name="filler", target_file="patterns/ko-filler.md")
    ]


def test_unknown_msg_id_in_response_is_ignored():
    msgs = [_mk("m1")]
    response = _envelope(
        {
            "classifications": [
                {"msg_id": "m1", "categories": ["filler"], "confidence": 0.5},
                {"msg_id": "ghost-id", "categories": ["filler"], "confidence": 0.5},
            ]
        }
    )
    client = _FakeClient([response])
    result = Classifier(client, taxonomy=BASELINE_7).classify(msgs)
    assert [c.msg_id for c in result.classifications] == ["m1"]


# ---------------------------------------------------------------------------
# R5: _new_* cap + exemplar floor
# ---------------------------------------------------------------------------


def test_new_proposals_capped_at_max_per_run():
    """When GLM proposes more than ``MAX_NEW_PROPOSALS_PER_RUN``, only the top
    N (by exemplar count) survive."""

    total = MAX_NEW_PROPOSALS_PER_RUN + 3  # 8 proposals against a cap of 5
    msgs = [_mk(f"m{i:02d}") for i in range(10)]
    proposals = []
    for i in range(total):
        # Strictly descending exemplar counts ⇒ deterministic priority order.
        n_ex = MIN_EXEMPLARS_PER_NEW_PROPOSAL + (total - i)
        proposals.append(
            {
                "name": f"_new_cat_{i:02d}",
                "exemplar_msg_ids": [f"m{j:02d}" for j in range(min(n_ex, 10))],
                "candidate_target_file": "patterns/ko-content.md",
                "rationale": "x",
            }
        )

    response = _envelope(
        {
            "classifications": [
                {"msg_id": f"m{i:02d}", "categories": ["filler"], "confidence": 0.5}
                for i in range(10)
            ],
            "_new_proposals": proposals,
        }
    )
    client = _FakeClient([response])
    result = Classifier(client, taxonomy=BASELINE_7).classify(msgs)
    assert len(result.new_proposals) == MAX_NEW_PROPOSALS_PER_RUN
    # The five kept proposals must be the five with the most exemplars (first
    # five in the input, since we built them strictly-descending by count).
    assert [p.name for p in result.new_proposals] == [
        f"_new_cat_{i:02d}" for i in range(MAX_NEW_PROPOSALS_PER_RUN)
    ]


def test_new_proposals_below_exemplar_floor_dropped():
    msgs = [_mk("m1"), _mk("m2")]
    too_few_exemplars = ["m1"]  # 1 < MIN_EXEMPLARS_PER_NEW_PROPOSAL
    response = _envelope(
        {
            "classifications": [
                {"msg_id": "m1", "categories": ["filler"], "confidence": 0.5},
                {"msg_id": "m2", "categories": ["filler"], "confidence": 0.5},
            ],
            "_new_proposals": [
                {
                    "name": "_new_underfunded",
                    "exemplar_msg_ids": too_few_exemplars,
                    "candidate_target_file": "patterns/ko-content.md",
                    "rationale": "not enough exemplars",
                }
            ],
        }
    )
    client = _FakeClient([response])
    result = Classifier(client, taxonomy=BASELINE_7).classify(msgs)
    assert result.new_proposals == []  # dropped by R5 floor


def test_new_proposal_with_enough_exemplars_kept():
    msgs = [_mk(f"m{i}") for i in range(1, 5)]
    response = _envelope(
        {
            "classifications": [
                {"msg_id": f"m{i}", "categories": ["filler"], "confidence": 0.5} for i in range(1, 5)
            ],
            "_new_proposals": [
                {
                    "name": "_new_chimaek_register",
                    "exemplar_msg_ids": ["m1", "m2", "m3"],
                    "candidate_target_file": "patterns/ko-communication.md",
                    "rationale": "casual food banter",
                }
            ],
        }
    )
    client = _FakeClient([response])
    result = Classifier(client, taxonomy=BASELINE_7).classify(msgs)
    assert len(result.new_proposals) == 1
    proposal = result.new_proposals[0]
    assert proposal.name == "_new_chimaek_register"
    assert proposal.exemplar_msg_ids == ["m1", "m2", "m3"]
    assert proposal.candidate_target_file == "patterns/ko-communication.md"


def test_new_proposals_merge_across_batches():
    """Same ``_new_*`` name across two batches keeps the union of exemplars."""

    msgs = [_mk(f"m{i:02d}") for i in range(40)]
    a = _envelope(
        {
            "classifications": [
                {"msg_id": f"m{i:02d}", "categories": ["filler"], "confidence": 0.5}
                for i in range(20)
            ],
            "_new_proposals": [
                {
                    "name": "_new_meme",
                    "exemplar_msg_ids": ["m00", "m01"],
                    "candidate_target_file": "patterns/ko-viral-hook.md",
                    "rationale": "x",
                }
            ],
        }
    )
    b = _envelope(
        {
            "classifications": [
                {"msg_id": f"m{i:02d}", "categories": ["filler"], "confidence": 0.5}
                for i in range(20, 40)
            ],
            "_new_proposals": [
                {
                    "name": "_new_meme",
                    "exemplar_msg_ids": ["m20", "m21"],
                    "candidate_target_file": "patterns/ko-viral-hook.md",
                    "rationale": "x",
                }
            ],
        }
    )
    client = _FakeClient([a, b])
    result = Classifier(client, taxonomy=BASELINE_7, batch_size=20).classify(msgs)
    assert len(result.new_proposals) == 1
    assert result.new_proposals[0].exemplar_msg_ids == ["m00", "m01", "m20", "m21"]


def test_new_proposal_cited_in_categories_carries_target_file():
    msgs = [_mk("m1"), _mk("m2"), _mk("m3")]
    response = _envelope(
        {
            "classifications": [
                {
                    "msg_id": "m1",
                    "categories": ["_new_chimaek_register"],
                    "confidence": 0.8,
                },
                {"msg_id": "m2", "categories": ["filler"], "confidence": 0.5},
                {"msg_id": "m3", "categories": ["filler"], "confidence": 0.5},
            ],
            "_new_proposals": [
                {
                    "name": "_new_chimaek_register",
                    "exemplar_msg_ids": ["m1", "m2", "m3"],
                    "candidate_target_file": "patterns/ko-communication.md",
                    "rationale": "casual food banter",
                }
            ],
        }
    )
    client = _FakeClient([response])
    result = Classifier(client, taxonomy=BASELINE_7).classify(msgs)
    m1 = result.classifications[0]
    assert m1.categories == [
        ToneCategory(name="_new_chimaek_register", target_file="patterns/ko-communication.md")
    ]


def test_orphan_new_category_without_proposal_drops_category_but_keeps_msg():
    """``_new_X`` referenced in categories with no matching proposal: drop the
    category, keep the message with a synthesised filler so we honour the
    'no silent drop' invariant without inventing a target file out of nothing.
    """

    msgs = [_mk("m1")]
    response = _envelope(
        {
            "classifications": [
                {"msg_id": "m1", "categories": ["_new_orphan"], "confidence": 0.7}
            ]
        }
    )
    client = _FakeClient([response])
    result = Classifier(client, taxonomy=BASELINE_7).classify(msgs)
    assert result.classifications[0].categories == [
        ToneCategory(name="filler", target_file="patterns/ko-filler.md")
    ]


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_unknown_baseline_category_raises_taxonomy_violation():
    msgs = [_mk("m1")]
    response = _envelope(
        {"classifications": [{"msg_id": "m1", "categories": ["banana"], "confidence": 0.5}]}
    )
    client = _FakeClient([response])
    with pytest.raises(TaxonomyViolationError, match="banana"):
        Classifier(client, taxonomy=BASELINE_7).classify(msgs)


def test_non_json_content_raises_schema_error():
    msgs = [_mk("m1")]
    bad_envelope = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "not json"}}],
    }
    client = _FakeClient([bad_envelope])
    with pytest.raises(GLMResponseSchemaError):
        Classifier(client, taxonomy=BASELINE_7).classify(msgs)


def test_schema_violation_raises():
    msgs = [_mk("m1")]
    # Missing required ``categories`` field.
    bad_content = {"classifications": [{"msg_id": "m1", "confidence": 0.5}]}
    response = _envelope(bad_content)
    client = _FakeClient([response])
    with pytest.raises(GLMResponseSchemaError):
        Classifier(client, taxonomy=BASELINE_7).classify(msgs)


# ---------------------------------------------------------------------------
# Promoted ``_new_*`` taxonomy works as first-class baseline (AC5)
# ---------------------------------------------------------------------------


def test_promoted_new_category_treated_as_first_class():
    """A previously-promoted ``_new_*`` is a valid baseline target."""

    promoted = BASELINE_7 + ["_new_chimaek_register"]
    msgs = [_mk("m1")]
    response = _envelope(
        {
            "classifications": [
                {
                    "msg_id": "m1",
                    "categories": ["_new_chimaek_register"],
                    "confidence": 0.8,
                }
            ],
            "_new_proposals": [
                {
                    "name": "_new_chimaek_register",
                    "exemplar_msg_ids": ["m1", "m1", "m1"],
                    "candidate_target_file": "patterns/ko-communication.md",
                }
            ],
        }
    )
    client = _FakeClient([response])
    result = Classifier(client, taxonomy=promoted).classify(msgs)
    cat = result.classifications[0].categories[0]
    assert cat.name == "_new_chimaek_register"
    assert cat.target_file == "patterns/ko-communication.md"
