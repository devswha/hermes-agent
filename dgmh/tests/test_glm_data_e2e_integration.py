"""End-to-end integration test for DGM-H GLM Data Collection v1.

Codex review test-gap regression: prior to this file, no test exercised the
full
``parser → classifier → inventory + gap_analyzer + anchor_emit + patina_emit``
chain on a single synthetic corpus. The CRITICAL-3 ``KakaoMsg`` schema split
(parser local dataclass vs. ``schema.KakaoMsg``) would have surfaced here in
seconds. This test is the regression gate: any module that breaks the
canonical ``schema.KakaoMsg`` contract must fail here before it can ship.

Flow:

1. Synthesize a 12-message kakao_export.txt covering varied register, one
   PII case, one short reaction, one URL, one multi-line continuation.
2. ``parse_kakao_file`` → ``list[schema.KakaoMsg]``.
3. ``Classifier(client=_FakeGLMClient)`` → ``ClassifyRunResult``.
4. ``inventory.generate_inventory(result, messages)`` → markdown blob.
5. ``gap_analyzer.analyze_gaps(result, patina_coverage=<stub>)`` →
   ``GapAnalysis`` + ``render_gap_report``.
6. ``AnchorEmitter(StagingWriter).emit(messages, classifications=<by_id>)``
   → 3 files written under the tmp staging session.
7. ``patina_emit.flat_pattern_namespace(...)`` → confirm the language-scoped
   flat namespace is intact (J1).

Assertions cover: (a) no ``AttributeError``/``TypeError`` on field access
across modules, (b) every ``KakaoMsg`` exposes the canonical 5 fields, (c)
``AnchorEmitter`` sidecar rows mirror ``msg_id``/``speaker_id``/``timestamp``
from the parser, (d) flat-namespace numbering stays language-scoped.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dgmh.glm_data.anchor_emit import AnchorEmitter  # noqa: E402
from dgmh.glm_data.classifier import Classifier  # noqa: E402
from dgmh.glm_data.gap_analyzer import (  # noqa: E402
    PatinaCoverage,
    analyze_gaps,
    render_gap_report,
)
from dgmh.glm_data.inventory import generate_inventory  # noqa: E402
from dgmh.glm_data.kakao_parser import parse_kakao_file  # noqa: E402
from dgmh.glm_data.patina_emit import flat_pattern_namespace  # noqa: E402
from dgmh.glm_data.schema import KakaoMsg  # noqa: E402
from dgmh.glm_data.staging_writer import StagingWriter  # noqa: E402


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class _FakeGLMClient:
    """Stand-in for ``GLMClient`` that returns a single scripted envelope.

    The envelope echoes every input ``msg_id`` with a deterministic baseline
    category so the classifier can build a ``ClassifyRunResult`` without any
    network access.
    """

    def __init__(self, category: str = "communication") -> None:
        self._category = category
        self.calls: list[dict] = []

    def chat(
        self,
        messages,
        *,
        model,
        response_format=None,
        max_retries=None,
        **kwargs,
    ):
        self.calls.append({"model": model, "messages": messages})
        # The classifier passes the user-payload as one JSON-per-line block.
        user_payload = messages[-1]["content"]
        msg_ids = [
            json.loads(line)["msg_id"]
            for line in user_payload.splitlines()
            if line.strip()
        ]
        content = {
            "classifications": [
                {
                    "msg_id": mid,
                    "categories": [self._category],
                    "confidence": 0.85,
                }
                for mid in msg_ids
            ]
        }
        return {
            "id": "chatcmpl-e2e",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(content),
                    },
                    "finish_reason": "stop",
                }
            ],
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_KAKAO_BODY = (
    "테스트방 님과 카카오톡 대화\n"
    "저장한 날짜 : 2026-05-12 12:00:00\n"
    "\n"
    "--------------- 2026년 5월 1일 목요일 ---------------\n"
    "[알파] [오전 9:20] 안녕하세요\n"
    "[베타] [오전 9:21] 안녕!\n"
    "[감마] [오전 9:22] ㅋㅋ\n"
    "[알파] [오전 9:25] 오늘 일정 공유합니다\n"
    "이건 같은 메시지의 둘째 줄\n"
    "[베타] [오전 9:30] 010-1234-5678 로 연락주세요\n"
    "[감마] [오전 9:31] 자료: https://example.com/doc\n"
    "[알파] [오전 9:32] 응 알겠어요\n"
    "감마님이 들어왔습니다.\n"
    "--------------- 2026년 5월 2일 금요일 ---------------\n"
    "[알파] [오전 10:00] 새 날 새 인사\n"
    "[베타] [오전 10:01] 굿모닝\n"
    "[감마] [오전 10:02] 좋은 아침입니다\n"
    "[알파] [오전 10:03] 회의 자료 준비됐어요\n"
    "[베타] [오전 10:05] 네 확인했어요\n"
)


@pytest.fixture()
def kakao_export(tmp_path: Path) -> Path:
    """Synthesize a 12-message kakao export covering varied register."""
    p = tmp_path / "synthetic_kakao.txt"
    p.write_text(_KAKAO_BODY, encoding="utf-8")
    return p


@pytest.fixture()
def staging_writer(tmp_path: Path) -> StagingWriter:
    """Per-test ``StagingWriter`` rooted at ``tmp_path``."""
    return StagingWriter("e2e-session-001", home=tmp_path)


# ---------------------------------------------------------------------------
# End-to-end flow
# ---------------------------------------------------------------------------


def test_e2e_parser_classifier_inventory_gap_anchor_patina(
    kakao_export: Path, staging_writer: StagingWriter, tmp_path: Path
) -> None:
    """Single-mission integration: every stage of the v1 pipeline must agree
    on the canonical ``schema.KakaoMsg`` contract."""

    # --------------------------------------------------------------- 1. Parser
    name_map: dict[str, str] = {}
    messages: list[KakaoMsg] = list(
        parse_kakao_file(kakao_export, name_map=name_map)
    )

    # Canonical schema invariants on every yielded message.
    assert len(messages) == 12
    for i, m in enumerate(messages):
        assert isinstance(m, KakaoMsg)
        assert m.msg_id == f"msg_{i:06d}"
        assert m.speaker_id.startswith("user_")
        # Timestamp populated; date-context-aware on every msg post-line-3.
        assert "T" in m.timestamp
        assert m.timestamp.startswith("2026-")
        assert isinstance(m.raw_text, str)
        assert isinstance(m.redacted_text, str)

    # PII redaction sanity: the raw line has a phone number, the redacted
    # body must not.
    phone_msg = next(m for m in messages if "010-1234-5678" in m.raw_text)
    assert "010-1234-5678" not in phone_msg.redacted_text

    # ------------------------------------------------------------ 2. Classifier
    classifier = Classifier(
        _FakeGLMClient(category="communication"),
        taxonomy=[
            "communication",
            "content",
            "filler",
            "language",
            "structure",
            "style",
            "viral-hook",
        ],
        batch_size=20,
        max_retries=1,
    )
    result = classifier.classify(messages)
    assert result.n_messages == 12
    assert len(result.classifications) == 12
    for c in result.classifications:
        # Every classification slots into the message id space the parser produced.
        assert c.msg_id.startswith("msg_")
        assert c.categories, "classifier must emit ≥1 category per msg"

    # ------------------------------------------------------------- 3. Inventory
    inventory_md = generate_inventory(result, messages)
    assert "DGM-H GLM data v1 — inventory report" in inventory_md
    assert "Messages classified: **12**" in inventory_md

    # ---------------------------------------------------------- 4. Gap analyzer
    # Stub coverage so we don't read the operator's local patina checkout.
    patina_coverage = PatinaCoverage(
        pattern_counts={
            "communication": 12,
            "content": 12,
            "filler": 12,
            "language": 12,
            "structure": 12,
            "style": 12,
            "viral-hook": 12,
        },
        lexicon_entries=80,
    )
    gap = analyze_gaps(result, patina_coverage=patina_coverage)
    assert gap.n_messages == 12
    # All 12 messages were classified as `communication`, so every other
    # baseline category is under-represented and surfaces as a gap row.
    gap_categories = {row.category for row in gap.gaps}
    assert "content" in gap_categories
    assert "filler" in gap_categories

    gap_md = render_gap_report(gap)
    assert "gap report" in gap_md

    # ------------------------------------------------------------ 5. Anchors
    classifications_by_id = {c.msg_id: c for c in result.classifications}
    report = AnchorEmitter(staging_writer).emit(
        messages, classifications=classifications_by_id
    )

    # Sidecar rows must mirror canonical schema fields. We round-trip them
    # via JSON to assert the contract holds across the I/O boundary too.
    sidecar_rows = [
        json.loads(line)
        for line in report.sidecar_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert sidecar_rows, "emit must produce at least one anchor candidate row"
    parser_ids = {m.msg_id for m in messages}
    for row in sidecar_rows:
        assert row["msg_id"] in parser_ids
        assert row["speaker_id"].startswith("user_")
        assert "T" in row["timestamp"]
        assert row["category"] == "communication"

    # The emit report and the on-disk artifact must agree on a non-empty
    # candidate list. (We don't equate `splitlines` to ``emit_count`` because
    # parser continuations may carry embedded newlines inside one message.)
    assert report.emit_count > 0
    assert report.txt_path.read_text(encoding="utf-8").strip()

    # ----------------------------------------------------- 6. patina_emit (J1)
    # Build a minimal language-scoped patina pack so we can verify the flat
    # numbering contract without touching the operator's real patina/ dir.
    patina_root = tmp_path / "patina"
    (patina_root / "patterns").mkdir(parents=True)
    (patina_root / "lexicon").mkdir()
    (patina_root / "patterns" / "ko-communication.md").write_text(
        "---\nlang: ko\n---\n\n### 1. greeting\nbody\n\n### 2. farewell\nbody\n",
        encoding="utf-8",
    )
    (patina_root / "patterns" / "ko-filler.md").write_text(
        "---\nlang: ko\n---\n\n### 3. ㅋㅋ\nbody\n",
        encoding="utf-8",
    )
    used = flat_pattern_namespace(patina_root, language="ko")
    # J1: numbers are language-scoped flat namespace (union across files).
    assert used == {1, 2, 3}


def test_e2e_kakaomsg_canonical_schema_contract(kakao_export: Path) -> None:
    """C3 regression: parser → classifier interface must not raise on the
    canonical ``msg_id`` / ``redacted_text`` accessors that the classifier reads.
    """
    messages = list(parse_kakao_file(kakao_export, name_map={}))
    # The classifier's _format_message_block reaches into these two fields
    # directly. Either dropping back to the old schema (.user/.text) would
    # raise AttributeError here.
    for m in messages:
        assert hasattr(m, "msg_id")
        assert hasattr(m, "redacted_text")
        # And the legacy fields from the pre-C3 parser must be absent.
        for legacy in ("user", "period", "hour", "minute", "text", "date_context"):
            assert not hasattr(m, legacy), (
                f"legacy field .{legacy} must not linger on schema.KakaoMsg"
            )
