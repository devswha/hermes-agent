"""Tests for ``dgmh.glm_data.anchor_emit``.

Plan reference: ``.omc/plans/dgmh-glm-data-v1.md`` §4 Step 10, AC10, AC15,
V4, V10, D-5, D-8, C2.

The contract under test is twofold:

1. Every line in ``kakao_anchors_candidate.txt`` survives the *real*
   ``kakao_style_retrieval._load_corpus`` (5-filter compliance, AC10).
2. The canonical corpus at ``~/.hermes/dgmh/kakao_hako_corpus.txt`` is
   never read, modified, or shadowed (AC15 / D-8).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pytest

from dgmh.glm_data import STAGING_ROOT_RELATIVE
from dgmh.glm_data.anchor_emit import (
    AnchorEmitter,
    DropReason,
    EmitReport,
    verify_emitted_file_loads,
)
from dgmh.glm_data.schema import KakaoMsg
from dgmh.glm_data.staging_writer import StagingWriter
from dgmh.kakao_style_retrieval import _load_corpus, build_kakao_rag_profile_body


# --------------------------------------------------------------------- fixtures


@pytest.fixture()
def fake_home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    (home / ".hermes" / "dgmh").mkdir(parents=True)
    return home


@pytest.fixture()
def writer(fake_home: Path) -> StagingWriter:
    return StagingWriter("anchor-test", home=fake_home)


@pytest.fixture()
def emitter(writer: StagingWriter) -> AnchorEmitter:
    return AnchorEmitter(writer)


def _mk(
    msg_id: str,
    redacted: str,
    *,
    speaker: str = "user_01",
    ts: str = "2026-05-12T00:00:00",
) -> KakaoMsg:
    return KakaoMsg(
        msg_id=msg_id,
        speaker_id=speaker,
        timestamp=ts,
        raw_text=redacted,
        redacted_text=redacted,
    )


# --------------------------------------------------------- 5-filter compliance


def _all_lines_pass_5_filters(txt_path: Path) -> bool:
    """Cross-check: does every emitted line survive _load_corpus by itself?"""

    raw = txt_path.read_text(encoding="utf-8")
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            return False
        if stripped.startswith("@") or stripped.startswith("[20"):
            return False
        if len(stripped) < 4:
            return False
        import re

        if re.search(r"https?://|www\.", stripped):
            return False
    return True


def test_emit_filters_drop_each_5filter_violation(
    emitter: AnchorEmitter,
) -> None:
    msgs = [
        _mk("m1", "오늘 회의 끝나고 한잔할래?"),  # OK
        _mk("m2", "ㄷㄷ"),  # short reaction → diverted
        _mk("m3", "@everyone 안녕"),  # @-prefix → dropped
        _mk("m4", "[2026-05-12] 회의록"),  # [20-prefix → dropped
        _mk("m5", "https://example.com/foo bar"),  # URL → dropped
        _mk("m6", "   "),  # empty after strip → dropped
        _mk("m7", "오늘 회의 끝나고 한잔할래?"),  # duplicate of m1
        _mk("m8", "x" * 250),  # too long
        _mk("m9", "괜찮은 식당 찾았어!"),  # OK
        _mk("m10", "ㅋㅋ"),  # short reaction → diverted
    ]
    report = emitter.emit(msgs)
    assert report.emit_count == 2
    assert report.short_reaction_count == 2
    assert report.dropped.get(DropReason.AT_PREFIX) == 1
    assert report.dropped.get(DropReason.HEADER_PREFIX) == 1
    assert report.dropped.get(DropReason.URL) == 1
    assert report.dropped.get(DropReason.EMPTY) == 1
    assert report.dropped.get(DropReason.DUPLICATE) == 1
    assert report.dropped.get(DropReason.TOO_LONG) == 1
    assert _all_lines_pass_5_filters(report.txt_path)


def test_emit_passes_real_load_corpus(emitter: AnchorEmitter) -> None:
    """AC10: the *actual* _load_corpus accepts every emitted line."""

    msgs = [
        _mk("m1", "오늘 점심 뭐 먹지"),
        _mk("m2", "그래 한 시에 보자"),
        _mk("m3", "내일은 회의가 길어질 듯"),
        _mk("m4", "Bob said he'll join us"),
        _mk("m5", "퇴근하고 노래방 콜?"),
    ]
    report = emitter.emit(msgs)
    assert report.emit_count == 5

    # The plan demands this exact call shape (V4):
    corpus = _load_corpus(report.txt_path)
    assert len(corpus.messages) == report.emit_count
    assert len(corpus.messages) > 0


def test_verify_helper_returns_load_count(emitter: AnchorEmitter) -> None:
    msgs = [_mk(f"m{i}", f"메시지 라인 {i}번 입니다") for i in range(7)]
    report = emitter.emit(msgs)
    assert verify_emitted_file_loads(report.txt_path) == report.emit_count


# ----------------------------------------------------- register_mode round-trip


@pytest.mark.parametrize("mode", ["mirror", "public_haeyo"])
def test_build_kakao_rag_profile_body_accepts_emitted_anchors(
    emitter: AnchorEmitter, mode: str
) -> None:
    """V4: both register_mode values must render without raising."""

    msgs = [_mk(f"m{i}", f"테스트 메시지 {i} 어쩌고") for i in range(10)]
    report = emitter.emit(msgs, register_mode_hint=mode)
    anchors = report.txt_path.read_text(encoding="utf-8").splitlines()
    body = build_kakao_rag_profile_body(anchors, register_mode=mode)
    assert isinstance(body, str)
    assert body, "profile body must be non-empty for >=1 anchor"


# --------------------------------------------------------------- sidecar shape


def test_sidecar_jsonl_decorates_each_emitted_line(emitter: AnchorEmitter) -> None:
    msgs = [_mk("m1", "안녕 오늘 뭐했어"), _mk("m2", "그냥 집에 있었지")]
    report = emitter.emit(msgs, register_mode_hint="public_haeyo")
    rows = [
        json.loads(line)
        for line in report.sidecar_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == report.emit_count
    for row in rows:
        assert set(
            ["msg_id", "speaker_id", "timestamp", "text_sha256_12",
             "category", "register_mode_hint", "length"]
        ) <= set(row.keys())
        assert row["register_mode_hint"] == "public_haeyo"
        assert row["category"] == "unknown"
        assert len(row["text_sha256_12"]) == 12


def test_short_reactions_routed_to_sidecar_only(
    emitter: AnchorEmitter,
) -> None:
    msgs = [_mk(f"r{i}", "ㄷㄷ") for i in range(3)] + [
        _mk(f"r{i}", "ㅋㅋ") for i in range(3, 6)
    ]
    report = emitter.emit(msgs)
    assert report.emit_count == 0
    assert report.short_reaction_count == 6
    short_rows = [
        json.loads(line)
        for line in report.short_reactions_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(short_rows) == 6
    # No short reaction must leak into the txt file (AC10).
    assert report.txt_path.read_text(encoding="utf-8") == ""


# ------------------------------------------------------- per-run cap


def test_max_lines_cap_enforced(writer: StagingWriter) -> None:
    em = AnchorEmitter(writer, max_lines=3)
    msgs = [_mk(f"m{i}", f"메시지 라인 번호 {i}") for i in range(10)]
    report = em.emit(msgs)
    assert report.emit_count == 3
    # Remaining 7 messages are dropped under RUN_CAP, not silently lost.
    assert report.dropped.get(DropReason.RUN_CAP) == 7


# ------------------------------------------------- canonical corpus integrity


def test_canonical_corpus_sha256_unchanged_across_emit(
    emitter: AnchorEmitter, fake_home: Path
) -> None:
    """AC15 / D-8: emit must never touch ~/.hermes/dgmh/kakao_hako_corpus.txt."""

    canonical = fake_home / ".hermes" / "dgmh" / "kakao_hako_corpus.txt"
    canonical.write_text(
        "원래 메시지 1\n원래 메시지 2\n원래 메시지 3\n", encoding="utf-8"
    )
    pre = hashlib.sha256(canonical.read_bytes()).hexdigest()

    msgs = [_mk(f"m{i}", f"새 후보 라인 {i}") for i in range(20)]
    report = emitter.emit(msgs)
    assert report.emit_count == 20

    post = hashlib.sha256(canonical.read_bytes()).hexdigest()
    assert pre == post, "canonical kakao_hako_corpus.txt was mutated (AC15 fail)"


def test_output_paths_under_session_staging_root(
    emitter: AnchorEmitter, fake_home: Path
) -> None:
    """AC11/AC13: all three artifacts live under the per-session root."""

    msgs = [_mk("m1", "정상적인 메시지 한 줄")]
    report = emitter.emit(msgs)
    session_root = (
        fake_home / STAGING_ROOT_RELATIVE / "anchor-test"
    ).resolve()
    for path in (
        report.txt_path,
        report.sidecar_path,
        report.short_reactions_path,
    ):
        path.relative_to(session_root)  # raises if escape


# ---------------------------------------------------- classification decoration


def test_sidecar_picks_up_classification_when_provided(
    emitter: AnchorEmitter,
) -> None:
    from dgmh.glm_data.schema import Classification, ToneCategory

    cls_map = {
        "m1": Classification(
            msg_id="m1",
            categories=[
                ToneCategory(name="style", target_file="patterns/ko-style.md")
            ],
            confidence=0.92,
        ),
    }
    msgs = [
        _mk("m1", "이건 분류된 메시지야"),
        _mk("m2", "이건 분류 안 된 메시지야"),
    ]
    report = emitter.emit(msgs, classifications=cls_map)
    rows = [
        json.loads(line)
        for line in report.sidecar_path.read_text(encoding="utf-8").splitlines()
    ]
    rows_by_id = {r["msg_id"]: r for r in rows}
    assert rows_by_id["m1"]["category"] == "style"
    assert rows_by_id["m2"]["category"] == "unknown"


def test_emit_rejects_non_kakaomsg_input(emitter: AnchorEmitter) -> None:
    with pytest.raises(TypeError):
        emitter.emit([{"redacted_text": "not a KakaoMsg"}])  # type: ignore[list-item]
