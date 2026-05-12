"""Tests for ``dgmh.glm_data.review``.

Plan reference: ``.omc/plans/dgmh-glm-data-v1.md`` §4 Step 11, AC12, V9, H4, M3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pytest
import yaml

from dgmh.glm_data.review import (
    APPROVALS_RELPATH,
    REPORT_RELPATH,
    REQUIRED_APPROVALS,
    WARNINGS_RELPATH,
    ApplyDryrunBlocked,
    ReviewManager,
    ReviewSample,
    UnknownDecisionError,
)
from dgmh.glm_data.staging_writer import StagingWriter
from dgmh.glm_data.taxonomy_loader import BASELINE_KO_7, load as load_taxonomy


# --------------------------------------------------------------------- fixtures


@pytest.fixture()
def fake_home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    (home / ".hermes" / "dgmh").mkdir(parents=True)
    return home


@pytest.fixture()
def writer(fake_home: Path) -> StagingWriter:
    return StagingWriter("review-sess", home=fake_home)


@pytest.fixture()
def clock() -> dict:
    """Monotonic clock under test control."""

    return {"now": 1_700_000_000.0}


@pytest.fixture()
def manager(writer: StagingWriter, clock: dict) -> ReviewManager:
    return ReviewManager(
        writer,
        min_review_seconds=60,
        now=lambda: clock["now"],
    )


def _samples(n: int = 12) -> list[ReviewSample]:
    return [
        ReviewSample(
            sample_id=f"S{i:02d}",
            source="kakao",
            text=f"샘플 문장 {i}",
            classification=("style" if i % 2 == 0 else "communication"),
            redaction_trace={"phone": (1 if i == 0 else 0)},
            rationale=f"이유 {i}",
            target_file="patterns/ko-style.md" if i % 2 == 0 else "lexicon/ai-ko.md",
        )
        for i in range(n)
    ]


# --------------------------------------------------------------- report render


def test_generate_report_renders_all_samples(manager: ReviewManager) -> None:
    samples = _samples(10)
    path = manager.generate_report(samples)
    text = path.read_text(encoding="utf-8")

    assert path.relative_to(manager._root) == Path(REPORT_RELPATH)
    assert "# DGM-H GLM Data v1 — Review Report" in text
    for s in samples:
        assert f"## {s.sample_id}" in text
        assert s.text in text
        assert s.classification in text


def test_status_pending_lists_undecided_samples(manager: ReviewManager) -> None:
    samples = _samples(5)
    manager.generate_report(samples)
    manager.approve("S00")
    st = manager.status()
    assert st["pending"] == ["S01", "S02", "S03", "S04"]


# --------------------------------------------------------------- verb appends


def test_each_verb_appends_exactly_one_entry(
    manager: ReviewManager, clock: dict
) -> None:
    manager.generate_report(_samples(10))
    clock["now"] += 120  # past min_review_seconds

    manager.approve("S00", notes="looks good")
    manager.reject("S01", reason="too noisy")
    manager.defer("S02")

    entries = [
        json.loads(line)
        for line in (manager._root / APPROVALS_RELPATH)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(entries) == 3
    decisions = [e["decision"] for e in entries]
    assert decisions == ["approve", "reject", "defer"]
    assert entries[0]["notes_or_reason"] == "looks good"
    assert entries[1]["notes_or_reason"] == "too noisy"
    assert entries[2]["notes_or_reason"] == ""
    for entry in entries:
        assert set(entry.keys()) == {
            "sample_id",
            "decision",
            "ts",
            "reviewer",
            "notes_or_reason",
            "review_duration_s",
        }


def test_reject_requires_non_empty_reason(manager: ReviewManager) -> None:
    manager.generate_report(_samples(10))
    with pytest.raises(ValueError):
        manager.reject("S00", reason="")


def test_unknown_decision_raises(manager: ReviewManager) -> None:
    with pytest.raises(UnknownDecisionError):
        manager._record(
            sample_id="x",
            decision="archive",
            reviewer="op",
            notes_or_reason="",
        )


# --------------------------------------------------------------- min-time warn


def test_fast_approval_warns_but_does_not_block(
    manager: ReviewManager, clock: dict
) -> None:
    manager.generate_report(_samples(10))
    # Approve within 5s of report mtime → warning, but entry still recorded.
    clock["now"] += 5

    manager.approve("S00")

    approvals = (
        (manager._root / APPROVALS_RELPATH).read_text(encoding="utf-8").splitlines()
    )
    warnings = (
        (manager._root / WARNINGS_RELPATH).read_text(encoding="utf-8").splitlines()
    )
    assert len(approvals) == 1  # not blocked
    assert len(warnings) == 1
    warning = json.loads(warnings[0])
    assert warning["sample_id"] == "S00"
    assert warning["review_duration_s"] < 60
    assert warning["min_review_seconds"] == 60


def test_slow_approval_emits_no_warning(
    manager: ReviewManager, clock: dict
) -> None:
    manager.generate_report(_samples(10))
    clock["now"] += 90
    manager.approve("S00")
    assert not (manager._root / WARNINGS_RELPATH).is_file()


def test_reject_under_min_time_does_not_warn(
    manager: ReviewManager, clock: dict
) -> None:
    """Only approvals warn — rejects/defers can land instantly."""

    manager.generate_report(_samples(10))
    clock["now"] += 1
    manager.reject("S00", reason="x")
    manager.defer("S01")
    assert not (manager._root / WARNINGS_RELPATH).is_file()


# --------------------------------------------------------------- status counts


def test_status_counts_correct_after_mixed_verbs(
    manager: ReviewManager, clock: dict
) -> None:
    manager.generate_report(_samples(10))
    clock["now"] += 120
    for i in range(7):
        manager.approve(f"S{i:02d}")
    manager.reject("S07", reason="r")
    manager.defer("S08")
    # S09 still pending

    st = manager.status()
    assert st["approved"] == 7
    assert st["rejected"] == 1
    assert st["deferred"] == 1
    assert st["required"] == REQUIRED_APPROVALS == 10
    assert st["pending"] == ["S09"]
    assert st["can_apply_dryrun"] is False


# --------------------------------------------------------------- apply gate


def test_apply_dryrun_blocked_under_10_approvals(
    manager: ReviewManager, clock: dict
) -> None:
    manager.generate_report(_samples(12))
    clock["now"] += 120
    for i in range(9):
        manager.approve(f"S{i:02d}")
    with pytest.raises(ApplyDryrunBlocked):
        manager.check_apply_dryrun()


def test_apply_dryrun_succeeds_at_10_approvals(
    manager: ReviewManager, clock: dict
) -> None:
    manager.generate_report(_samples(12))
    clock["now"] += 120
    for i in range(10):
        manager.approve(f"S{i:02d}")
    # Must not raise.
    manager.check_apply_dryrun()
    assert manager.status()["can_apply_dryrun"] is True


# --------------------------------------------------------------- promotion hook


def test_approving_new_proposal_promotes_taxonomy(
    manager: ReviewManager, clock: dict
) -> None:
    """Approving a _new_* classification promotes it via taxonomy_loader."""

    manager.generate_report(_samples(10))
    clock["now"] += 120

    manager.approve(
        "S00",
        sample_classification="_new_kakao_register_chimaek",
    )

    # The taxonomy_loader ledger should now contain the new category.
    cats = load_taxonomy(manager._root)
    assert cats == BASELINE_KO_7 + ["_new_kakao_register_chimaek"]


def test_baseline_approval_does_not_promote(
    manager: ReviewManager, clock: dict
) -> None:
    manager.generate_report(_samples(10))
    clock["now"] += 120
    manager.approve("S00", sample_classification="style")
    assert load_taxonomy(manager._root) == BASELINE_KO_7
    assert not (manager._root / "promoted_taxonomy.yaml").is_file()


# --------------------------------------------------------------- append-only


def test_approvals_jsonl_is_append_only(
    manager: ReviewManager, clock: dict
) -> None:
    manager.generate_report(_samples(10))
    clock["now"] += 120
    manager.approve("S00")
    first_bytes = (manager._root / APPROVALS_RELPATH).read_bytes()

    clock["now"] += 30
    manager.approve("S01")
    second_bytes = (manager._root / APPROVALS_RELPATH).read_bytes()

    # The file grew; the first entry's bytes remain identical at the front.
    assert second_bytes.startswith(first_bytes)
    assert len(second_bytes) > len(first_bytes)
