"""Tests for ``dgmh.glm_data.__main__`` CLI dispatcher.

Plan reference: ``.omc/plans/dgmh-glm-data-v1.md`` §4 Step 12, AC10, AC14,
D-7, D-9.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable

import pytest

from dgmh.glm_data import __main__ as cli
from dgmh.glm_data.__main__ import (
    DEFAULT_FILE_NAME,
    MEDICAL_FILE_NAMES,
    build_parser,
    main,
    resolve_corpus_files,
)


# --------------------------------------------------------------------- fixtures


@pytest.fixture()
def fake_corpus_dir(tmp_path: Path) -> Path:
    """A directory holding all 7 corpus files (5 non-medical + 2 medical)."""

    names = [
        "058_45_466_group.txt",  # medical, smallest
        "059_34_202_group.txt",  # medical
        "012_31_001_group.txt",
        "045_18_003_group.txt",
        "098_22_010_group.txt",
        "121_44_055_group.txt",
        "147_07_088_group.txt",
    ]
    for name in names:
        (tmp_path / name).write_text(
            f"--- preamble for {name} ---\n",
            encoding="utf-8",
        )
    return tmp_path


@pytest.fixture()
def network_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DGMH_GLM_DATA_V1_ENABLE_NETWORK", raising=False)


@pytest.fixture()
def network_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DGMH_GLM_DATA_V1_ENABLE_NETWORK", "1")


# ----------------------------------------------------------------- parser shape


def test_help_lists_all_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for sub in (
        "analyze",
        "gap",
        "collect",
        "patina",
        "anchor",
        "review",
        "apply-dryrun",
    ):
        assert sub in out


def test_review_subparser_exposes_all_four_verbs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["review", "--help"])
    out = capsys.readouterr().out
    for verb in ("--status", "--approve", "--reject", "--defer"):
        assert verb in out


def test_review_verbs_are_mutually_exclusive() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["review", "--status", "--approve", "S00"])


# ----------------------------------------------------------------- scope rules


def test_default_scope_is_one_file_the_smallest(
    fake_corpus_dir: Path,
) -> None:
    files = resolve_corpus_files(fake_corpus_dir)
    assert len(files) == 1
    assert files[0].name == DEFAULT_FILE_NAME


def test_full_corpus_returns_five_non_medical(fake_corpus_dir: Path) -> None:
    files = resolve_corpus_files(fake_corpus_dir, full_corpus=True)
    names = {f.name for f in files}
    assert len(files) == 5
    # Medical files must NOT appear (interpretation B per D5).
    assert names.isdisjoint(MEDICAL_FILE_NAMES)


def test_full_corpus_with_medical_returns_all_seven(
    fake_corpus_dir: Path,
) -> None:
    files = resolve_corpus_files(
        fake_corpus_dir, full_corpus=True, include_medical=True
    )
    assert len(files) == 7
    names = {f.name for f in files}
    assert names.issuperset(MEDICAL_FILE_NAMES)


def test_include_medical_alone_does_not_widen_default_scope(
    fake_corpus_dir: Path,
) -> None:
    """Just --include-medical-groups (no --full-corpus) stays at 1 file."""

    files = resolve_corpus_files(fake_corpus_dir, include_medical=True)
    assert len(files) == 1
    assert files[0].name == DEFAULT_FILE_NAME


def test_max_corpus_caps_full_corpus(fake_corpus_dir: Path) -> None:
    files = resolve_corpus_files(
        fake_corpus_dir, full_corpus=True, max_corpus=3
    )
    assert len(files) == 3


def test_single_file_path_passes_through(fake_corpus_dir: Path) -> None:
    target = fake_corpus_dir / "012_31_001_group.txt"
    files = resolve_corpus_files(target)
    assert files == [target]


def test_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_corpus_files(tmp_path / "does-not-exist")


# ----------------------------------------------------------------- analyze


_ESTIMATE_FIELDS = {
    "input_tokens",
    "output_tokens",
    "projected_usd",
    "projected_wall_seconds",
    "n_calls",
    "n_samples",
    "batch_size",
    "model",
    "corpus_path",
    "per_call_median_s",
}


def test_analyze_estimate_works_without_network(
    fake_corpus_dir: Path,
    network_off: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["analyze", "--estimate", str(fake_corpus_dir)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["mode"] == "estimate"
    assert payload["over_budget"] is False
    assert len(payload["estimates"]) == 1
    assert _ESTIMATE_FIELDS.issubset(payload["estimates"][0].keys())


def test_analyze_non_estimate_requires_network(
    fake_corpus_dir: Path, network_off: None
) -> None:
    rc = main(["analyze", str(fake_corpus_dir)])
    assert rc != 0


def test_analyze_full_corpus_estimate_five_files(
    fake_corpus_dir: Path,
    network_off: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(
        ["analyze", "--estimate", "--full-corpus", str(fake_corpus_dir)]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert len(payload["estimates"]) == 5


def test_analyze_full_corpus_with_medical_seven_files(
    fake_corpus_dir: Path,
    network_off: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(
        [
            "analyze",
            "--estimate",
            "--full-corpus",
            "--include-medical-groups",
            str(fake_corpus_dir),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert len(payload["estimates"]) == 7


def test_analyze_estimate_abort_over_usd_zero_exits_two(
    fake_corpus_dir: Path,
    network_off: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """C1 regression: `--abort-over-usd 0` must short-circuit with rc=2 and
    still emit the cost projection JSON on stdout so callers can log it."""
    rc = main(
        [
            "analyze",
            "--estimate",
            "--abort-over-usd",
            "0",
            str(fake_corpus_dir),
        ]
    )
    assert rc == 2
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["mode"] == "estimate"
    assert payload["over_budget"] is True
    assert payload["abort_over_usd"] == 0.0
    assert len(payload["estimates"]) == 1
    est = payload["estimates"][0]
    assert _ESTIMATE_FIELDS.issubset(est.keys())
    assert est["projected_usd"] > 0


# ----------------------------------------------------------------- collect


def test_collect_refuses_without_accept_tos_risk(
    network_on: None,
) -> None:
    rc = main(["collect", "vid1"])
    assert rc != 0


def test_collect_refuses_without_network(
    network_off: None,
) -> None:
    rc = main(["collect", "--accept-tos-risk", "vid1"])
    assert rc != 0


def test_collect_dispatches_when_gates_pass(
    network_on: None, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["collect", "--accept-tos-risk", "v1", "v2"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["mode"] == "collect"
    assert payload["video_ids"] == ["v1", "v2"]


# ----------------------------------------------------------------- review


@pytest.fixture()
def staging_session(tmp_path: Path) -> tuple[Path, str]:
    """Build an empty staging session under ``tmp_path`` and return
    ``(staging_home, session_id)`` so tests can pass them through the CLI.

    The session root is created with no approvals / report so the CLI starts
    from a clean slate — individual tests then drive `review --approve` etc.
    """
    from dgmh.glm_data import STAGING_ROOT_RELATIVE

    session_id = "test-session-001"
    (tmp_path / STAGING_ROOT_RELATIVE / session_id).mkdir(parents=True)
    return tmp_path, session_id


def _common_args(staging_home: Path, session_id: str) -> list[str]:
    return [
        "--session-id",
        session_id,
        "--staging-home",
        str(staging_home),
    ]


def test_review_status_no_session_exits_three(
    tmp_path: Path,
) -> None:
    """No staging session → `review --status` returns EXIT_GATE_BLOCKED."""
    rc = main(
        ["--staging-home", str(tmp_path), "review", "--status"]
    )
    assert rc == 3


def test_review_status_returns_review_manager_payload(
    staging_session: tuple[Path, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    staging_home, session_id = staging_session
    rc = main(
        _common_args(staging_home, session_id) + ["review", "--status"]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["verb"] == "status"
    assert out["session_id"] == session_id
    # ReviewManager.status() shape — none of these existed in the dispatcher
    # stub that this test replaces.
    assert out["approved"] == 0
    assert out["rejected"] == 0
    assert out["deferred"] == 0
    assert out["required"] == 10
    assert out["can_apply_dryrun"] is False


def test_review_approve_writes_to_approvals_jsonl(
    staging_session: tuple[Path, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """C2 regression: --approve must actually call ReviewManager.approve and
    persist an entry to approvals.jsonl (not just echo the dispatcher payload).
    """
    from dgmh.glm_data import STAGING_ROOT_RELATIVE

    staging_home, session_id = staging_session
    rc = main(
        _common_args(staging_home, session_id)
        + ["review", "--approve", "S00", "--notes", "looks good"]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["verb"] == "approve"
    assert out["session_id"] == session_id
    assert out["entry"]["sample_id"] == "S00"
    assert out["entry"]["decision"] == "approve"
    assert out["entry"]["notes_or_reason"] == "looks good"

    approvals = (
        staging_home / STAGING_ROOT_RELATIVE / session_id / "approvals.jsonl"
    )
    assert approvals.is_file(), "approve must persist to approvals.jsonl"
    rows = [json.loads(ln) for ln in approvals.read_text().splitlines() if ln.strip()]
    assert len(rows) == 1
    assert rows[0]["sample_id"] == "S00"
    assert rows[0]["decision"] == "approve"


def test_review_reject_requires_reason(
    staging_session: tuple[Path, str],
) -> None:
    staging_home, session_id = staging_session
    rc = main(
        _common_args(staging_home, session_id)
        + ["review", "--reject", "S01"]
    )
    assert rc != 0


def test_review_reject_with_reason(
    staging_session: tuple[Path, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    staging_home, session_id = staging_session
    rc = main(
        _common_args(staging_home, session_id)
        + ["review", "--reject", "S01", "--reason", "noisy"]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["verb"] == "reject"
    assert out["entry"]["sample_id"] == "S01"
    assert out["entry"]["notes_or_reason"] == "noisy"


def test_review_defer(
    staging_session: tuple[Path, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    staging_home, session_id = staging_session
    rc = main(
        _common_args(staging_home, session_id)
        + ["review", "--defer", "S02"]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["verb"] == "defer"
    assert out["entry"]["sample_id"] == "S02"
    assert out["entry"]["decision"] == "defer"


# ----------------------------------------------------------------- apply-dryrun + D-7


def test_apply_dryrun_blocked_when_approvals_below_threshold(
    staging_session: tuple[Path, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """C2 regression: apply-dryrun must be gated by REQUIRED_APPROVALS via
    ReviewManager.check_apply_dryrun, returning rc=3 with `blocked=true`."""
    staging_home, session_id = staging_session
    rc = main(
        _common_args(staging_home, session_id) + ["apply-dryrun"]
    )
    assert rc == 3
    out = json.loads(capsys.readouterr().out.strip())
    assert out["mode"] == "apply-dryrun"
    assert out["would_invoke_git_apply"] is False
    assert out["blocked"] is True
    assert out["status"]["approved"] < 10


def test_apply_dryrun_zero_with_ten_approvals(
    staging_session: tuple[Path, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With 10 approvals on the ledger, apply-dryrun emits an operator-readable
    plan and exits 0 — no `git apply` shellout (D-7)."""
    from dgmh.glm_data import STAGING_ROOT_RELATIVE

    staging_home, session_id = staging_session
    approvals = (
        staging_home / STAGING_ROOT_RELATIVE / session_id / "approvals.jsonl"
    )
    approvals.parent.mkdir(parents=True, exist_ok=True)
    with approvals.open("w", encoding="utf-8") as fh:
        for i in range(10):
            fh.write(
                json.dumps(
                    {
                        "sample_id": f"S{i:02d}",
                        "decision": "approve",
                        "ts": 1.0,
                        "reviewer": "operator",
                        "notes_or_reason": "",
                        "review_duration_s": 120,
                    }
                )
                + "\n"
            )
    rc = main(
        _common_args(staging_home, session_id) + ["apply-dryrun"]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["mode"] == "apply-dryrun"
    assert out["would_invoke_git_apply"] is False
    assert out["blocked"] is False
    assert out["status"]["approved"] == 10
    assert "patches" in out
    assert "artifacts" in out


def test_apply_dryrun_no_session_exits_three(
    tmp_path: Path,
) -> None:
    rc = main(
        ["--staging-home", str(tmp_path), "apply-dryrun"]
    )
    assert rc == 3


def test_module_source_has_no_git_apply_invocation() -> None:
    """D-7: the CLI must never shell out to ``git apply``.

    Grep guard: ``git apply`` must not appear as an executable string
    anywhere in the module source (banner comments excluded — we only
    check for invocation patterns).
    """

    source = Path(cli.__file__).read_text(encoding="utf-8")
    for forbidden in ('"git", "apply"', "['git', 'apply']", "git apply\""):
        assert forbidden not in source, f"D-7 violation: found {forbidden!r}"
    # Belt-and-braces: no subprocess.run / Popen calls anywhere.
    assert "subprocess.run" not in source
    assert "subprocess.Popen" not in source


# ----------------------------------------------------------------- gap / patina / anchor


def test_gap_requires_network(network_off: None) -> None:
    rc = main(["gap"])
    assert rc != 0


def test_gap_dispatch_when_network_on(
    network_on: None, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["gap"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["mode"] == "gap"


def test_patina_dispatch(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["patina"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out.strip())["mode"] == "patina"


def test_anchor_dispatch(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["anchor"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out.strip())["mode"] == "anchor"
