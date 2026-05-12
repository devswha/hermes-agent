"""Tests for ``dgmh.glm_data.youtube_collector``.

Plan reference: ``.omc/plans/dgmh-glm-data-v1.md`` §4 Step 8, AC7, R11, H6, J6.

These tests must run hermetically — no real yt-dlp subprocess, no network.
The collector exposes a ``subprocess_runner`` injection point for exactly
this purpose; the runner here writes a synthetic ``*.info.json`` into the
working directory the collector hands it.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Callable

import pytest

from dgmh.glm_data.glm_client import NetworkDeniedError
from dgmh.glm_data.youtube_collector import (
    EXCERPT_CAP_CHARS,
    MAX_VIDEOS_PER_RUN,
    ConsentMissingError,
    ToSRiskNotAcceptedError,
    TooManyVideosError,
    YouTubeCollector,
)


# --------------------------------------------------------------------- helpers


SIGNED_CONSENT_BODY = (
    "# Consent\n"
    "signed-by: Test Operator\n"
    "signed-at: 2026-05-12\n"
)

UNSIGNED_CONSENT_BODY = (
    "# Consent\n"
    "signed-by: <REPLACE-WITH-OPERATOR-NAME>\n"
    "signed-at: <REPLACE-WITH-YYYY-MM-DD>\n"
)


@pytest.fixture()
def consent_path(tmp_path: Path) -> Path:
    path = tmp_path / "yt-dlp-consent.md"
    path.write_text(SIGNED_CONSENT_BODY, encoding="utf-8")
    return path


@pytest.fixture()
def network_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DGMH_GLM_DATA_V1_ENABLE_NETWORK", "1")


def _make_runner(comments_by_video: dict[str, list[dict]]) -> Callable[..., subprocess.CompletedProcess]:
    """Build a fake yt-dlp runner that drops a ``*.info.json`` per video.

    The collector's _yt_dlp_argv ends with the watch URL, so we recover the
    video_id from the last argv element. The runner writes ``<video_id>.info.json``
    matching yt-dlp's ``-o %(id)s.%(ext)s`` template.
    """

    def _runner(argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
        watch_url = argv[-1]
        video_id = watch_url.rsplit("=", 1)[-1]
        info = {"id": video_id, "comments": comments_by_video.get(video_id, [])}
        (cwd / f"{video_id}.info.json").write_text(
            json.dumps(info), encoding="utf-8"
        )
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    return _runner


# -------------------------------------------------------------- consent gating


def test_consent_doc_absent_fails(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.md"
    collector = YouTubeCollector(missing, subprocess_runner=_make_runner({}))
    with pytest.raises(ConsentMissingError):
        collector.collect(["abc123"], accept_tos_risk=True)


def test_consent_doc_unsigned_placeholder_fails(tmp_path: Path) -> None:
    consent = tmp_path / "consent.md"
    consent.write_text(UNSIGNED_CONSENT_BODY, encoding="utf-8")
    collector = YouTubeCollector(consent, subprocess_runner=_make_runner({}))
    with pytest.raises(ConsentMissingError):
        collector.collect(["abc123"], accept_tos_risk=True)


def test_consent_doc_missing_signed_at_fails(tmp_path: Path) -> None:
    consent = tmp_path / "consent.md"
    consent.write_text("signed-by: Test Op\n", encoding="utf-8")
    collector = YouTubeCollector(consent, subprocess_runner=_make_runner({}))
    with pytest.raises(ConsentMissingError):
        collector.collect(["abc123"], accept_tos_risk=True)


def test_consent_doc_missing_signed_by_fails(tmp_path: Path) -> None:
    consent = tmp_path / "consent.md"
    consent.write_text("signed-at: 2026-05-12\n", encoding="utf-8")
    collector = YouTubeCollector(consent, subprocess_runner=_make_runner({}))
    with pytest.raises(ConsentMissingError):
        collector.collect(["abc123"], accept_tos_risk=True)


# ---------------------------------------------------------- ToS-risk gating


def test_accept_tos_risk_default_off_fails(consent_path: Path) -> None:
    collector = YouTubeCollector(consent_path, subprocess_runner=_make_runner({}))
    with pytest.raises(ToSRiskNotAcceptedError):
        collector.collect(["abc123"], accept_tos_risk=False)


def test_accept_tos_risk_must_be_explicitly_true(consent_path: Path) -> None:
    collector = YouTubeCollector(consent_path, subprocess_runner=_make_runner({}))
    # Same as above but in dry-run mode — still must fail without opt-in.
    with pytest.raises(ToSRiskNotAcceptedError):
        collector.plan(["abc123"], accept_tos_risk=False)


# ---------------------------------------------------------- network gating


def test_network_denied_when_env_unset(
    consent_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DGMH_GLM_DATA_V1_ENABLE_NETWORK", raising=False)
    collector = YouTubeCollector(consent_path, subprocess_runner=_make_runner({}))
    with pytest.raises(NetworkDeniedError):
        collector.collect(["abc123"], accept_tos_risk=True)


def test_dry_run_does_not_require_network(consent_path: Path) -> None:
    """plan() must work even when network is disabled."""

    os.environ.pop("DGMH_GLM_DATA_V1_ENABLE_NETWORK", None)
    collector = YouTubeCollector(consent_path, subprocess_runner=_make_runner({}))
    plan = collector.plan(["abc123", "def456"], accept_tos_risk=True)
    assert plan.video_ids == ["abc123", "def456"]
    assert plan.network_enabled is False
    assert len(plan.argv_per_video) == 2
    assert plan.argv_per_video[0][0] == "yt-dlp"


# ---------------------------------------------------------- rate cap


def test_rate_cap_enforced(consent_path: Path, network_on: None) -> None:
    collector = YouTubeCollector(consent_path, subprocess_runner=_make_runner({}))
    too_many = [f"vid{i}" for i in range(MAX_VIDEOS_PER_RUN + 1)]
    with pytest.raises(TooManyVideosError):
        collector.collect(too_many, accept_tos_risk=True)


# ---------------------------------------------------------- happy path


def test_happy_path_produces_redacted_rows(
    consent_path: Path, network_on: None
) -> None:
    comments = {
        "vid1": [
            {"text": "안녕하세요 010-1234-5678 입니다", "author": "Alice"},
            {"text": "Hello world", "author": "Bob"},
            {"text": "  ", "author": "Empty"},  # skipped: blank text
        ],
    }
    collector = YouTubeCollector(
        consent_path, subprocess_runner=_make_runner(comments)
    )
    rows = collector.collect(["vid1"], accept_tos_risk=True, gap_target="ko-style")

    assert len(rows) == 2
    # Schema check.
    for row in rows:
        assert set(row.keys()) == {
            "source",
            "video_id",
            "author_hash",
            "text",
            "lang_detect",
            "gap_target",
            "collected_at",
        }
        assert row["source"] == "youtube"
        assert row["video_id"] == "vid1"
        assert row["gap_target"] == "ko-style"
        assert len(row["author_hash"]) == 12

    # Phone number must be redacted from row 0 (AC7).
    assert "010-1234-5678" not in rows[0]["text"]
    assert rows[0]["lang_detect"] == "ko"
    assert rows[1]["lang_detect"] == "en"


def test_author_hash_is_sha256_prefix(
    consent_path: Path, network_on: None
) -> None:
    comments = {"vid1": [{"text": "hi", "author": "alice@example.com"}]}
    collector = YouTubeCollector(
        consent_path, subprocess_runner=_make_runner(comments)
    )
    rows = collector.collect(["vid1"], accept_tos_risk=True)
    import hashlib

    expected = hashlib.sha256(b"alice@example.com").hexdigest()[:12]
    assert rows[0]["author_hash"] == expected
    # Raw author handle must NEVER appear in the row.
    flat = json.dumps(rows[0])
    assert "alice@example.com" not in flat
    assert "alice" not in flat.lower() or "alice" in expected.lower()


def test_excerpt_capped_at_200_chars(
    consent_path: Path, network_on: None
) -> None:
    long_text = "x" * 500
    comments = {"vid1": [{"text": long_text, "author": "A"}]}
    collector = YouTubeCollector(
        consent_path, subprocess_runner=_make_runner(comments)
    )
    rows = collector.collect(["vid1"], accept_tos_risk=True)
    assert len(rows[0]["text"]) == EXCERPT_CAP_CHARS


def test_no_raw_author_handle_persisted(
    consent_path: Path, network_on: None
) -> None:
    comments = {
        "vid1": [{"text": "ok", "author": "DistinctiveAuthorNameXYZ"}]
    }
    collector = YouTubeCollector(
        consent_path, subprocess_runner=_make_runner(comments)
    )
    rows = collector.collect(["vid1"], accept_tos_risk=True)
    payload = json.dumps(rows)
    assert "DistinctiveAuthorNameXYZ" not in payload


def test_multiple_videos_round_trip(
    consent_path: Path, network_on: None
) -> None:
    comments = {
        "v1": [{"text": "comment-A", "author": "A"}],
        "v2": [{"text": "comment-B", "author": "B"}],
        "v3": [{"text": "comment-C", "author": "C"}],
    }
    collector = YouTubeCollector(
        consent_path, subprocess_runner=_make_runner(comments)
    )
    rows = collector.collect(["v1", "v2", "v3"], accept_tos_risk=True)
    assert [r["video_id"] for r in rows] == ["v1", "v2", "v3"]
    assert [r["text"] for r in rows] == ["comment-A", "comment-B", "comment-C"]


def test_no_comments_in_info_json_returns_empty(
    consent_path: Path, network_on: None
) -> None:
    collector = YouTubeCollector(
        consent_path, subprocess_runner=_make_runner({"vid1": []})
    )
    rows = collector.collect(["vid1"], accept_tos_risk=True)
    assert rows == []
