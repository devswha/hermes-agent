"""YouTube comment collector for DGM-H GLM data v1 (Step 8).

Plan reference: ``.omc/plans/dgmh-glm-data-v1.md`` §4 Step 8, AC7, R11, H6, J6.

This module wraps ``yt-dlp --skip-download --write-comments --write-info-json``
for an **explicit operator-curated list of video_id values** and emits
redacted, capped, hashed fragments suitable for downstream gap-fill use.

Three gates block execution (any failure → no network call, no write):

* **J6** — operator must have signed ``dgmh/docs/yt-dlp-consent.md`` with
  literal ``signed-by:`` and ``signed-at:`` lines populated.
* **AC7 / R11** — caller must pass ``accept_tos_risk=True`` (default off).
* **AC14** — ``DGMH_GLM_DATA_V1_ENABLE_NETWORK=1`` must be set in the env,
  unless ``dry_run=True`` (which only prints the call plan).

The yt-dlp invocation is injectable via the ``subprocess_runner`` parameter
so tests never hit the network or shell out to a real ``yt-dlp``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from .glm_client import NetworkDeniedError
from .pii_redact import redact_kakao_text

# AC7: cap output excerpts at 200 chars.
EXCERPT_CAP_CHARS = 200

# Step 8 acceptance: per-run rate cap of ≤3 videos.
MAX_VIDEOS_PER_RUN = 3

# Mirror the env flag name from glm_client (avoids accidental drift).
_ENABLE_NETWORK_ENV = "DGMH_GLM_DATA_V1_ENABLE_NETWORK"

# J6: consent-doc grep targets.
_SIGNED_BY_RE = re.compile(r"^signed-by:\s*(.+?)\s*$", re.MULTILINE)
_SIGNED_AT_RE = re.compile(r"^signed-at:\s*(\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE)
# Placeholder values from the unsigned template — must NOT count as signed.
_PLACEHOLDER_TOKENS = ("<REPLACE", "TODO", "FIXME", "operator-name")


# --------------------------------------------------------------- error classes


class YouTubeCollectorError(RuntimeError):
    """Base error for ``youtube_collector``."""


class ConsentMissingError(YouTubeCollectorError):
    """Raised when the operator consent doc is missing or unsigned (J6)."""


class ToSRiskNotAcceptedError(YouTubeCollectorError):
    """Raised when ``accept_tos_risk`` is not explicitly True (AC7)."""


class TooManyVideosError(YouTubeCollectorError):
    """Raised when the caller exceeds ``MAX_VIDEOS_PER_RUN`` (Step 8 cap)."""


# --------------------------------------------------------------- consent check


def _read_consent_signature(consent_path: Path) -> tuple[str, str]:
    """Return ``(signed_by, signed_at)`` or raise ``ConsentMissingError``.

    The consent doc is the unsigned template at ``dgmh/docs/yt-dlp-consent.md``
    until the operator replaces both placeholder lines.
    """

    if not consent_path.is_file():
        raise ConsentMissingError(
            f"yt-dlp consent doc not found at {consent_path}; refusing to run "
            f"(see plan §4 Step 8 / J6)."
        )
    text = consent_path.read_text(encoding="utf-8")

    by_match = _SIGNED_BY_RE.search(text)
    at_match = _SIGNED_AT_RE.search(text)
    if not by_match or not at_match:
        raise ConsentMissingError(
            f"consent doc {consent_path} is missing 'signed-by:' or "
            f"'signed-at: YYYY-MM-DD' lines (J6)."
        )

    signed_by = by_match.group(1).strip()
    signed_at = at_match.group(1).strip()
    if any(token in signed_by for token in _PLACEHOLDER_TOKENS):
        raise ConsentMissingError(
            f"consent doc {consent_path} still has placeholder signed-by; "
            f"replace it with the operator's name (J6)."
        )
    if not signed_by:
        raise ConsentMissingError(
            f"consent doc {consent_path} has empty signed-by (J6)."
        )
    return signed_by, signed_at


# ------------------------------------------------------------- yt-dlp adapter


SubprocessRunner = Callable[[list[str], Path], subprocess.CompletedProcess]


def _default_yt_dlp_runner(
    argv: list[str], cwd: Path
) -> subprocess.CompletedProcess:
    """Default runner — shells out to ``yt-dlp``. Replaced by tests."""

    return subprocess.run(
        argv, cwd=cwd, check=True, capture_output=True, text=True, timeout=120
    )


def _yt_dlp_argv(video_id: str) -> list[str]:
    """Build the yt-dlp argv. Kept as a module-level helper for visibility."""

    # Output template constrains filenames to the video id so we can find
    # the produced .info.json deterministically.
    return [
        "yt-dlp",
        "--skip-download",
        "--write-comments",
        "--write-info-json",
        "--no-warnings",
        "-o",
        "%(id)s.%(ext)s",
        f"https://www.youtube.com/watch?v={video_id}",
    ]


# ------------------------------------------------------------------- helpers


def _hash_author(author: str) -> str:
    """sha256(author)[:12] — never persist raw author display names (AC7)."""

    return hashlib.sha256(author.encode("utf-8")).hexdigest()[:12]


_HANGUL_RE = re.compile(r"[가-힯]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def _lang_detect(text: str) -> str:
    """Cheap Hangul/Latin heuristic (no langdetect dep)."""

    if _HANGUL_RE.search(text):
        return "ko"
    if _LATIN_RE.search(text):
        return "en"
    return "und"


def _cap(text: str) -> str:
    """Truncate to ``EXCERPT_CAP_CHARS`` characters (AC7)."""

    if len(text) <= EXCERPT_CAP_CHARS:
        return text
    return text[:EXCERPT_CAP_CHARS]


# ----------------------------------------------------------------- main class


@dataclass
class CallPlan:
    """Dry-run summary returned in lieu of an actual yt-dlp invocation."""

    video_ids: list[str]
    consent_signed_by: str
    consent_signed_at: str
    accept_tos_risk: bool
    network_enabled: bool
    argv_per_video: list[list[str]]


class YouTubeCollector:
    """Collect, redact, and cap YouTube comments for v1 staging.

    Construction is side-effect free. ``collect`` (and ``plan``) perform all
    three pre-flight gates before doing any work. Subprocess access is
    fully injectable so tests run hermetically.
    """

    def __init__(
        self,
        consent_path: Path,
        *,
        subprocess_runner: SubprocessRunner = _default_yt_dlp_runner,
        now: Callable[[], _dt.datetime] = lambda: _dt.datetime.now(_dt.timezone.utc),
    ) -> None:
        self._consent_path = Path(consent_path)
        self._runner = subprocess_runner
        self._now = now

    # ----- gate orchestration -----

    def _preflight(
        self,
        video_ids: list[str],
        *,
        accept_tos_risk: bool,
        require_network: bool,
    ) -> tuple[str, str]:
        if not accept_tos_risk:
            raise ToSRiskNotAcceptedError(
                "yt-dlp ToS risk not accepted: pass accept_tos_risk=True "
                "(default OFF) and verify dgmh/docs/yt-dlp-consent.md (AC7)."
            )
        signed_by, signed_at = _read_consent_signature(self._consent_path)

        if len(video_ids) > MAX_VIDEOS_PER_RUN:
            raise TooManyVideosError(
                f"requested {len(video_ids)} videos; v1 cap is "
                f"{MAX_VIDEOS_PER_RUN} per run."
            )

        if require_network and os.environ.get(_ENABLE_NETWORK_ENV) != "1":
            raise NetworkDeniedError(
                f"network transport refused: set {_ENABLE_NETWORK_ENV}=1 to "
                f"opt in (plan §3 AC14)."
            )
        return signed_by, signed_at

    # ----- public API -----

    def plan(self, video_ids: list[str], *, accept_tos_risk: bool) -> CallPlan:
        """Dry-run: emit the call plan without executing yt-dlp."""

        signed_by, signed_at = self._preflight(
            video_ids, accept_tos_risk=accept_tos_risk, require_network=False
        )
        return CallPlan(
            video_ids=list(video_ids),
            consent_signed_by=signed_by,
            consent_signed_at=signed_at,
            accept_tos_risk=accept_tos_risk,
            network_enabled=os.environ.get(_ENABLE_NETWORK_ENV) == "1",
            argv_per_video=[_yt_dlp_argv(vid) for vid in video_ids],
        )

    def collect(
        self,
        video_ids: list[str],
        *,
        accept_tos_risk: bool,
        gap_target: Optional[str] = None,
    ) -> list[dict]:
        """Run yt-dlp per video, redact + cap each comment, return rows.

        Each row schema (matches AC7):
        ``{source, video_id, author_hash, text, lang_detect, gap_target,
        collected_at}``
        """

        self._preflight(
            video_ids, accept_tos_risk=accept_tos_risk, require_network=True
        )

        rows: list[dict] = []
        collected_at = self._now().replace(microsecond=0).isoformat()

        for video_id in video_ids:
            with tempfile.TemporaryDirectory(prefix=f"yt_dlp_{video_id}_") as td:
                tmpdir = Path(td)
                self._runner(_yt_dlp_argv(video_id), tmpdir)
                rows.extend(
                    self._parse_info_json(
                        tmpdir,
                        video_id=video_id,
                        collected_at=collected_at,
                        gap_target=gap_target,
                    )
                )
        return rows

    # ----- info.json parsing -----

    def _parse_info_json(
        self,
        tmpdir: Path,
        *,
        video_id: str,
        collected_at: str,
        gap_target: Optional[str],
    ) -> Iterable[dict]:
        """Read every ``*.info.json`` under ``tmpdir`` and yield redacted rows."""

        for info_path in sorted(tmpdir.glob("*.info.json")):
            payload = json.loads(info_path.read_text(encoding="utf-8"))
            comments = payload.get("comments") or []
            for comment in comments:
                text = comment.get("text") or ""
                author = comment.get("author") or comment.get("author_id") or ""
                if not text.strip():
                    continue
                redacted, _trace = redact_kakao_text(text, name_map={})
                row = {
                    "source": "youtube",
                    "video_id": video_id,
                    "author_hash": _hash_author(author),
                    "text": _cap(redacted),
                    "lang_detect": _lang_detect(redacted),
                    "gap_target": gap_target,
                    "collected_at": collected_at,
                }
                yield row


__all__ = [
    "YouTubeCollector",
    "YouTubeCollectorError",
    "ConsentMissingError",
    "ToSRiskNotAcceptedError",
    "TooManyVideosError",
    "CallPlan",
    "EXCERPT_CAP_CHARS",
    "MAX_VIDEOS_PER_RUN",
]
