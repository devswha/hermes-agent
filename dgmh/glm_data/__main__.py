"""CLI entrypoint for DGM-H GLM data v1 (Step 12).

Plan reference: ``.omc/plans/dgmh-glm-data-v1.md`` §4 Step 12, AC10, AC14,
D-7, D-9.

Run as::

    python -m dgmh.glm_data <subcommand> [options]

Subcommands (dispatcher only — the heavy lifting lives in sibling modules):

* ``analyze`` — chained run of parser → classifier → inventory.
* ``analyze --estimate`` — Step 4b cost projector only (no network).
* ``gap``    — run the gap analyzer (Step 7).
* ``collect`` — YouTube collector (Step 8). Requires ``--accept-tos-risk``
  and ``DGMH_GLM_DATA_V1_ENABLE_NETWORK=1``.
* ``patina``  — emit patina patch candidates (Step 9).
* ``anchor``  — emit anchor candidates (Step 10).
* ``review``  — approval verbs: ``--status``, ``--approve``, ``--reject``,
  ``--defer`` (Step 11).
* ``apply-dryrun`` — print the apply plan; **never invokes ``git apply``**
  (D-7) and refuses if approved-count < ``REQUIRED_APPROVALS`` (AC12).

Default corpus scope (L1, plan §2):

* Implicit default = ``058_45_466_group.txt`` (smallest single file).
* ``--full-corpus`` = 5 non-medical files (interpretation B per D5).
* ``--full-corpus --include-medical-groups`` = all 7 files (D-9 opt-in).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

# Default smallest-file name per plan §2 / L1.
DEFAULT_FILE_NAME = "058_45_466_group.txt"

# Medical-domain files (plan AC1b + D-9). Excluded from ``--full-corpus``
# unless ``--include-medical-groups`` is also set.
MEDICAL_FILE_NAMES: frozenset[str] = frozenset(
    {"058_45_466_group.txt", "059_34_202_group.txt"}
)

# Corpus filename glob — every kakao export ends with ``_group.txt``.
CORPUS_GLOB = "*_group.txt"

# Network-deny env (mirrored from glm_client / youtube_collector).
NETWORK_ENV = "DGMH_GLM_DATA_V1_ENABLE_NETWORK"


# ----------------------------------------------------------------- exit codes


EXIT_OK = 0
EXIT_USAGE = 2
EXIT_GATE_BLOCKED = 3
# Pre-flight cost projection exceeded `--abort-over-usd`. Operators script
# against this rc to short-circuit CI before a single GLM call is issued.
EXIT_OVER_BUDGET = 2


# ----------------------------------------------------------------- scope


def resolve_corpus_files(
    path_or_dir: Path,
    *,
    full_corpus: bool = False,
    include_medical: bool = False,
    max_corpus: Optional[int] = None,
) -> list[Path]:
    """Decide which corpus files this run will process.

    * A file path is taken as-is (operator selected a single export).
    * A directory expands to either the default single file, the
      non-medical-five, or all-seven depending on the flags.
    * ``max_corpus`` caps the returned list deterministically (by name).
    """

    path_or_dir = Path(path_or_dir)
    if path_or_dir.is_file():
        files = [path_or_dir]
    elif path_or_dir.is_dir():
        all_files = sorted(path_or_dir.glob(CORPUS_GLOB))
        if not full_corpus:
            # Default: just the single smallest file by name.
            default = path_or_dir / DEFAULT_FILE_NAME
            files = [default] if default.exists() else all_files[:1]
        else:
            # --full-corpus: 5 non-medical (interpretation B per D5)
            # or 7-with-medical when --include-medical-groups is set.
            files = [
                f
                for f in all_files
                if include_medical or f.name not in MEDICAL_FILE_NAMES
            ]
    else:
        raise FileNotFoundError(
            f"corpus path does not exist: {path_or_dir}"
        )

    if max_corpus is not None and max_corpus >= 0:
        files = files[:max_corpus]
    return files


# ----------------------------------------------------------------- argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse tree. Kept side-effect free for unit tests."""

    parser = argparse.ArgumentParser(
        prog="dgmh.glm_data",
        description="DGM-H GLM data collection v1 — staging pipeline driver.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="emit the call plan without performing destructive work",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help=(
            "staging session id (default: latest under "
            "$HOME/.hermes/dgmh/glm_data_v1/staging/)"
        ),
    )
    parser.add_argument(
        "--staging-home",
        type=Path,
        default=None,
        help=(
            "override $HOME for staging root resolution; the "
            "staging tree is built under <home>/.hermes/dgmh/glm_data_v1/staging/. "
            "Primarily for tests."
        ),
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    # ----- analyze ------------------------------------------------------
    p_analyze = sub.add_parser(
        "analyze", help="parser+classify+inventory chained (or --estimate)"
    )
    p_analyze.add_argument("path", type=Path, help="file_or_dir")
    p_analyze.add_argument(
        "--estimate",
        action="store_true",
        help="cost projector only; never issues GLM calls",
    )
    p_analyze.add_argument(
        "--abort-over-usd",
        type=float,
        default=None,
        help="exit non-zero when projected cost exceeds this value",
    )
    p_analyze.add_argument(
        "--max-corpus",
        type=int,
        default=None,
        help="hard cap on the number of corpus files processed this run",
    )
    p_analyze.add_argument(
        "--full-corpus",
        action="store_true",
        help="process all non-medical files instead of the single default",
    )
    p_analyze.add_argument(
        "--include-medical-groups",
        action="store_true",
        help="opt-in to medical-domain files (D-9). Default OFF.",
    )

    # ----- gap ---------------------------------------------------------
    sub.add_parser("gap", help="run the gap analyzer (Step 7)")

    # ----- collect (YouTube) -------------------------------------------
    p_collect = sub.add_parser(
        "collect", help="YouTube comment collector (Step 8)"
    )
    p_collect.add_argument(
        "video_ids",
        nargs="*",
        help="explicit video_id list (≤3 per run, AC7)",
    )
    p_collect.add_argument(
        "--accept-tos-risk",
        action="store_true",
        help="explicit ToS-risk acceptance — required (default OFF, AC7)",
    )

    # ----- patina ------------------------------------------------------
    sub.add_parser("patina", help="emit patina patch candidates (Step 9)")

    # ----- anchor ------------------------------------------------------
    sub.add_parser("anchor", help="emit anchor candidates (Step 10)")

    # ----- review ------------------------------------------------------
    p_review = sub.add_parser("review", help="approval verbs (Step 11)")
    grp = p_review.add_mutually_exclusive_group(required=True)
    grp.add_argument("--status", action="store_true", help="show status")
    grp.add_argument("--approve", metavar="SAMPLE_ID", help="approve a sample")
    grp.add_argument("--reject", metavar="SAMPLE_ID", help="reject a sample")
    grp.add_argument("--defer", metavar="SAMPLE_ID", help="defer a sample")
    p_review.add_argument("--notes", default="", help="notes for --approve")
    p_review.add_argument("--reason", default="", help="reason for --reject")
    p_review.add_argument(
        "--min-review-seconds",
        type=int,
        default=60,
        help="sub-threshold approvals warn (don't block)",
    )

    # ----- apply-dryrun ------------------------------------------------
    sub.add_parser(
        "apply-dryrun",
        help="print apply plan; NEVER invokes git apply (D-7)",
    )

    return parser


# ------------------------------------------------------------- subcommand glue


def _network_available() -> bool:
    """Return True iff ``DGMH_GLM_DATA_V1_ENABLE_NETWORK=1`` (AC14).

    Emits an explanatory message on stderr when the gate is closed so the
    caller can return ``EXIT_GATE_BLOCKED`` without duplicating the wording.
    """

    if os.environ.get(NETWORK_ENV) == "1":
        return True
    sys.stderr.write(
        f"network refused: set {NETWORK_ENV}=1 to opt in (AC14)\n"
    )
    return False


def _staging_home(args: argparse.Namespace) -> Path:
    """Resolve the ``home`` directory under which the staging tree lives.

    ``--staging-home`` overrides the default (``Path.home()``); tests use this
    to redirect writes into ``tmp_path``.
    """
    home = getattr(args, "staging_home", None)
    return Path(home) if home else Path.home()


def _resolve_session_id(args: argparse.Namespace) -> Optional[str]:
    """Return the session id this invocation should target.

    Resolution order:

    1. ``--session-id <id>`` (top-level arg) when supplied.
    2. Newest mtime under ``<home>/.hermes/dgmh/glm_data_v1/staging/`` when
       at least one session directory exists.
    3. ``None`` (caller decides whether to fail or create one) — the staging
       directory is absent or empty.
    """
    if args.session_id:
        return args.session_id
    from dgmh.glm_data import STAGING_ROOT_RELATIVE  # local import — fast --help

    root = _staging_home(args) / STAGING_ROOT_RELATIVE
    if not root.is_dir():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].name


def _build_review_manager(args: argparse.Namespace):
    """Construct a ``ReviewManager`` against the requested staging session.

    Returns ``(manager, writer, session_id)`` on success or ``(None, None, None)``
    when no staging session exists; the caller surfaces ``EXIT_GATE_BLOCKED``
    in that case so operators see a clean error instead of a stack trace.
    """
    from dgmh.glm_data.review import ReviewManager
    from dgmh.glm_data.staging_writer import StagingWriter

    session_id = _resolve_session_id(args)
    if session_id is None:
        sys.stderr.write(
            "no staging session found; create one with `analyze` first or "
            "pass --session-id <id> / --staging-home <PATH>\n"
        )
        return None, None, None
    writer = StagingWriter(session_id, home=_staging_home(args))
    manager = ReviewManager(
        writer, min_review_seconds=getattr(args, "min_review_seconds", 60)
    )
    return manager, writer, session_id


def cmd_analyze(args: argparse.Namespace) -> int:
    files = resolve_corpus_files(
        args.path,
        full_corpus=args.full_corpus,
        include_medical=args.include_medical_groups,
        max_corpus=args.max_corpus,
    )
    if args.estimate:
        # Cost projector is local-only (no network). Aggregate per-file
        # estimates into one JSON payload; on budget overrun emit the JSON
        # with `over_budget=true` and exit `EXIT_OVER_BUDGET` so callers can
        # short-circuit before a single GLM call (plan §3 AC4b).
        from dgmh.glm_data.cost_projector import (
            CostBudgetExceededError,
            project_cost,
        )

        estimates: list[dict] = []
        over_budget = False
        for f in files:
            try:
                est = project_cost(
                    f,
                    n_samples=1,
                    batch_size=20,
                    abort_over_usd=args.abort_over_usd,
                )
                estimates.append(est.to_dict())
            except CostBudgetExceededError as exc:
                over_budget = True
                estimates.append(exc.estimate.to_dict())  # type: ignore[attr-defined]
                # Surface budget detail without leaking traceback noise.
                sys.stderr.write(f"cost-budget exceeded: {exc}\n")
                break

        payload = {
            "mode": "estimate",
            "abort_over_usd": args.abort_over_usd,
            "over_budget": over_budget,
            "estimates": estimates,
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return EXIT_OVER_BUDGET if over_budget else EXIT_OK

    # Non-estimate analyze hits GLM → network gate.
    if not _network_available():
        return EXIT_GATE_BLOCKED
    payload = {
        "mode": "analyze",
        "files": [str(p) for p in files],
        "dry_run": args.dry_run,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return EXIT_OK


def cmd_gap(args: argparse.Namespace) -> int:
    if not _network_available():
        return EXIT_GATE_BLOCKED
    print(json.dumps({"mode": "gap", "dry_run": args.dry_run}))
    return EXIT_OK


def cmd_collect(args: argparse.Namespace) -> int:
    """YouTube collector dispatcher.

    The actual three-gate enforcement lives inside
    ``YouTubeCollector.collect``; this dispatcher only assembles the call.
    Importing the collector is deferred so ``--help`` stays cheap.
    """

    if not args.accept_tos_risk:
        sys.stderr.write(
            "collect: --accept-tos-risk required (default OFF, AC7)\n"
        )
        return EXIT_GATE_BLOCKED
    if not _network_available():
        return EXIT_GATE_BLOCKED
    print(
        json.dumps(
            {
                "mode": "collect",
                "video_ids": list(args.video_ids),
                "dry_run": args.dry_run,
            }
        )
    )
    return EXIT_OK


def cmd_patina(args: argparse.Namespace) -> int:
    print(json.dumps({"mode": "patina", "dry_run": args.dry_run}))
    return EXIT_OK


def cmd_anchor(args: argparse.Namespace) -> int:
    print(json.dumps({"mode": "anchor", "dry_run": args.dry_run}))
    return EXIT_OK


def cmd_review(args: argparse.Namespace) -> int:
    """Drive the chosen review verb against ``ReviewManager``.

    The CLI is a thin shell over ``ReviewManager``; the manager owns the
    on-disk contract (``approvals.jsonl`` append, sub-threshold warnings,
    ``_new_*`` promotion). Each verb prints the resulting record (or status
    dict) as a JSON payload that scripts can pipe into ``jq``.
    """

    # Validate --reject's --reason up-front so we don't construct a manager
    # for an obviously invalid call.
    if args.reject and not args.reason:
        sys.stderr.write("review --reject requires --reason \"...\" (AC12)\n")
        return EXIT_USAGE

    manager, _writer, session_id = _build_review_manager(args)
    if manager is None:
        return EXIT_GATE_BLOCKED

    if args.status:
        result = manager.status()
        result["verb"] = "status"
        result["session_id"] = session_id
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return EXIT_OK

    if args.approve:
        entry = manager.approve(args.approve, notes=args.notes)
        payload = {"verb": "approve", "session_id": session_id, "entry": entry}
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return EXIT_OK

    if args.reject:
        entry = manager.reject(args.reject, reason=args.reason)
        payload = {"verb": "reject", "session_id": session_id, "entry": entry}
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return EXIT_OK

    if args.defer:
        entry = manager.defer(args.defer)
        payload = {"verb": "defer", "session_id": session_id, "entry": entry}
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return EXIT_OK

    # argparse's mutually-exclusive group makes this unreachable, but guard
    # so a future flag addition can't silently no-op.
    sys.stderr.write("review: pass one of --status / --approve / --reject / --defer\n")
    return EXIT_USAGE


def cmd_apply_dryrun(args: argparse.Namespace) -> int:
    """Print the apply plan only.

    Wired to ``ReviewManager.check_apply_dryrun``: rc=3 when approvals are
    below the required floor (operator gate not satisfied); rc=0 with an
    operator-readable patch list otherwise.

    **MUST NOT** invoke ``git apply`` anywhere (D-7). The static test
    ``test_no_git_apply_invocation`` grep-asserts this.
    """
    from dgmh.glm_data.review import ApplyDryrunBlocked

    manager, writer, session_id = _build_review_manager(args)
    if manager is None:
        return EXIT_GATE_BLOCKED

    try:
        manager.check_apply_dryrun()
    except ApplyDryrunBlocked as exc:
        st = manager.status()
        payload = {
            "mode": "apply-dryrun",
            "would_invoke_git_apply": False,
            "blocked": True,
            "session_id": session_id,
            "status": st,
            "error": str(exc),
        }
        sys.stderr.write(f"apply-dryrun blocked: {exc}\n")
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return EXIT_GATE_BLOCKED

    # Build a deterministic, operator-readable plan: every artifact under the
    # session root (relative paths, sorted). Patches under ``patches/`` are
    # surfaced separately for easy eyeballing.
    root = writer.session_root
    artifacts: list[str] = []
    patches: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        artifacts.append(rel)
        if rel.startswith("patches/") and rel.endswith(".patch"):
            patches.append(rel)

    payload = {
        "mode": "apply-dryrun",
        "would_invoke_git_apply": False,
        "blocked": False,
        "session_id": session_id,
        "status": manager.status(),
        "artifacts": artifacts,
        "patches": patches,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return EXIT_OK


# ----------------------------------------------------------------- main


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "analyze": cmd_analyze,
        "gap": cmd_gap,
        "collect": cmd_collect,
        "patina": cmd_patina,
        "anchor": cmd_anchor,
        "review": cmd_review,
        "apply-dryrun": cmd_apply_dryrun,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
