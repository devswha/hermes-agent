"""Tests for ``dgmh.glm_data.cost_projector``.

Covers plan §3 AC4b:

- determinism given fixed corpus input
- no network calls during projection (HTTP-level monkeypatch)
- ``abort_over_usd`` raises when budget exceeded
- ``warmup_then_median`` skips when benchmark binaries are absent
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dgmh.glm_data import cost_projector  # noqa: E402
from dgmh.glm_data.cost_projector import (  # noqa: E402
    BENCH_BINARIES,
    CostBudgetExceededError,
    CostEstimate,
    CostProjectorError,
    OUTPUT_FACTOR_CLASSIFY,
    project_cost,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_corpus(tmp_path: Path, *, lines: int = 100, body: str = "안녕하세요 반가워요") -> Path:
    """Write a fixed-content corpus file."""

    path = tmp_path / "corpus.txt"
    path.write_text("\n".join(f"{body} {i}" for i in range(lines)), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# AC4b: deterministic given fixed corpus
# ---------------------------------------------------------------------------


def test_projector_deterministic_given_fixed_corpus(tmp_path):
    corpus = _write_corpus(tmp_path, lines=200)

    est1 = project_cost(corpus, run_benchmark=False)
    est2 = project_cost(corpus, run_benchmark=False)

    assert isinstance(est1, CostEstimate)
    assert est1.to_dict() == est2.to_dict()

    # Output token count must be input × classify factor (1.5×).
    assert est1.output_tokens == int(round(est1.input_tokens * OUTPUT_FACTOR_CLASSIFY))
    # Call count matches ceil(lines / batch_size).
    assert est1.n_calls == (200 + est1.batch_size - 1) // est1.batch_size
    # Projection respects published GLM-4.5 pricing.
    expected_usd = round(
        est1.input_tokens * 0.5 / 1_000_000 + est1.output_tokens * 1.5 / 1_000_000,
        6,
    )
    assert est1.projected_usd == expected_usd


def test_projector_handles_directory(tmp_path):
    """A directory of *.txt files is summed deterministically."""

    (tmp_path / "a.txt").write_text("a\n" * 100, encoding="utf-8")
    (tmp_path / "b.txt").write_text("b\n" * 50, encoding="utf-8")
    # Non-.txt files are ignored.
    (tmp_path / "notes.md").write_text("should be ignored", encoding="utf-8")

    est = project_cost(tmp_path, run_benchmark=False)
    assert est.n_calls > 0
    assert est.input_tokens > 0


# ---------------------------------------------------------------------------
# AC4b: no network during projection
# ---------------------------------------------------------------------------


def test_projector_no_network(tmp_path, monkeypatch):
    """Projector must not open any sockets or invoke ``requests``."""

    corpus = _write_corpus(tmp_path, lines=50)

    # Warm the tiktoken encoder cache before we cut the network. tiktoken
    # downloads BPE merges on first use; once cached, every subsequent call
    # is offline, which is what the projector relies on at runtime.
    cost_projector._get_encoder()

    def _boom_socket(*args, **kwargs):  # pragma: no cover - guard
        raise AssertionError("cost_projector opened a socket — must be offline")

    monkeypatch.setattr(socket, "socket", _boom_socket)

    # Block ``requests`` if it gets imported and used.
    import requests

    def _boom_request(*args, **kwargs):  # pragma: no cover - guard
        raise AssertionError("cost_projector hit requests — must be offline")

    monkeypatch.setattr(requests.Session, "request", _boom_request)
    monkeypatch.setattr(requests, "post", _boom_request, raising=False)
    monkeypatch.setattr(requests, "get", _boom_request, raising=False)

    # Also pin enable-network OFF to mirror real operator default.
    monkeypatch.delenv("DGMH_GLM_DATA_V1_ENABLE_NETWORK", raising=False)

    est = project_cost(corpus, run_benchmark=False)
    assert est.projected_usd >= 0.0
    assert est.n_calls > 0


# ---------------------------------------------------------------------------
# AC4b: --abort-over-usd
# ---------------------------------------------------------------------------


def test_abort_over_usd_raises_when_exceeded(tmp_path):
    corpus = _write_corpus(tmp_path, lines=500)

    # First, project without a budget to learn the actual projection.
    baseline = project_cost(corpus, run_benchmark=False)
    # Pick a budget well below the projection — must raise.
    budget = baseline.projected_usd / 2

    # Guard the test against the corner case where projection is so small the
    # halved budget rounds to zero but projection is also zero.
    if baseline.projected_usd == 0:
        pytest.skip("corpus too small to produce a non-zero projection")

    with pytest.raises(CostBudgetExceededError) as excinfo:
        project_cost(corpus, run_benchmark=False, abort_over_usd=budget)

    # The exception carries the full estimate for the CLI to emit.
    estimate = getattr(excinfo.value, "estimate", None)
    assert isinstance(estimate, CostEstimate)
    assert estimate.projected_usd > budget


def test_abort_over_usd_passes_when_under_budget(tmp_path):
    corpus = _write_corpus(tmp_path, lines=10)
    baseline = project_cost(corpus, run_benchmark=False)
    generous = baseline.projected_usd + 1_000.0
    # Should not raise.
    est = project_cost(corpus, run_benchmark=False, abort_over_usd=generous)
    assert est.projected_usd <= generous


# ---------------------------------------------------------------------------
# AC4b: warmup-then-median skips when binaries absent
# ---------------------------------------------------------------------------


def test_warmup_then_median_skips_when_binaries_absent(tmp_path, monkeypatch):
    """If neither benchmark binary is on PATH, medians are None and wall is None."""

    corpus = _write_corpus(tmp_path, lines=20)

    def _absent(_binary: str) -> None:
        return None

    monkeypatch.setattr(cost_projector, "_which", _absent)

    def _boom_run(*args, **kwargs):  # pragma: no cover - guard
        raise AssertionError("subprocess.run must not be called when binaries are absent")

    monkeypatch.setattr(cost_projector.subprocess, "run", _boom_run)

    est = project_cost(corpus, run_benchmark=True)

    for binary in BENCH_BINARIES:
        assert est.per_call_median_s[binary] is None
    assert est.projected_wall_seconds is None


# ---------------------------------------------------------------------------
# MAJOR-1 (codex review): tiktoken lazy import
# ---------------------------------------------------------------------------


def test_module_imports_without_tiktoken(monkeypatch):
    """Importing ``cost_projector`` must succeed even if tiktoken is absent.

    Before MAJOR-1 fix, top-level ``import tiktoken`` made the entire module
    un-importable without the optional dependency. After the fix, the import
    is deferred to ``_get_encoder()`` so callers that don't project costs
    (e.g. ``--skip-cost-estimate``) are not punished.
    """

    import importlib

    # Hide any already-loaded tiktoken from a fresh import.
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    # Drop the projector itself so reimport actually re-executes module-level
    # code (otherwise the cached module satisfies the import trivially).
    monkeypatch.delitem(sys.modules, "dgmh.glm_data.cost_projector", raising=False)

    reloaded = importlib.import_module("dgmh.glm_data.cost_projector")
    assert hasattr(reloaded, "CostProjectorError")
    assert hasattr(reloaded, "project_cost")


def test_get_encoder_raises_cost_projector_error_when_tiktoken_missing(monkeypatch):
    """When tiktoken is missing, _get_encoder must raise CostProjectorError
    (catchable) rather than ImportError (would crash the CLI)."""

    # Clear the encoder cache so the lazy import path actually runs.
    monkeypatch.setattr(cost_projector, "_encoder_cache", {})
    # Force ImportError for any future ``import tiktoken``.
    monkeypatch.setitem(sys.modules, "tiktoken", None)

    with pytest.raises(CostProjectorError) as excinfo:
        cost_projector._get_encoder()

    # Operator-facing guidance must be present so the CLI message is useful.
    assert "tiktoken" in str(excinfo.value)
    assert "pip install" in str(excinfo.value) or "--skip-cost-estimate" in str(excinfo.value)


def test_project_cost_raises_cost_projector_error_when_tiktoken_missing(
    tmp_path, monkeypatch
):
    """End-to-end: project_cost() surfaces the lazy-import failure as
    CostProjectorError, not as a bare ModuleNotFoundError."""

    corpus = _write_corpus(tmp_path, lines=5)

    monkeypatch.setattr(cost_projector, "_encoder_cache", {})
    monkeypatch.setitem(sys.modules, "tiktoken", None)

    with pytest.raises(CostProjectorError):
        project_cost(corpus, run_benchmark=False)


def test_warmup_then_median_uses_present_binary(tmp_path, monkeypatch):
    """When a binary is present, its median latency feeds the wall-clock estimate."""

    corpus = _write_corpus(tmp_path, lines=20)

    monkeypatch.setattr(
        cost_projector,
        "_which",
        lambda b: f"/usr/local/bin/{b}" if b == "claude_p" else None,
    )

    # Each (binary, prompt) invocation returns the same fake elapsed seconds.
    call_log: list[str] = []

    fake_clock = {"now": 100.0}

    def _fake_perf_counter() -> float:
        # Bump by exactly 0.25s every odd call so each subprocess.run measures 0.25s.
        fake_clock["now"] += 0.125
        return fake_clock["now"]

    def _fake_run(*args, **kwargs):
        call_log.append(args[0][0] if args and args[0] else "unknown")

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""

        return _R()

    monkeypatch.setattr(cost_projector.time, "perf_counter", _fake_perf_counter)
    monkeypatch.setattr(cost_projector.subprocess, "run", _fake_run)

    est = project_cost(corpus, run_benchmark=True)

    assert est.per_call_median_s["claude_p"] is not None
    assert est.per_call_median_s["hermes_chat"] is None
    assert est.projected_wall_seconds is not None
    # Only the present binary should have been exercised.
    assert all(name == "claude_p" for name in call_log)
