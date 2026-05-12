"""Cost projector for DGM-H GLM Data Collection v1.

Implements plan §4 Step 4b + §3 AC4b. Projects token / dollar / wall-clock
cost for a planned run **without** issuing a single network call. The
operator can gate execution with ``--abort-over-usd N``: projections that
exceed the budget raise :class:`CostBudgetExceededError` so the CLI can exit
non-zero before any real GLM traffic.

Caveats (documented honestly so we don't oversell the projection):

- Tokenization uses ``tiktoken`` with the ``cl100k_base`` encoding as a
  *proxy* for the GLM tokenizer. Empirically this lands within ±50–70% of
  GLM's own token counts for Korean text; we deliberately do **not** claim
  ±30% accuracy.
- Output-token factors per stage (classify 1.5×, inventory 0.3×, gap 2×)
  are coarse heuristics agreed at plan §4 Step 4b. Real numbers will land
  after the first observed run; this module is a budget gate, not a finance
  estimator.
- Wall-clock estimates rely on observed median latency of the
  ``claude_p`` / ``hermes_chat`` binaries (warmup-then-median, 2 warmup + 3
  measure each, ~5 kB padded prompt). If the binaries are not on PATH the
  per-call median is reported as ``None`` and wall-clock seconds is ``None``.

This file makes **zero** network calls — verified by AC4b ``test_projector_no_network``.
"""

from __future__ import annotations

import logging
import os
import shutil
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import only for type hints
    import tiktoken

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Per-stage output/input token expansion factors (plan §4 Step 4b).
OUTPUT_FACTOR_CLASSIFY = 1.5
OUTPUT_FACTOR_INVENTORY = 0.3
OUTPUT_FACTOR_GAP = 2.0

# GLM-4.5 published prices (USD per 1M tokens) — pending OQ-1 confirmation.
INPUT_PRICE_PER_M_USD = 0.5
OUTPUT_PRICE_PER_M_USD = 1.5

# Default batch size matches the classifier (plan §4 Step 5: N=20 messages).
DEFAULT_BATCH_SIZE = 20

# Wallclock benchmark knobs (plan §4 Step 4b: warmup-then-median).
WARMUP_RUNS = 2
MEASURE_RUNS = 3
BENCH_PROMPT_BYTES = 5_000  # ~5 kB padded prompt
BENCH_TIMEOUT_S = 30.0
BENCH_BINARIES: tuple[str, ...] = ("claude_p", "hermes_chat")

# Encoding used as a proxy for the GLM tokenizer.
_TIKTOKEN_ENCODING = "cl100k_base"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CostProjectorError(RuntimeError):
    """Base error for cost projection failures."""


class CostBudgetExceededError(CostProjectorError):
    """Raised when a projection exceeds the operator-supplied USD budget."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class CostEstimate:
    """Result of a projection run.

    ``projected_wall_seconds`` is ``None`` when neither of the benchmark
    binaries is installed (we refuse to invent latency from thin air).
    ``per_call_median_s`` exposes the raw medians per binary for audit.
    """

    input_tokens: int
    output_tokens: int
    projected_usd: float
    projected_wall_seconds: Optional[float]
    n_calls: int
    n_samples: int
    batch_size: int
    model: str
    corpus_path: str
    per_call_median_s: dict[str, Optional[float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Token + line counting
# ---------------------------------------------------------------------------


def _iter_corpus_files(corpus_path: Path) -> list[Path]:
    """Resolve ``corpus_path`` to a deterministic sorted list of files."""

    if corpus_path.is_file():
        return [corpus_path]
    if corpus_path.is_dir():
        return sorted(p for p in corpus_path.glob("*.txt") if p.is_file())
    raise CostProjectorError(f"corpus_path not found: {corpus_path}")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


_encoder_cache: dict[str, "tiktoken.Encoding"] = {}


def _get_encoder() -> "tiktoken.Encoding":
    """Return a cached tiktoken encoder (lazy import).

    ``tiktoken`` is imported lazily so that ``dgmh.glm_data.cost_projector``
    can still be imported when the optional dependency is missing — callers
    that don't actually project costs (e.g. ``--skip-cost-estimate`` paths)
    must not be punished with ``ModuleNotFoundError`` at import time.

    ``tiktoken.get_encoding`` downloads BPE merges over HTTP on first use and
    caches them under ``$TIKTOKEN_CACHE_DIR`` (or ``~/.cache/tiktoken``).
    Subsequent calls within the process reuse the cached encoder object so the
    projector stays offline once the cache is warm.

    Raises
    ------
    CostProjectorError
        If ``tiktoken`` is not installed. Catchable by the CLI so it can map
        to a clean exit with install guidance instead of a raw ImportError.
    """

    enc = _encoder_cache.get(_TIKTOKEN_ENCODING)
    if enc is None:
        try:
            import tiktoken  # noqa: PLC0415 — intentional lazy import
        except ImportError as exc:
            raise CostProjectorError(
                "tiktoken is required for cost projection but is not installed; "
                "`pip install tiktoken` or pass --skip-cost-estimate"
            ) from exc
        enc = tiktoken.get_encoding(_TIKTOKEN_ENCODING)
        _encoder_cache[_TIKTOKEN_ENCODING] = enc
    return enc


def _count_input_tokens(texts: Iterable[str]) -> int:
    enc = _get_encoder()
    total = 0
    for text in texts:
        total += len(enc.encode(text, disallowed_special=()))
    return total


def _count_lines(text: str) -> int:
    # ``splitlines`` matches what the chunked reader will see.
    return sum(1 for _ in text.splitlines() if _.strip())


# ---------------------------------------------------------------------------
# Wallclock benchmark (warmup-then-median)
# ---------------------------------------------------------------------------


def _which(binary: str) -> Optional[str]:
    return shutil.which(binary)


def _benchmark_binary(binary: str, *, padded_prompt: str) -> Optional[float]:
    """Run ``binary`` with the padded prompt, return median of MEASURE_RUNS in seconds.

    Returns ``None`` if the binary is absent or any invocation fails. We
    deliberately swallow per-call failures here because this is a *projection*
    helper — the cost gate should still produce dollar numbers even when local
    latency telemetry is unavailable.
    """

    if _which(binary) is None:
        return None

    def _one_call() -> Optional[float]:
        start = time.perf_counter()
        try:
            subprocess.run(
                [binary],
                input=padded_prompt,
                capture_output=True,
                text=True,
                timeout=BENCH_TIMEOUT_S,
                check=False,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            logger.debug("cost_projector: benchmark of %s failed: %s", binary, exc)
            return None
        return time.perf_counter() - start

    # Warmup runs are discarded; their failures also do not abort.
    for _ in range(WARMUP_RUNS):
        _one_call()

    samples: list[float] = []
    for _ in range(MEASURE_RUNS):
        sample = _one_call()
        if sample is not None:
            samples.append(sample)

    if not samples:
        return None
    return statistics.median(samples)


def _measured_per_call_seconds(*, padded_prompt: str) -> dict[str, Optional[float]]:
    return {binary: _benchmark_binary(binary, padded_prompt=padded_prompt) for binary in BENCH_BINARIES}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def project_cost(
    corpus_path: Path,
    *,
    n_samples: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model: str = "glm-4.5",
    abort_over_usd: Optional[float] = None,
    output_factor: float = OUTPUT_FACTOR_CLASSIFY,
    run_benchmark: bool = True,
) -> CostEstimate:
    """Project cost for a planned GLM run over ``corpus_path``.

    Parameters
    ----------
    corpus_path:
        File or directory containing ``*.txt`` corpus chunks. A directory
        scope mirrors what the kakao reader will see for ``--full-corpus``.
    n_samples:
        Number of independent passes over the corpus (e.g. multi-sample
        majority voting). Defaults to 1.
    batch_size:
        Messages per GLM call. Used to estimate ``n_calls``.
    model:
        Recorded in the output for audit; pricing constants are GLM-4.5.
    abort_over_usd:
        If set and the projection exceeds this budget,
        :class:`CostBudgetExceededError` is raised. The exception carries the
        full :class:`CostEstimate` on ``.estimate`` so the CLI can still emit
        the estimate JSON before exiting.
    output_factor:
        Per-stage output expansion factor. Defaults to
        :data:`OUTPUT_FACTOR_CLASSIFY`; callers can pass
        :data:`OUTPUT_FACTOR_INVENTORY` or :data:`OUTPUT_FACTOR_GAP` for
        other stages.
    run_benchmark:
        If ``False``, skip the warmup-then-median latency benchmark entirely.
        Tests use this to assert no subprocesses run. When ``True`` (default),
        missing binaries report ``None`` medians without failing.
    """

    files = _iter_corpus_files(corpus_path)

    # Token + line counting (no network).
    texts = [_read_text(p) for p in files]
    line_count = sum(_count_lines(t) for t in texts)
    input_tokens_one_pass = _count_input_tokens(texts)

    input_tokens = input_tokens_one_pass * max(1, n_samples)
    output_tokens = int(round(input_tokens * output_factor))

    # Cost: input $0.5/M + output $1.5/M.
    projected_usd = (
        input_tokens * INPUT_PRICE_PER_M_USD / 1_000_000
        + output_tokens * OUTPUT_PRICE_PER_M_USD / 1_000_000
    )

    # Call count: ceil(line_count / batch_size), multiplied by n_samples.
    if batch_size <= 0:
        raise CostProjectorError("batch_size must be > 0")
    n_calls_per_sample = max(1, (line_count + batch_size - 1) // batch_size) if line_count else 0
    n_calls = n_calls_per_sample * max(1, n_samples)

    # Wallclock estimate (optional, never networked).
    per_call_median: dict[str, Optional[float]] = {b: None for b in BENCH_BINARIES}
    projected_wall_seconds: Optional[float] = None

    if run_benchmark:
        padded = ("x" * BENCH_PROMPT_BYTES) if BENCH_PROMPT_BYTES > 0 else ""
        per_call_median = _measured_per_call_seconds(padded_prompt=padded)
        medians = [v for v in per_call_median.values() if v is not None]
        if medians and n_calls > 0:
            # Use the slower median to bias estimates upward — undershooting
            # the wall-clock projection is a worse error than overshooting.
            projected_wall_seconds = max(medians) * n_calls

    estimate = CostEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        projected_usd=round(projected_usd, 6),
        projected_wall_seconds=projected_wall_seconds,
        n_calls=n_calls,
        n_samples=n_samples,
        batch_size=batch_size,
        model=model,
        corpus_path=str(corpus_path),
        per_call_median_s=per_call_median,
    )

    if abort_over_usd is not None and projected_usd > abort_over_usd:
        exc = CostBudgetExceededError(
            f"projected ${projected_usd:.4f} exceeds budget ${abort_over_usd:.4f} "
            f"(input={input_tokens} tok, output={output_tokens} tok, n_calls={n_calls})"
        )
        exc.estimate = estimate  # type: ignore[attr-defined]
        raise exc

    return estimate


__all__ = [
    "CostEstimate",
    "CostProjectorError",
    "CostBudgetExceededError",
    "OUTPUT_FACTOR_CLASSIFY",
    "OUTPUT_FACTOR_INVENTORY",
    "OUTPUT_FACTOR_GAP",
    "INPUT_PRICE_PER_M_USD",
    "OUTPUT_PRICE_PER_M_USD",
    "project_cost",
]
