"""
dgmh/select_parents.py — DGM-H paper §8.2 parent selection (Python port).

Port note: Direct translation of playground/dgmh-engine/selectParents.ts.
  - sigmoid_weight():       selectParents.ts lines 22-28
  - novelty_weight():       selectParents.ts lines 30-32
  - combined_weight():      selectParents.ts lines 34-41
  - archive_top_midpoint(): selectParents.ts lines 43-52
  - sample_parent():        selectParents.ts lines 54-76
  - select_parents():       selectParents.ts lines 78-96

Type contracts from playground/dgmh-engine/types.ts::Generation, Archive,
DgmhHyperparams, DEFAULT_HYPERPARAMS.

Algorithm reference: ALGORITHM.md §8.2 — paper-exact formulas:
  s_i = 1 / (1 + exp(-lambda * (alpha_i - alpha_mid)))   (sigmoid on performance)
  h_i = 1 / (1 + n_i)                                   (novelty bonus)
  w_i = s_i * h_i                                        (combined weight)
  alpha_mid = mean of top-m agents' performance scores

RNG note (v2 §CONCERN 6): Python random.Random and JS Math.random use different
algorithms; bit-exact parity is impossible. Distributional parity verified by
KS test (D < 0.1) against expected sigmoid×novelty weights over 1000 trials.

Reference: devswha/dgmh playground/dgmh-engine/selectParents.ts
           devswha/dgmh playground/dgmh-engine/types.ts
           hermes-skill-archive-dgmh-plan-v2-addendum.md §CONCERN 6
"""

from __future__ import annotations

import json
import logging
import math
import os
import random as _random_module
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameters — mirrors types.ts::DgmhHyperparams + DEFAULT_HYPERPARAMS
# ---------------------------------------------------------------------------

@dataclass
class DgmhHyperparams:
    """Paper §8.2 hyperparameters.

    Port note: mirrors types.ts lines 56-66 DgmhHyperparams + DEFAULT_HYPERPARAMS.
    lambda=10, top_m=3 are the paper §8.2 defaults.
    """
    lambda_: float = 10.0   # sigmoid sharpness λ; named lambda_ to avoid keyword clash
    top_m: int = 3          # top-m for alpha_mid computation


DEFAULT_HYPERPARAMS = DgmhHyperparams(lambda_=10.0, top_m=3)


# ---------------------------------------------------------------------------
# Generation dataclass — mirrors types.ts::Generation (minimal, selection-only)
# ---------------------------------------------------------------------------

@dataclass
class Generation:
    """Archive entry consumed by select_parents.

    Mirrors types.ts::Generation fields used in §8.2 selection:
    - id: unique identifier (maps to agent.id in TS)
    - score: EvalScore in [0, 1]
    - compiled_children: n_i for novelty bonus h_i = 1/(1+n_i)
    - generation_index: monotonic index t from Algorithm 1 §8.3
    - selectable: True if this entry is eligible for selection (v2 §BLOCKER 1)

    Additional fields carried through but not used in selection math.
    """
    id: str
    score: float
    compiled_children: int = 0
    generation_index: int = 0
    selectable: bool = True
    # Optional extra payload passed through to logs
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core math functions
# Port note: sigmoid_weight ↔ selectParents.ts:22-28
# ---------------------------------------------------------------------------

def sigmoid_weight(alpha_i: float, alpha_mid: float, lambda_: float = DEFAULT_HYPERPARAMS.lambda_) -> float:
    """Sigmoid performance weight.

    Port note: selectParents.ts::sigmoidWeight lines 22-28.
    Formula: s_i = 1 / (1 + exp(-lambda * (alpha_i - alpha_mid)))
    """
    return 1.0 / (1.0 + math.exp(-lambda_ * (alpha_i - alpha_mid)))


# Port note: novelty_weight ↔ selectParents.ts:30-32
def novelty_weight(n_i: int) -> float:
    """Novelty bonus weight.

    Port note: selectParents.ts::noveltyWeight lines 30-32.
    Formula: h_i = 1 / (1 + n_i)
    n_i = compiled_children count (number of accepted children spawned from this entry).
    """
    return 1.0 / (1.0 + n_i)


# Port note: combined_weight ↔ selectParents.ts:34-41
def combined_weight(s_i: float, h_i: float) -> float:
    """Combined selection weight w_i = s_i * h_i.

    Port note: selectParents.ts::combinedWeight lines 34-41.
    Note: TS combinedWeight takes (alphaI, alphaMid, compiledChildren, lambda) and
    calls sigmoid/novelty internally. Python version takes pre-computed s_i and h_i
    for cleaner separation; callers compute them explicitly.
    """
    return s_i * h_i


# ---------------------------------------------------------------------------
# archive_top_midpoint — selectParents.ts:43-52
# ---------------------------------------------------------------------------

def archive_top_midpoint(generations: list[Generation], top_m: int = DEFAULT_HYPERPARAMS.top_m) -> float:
    """Compute alpha_mid = mean score of top-m entries by score.

    Port note: selectParents.ts::archiveTopMidpoint lines 43-52.
    Returns 0.0 for empty list (matches TS: 'if (archive.generations.length === 0) return 0').
    Handles fewer-than-m gracefully: uses all available entries (matches TS slice behavior).
    """
    if not generations:
        return 0.0
    sorted_gens = sorted(generations, key=lambda g: g.score, reverse=True)
    top = sorted_gens[:top_m]  # slice(0, top_m) — graceful if len < top_m
    return sum(g.score for g in top) / len(top)


# ---------------------------------------------------------------------------
# Paths for selection log
# ---------------------------------------------------------------------------

def _dgmh_dir() -> Path:
    """~/.hermes/dgmh/ — DGM-H state home."""
    hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(hermes_home) / "dgmh"


def _select_log_path() -> Path:
    return _dgmh_dir() / "logs" / "select_parents.jsonl"


# ---------------------------------------------------------------------------
# sample_parent — selectParents.ts:54-76
# ---------------------------------------------------------------------------

def sample_parent(
    archive: list[Generation],
    rng: _random_module.Random,
    hyperparams: DgmhHyperparams = DEFAULT_HYPERPARAMS,
) -> Generation:
    """Sample one parent from the archive proportional to combined weight w_i.

    Port note: selectParents.ts::sampleParent lines 54-76.
    RNG injected as random.Random for reproducibility (distributional parity,
    not bit-exact — v2 §CONCERN 6).

    Raises ValueError for empty archive (unlike TS which returns null — Python
    convention is to raise on caller error).

    Degenerate case: if total weight <= 0 (all-zero scores, all compiled_children huge),
    falls back to uniform random selection so it never crashes.
    """
    if not archive:
        raise ValueError("sample_parent: archive is empty")

    selectable = [g for g in archive if g.selectable]
    if not selectable:
        # All entries are non-selectable; fall back to full archive to avoid crash
        selectable = archive

    alpha_mid = archive_top_midpoint(selectable, hyperparams.top_m)

    weights = []
    for g in selectable:
        s_i = sigmoid_weight(g.score, alpha_mid, hyperparams.lambda_)
        h_i = novelty_weight(g.compiled_children)
        w_i = combined_weight(s_i, h_i)
        weights.append(w_i)

    total = sum(weights)

    if total <= 0.0:
        # Degenerate: all weights zero — uniform fallback (matches TS line 69)
        return rng.choice(selectable)

    draw = rng.random() * total
    target = draw
    for i, g in enumerate(selectable):
        target -= weights[i]
        if target <= 0.0:
            return g
    # Floating-point overshoot guard (matches TS lines 74-75)
    return selectable[-1]


# ---------------------------------------------------------------------------
# select_parents — selectParents.ts:78-96
# ---------------------------------------------------------------------------

def select_parents(
    archive: list[Generation],
    count: int,
    rng: _random_module.Random,
    hyperparams: DgmhHyperparams = DEFAULT_HYPERPARAMS,
) -> list[Generation]:
    """Select `count` parents from archive using §8.2 weighted sampling.

    Port note: selectParents.ts::selectParents lines 78-96.
    Samples with replacement (paper does not specify; assumed with-replacement
    since same parent can re-appear — matches TS comment on line 81).

    Raises ValueError for empty archive.

    Writes one structured log entry to ~/.hermes/dgmh/logs/select_parents.jsonl
    per call with: timestamp, candidate_ids, scores, alpha_mid, sigmoid_weights,
    novelty_bonuses, combined_weights, total_weight, rng_draws, selected_ids.
    """
    if not archive:
        raise ValueError("select_parents: archive is empty")

    selectable = [g for g in archive if g.selectable]
    if not selectable:
        selectable = archive

    alpha_mid = archive_top_midpoint(selectable, hyperparams.top_m)

    # Pre-compute weights for all candidates (used in log + sampling)
    candidate_sigmoid: list[float] = []
    candidate_novelty: list[float] = []
    candidate_combined: list[float] = []

    for g in selectable:
        s_i = sigmoid_weight(g.score, alpha_mid, hyperparams.lambda_)
        h_i = novelty_weight(g.compiled_children)
        w_i = combined_weight(s_i, h_i)
        candidate_sigmoid.append(s_i)
        candidate_novelty.append(h_i)
        candidate_combined.append(w_i)

    total_weight = sum(candidate_combined)

    # Sample `count` parents, recording each rng draw
    selected: list[Generation] = []
    rng_draws: list[float] = []

    for _ in range(count):
        if total_weight <= 0.0:
            draw = rng.random()
            rng_draws.append(draw)
            selected.append(rng.choice(selectable))
            continue

        draw = rng.random()
        rng_draws.append(draw)
        target = draw * total_weight
        chosen = selectable[-1]  # default: last (overshoot guard)
        for i, g in enumerate(selectable):
            target -= candidate_combined[i]
            if target <= 0.0:
                chosen = g
                break
        selected.append(chosen)

    # Write structured selection log
    _write_selection_log(
        selectable=selectable,
        alpha_mid=alpha_mid,
        candidate_sigmoid=candidate_sigmoid,
        candidate_novelty=candidate_novelty,
        candidate_combined=candidate_combined,
        total_weight=total_weight,
        rng_draws=rng_draws,
        selected=selected,
    )

    return selected


# ---------------------------------------------------------------------------
# Selection log writer
# ---------------------------------------------------------------------------

def _write_selection_log(
    selectable: list[Generation],
    alpha_mid: float,
    candidate_sigmoid: list[float],
    candidate_novelty: list[float],
    candidate_combined: list[float],
    total_weight: float,
    rng_draws: list[float],
    selected: list[Generation],
) -> None:
    """Append one structured entry to select_parents.jsonl.

    Log schema (per W2 deliverable spec):
      timestamp, candidate_ids, scores, alpha_mid, sigmoid_weights,
      novelty_bonuses, combined_weights, total_weight, rng_draws, selected_ids
    """
    log_entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candidate_ids": [g.id for g in selectable],
        "scores": [g.score for g in selectable],
        "alpha_mid": alpha_mid,
        "sigmoid_weights": candidate_sigmoid,
        "novelty_bonuses": candidate_novelty,
        "combined_weights": candidate_combined,
        "total_weight": total_weight,
        "rng_draws": rng_draws,
        "selected_ids": [g.id for g in selected],
    }

    log_path = _select_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Append (never overwrite) — atomic line append via write mode "a"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("select_parents: could not write selection log: %s", exc)
