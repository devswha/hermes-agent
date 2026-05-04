"""
dgmh/bias_detector.py — Heuristic LLM-free bias detector (W5 deliverable).

Python port of playground/sandbox/biasDetector.ts with skill-domain
recalibration per plan v2 §CONCERN 5.

Recalibration rationale (plan v2 §CONCERN 5):
  "failure threshold needs recalibration for skill scores (Hermes skill
  scores likely cluster differently from synthetic task scores). New
  skill-domain signal: prompt-bloat."

  Sandbox defaults -> Skill-domain defaults:
    failure_run_threshold   0.1  -> 0.3   (skill scores cluster higher; 0.1
                                           would rarely fire on real skills)
    failure_run_window      3    -> 3     (unchanged; small window stays
                                           sensitive to short failure bursts)
    plateau_window          5    -> 5     (unchanged)
    plateau_epsilon         0.02 -> 0.05  (looser; skill score variance is
                                           naturally smaller on well-tuned
                                           evaluation suites)
    lineage_convergence_threshold 0.7 -> 0.7  (unchanged; convergence
                                               threshold is domain-agnostic)
    lineage_window          5    -> 5     (unchanged)
    prompt_bloat_window     —    -> 5     (new signal)
    prompt_bloat_growth_rate —   -> 0.5   (50% char-count growth across
                                           window without score gain)

Output envelope matches BiasReport shape from dgmh-engine/types.ts:
  { "signals": [{id, description, severity, evidenceGenerations}] }

Usage:
    from dgmh.bias_detector import make_heuristic_skill_bias_detector, BiasReport
    detector = make_heuristic_skill_bias_detector()
    report = detector.detect(archive, context_note="iteration-42")
    # report.signals is a list of BiasSignal dicts
    # str(report) is the Codex-envelope JSON

Reference:
  playground/sandbox/biasDetector.ts   (original 3-heuristic implementation)
  playground/dgmh-engine/biasDetection.ts (envelope + parse logic)
  playground/dgmh-engine/types.ts       (BiasSignal / BiasReport types)
  hermes-skill-archive-dgmh-plan-v2-addendum.md §CONCERN 5
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, NamedTuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types — mirror dgmh-engine/types.ts BiasSignal / BiasReport
# ---------------------------------------------------------------------------

BiasSeverity = Literal["info", "concern", "block"]


@dataclass
class BiasSignal:
    """One detected bias pattern.

    Mirrors types.ts::BiasSignal.
    """
    id: str
    description: str
    severity: BiasSeverity
    evidence_generations: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "severity": self.severity,
            "evidenceGenerations": self.evidence_generations,
        }


@dataclass
class BiasReport:
    """Structured bias detection result.

    Mirrors types.ts::BiasReport.
    signals: list of BiasSignal detected in the archive slice.
    envelope_json: Codex-shape JSON string ({ "signals": [...] }).
    inspected_generations: count of generations examined.
    produced_at: ISO 8601 timestamp.
    """
    signals: list[BiasSignal]
    inspected_generations: int
    produced_at: str
    envelope_json: str

    def __str__(self) -> str:
        return self.envelope_json

    def severity_counts(self) -> dict[str, int]:
        """Return {info, concern, block} counts."""
        counts: dict[str, int] = {"info": 0, "concern": 0, "block": 0}
        for s in self.signals:
            counts[s.severity] = counts.get(s.severity, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Generation shape expected by the detector
# ---------------------------------------------------------------------------

class _Generation(NamedTuple):
    """Minimal generation shape consumed by heuristics.

    Callers may pass objects with attribute access OR dicts — see
    _coerce_generation() below.
    """
    generation_index: int
    score: float
    lineage: list[str]   # parent ids in archive order
    skill_md: str        # SKILL.md text (for prompt-bloat signal)


def _coerce_generation(g: Any) -> _Generation:
    """Coerce a Generation-like object to _Generation.

    Accepts:
    - dgmh.select_parents.Generation (has .generation_index, .score, .extra)
    - plain dict with same keys
    - _Generation (pass-through)
    """
    if isinstance(g, _Generation):
        return g
    if isinstance(g, dict):
        lineage = g.get("lineage", g.get("extra", {}).get("lineage", []))
        return _Generation(
            generation_index=int(g.get("generation_index", 0)),
            score=float(g.get("score", 0.0)),
            lineage=list(lineage) if lineage else [],
            skill_md=str(g.get("skill_md", g.get("extra", {}).get("skill_md", ""))),
        )
    # attribute-access object (select_parents.Generation)
    lineage = []
    if hasattr(g, "extra") and isinstance(g.extra, dict):
        lineage = list(g.extra.get("lineage", []))
    elif hasattr(g, "lineage"):
        lineage = list(g.lineage)

    skill_md = ""
    if hasattr(g, "extra") and isinstance(g.extra, dict):
        skill_md = str(g.extra.get("skill_md", ""))
    elif hasattr(g, "skill_md"):
        skill_md = str(g.skill_md)

    return _Generation(
        generation_index=int(getattr(g, "generation_index", 0)),
        score=float(getattr(g, "score", 0.0)),
        lineage=lineage,
        skill_md=skill_md,
    )


# ---------------------------------------------------------------------------
# Detector options
# ---------------------------------------------------------------------------

@dataclass
class HeuristicBiasDetectorOpts:
    """Configurable thresholds for the skill-domain bias detector.

    Defaults are skill-domain recalibrations per plan v2 §CONCERN 5.
    """
    # failure-run signal
    failure_run_threshold: float = 0.3
    failure_run_window: int = 3

    # score-plateau signal
    plateau_window: int = 5
    plateau_epsilon: float = 0.05

    # lineage-convergence signal
    lineage_convergence_threshold: float = 0.7
    lineage_window: int = 5

    # prompt-bloat signal (new per plan v2 §CONCERN 5)
    prompt_bloat_window: int = 5
    prompt_bloat_growth_rate: float = 0.5  # 50% char growth without score gain


# ---------------------------------------------------------------------------
# Heuristic implementations
# ---------------------------------------------------------------------------

def _slice_recent(gens: list[_Generation], n: int) -> list[_Generation]:
    """Return the last n generations (or all if len < n)."""
    if n <= 0:
        return []
    return gens[-n:] if len(gens) >= n else list(gens)


def _detect_failure_run(
    gens: list[_Generation],
    threshold: float,
    min_window: int,
) -> BiasSignal | None:
    """Detect N consecutive low-score generations.

    Port of biasDetector.ts::detectFailureRun.
    Severity: concern if run_len in [min_window, 2*min_window),
              block if run_len >= 2*min_window.
    """
    best_start = -1
    best_len = 0
    cur_start = -1
    cur_len = 0

    for i, g in enumerate(gens):
        if g.score < threshold:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0
            cur_start = -1

    if best_len < min_window:
        return None

    evidence = [gens[i].generation_index for i in range(best_start, best_start + best_len)]
    severity: BiasSeverity = "block" if best_len >= 2 * min_window else "concern"
    return BiasSignal(
        id="failure-run",
        description=f"{best_len} consecutive generations with score < {threshold}",
        severity=severity,
        evidence_generations=evidence,
    )


def _detect_plateau(
    gens: list[_Generation],
    window: int,
    epsilon: float,
) -> BiasSignal | None:
    """Detect stagnant scores: max−min < epsilon over rolling window.

    Port of biasDetector.ts::detectPlateau.
    Severity: concern (score stuck is a concern, not an immediate block).
    """
    if len(gens) < window:
        return None
    recent = gens[-window:]
    scores = [g.score for g in recent]
    score_max = max(scores)
    score_min = min(scores)
    if score_max - score_min >= epsilon:
        return None
    return BiasSignal(
        id="score-plateau",
        description=(
            f"score range {score_max - score_min:.4f} < epsilon {epsilon} "
            f"over {window} generations"
        ),
        severity="concern",
        evidence_generations=[g.generation_index for g in recent],
    )


def _detect_lineage_convergence(
    gens: list[_Generation],
    threshold: float,
    window: int,
) -> BiasSignal | None:
    """Detect ancestor over-representation in recent lineages.

    Port of biasDetector.ts::detectLineageConvergence.
    Severity: info (convergence is observable but not necessarily harmful).
    """
    if len(gens) < window:
        return None
    recent = gens[-window:]

    tally: dict[str, int] = {}
    for g in recent:
        for anc in g.lineage:
            tally[anc] = tally.get(anc, 0) + 1

    if not tally:
        return None

    top_anc = max(tally, key=lambda k: tally[k])
    top_count = tally[top_anc]
    fraction = top_count / len(recent)

    if fraction < threshold:
        return None

    evidence = [
        g.generation_index for g in recent if top_anc in g.lineage
    ]
    return BiasSignal(
        id="lineage-convergence",
        description=(
            f"{fraction * 100:.0f}% of last {window} generations "
            f"descend from ancestor '{top_anc}'"
        ),
        severity="info",
        evidence_generations=evidence,
    )


def _detect_prompt_bloat(
    gens: list[_Generation],
    window: int,
    growth_rate: float,
) -> BiasSignal | None:
    """Detect monotonic skill_md growth without score gain (new signal).

    Plan v2 §CONCERN 5: "skill text grows monotonically across generations
    without score gain."

    Algorithm:
    1. Take the last `window` generations.
    2. Compute char counts of skill_md for each.
    3. Check if all consecutive pairs show non-decreasing char count
       (monotonic growth).
    4. Compute total growth rate: (last_chars - first_chars) / first_chars.
    5. Check if score did NOT improve from first to last generation.
    6. Fire if growth_rate >= threshold AND no score improvement.

    Severity: concern (bloat is a concern; may indicate cargo-cult copying).
    """
    if len(gens) < window:
        return None
    recent = gens[-window:]

    char_counts = [len(g.skill_md) for g in recent]

    # Check monotonic growth (each step non-decreasing)
    is_monotonic = all(
        char_counts[i] <= char_counts[i + 1]
        for i in range(len(char_counts) - 1)
    )
    if not is_monotonic:
        return None

    first_chars = char_counts[0]
    last_chars = char_counts[-1]
    if first_chars == 0:
        return None

    actual_growth = (last_chars - first_chars) / first_chars
    if actual_growth < growth_rate:
        return None

    # Check for no score gain (last score <= first score)
    first_score = recent[0].score
    last_score = recent[-1].score
    if last_score > first_score:
        # Score improved — bloat may be justified
        return None

    return BiasSignal(
        id="prompt-bloat",
        description=(
            f"skill_md grew {actual_growth * 100:.0f}% "
            f"({first_chars} -> {last_chars} chars) over {window} generations "
            f"with no score gain ({first_score:.3f} -> {last_score:.3f})"
        ),
        severity="concern",
        evidence_generations=[g.generation_index for g in recent],
    )


# ---------------------------------------------------------------------------
# Detector factory
# ---------------------------------------------------------------------------

class _HeuristicSkillBiasDetector:
    """Concrete heuristic bias detector for the skill domain.

    Instantiate via make_heuristic_skill_bias_detector().
    """

    def __init__(self, opts: HeuristicBiasDetectorOpts) -> None:
        self._opts = opts

    def detect(
        self,
        archive: list[Any],
        context_note: str | None = None,
        recent_n: int = 20,
    ) -> BiasReport:
        """Run all 4 heuristics on the recent slice of archive.

        Args:
            archive: list of Generation-like objects (select_parents.Generation,
                     dict, or _Generation).
            context_note: optional operator context injected into description
                          (logged at DEBUG level; not included in signal text).
            recent_n: how many most-recent generations to inspect (default 20).

        Returns:
            BiasReport with envelope_json = Codex-shape JSON string.
        """
        if context_note:
            logger.debug("bias_detector: detect called context=%r", context_note)

        coerced = [_coerce_generation(g) for g in archive]
        recent = _slice_recent(coerced, recent_n)
        signals: list[BiasSignal] = []

        opts = self._opts

        sig = _detect_failure_run(recent, opts.failure_run_threshold, opts.failure_run_window)
        if sig:
            signals.append(sig)

        sig = _detect_plateau(recent, opts.plateau_window, opts.plateau_epsilon)
        if sig:
            signals.append(sig)

        sig = _detect_lineage_convergence(
            recent, opts.lineage_convergence_threshold, opts.lineage_window
        )
        if sig:
            signals.append(sig)

        sig = _detect_prompt_bloat(recent, opts.prompt_bloat_window, opts.prompt_bloat_growth_rate)
        if sig:
            signals.append(sig)

        produced_at = datetime.now(timezone.utc).isoformat()
        envelope = {"signals": [s.to_dict() for s in signals]}
        envelope_json = json.dumps(envelope, ensure_ascii=False)

        report = BiasReport(
            signals=signals,
            inspected_generations=len(recent),
            produced_at=produced_at,
            envelope_json=envelope_json,
        )

        # Log concern/block signals to stderr so operators see them
        for s in signals:
            if s.severity in ("concern", "block"):
                logger.warning(
                    "bias_detector: %s signal=%r: %s (evidence=%s)",
                    s.severity.upper(),
                    s.id,
                    s.description,
                    s.evidence_generations,
                )

        return report


def make_heuristic_skill_bias_detector(
    opts: HeuristicBiasDetectorOpts | None = None,
) -> _HeuristicSkillBiasDetector:
    """Factory for the skill-domain heuristic bias detector.

    Returns a detector with .detect(archive, context_note=None, recent_n=20).

    Args:
        opts: HeuristicBiasDetectorOpts override. Defaults use skill-domain
              recalibrated thresholds per plan v2 §CONCERN 5.
    """
    return _HeuristicSkillBiasDetector(opts or HeuristicBiasDetectorOpts())
