"""Gap analyzer for DGM-H GLM Data Collection v1.

Implements plan §4 Step 7 + §3 AC6 + V7.

Given an :class:`ClassifyRunResult` and a read-only view of the patina
workspace (lexicon + ``patterns/ko-*.md``), produce a markdown
``gap_report.md`` that lists every register where one of these holds:

- The Kakao corpus underrepresents that register relative to the operator's
  expected distribution (default: uniform across the seven baseline
  categories). Underrepresentation is anything below
  :data:`DEFAULT_GAP_THRESHOLD_RATIO` of the expected share.
- The patina pack lacks pattern coverage (the ``patterns/ko-*.md`` file
  for that register has fewer than :data:`PATINA_MIN_PATTERNS` blocks).

Each gap row carries a recommended source:

- ``kakao_deeper`` — patina has decent coverage but the corpus does not
  surface enough exemplars. Re-scan more of the kakao corpus.
- ``youtube`` — patina coverage is also thin, so external collection is
  the cheaper path.
- ``both`` — corpus and patina are both severely under, so we want both
  the deeper kakao scan *and* external comments.

The analyzer also surfaces every kept ``_new_*`` proposal from the
classifier as a separate "discovery" gap.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .classifier import ClassifyRunResult, NewProposal
from .taxonomy_loader import BASELINE_KO_7

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# A category is underrepresented when its observed share falls below
# ``expected_share × DEFAULT_GAP_THRESHOLD_RATIO``.
DEFAULT_GAP_THRESHOLD_RATIO = 0.5

# Patina patterns/ko-*.md files with fewer than this many ``### N.`` blocks
# are considered thin and bias the source recommendation toward ``youtube``.
PATINA_MIN_PATTERNS = 3

DEFAULT_PATINA_ROOT = Path("/home/devswha/workspace/patina")
RECOMMENDED_SOURCES = ("kakao_deeper", "youtube", "both")


# Match the canonical pattern-block heading: ``### 19. 챗봇 표현``.
_PATTERN_BLOCK_RE = re.compile(r"^###\s+(\d+)\.\s+", re.MULTILINE)
# Match the ``entries: N`` line in ai-ko.md front-matter.
_LEXICON_ENTRIES_RE = re.compile(r"^entries:\s*(\d+)\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatinaCoverage:
    """Snapshot of the patina pack's current coverage for ko-*."""

    pattern_counts: dict[str, int] = field(default_factory=dict)
    lexicon_entries: int = 0

    def patterns_for(self, category: str) -> int:
        return self.pattern_counts.get(category, 0)


@dataclass(frozen=True)
class GapRow:
    """One identified gap in the corpus + patina coverage."""

    category: str
    kakao_count: int
    kakao_coverage_pct: float
    expected_share_pct: float
    patina_patterns: int
    recommended_source: str
    proposed_taxonomy_hook: str
    rationale: str


@dataclass(frozen=True)
class GapAnalysis:
    """Aggregate result of a gap-analysis run."""

    gaps: list[GapRow]
    discovery_proposals: list[NewProposal]
    patina_coverage: PatinaCoverage
    n_messages: int


# ---------------------------------------------------------------------------
# Patina reader (read-only)
# ---------------------------------------------------------------------------


def read_patina_coverage(patina_root: Path = DEFAULT_PATINA_ROOT) -> PatinaCoverage:
    """Read the patina workspace and return per-category pattern counts.

    Missing files are recorded as zero — never raise, because the operator
    may have a partial patina checkout and we still want a usable report.
    """

    patterns_dir = patina_root / "patterns"
    pattern_counts: dict[str, int] = {}
    for name in BASELINE_KO_7:
        path = patterns_dir / f"ko-{name}.md"
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.debug("gap_analyzer: %s unreadable (%s); counting zero", path, exc)
            pattern_counts[name] = 0
            continue
        pattern_counts[name] = len(_PATTERN_BLOCK_RE.findall(text))

    lexicon_path = patina_root / "lexicon" / "ai-ko.md"
    lexicon_entries = 0
    try:
        text = lexicon_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.debug("gap_analyzer: %s unreadable (%s); entries=0", lexicon_path, exc)
    else:
        match = _LEXICON_ENTRIES_RE.search(text)
        if match:
            lexicon_entries = int(match.group(1))

    return PatinaCoverage(pattern_counts=pattern_counts, lexicon_entries=lexicon_entries)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _per_category_counts(result: ClassifyRunResult) -> dict[str, int]:
    """Sum classifier hits per category, deduplicated per message."""

    counts: dict[str, int] = {}
    for classification in result.classifications:
        seen: set[str] = set()
        for cat in classification.categories:
            if cat.name in seen:
                continue
            seen.add(cat.name)
            counts[cat.name] = counts.get(cat.name, 0) + 1
    return counts


def _recommend_source(
    kakao_pct: float, expected_pct: float, patina_patterns: int
) -> str:
    """Pick a recommendation from :data:`RECOMMENDED_SOURCES`.

    Heuristic:

    - Both severely under (kakao < 0.25 × expected AND patina thin) → ``both``.
    - Patina thin but kakao plausible → ``youtube`` (external fills patina).
    - Patina solid but kakao thin → ``kakao_deeper`` (look harder in corpus).
    """

    kakao_severe = expected_pct > 0 and kakao_pct < 0.25 * expected_pct
    patina_thin = patina_patterns < PATINA_MIN_PATTERNS
    if kakao_severe and patina_thin:
        return "both"
    if patina_thin:
        return "youtube"
    return "kakao_deeper"


def _expected_shares(
    taxonomy: Sequence[str],
    expected_distribution: Optional[dict[str, float]],
) -> dict[str, float]:
    """Return expected share (percentage of corpus) per baseline category.

    If the operator passes ``expected_distribution`` it is honoured verbatim
    (after re-normalising to 100% if needed). Otherwise we spread the
    expectation uniformly across the seven baseline categories.
    """

    if expected_distribution:
        # Re-normalise so operator values sum to 100% — they may pass
        # weights rather than percentages.
        total = sum(max(0.0, v) for v in expected_distribution.values())
        if total <= 0:
            raise ValueError("expected_distribution must contain at least one positive weight")
        return {
            name: max(0.0, expected_distribution.get(name, 0.0)) / total * 100.0
            for name in BASELINE_KO_7
        }
    uniform = 100.0 / len(BASELINE_KO_7)
    return {name: uniform for name in BASELINE_KO_7}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_gaps(
    result: ClassifyRunResult,
    *,
    patina_coverage: Optional[PatinaCoverage] = None,
    patina_root: Path = DEFAULT_PATINA_ROOT,
    expected_distribution: Optional[dict[str, float]] = None,
    threshold_ratio: float = DEFAULT_GAP_THRESHOLD_RATIO,
) -> GapAnalysis:
    """Identify under-represented registers in the corpus."""

    if patina_coverage is None:
        patina_coverage = read_patina_coverage(patina_root)

    counts = _per_category_counts(result)
    expected = _expected_shares(result.taxonomy, expected_distribution)
    n = max(1, result.n_messages)

    gaps: list[GapRow] = []
    for name in BASELINE_KO_7:
        observed_count = counts.get(name, 0)
        observed_pct = observed_count / n * 100
        expected_pct = expected[name]
        patina_patterns = patina_coverage.patterns_for(name)

        underrep = observed_pct < expected_pct * threshold_ratio
        patina_thin = patina_patterns < PATINA_MIN_PATTERNS
        if not (underrep or patina_thin):
            continue

        rationale_bits: list[str] = []
        if underrep:
            rationale_bits.append(
                f"corpus share {observed_pct:.2f}% < threshold "
                f"{expected_pct * threshold_ratio:.2f}% "
                f"(expected {expected_pct:.2f}%)"
            )
        if patina_thin:
            rationale_bits.append(
                f"patina patterns/ko-{name}.md has only {patina_patterns} blocks "
                f"(< {PATINA_MIN_PATTERNS})"
            )

        gaps.append(
            GapRow(
                category=name,
                kakao_count=observed_count,
                kakao_coverage_pct=round(observed_pct, 2),
                expected_share_pct=round(expected_pct, 2),
                patina_patterns=patina_patterns,
                recommended_source=_recommend_source(observed_pct, expected_pct, patina_patterns),
                proposed_taxonomy_hook=f"ko-{name}",
                rationale="; ".join(rationale_bits),
            )
        )

    return GapAnalysis(
        gaps=gaps,
        discovery_proposals=list(result.new_proposals),
        patina_coverage=patina_coverage,
        n_messages=result.n_messages,
    )


def render_gap_report(
    analysis: GapAnalysis,
    *,
    title: str = "DGM-H GLM data v1 — gap report",
) -> str:
    """Render :class:`GapAnalysis` to plan §3 AC6 markdown."""

    lines: list[str] = [
        f"# {title}",
        "",
        f"- Messages analyzed: **{analysis.n_messages}**",
        f"- Patina lexicon entries: **{analysis.patina_coverage.lexicon_entries}**",
        f"- Gaps identified: **{len(analysis.gaps)}**",
        f"- New-category proposals carried over: **{len(analysis.discovery_proposals)}**",
        "",
        "## Gaps",
        "",
    ]

    if not analysis.gaps:
        lines.append("_No gaps identified for the baseline taxonomy._")
        lines.append("")
    else:
        lines.append(
            "| Category | Kakao count | Kakao % | Expected % | Patina patterns | Recommended source | Hook |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | --- | --- |"
        )
        for gap in analysis.gaps:
            lines.append(
                f"| `{gap.category}` | {gap.kakao_count} | "
                f"{gap.kakao_coverage_pct:.2f}% | {gap.expected_share_pct:.2f}% | "
                f"{gap.patina_patterns} | `{gap.recommended_source}` | "
                f"`{gap.proposed_taxonomy_hook}` |"
            )
        lines.append("")

        lines.append("### Rationale")
        lines.append("")
        for gap in analysis.gaps:
            lines.append(f"- `{gap.category}`: {gap.rationale}")
        lines.append("")

    lines.append("## Discovery proposals (kept after R5 cap)")
    lines.append("")
    if not analysis.discovery_proposals:
        lines.append("_No `_new_*` proposals this run._")
        lines.append("")
    else:
        for proposal in analysis.discovery_proposals:
            lines.append(f"### `{proposal.name}`")
            lines.append("")
            lines.append(f"- **Candidate target file**: `{proposal.candidate_target_file}`")
            lines.append(f"- **Exemplar count**: {len(proposal.exemplar_msg_ids)}")
            if proposal.rationale:
                lines.append(f"- **Rationale**: {proposal.rationale}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "DEFAULT_GAP_THRESHOLD_RATIO",
    "DEFAULT_PATINA_ROOT",
    "GapAnalysis",
    "GapRow",
    "PATINA_MIN_PATTERNS",
    "PatinaCoverage",
    "RECOMMENDED_SOURCES",
    "analyze_gaps",
    "read_patina_coverage",
    "render_gap_report",
]
