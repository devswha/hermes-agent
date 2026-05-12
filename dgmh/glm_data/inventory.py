"""Inventory report generator for DGM-H GLM Data Collection v1.

Implements plan §4 Step 6 + §3 AC3 + V7. Given a :class:`ClassifyRunResult`
and the source :class:`KakaoMsg` list that produced it, render a deterministic
markdown report covering:

- one table row per baseline + promoted ``_new_*`` category with
  ``count`` and ``percentage`` of the corpus,
- a ``### Examples`` block listing up to three **redacted** exemplars
  per category (we only emit ``KakaoMsg.redacted_text``, never
  ``raw_text``), and
- a ``## New category proposals`` section with rationale + target file
  for every ``_new_*`` proposal the classifier kept after R5 capping.

The module produces plain markdown; writing it to staging is the caller's
responsibility (e.g. via :class:`StagingWriter`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from .classifier import ClassifyRunResult, NewProposal
from .schema import Classification, KakaoMsg
from .taxonomy_loader import BASELINE_KO_7

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EXAMPLES_PER_CATEGORY = 3
NO_COVERAGE_MARKER = "_(no coverage)_"


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CategoryRow:
    name: str
    count: int
    percentage: float
    exemplar_msg_ids: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _index_messages(messages: Sequence[KakaoMsg]) -> dict[str, KakaoMsg]:
    """Build a deterministic msg_id → KakaoMsg lookup.

    Later occurrences of the same msg_id win — but in practice the parser
    emits unique ids, so the choice is academic. We keep the dict insertion
    order so callers iterating it observe original corpus order.
    """

    return {m.msg_id: m for m in messages}


def _collect_category_rows(
    result: ClassifyRunResult,
    *,
    examples_per_category: int,
) -> list[_CategoryRow]:
    """Aggregate per-category counts + first-seen exemplar ids."""

    # Map category name → ordered list of msg_ids that hit that category.
    hits: dict[str, list[str]] = {}
    for classification in result.classifications:
        seen_in_msg: set[str] = set()
        for cat in classification.categories:
            if cat.name in seen_in_msg:
                continue
            seen_in_msg.add(cat.name)
            hits.setdefault(cat.name, []).append(classification.msg_id)

    n = max(1, result.n_messages)  # avoid div-by-zero in synthetic edge cases

    rows: list[_CategoryRow] = []
    # Render every category in the active taxonomy — baseline first, then
    # promoted — even when coverage is zero (plan §4 Step 6 acceptance:
    # "renders every ko-* 7 baseline + every currently-promoted category
    # with count + 3 redacted exemplars (or 'no coverage' marker)").
    for name in result.taxonomy:
        msg_ids = hits.get(name, [])
        rows.append(
            _CategoryRow(
                name=name,
                count=len(msg_ids),
                percentage=round(len(msg_ids) / n * 100, 2),
                exemplar_msg_ids=msg_ids[:examples_per_category],
            )
        )
    return rows


def _render_redacted(text: str) -> str:
    """Strip line breaks so a single exemplar fits in a markdown bullet."""

    return text.replace("\n", " ").strip()


def _render_examples_section(
    rows: Sequence[_CategoryRow],
    by_id: dict[str, KakaoMsg],
) -> str:
    lines: list[str] = ["## Examples per category", ""]
    for row in rows:
        lines.append(f"### `{row.name}`")
        if not row.exemplar_msg_ids:
            lines.append(NO_COVERAGE_MARKER)
            lines.append("")
            continue
        for msg_id in row.exemplar_msg_ids:
            msg = by_id.get(msg_id)
            redacted = _render_redacted(msg.redacted_text) if msg is not None else ""
            lines.append(f"- `{msg_id}`: {redacted}")
        lines.append("")
    return "\n".join(lines)


def _render_summary_table(
    rows: Sequence[_CategoryRow],
    *,
    new_proposal_count: int,
) -> str:
    lines: list[str] = [
        "## Summary",
        "",
        "| Category | Count | % of corpus |",
        "| --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| `{row.name}` | {row.count} | {row.percentage:.2f}% |")
    lines.append("")
    lines.append(f"_New category proposals (after R5 cap): **{new_proposal_count}**_")
    lines.append("")
    return "\n".join(lines)


def _render_new_proposals_section(
    proposals: Sequence[NewProposal],
    by_id: dict[str, KakaoMsg],
    *,
    examples_per_proposal: int,
) -> str:
    lines: list[str] = ["## New category proposals", ""]
    if not proposals:
        lines.append("_No new category proposals this run._")
        lines.append("")
        return "\n".join(lines)
    for proposal in proposals:
        lines.append(f"### `{proposal.name}`")
        lines.append("")
        lines.append(f"- **Candidate target file**: `{proposal.candidate_target_file}`")
        lines.append(f"- **Exemplar count**: {len(proposal.exemplar_msg_ids)}")
        if proposal.rationale:
            lines.append(f"- **Rationale**: {proposal.rationale}")
        lines.append("")
        lines.append("**Exemplars:**")
        lines.append("")
        shown = 0
        for msg_id in proposal.exemplar_msg_ids:
            if shown >= examples_per_proposal:
                break
            msg = by_id.get(msg_id)
            if msg is None:
                continue
            redacted = _render_redacted(msg.redacted_text)
            lines.append(f"- `{msg_id}`: {redacted}")
            shown += 1
        if shown == 0:
            lines.append(NO_COVERAGE_MARKER)
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_inventory(
    result: ClassifyRunResult,
    messages: Sequence[KakaoMsg],
    *,
    examples_per_category: int = DEFAULT_EXAMPLES_PER_CATEGORY,
    examples_per_proposal: int = DEFAULT_EXAMPLES_PER_CATEGORY,
    title: str = "DGM-H GLM data v1 — inventory report",
) -> str:
    """Render the inventory markdown for a classifier run.

    Determinism rules (verified by tests):

    - Categories appear in :attr:`ClassifyRunResult.taxonomy` order (baseline
      seven first, then promoted ``_new_*`` in ledger order).
    - Within a category, exemplars are listed in first-seen corpus order.
    - Proposals appear in :attr:`ClassifyRunResult.new_proposals` order
      (already deterministically sorted by exemplar count by the classifier).
    """

    if examples_per_category <= 0:
        raise ValueError("examples_per_category must be > 0")
    if examples_per_proposal <= 0:
        raise ValueError("examples_per_proposal must be > 0")

    by_id = _index_messages(messages)
    rows = _collect_category_rows(result, examples_per_category=examples_per_category)

    header = [
        f"# {title}",
        "",
        f"- Messages classified: **{result.n_messages}**",
        f"- GLM calls: **{result.n_calls}**",
        f"- Baseline categories: **{len([c for c in result.taxonomy if c in BASELINE_KO_7])}**",
        f"- Promoted `_new_*` categories: **{len([c for c in result.taxonomy if c.startswith('_new_')])}**",
        "",
    ]

    sections = [
        "\n".join(header),
        _render_summary_table(rows, new_proposal_count=len(result.new_proposals)),
        _render_examples_section(rows, by_id),
        _render_new_proposals_section(
            result.new_proposals, by_id, examples_per_proposal=examples_per_proposal
        ),
    ]
    return "\n".join(sections).rstrip() + "\n"


__all__ = [
    "DEFAULT_EXAMPLES_PER_CATEGORY",
    "NO_COVERAGE_MARKER",
    "generate_inventory",
]
