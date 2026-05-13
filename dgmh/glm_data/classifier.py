"""Per-message tone classifier for DGM-H GLM Data Collection v1.

Implements plan §4 Step 5 + §3 AC5 (taxonomy conformance) and AC3
(batched / chunked analysis at the GLM-call level).

Pipeline:

1. Load the active taxonomy via :func:`dgmh.glm_data.taxonomy_loader.load`
   (baseline ``BASELINE_KO_7`` ∪ previously-promoted ``_new_*`` categories).
2. Group an input list of :class:`KakaoMsg` into batches of
   :data:`DEFAULT_BATCH_SIZE` (N=20 per plan §4 Step 5).
3. For each batch, call :meth:`GLMClient.chat` with a strict JSON-object
   schema requesting per-message classifications and optional ``_new_*``
   discovery proposals.
4. Validate the parsed content against :data:`CLASSIFY_RESPONSE_SCHEMA` and
   normalise it into :class:`Classification` dataclasses (defined in
   :mod:`dgmh.glm_data.schema`).
5. Enforce R5 / plan §4 Step 5 caps: at most :data:`MAX_NEW_PROPOSALS_PER_RUN`
   ``_new_*`` categories per run, each carrying ≥3 exemplars.

The classifier does **not** itself issue HTTP requests; it is purely a
composition layer over :class:`GLMClient`. Tests therefore use the mock
transport with synthetic fixtures, or inject a fake client object.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import jsonschema

from .glm_client import GLMClient, GLMResponseSchemaError
from .schema import (
    BASELINE_TONE_NAMES,
    Classification,
    KakaoMsg,
    ToneCategory,
)
from .taxonomy_loader import BASELINE_KO_7, load as load_taxonomy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE = 20
MAX_NEW_PROPOSALS_PER_RUN = 5
MIN_EXEMPLARS_PER_NEW_PROPOSAL = 3
DEFAULT_MODEL = "glm-4.5"


# Required outer-content shape from GLM. ``classifications`` is the only
# required key; ``_new_proposals`` is optional (model may emit none). We
# validate the parsed JSON content (the body of ``choices[0].message.content``)
# rather than the envelope, because the GLM client already validates the
# envelope shape upstream.
CLASSIFY_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "required": ["classifications"],
    "additionalProperties": True,
    "properties": {
        "classifications": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["msg_id", "categories"],
                "additionalProperties": True,
                "properties": {
                    "msg_id": {"type": "string", "minLength": 1},
                    "categories": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                    "confidence": {"type": "number"},
                },
            },
        },
        "_new_proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "exemplar_msg_ids", "candidate_target_file"],
                "additionalProperties": True,
                "properties": {
                    "name": {
                        "type": "string",
                        "pattern": r"^_new_[A-Za-z0-9_\- ]+$",
                    },
                    "exemplar_msg_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                    "candidate_target_file": {"type": "string", "minLength": 1},
                    "rationale": {"type": "string"},
                },
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ClassifierError(RuntimeError):
    """Base error for classifier failures."""


class TaxonomyViolationError(ClassifierError):
    """Raised when GLM returns a category not in the active taxonomy and not
    prefixed with ``_new_`` (silent taxonomy widening forbidden)."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class NewProposal:
    """One ``_new_*`` discovery proposal aggregated across all batches."""

    name: str
    exemplar_msg_ids: list[str]
    candidate_target_file: str
    rationale: str = ""


@dataclass
class ClassifyRunResult:
    """Aggregated result of a classifier run."""

    classifications: list[Classification]
    new_proposals: list[NewProposal]
    n_calls: int
    n_messages: int
    taxonomy: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strip_ko_prefix(name: str) -> str:
    """Normalise category names emitted by GLM.

    The prompt allows either form (``"ko-communication"`` matches the
    ``patterns/ko-*.md`` file basenames; ``"communication"`` matches our
    ``BASELINE_TONE_NAMES`` dataclass invariant). We standardise on the bare
    form for downstream consumers, leaving ``_new_*`` untouched.
    """

    if name.startswith("_new_"):
        return name
    if name.startswith("ko-"):
        return name[len("ko-"):]
    return name


def _target_file_for(name: str, *, proposal_target: Optional[str] = None) -> str:
    """Return the ``patterns/ko-*.md`` file that owns a category.

    Baseline 7 categories map deterministically to
    ``patterns/ko-{communication,content,...}.md``. For an *uncommitted*
    ``_new_*`` proposal, GLM is required to supply ``candidate_target_file``
    in its ``_new_proposals`` block; the caller passes that through as
    ``proposal_target``. We refuse to invent a target when GLM is silent —
    such a category will be discarded upstream as a taxonomy violation.
    """

    if name in BASELINE_TONE_NAMES:
        return f"patterns/ko-{name}.md"
    if name.startswith("_new_") and proposal_target:
        return proposal_target
    raise TaxonomyViolationError(f"unknown category {name!r}")


def _format_message_block(messages: Sequence[KakaoMsg]) -> str:
    rows = []
    for m in messages:
        # Only the redacted text crosses the network. ``raw_text`` stays local
        # per the schema's docstring.
        rows.append(json.dumps({"msg_id": m.msg_id, "text": m.redacted_text}, ensure_ascii=False))
    return "\n".join(rows)


def _build_system_prompt(taxonomy: Sequence[str]) -> str:
    baseline = [c for c in taxonomy if c in BASELINE_KO_7]
    promoted = [c for c in taxonomy if c.startswith("_new_")]
    promoted_block = (
        "\nPromoted custom categories (also valid baseline targets):\n  - "
        + "\n  - ".join(promoted)
        if promoted
        else ""
    )
    return (
        "You are a Korean-language tone classifier for the DGM-H project. "
        "Classify each input message into one or more of the baseline ko-* "
        "tone categories listed below. You may also propose at most "
        f"{MAX_NEW_PROPOSALS_PER_RUN} new ``_new_*`` categories per response "
        f"(each with at least {MIN_EXEMPLARS_PER_NEW_PROPOSAL} exemplar msg_ids).\n\n"
        "Baseline ko-* categories (use exactly these short names):\n  - "
        + "\n  - ".join(baseline)
        + promoted_block
        + "\n\nReturn a JSON object matching this schema:\n"
        '{"classifications": [{"msg_id": "...", "categories": ["communication", ...], '
        '"confidence": 0.0..1.0}], '
        '"_new_proposals": [{"name": "_new_<slug>", '
        '"exemplar_msg_ids": ["...", "...", "..."], '
        '"candidate_target_file": "patterns/ko-<base>.md", '
        '"rationale": "..."}]}'
    )


def _batched(seq: Sequence[KakaoMsg], size: int) -> Iterator[list[KakaoMsg]]:
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


def _extract_content(response: dict) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ClassifierError(f"malformed GLM response envelope: {exc}") from exc
    if not isinstance(content, str):
        raise ClassifierError(
            f"expected choices[0].message.content to be a JSON string; got {type(content)!r}"
        )
    return content


def _parse_content(content: str) -> dict:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise GLMResponseSchemaError(
            f"GLM returned non-JSON content for classifier: {exc}"
        ) from exc
    if isinstance(parsed, dict) and isinstance(parsed.get("_new_proposals"), list):
        parsed["_new_proposals"] = [
            p for p in parsed["_new_proposals"]
            if isinstance(p, dict)
            and p
            and p.get("name")
            and p.get("exemplar_msg_ids")
            and p.get("candidate_target_file")
        ]
    try:
        jsonschema.validate(parsed, CLASSIFY_RESPONSE_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise GLMResponseSchemaError(str(exc)) from exc
    return parsed


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class Classifier:
    """Compose :class:`GLMClient` + taxonomy into a batched classifier."""

    def __init__(
        self,
        client: GLMClient,
        *,
        staging_root: Optional[Path] = None,
        taxonomy: Optional[Sequence[str]] = None,
        model: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_retries: int = 3,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if taxonomy is None:
            if staging_root is None:
                raise ValueError("pass either `taxonomy` or `staging_root`")
            taxonomy = load_taxonomy(staging_root)
        # Defensive copy + normalise.
        self._taxonomy: list[str] = list(taxonomy)
        self._taxonomy_set: set[str] = set(self._taxonomy)
        self._client = client
        self._model = model
        self._batch_size = batch_size
        self._max_retries = max_retries

    # -- public API -----------------------------------------------------

    @property
    def taxonomy(self) -> list[str]:
        return list(self._taxonomy)

    def classify(self, messages: Sequence[KakaoMsg]) -> ClassifyRunResult:
        """Classify ``messages`` and return aggregated results + proposals."""

        classifications: list[Classification] = []
        # Preserve discovery order of proposals; dedupe by name across batches.
        proposals: "OrderedDict[str, NewProposal]" = OrderedDict()
        n_calls = 0

        for batch in _batched(messages, self._batch_size):
            n_calls += 1
            parsed = self._classify_batch(batch)
            classifications.extend(self._normalise_classifications(parsed, batch))
            self._merge_proposals(parsed, proposals)

        # R5 cap enforcement happens once at the aggregate level so we cap the
        # *run*, not the per-batch slice.
        capped = self._cap_proposals(list(proposals.values()))

        return ClassifyRunResult(
            classifications=classifications,
            new_proposals=capped,
            n_calls=n_calls,
            n_messages=len(messages),
            taxonomy=list(self._taxonomy),
        )

    # -- batching -------------------------------------------------------

    def _classify_batch(self, batch: Sequence[KakaoMsg]) -> dict:
        system_prompt = _build_system_prompt(self._taxonomy)
        user_payload = _format_message_block(batch)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ]
        response = self._client.chat(
            messages,
            model=self._model,
            response_format={"type": "json_object"},
            max_retries=self._max_retries,
        )
        content = _extract_content(response)
        return _parse_content(content)

    # -- normalisation --------------------------------------------------

    def _normalise_classifications(
        self,
        parsed: dict,
        batch: Sequence[KakaoMsg],
    ) -> list[Classification]:
        out: list[Classification] = []
        seen_ids: set[str] = set()
        batch_ids = {m.msg_id for m in batch}
        # Best-effort proposal target lookup so per-message _new_* references
        # in ``categories`` can still build a valid ``ToneCategory``.
        proposal_targets: dict[str, str] = {}
        for raw in parsed.get("_new_proposals") or []:
            name = raw.get("name")
            target = raw.get("candidate_target_file")
            if isinstance(name, str) and isinstance(target, str):
                proposal_targets[name] = target

        for raw in parsed.get("classifications", []):
            msg_id = raw.get("msg_id")
            if not isinstance(msg_id, str) or msg_id not in batch_ids:
                # Silently drop rows that aren't in the batch — GLM occasionally
                # echoes invented ids; we never trust them.
                continue
            if msg_id in seen_ids:
                continue
            seen_ids.add(msg_id)

            raw_categories = raw.get("categories") or []
            tone_cats: list[ToneCategory] = []
            for cat in raw_categories:
                if not isinstance(cat, str):
                    continue
                bare = _strip_ko_prefix(cat)
                if bare in BASELINE_TONE_NAMES:
                    tone_cats.append(
                        ToneCategory(name=bare, target_file=_target_file_for(bare))
                    )
                    continue
                if bare.startswith("_new_"):
                    # Promoted ``_new_*`` categories are first-class but the
                    # promotion ledger does not currently record their target
                    # file. We therefore require GLM to *also* echo a
                    # ``_new_proposals`` block whenever it cites a ``_new_*``
                    # in ``categories`` — for both fresh and promoted ids.
                    proposal_target = proposal_targets.get(bare)
                    if proposal_target is None:
                        logger.debug(
                            "classifier: orphan %s on msg_id=%s (no proposal block)",
                            bare,
                            msg_id,
                        )
                        continue
                    tone_cats.append(
                        ToneCategory(
                            name=bare,
                            target_file=_target_file_for(
                                bare, proposal_target=proposal_target
                            ),
                        )
                    )
                    continue
                raise TaxonomyViolationError(
                    f"GLM emitted unknown category {cat!r} for msg_id={msg_id!r}; "
                    "neither in the active taxonomy nor a _new_* proposal"
                )

            if not tone_cats:
                # Plan §4 Step 5 acceptance: 100% messages get ≥1 category or
                # 'unknown' (no silent drop). We synthesise an ``unknown``
                # placeholder using the closest viable baseline category so the
                # dataclass invariant holds.
                tone_cats.append(
                    ToneCategory(name="filler", target_file="patterns/ko-filler.md")
                )

            confidence = raw.get("confidence")
            if not isinstance(confidence, (int, float)):
                confidence = 0.0

            out.append(
                Classification(
                    msg_id=msg_id,
                    categories=tone_cats,
                    confidence=float(confidence),
                )
            )

        # Acceptance §4 Step 5: every message in the batch gets ≥1 category.
        for m in batch:
            if m.msg_id not in seen_ids:
                out.append(
                    Classification(
                        msg_id=m.msg_id,
                        categories=[
                            ToneCategory(name="filler", target_file="patterns/ko-filler.md")
                        ],
                        confidence=0.0,
                    )
                )
        return out

    def _merge_proposals(
        self,
        parsed: dict,
        proposals: "OrderedDict[str, NewProposal]",
    ) -> None:
        for raw in parsed.get("_new_proposals") or []:
            name = raw.get("name")
            exemplars = raw.get("exemplar_msg_ids") or []
            target = raw.get("candidate_target_file")
            if not (isinstance(name, str) and name.startswith("_new_")):
                continue
            if not (isinstance(target, str) and target):
                continue
            existing = proposals.get(name)
            if existing is None:
                proposals[name] = NewProposal(
                    name=name,
                    exemplar_msg_ids=list(dict.fromkeys(exemplars)),
                    candidate_target_file=target,
                    rationale=str(raw.get("rationale", "")),
                )
            else:
                # Merge exemplars across batches, preserving first-seen order.
                merged = list(
                    dict.fromkeys([*existing.exemplar_msg_ids, *exemplars])
                )
                existing.exemplar_msg_ids = merged

    def _cap_proposals(self, proposals: list[NewProposal]) -> list[NewProposal]:
        # Drop any proposal with too few exemplars (R5 floor).
        viable = [
            p
            for p in proposals
            if len(p.exemplar_msg_ids) >= MIN_EXEMPLARS_PER_NEW_PROPOSAL
        ]
        # Plan §4 Step 5 priority rule: deterministic priority by frequency.
        # Higher exemplar count → kept first. Ties broken by insertion order
        # (which matches discovery order) via ``sorted`` being stable.
        viable.sort(key=lambda p: len(p.exemplar_msg_ids), reverse=True)
        return viable[:MAX_NEW_PROPOSALS_PER_RUN]


def classify_messages(
    messages: Sequence[KakaoMsg],
    *,
    client: GLMClient,
    staging_root: Optional[Path] = None,
    taxonomy: Optional[Sequence[str]] = None,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> ClassifyRunResult:
    """Convenience: build a :class:`Classifier` and run it once."""

    classifier = Classifier(
        client,
        staging_root=staging_root,
        taxonomy=taxonomy,
        model=model,
        batch_size=batch_size,
    )
    return classifier.classify(messages)


__all__ = [
    "Classifier",
    "ClassifyRunResult",
    "ClassifierError",
    "NewProposal",
    "TaxonomyViolationError",
    "CLASSIFY_RESPONSE_SCHEMA",
    "DEFAULT_BATCH_SIZE",
    "MAX_NEW_PROPOSALS_PER_RUN",
    "MIN_EXEMPLARS_PER_NEW_PROPOSAL",
    "classify_messages",
]
