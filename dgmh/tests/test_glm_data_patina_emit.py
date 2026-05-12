"""
dgmh/tests/test_glm_data_patina_emit.py — Step 9 tests for `patina_emit.py`.

Covers Plan v2.1 §3 AC8, AC9 + v2.1 J1:
  - Language-scoped FLAT pattern-block numbering across ALL ko-*.md.
  - `entries:` (lexicon) and `patterns:` (pattern files) bumped exactly.
  - All front-matter fields round-trip bit-for-bit (incl. `score_only: true`).
  - 8-field block schema present in rendered output.
  - Markdown_it parses both patches without errors.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from markdown_it import MarkdownIt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dgmh.glm_data.patina_emit import (  # noqa: E402
    PatinaLexiconEntries,
    PatinaPatternBlock,
    emit_lexicon_patch,
    emit_pattern_patch,
    flat_pattern_namespace,
    next_free_pattern_number,
    scan_pattern_numbers,
)

PATINA_ROOT = Path("/home/devswha/workspace/patina")
LANG = "ko"

# Reusable block fixtures.
SAMPLE_BLOCK_A = PatinaPatternBlock(
    title="새 패턴 A",
    keywords="어휘1, 어휘2, 어휘3",
    problem="문제 설명 한 줄.",
    fire_condition="조건 한 줄.",
    exclusion="- 예외 1\n- 예외 2",
    severity="LOW",
    before="원래 문장.",
    after="다듬은 문장.",
)
SAMPLE_BLOCK_B = PatinaPatternBlock(
    title="새 패턴 B",
    keywords="어휘4",
    problem="문제 B",
    fire_condition="조건 B",
    exclusion="- 예외 B",
    severity="MEDIUM",
    before="전 B",
    after="후 B",
)


# --------------------------------------------------------------------------- #
# Namespace scanning (J1)
# --------------------------------------------------------------------------- #


def test_scan_pattern_numbers_returns_one_set_per_pack_file():
    per_file = scan_pattern_numbers(PATINA_ROOT, LANG)
    # All 7 ko-*.md files
    assert len(per_file) == 7
    # Each pack contributes ≥ 1 block.
    for nums in per_file.values():
        assert len(nums) >= 1


def test_flat_namespace_is_union_across_all_pack_files():
    per_file = scan_pattern_numbers(PATINA_ROOT, LANG)
    flat = flat_pattern_namespace(PATINA_ROOT, LANG)
    expected = set().union(*per_file.values())
    assert flat == expected


def test_flat_namespace_includes_known_landmarks():
    """ko-language #8 (~적 접미사) and ko-style #14 (볼드체) — sanity probes
    from scoring.md:147-148."""
    flat = flat_pattern_namespace(PATINA_ROOT, LANG)
    assert 8 in flat
    assert 14 in flat


def test_next_free_pattern_number_equals_max_plus_one():
    flat = flat_pattern_namespace(PATINA_ROOT, LANG)
    assert next_free_pattern_number(PATINA_ROOT, LANG) == max(flat) + 1


# --------------------------------------------------------------------------- #
# Pattern patch emission
# --------------------------------------------------------------------------- #


def test_pattern_patch_assigns_global_max_plus_one_regardless_of_target_file(
    tmp_path: Path,
):
    """A new block destined for ko-communication.md should still receive the
    LANGUAGE-WIDE next free number, not communication's local max+1."""
    starting = next_free_pattern_number(PATINA_ROOT, LANG)
    out = emit_pattern_patch(
        PATINA_ROOT,
        "ko-communication",
        [SAMPLE_BLOCK_A, SAMPLE_BLOCK_B],
        tmp_path / "ko-communication.patch.md",
    )
    text = out.read_text(encoding="utf-8")
    headers = [int(m) for m in re.findall(r"^###\s+(\d+)\.\s", text, re.M)]
    # New blocks present and renumbered at the global max+1, +2.
    assert starting in headers
    assert (starting + 1) in headers


def test_pattern_patch_bumps_patterns_count_exactly(tmp_path: Path):
    src = (PATINA_ROOT / "patterns" / "ko-filler.md").read_text(encoding="utf-8")
    old = int(re.search(r"^patterns:\s*(\d+)", src, re.M).group(1))
    out = emit_pattern_patch(
        PATINA_ROOT,
        "ko-filler",
        [SAMPLE_BLOCK_A, SAMPLE_BLOCK_B],
        tmp_path / "ko-filler.patch.md",
    )
    text = out.read_text(encoding="utf-8")
    new = int(re.search(r"^patterns:\s*(\d+)", text, re.M).group(1))
    assert new == old + 2


def test_pattern_patch_renders_8_field_block_schema(tmp_path: Path):
    out = emit_pattern_patch(
        PATINA_ROOT,
        "ko-style",
        [SAMPLE_BLOCK_A],
        tmp_path / "ko-style.patch.md",
    )
    text = out.read_text(encoding="utf-8")
    # Locate the new block by its title and verify the 8 markers exist after it.
    idx = text.index("새 패턴 A")
    tail = text[idx:]
    for marker in (
        "**주의 어휘:**",
        "**문제:**",
        "**발화 조건:**",
        "**제외 조건:**",
        "**의미 위험도:**",
        "**수정 전:**",
        "**수정 후:**",
    ):
        assert marker in tail, f"missing field marker: {marker}"


def test_pattern_patch_preserves_front_matter_fields(tmp_path: Path):
    """All keys present in the source front-matter must appear in the patch,
    only `patterns:` is allowed to differ in value."""
    src_text = (PATINA_ROOT / "patterns" / "ko-structure.md").read_text(
        encoding="utf-8"
    )
    src_fm = _front_matter_kv(src_text)

    out = emit_pattern_patch(
        PATINA_ROOT,
        "ko-structure",
        [SAMPLE_BLOCK_A],
        tmp_path / "ko-structure.patch.md",
    )
    patch_fm = _front_matter_kv(out.read_text(encoding="utf-8"))

    # Same key set + same order.
    assert list(patch_fm.keys()) == list(src_fm.keys())
    for k, v in src_fm.items():
        if k == "patterns":
            continue
        assert patch_fm[k] == v, f"field {k!r} changed: {v!r} → {patch_fm[k]!r}"


def test_block_numbers_globally_unique_after_patch(tmp_path: Path):
    """After emitting a patch, the union of (existing pack numbers ∪ new block
    numbers) must contain no duplicates."""
    flat_before = flat_pattern_namespace(PATINA_ROOT, LANG)
    out = emit_pattern_patch(
        PATINA_ROOT,
        "ko-language",
        [SAMPLE_BLOCK_A, SAMPLE_BLOCK_B],
        tmp_path / "ko-language.patch.md",
    )
    text = out.read_text(encoding="utf-8")
    nums_in_patch = [int(m) for m in re.findall(r"^###\s+(\d+)\.\s", text, re.M)]
    assert len(nums_in_patch) == len(set(nums_in_patch)), "duplicate within patch"
    # The two NEW numbers must not have been used anywhere in the existing pack.
    new_nums = nums_in_patch[-2:]
    assert set(new_nums).isdisjoint(flat_before)


# --------------------------------------------------------------------------- #
# ko-viral-hook — score_only preservation (v2.1 D-6)
# --------------------------------------------------------------------------- #


def test_viral_hook_score_only_preserved_after_patch(tmp_path: Path):
    """score_only: true must survive a synthetic patch byte-for-byte."""
    out = emit_pattern_patch(
        PATINA_ROOT,
        "ko-viral-hook",
        [SAMPLE_BLOCK_A],
        tmp_path / "ko-viral-hook.patch.md",
    )
    text = out.read_text(encoding="utf-8")
    fm = _front_matter_kv(text)
    assert fm.get("score_only") == "true"


def test_viral_hook_new_block_uses_language_wide_max_plus_one_not_local_6(
    tmp_path: Path,
):
    """The CRITICAL J1 contract: ko-viral-hook editorially uses 1–5 but shares
    the language-wide flat namespace. A new block must claim
    `next_free_pattern_number()`, not 6."""
    starting = next_free_pattern_number(PATINA_ROOT, LANG)
    assert starting > 6, "premise failed: language-wide max is already > 5"
    out = emit_pattern_patch(
        PATINA_ROOT,
        "ko-viral-hook",
        [SAMPLE_BLOCK_A],
        tmp_path / "ko-viral-hook.patch.md",
    )
    text = out.read_text(encoding="utf-8")
    headers = [int(m) for m in re.findall(r"^###\s+(\d+)\.\s", text, re.M)]
    # Original 1-5 still present and the new block uses `starting`, not 6.
    for n in (1, 2, 3, 4, 5):
        assert n in headers
    assert 6 not in headers
    assert starting in headers


# --------------------------------------------------------------------------- #
# Lexicon patch
# --------------------------------------------------------------------------- #


def test_lexicon_patch_bumps_entries_count_exactly(tmp_path: Path):
    src = (PATINA_ROOT / "lexicon" / "ai-ko.md").read_text(encoding="utf-8")
    old = int(re.search(r"^entries:\s*(\d+)", src, re.M).group(1))

    new = PatinaLexiconEntries(
        strict=("새단어1", "새단어2", "새단어3"),
        phrases=("새 구문 하나", "새 구문 둘"),
    )
    out = emit_lexicon_patch(PATINA_ROOT, new, tmp_path / "ai-ko.patch.md")
    text = out.read_text(encoding="utf-8")
    new_count = int(re.search(r"^entries:\s*(\d+)", text, re.M).group(1))
    assert new_count == old + 5


def test_lexicon_patch_appends_under_correct_sections(tmp_path: Path):
    new = PatinaLexiconEntries(
        strict=("XYZ_STRICT",), phrases=("ABC PHRASE",)
    )
    out = emit_lexicon_patch(PATINA_ROOT, new, tmp_path / "ai-ko.patch.md")
    text = out.read_text(encoding="utf-8")

    strict_h = text.index("## Strict matches")
    phrase_h = text.index("## Multi-word phrases")
    xyz = text.index("XYZ_STRICT")
    abc = text.index("ABC PHRASE")

    # Strict entry lands in the Strict-matches section.
    assert strict_h < xyz < phrase_h
    # Phrase entry lands after the Multi-word-phrases header.
    assert phrase_h < abc


def test_lexicon_patch_preserves_front_matter_fields(tmp_path: Path):
    src_text = (PATINA_ROOT / "lexicon" / "ai-ko.md").read_text(encoding="utf-8")
    src_fm = _front_matter_kv(src_text)

    new = PatinaLexiconEntries(strict=("aaa",))
    out = emit_lexicon_patch(PATINA_ROOT, new, tmp_path / "ai-ko.patch.md")
    patch_fm = _front_matter_kv(out.read_text(encoding="utf-8"))

    assert list(patch_fm.keys()) == list(src_fm.keys())
    for k, v in src_fm.items():
        if k == "entries":
            continue
        assert patch_fm[k] == v


# --------------------------------------------------------------------------- #
# Markdown_it parses both patches
# --------------------------------------------------------------------------- #


def test_markdown_it_parses_pattern_patch_without_error(tmp_path: Path):
    out = emit_pattern_patch(
        PATINA_ROOT,
        "ko-style",
        [SAMPLE_BLOCK_A, SAMPLE_BLOCK_B],
        tmp_path / "ko-style.patch.md",
    )
    md = MarkdownIt("commonmark")
    tokens = md.parse(out.read_text(encoding="utf-8"))
    assert len(tokens) > 0
    # ### headers parse to heading_open tokens.
    heading_opens = [t for t in tokens if t.type == "heading_open"]
    assert any(t.tag == "h3" for t in heading_opens)


def test_markdown_it_parses_lexicon_patch_without_error(tmp_path: Path):
    new = PatinaLexiconEntries(strict=("aa",), phrases=("bb cc",))
    out = emit_lexicon_patch(PATINA_ROOT, new, tmp_path / "ai-ko.patch.md")
    md = MarkdownIt("commonmark")
    tokens = md.parse(out.read_text(encoding="utf-8"))
    assert len(tokens) > 0
    # ## section headers present.
    h2_opens = [t for t in tokens if t.type == "heading_open" and t.tag == "h2"]
    assert len(h2_opens) >= 2


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _front_matter_kv(text: str) -> dict[str, str]:
    """Tiny parser: capture key→value pairs from the leading `--- … ---` block.

    Preserves first-seen ordering via dict insertion order.
    """
    out: dict[str, str] = {}
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        raise AssertionError("front-matter marker missing at line 0")
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return out
        if ":" in lines[i]:
            k, _, v = lines[i].partition(":")
            out[k.strip()] = v.strip()
    raise AssertionError("unterminated front-matter")
