"""
dgmh/glm_data/patina_emit.py — Patina patch generator (Plan v2.1 §4 Step 9).

Generates v1-staging candidate patches for two patina surfaces:

  1. `lexicon/ai-ko.md`         — append entries to `## Strict matches` / `## Multi-word phrases`;
                                  bump `entries: N → N + M_new`.
  2. `patterns/ko-<category>.md` — append new 8-field pattern blocks; bump
                                  `patterns: N → N + M_new`.

CRITICAL contract (Plan §3 AC9 + v2.1 J1 + scoring.md §5):
  - Pattern-block numbering is **language-scoped flat** — when we add new blocks,
    each gets `max(### N across ALL ko-*.md) + 1, +2, ...` *regardless of which
    file it ends up in*. `ko-viral-hook.md` editorially uses 1–5 today but
    shares the same flat namespace; a NEW viral-hook block must claim the next
    free language-wide number (currently 33), NOT 6.
  - Front-matter is preserved bit-for-bit except for the `entries:` / `patterns:`
    count. `pack`, `language`, `name`, `version`, `phase`, `score_only`, etc.
    round-trip exactly.

Rendering is via Jinja2 templates — never raw GLM markdown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from jinja2 import Environment, StrictUndefined

# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PatinaPatternBlock:
    """8-field pattern block (per `ko-communication.md` entries 19/20).

    The block's number is assigned by the emitter from the global flat
    namespace; callers do not set it.
    """

    title: str  # "### N. <title>"
    keywords: str  # "**주의 어휘:** ..."
    problem: str  # "**문제:** ..."
    fire_condition: str  # "**발화 조건:** ..."
    exclusion: str  # "**제외 조건:** ..." — may include bullet lines
    severity: str  # "**의미 위험도:** LOW|MEDIUM|HIGH"
    before: str  # "**수정 전:** > ..."
    after: str  # "**수정 후:** > ..."


@dataclass(frozen=True)
class PatinaLexiconEntries:
    """New entries to append to `lexicon/ai-ko.md`."""

    strict: tuple[str, ...] = ()
    phrases: tuple[str, ...] = ()

    @property
    def total(self) -> int:
        return len(self.strict) + len(self.phrases)


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #

_env = Environment(
    autoescape=False,
    keep_trailing_newline=True,
    trim_blocks=False,
    lstrip_blocks=False,
    undefined=StrictUndefined,
)

_PATTERN_BLOCK_TEMPLATE = _env.from_string(
    "### {{ number }}. {{ block.title }}\n"
    "\n"
    "**주의 어휘:** {{ block.keywords }}\n"
    "\n"
    "**문제:** {{ block.problem }}\n"
    "\n"
    "**발화 조건:** {{ block.fire_condition }}\n"
    "\n"
    "**제외 조건:**\n"
    "{{ block.exclusion }}\n"
    "\n"
    "**의미 위험도:** {{ block.severity }}\n"
    "\n"
    "**수정 전:**\n"
    "> {{ block.before }}\n"
    "\n"
    "**수정 후:**\n"
    "> {{ block.after }}\n"
)


# --------------------------------------------------------------------------- #
# Pattern numbering — language-scoped flat namespace (v2.1 J1)
# --------------------------------------------------------------------------- #

_BLOCK_HEADER_RE = re.compile(r"^###\s+(\d+)\.\s+\S")


def _pack_files(patina_root: Path, language: str) -> list[Path]:
    """Return all `patterns/<language>-*.md` files in sorted order."""
    patterns_dir = patina_root / "patterns"
    return sorted(patterns_dir.glob(f"{language}-*.md"))


def scan_pattern_numbers(
    patina_root: Path, language: str = "ko"
) -> dict[Path, set[int]]:
    """Return per-file set of `### N. ...` block numbers."""
    out: dict[Path, set[int]] = {}
    for p in _pack_files(patina_root, language):
        nums: set[int] = set()
        for line in p.read_text(encoding="utf-8").splitlines():
            m = _BLOCK_HEADER_RE.match(line)
            if m:
                nums.add(int(m.group(1)))
        out[p] = nums
    return out


def flat_pattern_namespace(
    patina_root: Path, language: str = "ko"
) -> set[int]:
    """Union of all `### N. ...` block numbers across the language pack."""
    union: set[int] = set()
    for nums in scan_pattern_numbers(patina_root, language).values():
        union |= nums
    return union


def next_free_pattern_number(
    patina_root: Path, language: str = "ko"
) -> int:
    """Next free integer in the language-scoped flat namespace.

    Computed as ``max(used) + 1`` (or 1 if no patterns yet). The value is the
    same regardless of which target file the new block is destined for —
    this is the v2.1 J1 contract.
    """
    used = flat_pattern_namespace(patina_root, language)
    return (max(used) + 1) if used else 1


# --------------------------------------------------------------------------- #
# Front-matter handling (preserves order, comments, formatting bit-for-bit)
# --------------------------------------------------------------------------- #


@dataclass
class _FrontMatter:
    """A list-backed YAML-front-matter editor.

    We keep the original line text and only rewrite lines whose key matches a
    requested override. This preserves quoting, spacing, comments, and key
    order — important for AC9 ("All front-matter fields … round-trip preserved
    bit-for-bit except for the bumped fields").
    """

    raw_lines: list[str]  # lines BETWEEN the two `---` markers, no trailing \n
    leading: str = "---"  # always literally "---"
    trailing: str = "---"

    def value(self, key: str) -> Optional[str]:
        for line in self.raw_lines:
            k, _, v = _split_kv(line)
            if k == key:
                return v
        return None

    def set_value(self, key: str, value: str) -> None:
        for i, line in enumerate(self.raw_lines):
            k, sep, _ = _split_kv(line)
            if k == key:
                # Preserve "<key><sep_after_colon><new_value>" exactly.
                self.raw_lines[i] = f"{key}:{sep}{value}"
                return
        raise KeyError(key)

    def serialize(self) -> str:
        body = "\n".join(self.raw_lines)
        return f"{self.leading}\n{body}\n{self.trailing}\n"


def _split_kv(line: str) -> tuple[str, str, str]:
    """Return (key, separator_after_colon, value).

    ``separator_after_colon`` is the whitespace between the colon and value
    in the source line (usually `" "`); preserving it keeps round-trip exact.
    """
    if ":" not in line:
        return "", "", line
    key, _, rest = line.partition(":")
    key = key.strip()
    # Capture exact whitespace after colon.
    stripped = rest.lstrip(" ")
    sep = " " * (len(rest) - len(stripped))
    if not sep and rest.startswith(" "):
        sep = " "
    return key, sep or " ", stripped


def _parse_front_matter(text: str) -> tuple[_FrontMatter, str]:
    """Split front-matter from body. Body is everything after the trailing `---\\n`."""
    lines = text.split("\n")
    if not lines or lines[0].rstrip() != "---":
        raise ValueError("file does not start with `---` front-matter marker")
    end = None
    for i in range(1, len(lines)):
        if lines[i].rstrip() == "---":
            end = i
            break
    if end is None:
        raise ValueError("unterminated front-matter (no closing `---`)")
    raw = lines[1:end]
    body = "\n".join(lines[end + 1 :])
    fm = _FrontMatter(raw_lines=raw)
    return fm, body


# --------------------------------------------------------------------------- #
# Lexicon patch
# --------------------------------------------------------------------------- #

_STRICT_HEADER_RE = re.compile(r"^##\s+Strict matches\b")
_PHRASE_HEADER_RE = re.compile(r"^##\s+Multi-word phrases\b")
_NEXT_SECTION_RE = re.compile(r"^##\s+\S")


def _insert_under_section(
    body: str, header_re: re.Pattern[str], entries: Iterable[str]
) -> str:
    """Append `- entry` lines at the END of the section matched by `header_re`.

    The insertion point is immediately before the next `## ` header or EOF,
    trimming trailing blank lines so we don't leave double-blanks.
    """
    new_lines = [f"- {e}" for e in entries]
    if not new_lines:
        return body

    lines = body.split("\n")
    start = None
    for i, line in enumerate(lines):
        if header_re.match(line):
            start = i
            break
    if start is None:
        raise ValueError(f"section header not found: {header_re.pattern}")

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if _NEXT_SECTION_RE.match(lines[j]):
            end = j
            break

    # Strip trailing blank lines inside the section before our insertion point.
    insert_at = end
    while insert_at > start + 1 and lines[insert_at - 1].strip() == "":
        insert_at -= 1

    return "\n".join(
        lines[:insert_at] + new_lines + [""] + lines[insert_at:]
    )


def emit_lexicon_patch(
    patina_root: Path,
    new_entries: PatinaLexiconEntries,
    out_path: Path,
) -> Path:
    """Render the updated `lexicon/ai-ko.md` and write to `out_path`.

    Side effects: creates parent directories of `out_path` and writes the file.
    Returns `out_path`.
    """
    src = patina_root / "lexicon" / "ai-ko.md"
    text = src.read_text(encoding="utf-8")
    fm, body = _parse_front_matter(text)

    cur = fm.value("entries")
    if cur is None:
        raise ValueError("lexicon/ai-ko.md missing `entries:` field")
    new_count = int(cur) + new_entries.total
    fm.set_value("entries", str(new_count))

    body = _insert_under_section(body, _STRICT_HEADER_RE, new_entries.strict)
    body = _insert_under_section(body, _PHRASE_HEADER_RE, new_entries.phrases)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(fm.serialize() + body, encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------- #
# Pattern patch
# --------------------------------------------------------------------------- #


def emit_pattern_patch(
    patina_root: Path,
    target_pack: str,
    new_blocks: list[PatinaPatternBlock],
    out_path: Path,
    *,
    starting_number: Optional[int] = None,
    language: str = "ko",
) -> Path:
    """Render the updated `patterns/<target_pack>.md` and write to `out_path`.

    The block numbers are assigned from the language-scoped flat namespace
    starting at `starting_number` (default: `next_free_pattern_number(...)`).
    """
    src = patina_root / "patterns" / f"{target_pack}.md"
    text = src.read_text(encoding="utf-8")
    fm, body = _parse_front_matter(text)

    cur = fm.value("patterns")
    if cur is None:
        raise ValueError(f"{src.name} missing `patterns:` field")
    new_count = int(cur) + len(new_blocks)
    fm.set_value("patterns", str(new_count))

    if starting_number is None:
        starting_number = next_free_pattern_number(patina_root, language)

    rendered_blocks: list[str] = []
    for i, block in enumerate(new_blocks):
        number = starting_number + i
        rendered_blocks.append(
            _PATTERN_BLOCK_TEMPLATE.render(number=number, block=block)
        )

    body = body.rstrip("\n") + "\n\n---\n\n" + "\n---\n\n".join(rendered_blocks)
    if not body.endswith("\n"):
        body += "\n"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(fm.serialize() + body, encoding="utf-8")
    return out_path


__all__ = [
    "PatinaPatternBlock",
    "PatinaLexiconEntries",
    "scan_pattern_numbers",
    "flat_pattern_namespace",
    "next_free_pattern_number",
    "emit_lexicon_patch",
    "emit_pattern_patch",
]
