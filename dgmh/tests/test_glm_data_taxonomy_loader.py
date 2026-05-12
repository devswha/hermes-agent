"""Tests for ``dgmh.glm_data.taxonomy_loader``.

Plan reference: ``.omc/plans/dgmh-glm-data-v1.md`` §4 Step 5b (promotion
ledger) and AC5 (taxonomy conformance). The three named tests in the task
brief plus a few sanity guards.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dgmh.glm_data import taxonomy_loader
from dgmh.glm_data.taxonomy_loader import BASELINE_KO_7, load, promote


@pytest.fixture()
def staging_root(tmp_path: Path) -> Path:
    root = tmp_path / "staging"
    root.mkdir()
    return root


# --------------------------------------------------------------- core scenarios


def test_taxonomy_loader_empty_promoted_returns_baseline_7(
    staging_root: Path,
) -> None:
    """No promoted_taxonomy.yaml on disk → load() returns baseline only (J5)."""

    assert not (staging_root / "promoted_taxonomy.yaml").exists()
    assert load(staging_root) == BASELINE_KO_7
    assert len(load(staging_root)) == 7


def test_taxonomy_loader_after_promote_includes_new(staging_root: Path) -> None:
    """promote() then load() — the new category appears after baseline."""

    promote("_new_kakao_register_chimaek", staging_root)

    result = load(staging_root)
    assert result[: len(BASELINE_KO_7)] == BASELINE_KO_7
    assert result[-1] == "_new_kakao_register_chimaek"
    assert len(result) == 8


def test_taxonomy_loader_round_trip_preserves_order(staging_root: Path) -> None:
    """Multiple promotions retain their insertion order across baseline."""

    promote("_new_a", staging_root)
    promote("_new_b", staging_root)
    promote("_new_c", staging_root)

    result = load(staging_root)
    assert result == BASELINE_KO_7 + ["_new_a", "_new_b", "_new_c"]


# ---------------------------------------------------------- append-only invariant


def test_promote_is_append_only_no_reorder(staging_root: Path) -> None:
    """Re-promoting an existing id is a no-op (ledger never reorders)."""

    promote("_new_a", staging_root)
    promote("_new_b", staging_root)
    promote("_new_a", staging_root)  # duplicate — must be ignored

    result = load(staging_root)
    assert result == BASELINE_KO_7 + ["_new_a", "_new_b"]

    raw = yaml.safe_load((staging_root / "promoted_taxonomy.yaml").read_text())
    assert [entry["category_id"] for entry in raw] == ["_new_a", "_new_b"]


def test_promote_records_promoted_at_timestamp(staging_root: Path) -> None:
    promote("_new_x", staging_root)
    raw = yaml.safe_load((staging_root / "promoted_taxonomy.yaml").read_text())
    assert "promoted_at" in raw[0]
    # ISO-8601 with explicit timezone suffix
    assert raw[0]["promoted_at"].endswith("+00:00")


# ----------------------------------------------------------- input validation


def test_promote_rejects_non_new_prefix(staging_root: Path) -> None:
    with pytest.raises(ValueError):
        promote("communication", staging_root)  # baseline cannot be promoted
    with pytest.raises(ValueError):
        promote("kakao_register_chimaek", staging_root)  # missing prefix


def test_promote_rejects_empty(staging_root: Path) -> None:
    with pytest.raises(ValueError):
        promote("", staging_root)


def test_load_rejects_corrupted_ledger(staging_root: Path) -> None:
    (staging_root / "promoted_taxonomy.yaml").write_text("not_a_list: true\n")
    with pytest.raises(ValueError):
        load(staging_root)


def test_load_handles_empty_yaml_file(staging_root: Path) -> None:
    (staging_root / "promoted_taxonomy.yaml").write_text("")
    assert load(staging_root) == BASELINE_KO_7


def test_iter_promoted_excludes_baseline(staging_root: Path) -> None:
    promote("_new_alpha", staging_root)
    promote("_new_beta", staging_root)
    assert list(taxonomy_loader.iter_promoted(staging_root)) == [
        "_new_alpha",
        "_new_beta",
    ]
