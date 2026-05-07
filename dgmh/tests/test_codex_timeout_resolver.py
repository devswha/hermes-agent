"""Unit tests for DGMH_CODEX_TIMEOUT_S env override resolver.

The resolver lives identically in modifier.py, critic.py, and judge.py
(copied rather than shared to keep each module self-contained, matching
their existing pattern of duplicating _invoke_codex). Tests exercise all
three to ensure consistent behavior.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from dgmh.critic import _resolve_timeout_s as critic_resolve
from dgmh.judge import _resolve_timeout_s as judge_resolve
from dgmh.modifier import _resolve_timeout_s as modifier_resolve

_RESOLVERS = (modifier_resolve, critic_resolve, judge_resolve)
_BUMPED_DEFAULT = 180.0


def _clear_env() -> dict[str, str]:
    """Return an env dict identical to os.environ but with the override removed."""
    env = dict(os.environ)
    env.pop("DGMH_CODEX_TIMEOUT_S", None)
    return env


@pytest.mark.parametrize("resolve", _RESOLVERS)
def test_default_when_env_unset(resolve):
    with patch.dict(os.environ, _clear_env(), clear=True):
        assert resolve() == _BUMPED_DEFAULT


@pytest.mark.parametrize("resolve", _RESOLVERS)
def test_env_override_valid(resolve):
    with patch.dict(os.environ, {"DGMH_CODEX_TIMEOUT_S": "120"}, clear=False):
        assert resolve() == 120.0


@pytest.mark.parametrize("resolve", _RESOLVERS)
def test_env_invalid_falls_back_to_default(resolve):
    with patch.dict(os.environ, {"DGMH_CODEX_TIMEOUT_S": "abc"}, clear=False):
        assert resolve() == _BUMPED_DEFAULT


@pytest.mark.parametrize("resolve", _RESOLVERS)
def test_env_below_floor_clamps_to_floor(resolve):
    with patch.dict(os.environ, {"DGMH_CODEX_TIMEOUT_S": "5"}, clear=False):
        assert resolve() == 10.0


@pytest.mark.parametrize("resolve", _RESOLVERS)
def test_env_above_ceiling_clamps_to_ceiling(resolve):
    with patch.dict(os.environ, {"DGMH_CODEX_TIMEOUT_S": "9999"}, clear=False):
        assert resolve() == 600.0


@pytest.mark.parametrize("resolve", _RESOLVERS)
def test_env_empty_string_falls_back_to_default(resolve):
    with patch.dict(os.environ, {"DGMH_CODEX_TIMEOUT_S": ""}, clear=False):
        assert resolve() == _BUMPED_DEFAULT
