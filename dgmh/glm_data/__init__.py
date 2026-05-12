"""DGM-H GLM Data Collection v1 — staging pipeline for patina + anchor candidates.

All outputs land under ``~/.hermes/dgmh/glm_data_v1/staging/<session_id>/``
and are kept *outside* every git working tree (AC13 primary defense).

See ``.omc/plans/dgmh-glm-data-v1.md`` for the full design.
"""

from pathlib import Path

__version__ = "0.1.0"

# Canonical staging root. Resolved lazily by callers so unit tests can monkey-
# patch ``Path.home`` without import-time side effects.
STAGING_ROOT_RELATIVE = Path(".hermes/dgmh/glm_data_v1/staging")


def staging_root() -> Path:
    """Return the resolved canonical staging root for the current operator."""

    return (Path.home() / STAGING_ROOT_RELATIVE).resolve()


__all__ = ["__version__", "STAGING_ROOT_RELATIVE", "staging_root"]
