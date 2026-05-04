"""
dgmh/hermes_integration/install_hooks.py — Install DGM-H hooks into ~/.hermes/hooks/.

Run once before (or after) service restart:
  python3 -m dgmh.hermes_integration.install_hooks

This script:
  1. Copies dgmh-reaction hook to ~/.hermes/hooks/dgmh-reaction/
  2. Adds dgmh.gate_native_review: true to ~/.hermes/config.yaml (idempotent)
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


def install_reaction_hook() -> None:
    """Copy the dgmh-reaction hook directory to ~/.hermes/hooks/dgmh-reaction/."""
    src = Path(__file__).parent / "hooks" / "dgmh-reaction"
    dst = _hermes_home() / "hooks" / "dgmh-reaction"
    dst.mkdir(parents=True, exist_ok=True)
    for filename in ("HOOK.yaml", "handler.py"):
        shutil.copy2(src / filename, dst / filename)
    print(f"[install_hooks] Installed dgmh-reaction hook to {dst}")


def install_humanness_hook() -> None:
    """Copy the dgmh-humanness hook directory to ~/.hermes/hooks/dgmh-humanness/."""
    src = Path(__file__).parent / "hooks" / "dgmh-humanness"
    dst = _hermes_home() / "hooks" / "dgmh-humanness"
    dst.mkdir(parents=True, exist_ok=True)
    for filename in ("HOOK.yaml", "handler.py"):
        shutil.copy2(src / filename, dst / filename)
    print(f"[install_hooks] Installed dgmh-humanness hook to {dst}")


def patch_config_yaml() -> None:
    """Add dgmh.gate_native_review: true to ~/.hermes/config.yaml (idempotent)."""
    import yaml  # type: ignore[import]

    config_path = _hermes_home() / "config.yaml"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    dgmh_section = cfg.get("dgmh") or {}
    if dgmh_section.get("gate_native_review") is True:
        print("[install_hooks] config.yaml: dgmh.gate_native_review already set")
        return

    dgmh_section["gate_native_review"] = True
    cfg["dgmh"] = dgmh_section

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"[install_hooks] Patched {config_path}: dgmh.gate_native_review = true")


def main() -> None:
    install_reaction_hook()
    install_humanness_hook()
    patch_config_yaml()
    print("[install_hooks] Done. Restart hermes-gateway to activate.")


if __name__ == "__main__":
    main()
