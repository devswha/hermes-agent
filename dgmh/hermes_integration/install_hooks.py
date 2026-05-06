"""
dgmh/hermes_integration/install_hooks.py — Install DGM-H hooks into ~/.hermes/hooks/.

Run once before (or after) service restart:
  python3 -m dgmh.hermes_integration.install_hooks

This script:
  1. Copies dgmh-reaction hook to ~/.hermes/hooks/dgmh-reaction/
  2. Adds dgmh.gate_native_review: true to ~/.hermes/config.yaml (idempotent)

Phase-1 environment variables (v3 Step 9):

  DGMH_PUBLIC_HUMAN_MODE
    Master switch for the public-channel "indistinguishable persona"
    pipeline (Phase 1: voice rewrite + bimodal latency + channel-tag
    Honcho isolation + persona pinning).
      unset / "0" / "false"  → OFF (default; behavior identical to pre-v3)
      "1"                    → ON (public channels run the full pipeline)
    Single env flag revert: setting back to unset restores prior behavior
    on the next gateway restart.

  DGMH_PUBLIC_CHANNELS
    Comma-separated list of Discord channel ids that are PUBLIC. The
    operator's 1:1 channel id (default 1496872245027541062) MUST NOT
    appear here. Default empty → no channel is treated as public, so
    the bimodal latency / public-mode rewrite never fires even with
    DGMH_PUBLIC_HUMAN_MODE=1.

  DGMH_OPERATOR_USER_ID
    Operator's Discord user id; comma-separated for multi-id allowlist.
    Default 266436073557590016. Drives the AC1.1 operator-shortcut
    latency bucket (operator pings stay in [3, 15] seconds regardless
    of channel activity).

  DGMH_PUBLIC_PERSONA_NAME
    Display name the persona uses on public surfaces. Default "flask".
    Phase 1 still posts via the bot account, so this just informs the
    SOUL.md PERSONA-IDENTITY block; Phase 2 (webhook relay) will pass
    this value as the webhook display name.

  DGMH_PUBLIC_PERSONA_AVATAR_URL
    Phase-2 placeholder. Webhook avatar URL the persona will use when
    the webhook relay (Step W) ships. Phase 1 reads it but does not
    use it.

Other already-documented env knobs (unchanged): DGMH_HUMANNESS_DISABLED,
DGMH_HUMANNESS_MIN_CHARS, DGMH_PRUNE_DISABLED, DGMH_PRUNE_AI_THRESHOLD,
DGMH_REWRITE_ENABLED, DGMH_HONCHO_DISABLED, DGMH_HONCHO_ALLOWED_CHANNELS,
DGMH_PATINA_BIN, DGMH_CODEX_BIN.
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


def install_honcho_hook() -> None:
    """Copy the dgmh-honcho hook directory to ~/.hermes/hooks/dgmh-honcho/."""
    src = Path(__file__).parent / "hooks" / "dgmh-honcho"
    dst = _hermes_home() / "hooks" / "dgmh-honcho"
    dst.mkdir(parents=True, exist_ok=True)
    for filename in ("HOOK.yaml", "handler.py"):
        shutil.copy2(src / filename, dst / filename)
    print(f"[install_hooks] Installed dgmh-honcho hook to {dst}")


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
    install_honcho_hook()
    patch_config_yaml()
    print("[install_hooks] Done. Restart hermes-gateway to activate.")


if __name__ == "__main__":
    main()
