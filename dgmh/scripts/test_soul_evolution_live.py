"""Direct end-to-end test of SOUL.md evolution using REAL Codex.

Bypasses Discord. Exercises:
- soul archive bootstrap (gen-0 = current SOUL.md)
- SkillModifier (real Codex CLI)
- SkillCritic (real Codex CLI)
- atomic SOUL.md replace on approve

Usage:
    cd /home/devswha/workspace/hermes-agent
    python -m dgmh.scripts.test_soul_evolution_live
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("test_soul_live")


def main() -> int:
    from dgmh.soul_evolution import run_soul_evolution, SoulEvolutionOpts

    soul_path = Path.home() / ".hermes" / "SOUL.md"
    if not soul_path.exists():
        logger.error("SOUL.md not found at %s", soul_path)
        return 1

    pre_content = soul_path.read_text(encoding="utf-8")
    pre_hash_short = str(hash(pre_content) & 0xFFFFFFFF)
    logger.info("=" * 60)
    logger.info("PRE SOUL.md hash=%s len=%d", pre_hash_short, len(pre_content))
    logger.info("First 200: %s", pre_content[:200].replace("\n", " | "))
    logger.info("=" * 60)

    logger.info("Calling run_soul_evolution(neg=1)...")
    try:
        admitted = run_soul_evolution(SoulEvolutionOpts(pos_reactions=0, neg_reactions=1))
    except Exception as exc:
        logger.exception("raised: %s", exc)
        return 2

    post_content = soul_path.read_text(encoding="utf-8")
    post_hash_short = str(hash(post_content) & 0xFFFFFFFF)
    logger.info("=" * 60)
    logger.info("admitted=%s", admitted)
    logger.info("POST SOUL.md hash=%s len=%d", post_hash_short, len(post_content))
    logger.info("First 200: %s", post_content[:200].replace("\n", " | "))
    logger.info("=" * 60)

    if pre_content == post_content:
        logger.warning("SOUL.md UNCHANGED")
    else:
        logger.info("SOUL.md MUTATED OK (hash %s -> %s)", pre_hash_short, post_hash_short)

    return 0 if admitted else 3


if __name__ == "__main__":
    sys.exit(main())
