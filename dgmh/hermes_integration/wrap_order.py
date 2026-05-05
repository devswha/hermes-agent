"""DGM-H wrap-order verification helper.

Both ``humanness_hook._wrap_send`` and ``honcho_hook._wrap_send`` monkey-patch
``adapter.send``. The order matters:

  - humanness wraps **innermost** (closest to the real ``adapter.send``).
    It performs the patina rewrite + score side-effect on the actual outbound
    text. Priority 100 in HOOK.yaml.
  - honcho wraps **outermost**. It mirrors the post-rewrite content into the
    Honcho memory layer so the peer model never sees pre-rewrite drafts.
    Priority 50 in HOOK.yaml.

Lower priority runs later → wraps outer. Higher priority runs first → wraps
inner. The HOOK.yaml field is documentation today; Hermes' loader does not
yet honour it. Runtime ordering is enforced by ``_verify_wrap_chain`` below
which walks the ``__wrapped__`` chain from outer to inner and asserts the
sequence ``["honcho", "humanness"]``.

Each wrapper sets two attributes on its ``wrapped_send`` callable:

  - ``__dgmh_hook_name__``: short string identifier (e.g. ``"humanness"``)
  - ``__wrapped__``: the original ``send`` it wrapped over

Walking the chain reveals the pipeline order. If the chain is incomplete (one
hook hasn't wrapped yet at startup), ``_verify_wrap_chain`` returns ``False``
without raising — the caller logs a warning and the partner hook will run
verification once it finishes wrapping.
"""

from __future__ import annotations

import logging
from typing import Any, List

logger = logging.getLogger(__name__)


# Outer-to-inner expected order: honcho is outermost, humanness is innermost.
EXPECTED_CHAIN: List[str] = ["honcho", "humanness"]


def _walk_chain(send_fn: Any, max_depth: int = 8) -> List[str]:
    """Walk ``__wrapped__`` from outer to inner; return list of hook names seen.

    Stops at the first link that has no ``__dgmh_hook_name__`` (the original
    ``adapter.send``) or after ``max_depth`` to guard against pathological
    self-loops.
    """
    seen: List[str] = []
    fn = send_fn
    for _ in range(max_depth):
        name = getattr(fn, "__dgmh_hook_name__", None)
        if name is None:
            break
        seen.append(str(name))
        fn = getattr(fn, "__wrapped__", None)
        if fn is None:
            break
    return seen


def verify_wrap_chain(adapter: Any, *, expected: List[str] | None = None) -> bool:
    """Verify ``adapter.send`` is wrapped in the documented order.

    Returns ``True`` only when the chain matches ``expected`` exactly. Any
    mismatch — including a partially-wrapped chain — returns ``False`` and
    logs at WARNING / ERROR level so the gateway log captures the state.

    Caller invokes this at the tail of its own ``_wait_and_patch`` so the
    last hook to finish wrapping is the one whose check actually fires.
    """
    expected = expected or EXPECTED_CHAIN
    send_fn = getattr(adapter, "send", None)
    if send_fn is None:
        logger.error("[wrap_order] adapter.send missing; cannot verify chain")
        return False

    seen = _walk_chain(send_fn)
    if seen == expected:
        logger.info(
            "[wrap_order] adapter.send chain OK: outer→inner = %s", seen
        )
        return True

    # Partial chain (one hook not wrapped yet) is expected during startup.
    if len(seen) < len(expected):
        logger.warning(
            "[wrap_order] adapter.send chain incomplete: got %s, expected %s "
            "(partner hook may not have wrapped yet)",
            seen, expected,
        )
        return False

    logger.error(
        "[wrap_order] adapter.send wrap-order broken: got %s, expected %s",
        seen, expected,
    )
    return False
