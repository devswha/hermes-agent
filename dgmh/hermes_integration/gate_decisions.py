"""DGM-H gate decision cache — track pre_send_gate outcomes per
outbound message_id so reaction_hook can classify thumbs-down reactions
by whether the message was truncated mid-sentence.

In-memory only. Bounded LRU. Gateway restart clears state — that is
fine because 👎 typically fires within minutes of the send.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional


_CACHE_MAX = 256
_lock = threading.Lock()
_cache: "OrderedDict[str, GateDecision]" = OrderedDict()


@dataclass(frozen=True)
class GateDecision:
    decision: str
    in_len: int
    out_len: int
    ends_in_ellipsis: bool

    @property
    def truncated(self) -> bool:
        return self.decision == "compress" or self.ends_in_ellipsis


def record_decision(
    msg_id: str,
    *,
    decision: str,
    in_len: int,
    out_len: int,
    ends_in_ellipsis: bool = False,
) -> None:
    """Cache a gate decision for the given outbound message id."""
    if not msg_id:
        return
    with _lock:
        if msg_id in _cache:
            _cache.move_to_end(msg_id)
        _cache[msg_id] = GateDecision(
            decision=decision,
            in_len=in_len,
            out_len=out_len,
            ends_in_ellipsis=ends_in_ellipsis,
        )
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)


def get_decision(msg_id: str) -> Optional[GateDecision]:
    """Return the cached gate decision, or None when unknown."""
    if not msg_id:
        return None
    with _lock:
        return _cache.get(msg_id)


def _reset_for_tests() -> None:
    with _lock:
        _cache.clear()


def _ends_with_ellipsis_marker(text: str) -> bool:
    """Return True when ``text`` ends with one of the truncation markers
    that the gate appends (Unicode `…` U+2026 or ASCII `...`).
    """
    if not text:
        return False
    return text.endswith("…") or text.endswith("...")
