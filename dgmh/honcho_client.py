"""Honcho memory layer integration for the dgmh / flask Discord assistant.

Wraps the local self-hosted Honcho server (http://127.0.0.1:8000) so the
rest of dgmh can write Discord messages into structured peer/session memory
and pull peer representations + session summaries into the bot's system
prompt without each call site re-implementing the SDK boilerplate.

Concepts (mapped to our world):

  - Workspace: a single namespace, default ``dgmh-flask``.
  - Peers: ``devswha`` (operator), ``flask`` (the bot itself).
  - Session: one per Discord chat surface — keyed by channel/thread id.

Honcho is configured for the $0 self-host path: Ollama LLM (``llama3.2:3b``)
+ ``nomic-embed-text`` embeddings, all on localhost. The deriver service
runs background memory extraction asynchronously after each message — we
do not await it.

Disable everything by setting ``DGMH_HONCHO_DISABLED=1`` in the env. All
public functions degrade gracefully on connection failure (log + skip)
so a Honcho outage never blocks the gateway response path.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


_DEFAULT_BASE_URL = "http://127.0.0.1:8000"
_DEFAULT_WORKSPACE = "dgmh-flask"
_DEFAULT_OPERATOR_PEER = "devswha"
_DEFAULT_BOT_PEER = "flask"


@dataclass(frozen=True)
class HonchoConfig:
    base_url: str
    api_key: str
    workspace_id: str
    operator_peer: str
    bot_peer: str

    @classmethod
    def from_env(cls) -> "HonchoConfig":
        return cls(
            base_url=os.environ.get("DGMH_HONCHO_BASE_URL", _DEFAULT_BASE_URL),
            api_key=os.environ.get("DGMH_HONCHO_API_KEY", "unused-but-required"),
            workspace_id=os.environ.get("DGMH_HONCHO_WORKSPACE", _DEFAULT_WORKSPACE),
            operator_peer=os.environ.get(
                "DGMH_HONCHO_OPERATOR_PEER", _DEFAULT_OPERATOR_PEER
            ),
            bot_peer=os.environ.get("DGMH_HONCHO_BOT_PEER", _DEFAULT_BOT_PEER),
        )


def _is_disabled() -> bool:
    return bool(os.environ.get("DGMH_HONCHO_DISABLED"))


@lru_cache(maxsize=1)
def get_client():
    """Return a cached Honcho SDK client bound to the configured workspace.

    Returns None on import / connection error so callers can short-circuit.
    """
    if _is_disabled():
        return None
    try:
        from honcho import Honcho  # type: ignore[import]
    except ImportError:
        logger.warning("honcho SDK not installed; memory wiring disabled")
        return None

    cfg = HonchoConfig.from_env()
    try:
        return Honcho(
            workspace_id=cfg.workspace_id,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
        )
    except Exception:
        logger.exception("honcho client init failed; memory wiring disabled")
        return None


def session_id_for_channel(channel_id: str, thread_id: Optional[str] = None) -> str:
    """Stable session key per Discord chat surface.

    Threads get their own session — they are independent conversational
    contexts. Channel-only messages share the channel session.
    """
    base = f"discord-{channel_id}"
    if thread_id:
        return f"{base}-thread-{thread_id}"
    return base


def add_user_message(
    *,
    channel_id: str,
    thread_id: Optional[str],
    content: str,
) -> bool:
    """Persist an inbound (operator) message to Honcho. Returns True on success."""
    if _is_disabled():
        return False
    client = get_client()
    if client is None:
        return False
    cfg = HonchoConfig.from_env()
    try:
        operator = client.peer(cfg.operator_peer)
        session = client.session(session_id_for_channel(channel_id, thread_id))
        session.add_messages([operator.message(content)])
        return True
    except Exception:
        logger.exception("honcho add_user_message failed")
        return False


def add_bot_message(
    *,
    channel_id: str,
    thread_id: Optional[str],
    content: str,
) -> bool:
    """Persist an outbound (bot) message to Honcho. Returns True on success."""
    if _is_disabled():
        return False
    client = get_client()
    if client is None:
        return False
    cfg = HonchoConfig.from_env()
    try:
        bot = client.peer(cfg.bot_peer)
        session = client.session(session_id_for_channel(channel_id, thread_id))
        session.add_messages([bot.message(content)])
        return True
    except Exception:
        logger.exception("honcho add_bot_message failed")
        return False


def get_operator_representation(channel_id: str, thread_id: Optional[str] = None) -> str:
    """Return a session-scoped natural-language snapshot of the operator.

    This is what the bot sees as durable peer-modeling: who the operator
    is, what their preferences are, what register they use in this
    surface. Empty string on failure or empty memory.
    """
    if _is_disabled():
        return ""
    client = get_client()
    if client is None:
        return ""
    cfg = HonchoConfig.from_env()
    try:
        session = client.session(session_id_for_channel(channel_id, thread_id))
        operator = client.peer(cfg.operator_peer)
        rep = session.representation(operator)
        if isinstance(rep, str):
            return rep
        if rep is None:
            return ""
        # Some SDK versions return a structured object; coerce to string.
        return str(rep)
    except Exception:
        logger.exception("honcho representation lookup failed")
        return ""


def chat_about_operator(query: str) -> str:
    """Ask Honcho a natural-language question about the operator. Empty on fail."""
    if _is_disabled():
        return ""
    client = get_client()
    if client is None:
        return ""
    cfg = HonchoConfig.from_env()
    try:
        operator = client.peer(cfg.operator_peer)
        response = operator.chat(query)
        if hasattr(response, "content"):
            return str(response.content)
        return str(response)
    except Exception:
        logger.exception("honcho chat lookup failed")
        return ""


def bootstrap() -> dict:
    """Idempotent bootstrap: ensure workspace, peers, and (no) sessions exist.

    Sessions are created on first add_messages, so we don't pre-create them.
    Run once at install time or via the gateway:startup hook.
    """
    if _is_disabled():
        return {"status": "disabled"}
    client = get_client()
    if client is None:
        return {"status": "no-client"}
    cfg = HonchoConfig.from_env()
    try:
        # peer() is implicit creation — calling it ensures the peer exists.
        operator = client.peer(cfg.operator_peer)
        bot = client.peer(cfg.bot_peer)
        return {
            "status": "ok",
            "workspace": cfg.workspace_id,
            "operator_peer": str(getattr(operator, "id", cfg.operator_peer)),
            "bot_peer": str(getattr(bot, "id", cfg.bot_peer)),
        }
    except Exception as exc:
        logger.exception("honcho bootstrap failed")
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
