"""GLM REST client for DGM-H GLM Data Collection v1.

Implements plan §4 Step 4 + §3 AC4, AC14:

- Thin wrapper over the OpenAI-compatible chat-completions endpoint
  ``https://open.bigmodel.cn/api/paas/v4/chat/completions`` (zhipu / GLM-4).
- Two transports: ``"network"`` (real HTTP via ``requests``) and ``"mock"``
  (deterministic replay of recorded fixtures).
- **Deny-by-default network**: the ``"network"`` transport refuses to issue
  HTTP unless ``DGMH_GLM_DATA_V1_ENABLE_NETWORK=1`` is set (AC14).
- 429 / 5xx / timeout retries with exponential backoff (1s, 2s, 4s).
- Optional response-body validation against a caller-supplied JSON schema
  using the ``jsonschema`` library (already a hermes-agent dep).

Reference: plan §4 Step 4 "GLM client" + Option Group A1 (REST via requests).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import jsonschema
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_MODEL = "glm-4.5"
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_MAX_RETRIES = 3
# Exponential backoff schedule (seconds). One slot per retry beyond the first
# attempt; we cap at ``DEFAULT_MAX_RETRIES`` retries total, so we need
# ``max_retries`` entries.
_BACKOFF_EXPONENTIAL = (1.0, 2.0, 4.0, 8.0, 16.0)

_ENABLE_NETWORK_ENV = "DGMH_GLM_DATA_V1_ENABLE_NETWORK"
_API_KEY_ENV = "GLM_API"
_BASE_URL_ENV = "GLM_API_BASE"

Transport = Literal["network", "mock"]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class GLMClientError(RuntimeError):
    """Base error for ``glm_client`` failures."""


class NetworkDeniedError(GLMClientError):
    """Raised when the ``"network"`` transport is invoked without the explicit
    ``DGMH_GLM_DATA_V1_ENABLE_NETWORK=1`` opt-in (plan §3 AC14)."""


class GLMResponseSchemaError(GLMClientError):
    """Raised when a response fails ``jsonschema`` validation after every
    permitted retry has been exhausted (plan §4 Step 4 acceptance)."""


class GLMTransportError(GLMClientError):
    """Raised when the transport returns an unrecoverable HTTP / network
    error after retries (4xx other than 429, persistent 5xx, etc.)."""


# ---------------------------------------------------------------------------
# Env loader (.env.local helper)
# ---------------------------------------------------------------------------


def _read_env_local_value(key: str, env_path: Optional[Path] = None) -> Optional[str]:
    """Read ``key`` from a ``KEY=VALUE``-style ``.env.local`` file.

    The hermes-agent repo follows the convention of stashing GLM credentials
    in a ``.env.local`` at the repo root. Plan §4 Step 4 specifies this path
    explicitly: *"Read api_key from .env.local GLM_API if not passed."* We
    implement a minimal parser (plain ``KEY=value`` lines, optional surrounding
    quotes, ``#`` comments) instead of pulling in ``python-dotenv``.

    ``env_path`` overrides the default location and is used by tests.
    Returns ``None`` if the file is missing or the key is absent.
    """

    if env_path is None:
        # ``dgmh/glm_data/glm_client.py`` → ``hermes-agent/`` repo root.
        env_path = Path(__file__).resolve().parents[2] / ".env.local"

    try:
        text = env_path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() != key:
            continue
        value = v.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        return value
    return None


# ---------------------------------------------------------------------------
# Mock fixture loader
# ---------------------------------------------------------------------------


def _fixture_key(messages: list[dict], model: str) -> str:
    """Build a deterministic lookup key from the request envelope."""

    payload = {
        "model": model,
        "messages": messages,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _load_fixtures(fixtures_dir: Path) -> dict[str, dict]:
    """Discover ``glm_*_real_stdout_*.json`` fixtures under ``fixtures_dir``.

    Each fixture file must be JSON of the form::

        {
          "request": {"model": "...", "messages": [...]},
          "response": {<full GLM response envelope>}
        }

    The returned mapping is keyed by ``_fixture_key(request.messages, request.model)``.
    Fixtures without a matching ``request`` block fall back to keying by file
    name so callers can still address them by stem.
    """

    fixtures: dict[str, dict] = {}
    if not fixtures_dir.is_dir():
        return fixtures
    for path in sorted(fixtures_dir.glob("glm_*_real_stdout_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("glm_client: skipping unreadable fixture %s: %s", path, exc)
            continue
        request = payload.get("request") or {}
        response = payload.get("response")
        if response is None:
            logger.warning("glm_client: fixture %s missing 'response' block", path)
            continue
        key = _fixture_key(
            request.get("messages", []) or [],
            request.get("model", DEFAULT_MODEL),
        )
        fixtures[key] = response
        # Secondary index: filename stem, so tests can synthesize fixtures
        # without matching the full request envelope.
        fixtures[path.stem] = response
    return fixtures


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class GLMClient:
    """REST-based GLM client with mock transport + retry + schema validation.

    Construct with ``transport="mock"`` for deterministic replay in tests or
    cost-projection scenarios. Construct with ``transport="network"`` for live
    calls — the client will refuse to talk to the network unless the operator
    sets ``DGMH_GLM_DATA_V1_ENABLE_NETWORK=1``.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        transport: Transport = "network",
        fixtures_dir: Optional[Path] = None,
        session: Optional[requests.Session] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if transport not in ("network", "mock"):
            raise ValueError(f"transport must be 'network' or 'mock'; got {transport!r}")

        self._transport: Transport = transport
        self._base_url = os.environ.get(_BASE_URL_ENV) or base_url
        self._session = session
        self._sleep = sleep
        self._fixtures_dir = fixtures_dir
        self._fixtures: dict[str, dict] = (
            _load_fixtures(fixtures_dir) if (transport == "mock" and fixtures_dir is not None) else {}
        )

        # API key resolution: explicit > env var > .env.local. We only require
        # a key for live network calls; mock transport never reads it.
        if api_key is None:
            api_key = os.environ.get(_API_KEY_ENV) or _read_env_local_value(_API_KEY_ENV)
        self._api_key = api_key

    # -- public API -----------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        *,
        model: str = DEFAULT_MODEL,
        response_format: Optional[dict] = None,
        json_schema: Optional[dict] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff: str = "exponential",
        thinking: Optional[dict] = None,
    ) -> dict:
        """Issue a GLM chat-completions call and return the parsed response.

        Parameters mirror plan §4 Step 4. ``json_schema`` is validated via the
        ``jsonschema`` library against the JSON body of the first choice's
        message content; on failure we retry up to ``max_retries`` times and
        then raise :class:`GLMResponseSchemaError`.
        """

        if backoff != "exponential":
            raise ValueError(f"only exponential backoff is supported in v1; got {backoff!r}")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        payload: dict[str, Any] = {"model": model, "messages": messages}
        if response_format is not None:
            payload["response_format"] = response_format
        elif json_schema is not None:
            # Force JSON object output whenever a schema is supplied — keeps the
            # validator happy and matches plan §4 Step 4 "json envelope wrap".
            payload["response_format"] = {"type": "json_object"}
        if thinking is not None:
            payload["thinking"] = thinking

        if self._transport == "mock":
            return self._chat_mock(messages, model=model, json_schema=json_schema)
        return self._chat_network(
            payload,
            timeout_s=timeout_s,
            max_retries=max_retries,
            json_schema=json_schema,
        )

    # -- transports -----------------------------------------------------

    def _chat_mock(
        self,
        messages: list[dict],
        *,
        model: str,
        json_schema: Optional[dict],
    ) -> dict:
        if not self._fixtures:
            raise GLMClientError(
                "mock transport selected but no fixtures were loaded — "
                "pass fixtures_dir pointing at dgmh/tests/fixtures/glm_data/"
            )
        key = _fixture_key(messages, model)
        response = self._fixtures.get(key)
        if response is None:
            raise GLMClientError(
                "mock transport: no recorded fixture matches this request "
                f"(model={model!r}, messages={messages!r})"
            )
        if json_schema is not None:
            self._validate_schema(response, json_schema)
        return response

    def _chat_network(
        self,
        payload: dict,
        *,
        timeout_s: float,
        max_retries: int,
        json_schema: Optional[dict],
    ) -> dict:
        if os.environ.get(_ENABLE_NETWORK_ENV) != "1":
            raise NetworkDeniedError(
                f"network transport refused: set {_ENABLE_NETWORK_ENV}=1 to opt in "
                "(plan §3 AC14, deny-by-default)"
            )
        if not self._api_key:
            raise GLMClientError(
                f"missing API key — set {_API_KEY_ENV} in env or .env.local "
                "before issuing a live GLM call"
            )

        session = self._session or requests.Session()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_error: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                resp = session.post(
                    self._base_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout_s,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                self._maybe_backoff(attempt, max_retries)
                continue

            status = getattr(resp, "status_code", 0)
            if status == 200:
                try:
                    body = resp.json()
                except ValueError as exc:
                    raise GLMTransportError(f"non-JSON 200 response: {exc}") from exc
                if json_schema is not None:
                    try:
                        self._validate_schema(body, json_schema)
                    except GLMResponseSchemaError:
                        last_error = GLMResponseSchemaError("schema validation failed")
                        self._maybe_backoff(attempt, max_retries)
                        continue
                return body

            if status == 429 or 500 <= status < 600:
                last_error = GLMTransportError(
                    f"retryable HTTP status {status}: {resp.text[:200] if hasattr(resp, 'text') else ''}"
                )
                self._maybe_backoff(attempt, max_retries)
                continue

            # Non-retryable client error.
            text = resp.text[:200] if hasattr(resp, "text") else ""
            raise GLMTransportError(f"non-retryable HTTP status {status}: {text}")

        if isinstance(last_error, GLMResponseSchemaError):
            raise last_error
        raise GLMTransportError(
            f"exhausted {max_retries} retries; last error: {last_error}"
        )

    # -- helpers --------------------------------------------------------

    def _maybe_backoff(self, attempt: int, max_retries: int) -> None:
        """Sleep before the next attempt (no-op after the final attempt)."""

        if attempt >= max_retries:
            return
        delay = (
            _BACKOFF_EXPONENTIAL[attempt]
            if attempt < len(_BACKOFF_EXPONENTIAL)
            else _BACKOFF_EXPONENTIAL[-1]
        )
        self._sleep(delay)

    @staticmethod
    def _validate_schema(body: dict, json_schema: dict) -> None:
        """Validate the response body against ``json_schema``.

        Callers are expected to shape ``json_schema`` against whatever level of
        the response envelope they care about (typically the whole body for
        v1). Raises :class:`GLMResponseSchemaError` on validation failure.
        """

        try:
            jsonschema.validate(body, json_schema)
        except jsonschema.ValidationError as exc:
            raise GLMResponseSchemaError(str(exc)) from exc


__all__ = [
    "GLMClient",
    "GLMClientError",
    "NetworkDeniedError",
    "GLMResponseSchemaError",
    "GLMTransportError",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
]
