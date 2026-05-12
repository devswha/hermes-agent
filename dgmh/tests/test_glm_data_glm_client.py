"""Tests for ``dgmh.glm_data.glm_client``.

Covers plan §3 AC4 (mockability + ≥3 recorded fixtures) and AC14 (deny-by-
default network) plus the Step 4 acceptance bullets:

- mock fixture replay is deterministic
- 429 retries with exponential backoff (1s, 2s, 4s)
- schema validation failure raises ``GLMResponseSchemaError`` after retries
- network transport refuses to run without the explicit env opt-in
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dgmh.glm_data.glm_client import (  # noqa: E402
    DEFAULT_MODEL,
    GLMClient,
    GLMResponseSchemaError,
    GLMTransportError,
    NetworkDeniedError,
)


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "glm_data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, status_code: int, body: dict | str = "", *, text: str = "") -> None:
        self.status_code = status_code
        self._body = body
        self.text = text or (json.dumps(body) if isinstance(body, dict) else str(body))

    def json(self) -> dict:
        if isinstance(self._body, dict):
            return self._body
        return json.loads(self._body or "{}")


class _ScriptedSession:
    """Replays a fixed list of responses in order; records each POST call."""

    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def post(self, url, *, headers, json, timeout):  # noqa: A002 - mirrors requests' API
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        if not self._responses:
            raise AssertionError("no more scripted responses available")
        return self._responses.pop(0)


@pytest.fixture()
def network_enabled(monkeypatch):
    monkeypatch.setenv("DGMH_GLM_DATA_V1_ENABLE_NETWORK", "1")
    monkeypatch.setenv("GLM_API", "test-api-key")
    yield


@pytest.fixture()
def sleep_log():
    """Records sleep durations so we can assert backoff schedule."""

    log: list[float] = []

    def _sleep(duration: float) -> None:
        log.append(duration)

    return log, _sleep


# ---------------------------------------------------------------------------
# AC4: mockability + deterministic replay
# ---------------------------------------------------------------------------


def test_mock_fixture_replay_deterministic():
    """Three recorded fixtures replay deterministically via mock transport."""

    client = GLMClient(transport="mock", fixtures_dir=FIXTURES_DIR)

    # Fixture 1: classify casual reaction
    resp1 = client.chat(
        [
            {"role": "system", "content": "Classify the message into one ko-* category."},
            {"role": "user", "content": "ㅋㅋㅋ 이거 진짜 웃기다"},
        ]
    )
    assert resp1["id"] == "chatcmpl-test-001"
    parsed = json.loads(resp1["choices"][0]["message"]["content"])
    assert parsed["categories"] == ["ko-communication"]

    # Fixture 2: classify structured Korean
    resp2 = client.chat(
        [
            {"role": "system", "content": "Classify the message into one ko-* category."},
            {"role": "user", "content": "내일 오전에 시간 괜찮으신가요?"},
        ]
    )
    assert resp2["id"] == "chatcmpl-test-002"
    assert json.loads(resp2["choices"][0]["message"]["content"])["categories"] == [
        "ko-structure",
        "ko-language",
    ]

    # Fixture 3: inventory aggregation
    resp3 = client.chat(
        [
            {"role": "system", "content": "Aggregate per-category counts from the messages."},
            {
                "role": "user",
                "content": '[{"msg_id":"m001","categories":["ko-communication"]},{"msg_id":"m002","categories":["ko-structure"]}]',
            },
        ]
    )
    assert resp3["id"] == "chatcmpl-test-003"
    counts = json.loads(resp3["choices"][0]["message"]["content"])["counts"]
    assert counts == {"ko-communication": 1, "ko-structure": 1}

    # Determinism: second pass produces identical envelopes.
    resp1b = client.chat(
        [
            {"role": "system", "content": "Classify the message into one ko-* category."},
            {"role": "user", "content": "ㅋㅋㅋ 이거 진짜 웃기다"},
        ]
    )
    assert resp1b == resp1


def test_mock_transport_requires_fixtures_dir():
    """Mock transport without fixtures raises a clear error."""

    from dgmh.glm_data.glm_client import GLMClientError

    client = GLMClient(transport="mock")
    with pytest.raises(GLMClientError, match="no fixtures were loaded"):
        client.chat([{"role": "user", "content": "anything"}])


def test_mock_unknown_request_raises():
    """Mock transport with a non-matching request fails loudly (no silent pass)."""

    from dgmh.glm_data.glm_client import GLMClientError

    client = GLMClient(transport="mock", fixtures_dir=FIXTURES_DIR)
    with pytest.raises(GLMClientError, match="no recorded fixture matches"):
        client.chat([{"role": "user", "content": "this prompt is not in any fixture"}])


# ---------------------------------------------------------------------------
# Retry + backoff
# ---------------------------------------------------------------------------


def test_429_retry_with_backoff(network_enabled, sleep_log):
    """Mock transport returns 429 twice then 200; client makes 3 attempts and sleeps 1s, 2s."""

    log, sleep = sleep_log
    success_body = {
        "id": "chatcmpl-retry-ok",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "{\"ok\": true}"}}],
    }
    session = _ScriptedSession(
        [
            _FakeResponse(429, text="rate limited"),
            _FakeResponse(429, text="rate limited"),
            _FakeResponse(200, success_body),
        ]
    )

    client = GLMClient(transport="network", session=session, sleep=sleep)
    result = client.chat(
        [{"role": "user", "content": "hi"}],
        max_retries=3,
        timeout_s=5.0,
    )
    assert result == success_body
    assert len(session.calls) == 3
    # Two backoff sleeps before the third attempt; no sleep after success.
    assert log == [1.0, 2.0]


def test_5xx_retry_then_failure(network_enabled, sleep_log):
    """Persistent 500 errors exhaust retries and raise GLMTransportError."""

    log, sleep = sleep_log
    session = _ScriptedSession(
        [
            _FakeResponse(500, text="boom"),
            _FakeResponse(500, text="boom"),
            _FakeResponse(500, text="boom"),
            _FakeResponse(500, text="boom"),
        ]
    )

    client = GLMClient(transport="network", session=session, sleep=sleep)
    with pytest.raises(GLMTransportError, match="exhausted"):
        client.chat(
            [{"role": "user", "content": "hi"}],
            max_retries=3,
            timeout_s=5.0,
        )
    assert len(session.calls) == 4  # initial attempt + 3 retries
    assert log == [1.0, 2.0, 4.0]


def test_non_retryable_4xx_fails_immediately(network_enabled, sleep_log):
    """A 400 response is not retried."""

    log, sleep = sleep_log
    session = _ScriptedSession([_FakeResponse(400, text="bad request")])
    client = GLMClient(transport="network", session=session, sleep=sleep)
    with pytest.raises(GLMTransportError, match="non-retryable"):
        client.chat([{"role": "user", "content": "hi"}], max_retries=3)
    assert len(session.calls) == 1
    assert log == []


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_schema_failure_raises_after_retries(network_enabled, sleep_log):
    """When every response fails schema validation, raise ``GLMResponseSchemaError``."""

    log, sleep = sleep_log
    # Schema demands a top-level "ok" field set to true; response omits it.
    schema = {
        "type": "object",
        "properties": {"ok": {"const": True}},
        "required": ["ok"],
    }
    bad_body = {"id": "no-ok-field", "choices": []}
    session = _ScriptedSession(
        [
            _FakeResponse(200, bad_body),
            _FakeResponse(200, bad_body),
            _FakeResponse(200, bad_body),
            _FakeResponse(200, bad_body),
        ]
    )
    client = GLMClient(transport="network", session=session, sleep=sleep)
    with pytest.raises(GLMResponseSchemaError):
        client.chat(
            [{"role": "user", "content": "hi"}],
            json_schema=schema,
            max_retries=3,
        )
    # 1 initial + 3 retries = 4 attempts, 3 backoff sleeps.
    assert len(session.calls) == 4
    assert log == [1.0, 2.0, 4.0]


def test_schema_validation_passes(network_enabled, sleep_log):
    """A valid response is returned without retry."""

    log, sleep = sleep_log
    schema = {
        "type": "object",
        "properties": {"ok": {"const": True}},
        "required": ["ok"],
    }
    good_body = {"ok": True, "id": "fine"}
    session = _ScriptedSession([_FakeResponse(200, good_body)])
    client = GLMClient(transport="network", session=session, sleep=sleep)
    result = client.chat(
        [{"role": "user", "content": "hi"}],
        json_schema=schema,
    )
    assert result == good_body
    assert len(session.calls) == 1
    assert log == []


# ---------------------------------------------------------------------------
# AC14: deny-by-default network
# ---------------------------------------------------------------------------


def test_network_deny_default_raises_NetworkDeniedError(monkeypatch):
    """Without DGMH_GLM_DATA_V1_ENABLE_NETWORK=1, network transport refuses."""

    monkeypatch.delenv("DGMH_GLM_DATA_V1_ENABLE_NETWORK", raising=False)

    class _ExplodingSession:
        def post(self, *args, **kwargs):
            raise AssertionError("session.post must not be called when network is denied")

    client = GLMClient(
        transport="network",
        api_key="placeholder-should-not-be-used",
        session=_ExplodingSession(),
    )
    with pytest.raises(NetworkDeniedError, match="DGMH_GLM_DATA_V1_ENABLE_NETWORK"):
        client.chat([{"role": "user", "content": "hi"}])


def test_network_denied_when_flag_not_one(monkeypatch):
    """``DGMH_GLM_DATA_V1_ENABLE_NETWORK=true`` (non-"1") still denies."""

    monkeypatch.setenv("DGMH_GLM_DATA_V1_ENABLE_NETWORK", "true")
    client = GLMClient(transport="network", api_key="k")
    with pytest.raises(NetworkDeniedError):
        client.chat([{"role": "user", "content": "hi"}])


def test_missing_api_key_raises(monkeypatch):
    """Network call with the opt-in but no API key fails cleanly."""

    from dgmh.glm_data.glm_client import GLMClientError

    monkeypatch.setenv("DGMH_GLM_DATA_V1_ENABLE_NETWORK", "1")
    monkeypatch.delenv("GLM_API", raising=False)
    # Point .env.local at a non-existent path so we don't accidentally pick up
    # an operator-side key during local test runs.
    client = GLMClient(transport="network", api_key=None)
    # Force the resolved key to None even if .env.local exists.
    client._api_key = None  # type: ignore[attr-defined]
    with pytest.raises(GLMClientError, match="missing API key"):
        client.chat([{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# Sanity: default model + base URL
# ---------------------------------------------------------------------------


def test_default_model_and_payload_shape(network_enabled, sleep_log):
    """Request payload sets model + messages and optional response_format when schema present."""

    log, sleep = sleep_log
    session = _ScriptedSession([_FakeResponse(200, {"ok": True})])
    client = GLMClient(transport="network", session=session, sleep=sleep)
    schema = {"type": "object", "required": ["ok"], "properties": {"ok": {"const": True}}}
    client.chat(
        [{"role": "user", "content": "ping"}],
        json_schema=schema,
    )
    call = session.calls[0]
    assert call["json"]["model"] == DEFAULT_MODEL
    assert call["json"]["messages"] == [{"role": "user", "content": "ping"}]
    assert call["json"]["response_format"] == {"type": "json_object"}
    assert call["headers"]["Authorization"] == "Bearer test-api-key"
