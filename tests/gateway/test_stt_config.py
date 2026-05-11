"""Gateway STT config tests — honor stt.enabled: false from config.yaml."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from gateway.config import GatewayConfig, Platform, PlatformConfig, load_gateway_config
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def test_gateway_config_stt_disabled_from_dict_nested():
    config = GatewayConfig.from_dict({"stt": {"enabled": False}})
    assert config.stt_enabled is False


def test_load_gateway_config_bridges_stt_enabled_from_config_yaml(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.dump({"stt": {"enabled": False}}),
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    config = load_gateway_config()

    assert config.stt_enabled is False


@pytest.mark.asyncio
async def test_enrich_message_with_transcription_skips_when_stt_disabled():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=False)

    with patch(
        "tools.transcription_tools.transcribe_audio",
        side_effect=AssertionError("transcribe_audio should not be called when STT is disabled"),
    ):
        result = await runner._enrich_message_with_transcription(
            "caption",
            ["/tmp/voice.ogg"],
        )

    assert "transcription is disabled" in result.lower()
    assert "caption" in result


@pytest.mark.asyncio
async def test_enrich_message_with_transcription_avoids_bogus_no_provider_message_for_backend_key_errors():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": False, "error": "VOICE_TOOLS_OPENAI_KEY not set"},
    ):
        result = await runner._enrich_message_with_transcription(
            "caption",
            ["/tmp/voice.ogg"],
        )

    assert "No STT provider is configured" not in result
    assert "trouble transcribing" in result
    assert "caption" in result


@pytest.mark.asyncio
async def test_prepare_inbound_message_text_transcribes_queued_voice_event():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner.adapters = {}
    runner._model = "test-model"
    runner._base_url = ""
    runner._has_setup_skill = lambda: False

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
    )
    event = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=source,
        media_urls=["/tmp/queued-voice.ogg"],
        media_types=["audio/ogg"],
    )

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={
            "success": True,
            "transcript": "queued voice transcript",
            "provider": "local_command",
        },
    ):
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    assert result is not None
    assert "queued voice transcript" in result
    assert "voice message" in result.lower()


@pytest.mark.asyncio
async def test_prepare_inbound_mixed_media_uses_per_attachment_mime_type():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    runner._decide_image_input_mode = lambda: "native"
    runner._has_setup_skill = lambda: False

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="1496735735078715542",
        chat_type="group",
        user_id="1487083003946729542",
        user_name="쿠마",
        is_bot=True,
    )
    event = MessageEvent(
        text="[image: neutral_smile] [voice: un] <@1496771171151253636>",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["/tmp/neutral_smile.png", "/tmp/un.wav"],
        media_types=["image/png", "audio/x-wav"],
    )

    with patch.dict("os.environ", {"DGMH_PUBLIC_BOT_TEXT_ONLY": "0"}, clear=False):
        with patch(
            "tools.transcription_tools.transcribe_audio",
            return_value={
                "success": True,
                "transcript": "voice transcript",
                "provider": "local_command",
            },
        ):
            result = await runner._prepare_inbound_message_text(
                event=event,
                source=source,
                history=[],
            )

    assert result is not None
    assert "voice transcript" in result
    pending = runner._consume_pending_native_image_paths(
        runner._session_key_for_source(source)
    )
    assert pending == ["/tmp/neutral_smile.png"]


@pytest.mark.asyncio
async def test_public_discord_bot_peer_media_is_text_only_by_default():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        stt_enabled=True,
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="fake")},
    )
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    runner._decide_image_input_mode = lambda: (_ for _ in ()).throw(
        AssertionError("public bot media should not reach image routing")
    )
    runner._has_setup_skill = lambda: False

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="1496735735078715542",
        chat_type="group",
        user_id="1487083003946729542",
        user_name="쿠마",
        is_bot=True,
    )
    event = MessageEvent(
        text="[image: neutral_smile] [voice: un] <@1496771171151253636>",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["/tmp/neutral_smile.png", "/tmp/un.wav"],
        media_types=["image/png", "audio/x-wav"],
    )

    with patch.dict(
        "os.environ",
        {"DGMH_PUBLIC_CHANNELS": "1496735735078715542"},
        clear=False,
    ):
        with patch(
            "tools.transcription_tools.transcribe_audio",
            side_effect=AssertionError("public bot audio should not be transcribed"),
        ):
            result = await runner._prepare_inbound_message_text(
                event=event,
                source=source,
                history=[],
            )

    assert result is not None
    assert "[쿠마]" in result
    assert runner._consume_pending_native_image_paths(
        runner._session_key_for_source(source)
    ) == []
