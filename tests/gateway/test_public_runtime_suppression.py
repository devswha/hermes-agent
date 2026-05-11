from __future__ import annotations

import os
from unittest import mock

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def test_public_discord_runtime_messages_suppressed_by_default():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="fake")},
    )
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="1496735735078715542",
        chat_type="group",
    )

    with mock.patch.dict(
        os.environ,
        {"DGMH_PUBLIC_CHANNELS": "1496735735078715542"},
        clear=False,
    ):
        assert runner._should_suppress_public_runtime_message(source) is True


def test_public_discord_runtime_suppression_can_be_disabled():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="fake")},
    )
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="1496735735078715542",
        chat_type="group",
    )

    with mock.patch.dict(
        os.environ,
        {
            "DGMH_PUBLIC_CHANNELS": "1496735735078715542",
            "DGMH_PUBLIC_SUPPRESS_RUNTIME_STATUS": "0",
        },
        clear=False,
    ):
        assert runner._should_suppress_public_runtime_message(source) is False


def test_non_public_discord_runtime_messages_not_suppressed():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="fake")},
    )
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="not-public",
        chat_type="group",
    )

    with mock.patch.dict(
        os.environ,
        {"DGMH_PUBLIC_CHANNELS": "1496735735078715542"},
        clear=False,
    ):
        assert runner._should_suppress_public_runtime_message(source) is False

