from __future__ import annotations

import json
import os
from unittest import mock

from dgmh.self_evolution_status import (
    DEFAULT_NOTICE,
    clear_self_evolution_active,
    get_self_evolution_notice,
    load_self_evolution_status,
    mark_self_evolution_active,
    status_path,
)


def test_env_flag_enables_notice(tmp_path):
    with mock.patch.dict(
        os.environ,
        {
            "HERMES_HOME": str(tmp_path),
            "DGMH_SELF_EVOLVING": "1",
            "DGMH_SELF_EVOLUTION_NOTICE": "지금 자체진화 중이에요.",
        },
        clear=False,
    ):
        assert get_self_evolution_notice() == "지금 자체진화 중이에요."


def test_marker_file_enables_and_clears_notice(tmp_path):
    with mock.patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}, clear=False):
        mark_self_evolution_active(reason="test", source="unit", ttl_s=60)
        assert status_path().exists()
        assert get_self_evolution_notice() == DEFAULT_NOTICE

        clear_self_evolution_active()
        assert get_self_evolution_notice() is None


def test_expired_marker_is_inactive(tmp_path):
    with mock.patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}, clear=False):
        path = status_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"active": True, "expires_at_epoch": 10, "message": "old"}),
            encoding="utf-8",
        )

        assert load_self_evolution_status(now_epoch=11)["active"] is False
        assert get_self_evolution_notice(now_epoch=11) is None

