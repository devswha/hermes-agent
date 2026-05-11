"""Tests for dgmh.hermes_integration.ai_callout_hook."""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from dgmh.hermes_integration import ai_callout_hook as hook


class TestEnvResolution(unittest.TestCase):
    def setUp(self) -> None:
        # Clear relevant env vars before each test to avoid bleed.
        for k in (
            "DGMH_AI_CALLOUT_MODE",
            "DGMH_AI_CALLOUT_CHANNELS",
            "DGMH_AI_CALLOUT_THRESHOLD",
            "DGMH_AI_CALLOUT_COOLDOWN_S",
            "DGMH_AI_CALLOUT_OPERATOR_QUIET_S",
        ):
            os.environ.pop(k, None)

    def test_mode_default_disabled(self) -> None:
        self.assertEqual(hook.get_mode(), "disabled")

    def test_mode_known_values(self) -> None:
        for v in ("disabled", "dryrun", "fixed", "agent"):
            with mock.patch.dict(os.environ, {"DGMH_AI_CALLOUT_MODE": v}):
                self.assertEqual(hook.get_mode(), v)

    def test_mode_unknown_clamped_to_disabled(self) -> None:
        with mock.patch.dict(os.environ, {"DGMH_AI_CALLOUT_MODE": "boom"}):
            self.assertEqual(hook.get_mode(), "disabled")

    def test_callout_channels_parsing(self) -> None:
        with mock.patch.dict(
            os.environ, {"DGMH_AI_CALLOUT_CHANNELS": "111, 222 ,, 333"}
        ):
            self.assertEqual(hook.get_callout_channels(), {"111", "222", "333"})

    def test_threshold_default_and_override(self) -> None:
        self.assertAlmostEqual(hook.get_threshold(), 70.0)
        with mock.patch.dict(os.environ, {"DGMH_AI_CALLOUT_THRESHOLD": "55.5"}):
            self.assertAlmostEqual(hook.get_threshold(), 55.5)

    def test_threshold_invalid_falls_back(self) -> None:
        with mock.patch.dict(os.environ, {"DGMH_AI_CALLOUT_THRESHOLD": "garbage"}):
            self.assertAlmostEqual(hook.get_threshold(), 70.0)


class TestShouldObserve(unittest.TestCase):
    def test_skips_non_bot(self) -> None:
        ok, reason = hook.should_observe(
            is_bot=False,
            is_self=False,
            channel_id="111",
            text="x" * 100,
            callout_channels={"111"},
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "not_bot")

    def test_skips_self(self) -> None:
        ok, reason = hook.should_observe(
            is_bot=True,
            is_self=True,
            channel_id="111",
            text="x" * 100,
            callout_channels={"111"},
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "self")

    def test_skips_channel_not_in_set(self) -> None:
        ok, reason = hook.should_observe(
            is_bot=True,
            is_self=False,
            channel_id="999",
            text="x" * 100,
            callout_channels={"111"},
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "channel_not_in_callout_set")

    def test_skips_too_short(self) -> None:
        ok, reason = hook.should_observe(
            is_bot=True,
            is_self=False,
            channel_id="111",
            text="hi",
            callout_channels={"111"},
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "too_short")

    def test_passes_valid_candidate(self) -> None:
        ok, _ = hook.should_observe(
            is_bot=True,
            is_self=False,
            channel_id="111",
            text="x" * 100,
            callout_channels={"111"},
        )
        self.assertTrue(ok)


class TestCooldownAndQuietWindow(unittest.TestCase):
    def setUp(self) -> None:
        hook._reset_state_for_tests()

    def test_cooldown_inactive_initially(self) -> None:
        self.assertFalse(hook._is_cooldown_active("c1", 60.0))

    def test_mark_callout_starts_cooldown(self) -> None:
        hook._mark_callout("c1")
        self.assertTrue(hook._is_cooldown_active("c1", 60.0))

    def test_cooldown_expires(self) -> None:
        with mock.patch.object(time, "monotonic", side_effect=[100.0, 200.0]):
            hook._mark_callout("c1")
            self.assertFalse(hook._is_cooldown_active("c1", 60.0))

    def test_operator_quiet_window(self) -> None:
        hook.record_human_activity("c1")
        self.assertTrue(hook._is_operator_active("c1", 60.0))

    def test_operator_quiet_expires(self) -> None:
        with mock.patch.object(time, "monotonic", side_effect=[100.0, 200.0]):
            hook.record_human_activity("c1")
            self.assertFalse(hook._is_operator_active("c1", 60.0))


class TestStructuralPreCheck(unittest.TestCase):
    def test_three_bullets_flagged(self) -> None:
        text = "- a\n- b\n- c\n"
        self.assertTrue(hook._has_ai_structural_tells(text))

    def test_numbered_list_flagged(self) -> None:
        text = "1. one\n2. two\n3. three\n"
        self.assertTrue(hook._has_ai_structural_tells(text))

    def test_bold_label_flagged(self) -> None:
        text = "**핵심:** something here"
        self.assertTrue(hook._has_ai_structural_tells(text))

    def test_human_chat_not_flagged(self) -> None:
        text = "응 그거 맞아 ㅋㅋ 그렇게 가자"
        self.assertFalse(hook._has_ai_structural_tells(text))


class TestPhrasebook(unittest.TestCase):
    def test_load_default_phrasebook_returns_lines(self) -> None:
        # The repo ships a default phrasebook; this should be non-empty.
        lines = hook._load_phrasebook()
        self.assertGreater(len(lines), 0)
        for ln in lines:
            self.assertFalse(ln.startswith("#"))
            self.assertTrue(ln.strip())

    def test_pick_callout_line_returns_member_of_phrasebook(self) -> None:
        rng = random.Random(42)
        line = hook.pick_callout_line(rng=rng)
        self.assertIsNotNone(line)
        self.assertIn(line, hook._load_phrasebook())

    def test_phrasebook_env_override_to_missing_file(self) -> None:
        with TemporaryDirectory() as d:
            missing = Path(d) / "nope.txt"
            with mock.patch.dict(
                os.environ, {"DGMH_AI_CALLOUT_PHRASEBOOK": str(missing)}
            ):
                self.assertEqual(hook._load_phrasebook(), [])
                self.assertIsNone(hook.pick_callout_line())

    def test_phrasebook_env_override_to_custom_file(self) -> None:
        with TemporaryDirectory() as d:
            f = Path(d) / "p.txt"
            f.write_text("# header\nONE\nTWO\n\n# skip\nTHREE\n", encoding="utf-8")
            with mock.patch.dict(
                os.environ, {"DGMH_AI_CALLOUT_PHRASEBOOK": str(f)}
            ):
                self.assertEqual(
                    sorted(hook._load_phrasebook()), ["ONE", "THREE", "TWO"]
                )


class TestLogRecord(unittest.TestCase):
    def test_record_excludes_bot_text(self) -> None:
        r = hook.make_log_record(
            channel_id="111",
            msg_id="222",
            author_bot_id="333",
            text_length=512,
            ai_score=82.5,
            mode="dryrun",
            action="would_callout",
        )
        # Privacy: third-party text NEVER lands in the log row.
        for key in r:
            self.assertNotIn("text", key.replace("text_length", ""))
        self.assertEqual(r["text_length"], 512)
        self.assertEqual(r["ai_score"], 82.5)
        self.assertEqual(r["mode"], "dryrun")
        self.assertEqual(r["action"], "would_callout")
        self.assertIn("ts", r)

    def test_append_log_writes_jsonl(self) -> None:
        with TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"HERMES_HOME": d}):
                hook._append_log({"ts": "now", "channel_id": "111"})
                hook._append_log({"ts": "next", "channel_id": "222"})
            log = Path(d) / "dgmh" / "ai_callout_log.jsonl"
            lines = log.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["channel_id"], "111")
            self.assertEqual(json.loads(lines[1])["channel_id"], "222")


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


class TestObserveDispatch(unittest.TestCase):
    def setUp(self) -> None:
        hook._reset_state_for_tests()

    def _isolate_log(self):
        d = TemporaryDirectory()
        self.addCleanup(d.cleanup)
        return mock.patch.dict(os.environ, {"HERMES_HOME": d.name})

    def test_disabled_mode_short_circuits(self) -> None:
        with self._isolate_log(), mock.patch.dict(
            os.environ, {"DGMH_AI_CALLOUT_MODE": "disabled"}
        ):
            with mock.patch.object(hook, "_score_in_thread") as scorer:
                _run(
                    hook.observe_inbound_bot(
                        adapter=mock.AsyncMock(),
                        channel_id="111",
                        msg_id="m1",
                        author_bot_id="b1",
                        text="x" * 100,
                    )
                )
                scorer.assert_not_called()

    def test_operator_active_skips(self) -> None:
        d = TemporaryDirectory()
        self.addCleanup(d.cleanup)
        hook.record_human_activity("111")
        with mock.patch.dict(
            os.environ,
            {
                "DGMH_AI_CALLOUT_MODE": "dryrun",
                "HERMES_HOME": d.name,
                "DGMH_AI_CALLOUT_OPERATOR_QUIET_S": "60",
            },
        ):
            with mock.patch.object(hook, "_score_in_thread") as scorer:
                _run(
                    hook.observe_inbound_bot(
                        adapter=mock.AsyncMock(),
                        channel_id="111",
                        msg_id="m1",
                        author_bot_id="b1",
                        text="x" * 100,
                    )
                )
                scorer.assert_not_called()
            log = Path(d.name) / "dgmh" / "ai_callout_log.jsonl"
            row = json.loads(log.read_text().strip().splitlines()[-1])
            self.assertEqual(row["action"], "skipped")
            self.assertEqual(row["reason"], "operator_active")

    def test_cooldown_skips(self) -> None:
        d = TemporaryDirectory()
        self.addCleanup(d.cleanup)
        hook._mark_callout("111")
        with mock.patch.dict(
            os.environ,
            {
                "DGMH_AI_CALLOUT_MODE": "dryrun",
                "HERMES_HOME": d.name,
                "DGMH_AI_CALLOUT_COOLDOWN_S": "600",
            },
        ):
            with mock.patch.object(hook, "_score_in_thread") as scorer:
                _run(
                    hook.observe_inbound_bot(
                        adapter=mock.AsyncMock(),
                        channel_id="111",
                        msg_id="m1",
                        author_bot_id="b1",
                        text="x" * 100,
                    )
                )
                scorer.assert_not_called()
            row = json.loads(
                (Path(d.name) / "dgmh" / "ai_callout_log.jsonl")
                .read_text()
                .strip()
                .splitlines()[-1]
            )
            self.assertEqual(row["reason"], "cooldown")

    def test_dryrun_below_threshold_logs(self) -> None:
        d = TemporaryDirectory()
        self.addCleanup(d.cleanup)
        with mock.patch.dict(
            os.environ,
            {
                "DGMH_AI_CALLOUT_MODE": "dryrun",
                "HERMES_HOME": d.name,
                "DGMH_AI_CALLOUT_THRESHOLD": "70",
            },
        ):
            with mock.patch.object(hook, "_score_in_thread", return_value=42.0):
                _run(
                    hook.observe_inbound_bot(
                        adapter=mock.AsyncMock(),
                        channel_id="111",
                        msg_id="m1",
                        author_bot_id="b1",
                        text="x" * 100,
                    )
                )
            row = json.loads(
                (Path(d.name) / "dgmh" / "ai_callout_log.jsonl")
                .read_text()
                .strip()
                .splitlines()[-1]
            )
            self.assertEqual(row["action"], "below_threshold")
            self.assertEqual(row["ai_score"], 42.0)
            # Below threshold must NOT consume the cooldown.
            self.assertFalse(hook._is_cooldown_active("111", 600.0))

    def test_dryrun_above_threshold_marks_cooldown(self) -> None:
        d = TemporaryDirectory()
        self.addCleanup(d.cleanup)
        with mock.patch.dict(
            os.environ,
            {
                "DGMH_AI_CALLOUT_MODE": "dryrun",
                "HERMES_HOME": d.name,
                "DGMH_AI_CALLOUT_THRESHOLD": "70",
            },
        ):
            adapter = mock.AsyncMock()
            with mock.patch.object(hook, "_score_in_thread", return_value=85.0):
                _run(
                    hook.observe_inbound_bot(
                        adapter=adapter,
                        channel_id="111",
                        msg_id="m1",
                        author_bot_id="b1",
                        text="x" * 100,
                    )
                )
            row = json.loads(
                (Path(d.name) / "dgmh" / "ai_callout_log.jsonl")
                .read_text()
                .strip()
                .splitlines()[-1]
            )
            self.assertEqual(row["action"], "would_callout")
            adapter.send.assert_not_called()
            self.assertTrue(hook._is_cooldown_active("111", 600.0))

    def test_fixed_above_threshold_sends_and_logs(self) -> None:
        d = TemporaryDirectory()
        self.addCleanup(d.cleanup)
        adapter = mock.AsyncMock()
        with mock.patch.dict(
            os.environ,
            {
                "DGMH_AI_CALLOUT_MODE": "fixed",
                "HERMES_HOME": d.name,
                "DGMH_AI_CALLOUT_THRESHOLD": "70",
            },
        ):
            with mock.patch.object(hook, "_score_in_thread", return_value=85.0), \
                mock.patch.object(hook, "pick_callout_line", return_value="ㅋㅋ AI 같잖아"):
                _run(
                    hook.observe_inbound_bot(
                        adapter=adapter,
                        channel_id="111",
                        msg_id="m1",
                        author_bot_id="b1",
                        text="x" * 100,
                    )
                )
        adapter.send.assert_awaited_once_with("111", "ㅋㅋ AI 같잖아")
        row = json.loads(
            (Path(d.name) / "dgmh" / "ai_callout_log.jsonl")
            .read_text()
            .strip()
            .splitlines()[-1]
        )
        self.assertEqual(row["action"], "sent")
        self.assertTrue(hook._is_cooldown_active("111", 600.0))

    def test_fixed_empty_phrasebook_does_not_send(self) -> None:
        d = TemporaryDirectory()
        self.addCleanup(d.cleanup)
        adapter = mock.AsyncMock()
        with mock.patch.dict(
            os.environ,
            {"DGMH_AI_CALLOUT_MODE": "fixed", "HERMES_HOME": d.name},
        ):
            with mock.patch.object(hook, "_score_in_thread", return_value=99.0), \
                mock.patch.object(hook, "pick_callout_line", return_value=None):
                _run(
                    hook.observe_inbound_bot(
                        adapter=adapter,
                        channel_id="111",
                        msg_id="m1",
                        author_bot_id="b1",
                        text="x" * 100,
                    )
                )
        adapter.send.assert_not_called()
        row = json.loads(
            (Path(d.name) / "dgmh" / "ai_callout_log.jsonl")
            .read_text()
            .strip()
            .splitlines()[-1]
        )
        self.assertEqual(row["action"], "send_failed")
        # Phrasebook empty → no cooldown burned.
        self.assertFalse(hook._is_cooldown_active("111", 600.0))

    def test_agent_falls_back_to_fixed(self) -> None:
        d = TemporaryDirectory()
        self.addCleanup(d.cleanup)
        adapter = mock.AsyncMock()
        with mock.patch.dict(
            os.environ,
            {"DGMH_AI_CALLOUT_MODE": "agent", "HERMES_HOME": d.name},
        ):
            with mock.patch.object(hook, "_score_in_thread", return_value=85.0), \
                mock.patch.object(hook, "pick_callout_line", return_value="LINE"):
                _run(
                    hook.observe_inbound_bot(
                        adapter=adapter,
                        channel_id="111",
                        msg_id="m1",
                        author_bot_id="b1",
                        text="x" * 100,
                    )
                )
        adapter.send.assert_awaited_once_with("111", "LINE")
        row = json.loads(
            (Path(d.name) / "dgmh" / "ai_callout_log.jsonl")
            .read_text()
            .strip()
            .splitlines()[-1]
        )
        self.assertEqual(row["action"], "agent_fallback_to_fixed")


class TestSelfReferencedSkip(unittest.TestCase):
    def test_no_reference_returns_false(self) -> None:
        msg = mock.Mock(reference=None)
        self.assertFalse(hook._is_self_referenced(msg, self_user_id=999))

    def test_reference_to_self_returns_true(self) -> None:
        resolved = mock.Mock(author=mock.Mock(id=999))
        msg = mock.Mock(reference=mock.Mock(resolved=resolved))
        self.assertTrue(hook._is_self_referenced(msg, self_user_id=999))

    def test_reference_to_other_returns_false(self) -> None:
        resolved = mock.Mock(author=mock.Mock(id=111))
        msg = mock.Mock(reference=mock.Mock(resolved=resolved))
        self.assertFalse(hook._is_self_referenced(msg, self_user_id=999))


if __name__ == "__main__":
    unittest.main()
