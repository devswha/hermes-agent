"""Tests for dgmh.humanness_log."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dgmh.humanness_log import (
    append_record,
    humanness_log_path,
    make_error_record,
    make_success_record,
    mean_humanness,
    read_records,
)


class TestRecordShape(unittest.TestCase):
    def test_success_record_has_score_fields(self) -> None:
        rec = make_success_record(
            chat_id="123",
            thread_id="456",
            message_id="789",
            soul_md_hash="abc",
            text_length=42,
            ai_score=10.0,
            human_likeness=90.0,
            sub_scores={"communication": 0.0},
            interpretation="human",
            elapsed_s=8.5,
        )
        self.assertEqual(rec["chat_id"], "123")
        self.assertEqual(rec["ai_score"], 10.0)
        self.assertEqual(rec["human_likeness"], 90.0)
        self.assertNotIn("error", rec)
        self.assertIn("ts", rec)

    def test_error_record_has_error_field(self) -> None:
        rec = make_error_record(
            chat_id="123",
            thread_id=None,
            message_id=None,
            soul_md_hash="abc",
            text_length=10,
            error="PatinaScoreError: timeout",
        )
        self.assertIn("error", rec)
        self.assertNotIn("ai_score", rec)
        self.assertEqual(rec["thread_id"], None)


class TestAppendAndRead(unittest.TestCase):
    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"HERMES_HOME": tmpdir}):
                rec = make_success_record(
                    chat_id="C1",
                    thread_id=None,
                    message_id="M1",
                    soul_md_hash="h1",
                    text_length=50,
                    ai_score=20.0,
                    human_likeness=80.0,
                    sub_scores={},
                    interpretation="mostly human",
                    elapsed_s=7.0,
                )
                append_record(rec)

                records = read_records()
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["chat_id"], "C1")
                self.assertEqual(records[0]["ai_score"], 20.0)

    def test_filter_by_soul_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"HERMES_HOME": tmpdir}):
                for hash_id, score in [("h1", 10.0), ("h2", 50.0), ("h1", 12.0)]:
                    append_record(
                        make_success_record(
                            chat_id="C",
                            thread_id=None,
                            message_id=None,
                            soul_md_hash=hash_id,
                            text_length=40,
                            ai_score=score,
                            human_likeness=100 - score,
                            sub_scores={},
                            interpretation="x",
                            elapsed_s=5.0,
                        )
                    )

                h1_records = read_records(soul_md_hash="h1")
                self.assertEqual(len(h1_records), 2)
                for rec in h1_records:
                    self.assertEqual(rec["soul_md_hash"], "h1")

                h2_records = read_records(soul_md_hash="h2")
                self.assertEqual(len(h2_records), 1)

    def test_limit_returns_most_recent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"HERMES_HOME": tmpdir}):
                for i in range(5):
                    append_record(
                        make_success_record(
                            chat_id=f"C{i}",
                            thread_id=None,
                            message_id=None,
                            soul_md_hash="h",
                            text_length=40,
                            ai_score=float(i),
                            human_likeness=100 - i,
                            sub_scores={},
                            interpretation="x",
                            elapsed_s=5.0,
                        )
                    )
                last_two = read_records(limit=2)
                self.assertEqual(len(last_two), 2)
                self.assertEqual(last_two[0]["chat_id"], "C3")
                self.assertEqual(last_two[1]["chat_id"], "C4")

    def test_skips_malformed_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"HERMES_HOME": tmpdir}):
                path = humanness_log_path()
                path.write_text(
                    '{"valid": 1, "ai_score": 10, "human_likeness": 90}\n'
                    "garbage line not json\n"
                    '{"valid": 2, "ai_score": 20, "human_likeness": 80}\n',
                    encoding="utf-8",
                )
                records = read_records()
                self.assertEqual(len(records), 2)
                self.assertEqual(records[0]["valid"], 1)
                self.assertEqual(records[1]["valid"], 2)

    def test_missing_file_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"HERMES_HOME": tmpdir}):
                self.assertEqual(read_records(), [])


class TestMeanHumanness(unittest.TestCase):
    def test_arithmetic_mean(self) -> None:
        records = [
            {"human_likeness": 80.0},
            {"human_likeness": 90.0},
            {"human_likeness": 100.0},
        ]
        self.assertAlmostEqual(mean_humanness(records), 90.0, places=2)

    def test_skips_error_records(self) -> None:
        records = [
            {"human_likeness": 80.0},
            {"error": "timeout"},
            {"human_likeness": 90.0},
        ]
        self.assertAlmostEqual(mean_humanness(records), 85.0, places=2)

    def test_empty_returns_none(self) -> None:
        self.assertIsNone(mean_humanness([]))

    def test_only_errors_returns_none(self) -> None:
        self.assertIsNone(mean_humanness([{"error": "x"}, {"error": "y"}]))


if __name__ == "__main__":
    unittest.main()
