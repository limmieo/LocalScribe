import json
import tempfile
import unittest
from pathlib import Path

import transcribe_core


class ExportTests(unittest.TestCase):
    def test_timestamp(self):
        self.assertEqual(transcribe_core.timestamp(3661.234), "01:01:01.234")

    def test_timestamped_text_and_highlights(self):
        transcript = {
            "segments": [
                {
                    "start": 0,
                    "end": 20,
                    "text": "The biggest mistake actually changed how we built the company.",
                },
                {
                    "start": 20,
                    "end": 42,
                    "text": "Why did that happen? We learned three important lessons.",
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.json"
            timestamped = root / "sample.timestamps.txt"
            highlights = root / "sample.highlights.md"
            source.write_text(json.dumps(transcript), encoding="utf-8")

            segments = transcribe_core.load_segments(source)
            transcribe_core.write_timestamped_text(segments, timestamped)
            transcribe_core.write_highlights(segments, highlights, "sample.mp4")

            self.assertIn("[00:00:00.000 - 00:00:20.000]", timestamped.read_text())
            self.assertIn("Highlight candidates", highlights.read_text())
            self.assertIn("biggest mistake", highlights.read_text())

    def test_timestamped_text_without_speakers_unchanged(self):
        segments = [{"start": 0, "end": 5, "text": "Hello there."}]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "sample.timestamps.txt"
            transcribe_core.write_timestamped_text(segments, destination)
            self.assertEqual(
                destination.read_text(),
                "[00:00:00.000 - 00:00:05.000] Hello there.\n",
            )

    def test_timestamped_text_with_speakers(self):
        segments = [
            {"start": 0, "end": 5, "text": "Hello there.", "speaker": "Speaker 1"},
            {"start": 5, "end": 10, "text": "Hi back.", "speaker": "Speaker 2"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "sample.timestamps.txt"
            transcribe_core.write_timestamped_text(segments, destination)
            content = destination.read_text()
            self.assertIn("Speaker 1: Hello there.", content)
            self.assertIn("Speaker 2: Hi back.", content)


class AssignSpeakersTests(unittest.TestCase):
    def test_renumbers_by_first_appearance(self):
        segments = [
            {"start": 0, "end": 5, "text": "a"},
            {"start": 5, "end": 10, "text": "b"},
            {"start": 10, "end": 15, "text": "c"},
        ]
        turns = [
            (0, 5, "SPEAKER_01"),
            (5, 10, "SPEAKER_00"),
            (10, 15, "SPEAKER_01"),
        ]
        result = transcribe_core.assign_speakers(segments, turns)
        self.assertEqual(result[0]["speaker"], "Speaker 1")
        self.assertEqual(result[1]["speaker"], "Speaker 2")
        self.assertEqual(result[2]["speaker"], "Speaker 1")

    def test_overlap_picks_dominant_turn(self):
        segments = [{"start": 0, "end": 10, "text": "a"}]
        turns = [(0, 3, "SPEAKER_00"), (3, 10, "SPEAKER_01")]
        result = transcribe_core.assign_speakers(segments, turns)
        self.assertEqual(result[0]["speaker"], "Speaker 2")

    def test_no_overlap_falls_back_to_nearest_turn(self):
        segments = [{"start": 20, "end": 25, "text": "a"}]
        turns = [(0, 5, "SPEAKER_00"), (6, 10, "SPEAKER_01")]
        result = transcribe_core.assign_speakers(segments, turns)
        self.assertEqual(result[0]["speaker"], "Speaker 2")

    def test_no_turns_leaves_segments_unchanged(self):
        segments = [{"start": 0, "end": 5, "text": "a"}]
        result = transcribe_core.assign_speakers(segments, [])
        self.assertNotIn("speaker", result[0])


if __name__ == "__main__":
    unittest.main()
