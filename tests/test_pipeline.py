import logging
import tempfile
import unittest
from pathlib import Path

import diarization_pipeline as pipeline


class PipelineTests(unittest.TestCase):
    def test_config_hash_ignores_adaptive_batch_size(self):
        cfg_a = pipeline.DEFAULT_CONFIG.copy()
        cfg_b = pipeline.DEFAULT_CONFIG.copy()
        cfg_a["batch_size"] = 16
        cfg_b["batch_size"] = 4

        self.assertEqual(pipeline._config_hash(cfg_a), pipeline._config_hash(cfg_b))

    def test_config_hash_includes_speaker_embedding_mode(self):
        cfg_a = pipeline.DEFAULT_CONFIG.copy()
        cfg_b = pipeline.DEFAULT_CONFIG.copy()
        cfg_a["extract_speaker_embeddings"] = True
        cfg_b["extract_speaker_embeddings"] = False

        self.assertNotEqual(pipeline._config_hash(cfg_a), pipeline._config_hash(cfg_b))

    def test_completed_record_remains_resumable_after_skip_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "audio"
            state = Path(tmp) / "state"
            output = Path(tmp) / "out.txt"
            root.mkdir()
            audio = root / "sample.wav"
            audio.write_bytes(b"fake")
            output.write_text("done", encoding="utf-8")

            logger = logging.getLogger("test-progress")
            tracker = pipeline.ProgressTracker(state, root, "cfg", logger)
            fingerprint = pipeline._file_fingerprint(audio)
            tracker.update_file(
                audio,
                status="done",
                stage="done",
                fingerprint=fingerprint,
                config_hash="cfg",
                output_paths=[str(output)],
            )
            self.assertTrue(tracker.is_completed(audio, fingerprint, "cfg"))

            tracker.update_file(audio, status="done", stage="done", last_resume_skip_at="now")
            self.assertTrue(tracker.is_completed(audio, fingerprint, "cfg"))

    def test_completed_record_can_require_speaker_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "audio"
            state = Path(tmp) / "state"
            output = Path(tmp) / "out.json"
            root.mkdir()
            audio = root / "sample.wav"
            audio.write_bytes(b"fake")
            output.write_text('{"segments": []}', encoding="utf-8")

            logger = logging.getLogger("test-progress")
            tracker = pipeline.ProgressTracker(state, root, "cfg", logger)
            fingerprint = pipeline._file_fingerprint(audio)
            tracker.update_file(
                audio,
                status="done",
                stage="done",
                fingerprint=fingerprint,
                config_hash="cfg",
                output_paths=[str(output)],
            )

            self.assertTrue(tracker.is_completed(audio, fingerprint, "cfg"))
            self.assertFalse(
                tracker.is_completed(
                    audio,
                    fingerprint,
                    "cfg",
                    require_speaker_embeddings=True,
                )
            )

            output.write_text(
                '{"segments": [], "speaker_embeddings": {"SPEAKER_00": [0.1, 0.2]}}',
                encoding="utf-8",
            )
            self.assertTrue(
                tracker.is_completed(
                    audio,
                    fingerprint,
                    "cfg",
                    require_speaker_embeddings=True,
                )
            )

    def test_quality_scores_good_speaker_assignment_high(self):
        result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "speaker": "SPEAKER_00",
                    "text": "привіт",
                    "words": [
                        {"word": "привіт", "start": 0.1, "end": 0.8, "score": 0.92, "speaker": "SPEAKER_00"}
                    ],
                }
            ]
        }
        diarize_segments = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]

        quality = pipeline.calculate_quality(result, diarize_segments, 2.0)

        self.assertGreaterEqual(quality["overall_score"], 90)
        self.assertEqual(quality["grade"], "A")
        self.assertEqual(quality["speaker_count"], 1)


if __name__ == "__main__":
    unittest.main()
