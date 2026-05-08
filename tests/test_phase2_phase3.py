import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from embedding_extractor import EmbeddingExtractor
from speaker_clustering import SpeakerClusterer, run_phase3


class Phase2Phase3Tests(unittest.TestCase):
    def test_embedding_scan_uses_manifest_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            progress = output / "_progress"
            transcript_dir = output / "2020-10"
            progress.mkdir(parents=True)
            transcript_dir.mkdir()

            transcript = transcript_dir / "12-10-2020_06-11_0456307014_incoming_0443334085_user___e5f7db45.json"
            transcript.write_text(
                json.dumps({"speaker_embeddings": {"SPEAKER_00": [0.1, 0.2]}}),
                encoding="utf-8",
            )
            expected_key = "2020-10/12-10-2020_06-11_0456307014_incoming_0443334085_user_.mp3"
            (progress / "manifest.json").write_text(
                json.dumps(
                    {
                        "files": {
                            expected_key: {
                                "output_paths": [str(transcript)],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            file_map = EmbeddingExtractor(output).scan_output_directory()

            self.assertEqual(file_map, {expected_key: transcript})

    def test_clustering_handles_single_embedding(self):
        clusterer = SpeakerClusterer(Path("unused.json"))
        clusterer.speaker_to_file["file.mp3:::SPEAKER_00"] = ("file.mp3", "SPEAKER_00")

        clusters = clusterer.cluster_speakers(
            np.array([]),
            ["file.mp3:::SPEAKER_00"],
            similarity_threshold=0.5,
        )

        self.assertEqual(clusters, {1: ["file.mp3:::SPEAKER_00"]})

    def test_embedding_matrix_skips_non_finite_vectors(self):
        clusterer = SpeakerClusterer(Path("unused.json"))
        clusterer.embeddings = {
            "a.mp3": {
                "SPEAKER_00": [0.1, 0.2],
                "SPEAKER_01": [float("nan"), 0.3],
                "SPEAKER_02": [0.0, 0.0],
            }
        }

        matrix, speaker_ids = clusterer.build_embedding_matrix()

        self.assertEqual(matrix.shape, (1, 2))
        self.assertEqual(speaker_ids, ["a.mp3:::SPEAKER_00"])
        reasons = {item["reason"] for item in clusterer.skipped_embeddings}
        self.assertEqual(reasons, {"non_finite_embedding", "zero_norm_embedding"})

    def test_run_phase3_uses_configurable_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            progress = Path(tmp)
            embeddings_path = progress / "phase2_embeddings.json"
            embeddings_path.write_text(
                json.dumps(
                    {
                        "embeddings_by_file": {
                            "a.mp3": {
                                "SPEAKER_00": [1.0, 0.0],
                                "SPEAKER_01": [0.9, 0.1],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            result = run_phase3(
                embeddings_path,
                progress,
                similarity_threshold=0.67,
            )

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["similarity_threshold"], 0.67)


if __name__ == "__main__":
    unittest.main()
