import json
import tempfile
import unittest
from pathlib import Path

from person_linking import PersonLinker


class PersonLinkingTests(unittest.TestCase):
    def test_phone_based_export_links_calls_to_persons(self):
        with tempfile.TemporaryDirectory() as tmp:
            progress = Path(tmp)
            (progress / "phase1_call_records.json").write_text(
                json.dumps(
                    {
                        "call_log": [
                            {
                                "timestamp": "2020-10-12T06:11:00+02:00",
                                "file_key": "2020-10/a.mp3",
                                "direction": "incoming",
                                "originating_phone": "0456307014",
                                "contact_phone": "0443334085",
                                "contact_name": "Name",
                            },
                            {
                                "timestamp": "2020-10-12T06:58:00+02:00",
                                "file_key": "2020-10/b.mp3",
                                "direction": "outgoing",
                                "originating_phone": "0456307014",
                                "contact_phone": None,
                                "contact_name": None,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (progress / "manifest.json").write_text(
                json.dumps(
                    {
                        "files": {
                            "2020-10/a.mp3": {
                                "duration_sec": 12.5,
                                "detected_language": "ru",
                                "quality": {"overall_score": 99.0, "grade": "A", "speaker_count": 2},
                            },
                            "2020-10/b.mp3": {
                                "duration_sec": 7.5,
                                "detected_language": "uk",
                                "quality": {"overall_score": 95.0, "grade": "A", "speaker_count": 1},
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            result = PersonLinker(progress).export_persons()

            self.assertEqual(result["phone_persons"], 1)
            data = json.loads((progress / "persons.json").read_text(encoding="utf-8"))
            self.assertEqual(data["metadata"]["total_calls"], 2)
            self.assertEqual(data["persons"][0]["call_count"], 2)
            self.assertEqual(data["persons"][0]["total_duration_sec"], 20.0)
            self.assertEqual(data["call_log"][0]["participants"], ["PERSON_001"])

    def test_optional_voice_clusters_are_exported_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            progress = Path(tmp)
            (progress / "phase1_call_records.json").write_text(
                json.dumps(
                    {
                        "call_log": [
                            {
                                "timestamp": "2020-10-12T06:11:00+02:00",
                                "file_key": "2020-10/a.mp3",
                                "direction": "incoming",
                                "originating_phone": "0456307014",
                                "contact_phone": None,
                                "contact_name": None,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (progress / "manifest.json").write_text(json.dumps({"files": {}}), encoding="utf-8")
            (progress / "phase3_speaker_clusters.json").write_text(
                json.dumps(
                    {
                        "clusters": {
                            "cluster_1": [
                                {
                                    "file_key": "2020-10/a.mp3",
                                    "speaker_label": "SPEAKER_00",
                                    "person_id": 1,
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            result = PersonLinker(progress).export_persons()

            self.assertEqual(result["phone_persons"], 1)
            self.assertEqual(result["voice_persons"], 1)
            data = json.loads((progress / "persons.json").read_text(encoding="utf-8"))
            person_ids = {person["person_id"] for person in data["persons"]}
            self.assertEqual(person_ids, {"PERSON_001", "VOICE_PERSON_001"})
            self.assertEqual(data["call_log"][0]["participants"], ["PERSON_001", "VOICE_PERSON_001"])


if __name__ == "__main__":
    unittest.main()
