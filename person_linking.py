#!/usr/bin/env python3
"""Phase 4-5: link call records into person records and export persons.json."""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class PersonAggregate:
    """Aggregated identity built from call metadata and optional voice clusters."""

    person_id: str
    link_method: str
    primary_phone: Optional[str] = None
    phone_numbers: set[str] = field(default_factory=set)
    observed_contact_phones: set[str] = field(default_factory=set)
    observed_contact_names: set[str] = field(default_factory=set)
    call_file_keys: list[str] = field(default_factory=list)
    speaker_appearances: list[dict[str, Any]] = field(default_factory=list)
    direction_counts: Counter[str] = field(default_factory=Counter)
    language_counts: Counter[str] = field(default_factory=Counter)
    total_duration_sec: float = 0.0
    qa_scores: list[float] = field(default_factory=list)
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None

    def add_call(self, call: dict[str, Any]) -> None:
        """Attach a call-level observation to this person."""
        file_key = call.get("file_key")
        if file_key and file_key not in self.call_file_keys:
            self.call_file_keys.append(file_key)

        originating_phone = call.get("originating_phone")
        if originating_phone:
            self.phone_numbers.add(originating_phone)

        contact_phone = call.get("contact_phone")
        if contact_phone:
            self.observed_contact_phones.add(contact_phone)

        contact_name = call.get("contact_name")
        if contact_name:
            self.observed_contact_names.add(contact_name)

        direction = call.get("direction")
        if direction:
            self.direction_counts[direction] += 1

        language = call.get("language")
        if language:
            self.language_counts[language] += 1

        duration = _coerce_float(call.get("duration_sec"))
        if duration is not None:
            self.total_duration_sec += duration

        qa_score = _coerce_float(call.get("qa_score"))
        if qa_score is not None:
            self.qa_scores.append(qa_score)

        timestamp = call.get("timestamp")
        if timestamp:
            if self.first_seen is None or timestamp < self.first_seen:
                self.first_seen = timestamp
            if self.last_seen is None or timestamp > self.last_seen:
                self.last_seen = timestamp

    def to_dict(self) -> dict[str, Any]:
        """Convert to a stable JSON-serializable shape."""
        avg_qa = sum(self.qa_scores) / len(self.qa_scores) if self.qa_scores else None
        return {
            "person_id": self.person_id,
            "link_method": self.link_method,
            "primary_phone": self.primary_phone,
            "phone_numbers": sorted(self.phone_numbers),
            "observed_contact_phones": sorted(self.observed_contact_phones),
            "observed_contact_names": sorted(self.observed_contact_names),
            "call_count": len(self.call_file_keys),
            "call_file_keys": sorted(self.call_file_keys),
            "speaker_appearances": sorted(
                self.speaker_appearances,
                key=lambda item: (item.get("file_key", ""), item.get("speaker_label", "")),
            ),
            "direction_counts": dict(sorted(self.direction_counts.items())),
            "language_counts": dict(sorted(self.language_counts.items())),
            "total_duration_sec": round(self.total_duration_sec, 3),
            "avg_qa_score": round(avg_qa, 3) if avg_qa is not None else None,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


class PersonLinker:
    """Build a final persons.json from Phase 1 and optional Phase 3 output."""

    def __init__(
        self,
        progress_dir: Path = Path("output/_progress"),
        manifest_path: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.progress_dir = Path(progress_dir)
        self.manifest_path = Path(manifest_path) if manifest_path else self.progress_dir / "manifest.json"
        self.logger = logger or logging.getLogger(__name__)

    def load_phase1_calls(self, phase1_path: Optional[Path] = None) -> list[dict[str, Any]]:
        """Load Phase 1 call records."""
        path = Path(phase1_path) if phase1_path else self.progress_dir / "phase1_call_records.json"
        data = _load_json(path)
        calls = data.get("call_log", [])
        if not isinstance(calls, list):
            raise ValueError(f"Invalid call_log in {path}")
        return [call for call in calls if isinstance(call, dict)]

    def load_manifest(self) -> dict[str, Any]:
        """Load diarization manifest if available."""
        if not self.manifest_path.exists():
            self.logger.warning("Manifest not found: %s", self.manifest_path)
            return {"files": {}}
        return _load_json(self.manifest_path)

    def load_phase3_clusters(self, clusters_path: Optional[Path] = None) -> dict[str, Any]:
        """Load Phase 3 clusters when they exist and contain speaker instances."""
        path = Path(clusters_path) if clusters_path else self.progress_dir / "phase3_speaker_clusters.json"
        if not path.exists():
            return {}
        data = _load_json(path)
        clusters = data.get("clusters", {})
        if not isinstance(clusters, dict) or not clusters:
            return {}
        return data

    def enrich_calls(
        self,
        phase1_calls: list[dict[str, Any]],
        manifest: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Merge Phase 1 metadata with manifest QA/language/duration details."""
        files = manifest.get("files", {})
        enriched = []

        for call in phase1_calls:
            item = dict(call)
            file_info = files.get(call.get("file_key"), {})
            quality = file_info.get("quality", {}) if isinstance(file_info, dict) else {}

            item["speaker_count"] = quality.get("speaker_count", item.get("speaker_count", 0))
            item["language"] = file_info.get("detected_language", item.get("language")) if isinstance(file_info, dict) else item.get("language")
            item["duration_sec"] = file_info.get("duration_sec", item.get("duration_sec", 0.0)) if isinstance(file_info, dict) else item.get("duration_sec", 0.0)
            item["qa_score"] = quality.get("overall_score", item.get("qa_score"))
            item["quality_grade"] = quality.get("grade", item.get("quality_grade"))
            item["participants"] = []
            enriched.append(item)

        return enriched

    def build_phone_persons(self, calls: list[dict[str, Any]]) -> tuple[dict[str, PersonAggregate], dict[str, str]]:
        """Build conservative phone-based person records from originating phones."""
        phones = sorted({call.get("originating_phone") for call in calls if call.get("originating_phone")})
        phone_to_person_id = {phone: f"PERSON_{idx:03d}" for idx, phone in enumerate(phones, start=1)}

        persons = {
            person_id: PersonAggregate(
                person_id=person_id,
                link_method="phone_metadata",
                primary_phone=phone,
                phone_numbers={phone},
            )
            for phone, person_id in phone_to_person_id.items()
        }

        for call in calls:
            phone = call.get("originating_phone")
            person_id = phone_to_person_id.get(phone)
            if not person_id:
                continue
            persons[person_id].add_call(call)
            call["participants"] = _append_unique(call.get("participants", []), person_id)

        return persons, phone_to_person_id

    def build_voice_persons(
        self,
        clusters_data: dict[str, Any],
        calls_by_file: dict[str, dict[str, Any]],
    ) -> dict[str, PersonAggregate]:
        """Build voice-cluster persons without forcing them onto phone identities."""
        clusters = clusters_data.get("clusters", {})
        persons: dict[str, PersonAggregate] = {}

        for idx, (_cluster_key, appearances) in enumerate(sorted(clusters.items()), start=1):
            if not isinstance(appearances, list):
                continue
            person_id = f"VOICE_PERSON_{idx:03d}"
            person = PersonAggregate(person_id=person_id, link_method="voice_cluster")

            for appearance in appearances:
                if not isinstance(appearance, dict):
                    continue
                file_key = appearance.get("file_key")
                speaker_label = appearance.get("speaker_label")
                if not file_key or not speaker_label:
                    continue

                person.speaker_appearances.append(
                    {
                        "file_key": file_key,
                        "speaker_label": speaker_label,
                        "cluster_person_id": appearance.get("person_id"),
                    }
                )

                call = calls_by_file.get(file_key)
                if call:
                    person.add_call(call)
                    call["participants"] = _append_unique(call.get("participants", []), person_id)

            if person.speaker_appearances:
                persons[person_id] = person

        return persons

    def export_persons(
        self,
        output_path: Optional[Path] = None,
        phase1_path: Optional[Path] = None,
        clusters_path: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Run Phase 4-5 and write the final persons.json."""
        phase1_calls = self.load_phase1_calls(phase1_path)
        manifest = self.load_manifest()
        calls = self.enrich_calls(phase1_calls, manifest)
        calls_by_file = {call.get("file_key"): call for call in calls if call.get("file_key")}

        phone_persons, _phone_lookup = self.build_phone_persons(calls)
        clusters_data = self.load_phase3_clusters(clusters_path)
        voice_persons = self.build_voice_persons(clusters_data, calls_by_file) if clusters_data else {}

        persons = {**phone_persons, **voice_persons}
        export = {
            "metadata": {
                "export_date": datetime.now().astimezone().isoformat(timespec="seconds"),
                "source_phase1": str(Path(phase1_path) if phase1_path else self.progress_dir / "phase1_call_records.json"),
                "source_phase3": str(Path(clusters_path) if clusters_path else self.progress_dir / "phase3_speaker_clusters.json"),
                "voice_clusters_loaded": bool(voice_persons),
                "linking_strategy": "phone_metadata_with_optional_voice_clusters",
                "total_calls": len(calls),
                "phone_persons": len(phone_persons),
                "voice_persons": len(voice_persons),
                "total_person_records": len(persons),
            },
            "persons": [persons[key].to_dict() for key in sorted(persons)],
            "call_log": sorted(calls, key=lambda call: (call.get("timestamp", ""), call.get("file_key", ""))),
        }

        path = Path(output_path) if output_path else self.progress_dir / "persons.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(export, f, ensure_ascii=False, indent=2)

        self.logger.info("Exported persons to %s", path)
        return {
            "status": "success",
            "export_path": str(path),
            "total_calls": len(calls),
            "phone_persons": len(phone_persons),
            "voice_persons": len(voice_persons),
            "total_person_records": len(persons),
        }


def run_phase4_5(
    progress_dir: Path = Path("output/_progress"),
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    """Execute Phase 4-5: person linking and final persons.json export."""
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("PHASE 4-5: Person Linking and persons.json Export")
    logger.info("=" * 70)

    linker = PersonLinker(progress_dir=progress_dir, logger=logger)
    return linker.export_persons()


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON in {path}")
    return data


def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _append_unique(items: list[str], item: str) -> list[str]:
    if item not in items:
        items.append(item)
    return items
