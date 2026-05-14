#!/usr/bin/env python3
"""Speaker identification and cross-file person tracking system."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional

from filename_parser import parse_telegram_filename, TelegramAudioMetadata


@dataclass
class SpeakerInstance:
    """Single speaker appearance in a file."""

    file_key: str  # Relative path key (e.g., "2020-10/12-10-2020...")
    speaker_label: str  # SPEAKER_00, SPEAKER_01, etc.
    duration_sec: float = 0.0
    word_count: int = 0


@dataclass
class Person:
    """Identified person across multiple calls."""

    person_id: str  # PERSON_001, PERSON_002, etc.
    phone_numbers: list[str] = field(default_factory=list)
    contact_names: set[str] = field(default_factory=set)
    appearances: list[SpeakerInstance] = field(default_factory=list)
    embedding_centroid: Optional[list[float]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "person_id": self.person_id,
            "phone_numbers": list(set(self.phone_numbers)),  # Deduplicate
            "contact_names": sorted(list(self.contact_names)),
            "appearance_count": len(self.appearances),
            "appearances": [
                {
                    "file_key": app.file_key,
                    "speaker_label": app.speaker_label,
                    "duration_sec": app.duration_sec,
                    "word_count": app.word_count,
                }
                for app in self.appearances
            ],
        }


@dataclass
class CallRecord:
    """Represents a single call/interaction."""

    timestamp: str  # ISO 8601
    file_key: str
    direction: str  # "incoming" or "outgoing"
    originating_phone: str  # The "call_id" from filename
    contact_phone: Optional[str]
    contact_name: Optional[str]
    speaker_count: int
    language: str  # "uk", "ru", etc.
    duration_sec: float
    qa_score: float
    participants: list[str] = field(default_factory=list)  # List of PERSON_IDs


class SpeakerIdentificationEngine:
    """Main speaker tracking and identification engine."""

    def __init__(
        self,
        manifest_path: Path,
        output_dir: Path | None = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.output_dir = Path(output_dir) if output_dir else self.manifest_path.parent
        self.logger = logger or logging.getLogger(__name__)

        self.people: dict[str, Person] = {}  # person_id -> Person
        self.next_person_id = 1
        self.call_records: list[CallRecord] = []
        self.phone_to_persons: dict[str, list[str]] = defaultdict(list)  # phone -> [person_ids]

    def load_manifest(self) -> dict[str, Any] | None:
        """Load manifest.json from diarization pipeline."""
        if not self.manifest_path.exists():
            self.logger.error(f"Manifest not found: {self.manifest_path}")
            return None

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.logger.info(f"Loaded manifest: {len(data.get('files', {}))} files")
            return data
        except Exception as exc:
            self.logger.error(f"Failed to load manifest: {exc}")
            return None

    def parse_all_filenames(self, manifest: dict[str, Any]) -> dict[str, TelegramAudioMetadata]:
        """Parse all filenames in manifest."""
        parsed = {}
        failures = []

        for file_key in manifest.get("files", {}).keys():
            # Extract just the filename from the key
            filename = Path(file_key).name
            metadata = parse_telegram_filename(filename)

            if metadata:
                parsed[file_key] = metadata
            else:
                failures.append(file_key)

        self.logger.info(f"Parsed {len(parsed)} filenames; {len(failures)} failures")
        if failures[:10]:  # Log first 10 failures
            self.logger.debug(f"Sample failures: {failures[:10]}")

        return parsed

    def build_call_records(
        self, manifest: dict[str, Any], parsed_filenames: dict[str, TelegramAudioMetadata]
    ) -> list[CallRecord]:
        """Build call records from manifest and parsed metadata."""
        records = []

        for file_key, file_info in manifest.get("files", {}).items():
            if file_info.get("status") != "done":
                continue

            # Get parsed filename metadata
            metadata = parsed_filenames.get(file_key)
            if not metadata:
                continue

            # Build call record
            timestamp = None
            if metadata.datetime_str:
                # Parse datetime and add timezone
                try:
                    from datetime import datetime

                    dt = datetime.strptime(metadata.datetime_str, "%d-%m-%Y_%H-%M")
                    timestamp = dt.isoformat() + "+02:00"  # Assume EET
                except Exception:
                    timestamp = f"{metadata.datetime_str}+02:00"

            quality_info = file_info.get("quality", {})

            record = CallRecord(
                timestamp=timestamp or "unknown",
                file_key=file_key,
                direction=metadata.direction,
                originating_phone=metadata.call_id,
                contact_phone=metadata.contact_id,
                contact_name=metadata.contact_name,
                speaker_count=quality_info.get("speaker_count", 0),
                language=file_info.get("detected_language", "uk"),
                duration_sec=file_info.get("duration_sec", 0.0),
                qa_score=quality_info.get("overall_score", 0.0),
            )
            records.append(record)

        return sorted(records, key=lambda r: r.timestamp)

    def build_phone_lookup(
        self, call_records: list[CallRecord]
    ) -> dict[str, dict[str, list[str]]]:
        """Build lookup of originating/contact phones -> file_keys."""
        lookup = {"originating": defaultdict(list), "contact": defaultdict(list)}

        for rec in call_records:
            lookup["originating"][rec.originating_phone].append(rec.file_key)
            if rec.contact_phone:
                lookup["contact"][rec.contact_phone].append(rec.file_key)

        return lookup

    def create_persons_json(self) -> dict[str, Any]:
        """Create the final persons.json export structure."""
        return {
            "persons": [person.to_dict() for person in self.people.values()],
            "call_log": [
                {
                    "timestamp": rec.timestamp,
                    "file_key": rec.file_key,
                    "direction": rec.direction,
                    "originating_phone": rec.originating_phone,
                    "contact_phone": rec.contact_phone,
                    "contact_name": rec.contact_name,
                    "speaker_count": rec.speaker_count,
                    "language": rec.language,
                    "duration_sec": rec.duration_sec,
                    "qa_score": rec.qa_score,
                    "participants": rec.participants,
                }
                for rec in self.call_records
            ],
            "metadata": {
                "total_files": len(self.call_records),
                "unique_persons": len(self.people),
                "export_date": None,  # Will be filled by caller
                "manifest_path": str(self.manifest_path),
            },
        }

    def run_phase1(self) -> dict[str, Any]:
        """Execute Phase 1: Filename parsing and call record building."""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 1: Filename Parsing & Call Records")
        self.logger.info("=" * 60)

        # Load manifest
        manifest = self.load_manifest()
        if not manifest:
            return {"status": "error", "message": "Failed to load manifest"}

        # Parse filenames
        parsed_files = self.parse_all_filenames(manifest)
        if not parsed_files:
            return {"status": "error", "message": "No filenames parsed successfully"}

        # Build call records
        self.call_records = self.build_call_records(manifest, parsed_files)
        self.logger.info(f"Built {len(self.call_records)} call records")

        # Build phone lookup
        phone_lookup = self.build_phone_lookup(self.call_records)
        unique_originating = len(phone_lookup["originating"])
        unique_contacts = len(phone_lookup["contact"])

        self.logger.info(f"Unique originating phones: {unique_originating}")
        self.logger.info(f"Unique contact phones: {unique_contacts}")

        # Export interim results
        interim_path = self.output_dir / "phase1_call_records.json"
        try:
            with open(interim_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "call_log": [
                            {
                                "timestamp": rec.timestamp,
                                "file_key": rec.file_key,
                                "direction": rec.direction,
                                "originating_phone": rec.originating_phone,
                                "contact_phone": rec.contact_phone,
                                "contact_name": rec.contact_name,
                            }
                            for rec in self.call_records
                        ],
                        "summary": {
                            "total_calls": len(self.call_records),
                            "unique_originating_phones": unique_originating,
                            "unique_contact_phones": unique_contacts,
                        },
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            self.logger.info(f"Exported phase 1 results to {interim_path}")
        except Exception as exc:
            self.logger.error(f"Failed to export phase 1 results: {exc}")

        return {
            "status": "success",
            "call_records_count": len(self.call_records),
            "unique_originating_phones": unique_originating,
            "unique_contact_phones": unique_contacts,
            "interim_export": str(interim_path),
        }
