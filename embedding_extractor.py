#!/usr/bin/env python3
"""Phase 2: Speaker Embedding Extraction and Management."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np


class EmbeddingExtractor:
    """Extract and manage speaker embeddings from diarization outputs."""

    def __init__(
        self,
        output_dir: Path,
        logger: Optional[logging.Logger] = None,
    ):
        self.output_dir = Path(output_dir)
        self.logger = logger or logging.getLogger(__name__)
        self.embeddings_index: dict[str, dict[str, Any]] = {}  # file_key -> {speaker_id: metadata}
        self.embedding_vectors: dict[str, dict[str, list[float]]] = {}  # file_key -> {speaker_id: vector}

    def load_embeddings_from_json(self, json_path: Path) -> dict[str, list[float]] | None:
        """Load speaker embeddings from a WhisperX JSON output file."""
        if not json_path.exists():
            return None

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            speaker_embeddings = data.get("speaker_embeddings", {})
            if speaker_embeddings:
                return speaker_embeddings
        except Exception as exc:
            self.logger.debug(f"Failed to load embeddings from {json_path}: {exc}")

        return None

    def scan_output_directory(self) -> dict[str, Path]:
        """Scan for all WhisperX JSON output files."""
        json_files = {}
        manifest_outputs = self._manifest_json_outputs()

        for json_path in self.output_dir.glob("**/*.json"):
            # Skip manifest and other non-transcript files
            if json_path.name in [
                "manifest.json",
                "phase1_call_records.json",
                "phase2_embeddings.json",
                "phase3_speaker_clusters.json",
                "persons.json",
            ]:
                continue

            key = manifest_outputs.get(_path_lookup_key(json_path))
            if key is None:
                key = manifest_outputs.get(_path_lookup_key(json_path.resolve()))
            if key is None:
                key = self._file_key_from_json(json_path)

            if key:
                json_files[key] = json_path

        return json_files

    def _manifest_json_outputs(self) -> dict[str, str]:
        """Map transcript JSON paths to manifest file keys."""
        manifest_path = self.output_dir / "_progress" / "manifest.json"
        if not manifest_path.exists():
            return {}

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as exc:
            self.logger.debug(f"Failed to load manifest for output mapping: {exc}")
            return {}

        path_to_key: dict[str, str] = {}
        for file_key, file_info in manifest.get("files", {}).items():
            for output_path in file_info.get("output_paths") or []:
                path = Path(output_path)
                if path.suffix.lower() != ".json":
                    continue
                path_to_key[_path_lookup_key(path)] = file_key
                try:
                    path_to_key[_path_lookup_key(path.resolve())] = file_key
                except Exception:
                    pass

        return path_to_key

    def _file_key_from_json(self, json_path: Path) -> str | None:
        """Best-effort fallback for transcript JSONs not present in manifest."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            relative_key = data.get("source", {}).get("relative_key")
            if relative_key:
                return relative_key
        except Exception:
            pass

        try:
            relative = json_path.relative_to(self.output_dir)
        except ValueError:
            return None

        original_stem = json_path.stem.rsplit("__", 1)[0]
        return f"{relative.parent.as_posix()}/{original_stem}.mp3"

    def extract_all_embeddings(self, file_map: dict[str, Path]) -> tuple[int, int]:
        """Extract embeddings from all JSON files."""
        success = 0
        failed = 0

        for file_key, json_path in file_map.items():
            embeddings = self.load_embeddings_from_json(json_path)
            if embeddings:
                self.embedding_vectors[file_key] = embeddings
                self.embeddings_index[file_key] = {
                    speaker_id: {"embedding_dim": len(emb) if isinstance(emb, list) else None}
                    for speaker_id, emb in embeddings.items()
                }
                success += 1
            else:
                failed += 1

        return success, failed

    def compute_embedding_statistics(self) -> dict[str, Any]:
        """Compute statistics on extracted embeddings."""
        stats = {
            "files_with_embeddings": len(self.embedding_vectors),
            "total_speaker_embeddings": sum(
                len(embeddings) for embeddings in self.embedding_vectors.values()
            ),
            "unique_speakers_per_file": [],
            "embedding_dimensions": set(),
        }

        for file_key, embeddings in self.embedding_vectors.items():
            stats["unique_speakers_per_file"].append(len(embeddings))
            for embedding in embeddings.values():
                if isinstance(embedding, list):
                    stats["embedding_dimensions"].add(len(embedding))

        stats["unique_speakers_per_file"] = {
            "min": min(stats["unique_speakers_per_file"]) if stats["unique_speakers_per_file"] else 0,
            "max": max(stats["unique_speakers_per_file"]) if stats["unique_speakers_per_file"] else 0,
            "avg": (
                sum(stats["unique_speakers_per_file"]) / len(stats["unique_speakers_per_file"])
                if stats["unique_speakers_per_file"]
                else 0
            ),
        }
        stats["embedding_dimensions"] = sorted(list(stats["embedding_dimensions"]))

        return stats

    def export_embeddings(self, export_path: Path) -> None:
        """Export embeddings to JSON file for Phase 3 clustering."""
        export = {
            "metadata": {
                "export_date": datetime.now().astimezone().isoformat(timespec="seconds"),
                "total_files": len(self.embedding_vectors),
                "total_speaker_embeddings": sum(len(e) for e in self.embedding_vectors.values()),
                "embedding_stats": self.compute_embedding_statistics(),
            },
            "embeddings_by_file": {},
        }

        # Convert embeddings to JSON-serializable format
        for file_key, embeddings in self.embedding_vectors.items():
            export["embeddings_by_file"][file_key] = {
                speaker_id: emb if isinstance(emb, list) else emb.tolist()
                for speaker_id, emb in embeddings.items()
            }

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Exported embeddings to {export_path}")


def run_phase2(
    output_dir: Path = Path("output"),
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    """Execute Phase 2: Extract speaker embeddings from all diarization outputs."""

    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("PHASE 2: Speaker Embedding Extraction")
    logger.info("=" * 70)

    # Initialize extractor
    extractor = EmbeddingExtractor(output_dir, logger)

    # Scan for JSON files
    logger.info("Scanning for WhisperX output files...")
    file_map = extractor.scan_output_directory()
    logger.info(f"Found {len(file_map)} transcript files")

    if not file_map:
        return {
            "status": "warning",
            "message": "No WhisperX JSON output files found",
            "files_scanned": 0,
        }

    # Extract embeddings
    logger.info("Extracting speaker embeddings...")
    success, failed = extractor.extract_all_embeddings(file_map)

    logger.info(f"✓ Extracted embeddings from {success} files")
    if failed > 0:
        logger.warning(f"⚠ {failed} files did not have embeddings in JSON")

    # Compute statistics
    stats = extractor.compute_embedding_statistics()
    logger.info(f"Statistics:")
    logger.info(f"  • Files with embeddings: {stats['files_with_embeddings']}")
    logger.info(f"  • Total speaker embeddings: {stats['total_speaker_embeddings']}")
    logger.info(f"  • Embedding dimensions: {stats['embedding_dimensions']}")
    logger.info(f"  • Speakers per file: min={stats['unique_speakers_per_file']['min']}, " +
                f"max={stats['unique_speakers_per_file']['max']}, " +
                f"avg={stats['unique_speakers_per_file']['avg']:.1f}")

    # Export embeddings
    export_path = output_dir / "_progress" / "phase2_embeddings.json"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    extractor.export_embeddings(export_path)

    return {
        "status": "success" if success > 0 else "warning",
        "files_processed": success,
        "files_without_embeddings": failed,
        "total_embeddings": stats["total_speaker_embeddings"],
        "export_path": str(export_path),
        "statistics": stats,
    }


def _path_lookup_key(path: Path) -> str:
    return str(path).replace("\\", "/").lower()
