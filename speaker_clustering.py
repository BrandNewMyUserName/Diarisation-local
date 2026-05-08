#!/usr/bin/env python3
"""Phase 3: Speaker Clustering and Unique Person Identification."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


class SpeakerClusterer:
    """Cluster speakers using embedding similarity to identify unique persons."""

    DEFAULT_SIMILARITY_THRESHOLD = 0.67

    def __init__(
        self,
        embeddings_path: Path,
        logger: Optional[logging.Logger] = None,
    ):
        self.embeddings_path = Path(embeddings_path)
        self.logger = logger or logging.getLogger(__name__)
        self.embeddings: dict[str, dict[str, list[float]]] = {}
        self.speaker_to_file: dict[str, tuple[str, str]] = {}  # global_id -> (file_key, speaker_label)
        self.clusters: dict[int, list[str]] = {}  # cluster_id -> [speaker_ids]
        self.person_assignments: dict[str, int] = {}  # (file, speaker) -> person_id
        self.similarity_threshold = self.DEFAULT_SIMILARITY_THRESHOLD
        self.skipped_embeddings: list[dict[str, Any]] = []

    def load_embeddings(self) -> bool:
        """Load speaker embeddings from Phase 2 output."""
        if not self.embeddings_path.exists():
            self.logger.error(f"Embeddings file not found: {self.embeddings_path}")
            return False

        try:
            with open(self.embeddings_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.embeddings = data.get("embeddings_by_file", {})
            self.logger.info(f"Loaded embeddings for {len(self.embeddings)} files")
            return True
        except Exception as exc:
            self.logger.error(f"Failed to load embeddings: {exc}")
            return False

    def build_embedding_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Build matrix of all speaker embeddings with speaker IDs."""
        embeddings_list = []
        speaker_ids = []

        for file_key, speakers in self.embeddings.items():
            for speaker_label, embedding in speakers.items():
                if not embedding:
                    self.skipped_embeddings.append(
                        {"file_key": file_key, "speaker_label": speaker_label, "reason": "empty_embedding"}
                    )
                    continue

                vector = np.asarray(embedding, dtype=np.float32)
                if vector.ndim != 1:
                    self.skipped_embeddings.append(
                        {"file_key": file_key, "speaker_label": speaker_label, "reason": "non_vector_embedding"}
                    )
                    continue
                if not np.isfinite(vector).all():
                    self.skipped_embeddings.append(
                        {"file_key": file_key, "speaker_label": speaker_label, "reason": "non_finite_embedding"}
                    )
                    continue
                if float(np.linalg.norm(vector)) == 0.0:
                    self.skipped_embeddings.append(
                        {"file_key": file_key, "speaker_label": speaker_label, "reason": "zero_norm_embedding"}
                    )
                    continue

                embeddings_list.append(vector)
                global_id = f"{file_key}:::{speaker_label}"
                speaker_ids.append(global_id)
                self.speaker_to_file[global_id] = (file_key, speaker_label)

        if not embeddings_list:
            self.logger.error("No embeddings found to cluster!")
            return np.array([]), []

        matrix = np.array(embeddings_list, dtype=np.float32)
        self.logger.info(f"Built embedding matrix: {matrix.shape}")
        if self.skipped_embeddings:
            self.logger.warning(
                "Skipped %d invalid embedding(s) before clustering",
                len(self.skipped_embeddings),
            )
        return matrix, speaker_ids

    def compute_linkage(self, embedding_matrix: np.ndarray) -> np.ndarray:
        """Compute hierarchical clustering linkage matrix."""
        if embedding_matrix.shape[0] < 2:
            self.logger.warning("Not enough embeddings for clustering")
            return np.array([])

        self.logger.info("Computing pairwise distances...")
        distances = pdist(embedding_matrix, metric="cosine")

        self.logger.info("Computing hierarchical clustering linkage...")
        linkage_matrix = linkage(distances, method="average")
        return linkage_matrix

    def cluster_speakers(
        self,
        linkage_matrix: np.ndarray,
        speaker_ids: list[str],
        similarity_threshold: float = 0.5,
    ) -> dict[int, list[str]]:
        """Cluster speakers using agglomerative clustering."""
        if not speaker_ids:
            return {}
        if len(speaker_ids) == 1 or linkage_matrix.size == 0:
            return {1: [speaker_ids[0]]}

        # Cosine distance is 1 - cosine similarity.
        distance_cutoff = 1.0 - similarity_threshold
        clusters_array = fcluster(linkage_matrix, distance_cutoff, criterion="distance")

        clusters = {}
        for speaker_id, cluster_id in zip(speaker_ids, clusters_array):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(speaker_id)

        self.logger.info(f"Clustered speakers into {len(clusters)} groups")
        return clusters

    def assign_person_ids(self, clusters: dict[int, list[str]]) -> dict[str, int]:
        """Assign global PERSON_IDs to speaker clusters."""
        assignments = {}
        person_counter = 1

        for cluster_id, speaker_ids in clusters.items():
            person_id = person_counter
            for speaker_id in speaker_ids:
                assignments[speaker_id] = person_id
            person_counter += 1

        self.logger.info(f"Assigned {person_counter - 1} unique persons")
        return assignments

    def export_clusters(self, output_path: Path) -> None:
        """Export clustering results to JSON."""
        export = {
            "metadata": {
                "export_date": datetime.now().astimezone().isoformat(timespec="seconds"),
                "total_cluster_groups": len(self.clusters),
                "total_speaker_instances": sum(len(s) for s in self.clusters.values()),
                "clustering_method": "agglomerative_hierarchical",
                "distance_metric": "cosine",
                "linkage_method": "average",
                "similarity_threshold": self.similarity_threshold,
                "skipped_embeddings": len(self.skipped_embeddings),
            },
            "skipped_embeddings": self.skipped_embeddings,
            "clusters": {},
        }

        for cluster_id, speaker_ids in self.clusters.items():
            cluster_info = []
            for speaker_id in speaker_ids:
                file_key, speaker_label = self.speaker_to_file.get(speaker_id, ("", ""))
                person_id = self.person_assignments.get(speaker_id, 0)
                cluster_info.append(
                    {
                        "file_key": file_key,
                        "speaker_label": speaker_label,
                        "person_id": person_id,
                    }
                )
            export["clusters"][f"cluster_{cluster_id}"] = cluster_info

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Exported clusters to {output_path}")


def run_phase3(
    embeddings_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
    similarity_threshold: float = SpeakerClusterer.DEFAULT_SIMILARITY_THRESHOLD,
) -> dict[str, Any]:
    """Execute Phase 3: Speaker clustering using embeddings."""

    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

    if embeddings_path is None:
        embeddings_path = Path("output/_progress/phase2_embeddings.json")
    if output_dir is None:
        output_dir = Path("output/_progress")

    logger.info("=" * 70)
    logger.info("PHASE 3: Speaker Clustering")
    logger.info("=" * 70)

    # Initialize clusterer
    clusterer = SpeakerClusterer(embeddings_path, logger)

    # Load embeddings
    if not clusterer.load_embeddings():
        return {
            "status": "error",
            "message": "Failed to load embeddings",
        }

    # Build embedding matrix
    embedding_matrix, speaker_ids = clusterer.build_embedding_matrix()
    if embedding_matrix.size == 0:
        return {
            "status": "error",
            "message": "No embeddings to cluster",
        }

    # Compute linkage and cluster
    linkage_matrix = clusterer.compute_linkage(embedding_matrix)
    if embedding_matrix.shape[0] > 1 and linkage_matrix.size == 0:
        return {
            "status": "error",
            "message": "Clustering failed",
        }

    # Perform clustering
    clusterer.similarity_threshold = similarity_threshold
    clusters = clusterer.cluster_speakers(
        linkage_matrix,
        speaker_ids,
        similarity_threshold=clusterer.similarity_threshold,
    )
    clusterer.clusters = clusters

    # Assign person IDs
    person_assignments = clusterer.assign_person_ids(clusters)
    clusterer.person_assignments = person_assignments

    # Export results
    export_path = output_dir / "phase3_speaker_clusters.json"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    clusterer.export_clusters(export_path)

    return {
        "status": "success",
        "total_speaker_instances": len(speaker_ids),
        "skipped_embeddings": len(clusterer.skipped_embeddings),
        "unique_clusters": len(clusters),
        "unique_persons": len(set(person_assignments.values())),
        "similarity_threshold": clusterer.similarity_threshold,
        "export_path": str(export_path),
    }
