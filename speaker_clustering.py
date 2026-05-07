#!/usr/bin/env python3
"""Phase 3: Speaker Clustering and Unique Person Identification."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform


class SpeakerClusterer:
    """Cluster speakers using embedding similarity to identify unique persons."""

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
                if embedding:  # Skip empty embeddings
                    embeddings_list.append(embedding)
                    global_id = f"{file_key}:::{speaker_label}"
                    speaker_ids.append(global_id)
                    self.speaker_to_file[global_id] = (file_key, speaker_label)

        if not embeddings_list:
            self.logger.error("No embeddings found to cluster!")
            return np.array([]), []

        matrix = np.array(embeddings_list, dtype=np.float32)
        self.logger.info(f"Built embedding matrix: {matrix.shape}")
        return matrix, speaker_ids

    def compute_linkage(self, embedding_matrix: np.ndarray) -> np.ndarray:
        """Compute hierarchical clustering linkage matrix."""
        if embedding_matrix.shape[0] < 2:
            self.logger.warning("Not enough embeddings for clustering")
            return np.array([])

        self.logger.info("Computing pairwise distances...")
        distances = pdist(embedding_matrix, metric="cosine")

        self.logger.info("Computing hierarchical clustering linkage...")
        linkage_matrix = linkage(distances, method="ward")
        return linkage_matrix

    def cluster_speakers(self, linkage_matrix: np.ndarray, threshold: float = 0.5) -> dict[int, list[str]]:
        """Cluster speakers using agglomerative clustering."""
        if linkage_matrix.size == 0:
            return {}

        # Convert cosine distance threshold to dendrogram distance cutoff
        # threshold=0.5 cosine similarity = 0.5 distance
        clusters_array = fcluster(linkage_matrix, threshold, criterion="distance")

        clusters = {}
        for speaker_id, cluster_id in zip(self.speaker_to_file.keys(), clusters_array):
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
                "export_date": None,  # To be filled by caller
                "total_cluster_groups": len(self.clusters),
                "total_speaker_instances": sum(len(s) for s in self.clusters.values()),
                "clustering_method": "agglomerative_hierarchical",
                "distance_metric": "cosine",
                "linkage_method": "ward",
                "similarity_threshold": 0.5,
            },
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
    if linkage_matrix.size == 0:
        return {
            "status": "error",
            "message": "Clustering failed",
        }

    # Perform clustering
    clusters = clusterer.cluster_speakers(linkage_matrix, threshold=0.5)
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
        "unique_clusters": len(clusters),
        "unique_persons": len(set(person_assignments.values())),
        "export_path": str(export_path),
    }
