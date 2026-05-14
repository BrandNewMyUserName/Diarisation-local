#!/usr/bin/env python3
"""Run Phase 2 extraction, Phase 3 clustering, and Phase 4-5 final export."""

import argparse
import json
import logging
from pathlib import Path

from embedding_extractor import run_phase2
from person_linking import run_phase4_5
from speaker_clustering import SpeakerClusterer, run_phase3


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 2 extraction through Phase 5 export.")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=SpeakerClusterer.DEFAULT_SIMILARITY_THRESHOLD,
        help="Cosine similarity threshold for Phase 3 speaker clustering.",
    )
    args = parser.parse_args()

    output_dir = Path("output")
    progress_dir = output_dir / "_progress"

    logger.info("Starting Phase 2 -> 5 chain")

    phase2 = run_phase2(output_dir, logger)
    logger.info("Phase 2 result:")
    logger.info(json.dumps(phase2, indent=2, ensure_ascii=False))

    if phase2["status"] != "success":
        logger.error("Stopping: Phase 2 did not extract embeddings. Re-run diarization first.")
        return 1

    embeddings_path = Path(phase2["export_path"])
    phase3 = run_phase3(
        embeddings_path,
        progress_dir,
        logger,
        similarity_threshold=args.similarity_threshold,
    )
    logger.info("Phase 3 result:")
    logger.info(json.dumps(phase3, indent=2, ensure_ascii=False))

    if phase3["status"] != "success":
        logger.error("Stopping: Phase 3 clustering failed.")
        return 1

    phase4_5 = run_phase4_5(progress_dir, logger)
    logger.info("Phase 4-5 result:")
    logger.info(json.dumps(phase4_5, indent=2, ensure_ascii=False))

    return 0 if phase4_5["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
