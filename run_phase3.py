#!/usr/bin/env python3
"""Phase 3 execution: Cluster speakers using embeddings."""

import json
import logging
from pathlib import Path

from speaker_clustering import run_phase3

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    output_dir = Path("output/_progress")
    embeddings_path = output_dir / "phase2_embeddings.json"

    logger.info("Starting Phase 3: Speaker Clustering")
    logger.info(f"Input: {embeddings_path}")

    # Check if embeddings exist
    if not embeddings_path.exists():
        logger.error(f"❌ Embeddings file not found: {embeddings_path}")
        logger.error("   Please run Phase 2 first: python run_phase2.py")
        return 1

    # Run Phase 3
    result = run_phase3(embeddings_path, output_dir, logger)

    logger.info(f"\nPhase 3 result:")
    logger.info(json.dumps(result, indent=2))

    if result["status"] == "success":
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 3 SUMMARY")
        logger.info("=" * 70)
        logger.info(f"  • Speaker instances clustered: {result['total_speaker_instances']}")
        logger.info(f"  • Unique cluster groups: {result['unique_clusters']}")
        logger.info(f"  • Unique persons identified: {result['unique_persons']}")
        logger.info(f"  • Export: {result['export_path']}")
        logger.info("\nPhase 3 COMPLETE - Ready for Phase 4 (person linking)")
        return 0
    else:
        logger.error(f"Phase 3 failed: {result.get('message', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    exit(main())
