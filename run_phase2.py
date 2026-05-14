#!/usr/bin/env python3
"""Phase 2 execution: Extract speaker embeddings from all WhisperX outputs."""

import json
import logging
from pathlib import Path

from embedding_extractor import run_phase2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    output_dir = Path("output")

    # Run Phase 2
    logger.info("Starting Phase 2: Speaker Embedding Extraction")
    result = run_phase2(output_dir, logger)

    logger.info(f"\nPhase 2 result:")
    logger.info(json.dumps(result, indent=2))

    if result["status"] in ("success", "warning"):
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 2 SUMMARY")
        logger.info("=" * 70)
        logger.info(f"  • Files processed: {result['files_processed']}")
        logger.info(f"  • Files without embeddings: {result['files_without_embeddings']}")
        logger.info(f"  • Total embeddings extracted: {result['total_embeddings']}")
        logger.info(f"  • Export: {result['export_path']}")
        
        if result['status'] == 'warning':
            logger.warning("\n⚠️  WARNING: Some/all files do not have speaker embeddings in JSON")
            logger.warning("   This may be because:")
            logger.warning("   1. Diarization pipeline was not run with return_embeddings=True")
            logger.warning("   2. Embeddings were computed but not saved to JSON")
            logger.warning("\n   NEXT STEP: Re-run diarization pipeline with updated code")
            logger.warning("   to extract and save embeddings from pyannote")
            return 1

        return 0
    else:
        logger.error(f"Phase 2 failed: {result.get('message', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    exit(main())
