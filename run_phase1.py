#!/usr/bin/env python3
"""Phase 1 execution: Parse filenames and build call records from 1,073 files."""

import json
import logging
from pathlib import Path

from speaker_identification import SpeakerIdentificationEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    # Paths
    manifest_path = Path("output/_progress/manifest.json")
    output_dir = Path("output/_progress")

    # Initialize engine
    logger.info("Initializing Speaker Identification Engine...")
    engine = SpeakerIdentificationEngine(manifest_path, output_dir, logger)

    # Run Phase 1
    result = engine.run_phase1()

    logger.info(f"Phase 1 result: {json.dumps(result, indent=2)}")

    if result["status"] == "success":
        logger.info("="*60)
        logger.info("PHASE 1 COMPLETE")
        logger.info("="*60)
        logger.info(f"  • Call records: {result['call_records_count']}")
        logger.info(f"  • Unique originating phones: {result['unique_originating_phones']}")
        logger.info(f"  • Unique contact phones: {result['unique_contact_phones']}")
        logger.info(f"  • Interim export: {result['interim_export']}")
        return 0
    else:
        logger.error(f"Phase 1 failed: {result.get('message', 'Unknown error')}")
        return 1

if __name__ == "__main__":
    exit(main())
