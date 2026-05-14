#!/usr/bin/env python3
"""Phase 4-5 execution: build final persons.json."""

import json
import logging
from pathlib import Path

from person_linking import run_phase4_5


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    logger.info("Starting Phase 4-5: Person Linking and Final Export")
    result = run_phase4_5(Path("output/_progress"), logger)

    logger.info("\nPhase 4-5 result:")
    logger.info(json.dumps(result, indent=2, ensure_ascii=False))

    if result["status"] == "success":
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 4-5 SUMMARY")
        logger.info("=" * 70)
        logger.info("  * Calls linked: %s", result["total_calls"])
        logger.info("  * Phone persons: %s", result["phone_persons"])
        logger.info("  * Voice persons: %s", result["voice_persons"])
        logger.info("  * Total person records: %s", result["total_person_records"])
        logger.info("  * Export: %s", result["export_path"])
        return 0

    logger.error("Phase 4-5 failed: %s", result.get("message", "Unknown error"))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
