import logging

from app.services.reconciliation_service import (
    load_canonical_chunks,
    reconcile_incomplete_operations,
)

logger = logging.getLogger(__name__)


def main() -> int:
    """Reconcile incomplete document-indexing operations."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        outcomes = reconcile_incomplete_operations(
            load_canonical_chunks=load_canonical_chunks,
        )
    except Exception:
        logger.exception("Unable to run indexing reconciliation.")
        return 1

    recovered_count = 0
    superseded_count = 0
    failed_count = 0

    for outcome in outcomes:
        if outcome.recovered:
            recovered_count += 1
            logger.info(
                "Recovered operation %s for document %s",
                outcome.operation_id,
                outcome.document_id,
            )
        elif outcome.superseded:
            superseded_count += 1
            logger.warning(
                "Superseded operation %s for document %s: %s",
                outcome.operation_id,
                outcome.document_id,
                outcome.message,
            )
        else:
            failed_count += 1
            logger.error(
                "Recovery failed for operation %s, document %s: %s",
                outcome.operation_id,
                outcome.document_id,
                outcome.message,
            )

    logger.info(
        "Reconciliation summary: recovered=%d superseded=%d failed=%d",
        recovered_count,
        superseded_count,
        failed_count,
    )

    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())