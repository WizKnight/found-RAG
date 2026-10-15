
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class IndexingOperationStatus(StrEnum):
    PENDING = "pending"
    QDRANT_UPDATED = "qdrant_updated"
    BM25_UPDATED = "bm25_updated"
    COMPLETED = "completed"
    NEEDS_RECONCILIATION = "needs_reconciliation"
    SUPERSEDED = "superseded"


class IndexingOperation(BaseModel):
    operation_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    chunk_ids: list[UUID]

    status: IndexingOperationStatus = IndexingOperationStatus.PENDING
    attempts: int = Field(default=1, ge=1)

    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
