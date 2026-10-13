
from uuid import UUID

from pydantic import BaseModel, Field


class IndexConsistencyReport(BaseModel):
    document_id: UUID
    expected_chunk_ids: set[UUID] = Field(default_factory=set)
    bm25_chunk_ids: set[UUID] = Field(default_factory=set)
    qdrant_chunk_ids: set[UUID] = Field(default_factory=set)

    bm25_missing_ids: set[UUID] = Field(default_factory=set)
    bm25_extra_ids: set[UUID] = Field(default_factory=set)
    qdrant_missing_ids: set[UUID] = Field(default_factory=set)
    qdrant_extra_ids: set[UUID] = Field(default_factory=set)

    @property
    def is_consistent(self) -> bool:
        return not any(
            (
                self.bm25_missing_ids,
                self.bm25_extra_ids,
                self.qdrant_missing_ids,
                self.qdrant_extra_ids,
            )
        )
