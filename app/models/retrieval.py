from uuid import UUID

from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    chunk_id: UUID
    text: str

    score: float
    rank: int

    retrieval_method: str

    document_id: UUID | None = None
    page_number: int | None = None
    section: str | None = None

    metadata: dict[str, str] = Field(default_factory=dict)