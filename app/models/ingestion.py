from pydantic import BaseModel

from app.models.chunk import Chunk
from app.models.document import Document


class IngestionResult(BaseModel):
    """Result produced by the document ingestion pipeline."""

    document: Document
    chunks: list[Chunk]

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)