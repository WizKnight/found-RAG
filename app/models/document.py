from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DocumentType(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "markdown"
    HTML = "html"


class Document(BaseModel):
    """Represents an ingested source document."""

    id: UUID = Field(default_factory=uuid4)
    filename: str
    document_type: DocumentType
    source: str | None = None

    content_hash: str

    created_at: datetime = Field(default_factory=datetime.utcnow)

    metadata: dict[str, str] = Field(default_factory=dict)