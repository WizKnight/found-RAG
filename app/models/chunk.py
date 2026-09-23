from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A retrievable piece of a source document."""

    id: UUID = Field(default_factory=uuid4)

    document_id: UUID

    text: str

    chunk_index: int

    page_number: int | None = None

    section: str | None = None

    start_char: int | None = None
    end_char: int | None = None

    metadata: dict[str, str] = Field(default_factory=dict)