from pydantic import BaseModel, Field


class Citation(BaseModel):
    citation_id: str
    chunk_id: str
    document_id: str | None = None
    page_number: int | None = None
    section: str | None = None
    text: str


class GeneratedAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)