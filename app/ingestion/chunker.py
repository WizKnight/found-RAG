from uuid import UUID

import tiktoken

from app.models.chunk import Chunk


class TextChunker:
    """Create overlapping, paragraph-aware chunks."""

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def chunk(
        self,
        text: str,
        document_id: UUID,
        page_number: int | None = None,
        section: str | None = None,
    ) -> list[Chunk]:
        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

        chunks: list[Chunk] = []

        current_tokens: list[int] = []

        for paragraph in paragraphs:
            paragraph_tokens = self.encoder.encode(paragraph)

            if (
                current_tokens
                and len(current_tokens) + len(paragraph_tokens)
                > self.chunk_size
            ):
                chunks.append(
                    self._create_chunk(
                        current_tokens,
                        document_id,
                        len(chunks),
                        page_number,
                        section,
                    )
                )

                overlap_start = max(
                    0,
                    len(current_tokens) - self.chunk_overlap,
                )

                current_tokens = current_tokens[overlap_start:]

            current_tokens.extend(paragraph_tokens)

        if current_tokens:
            chunks.append(
                self._create_chunk(
                    current_tokens,
                    document_id,
                    len(chunks),
                    page_number,
                    section,
                )
            )

        return chunks

    def _create_chunk(
        self,
        tokens: list[int],
        document_id: UUID,
        chunk_index: int,
        page_number: int | None,
        section: str | None,
    ) -> Chunk:
        text = self.encoder.decode(tokens)

        return Chunk(
            document_id=document_id,
            text=text.strip(),
            chunk_index=chunk_index,
            page_number=page_number,
            section=section,
        )