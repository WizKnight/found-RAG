
import json
from pathlib import Path
from uuid import UUID

from app.models.chunk import Chunk


class BM25IndexStore:
    """Persist canonical chunk records used to rebuild the BM25 index."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load_chunks(self) -> list[Chunk]:
        if not self.path.exists():
            return []

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise TypeError("The BM25 corpus must be a JSON list.")

            return [Chunk.model_validate(item) for item in data]
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise RuntimeError(
                f"Unable to load BM25 corpus from {self.path}. "
                "The file may be corrupted; repair or restore it "
                "before rebuilding the index."
            ) from exc

    def save_chunks(self, chunks: list[Chunk]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Stable ordering makes the persisted corpus easier to inspect
        # and compare between runs.
        ordered_chunks = sorted(
            chunks,
            key=lambda chunk: (
                str(chunk.document_id),
                chunk.chunk_index,
                str(chunk.id),
            ),
        )

        serialized = [
            chunk.model_dump(mode="json")
            for chunk in ordered_chunks
        ]

        temporary_path = self.path.with_name(
            f"{self.path.name}.tmp"
        )

        try:
            temporary_path.write_text(
                json.dumps(
                    serialized,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            temporary_path.replace(self.path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def upsert_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Insert or replace chunks by chunk ID, then persist."""
        existing = {
            chunk.id: chunk
            for chunk in self.load_chunks()
        }

        for chunk in chunks:
            existing[chunk.id] = chunk

        merged = list(existing.values())
        self.save_chunks(merged)
        return merged

    def delete_document(self, document_id: UUID) -> list[Chunk]:
        """Remove every persisted chunk belonging to a document."""
        remaining = [
            chunk
            for chunk in self.load_chunks()
            if chunk.document_id != document_id
        ]

        self.save_chunks(remaining)
        return remaining

    
    def replace_document(
        self,
        document_id: UUID,
        chunks: list[Chunk],
    ) -> list[Chunk]:
        """Atomically replace the persisted chunks for one document."""
        if any(chunk.document_id != document_id for chunk in chunks):
            raise ValueError(
                "All replacement chunks must belong to the target document."
            )

        existing = {
            chunk.id: chunk
            for chunk in self.load_chunks()
            if chunk.document_id != document_id
        }

        for chunk in chunks:
            existing[chunk.id] = chunk

        merged = list(existing.values())
        self.save_chunks(merged)
        return merged

