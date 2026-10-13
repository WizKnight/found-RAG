import json
import os
import tempfile
from pathlib import Path
from threading import RLock
from uuid import UUID

from app.models.chunk import Chunk


class CanonicalChunkStore:
    """Persistent source of truth for document chunks.

    This JSON implementation is intended for single-process development.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = RLock()

    def load_chunks(self) -> list[Chunk]:
        """Load all canonical chunks."""
        with self._lock:
            return self._read_all()

    def get_document_chunks(self, document_id: UUID) -> list[Chunk]:
        """Return all canonical chunks belonging to a document."""
        with self._lock:
            chunks = [
                chunk
                for chunk in self._read_all()
                if chunk.document_id == document_id
            ]

        return sorted(
            chunks,
            key=lambda chunk: (chunk.chunk_index, str(chunk.id)),
        )

    def get_chunks_by_ids(
        self,
        document_id: UUID,
        chunk_ids: set[UUID],
    ) -> list[Chunk]:
        """Load exact chunk IDs for a document or fail explicitly."""
        if not chunk_ids:
            return []

        chunks = self.get_document_chunks(document_id)
        chunks_by_id = {chunk.id: chunk for chunk in chunks}

        missing_ids = chunk_ids - chunks_by_id.keys()

        if missing_ids:
            raise ValueError(
                "Canonical chunk repository is missing requested IDs: "
                f"{sorted(str(chunk_id) for chunk_id in missing_ids)}"
            )

        return [chunks_by_id[chunk_id] for chunk_id in sorted(
            chunk_ids, key=str
        )]

    def replace_document(
        self,
        document_id: UUID,
        chunks: list[Chunk],
    ) -> list[Chunk]:
        """Atomically replace a document's canonical chunk set."""
        if any(chunk.document_id != document_id for chunk in chunks):
            raise ValueError(
                "All replacement chunks must belong to the target document."
            )

        chunk_ids = [chunk.id for chunk in chunks]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(
                "Replacement chunks must have unique IDs."
            )

        with self._lock:
            existing_chunks = self._read_all()

            # Preserve chunks from every other document.
            merged = {
                chunk.id: chunk
                for chunk in existing_chunks
                if chunk.document_id != document_id
            }

            # Replace this document's entire canonical chunk set.
            for chunk in chunks:
                merged[chunk.id] = chunk

            updated_chunks = sorted(
                merged.values(),
                key=lambda chunk: (
                    str(chunk.document_id),
                    chunk.chunk_index,
                    str(chunk.id),
                ),
            )

            self._write_all(updated_chunks)
            return updated_chunks

    def delete_document(self, document_id: UUID) -> list[Chunk]:
        """Remove a document's canonical chunks."""
        with self._lock:
            remaining_chunks = [
                chunk
                for chunk in self._read_all()
                if chunk.document_id != document_id
            ]

            self._write_all(remaining_chunks)
            return remaining_chunks

    def _read_all(self) -> list[Chunk]:
        if not self.path.exists():
            return []

        with self.path.open("r", encoding="utf-8") as file:
            raw_chunks = json.load(file)

        if not isinstance(raw_chunks, list):
            raise TypeError(
                "Canonical chunk file must contain a JSON list."
            )

        return [Chunk.model_validate(item) for item in raw_chunks]

    def _write_all(self, chunks: list[Chunk]) -> None:
        """Write atomically so a failed write does not truncate the index."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: str | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = temp_file.name

                json.dump(
                    [
                        chunk.model_dump(mode="json")
                        for chunk in chunks
                    ],
                    temp_file,
                    ensure_ascii=False,
                    indent=2,
                )

                temp_file.flush()
                os.fsync(temp_file.fileno())

            os.replace(temp_path, self.path)
            temp_path = None

        finally:
            if temp_path is not None and os.path.exists(temp_path):
                os.unlink(temp_path)