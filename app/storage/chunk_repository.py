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
    Chunk data and per-document version history are persisted atomically
    in the same file.
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = RLock()

    def load_chunks(self) -> list[Chunk]:
        """Load all canonical chunks."""
        with self._lock:
            chunks, _ = self._read_state()
            return chunks

    def get_document_chunks(self, document_id: UUID) -> list[Chunk]:
        """Return all canonical chunks belonging to a document."""
        with self._lock:
            chunks, _ = self._read_state()
            document_chunks = [
                chunk
                for chunk in chunks
                if chunk.document_id == document_id
            ]

        return self._sort_document_chunks(document_chunks)

    def get_document_version(self, document_id: UUID) -> int:
        """Return the current version, or zero if never recorded."""
        with self._lock:
            _, versions = self._read_state()
            return versions.get(str(document_id), 0)

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

        return [
            chunks_by_id[chunk_id]
            for chunk_id in sorted(chunk_ids, key=str)
        ]

    def replace_document(
        self,
        document_id: UUID,
        chunks: list[Chunk],
    ) -> list[Chunk]:
        """Replace a document's chunks and increment its version."""
        self._validate_replacement(document_id, chunks)

        with self._lock:
            existing_chunks, versions = self._read_state()
            next_version = versions.get(str(document_id), 0) + 1

            updated_chunks = self._merge_document_chunks(
                existing_chunks,
                document_id,
                chunks,
            )
            versions[str(document_id)] = next_version

            self._write_state(updated_chunks, versions)
            return updated_chunks

    def replace_document_at_version(
        self,
        document_id: UUID,
        chunks: list[Chunk],
        version: int,
    ) -> list[Chunk]:
        """Apply a specific version, allowing safe retries of that version.

        Rejects stale versions and version gaps. Reapplying the current
        version succeeds only when its canonical chunks are identical.
        """
        self._validate_replacement(document_id, chunks)

        if version < 1:
            raise ValueError("Document version must be positive.")

        with self._lock:
            existing_chunks, versions = self._read_state()
            current_version = versions.get(str(document_id), 0)
            current_chunks = self._sort_document_chunks(
                chunk
                for chunk in existing_chunks
                if chunk.document_id == document_id
            )
            requested_chunks = self._sort_document_chunks(chunks)

            if version == current_version:
                if current_chunks != requested_chunks:
                    raise ValueError(
                        "Cannot change canonical chunks without "
                        "advancing the document version."
                    )
                return existing_chunks

            if version < current_version:
                raise ValueError(
                    f"Stale document version {version}; "
                    f"current version is {current_version}."
                )

            if version != current_version + 1:
                raise ValueError(
                    f"Document version must advance from "
                    f"{current_version} to {current_version + 1}; "
                    f"received {version}."
                )

            updated_chunks = self._merge_document_chunks(
                existing_chunks,
                document_id,
                chunks,
            )
            versions[str(document_id)] = version

            self._write_state(updated_chunks, versions)
            return updated_chunks

    def delete_document(self, document_id: UUID) -> list[Chunk]:
        """Delete chunks while retaining an incremented version tombstone."""
        with self._lock:
            existing_chunks, versions = self._read_state()
            remaining_chunks = [
                chunk
                for chunk in existing_chunks
                if chunk.document_id != document_id
            ]

            # Keep version history even when no chunks remain. Otherwise,
            # an old operation could recreate a deleted document.
            versions[str(document_id)] = (
                versions.get(str(document_id), 0) + 1
            )

            self._write_state(remaining_chunks, versions)
            return remaining_chunks

    @staticmethod
    def _validate_replacement(
        document_id: UUID,
        chunks: list[Chunk],
    ) -> None:
        if any(chunk.document_id != document_id for chunk in chunks):
            raise ValueError(
                "All replacement chunks must belong to the target document."
            )

        chunk_ids = [chunk.id for chunk in chunks]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Replacement chunks must have unique IDs.")

    @staticmethod
    def _sort_document_chunks(chunks) -> list[Chunk]:
        return sorted(
            chunks,
            key=lambda chunk: (chunk.chunk_index, str(chunk.id)),
        )

    @staticmethod
    def _merge_document_chunks(
        existing_chunks: list[Chunk],
        document_id: UUID,
        replacement_chunks: list[Chunk],
    ) -> list[Chunk]:
        merged = {
            chunk.id: chunk
            for chunk in existing_chunks
            if chunk.document_id != document_id
        }

        for chunk in replacement_chunks:
            merged[chunk.id] = chunk

        return sorted(
            merged.values(),
            key=lambda chunk: (
                str(chunk.document_id),
                chunk.chunk_index,
                str(chunk.id),
            ),
        )

    def _read_state(self) -> tuple[list[Chunk], dict[str, int]]:
        """Read either legacy list-format or versioned canonical state."""
        if not self.path.exists():
            return [], {}

        with self.path.open("r", encoding="utf-8") as file:
            raw_state = json.load(file)

        # Backward compatibility with the original JSON list format.
        if isinstance(raw_state, list):
            chunks = [
                Chunk.model_validate(item)
                for item in raw_state
            ]
            versions: dict[str, int] = {}

            # Existing documents begin at version 1 after migration.
            for chunk in chunks:
                versions.setdefault(str(chunk.document_id), 1)

            return chunks, versions

        if not isinstance(raw_state, dict):
            raise TypeError(
                "Canonical chunk file must contain a JSON list "
                "or a versioned state object."
            )

        if raw_state.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError(
                "Unsupported canonical chunk schema version: "
                f"{raw_state.get('schema_version')!r}"
            )

        raw_chunks = raw_state.get("chunks")
        raw_versions = raw_state.get("document_versions")

        if not isinstance(raw_chunks, list):
            raise TypeError(
                "Canonical state 'chunks' must be a JSON list."
            )

        if not isinstance(raw_versions, dict):
            raise TypeError(
                "Canonical state 'document_versions' must be an object."
            )

        chunks = [
            Chunk.model_validate(item)
            for item in raw_chunks
        ]

        versions = {}
        for document_id, version in raw_versions.items():
            try:
                UUID(document_id)
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(
                    f"Invalid document ID in version history: {document_id!r}"
                ) from exc

            if type(version) is not int or version < 1:
                raise ValueError(
                    f"Invalid version for document {document_id!r}: "
                    f"{version!r}"
                )

            versions[document_id] = version

        for chunk in chunks:
            if str(chunk.document_id) not in versions:
                raise ValueError(
                    "Canonical state is missing version history for "
                    f"document {chunk.document_id}."
                )

        return chunks, versions

    def _write_state(
        self,
        chunks: list[Chunk],
        versions: dict[str, int],
    ) -> None:
        """Atomically persist chunks and version history together."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: str | None = None

        state = {
            "schema_version": self.SCHEMA_VERSION,
            "chunks": [
                chunk.model_dump(mode="json")
                for chunk in chunks
            ],
            "document_versions": dict(sorted(versions.items())),
        }

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
                    state,
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
