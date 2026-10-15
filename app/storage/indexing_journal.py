
import json
from pathlib import Path
from threading import RLock
from uuid import UUID

from app.models.indexing_operation import (
    IndexingOperation,
    IndexingOperationStatus,
    utc_now,
)


class IndexingJournal:
    """Persistent journal for document indexing operations."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = RLock()

    def _read_all(self) -> dict[str, IndexingOperation]:
        if not self.path.exists():
            return {}

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise TypeError("Journal root must be a JSON object.")

            operations = {
                key: IndexingOperation.model_validate(value)
                for key, value in data.items()
            }

            for key, operation in operations.items():
                if key != str(operation.operation_id):
                    raise ValueError(
                        "Journal key does not match operation ID."
                    )

            return operations

        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise RuntimeError(
                f"Unable to load indexing journal from {self.path}."
            ) from exc

    def _write_all(
        self,
        operations: dict[str, IndexingOperation],
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_name(f"{self.path.name}.tmp")

        try:
            serialized = {
                key: operation.model_dump(mode="json")
                for key, operation in operations.items()
            }

            temporary_path.write_text(
                json.dumps(serialized, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary_path.replace(self.path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def create(
        self,
        operation: IndexingOperation,
    ) -> IndexingOperation:
        with self._lock:
            operations = self._read_all()
            key = str(operation.operation_id)

            if key in operations:
                raise ValueError(
                    f"Operation already exists: {operation.operation_id}"
                )

            operations[key] = operation
            self._write_all(operations)
            return operation

    def get(self, operation_id: UUID) -> IndexingOperation:
        with self._lock:
            operations = self._read_all()
            operation = operations.get(str(operation_id))

            if operation is None:
                raise KeyError(
                    f"Indexing operation not found: {operation_id}"
                )

            return operation

    def update(
        self,
        operation_id: UUID,
        *,
        status: IndexingOperationStatus,
        error: str | None = None,
        increment_attempts: bool = False,
    ) -> IndexingOperation:
        with self._lock:
            operations = self._read_all()
            key = str(operation_id)
            current = operations.get(key)

            if current is None:
                raise KeyError(
                    f"Indexing operation not found: {operation_id}"
                )

            updated = current.model_copy(
                update={
                    "status": status,
                    "error": error,
                    "attempts": (
                        current.attempts + 1
                        if increment_attempts
                        else current.attempts
                    ),
                    "updated_at": utc_now(),
                }
            )

            operations[key] = updated
            self._write_all(operations)
            return updated

    def list_incomplete(self) -> list[IndexingOperation]:
        terminal_statuses = {
            IndexingOperationStatus.COMPLETED,
            IndexingOperationStatus.SUPERSEDED,
        }

        with self._lock:
            operations = self._read_all()

            return [
                operation
                for operation in operations.values()
                if operation.status not in terminal_statuses
            ]
