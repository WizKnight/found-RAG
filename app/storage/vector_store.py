from abc import ABC, abstractmethod
from uuid import UUID


class VectorStore(ABC):
    """Interface for vector storage."""

    @abstractmethod
    def create_collection(self, vector_size: int) -> None:
        """Create the vector collection if necessary."""
        raise NotImplementedError

    @abstractmethod
    def upsert(
        self,
        ids: list[UUID],
        vectors: list[list[float]],
        payloads: list[dict],
    ) -> None:
        """Insert or update vectors."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        vector: list[float],
        limit: int,
    ) -> list[dict]:
        """Search for similar vectors."""
        raise NotImplementedError