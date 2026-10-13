from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from app.storage.vector_store import VectorStore


class QdrantVectorStore(VectorStore):
    """Qdrant-backed vector store."""

    def __init__(
        self,
        url: str,
        collection_name: str,
        api_key: str | None = None,
    ) -> None:
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
        )
        self.collection_name = collection_name

    def create_collection(self, vector_size: int) -> None:
        """Create the collection if it does not already exist."""
        if vector_size <= 0:
            raise ValueError("vector_size must be greater than zero.")

        collections = self.client.get_collections()
        exists = any(
            collection.name == self.collection_name
            for collection in collections.collections
        )

        if exists:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    def upsert(
        self,
        ids: list[UUID],
        vectors: list[list[float]],
        payloads: list[dict],
    ) -> None:
        """Insert or update points in the collection."""
        if not (len(ids) == len(vectors) == len(payloads)):
            raise ValueError(
                "ids, vectors, and payloads must have the same length."
            )

        if len(set(ids)) != len(ids):
            raise ValueError("Point IDs must be unique within an upsert.")

        if not ids:
            return

        points = [
            PointStruct(
                id=str(chunk_id),
                vector=vector,
                payload=payload,
            )
            for chunk_id, vector, payload in zip(
                ids,
                vectors,
                payloads,
                strict=True,
            )
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def search(
        self,
        vector: list[float],
        limit: int,
    ) -> list[dict]:
        """Search for the nearest vector points."""
        if limit <= 0:
            return []

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
        ).points

        return [
            {
                "id": result.id,
                "score": result.score,
                "payload": result.payload,
            }
            for result in results
        ]

    def delete_points(self, ids: list[UUID]) -> None:
        """Delete specific points without affecting other chunks."""
        if not ids:
            return

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(
                points=[str(point_id) for point_id in ids],
            ),
            wait=True,
        )

    def delete_document(self, document_id: UUID) -> None:
        """Delete all points belonging to a document."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=str(document_id)),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def get_document_chunk_ids(
        self,
        document_id: UUID,
    ) -> set[UUID]:
        """Read every Qdrant point belonging to a document."""
        chunk_ids: set[UUID] = set()
        offset = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=str(document_id)),
                        )
                    ]
                ),
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )

            for point in points:
                chunk_ids.add(UUID(str(point.id)))

            if next_offset is None:
                break

            offset = next_offset

        return chunk_ids