from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
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
            )
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        vector: list[float],
        limit: int,
    ) -> list[dict]:
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