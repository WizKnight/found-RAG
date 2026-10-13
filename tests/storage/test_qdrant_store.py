
from uuid import uuid4

from app.storage.qdrant_store import QdrantVectorStore


class FakeScrollClient:
    def __init__(self, pages):
        self.pages = iter(pages)

    def scroll(self, **kwargs):
        return next(self.pages)


class FakePoint:
    def __init__(self, point_id):
        self.id = point_id


def test_get_document_chunk_ids_reads_all_pages() -> None:
    first_id = uuid4()
    second_id = uuid4()

    store = QdrantVectorStore.__new__(QdrantVectorStore)
    object.__setattr__(store, "collection_name", "test_collection")
    object.__setattr__(store, "client", FakeScrollClient([
        ([FakePoint(str(first_id))], "next-page"),
        ([FakePoint(str(second_id))], None),
    ]))

    result = store.get_document_chunk_ids(uuid4())

    assert result == {first_id, second_id}
