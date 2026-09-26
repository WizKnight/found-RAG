from pathlib import Path

from app.ingestion.pipeline import ingest_document
from app.services.bm25_service import (
    bm25_search,
    index_chunks,
)


SAMPLE_PATH = Path("data/raw/sample.txt")


def main() -> None:
    result = ingest_document(SAMPLE_PATH)

    print(f"Document: {result.document.filename}")
    print(f"Chunks: {result.chunk_count}")

    index_chunks(result.chunks)

    query = "How many paid anuual leave?"

    print("=" * 80)
    print(f"Query: {query}")
    print("=" * 80)

    results = bm25_search(
        query=query,
        top_k=5,
    )

    for result in results:
        print(f"Rank: {result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Chunk ID: {result['chunk_id']}")
        print()
        print(result["text"])
        print("=" * 80)


if __name__ == "__main__":
    main()