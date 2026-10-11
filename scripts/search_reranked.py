from pathlib import Path

from app.ingestion.pipeline import ingest_document
from app.services.bm25_service import (
    index_chunks as index_bm25_chunks,
)
from app.services.indexing_service import (
    index_chunks as index_dense_chunks,
)
from app.services.retrieval_pipeline_service import (
    get_retrieval_pipeline,
)


SAMPLE_PATH = Path("data/raw/sample.txt")


def main() -> None:
    result = ingest_document(SAMPLE_PATH)

    print(f"Document: {result.document.filename}")
    print(f"Chunks: {result.chunk_count}")

    index_dense_chunks(result.chunks)
    index_bm25_chunks(result.chunks)

    pipeline = get_retrieval_pipeline()

    query = "What is the maximum carry-forward allowance?"

    results = pipeline.search(query)

    print()
    print("=" * 80)
    print("FINAL RERANKED RESULTS")
    print("=" * 80)

    for result in results:
        print(f"Rank: {result.rank}")
        print(f"Score: {result.score:.6f}")
        print(f"Method: {result.retrieval_method}")
        print(f"Chunk ID: {result.chunk_id}")
        print()
        print(result.text)
        print("=" * 80)


if __name__ == "__main__":
    main()