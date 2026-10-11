from pathlib import Path

from app.ingestion.pipeline import ingest_document
from app.retrieval.dense import dense_search
from app.retrieval.rrf import ReciprocalRankFusion
from app.services.bm25_service import (
    bm25_search,
    index_chunks as index_bm25_chunks,
)
from app.services.indexing_service import (
    index_chunks as index_dense_chunks,
)


SAMPLE_PATH = Path("data/raw/sample.txt")


def print_results(
    title: str,
    results,
) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    for result in results:
        print(f"Rank: {result.rank}")
        print(f"Score: {result.score:.6f}")
        print(f"Method: {result.retrieval_method}")
        print(f"Chunk ID: {result.chunk_id}")
        print()
        print(result.text)
        print("-" * 80)


def main() -> None:
    result = ingest_document(SAMPLE_PATH)

    print(f"Document: {result.document.filename}")
    print(f"Chunks: {result.chunk_count}")

    index_dense_chunks(result.chunks)
    index_bm25_chunks(result.chunks)

    query = "What is the maximum carry-forward allowance?"

    dense_results = dense_search(
        query=query,
        top_k=20,
    )

    bm25_results = bm25_search(
        query=query,
        top_k=20,
    )

    fusion = ReciprocalRankFusion(k=60)

    hybrid_results = fusion.fuse(
        result_lists=[
            dense_results,
            bm25_results,
        ],
        top_k=30,
    )

    print_results(
        "DENSE RESULTS",
        dense_results,
    )

    print_results(
        "BM25 RESULTS",
        bm25_results,
    )

    print_results(
        "RRF HYBRID RESULTS",
        hybrid_results,
    )


if __name__ == "__main__":
    main()