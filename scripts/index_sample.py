from pathlib import Path

from app.ingestion.pipeline import ingest_document
from app.services.indexing_service import index_chunks


def main() -> None:
    path = Path("data/raw/sample.txt")

    result = ingest_document(path)

    print(f"Document: {result.document.filename}")
    print(f"Chunks: {result.chunk_count}")

    index_chunks(result.chunks)

    print("Indexing complete.")


if __name__ == "__main__":
    main()