from pathlib import Path

from app.ingestion.pipeline import ingest_document


def main() -> None:
    path = Path("data/raw/sample.txt")

    result = ingest_document(path)

    print(f"Document: {result.document.filename}")
    print(f"Type: {result.document.document_type}")
    print(f"Hash: {result.document.content_hash}")
    print(f"Chunks: {result.chunk_count}")
    print()

    for chunk in result.chunks:
        print("=" * 80)
        print(f"Chunk: {chunk.chunk_index}")
        print(f"Page: {chunk.page_number}")
        print(f"ID: {chunk.id}")
        print()
        print(chunk.text)


if __name__ == "__main__":
    main()