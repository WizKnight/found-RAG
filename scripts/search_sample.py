from app.retrieval.dense import dense_search


def main() -> None:
    query = "How many annual leave days do employees receive?"

    results = dense_search(
        query=query,
        top_k=5,
    )

    for index, result in enumerate(results, start=1):
        print("=" * 80)
        print(f"Rank: {index}")
        print(f"Score: {result['score']}")
        print(f"Chunk ID: {result['id']}")
        print()
        print(result["payload"]["text"])


if __name__ == "__main__":
    main()