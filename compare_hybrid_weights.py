from src.loader import load_documents
from src.chunker import chunk_documents
from src.hybrid_retriever import HybridRetriever


TEST_QUERIES = [
    "How many annual leave days do employees get?",
    "Can employees work from home?",
    "Do employees need VPN access?",
    "How do I reset my password?",
    "What does the onboarding document say?",
]


WEIGHT_SETTINGS = [
    {
        "name": "Balanced",
        "keyword_weight": 0.5,
        "vector_weight": 0.5,
    },
    {
        "name": "Keyword-heavy",
        "keyword_weight": 0.7,
        "vector_weight": 0.3,
    },
    {
        "name": "Vector-heavy",
        "keyword_weight": 0.3,
        "vector_weight": 0.7,
    },
]


def print_results(setting_name: str, query: str, results: list[dict]):
    print("\n" + "=" * 90)
    print(f"SETTING: {setting_name}")
    print(f"QUERY: {query}")
    print("-" * 90)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print(f"Source: {result['source']}")

        if result.get("location"):
            print(f"Location: {result['location']}")

        print(f"Hybrid Score: {result['hybrid_score']:.6f}")
        print(f"Keyword Rank: {result.get('keyword_rank')}")
        print(f"Vector Rank: {result.get('vector_rank')}")
        print(f"Keyword Score: {result.get('keyword_score')}")
        print(f"Vector Score: {result.get('vector_score')}")
        print("Text:")
        print(result["text"][:400])


def main():
    documents = load_documents("data/uploads")
    chunks = chunk_documents(documents)

    for query in TEST_QUERIES:
        for setting in WEIGHT_SETTINGS:
            retriever = HybridRetriever(
                chunks,
                keyword_weight=setting["keyword_weight"],
                vector_weight=setting["vector_weight"],
            )

            results = retriever.search(query, top_k=3, candidate_k=10)

            print_results(setting["name"], query, results)


if __name__ == "__main__":
    main()