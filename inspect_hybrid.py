from src.loader import load_documents
from src.chunker import chunk_documents
from src.hybrid_retriever import HybridRetriever


def inspect_query(query: str):
    documents = load_documents("data/uploads")
    chunks = chunk_documents(documents)

    retriever = HybridRetriever(chunks)

    print("\n" + "=" * 80)
    print("QUERY:")
    print(query)

    results = retriever.search(query, top_k=5)

    print("\nTOP HYBRID RESULTS:")

    for index, result in enumerate(results, start=1):
        print("\n" + "-" * 80)
        print(f"Rank: {index}")
        print(f"Source: {result['source']}")
        print(f"Hybrid Score: {result['hybrid_score']:.4f}")

        keyword_score = result["keyword_score"]
        vector_score = result["vector_score"]

        print(f"Keyword Score: {keyword_score if keyword_score is not None else 'Not found by keyword'}")
        print(f"Vector Score: {vector_score if vector_score is not None else 'Not found by vector'}")

        print("\nText:")
        print(result["text"])


if __name__ == "__main__":
    test_queries = [
        "How many annual leave days do employees get?",
        "Can employees work remotely?",
        "Do I need VPN access?",
        "How do I reset my password?",
        "Can I work from home?",
        "Who do I contact if my laptop is broken?",
    ]

    for query in test_queries:
        inspect_query(query)