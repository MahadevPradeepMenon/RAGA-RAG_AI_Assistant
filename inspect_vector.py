from src.loader import load_documents
from src.chunker import chunk_documents
from src.vector_retriever import VectorRetriever


def inspect_query(query: str):
    documents = load_documents("data/uploads")
    chunks = chunk_documents(documents)

    retriever = VectorRetriever(chunks)

    print("\n" + "=" * 70)
    print("QUERY:")
    print(query)

    results = retriever.search(query, top_k=5)

    print("\nTOP VECTOR RESULTS:")

    for index, result in enumerate(results, start=1):
        print("\n" + "-" * 70)
        print(f"Rank: {index}")
        print(f"Source: {result['source']}")
        print(f"Score: {result['score']:.4f}")
        print("Text:")
        print(result["text"])


if __name__ == "__main__":
    test_queries = [
        "How many annual leave days do employees get?",
        "Can employees work remotely?",
        "Do I need VPN access?",
        "How do I reset my password?",
        "Can I work from home?",
    ]

    for query in test_queries:
        inspect_query(query)