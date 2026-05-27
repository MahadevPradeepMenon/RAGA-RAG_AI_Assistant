from src.loader import load_documents
from src.chunker import chunk_documents
from src.retriever import KeywordRetriever
from src.vector_retriever import VectorRetriever


def print_results(title: str, results: list[dict]):
    print("\n" + title)
    print("-" * 70)

    for index, result in enumerate(results, start=1):
        print(f"\nRank {index}")
        print(f"Source: {result['source']}")
        print(f"Score: {result['score']:.4f}")
        print(result["text"])


def compare_query(query: str, keyword_retriever, vector_retriever):
    print("\n" + "=" * 80)
    print("QUERY:")
    print(query)

    keyword_results = keyword_retriever.search(query, top_k=3)
    vector_results = vector_retriever.search(query, top_k=3)

    print_results("BM25 KEYWORD RESULTS", keyword_results)
    print_results("VECTOR SEMANTIC RESULTS", vector_results)


def main():
    documents = load_documents("data/uploads")
    chunks = chunk_documents(documents)

    keyword_retriever = KeywordRetriever(chunks)
    vector_retriever = VectorRetriever(chunks)

    test_queries = [
        "How many annual leave days do employees get?",
        "Can employees work remotely?",
        "Do I need VPN access?",
        "How do I reset my password?",
        "Can I work from home?",
    ]

    for query in test_queries:
        compare_query(query, keyword_retriever, vector_retriever)


if __name__ == "__main__":
    main()