from src.loader import load_documents
from src.chunker import chunk_documents
from src.retriever import KeywordRetriever


def build_pipeline():
    """
    Load documents, chunk them, and create the retriever.
    """
    documents = load_documents("data/uploads")
    chunks = chunk_documents(documents)

    retriever = KeywordRetriever(chunks)

    return retriever, chunks


def print_results(question: str, results: list[dict]):
    """
    Display retrieved chunks in a readable way.
    """
    print("\nQUESTION:")
    print(question)

    print("\nTOP MATCHES:")

    for index, result in enumerate(results, start=1):
        print("\n" + "=" * 60)
        print(f"Result {index}")
        print(f"Source: {result['source']}")
        print(f"Score: {result['score']:.2f}")
        print("-" * 60)
        print(result["text"])


def main():
    print("Company Knowledge Assistant - V1")
    print("Type 'exit' to quit.\n")

    retriever, chunks = build_pipeline()

    print(f"Loaded {len(chunks)} chunks from company documents.\n")

    while True:
        question = input("Ask a question: ")

        if question.lower().strip() == "exit":
            print("Goodbye.")
            break

        results = retriever.search(question, top_k=3)
        print_results(question, results)


if __name__ == "__main__":
    main()