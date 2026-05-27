from src.loader import load_documents
from src.chunker import chunk_documents
from src.hybrid_retriever import HybridRetriever
from src.local_answerer import generate_local_answer


def build_pipeline():
    """
    Load documents, chunk them, and create the hybrid retriever.
    """
    documents = load_documents("data/uploads")
    chunks = chunk_documents(documents)

    retriever = HybridRetriever(chunks)

    return retriever, chunks


def print_answer(question: str, answer_data: dict, results: list[dict]):
    """
    Display the locally generated answer and supporting evidence.
    """
    print("\n" + "=" * 70)
    print("QUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer_data["answer"])

    print("\nSOURCES USED:")
    if answer_data["sources"]:
        for source in answer_data["sources"]:
            print(f"- {source}")
    else:
        print("No sources found")

    print(f"\nCONFIDENCE SCORE: {answer_data['confidence']:.2f}")

    print("\nRETRIEVED EVIDENCE:")
    for index, result in enumerate(results, start=1):
        print("\n" + "-" * 70)
        print(f"Evidence {index}")
        print(f"Source: {result['source']}")
        print(f"Hybrid Score: {result['hybrid_score']:.2f}")
        print(f"Keyword Score: {result['keyword_score']}")
        print(f"Vector Score: {result['vector_score']}")
        print(result["text"])


def main():
    print("Company Knowledge Assistant - V4 Free Local RAG")
    print("Type 'exit' to quit.\n")

    retriever, chunks = build_pipeline()

    print(f"Loaded {len(chunks)} chunks from company documents.\n")

    while True:
        question = input("Ask a question: ")

        if question.lower().strip() == "exit":
            print("Goodbye.")
            break

        results = retriever.search(question, top_k=3)
        answer_data = generate_local_answer(question, results)

        print_answer(question, answer_data, results)


if __name__ == "__main__":
    main()