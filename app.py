from src.loader import load_documents
from src.chunker import chunk_documents
from src.retriever import KeywordRetriever
from src.answerer import generate_basic_answer


def build_pipeline():
    """
    Load documents, chunk them, and create the retriever.
    """
    documents = load_documents("data/uploads")
    chunks = chunk_documents(documents)

    retriever = KeywordRetriever(chunks)

    return retriever, chunks


def print_answer(question: str, answer_data: dict, results: list[dict]):
    """
    Display the generated answer and sources.
    """
    print("\n" + "=" * 60)
    print("QUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer_data["answer"])

    print("\nSOURCE:")
    if answer_data["source"]:
        print(answer_data["source"])
    else:
        print("No source found")

    print(f"\nCONFIDENCE SCORE: {answer_data['confidence']:.2f}")

    print("\nRETRIEVED EVIDENCE:")
    for index, result in enumerate(results, start=1):
        print("\n" + "-" * 60)
        print(f"Evidence {index}")
        print(f"Source: {result['source']}")
        print(f"Score: {result['score']:.2f}")
        print(result["text"])


def main():
    print("Company Knowledge Assistant - V1.1")
    print("Type 'exit' to quit.\n")

    retriever, chunks = build_pipeline()

    print(f"Loaded {len(chunks)} chunks from company documents.\n")

    while True:
        question = input("Ask a question: ")

        if question.lower().strip() == "exit":
            print("Goodbye.")
            break

        results = retriever.search(question, top_k=3)
        answer_data = generate_basic_answer(results)

        print_answer(question, answer_data, results)


if __name__ == "__main__":
    main()