from src.loader import load_documents
from src.chunker import chunk_documents


def main():
    documents = load_documents("data/uploads")
    chunks = chunk_documents(documents)

    print(f"\nLoaded {len(documents)} document units")
    print(f"Created {len(chunks)} chunks\n")

    for chunk in chunks:
        print("=" * 80)
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Source: {chunk['source']}")
        print(f"File Type: {chunk['file_type']}")
        print(f"Location: {chunk['location']}")
        print(f"Strategy: {chunk['chunk_strategy']}")
        print(f"Word Count: {chunk['word_count']}")
        print("-" * 80)
        print(chunk["text"][:500])
        print()


if __name__ == "__main__":
    main()