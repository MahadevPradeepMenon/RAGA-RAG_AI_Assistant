def chunk_text(text: str, chunk_size: int = 80, overlap: int = 20) -> list[str]:
    """
    Split text into word-based chunks with overlap.

    chunk_size = max number of words per chunk
    overlap = number of words repeated between chunks
    """
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def chunk_documents(documents: list[dict], chunk_size: int = 80, overlap: int = 20) -> list[dict]:
    """
    Convert loaded documents into chunks with metadata.

    Returns:
    [
        {
            "chunk_id": "vacation_policy.txt_chunk_1",
            "source": "vacation_policy.txt",
            "text": "Employees are entitled..."
        }
    ]
    """
    all_chunks = []

    for document in documents:
        source = document["source"]
        text = document["text"]

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        for index, chunk in enumerate(chunks, start=1):
            all_chunks.append({
                "chunk_id": f"{source}_chunk_{index}",
                "source": source,
                "text": chunk
            })

    return all_chunks