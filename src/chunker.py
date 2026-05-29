def chunk_text(text: str, chunk_size: int = 80, overlap: int = 20) -> list[str]:
    """
    Split text into word-based chunks with overlap.
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


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 80,
    overlap: int = 20
) -> list[dict]:
    """
    Convert loaded documents into chunks with metadata.

    Keeps source file and location metadata.
    For PPTX, location can be "Slide 1", "Slide 2", etc.
    """
    all_chunks = []

    for document in documents:
        source = document["source"]
        text = document["text"]
        file_type = document.get("file_type")
        location = document.get("location")

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        for index, chunk in enumerate(chunks, start=1):
            location_part = f"_{location.replace(' ', '_')}" if location else ""

            all_chunks.append({
                "chunk_id": f"{source}{location_part}_chunk_{index}",
                "source": source,
                "file_type": file_type,
                "location": location,
                "text": chunk,
            })

    return all_chunks