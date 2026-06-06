def count_words(text: str) -> int:
    """
    Count words in a piece of text.
    """
    return len(text.split())


def split_into_paragraphs(text: str) -> list[str]:
    """
    Split text into paragraph-like sections.

    Works for:
    - PDF extracted text
    - DOCX paragraphs
    - text with line breaks
    """
    raw_paragraphs = text.replace("\r\n", "\n").split("\n")

    paragraphs = []

    for paragraph in raw_paragraphs:
        cleaned = paragraph.strip()

        if cleaned:
            paragraphs.append(cleaned)

    return paragraphs


def chunk_by_words(text: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
    """
    Split text into word-based chunks with overlap.

    Best for plain TXT files where structure may be limited.
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


def chunk_by_paragraphs(
    text: str,
    max_words: int = 140,
    paragraph_overlap: int = 1
) -> list[str]:
    """
    Split text into paragraph-based chunks.

    Best for PDFs and DOCX files because they often have paragraph/section structure.
    """
    paragraphs = split_into_paragraphs(text)

    if not paragraphs:
        return []

    chunks = []
    current_chunk = []
    current_word_count = 0

    for paragraph in paragraphs:
        paragraph_word_count = count_words(paragraph)

        if current_chunk and current_word_count + paragraph_word_count > max_words:
            chunks.append("\n".join(current_chunk))

            if paragraph_overlap > 0:
                current_chunk = current_chunk[-paragraph_overlap:]
                current_word_count = sum(count_words(p) for p in current_chunk)
            else:
                current_chunk = []
                current_word_count = 0

        current_chunk.append(paragraph)
        current_word_count += paragraph_word_count

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def chunk_pptx_slide(text: str, max_words: int = 120, overlap: int = 20) -> list[str]:
    """
    Chunk PowerPoint slide text.

    If the slide is short, keep it as one chunk.
    If the slide is long, split it further.
    """
    if count_words(text) <= max_words:
        return [text]

    return chunk_by_words(text, chunk_size=max_words, overlap=overlap)


def choose_chunking_strategy(document: dict) -> str:
    """
    Choose a chunking strategy based on file type.

    This makes chunking document-type-aware instead of one-size-fits-all.
    """
    file_type = document.get("file_type")

    if file_type == ".pptx":
        return "slide"

    if file_type in {".pdf", ".docx"}:
        return "paragraph"

    return "word"


def chunk_document(document: dict) -> list[dict]:
    """
    Chunk a single document using the best strategy for its file type.
    """
    source = document["source"]
    text = document["text"]
    file_type = document.get("file_type")
    location = document.get("location")

    strategy = choose_chunking_strategy(document)

    if strategy == "slide":
        chunks = chunk_pptx_slide(text)

    elif strategy == "paragraph":
        chunks = chunk_by_paragraphs(text)

    else:
        chunks = chunk_by_words(text)

    chunked_documents = []

    for index, chunk in enumerate(chunks, start=1):
        location_part = f"_{location.replace(' ', '_')}" if location else ""

        chunked_documents.append({
            "chunk_id": f"{source}{location_part}_chunk_{index}",
            "source": source,
            "file_type": file_type,
            "location": location,
            "chunk_strategy": strategy,
            "word_count": count_words(chunk),
            "text": chunk,
        })

    return chunked_documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Convert loaded documents into chunks with metadata.

    Supports document-type-aware chunking:
    - TXT uses word chunking
    - PDF/DOCX use paragraph chunking
    - PPTX uses slide-aware chunking
    """
    all_chunks = []

    for document in documents:
        document_chunks = chunk_document(document)
        all_chunks.extend(document_chunks)

    return all_chunks