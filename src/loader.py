from pathlib import Path
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


def load_txt_file(file_path: Path) -> str:
    """
    Read a single .txt file and return its text.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def load_pdf_file(file_path: Path) -> str:
    """
    Read a PDF file and extract text from each page.

    This works best for text-based PDFs.
    Scanned image PDFs may require OCR later.
    """
    reader = PdfReader(str(file_path))
    text_parts = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text:
            text_parts.append(f"\n[Page {page_number}]\n{page_text}")

    return "\n".join(text_parts)


def load_single_document(file_path: Path) -> dict:
    """
    Load one supported document and return source + text.
    """
    extension = file_path.suffix.lower()

    if extension == ".txt":
        text = load_txt_file(file_path)
    elif extension == ".pdf":
        text = load_pdf_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")

    return {
        "source": file_path.name,
        "text": text,
        "file_type": extension,
    }


def load_documents(folder_path: str) -> list[dict]:
    """
    Load all supported documents from a folder.

    Currently supports:
    - .txt
    - .pdf
    """
    folder = Path(folder_path)
    documents = []

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            document = load_single_document(file_path)

            if document["text"].strip():
                documents.append(document)

    return documents