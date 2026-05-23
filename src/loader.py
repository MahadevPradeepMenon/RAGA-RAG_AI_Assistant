from pathlib import Path


def load_txt_file(file_path: Path) -> str:
    """
    Read a single .txt file and return its text.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def load_documents(folder_path: str) -> list[dict]:
    """
    Load all .txt documents from a folder.

    Returns a list like:
    [
        {
            "source": "vacation_policy.txt",
            "text": "Employees are entitled..."
        }
    ]
    """
    folder = Path(folder_path)
    documents = []

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for file_path in folder.glob("*.txt"):
        text = load_txt_file(file_path)

        documents.append({
            "source": file_path.name,
            "text": text
        })

    return documents