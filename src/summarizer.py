import re
from collections import Counter


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "to", "of", "in", "on",
    "for", "with", "as", "by", "is", "are", "was", "were", "be", "been",
    "being", "it", "this", "that", "these", "those", "from", "at", "into",
    "about", "employees", "employee", "company", "must", "should", "may",
    "can", "will", "shall", "do", "does", "did", "have", "has", "had"
}


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into simple sentences.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def tokenize_words(text: str) -> list[str]:
    """
    Convert text into lowercase word tokens.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    return [word for word in words if word not in STOPWORDS]


def summarize_text(text: str, max_sentences: int = 3) -> list[str]:
    """
    Create a simple extractive summary by selecting important sentences.

    This does not rewrite text like an LLM.
    It selects the most important sentences from the document.
    """
    sentences = split_into_sentences(text)

    if len(sentences) <= max_sentences:
        return sentences

    words = tokenize_words(text)
    word_frequencies = Counter(words)

    if not word_frequencies:
        return sentences[:max_sentences]

    scored_sentences = []

    for index, sentence in enumerate(sentences):
        sentence_words = tokenize_words(sentence)

        if not sentence_words:
            score = 0
        else:
            score = sum(word_frequencies[word] for word in sentence_words) / len(sentence_words)

        scored_sentences.append({
            "index": index,
            "sentence": sentence,
            "score": score
        })

    top_sentences = sorted(
        scored_sentences,
        key=lambda item: item["score"],
        reverse=True
    )[:max_sentences]

    top_sentences = sorted(top_sentences, key=lambda item: item["index"])

    return [item["sentence"] for item in top_sentences]


def is_summary_request(question: str) -> bool:
    """
    Detect whether the user is asking for a summary.
    """
    question_lower = question.lower()

    summary_keywords = [
        "summarize",
        "summary",
        "summarise",
        "overview",
        "brief",
        "what are these documents about",
        "what is this document about",
        "explain the documents"
    ]

    return any(keyword in question_lower for keyword in summary_keywords)


def is_simple_request(question: str) -> bool:
    """
    Detect whether the user wants a simpler explanation.
    """
    question_lower = question.lower()

    simple_keywords = [
        "simple",
        "simply",
        "layman",
        "plain english",
        "easy terms",
        "i don't understand",
        "i do not understand"
    ]

    return any(keyword in question_lower for keyword in simple_keywords)


def select_documents_for_summary(question: str, documents: list[dict]) -> list[dict]:
    """
    If the user mentions a specific document name, summarize that document.
    Otherwise, summarize all documents.
    """
    question_lower = question.lower()
    matching_documents = []

    for document in documents:
        source = document["source"]
        source_lower = source.lower()
        source_name_without_extension = source_lower.rsplit(".", 1)[0]
        readable_source_name = source_name_without_extension.replace("_", " ").replace("-", " ")

        if source_lower in question_lower or readable_source_name in question_lower:
            matching_documents.append(document)

    if matching_documents:
        return matching_documents

    return documents


def generate_summary_response(question: str, documents: list[dict]) -> str:
    """
    Generate a summary response for one or more documents.
    """
    selected_documents = select_documents_for_summary(question, documents)
    simple_mode = is_simple_request(question)

    max_sentences = 2 if simple_mode else 3

    if simple_mode:
        response_lines = ["Here is a simple summary of the relevant document content:\n"]
    else:
        response_lines = ["Here is a summary of the relevant document content:\n"]

    for document in selected_documents:
        source = document["source"]
        text = document["text"]

        summary_sentences = summarize_text(text, max_sentences=max_sentences)

        response_lines.append(f"**{source}**")

        for sentence in summary_sentences:
            response_lines.append(f"- {sentence}")

        response_lines.append("")

    return "\n".join(response_lines)