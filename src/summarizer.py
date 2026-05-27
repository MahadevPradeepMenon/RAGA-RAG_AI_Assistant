import re
from collections import Counter


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "to", "of", "in", "on",
    "for", "with", "as", "by", "is", "are", "was", "were", "be", "been",
    "being", "it", "this", "that", "these", "those", "from", "at", "into",
    "about", "employees", "employee", "company", "must", "should", "may",
    "can", "will", "shall", "do", "does", "did", "have", "has", "had"
}


SIMPLE_REPLACEMENTS = {
    "annual leave": "paid time off",
    "vacation requests": "requests for time off",
    "unused leave days": "unused days off",
    "carried over": "moved to the next year",
    "remotely": "from home or outside the office",
    "remote work": "working from home or outside the office",
    "scheduled team meetings": "planned team meetings",
    "internal company systems": "company tools and websites",
    "technical issues": "computer or system problems",
    "IT helpdesk": "IT support team",
    "onboarding portal": "new-starter website",
    "self-service password reset tool": "online password reset tool",
    "VPN": "secure company connection",
    "HR": "Human Resources",
    "entitled to": "allowed to have",
    "submitted": "sent",
    "approved": "agreed to",
    "required": "needed",
}


def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def tokenize_words(text: str) -> list[str]:
    words = re.findall(r"\b\w+\b", text.lower())
    return [word for word in words if word not in STOPWORDS]


def simplify_sentence(sentence: str) -> str:
    simplified = sentence

    for phrase, replacement in SIMPLE_REPLACEMENTS.items():
        pattern = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
        simplified = pattern.sub(replacement, simplified)

    simplified = simplified.replace(" per year", " each year")

    return simplified


def summarize_text(text: str, max_sentences: int = 2) -> list[str]:
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
    question_lower = question.lower()

    if "all documents" in question_lower or "all docs" in question_lower:
        return documents

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
    selected_documents = select_documents_for_summary(question, documents)
    simple_mode = is_simple_request(question)

    response_lines = []

    if simple_mode:
        response_lines.append("### Simple Summary")
        response_lines.append("Here is a plain-English summary of the relevant documents:\n")
    else:
        response_lines.append("### Document Summary")
        response_lines.append("Here is a summary of the relevant documents:\n")

    for document in selected_documents:
        source = document["source"]
        text = document["text"]

        summary_sentences = summarize_text(text, max_sentences=2)

        response_lines.append(f"#### {source}")

        for sentence in summary_sentences:
            if simple_mode:
                sentence = simplify_sentence(sentence)

            response_lines.append(f"- {sentence}")

        response_lines.append("")

    return "\n".join(response_lines)