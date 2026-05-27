import re


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into simple sentences.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def tokenize(text: str) -> set[str]:
    """
    Convert text into lowercase word tokens.
    """
    return set(re.findall(r"\b\w+\b", text.lower()))


def select_best_sentence(question: str, chunk_text: str) -> str:
    """
    Pick the sentence from the retrieved chunk that overlaps most with the question.
    """
    question_tokens = tokenize(question)
    sentences = split_into_sentences(chunk_text)

    if not sentences:
        return chunk_text

    best_sentence = sentences[0]
    best_score = 0

    for sentence in sentences:
        sentence_tokens = tokenize(sentence)
        overlap_score = len(question_tokens.intersection(sentence_tokens))

        if overlap_score > best_score:
            best_score = overlap_score
            best_sentence = sentence

    return best_sentence


def generate_local_answer(question: str, results: list[dict]) -> dict:
    """
    Generate a basic answer without using any paid API.

    It uses the highest-ranked retrieved chunk and extracts the most relevant sentence.
    """
    if not results:
        return {
            "answer": "I could not find this in the available company documents.",
            "sources": [],
            "confidence": 0
        }

    best_result = results[0]

    confidence = best_result.get("hybrid_score", best_result.get("score", 0))

    if confidence <= 0:
        return {
            "answer": "I could not find this in the available company documents.",
            "sources": [],
            "confidence": confidence
        }

    best_sentence = select_best_sentence(question, best_result["text"])

    return {
        "answer": best_sentence,
        "sources": [best_result["source"]],
        "confidence": confidence
    }