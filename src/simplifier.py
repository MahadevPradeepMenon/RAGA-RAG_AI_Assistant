import re


JARGON_REPLACEMENTS = {
    "annual leave": "paid time off",
    "vacation requests": "requests for time off",
    "unused leave days": "unused days off",
    "carried over": "moved",
    "remotely": "from home or outside the office",
    "remote work": "working from home or outside the office",
    "scheduled team meetings": "planned team meetings",
    "internal company systems": "company tools and websites",
    "technical issues": "computer or system problems",
    "it helpdesk": "IT support team",
    "onboarding portal": "new-starter website",
    "self-service password reset tool": "online password reset tool",
    "vpn": "secure company connection",
    "hr": "Human Resources",
    "entitled to": "allowed to have",
    "submitted": "sent",
    "approved": "agreed to",
    "required": "needed",
}


def is_simple_request(question: str) -> bool:
    """
    Detect whether the user wants a simpler explanation.
    """
    question_lower = question.lower()

    simple_keywords = [
        "simple",
        "simply",
        "layman",
        "layman's",
        "plain english",
        "easy terms",
        "easier terms",
        "i don't understand",
        "i do not understand",
        "dont understand",
        "explain that",
        "explain this",
        "make it clearer",
        "can you clarify",
    ]

    return any(keyword in question_lower for keyword in simple_keywords)


def is_follow_up_simplification(question: str) -> bool:
    """
    Detect whether the user is asking to simplify the previous assistant answer.
    """
    question_lower = question.lower().strip()

    follow_up_phrases = [
        "i don't understand",
        "i do not understand",
        "dont understand",
        "explain that",
        "explain this",
        "make it simpler",
        "make this simpler",
        "explain in simple terms",
        "explain in plain english",
        "can you clarify",
    ]

    return any(phrase in question_lower for phrase in follow_up_phrases)


def replace_jargon(text: str) -> str:
    """
    Replace common workplace jargon with simpler wording.
    Uses word boundaries so short terms like HR do not break words like 'three'.
    """
    simplified = text

    for jargon, plain_wording in JARGON_REPLACEMENTS.items():
        pattern = re.compile(rf"\b{re.escape(jargon)}\b", re.IGNORECASE)
        simplified = pattern.sub(plain_wording, simplified)

    return simplified


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into simple sentences.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def simplify_sentence(sentence: str) -> str:
    """
    Make one sentence easier to read.
    """
    sentence = replace_jargon(sentence)

    sentence = sentence.replace(" per year", " each year")
    sentence = sentence.replace(" at least ", " minimum of ")

    return sentence.strip()


def simplify_answer(answer: str) -> str:
    """
    Convert an answer into a simpler local explanation.
    """
    if not answer.strip():
        return "I do not have an answer to simplify yet."

    sentences = split_into_sentences(answer)

    if not sentences:
        simplified_text = replace_jargon(answer)
        return f"In simple terms: {simplified_text}"

    simplified_sentences = []

    for sentence in sentences[:3]:
        simplified_sentences.append(simplify_sentence(sentence))

    simplified_answer = " ".join(simplified_sentences)

    return f"In simple terms: {simplified_answer}"