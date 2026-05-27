def generate_basic_answer(results: list[dict]) -> dict:
    """
    Generate a basic answer using the highest-scoring retrieved chunk.

    This is not using AI yet.
    It simply returns the most relevant chunk as the answer.
    """
    if not results:
        return {
            "answer": "I could not find a relevant answer in the company documents.",
            "source": None,
            "confidence": 0
        }

    best_result = results[0]

    confidence = best_result.get("hybrid_score", best_result.get("score", 0))

    if confidence <= 0:
        return {
            "answer": "I could not find a relevant answer in the company documents.",
            "source": None,
            "confidence": confidence
        }

    return {
        "answer": best_result["text"],
        "source": best_result["source"],
        "confidence": confidence
    }