import json
from pathlib import Path

from src.loader import load_documents
from src.chunker import chunk_documents
from src.hybrid_retriever import HybridRetriever


EVAL_PATH = Path("data/evaluation/questions.json")
UPLOAD_FOLDER = Path("data/uploads")


WEIGHT_CONFIGS = [
    {
        "name": "Keyword-heavy",
        "keyword_weight": 0.7,
        "vector_weight": 0.3,
    },
    {
        "name": "Balanced",
        "keyword_weight": 0.5,
        "vector_weight": 0.5,
    },
    {
        "name": "Vector-heavy",
        "keyword_weight": 0.3,
        "vector_weight": 0.7,
    },
]


def load_evaluation_questions(file_path: Path) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def is_relevant_result(result: dict, relevant_sources: list[str]) -> bool:
    return result["source"] in relevant_sources


def calculate_hit_at_k(results: list[dict], relevant_sources: list[str], k: int) -> int:
    for result in results[:k]:
        if is_relevant_result(result, relevant_sources):
            return 1

    return 0


def calculate_recall_at_k(results: list[dict], relevant_sources: list[str], k: int) -> float:
    found_sources = set()

    for result in results[:k]:
        if is_relevant_result(result, relevant_sources):
            found_sources.add(result["source"])

    return len(found_sources) / len(relevant_sources)


def calculate_reciprocal_rank(results: list[dict], relevant_sources: list[str]) -> float:
    for rank, result in enumerate(results, start=1):
        if is_relevant_result(result, relevant_sources):
            return 1 / rank

    return 0.0


def evaluate_weight_config(
    config_name: str,
    keyword_weight: float,
    vector_weight: float,
    top_k: int = 3,
) -> dict:
    questions = load_evaluation_questions(EVAL_PATH)

    documents = load_documents(str(UPLOAD_FOLDER))
    chunks = chunk_documents(documents)

    retriever = HybridRetriever(
        chunks,
        keyword_weight=keyword_weight,
        vector_weight=vector_weight,
    )

    total_hit_at_1 = 0
    total_hit_at_k = 0
    total_recall_at_k = 0
    total_mrr = 0

    failed_questions = []

    for item in questions:
        question = item["question"]
        relevant_sources = item["relevant_sources"]

        results = retriever.search(question, top_k=top_k)

        hit_at_1 = calculate_hit_at_k(results, relevant_sources, k=1)
        hit_at_k = calculate_hit_at_k(results, relevant_sources, k=top_k)
        recall_at_k = calculate_recall_at_k(results, relevant_sources, k=top_k)
        reciprocal_rank = calculate_reciprocal_rank(results, relevant_sources)

        total_hit_at_1 += hit_at_1
        total_hit_at_k += hit_at_k
        total_recall_at_k += recall_at_k
        total_mrr += reciprocal_rank

        if hit_at_1 == 0:
            failed_questions.append({
                "question": question,
                "expected": relevant_sources,
                "top_result": results[0]["source"] if results else None,
            })

    question_count = len(questions)

    return {
        "config_name": config_name,
        "keyword_weight": keyword_weight,
        "vector_weight": vector_weight,
        "hit_at_1": total_hit_at_1 / question_count,
        f"hit_at_{top_k}": total_hit_at_k / question_count,
        f"recall_at_{top_k}": total_recall_at_k / question_count,
        "mrr": total_mrr / question_count,
        "failed_questions": failed_questions,
    }


def main():
    print("\nRAGA Hybrid Weight Comparison")
    print("=" * 90)

    results = []

    for config in WEIGHT_CONFIGS:
        print(f"\nEvaluating: {config['name']}")

        result = evaluate_weight_config(
            config_name=config["name"],
            keyword_weight=config["keyword_weight"],
            vector_weight=config["vector_weight"],
            top_k=3,
        )

        results.append(result)

    print("\nFinal Comparison")
    print("=" * 90)

    print(
        f"{'Setting':<18} "
        f"{'BM25':<8} "
        f"{'Vector':<8} "
        f"{'Hit@1':<8} "
        f"{'Hit@3':<8} "
        f"{'Recall@3':<10} "
        f"{'MRR':<8}"
    )

    print("-" * 90)

    for result in results:
        print(
            f"{result['config_name']:<18} "
            f"{result['keyword_weight']:<8.2f} "
            f"{result['vector_weight']:<8.2f} "
            f"{result['hit_at_1']:<8.2f} "
            f"{result['hit_at_3']:<8.2f} "
            f"{result['recall_at_3']:<10.2f} "
            f"{result['mrr']:<8.2f}"
        )

    print("\nFailed Hit@1 Questions")
    print("=" * 90)

    for result in results:
        print(f"\n{result['config_name']}:")

        if not result["failed_questions"]:
            print("No Hit@1 failures.")
            continue

        for failure in result["failed_questions"]:
            print(f"- Question: {failure['question']}")
            print(f"  Expected: {failure['expected']}")
            print(f"  Top result: {failure['top_result']}")


if __name__ == "__main__":
    main()