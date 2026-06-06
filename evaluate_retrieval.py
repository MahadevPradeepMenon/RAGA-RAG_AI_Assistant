import json
from pathlib import Path

from src.loader import load_documents
from src.chunker import chunk_documents
from src.hybrid_retriever import HybridRetriever


EVAL_PATH = Path("data/evaluation/questions.json")
UPLOAD_FOLDER = Path("data/uploads")


def load_evaluation_questions(file_path: Path) -> list[dict]:
    """
    Load evaluation questions from a JSON file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def is_relevant_result(result: dict, relevant_sources: list[str]) -> bool:
    """
    Check whether a retrieved result comes from one of the expected source documents.
    """
    return result["source"] in relevant_sources


def calculate_hit_at_k(results: list[dict], relevant_sources: list[str], k: int) -> int:
    """
    Hit@K = 1 if at least one relevant source appears in the top K results.
    Otherwise 0.
    """
    top_k_results = results[:k]

    for result in top_k_results:
        if is_relevant_result(result, relevant_sources):
            return 1

    return 0


def calculate_recall_at_k(results: list[dict], relevant_sources: list[str], k: int) -> float:
    """
    Recall@K = number of relevant sources found in top K / total relevant sources.

    This is source-level recall, not chunk-level recall.
    """
    top_k_results = results[:k]

    found_sources = set()

    for result in top_k_results:
        if is_relevant_result(result, relevant_sources):
            found_sources.add(result["source"])

    return len(found_sources) / len(relevant_sources)


def calculate_reciprocal_rank(results: list[dict], relevant_sources: list[str]) -> float:
    """
    Reciprocal Rank = 1 / rank of first relevant result.

    If the first relevant result is ranked 1, score is 1.0.
    If ranked 2, score is 0.5.
    If ranked 3, score is 0.333.
    If not found, score is 0.
    """
    for rank, result in enumerate(results, start=1):
        if is_relevant_result(result, relevant_sources):
            return 1 / rank

    return 0.0


def evaluate_retrieval(top_k: int = 3):
    """
    Run retrieval evaluation over all evaluation questions.
    """
    questions = load_evaluation_questions(EVAL_PATH)

    documents = load_documents(str(UPLOAD_FOLDER))
    chunks = chunk_documents(documents)

    retriever = HybridRetriever(chunks)

    total_hit_at_1 = 0
    total_hit_at_k = 0
    total_recall_at_k = 0
    total_mrr = 0

    print("\nRAGA Retrieval Evaluation")
    print("=" * 80)
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Evaluation questions: {len(questions)}")
    print("=" * 80)

    for index, item in enumerate(questions, start=1):
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

        print(f"\nQuestion {index}: {question}")
        print(f"Expected Source(s): {relevant_sources}")

        print("\nRetrieved Results:")
        for rank, result in enumerate(results, start=1):
            relevant_marker = "✅" if is_relevant_result(result, relevant_sources) else "❌"
            print(
                f"{rank}. {relevant_marker} {result['source']} "
                f"| Hybrid Score: {result['hybrid_score']:.6f}"
            )

        print("\nScores:")
        print(f"Hit@1: {hit_at_1}")
        print(f"Hit@{top_k}: {hit_at_k}")
        print(f"Recall@{top_k}: {recall_at_k:.2f}")
        print(f"Reciprocal Rank: {reciprocal_rank:.2f}")
        print("-" * 80)

    question_count = len(questions)

    average_hit_at_1 = total_hit_at_1 / question_count
    average_hit_at_k = total_hit_at_k / question_count
    average_recall_at_k = total_recall_at_k / question_count
    mean_reciprocal_rank = total_mrr / question_count

    print("\nFinal Evaluation Summary")
    print("=" * 80)
    print(f"Hit@1: {average_hit_at_1:.2f}")
    print(f"Hit@{top_k}: {average_hit_at_k:.2f}")
    print(f"Recall@{top_k}: {average_recall_at_k:.2f}")
    print(f"MRR: {mean_reciprocal_rank:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_retrieval(top_k=3)