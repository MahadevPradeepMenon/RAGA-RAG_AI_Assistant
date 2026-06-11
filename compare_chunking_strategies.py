import json
from pathlib import Path

from src.loader import load_documents
from src.chunker import chunk_documents
from src.hybrid_retriever import HybridRetriever


EVAL_PATH = Path("data/evaluation/questions.json")
UPLOAD_FOLDER = Path("data/uploads")


CHUNKING_CONFIGS = [
    {
        "name": "Small chunks",
        "config": {
            "txt_chunk_size": 80,
            "txt_overlap": 20,
            "paragraph_max_words": 100,
            "paragraph_overlap": 1,
            "pptx_max_words": 80,
            "pptx_overlap": 20,
        },
    },
    {
        "name": "Default chunks",
        "config": {
            "txt_chunk_size": 100,
            "txt_overlap": 20,
            "paragraph_max_words": 140,
            "paragraph_overlap": 1,
            "pptx_max_words": 120,
            "pptx_overlap": 20,
        },
    },
    {
        "name": "Large chunks",
        "config": {
            "txt_chunk_size": 150,
            "txt_overlap": 30,
            "paragraph_max_words": 180,
            "paragraph_overlap": 1,
            "pptx_max_words": 160,
            "pptx_overlap": 30,
        },
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


def evaluate_config(config_name: str, chunking_config: dict, top_k: int = 3) -> dict:
    questions = load_evaluation_questions(EVAL_PATH)

    documents = load_documents(str(UPLOAD_FOLDER))
    chunks = chunk_documents(documents, config=chunking_config)

    retriever = HybridRetriever(chunks)

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
        "chunk_count": len(chunks),
        "hit_at_1": total_hit_at_1 / question_count,
        f"hit_at_{top_k}": total_hit_at_k / question_count,
        f"recall_at_{top_k}": total_recall_at_k / question_count,
        "mrr": total_mrr / question_count,
        "failed_questions": failed_questions,
    }


def main():
    print("\nRAGA Chunking Strategy Comparison")
    print("=" * 90)

    results = []

    for chunking_setup in CHUNKING_CONFIGS:
        config_name = chunking_setup["name"]
        config = chunking_setup["config"]

        print(f"\nEvaluating: {config_name}")

        result = evaluate_config(config_name, config, top_k=3)
        results.append(result)

    print("\nFinal Comparison")
    print("=" * 90)

    print(
        f"{'Strategy':<20} "
        f"{'Chunks':<8} "
        f"{'Hit@1':<8} "
        f"{'Hit@3':<8} "
        f"{'Recall@3':<10} "
        f"{'MRR':<8}"
    )

    print("-" * 90)

    for result in results:
        print(
            f"{result['config_name']:<20} "
            f"{result['chunk_count']:<8} "
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