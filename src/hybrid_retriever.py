from src.retriever import KeywordRetriever
from src.vector_retriever import VectorRetriever


class HybridRetriever:
    """
    Combines BM25 keyword retrieval and vector semantic retrieval.

    This uses a simple rank-fusion approach:
    - Run BM25 search
    - Run vector search
    - Give points based on ranking position
    - Merge and sort final results
    """

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.keyword_retriever = KeywordRetriever(chunks)
        self.vector_retriever = VectorRetriever(chunks)

    def search(self, query: str, top_k: int = 3, candidate_k: int = 5) -> list[dict]:
        """
        Search using both keyword and vector retrieval.

        top_k = final number of results returned
        candidate_k = number of results pulled from each retriever before merging
        """
        keyword_results = self.keyword_retriever.search(query, top_k=candidate_k)
        vector_results = self.vector_retriever.search(query, top_k=candidate_k)

        fused_results = {}

        self._add_results(
            fused_results,
            keyword_results,
            retrieval_type="keyword"
        )

        self._add_results(
            fused_results,
            vector_results,
            retrieval_type="vector"
        )

        final_results = list(fused_results.values())

        final_results.sort(
            key=lambda item: item["hybrid_score"],
            reverse=True
        )

        return final_results[:top_k]

    def _add_results(self, fused_results: dict, results: list[dict], retrieval_type: str):
        """
        Add retrieval results into the fusion dictionary.

        Higher-ranked results receive more points.
        """
        for rank, result in enumerate(results, start=1):
            chunk_id = result["chunk_id"]

            rank_score = 1 / rank

            if chunk_id not in fused_results:
                fused_results[chunk_id] = {
                    "chunk_id": result["chunk_id"],
                    "source": result["source"],
                    "text": result["text"],
                    "hybrid_score": 0,
                    "keyword_score": None,
                    "vector_score": None
                }

            fused_results[chunk_id]["hybrid_score"] += rank_score

            if retrieval_type == "keyword":
                fused_results[chunk_id]["keyword_score"] = result["score"]

            if retrieval_type == "vector":
                fused_results[chunk_id]["vector_score"] = result["score"]