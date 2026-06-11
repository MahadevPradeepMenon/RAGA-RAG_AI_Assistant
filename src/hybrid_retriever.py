from src.retriever import KeywordRetriever
from src.vector_retriever import VectorRetriever


class HybridRetriever:
    """
    Combines BM25 keyword retrieval and vector semantic retrieval.

    Uses Weighted Reciprocal Rank Fusion.

    RRF idea:
    - Higher-ranked results receive more points.
    - Results appearing in both keyword and vector retrieval are rewarded.
    - Weights allow us to control how much keyword vs vector retrieval matters.
    """

    def __init__(
        self,
        chunks: list[dict],
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7,
        rrf_k: int = 60,
    ):
        self.chunks = chunks
        self.keyword_retriever = KeywordRetriever(chunks)
        self.vector_retriever = VectorRetriever(chunks)

        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 3, candidate_k: int = 10) -> list[dict]:
        """
        Search using both keyword and vector retrieval.

        top_k = final number of results returned
        candidate_k = number of candidates pulled from each retriever
        """
        keyword_results = self.keyword_retriever.search(query, top_k=candidate_k)
        vector_results = self.vector_retriever.search(query, top_k=candidate_k)

        fused_results = {}

        self._add_results(
            fused_results=fused_results,
            results=keyword_results,
            retrieval_type="keyword",
            weight=self.keyword_weight,
        )

        self._add_results(
            fused_results=fused_results,
            results=vector_results,
            retrieval_type="vector",
            weight=self.vector_weight,
        )

        final_results = list(fused_results.values())

        final_results.sort(
            key=lambda item: item["hybrid_score"],
            reverse=True
        )

        return final_results[:top_k]

    def _add_results(
        self,
        fused_results: dict,
        results: list[dict],
        retrieval_type: str,
        weight: float,
    ):
        """
        Add retrieval results using Weighted Reciprocal Rank Fusion.

        Formula:
        weighted_score = weight * (1 / (rrf_k + rank))
        """
        for rank, result in enumerate(results, start=1):
            chunk_id = result["chunk_id"]
            rrf_score = weight * (1 / (self.rrf_k + rank))

            if chunk_id not in fused_results:
                fused_results[chunk_id] = {
                    "chunk_id": result["chunk_id"],
                    "source": result["source"],
                    "file_type": result.get("file_type"),
                    "location": result.get("location"),
                    "chunk_strategy": result.get("chunk_strategy"),
                    "text": result["text"],

                    "hybrid_score": 0.0,

                    "keyword_score": None,
                    "vector_score": None,

                    "keyword_rank": None,
                    "vector_rank": None,

                    "keyword_weight": self.keyword_weight,
                    "vector_weight": self.vector_weight,
                    "rrf_k": self.rrf_k,
                    "fusion_method": "weighted_rrf",
                }

            fused_results[chunk_id]["hybrid_score"] += rrf_score

            if retrieval_type == "keyword":
                fused_results[chunk_id]["keyword_score"] = result["score"]
                fused_results[chunk_id]["keyword_rank"] = rank

            if retrieval_type == "vector":
                fused_results[chunk_id]["vector_score"] = result["score"]
                fused_results[chunk_id]["vector_rank"] = rank