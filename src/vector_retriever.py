import numpy as np
from sentence_transformers import SentenceTransformer


class VectorRetriever:
    """
    Semantic retriever using sentence-transformers embeddings.

    Unlike BM25, this searches by meaning rather than exact word overlap.
    """

    def __init__(self, chunks: list[dict], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)

        self.chunk_texts = [chunk["text"] for chunk in chunks]

        self.chunk_embeddings = self.model.encode(
            self.chunk_texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Search chunks using semantic similarity.
        """
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        similarity_scores = np.dot(self.chunk_embeddings, query_embedding)

        results = []

        for chunk, score in zip(self.chunks, similarity_scores):
            results.append({
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "text": chunk["text"],
                "score": float(score)
            })

        results.sort(key=lambda item: item["score"], reverse=True)

        return results[:top_k]