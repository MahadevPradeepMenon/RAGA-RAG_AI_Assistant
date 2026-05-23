import re
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """
    Lowercase text and split it into simple word tokens.
    """
    return re.findall(r"\b\w+\b", text.lower())


class KeywordRetriever:
    """
    Keyword-based retriever using BM25.
    """

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.tokenized_chunks = [tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Search chunks using BM25 and return top results.
        """
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        results = []

        for chunk, score in zip(self.chunks, scores):
            results.append({
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "text": chunk["text"],
                "score": float(score)
            })

        results.sort(key=lambda item: item["score"], reverse=True)

        return results[:top_k]