"""
Sparse retrieval using BM25.
Pure CPU, no neural components.
"""

import re
from typing import Any

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """Simple tokenization for BM25."""
    return re.findall(r"\b\w+\b", str(text).lower())


class SparseRetriever:
    """Sparse retrieval via BM25."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._bm25 = None
        self._doc_ids = None

    def index(self, corpus: dict[str, dict[str, Any]]) -> None:
        """Index corpus documents."""
        self._doc_ids = list(corpus.keys())
        tokenized = [
            _tokenize(
                f"{corpus[did].get('title', '')} {corpus[did].get('text', '')}".strip()
            )
            for did in self._doc_ids
        ]
        self._bm25 = BM25Okapi(tokenized)

    def search(
        self,
        query: str,
        top_k: int = 100,
    ) -> list[tuple[str, float]]:
        """Return top-k (doc_id, score) pairs."""
        if self._bm25 is None:
            raise RuntimeError("Call index() before search()")

        q_tokens = _tokenize(query)
        scores = self._bm25.get_scores(q_tokens)
        top_indices = scores.argsort()[::-1][:top_k]
        return [(self._doc_ids[i], float(scores[i])) for i in top_indices]

    def batch_search(
        self,
        queries: dict[str, str],
        top_k: int = 100,
    ) -> dict[str, list[tuple[str, float]]]:
        """Search for multiple queries."""
        results = {}
        for qid, qtext in queries.items():
            results[qid] = self.search(qtext, top_k)
        return results
