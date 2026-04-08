"""
Hybrid retrieval: Reciprocal Rank Fusion (RRF) of dense and sparse results.
"""

from typing import Any

from .dense import DenseRetriever
from .sparse import SparseRetriever


def _reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Fuse multiple ranked lists using RRF.
    score(d) = sum over lists of 1 / (k + rank(d))
    """
    doc_scores = {}
    for rlist in ranked_lists:
        for rank, (doc_id, _) in enumerate(rlist, start=1):
            rrf_score = 1.0 / (k + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rrf_score

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs


class HybridRetriever:
    """Hybrid retrieval combining dense and sparse via RRF."""

    def __init__(
        self,
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        rrf_k: int = 60,
        batch_size: int = 32,
    ):
        self.dense = DenseRetriever(model_name=dense_model, batch_size=batch_size)
        self.sparse = SparseRetriever()
        self.rrf_k = rrf_k

    def index(self, corpus: dict[str, Any]) -> None:
        """Index for both retrievers."""
        self.dense.index(corpus)
        self.sparse.index(corpus)

    def search(
        self,
        query: str,
        top_k: int = 100,
    ) -> list[tuple[str, float]]:
        """Fuse dense and sparse results via RRF."""
        dense_results = self.dense.search(query, top_k=top_k * 2)
        sparse_results = self.sparse.search(query, top_k=top_k * 2)
        fused = _reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=self.rrf_k,
        )
        return fused[:top_k]

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
