"""
Dense retrieval using sentence-transformers.
CPU-optimized with configurable batch size.
"""

from typing import Any

from sentence_transformers import SentenceTransformer


class DenseRetriever:
    """Dense retrieval via learned embeddings and cosine similarity."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self._corpus_embeddings = None
        self._doc_ids = None

    def index(self, corpus: dict[str, dict[str, Any]]) -> None:
        """Index corpus documents."""
        doc_ids = list(corpus.keys())
        texts = [
            f"{corpus[did].get('title', '')} {corpus[did].get('text', '')}".strip()
            for did in doc_ids
        ]
        self._corpus_embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        self._doc_ids = doc_ids

    def search(
        self,
        query: str,
        top_k: int = 100,
    ) -> list[tuple[str, float]]:
        """Return top-k (doc_id, score) pairs."""
        if self._corpus_embeddings is None:
            raise RuntimeError("Call index() before search()")

        q_emb = self.model.encode([query], convert_to_numpy=True)
        scores = self._corpus_embeddings @ q_emb.T
        scores = scores.flatten()

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
