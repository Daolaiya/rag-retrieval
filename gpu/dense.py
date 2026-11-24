"""
Dense retrieval using sentence-transformers.
Supports CPU or GPU with configurable batch size.
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
            device=self.device,
            normalize_embeddings=True,
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

        q_emb = self.model.encode(
            [query],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            device=self.device,
            normalize_embeddings=True,
        )
        scores = self._corpus_embeddings @ q_emb.T
        scores = scores.flatten()

        top_indices = scores.argsort()[::-1][:top_k]
        return [(self._doc_ids[i], float(scores[i])) for i in top_indices]

    def batch_search(
        self,
        queries: dict[str, str],
        top_k: int = 100,
    ) -> dict[str, list[tuple[str, float]]]:
        """Search for multiple queries using batched query encoding."""
        if self._corpus_embeddings is None:
            raise RuntimeError("Call index() before batch_search()")
    
        qids = list(queries.keys())
        qtexts = [queries[qid] for qid in qids]
    
        q_embs = self.model.encode(
            qtexts,
            batch_size=self.batch_size,
            show_progress_bar=len(qtexts) > 100,
            convert_to_numpy=True,
            device=self.device,
            normalize_embeddings=True,
        )
    
        scores = q_embs @ self._corpus_embeddings.T  # shape: (n_queries, n_docs)
    
        results = {}
        for i, qid in enumerate(qids):
            row = scores[i]
            top_indices = row.argsort()[::-1][:top_k]
            results[qid] = [(self._doc_ids[j], float(row[j])) for j in top_indices]
    
        return results