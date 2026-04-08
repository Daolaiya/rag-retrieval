"""
Retrieval evaluation metrics: Recall@k, MRR, NDCG@k.
"""

import math
from typing import Any

def recall_at_k(retrieved: list[tuple[str, float]], relevant: set[str], k: int) -> float:
    """Recall@k: proportion of relevant docs in top-k."""
    top_k_ids = {doc_id for doc_id, _ in retrieved[:k]}
    hits = len(top_k_ids & relevant)
    return hits / len(relevant) if relevant else 0.0

def mrr(retrieved: list[tuple[str, float]], relevant: set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant doc."""
    for rank, (doc_id, _) in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved: list[tuple[str, float]], relevant: dict[str, int], k: int) -> float:
    """
    NDCG@k. Uses binary relevance (1 if relevant, 0 else).
    DCG = sum (2^rel - 1) / log2(rank+1)
    """
    dcg = 0.0
    for rank, (doc_id, _) in enumerate(retrieved[:k], start=1):
        rel = relevant.get(doc_id, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(rank + 1)

    # Ideal DCG
    ideal_relevances = sorted(relevant.values(), reverse=True)[:k]
    idcg = 0.0
    for rank, rel in enumerate(ideal_relevances, start=1):
        idcg += (2**rel - 1) / math.log2(rank + 1)

    if idcg == 0:
        return 0.0
    return dcg / idcg

def evaluate_retrieval(results: dict[str, list[tuple[str, float]]], qrels: dict[str, dict[str, int]],
                       k_values: list[int] = [1, 5, 10, 100]) -> dict[str, Any]:
    """
    Compute Recall@k, MRR, NDCG@k for retrieval results.

    Args:
        results: {query_id: [(doc_id, score), ...]}
        qrels: {query_id: {doc_id: relevance}}
        k_values: k values for Recall@k and NDCG@k

    Returns:
        Dict with mean Recall@k, MRR, NDCG@k
    """
    recall_sums = {k: 0.0 for k in k_values}
    mrr_sum = 0.0
    ndcg_sums = {k: 0.0 for k in k_values}
    n = 0

    for qid, retrieved in results.items():
        if qid not in qrels:
            continue
        relevant_docs = qrels[qid]
        relevant_set = set(relevant_docs.keys())

        for k in k_values:
            recall_sums[k] += recall_at_k(retrieved, relevant_set, k)
        mrr_sum += mrr(retrieved, relevant_set)
        for k in k_values:
            ndcg_sums[k] += ndcg_at_k(retrieved, relevant_docs, k)
        n += 1

    if n == 0:
        return {}

    metrics = {"n_queries": n, "MRR": mrr_sum / n}
    
    for k in k_values:
        metrics[f"Recall@{k}"] = recall_sums[k] / n
        metrics[f"NDCG@{k}"] = ndcg_sums[k] / n

    return metrics
