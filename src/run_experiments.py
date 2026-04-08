"""
Main experiment runner for RAG retrieval comparison.

Option A (Empirical evaluation): compare dense, sparse, and hybrid retrieval
strategies, including a simple chunk-size/granularity sweep.

CPU-only; self-contained relative to the project folder.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure `from src...` imports work regardless of the launch working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # project/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CHUNK_SIZES, DEFAULT_DENSE_MODEL, EVAL_TOP_K, TOP_K
from src.data.dataset_loader import load_beir_dataset
from src.evaluation import evaluate_retrieval
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.hybrid import HybridRetriever

def run_retrieval(method: str, corpus: dict[str, dict[str, Any]], queries: dict[str, str],
                  top_k: int = TOP_K) -> tuple[dict[str, list[tuple[str, float]]], float]:
    """Run one retrieval method; return (results, elapsed_time_seconds)."""
    if method == "dense":
        retriever = DenseRetriever(model_name=DEFAULT_DENSE_MODEL)
    elif method == "sparse":
        retriever = SparseRetriever()
    elif method == "hybrid":
        retriever = HybridRetriever(dense_model=DEFAULT_DENSE_MODEL)
    else:
        raise ValueError(f"Unknown method: {method}")

    t0 = time.perf_counter()
    retriever.index(corpus)
    results = retriever.batch_search(queries, top_k=top_k)
    elapsed = time.perf_counter() - t0
    return results, elapsed

def main() -> list[dict[str, Any]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="nfcorpus", choices=["nfcorpus", "fiqa", "scifact", "arguana", "dbpedia-entity"])
    parser.add_argument("--methods", nargs="+", default=["dense", "sparse", "hybrid"])
    parser.add_argument("--chunk_sizes", nargs="+", type=int, default=CHUNK_SIZES)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT/"results"), help="Where to write outputs (default: project/results).")
    parser.add_argument("--no_chunk_variation", action="store_true", help="Skip chunk size experiments (use original documents only).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(exist_ok=True)

    all_results: list[dict[str, Any]] = []

    # Chunk size None = original documents
    chunk_configs = [None] if args.no_chunk_variation else [None] + list(args.chunk_sizes)

    for chunk_size in chunk_configs:
        cs_label = "original" if chunk_size is None else f"chunk_{chunk_size}"
        print(f"\n--- Dataset: {args.dataset}, Chunk: {cs_label} ---")
        corpus, queries, qrels = load_beir_dataset(args.dataset, chunk_size=chunk_size, max_queries=args.max_queries)
        print(f"Corpus: {len(corpus)} docs, Queries: {len(queries)}")

        for method in args.methods:
            print(f"Running {method}...")
            results, elapsed = run_retrieval(method, corpus, queries)
            metrics = evaluate_retrieval(results, qrels, k_values=EVAL_TOP_K)
            metrics["method"] = method
            metrics["chunk_size"] = cs_label
            metrics["dataset"] = args.dataset
            metrics["time_seconds"] = round(elapsed, 2)
            metrics["n_docs"] = len(corpus)
            metrics["n_queries"] = len(queries)
            all_results.append(metrics)
            print(f"MRR: {metrics.get('MRR', 0.0):.4f},Recall@10: {metrics.get('Recall@10', 0.0):.4f}, Time: {elapsed:.1f}s")

    summary_path = output_dir / "metrics" / f"{args.dataset}_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {summary_path}")
    return all_results

if __name__ == "__main__":
    main()
