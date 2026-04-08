"""
Run a full grid of retrieval experiments on CPU.

Grid axes:
- datasets: all datasets in src.config.AVAILABLE_DATASETS (default)
- methods: dense, sparse, hybrid (default)
- chunk sizes: original (no chunks) + all values in src.config.CHUNK_SIZES (default)

Outputs are saved to a timestamped run folder under:
project/results/grid_runs/<timestamp>/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure `from src...` imports work from any launch directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AVAILABLE_DATASETS, CHUNK_SIZES, EVAL_TOP_K, TOP_K
from src.data.dataset_loader import load_beir_dataset
from src.evaluation import evaluate_retrieval
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sparse import SparseRetriever

# from src.config import DEFAULT_DENSE_MODEL # ==> "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # ==> ~80MB, fast
# DEFAULT_DENSE_MODEL = "sentence-transformers/all-mpnet-base-v2" # ==> larger, better quality

# Suitable CPU-first default:
# gives stable-enough metrics while keeping runtime practical for a full grid.
DEFAULT_MAX_QUERIES = 200

def _build_retriever(method: str):
    if method == "dense":
        return DenseRetriever(model_name=DEFAULT_DENSE_MODEL)
    if method == "sparse":
        return SparseRetriever()
    if method == "hybrid":
        return HybridRetriever(dense_model=DEFAULT_DENSE_MODEL)
    raise ValueError(f"Unknown method: {method}")

def run_retrieval(method: str, corpus: dict[str, dict[str, Any]], queries: dict[str, str],
                  top_k: int = TOP_K) -> tuple[dict[str, list[tuple[str, float]]], float]:
    retriever = _build_retriever(method)
    t0 = time.perf_counter()
    retriever.index(corpus)
    results = retriever.batch_search(queries, top_k=top_k)
    elapsed = time.perf_counter() - t0
    return results, elapsed

def _safe_chunk_label(chunk_size: int | None) -> str:
    return "original" if chunk_size is None else f"chunk_{chunk_size}"

def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["empty"])
        return
    preferred_order = ["run_id", "dataset", "method", "chunk_size", "max_queries", "n_docs", "n_queries", "time_seconds", "MRR", "Recall@1",
                       "Recall@5", "Recall@10", "Recall@100", "NDCG@1", "NDCG@5", "NDCG@10", "NDCG@100"]
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    fieldnames = [k for k in preferred_order if k in all_keys] + sorted(all_keys - set(preferred_order))

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=AVAILABLE_DATASETS)
    parser.add_argument("--methods", nargs="+", default=["dense", "sparse", "hybrid"])
    parser.add_argument("--chunk_sizes", nargs="+", type=int, default=CHUNK_SIZES)
    parser.add_argument("--max_queries", type=int, default=DEFAULT_MAX_QUERIES)
    parser.add_argument("--output_root", default=str(PROJECT_ROOT / "results" / "grid_runs"),
                        help="Root directory for timestamped grid outputs.")
    parser.add_argument("--include_original", action="store_true", default=True,
                        help="Include no-chunk/original documents in chunk grid (default: true).")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    configs_dir = run_dir / "configs"
    aggregates_dir = run_dir / "aggregate"
    configs_dir.mkdir(parents=True, exist_ok=True)
    aggregates_dir.mkdir(parents=True, exist_ok=True)

    chunk_grid: list[int | None] = [None] + list(args.chunk_sizes) if args.include_original else list(args.chunk_sizes)
    full_grid_size = len(args.datasets) * len(args.methods) * len(chunk_grid)
    print(f"Run ID: {run_id}")
    print(f"Output folder: {run_dir}")
    print(f"Grid size: {full_grid_size} configurations")
    print(f"Datasets: {args.datasets}")
    print(f"Methods: {args.methods}")
    print(f"Chunks: {[ _safe_chunk_label(c) for c in chunk_grid ]}")
    print(f"max_queries: {args.max_queries}")

    summary_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    completed = 0

    for dataset in args.datasets:
        for chunk_size in chunk_grid:
            chunk_label = _safe_chunk_label(chunk_size)
            print(f"\n=== Loading dataset={dataset}, chunk={chunk_label} ===")
            try:
                corpus, queries, qrels = load_beir_dataset(dataset_name=dataset, chunk_size=chunk_size, max_queries=args.max_queries)
            except Exception as e:
                msg = f"Dataset load failed for dataset={dataset}, chunk={chunk_label}: {e}"
                print(msg)
                errors.append({"stage": "load", "dataset": dataset, "chunk_size": chunk_label, "method": None, "error": str(e)})
                continue

            print(f"Corpus: {len(corpus)} docs | Queries: {len(queries)}")
            for method in args.methods:
                cfg_start = time.perf_counter()
                print(f"-> method={method} ...")
                try:
                    results, retrieval_time = run_retrieval(method, corpus, queries, top_k=TOP_K)
                    metrics = evaluate_retrieval(results, qrels, k_values=EVAL_TOP_K)
                    cfg_elapsed = time.perf_counter() - cfg_start

                    row = {"run_id": run_id, "dataset": dataset, "method": method, "chunk_size": chunk_label, "max_queries": args.max_queries,
                           "n_docs": len(corpus), "n_queries": len(queries), "time_seconds": round(retrieval_time, 3),
                           "config_elapsed_seconds": round(cfg_elapsed, 3)}

                    row.update(metrics)
                    summary_rows.append(row)

                    config_payload = {"metadata": {"run_id": run_id, "dataset": dataset, "method": method, "chunk_size": chunk_label,
                                                   "max_queries": args.max_queries, "n_docs": len(corpus), "n_queries": len(queries),
                                                   "top_k": TOP_K,"eval_top_k": EVAL_TOP_K, "retrieval_time_seconds": round(retrieval_time, 3),
                                                   "config_elapsed_seconds": round(cfg_elapsed, 3)},
                                    "metrics": metrics}
                    
                    cfg_name = f"dataset={dataset}__method={method}__chunk={chunk_label}__maxq={args.max_queries}.json"
                    _write_json(configs_dir / cfg_name, config_payload)
                    completed += 1
                    print(f"done | MRR={metrics.get('MRR', 0.0):.4f} | Recall@10={metrics.get('Recall@10', 0.0):.4f}")
                
                except Exception as e:
                    msg = f"Config failed for dataset={dataset}, chunk={chunk_label}, method={method}: {e}"
                    print(msg)
                    errors.append({"stage": "run", "dataset": dataset, "chunk_size": chunk_label, "method": method, "error": str(e)})

    # Aggregate outputs
    _write_json(aggregates_dir / "summary.json", summary_rows)
    _write_csv(aggregates_dir / "summary.csv", summary_rows)
    _write_json(aggregates_dir / "errors.json", errors)
    _write_json(aggregates_dir / "run_metadata.json", {"run_id": run_id, "datasets": args.datasets, "methods": args.methods,
                                                       "chunk_sizes": [_safe_chunk_label(c) for c in chunk_grid],
                                                       "max_queries": args.max_queries, "configs_total": full_grid_size,
                                                       "configs_completed": completed, "configs_failed": len(errors),
                                                       "output_dir": str(run_dir)})
    
    print("\n=== Grid run complete ===")
    print(f"Completed: {completed}/{full_grid_size}")
    print(f"Failed: {len(errors)}")
    print(f"Summary JSON: {aggregates_dir / 'summary.json'}")
    print(f"Summary CSV:  {aggregates_dir / 'summary.csv'}")
    print(f"Errors JSON:  {aggregates_dir / 'errors.json'}")

if __name__ == "__main__":
    main()
