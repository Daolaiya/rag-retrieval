# Project scripts inventory (`Proposal/project/`)

Inventory of **every project-authored Python file** under `Proposal/project/` (excluding `project/venv/`, which is third-party). There are **12** such files.

---

## Entry points (run directly)

| Path | Purpose | Primary inputs | Primary outputs | Who uses it |
|------|---------|----------------|-----------------|-------------|
| **`experiments/run_all.py`** | Batch driver: runs the main experiment runner for several datasets, then merges JSON summaries. | Hard-coded list of datasets (`nfcorpus`, `scifact`), chunk sizes, `max_queries`; spawns `python -m src.run_experiments ...` with `cwd=project/`. | Reads `project/results/metrics/*_summary.json`; writes **`project/results/full_summary.json`**. | You run it manually (`python experiments/run_all.py`). It does **not** import other project modules; it only **subprocess-invokes** `src.run_experiments`. |
| **`src/run_experiments.py`** | Main experiment loop: load BEIR data → run dense / sparse / hybrid retrieval → compute metrics → save JSON. | **CLI:** `--dataset`, `--methods`, `--chunk_sizes`, `--max_queries`, `--output_dir` (default `project/results`), `--no_chunk_variation`. **Data:** downloaded/cached BEIR via `ir_datasets` (not in `project/data/`). | **`{output_dir}/metrics/{dataset}_summary.json`** (list of metric dicts). Console logs. | Run manually (`python -m src.run_experiments` or `python src/run_experiments.py`). **Used as a subprocess** by `experiments/run_all.py`. **Imports:** `src.config`, `src.data.dataset_loader`, `src.evaluation`, `src.retrieval.{dense,sparse,hybrid}`. |
| **`scripts/generate_figures.py`** | Plot report-ready figures from per-dataset summaries. | Reads **`project/results/metrics/*_summary.json`** (falls back to `sample_results.json` only if no summaries exist). | **`project/results/metrics/figures/*.pdf`** (method comparison and chunk-impact plots across datasets for MRR/Recall@10/NDCG@10). | Run manually (`python scripts/generate_figures.py`). **Not imported** by other project scripts. |
| **`scripts/generate_figures_grid.py`** | Comprehensive plotting for one completed grid run. | Reads **`project/results/grid_runs/<run_id>/aggregate/summary.json`** (defaults to run `20260328_191443` if present, else latest run). | **`project/results/grid_runs/<run_id>/figures/*.pdf`** (original-corpus comparisons, chunk-impact, heatmaps, runtime tradeoffs). | Run manually (`python scripts/generate_figures_grid.py`). **Not imported** by other project scripts. |

---

## Library / package modules (imported, not meant as top-level scripts)

| Path | Purpose | Primary inputs | Primary outputs | Who imports / uses it |
|------|---------|----------------|-----------------|----------------------|
| **`src/config.py`** | Central constants: default embedding model, chunk sizes, `TOP_K`, metric k-list, BM25/RRF params. | (none at runtime; edited in code) | (none; values consumed by other modules) | **`src/run_experiments.py`**. Indirectly affects **`DenseRetriever` / `HybridRetriever`** via `DEFAULT_DENSE_MODEL`. |
| **`src/evaluation.py`** | Retrieval metrics: Recall@k, MRR, NDCG@k. | Dict of ranked lists per query + qrels dict. | Dict of aggregated metrics. | **`src/run_experiments.py`** (`evaluate_retrieval`). |
| **`src/data/dataset_loader.py`** | Load BEIR-style corpus/queries/qrels; optional word-token chunking and qrel expansion for chunks. | Dataset name string; optional `chunk_size`, `max_queries`. Uses **`ir_datasets`** (and optionally **`datasets`** if `ir_datasets` missing). | Three dicts: `corpus`, `queries`, `qrels`. | **`src/run_experiments.py`** (`load_beir_dataset`). |
| **`src/retrieval/dense.py`** | Dense retrieval with **sentence-transformers** + cosine similarity on CPU. | Corpus dict (title+text), queries dict. | Ranked `(doc_id, score)` lists per query. | **`src/run_experiments.py`** (`DenseRetriever`). **`src/retrieval/hybrid.py`** (instantiates `DenseRetriever`). |
| **`src/retrieval/sparse.py`** | BM25 (**rank_bm25**) retrieval. | Same corpus/query dicts as dense. | Ranked lists per query. | **`src/run_experiments.py`** (`SparseRetriever`). **`src/retrieval/hybrid.py`** (`SparseRetriever`). |
| **`src/retrieval/hybrid.py`** | Hybrid retrieval: RRF fusion of dense + sparse rankings. | Same as dense/sparse. | Fused ranked lists. | **`src/run_experiments.py`** (`HybridRetriever`). |
| **`src/__init__.py`** | Package docstring / marks `src` as a package. | — | — | Loaded implicitly when Python imports **`src.*`** (e.g. `python -m src.run_experiments`). Nothing in the repo imports `import src` alone for logic. |
| **`src/data/__init__.py`** | Re-exports `load_beir_dataset`, `chunk_documents` from `dataset_loader`. | — | — | Optional convenience import (`from src.data import ...`). **`run_experiments` bypasses this** and imports `dataset_loader` directly. |
| **`src/retrieval/__init__.py`** | Re-exports retriever classes. | — | — | Optional convenience import. **`run_experiments` bypasses this** and imports `dense` / `sparse` / `hybrid` directly. |

---

## Not counted as “project scripts” (but present)

- **`project/venv/`** — virtual environment; thousands of third-party `.py` files (pip, etc.). Not part of your assignment code.
- **`project/0_test.ipynb`** (if present) — a Jupyter notebook; not wired into the pipeline unless you use it yourself.

---

## Data / cache (not produced by these scripts into `project/data/`)

- BEIR corpora downloaded by **`ir_datasets`** live under the user cache, typically **`%USERPROFILE%\.ir_datasets\`** (e.g. `...\beir\...`), not under `project/data/`.

---

## Quick dependency graph (who calls whom)

- **`run_all.py`** → subprocess → **`src/run_experiments.py`**
- **`run_experiments.py`** → **`config`**, **`data/dataset_loader`**, **`evaluation`**, **`retrieval/{dense,sparse,hybrid}`**
- **`hybrid.py`** → **`dense.py`**, **`sparse.py`**
- **`generate_figures.py`** → reads JSON under **`results/metrics/`** only (no imports from `src/`)
- **`generate_figures_grid.py`** → reads one grid summary under **`results/grid_runs/<run_id>/aggregate/summary.json`**
