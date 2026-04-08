# COMP5801 RAG Retrieval Comparison Project

**Option A: Empirical Evaluation** — Systematic comparison of retrieval strategies (dense, sparse, hybrid) in Retrieval-Augmented Generation, with analysis of chunk size and granularity effects.

**Designed for CPU-only execution** — No GPU required.

**Note:** This is the project root folder. Run all commands from this directory.

---

## Environment Setup

### 1. Create Virtual Environment (Recommended)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Install PyTorch (CPU-only)

For CPU-only (no GPU):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import sentence_transformers; import rank_bm25; import datasets; print('OK')"
```

---

## Folder Structure

```
project/
├── README.md                 # This file
├── requirements.txt         # Python dependencies
│
├── report/                  # LaTeX report (Final Report)
│   ├── report.tex           # Main report source
│   ├── report.bib           # Bibliography
│   └── jmlr2e.sty          # JMLR style (copy from project root if needed)
│
├── src/                     # Source code
│   ├── __init__.py
│   ├── config.py            # Configuration constants
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset_loader.py # Dataset loading (BEIR/HuggingFace)
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── dense.py         # Dense retrieval (sentence-transformers)
│   │   ├── sparse.py        # Sparse retrieval (BM25)
│   │   └── hybrid.py        # Hybrid retrieval (RRF fusion)
│   ├── evaluation.py        # Metrics: Recall@k, MRR, NDCG
│   └── run_experiments.py   # Main experiment runner
│
├── experiments/             # Experiment scripts
│   ├── run_all.py          # Run fixed suite (selected datasets)
│   └── run_grid.py         # Run full Cartesian grid with timestamped outputs
├── scripts/                 # Utility scripts
│   ├── generate_figures.py        # Plots from results/metrics/*_summary.json
│   └── generate_figures_grid.py   # Comprehensive plots from one grid run
│
├── data/                    # Downloaded datasets (auto-created)
│   └── .gitkeep
│
├── results/                 # Experiment outputs (auto-created)
│   ├── metrics/             # JSON/CSV results
│   └── figures/             # Plots
│
└── notebooks/               # Optional Jupyter notebooks
    └── .gitkeep
```

---

## How to Run

### Quick Start (Single Dataset, Default Config)

```bash
python -m src.run_experiments --dataset nfcorpus --max_queries 100
```

### Full Experiment Suite

```bash
python experiments/run_all.py
```

This runs:
- Dense retrieval (all-MiniLM-L6-v2)
- Sparse retrieval (BM25)
- Hybrid retrieval (RRF)
- Chunk size variation (128, 256, 512 tokens)
- On nfcorpus and fiqa datasets (CPU-friendly sizes)

### Comprehensive Grid Run (CPU)

```bash
python experiments/run_grid.py
```

This command works with no extra arguments. Defaults are:
- Datasets: all values in `src/config.py` (`AVAILABLE_DATASETS`)
- Methods: dense, sparse, hybrid
- Chunk sizes: original (no chunking) + all values in `CHUNK_SIZES`
- `max_queries`: 200
- Output root: `results/grid_runs/<timestamp>/`

Example with overrides:

```bash
python experiments/run_grid.py --datasets nfcorpus scifact --methods sparse hybrid --chunk_sizes 128 256 --max_queries 300
```

### Custom Run

```bash
python -m src.run_experiments \
  --dataset nfcorpus \
  --methods dense sparse hybrid \
  --chunk_sizes 128 256 512 \
  --max_queries 200 \
  --output_dir results
```

---

## Output

- **results/metrics/** — per-dataset summary JSON from `src/run_experiments.py` / `run_all.py`
- **results/figures/** — Comparison plots
- **results/full_summary.json** — aggregated results from `experiments/run_all.py`
- **results/grid_runs/<timestamp>/** — full-grid outputs from `experiments/run_grid.py`:
  - `configs/*.json` (one JSON per dataset/method/chunk/maxq config)
  - `aggregate/summary.json`
  - `aggregate/summary.csv`
  - `aggregate/errors.json`
  - `aggregate/run_metadata.json`

---

## Report

The final report is in `report/report.tex`. Compile:

```bash
cd report
pdflatex report
bibtex report
pdflatex report
pdflatex report
```

Or use Overleaf: upload `report.tex`, `report.bib`, and `jmlr2e.sty` (copy from project root if needed).

**Before submitting:** Replace "Author Name" and "author@carleton.ca" in `report.tex`.

---

## Dataset Notes

- **nfcorpus**: ~3.6k docs, ~323 queries — fast on CPU
- **fiqa**: ~57k docs, ~648 queries — moderate
- **scifact**: ~5k docs, ~300 queries — small, fast

Larger datasets (MS MARCO, NQ) will be slower on CPU; reduce `--max_queries` for testing.

---

## CPU-Only Design Choices

| Component | Choice | Reason |
|-----------|---------|--------|
| Embedding model | all-MiniLM-L6-v2 | Small (~80MB), fast on CPU |
| Dataset size | nfcorpus, fiqa, scifact | Smaller BEIR datasets |
| Batch size | 32 | Conservative for CPU memory |
| LLM | Not included | Optional; use API for E2E eval |
| ColBERT | Not included | GPU-heavy; dense+BM25+hybrid sufficient |

---

## References

- Lewis et al. (2020) — RAG: Retrieval-Augmented Generation
- Karpukhin et al. (2020) — Dense Passage Retrieval
- Robertson & Zaragoza (2009) — BM25
- BEIR benchmark — https://github.com/beir-cellar/beir
