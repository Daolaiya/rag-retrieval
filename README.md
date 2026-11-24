# COMP5801 RAG Retrieval Comparison Project
**Option A: Empirical Evaluation** — I compare dense, sparse, and hybrid retrieval for RAG-style setups and look at how chunk size / granularity changes metrics.

The scripts in `experiments/` and `src/` are the CPU pipeline. `gpu/` is the CUDA variant of the grid runner; it produced `results/grid_runs/20260406_235803` (MiniLM on GPU, not a second encoder). Run commands from this folder (`project/`).

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
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

This matches `experiments/run_grid.py`. The scripts in `gpu/` need a CUDA build of PyTorch instead. See `gpu/README.md`: `gpu/run_grid.py` still imports the retrievers from `src/retrieval/`, so those three files only reproduce `20260406_235803` when they occupy `experiments/run_grid.py`, `src/retrieval/dense.py`, and `src/retrieval/hybrid.py`.

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import sentence_transformers; import rank_bm25; import datasets; print('OK')"
```

## Folder Structure
```
project/
├── README.md                # This file
├── requirements.txt         # Python dependencies
│
├── gpu/                     # CUDA copies of the grid runner and dense/hybrid retrievers
│   ├── README.md
│   ├── run_grid.py
│   ├── dense.py
│   └── hybrid.py
│
├── report_submitted/        # Overleaf source and graded Report.pdf
│   ├── report.tex           # Submitted source (calls the GPU run MPNet)
│   ├── report.bib
│   ├── jmlr2e.sty
│   ├── Report.pdf
│   └── figures/
│
├── report_corrected/        # Same paper with CPU/GPU labels corrected
│   ├── report.tex
│   ├── report.bib
│   └── jmlr2e.sty
│
├── src/                     # Source code
│   ├── __init__.py
│   ├── config.py            # Defaults (dense model for run_experiments / run_all)
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset_loader.py # BEIR / HuggingFace loading
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── dense.py         # Dense (sentence-transformers)
│   │   ├── sparse.py        # BM25
│   │   └── hybrid.py        # RRF hybrid
│   ├── evaluation.py        # Recall@k, MRR, NDCG
│   └── run_experiments.py   # One dataset at a time (run_all calls this)
│
├── experiments/
│   ├── run_grid.py          # Full grid — main run
│   ├── run_all.py           # Shorter fixed suite → metrics + full_summary.json
│   ├── run_compare.py       # Compare two grid runs
│   └── report_tables.ipynb  # Optional: check numbers against the report
│
├── scripts/
│   ├── generate_figures.py        # Plots from results/metrics/*_summary.json
│   └── generate_figures_grid.py   # Plots from a grid run’s aggregate/summary.json
│
├── data/                    # Downloaded datasets (created when you run)
│   └── .gitkeep
│
├── results/
│   ├── metrics/             # Summaries from run_experiments / run_all
│   ├── grid_runs/           # Three completed grids; see README.md in that folder
│   └── comparisons/         # Output from run_compare.py
│
└── notebooks/
    └── .gitkeep
```

## How to Run
### Primary: full grid (`run_grid.py`)
For the report I rely on a **full grid** over datasets, methods, and chunk settings:
```bash
python experiments/run_grid.py
```

No flags needed. By default it uses everything in `src/config.py` (`AVAILABLE_DATASETS`), methods `dense` / `sparse` / `hybrid`, chunk sizes in `CHUNK_SIZES` plus an **original** (no chunking) run, and `max_queries=200`. Each run goes to `results/grid_runs/<timestamp>/` with `aggregate/summary.json`, `summary.csv`, and per-config JSON under `configs/`.

Override example:

```bash
python experiments/run_grid.py --datasets nfcorpus scifact --methods sparse hybrid --chunk_sizes 128 256 --max_queries 300
```

### Quick: single dataset (`run_experiments`)
Good for debugging or a fast check:
```bash
python -m src.run_experiments --dataset nfcorpus --max_queries 100
```

### Smaller suite (`run_all.py`)
`run_all.py` calls `run_experiments` on a smaller fixed list of datasets and then builds `results/full_summary.json`. Faster than the full grid; it’s not the same thing as the main BEIR grid above.

```bash
python experiments/run_all.py
```

### Compare two grid runs (`run_compare.py`)
After I have two full grids, I compare them with:
```bash
python experiments/run_compare.py
```

With no flags it reads `RUN_A_ID` and `RUN_B_ID` from the top of `run_compare.py` and writes CSVs under `results/comparisons/<RUN_A_ID>_vs_<RUN_B_ID>/`. Those constants are the CPU MiniLM run `20260328_191443` and the GPU MiniLM run `20260406_235803`. If the second run isn’t there yet, the script can fabricate a placeholder folder for testing the pipeline—don’t use that for real numbers.

### Figures
Plots from `run_experiments` / `run_all` outputs:
```bash
python scripts/generate_figures.py
```

→ `results/metrics/figures/`

Plots from one grid run:
```bash
python scripts/generate_figures_grid.py
```

Defaults to `PREFERRED_RUN_ID` at the top of `generate_figures_grid.py`; if that folder isn’t there it falls back to the latest valid run under `results/grid_runs/`. Output: `results/grid_runs/<run_id>/figures/`.

### Manual tweaks (embedding + which run to use)
I didn’t wire everything through argparse. To switch the dense model or point scripts at a specific grid folder, edit the constants near the tops of these files:

| Goal | File |
|------|------|
| Dense model for **`run_grid.py`** | `experiments/run_grid.py` — `DEFAULT_DENSE_MODEL`. Active value is MiniLM. `all-mpnet-base-v2` is the commented alternative. This overrides the grid run separately from `config.py`. |
| Dense model for **`run_experiments` / `run_all`** | `src/config.py` — `DEFAULT_DENSE_MODEL` |
| Which grid run to plot | `scripts/generate_figures_grid.py` — `PREFERRED_RUN_ID` |
| Which two grids to compare | `experiments/run_compare.py` — `RUN_A_ID`, `RUN_B_ID` |

### Custom `run_experiments` example
```bash
python -m src.run_experiments \
  --dataset nfcorpus \
  --methods dense sparse hybrid \
  --chunk_sizes 128 256 512 \
  --max_queries 200 \
  --output_dir results
```

## Output
| Path | From |
|------|------|
| `results/metrics/*_summary.json` | `src/run_experiments.py`, `experiments/run_all.py` |
| `results/metrics/figures/` | `scripts/generate_figures.py` |
| `results/full_summary.json` | `experiments/run_all.py` |
| `results/grid_runs/<timestamp>/` | `experiments/run_grid.py` |
| `results/grid_runs/<timestamp>/figures/` | `scripts/generate_figures_grid.py` |
| `results/comparisons/<A>_vs_<B>/` | `experiments/run_compare.py` |

## Report
`report_submitted/` is the Overleaf source and the graded `Report.pdf`. That `.tex` calls the GPU run MPNet. `report_corrected/report.tex` is the same paper with those labels corrected. The figure files for a local build are in `report_submitted/figures/`.

## Dataset Notes
- **nfcorpus**: ~3.6k docs, ~323 queries — quick on CPU
- **fiqa**: ~57k docs, ~648 queries — heavier
- **scifact**: ~5k docs, ~300 queries — small

If you try huge BEIR sets, lower `--max_queries` while testing.

## Completed grids
Labels for the three folders under `results/grid_runs/` are in `results/grid_runs/README.md`.

| Run | What it is |
|-----|------------|
| `20260328_191443` | MiniLM on CPU. No `device` field. |
| `20260401_193340` | Heavier encoder on CPU. Sparse metrics match the MiniLM CPU run. Dense and hybrid do not. The result files do not store the model name. |
| `20260406_235803` | Same MiniLM grid on GPU (`device: cuda`, `batch_size` 128 for dense and hybrid). Quality matches `20260328_191443`. |

`experiments/run_grid.py` still defaults to MiniLM. `all-mpnet-base-v2` is only a comment in that file. Switching it and running the grid again is how a new encoder run is started. It does not describe `20260406_235803`.

## Design Choices (CPU pipeline)
| Piece | What I used | Why |
|-------|-------------|-----|
| Embeddings | MiniLM for the CPU baseline and the GPU rerun. One heavier-encoder CPU grid is `20260401_193340`. | MiniLM is the fast default. The heavier run is the encoder comparison; the GPU run is not. |
| Datasets | `nfcorpus`, `fiqa`, `scifact`, `arguana`, `dbpedia-entity` | The list in `src/config.py` |
| Batch size | 32 on the CPU scripts; 128 on the GPU run | 32 keeps CPU memory reasonable. The GPU metadata records 128. |
| LLM | Not in this repo | I only evaluate retrieval |
| ColBERT | Not implemented | Late-interaction retrieval was outside this scope |

## References
- Lewis et al. (2020) — RAG
- Karpukhin et al. (2020) — Dense Passage Retrieval
- Robertson & Zaragoza (2009) — BM25
- BEIR — https://github.com/beir-cellar/beir
