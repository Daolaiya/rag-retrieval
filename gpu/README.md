# GPU scripts

These three files are the CUDA variant of the grid runner. They produced `results/grid_runs/20260406_235803`. They are not the entry points used by the rest of this tree.

- `run_grid.py` — same grid as `experiments/run_grid.py`, plus `--device` and `--batch_size`. It defaults to CUDA when available and records `device` and `batch_size` in the results. The dense model constant is `sentence-transformers/all-MiniLM-L6-v2`.
- `dense.py` — passes `device` into encoding, normalizes embeddings, and scores queries in one batch.
- `hybrid.py` — passes `device` and `batch_size` through to the dense retriever.

The copies under `experiments/` and `src/retrieval/` are the CPU scripts. `src/retrieval/dense.py` accepts a `device` argument and does not pass it into `encode`.

`run_grid.py` imports `src.retrieval.dense` and `src.retrieval.hybrid`. Running `python gpu/run_grid.py` as it sits uses those CPU retrievers. These three files are the preserved GPU sources. They behave as they did for `20260406_235803` when `run_grid.py` is the file at `experiments/run_grid.py` and `dense.py` and `hybrid.py` are the files at `src/retrieval/dense.py` and `src/retrieval/hybrid.py`.
