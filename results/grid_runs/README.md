- `20260328_191443` is the MiniLM run on CPU. The CPU script does not record a device. Output path: `C:\Users\nodemachine\Downloads\COMP5801\project\...`.

- `20260401_193340` is the heavier-encoder run on CPU, same Windows path. Sparse metrics match `20260328_191443` exactly. Dense and hybrid metrics and runtime do not. The result JSON never stores the model name. The other model named in `experiments/run_grid.py` is the commented `sentence-transformers/all-mpnet-base-v2`.

- `20260406_235803` is the MiniLM grid again, rerun with `gpu/run_grid.py` on the machine at `/home/rapids/project/...`. `run_metadata.json` records `device: cuda` and `batch_size: 128`. Dense and hybrid rows say `cuda`; sparse rows say `cpu`. Quality matches `20260328_191443`. It is not the heavier-encoder run.
