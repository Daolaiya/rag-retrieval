"""
Configuration constants for RAG retrieval experiments.
CPU-only design.
"""

# Embedding models (small, CPU-friendly)
DENSE_MODEL_NAMES = ["sentence-transformers/all-MiniLM-L6-v2",  # ~80MB, fast
                     "sentence-transformers/all-mpnet-base-v2",  # larger, better quality
                    ]

DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# BEIR datasets (HuggingFace) - CPU-friendly sizes
AVAILABLE_DATASETS = ["nfcorpus", "fiqa", "scifact", "arguana", "dbpedia-entity"]

# Chunk sizes (in tokens) for TA feedback: granularity analysis
CHUNK_SIZES = [128, 256, 512]

# Retrieval
TOP_K = 100  # Retrieve top-k for evaluation
EVAL_TOP_K = [1, 5, 10, 100]  # Metrics at these k values

# BM25 parameters
BM25_K1 = 1.5
BM25_B = 0.75

# Hybrid: Reciprocal Rank Fusion
RRF_K = 60  # RRF constant

# Batch size for embeddings (CPU-friendly)
BATCH_SIZE = 32

# Random seed
SEED = 42
