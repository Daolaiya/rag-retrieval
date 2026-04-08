"""
Dataset loader for BEIR-style retrieval benchmarks.
Uses ir_datasets when available; falls back to Hugging Face datasets.
Supports chunk size variation.
"""

import re
from typing import Any

try:
    import ir_datasets
    HAS_IR_DATASETS = True
except ImportError:
    HAS_IR_DATASETS = False

try:
    from datasets import load_dataset as hf_load
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

def _simple_tokenize(text: str) -> list[str]:
    """Simple word tokenizer for chunk size estimation."""
    return re.findall(r"\b\w+\b", str(text).lower())

def chunk_text(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    """Split text into chunks of approximately chunk_size tokens."""
    tokens = _simple_tokenize(text)
    if len(tokens) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(" ".join(chunk_tokens))
        start = end - overlap if overlap > 0 else end

    return chunks

def chunk_documents(corpus: dict[str, dict[str, Any]], chunk_size: int,
                    overlap: int = 0) -> tuple[dict[str,dict[str, Any]],dict[str, str]]:
    """
    Chunk documents by token count.
    Returns (new_corpus, chunk_to_original_doc).
    """
    new_corpus = {}
    chunk_to_original = {}

    for doc_id, doc in corpus.items():
        text = doc.get("text", "") or ""
        title = doc.get("title", "") or ""
        full_text = f"{title} {text}".strip() if title else text

        chunks = chunk_text(full_text, chunk_size, overlap)

        if not chunks:
            new_corpus[doc_id] = {"text": full_text or " ", "title": title}
            chunk_to_original[doc_id] = doc_id
        else:
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                new_corpus[chunk_id] = {"text": chunk, "title": title}
                chunk_to_original[chunk_id] = doc_id

    return new_corpus, chunk_to_original

def _adapt_qrels_for_chunks(qrels: dict[str, dict[str, int]], chunk_to_original: dict[str, str]) -> dict[str, dict[str, int]]:
    """Expand qrels: if doc was relevant, all its chunks are relevant."""
    new_qrels = {}
    for qid, rel_docs in qrels.items():
        new_qrels[qid] = {}
        for chunk_id, orig_id in chunk_to_original.items():
            if orig_id in rel_docs:
                new_qrels[qid][chunk_id] = rel_docs[orig_id]
    return new_qrels

def _load_via_ir_datasets(dataset_name: str) -> tuple[dict, dict, dict]:
    """Load using ir_datasets. Use /test split for qrels."""
    ds_map = {"nfcorpus": "beir/nfcorpus/test",
              "fiqa": "beir/fiqa/test",
              "scifact": "beir/scifact/test",
              "arguana": "beir/arguana"
            }
    ds_name = ds_map.get(dataset_name, f"beir/{dataset_name}/test")
    dataset = ir_datasets.load(ds_name)

    corpus = {}
    for doc in dataset.docs_iter():
        doc_id = str(doc.doc_id)
        text = getattr(doc, "text", "") or ""
        title = getattr(doc, "title", "") or ""
        corpus[doc_id] = {"text": text, "title": title}

    queries = {}
    for query in dataset.queries_iter():
        qid = str(query.query_id)
        queries[qid] = getattr(query, "text", "") or ""

    qrels = {}
    for qrel in dataset.qrels_iter():
        qid = str(qrel.query_id)
        doc_id = str(qrel.doc_id)
        score = int(getattr(qrel, "relevance", 1))
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][doc_id] = score

    return corpus, queries, qrels

def _load_via_huggingface(dataset_name: str) -> tuple[dict, dict, dict]:
    """Load using Hugging Face datasets (BeIR format)."""
    hf_name = f"BeIR/{dataset_name}"
    corpus_ds = hf_load(hf_name, "corpus", trust_remote_code=True)
    queries_ds = hf_load(hf_name, "queries", trust_remote_code=True)
    try:
        qrels_ds = hf_load(hf_name, "qrels", trust_remote_code=True)
    except Exception:
        qrels_ds = hf_load(hf_name, "test", trust_remote_code=True)

    corpus = {}
    for row in corpus_ds["train"] if "train" in corpus_ds else corpus_ds[list(corpus_ds.keys())[0]]:
        doc_id = str(row.get("_id", row.get("id", "")))
        text = row.get("text", row.get("content", "")) or ""
        title = row.get("title", "") or ""
        corpus[doc_id] = {"text": text, "title": title}

    queries = {}
    q_split = queries_ds["train"] if "train" in queries_ds else queries_ds[list(queries_ds.keys())[0]]
    for row in q_split:
        qid = str(row.get("_id", row.get("id", "")))
        queries[qid] = row.get("text", row.get("content", "")) or ""

    qrels = {}
    qr_split = qrels_ds["train"] if "train" in qrels_ds else qrels_ds[list(qrels_ds.keys())[0]]
    for row in qr_split:
        qid = str(row.get("query-id", row.get("query_id", "")))
        doc_id = str(row.get("corpus-id", row.get("corpus_id", row.get("doc_id", ""))))
        score = int(row.get("score", row.get("label", 1)))
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][doc_id] = score

    return corpus, queries, qrels

def load_beir_dataset(dataset_name: str, chunk_size: int | None = None, max_queries: int | None = None,
                      cache_dir: str | None = None) -> tuple[dict, dict, dict]:
    """
    Load a BEIR dataset.

    Args:
        dataset_name: e.g. "nfcorpus", "fiqa", "scifact"
        chunk_size: If set, chunk documents. None = use original docs.
        max_queries: Limit queries for faster experiments.
        cache_dir: Unused.

    Returns:
        corpus: {doc_id: {"text": str, "title": str}}
        queries: {query_id: str}
        qrels: {query_id: {doc_id: relevance_score}}
    """
    if HAS_IR_DATASETS:
        corpus, queries, qrels = _load_via_ir_datasets(dataset_name)
    elif HAS_DATASETS:
        corpus, queries, qrels = _load_via_huggingface(dataset_name)
    else:
        raise ImportError("Install ir-datasets or datasets: pip install ir-datasets")

    valid_qids = {q for q in queries if q in qrels}
    queries = {k: v for k, v in queries.items() if k in valid_qids}
    qrels = {k: v for k, v in qrels.items() if k in valid_qids}

    if chunk_size is not None and chunk_size > 0:
        corpus, chunk_to_orig = chunk_documents(corpus, chunk_size)
        qrels = _adapt_qrels_for_chunks(qrels, chunk_to_orig)
        for qid in list(qrels.keys()):
            qrels[qid] = {d: s for d, s in qrels[qid].items() if d in corpus}

    if max_queries is not None:
        qids = list(queries.keys())[:max_queries]
        queries = {k: queries[k] for k in qids}
        qrels = {k: qrels[k] for k in qids if k in qrels}

    return corpus, queries, qrels
