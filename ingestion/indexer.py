"""Build/load the BM25 index and persistent ChromaDB `base_contracts` collection from chunked Documents."""

from __future__ import annotations

import functools
import hashlib
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Allow direct script invocation (`python ingestion/indexer.py`) by bootstrapping
# the project root onto sys.path before any first-party imports. No-op when run
# as `python -m ingestion.indexer` or imported as a package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Silence chromadb's anonymized telemetry posts on every startup; must be set
# before chromadb is imported (langchain_chroma re-exports it).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from ingestion.chunker import chunk_pdfs


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default I/O paths. `RAW_DIR` is the existing scaffold location for base PDFs;
# `CHROMA_DIR` matches the spec's `data/processed/chroma/` target.
RAW_DIR = PROJECT_ROOT / "data" / "raw_contracts"
CHROMA_DIR = PROJECT_ROOT / "data" / "processed" / "chroma"

COLLECTION_NAME = "base_contracts"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Tag stamped onto every base-contract chunk's metadata at index time. Per
# `.cursorrules`, retrievers can filter on `collection in {"base", "session"}`.
BASE_COLLECTION_TAG = "base"


@dataclass
class BM25Index:
    """In-memory BM25 index: the `BM25Okapi` plus the documents in corpus order."""

    bm25: BM25Okapi
    documents: List[Document]


def _tokenize(text: str) -> List[str]:
    """Lowercased whitespace tokenization; sufficient for BM25 over English contracts."""
    return text.lower().split()


def _chunk_id(doc: Document) -> str:
    """Deterministic 16-hex-char chunk ID. Stable across rebuilds when content + key metadata are unchanged."""
    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page", "")
    chunk_index = doc.metadata.get("chunk_index", 0)
    key = f"{source}|{page}|{chunk_index}|{doc.page_content}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _stamp_collection(docs: List[Document], tag: str) -> List[Document]:
    """
    Return a new list of Documents with `collection=tag` added to metadata, and
    `clause_number=None` replaced with `""` (ChromaDB requires primitive,
    non-None metadata values).
    """
    stamped: List[Document] = []
    for d in docs:
        meta = dict(d.metadata)
        meta["collection"] = tag
        if meta.get("clause_number") is None:
            meta["clause_number"] = ""
        stamped.append(Document(page_content=d.page_content, metadata=meta))
    return stamped


@functools.lru_cache(maxsize=1)
def _make_embeddings() -> HuggingFaceEmbeddings:
    """Construct the BGE embedder with cosine-normalized vectors (BGE author recommendation).

    Cached at module level: the BGE-small model is ~133MB and takes ~1s to
    load. Without this cache, every call to `session_store.get_session_index`
    triggers a fresh model load — meaningful Streamlit-cold-start tax. The
    cache is per-process, so all callers share the same singleton instance.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def _build_bm25(documents: List[Document]) -> BM25Index:
    """Whitespace-tokenize each chunk and build a BM25Okapi index aligned with the document order."""
    tokenized = [_tokenize(d.page_content) for d in documents]
    return BM25Index(bm25=BM25Okapi(tokenized), documents=documents)


def _build_chroma(documents: List[Document], persist_dir: Path) -> Chroma:
    """Build & persist a Chroma collection using BAAI/bge-small-en-v1.5 embeddings."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    ids = [_chunk_id(d) for d in documents]
    return Chroma.from_documents(
        documents=documents,
        embedding=_make_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_dir),
        ids=ids,
    )


def build_index(
    raw_dir: Optional[Path] = None,
    chroma_dir: Optional[Path] = None,
) -> Tuple[BM25Index, Chroma]:
    """
    Run the full ingestion pipeline:
      1. parse + chunk every `*.pdf` under `raw_dir` (via `chunker.chunk_pdfs`)
      2. stamp `collection: "base"` and sanitize metadata
      3. build the BM25 index in memory
      4. embed chunks with BGE and persist the Chroma collection to `chroma_dir`

    `chroma_dir` is wiped before rebuild so the persisted collection always
    reflects the current state of `raw_dir` (no stale chunks, no duplicate IDs).

    Args:
        raw_dir: folder of base contract PDFs (defaults to `RAW_DIR`).
        chroma_dir: persistence directory for ChromaDB (defaults to `CHROMA_DIR`).

    Returns:
        `(bm25_index, chroma_collection)` — pass these to the retrieval layer.
    """
    raw_dir = raw_dir or RAW_DIR
    chroma_dir = chroma_dir or CHROMA_DIR

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw contracts directory not found: {raw_dir}")

    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)

    print(f"[indexer] parsing + chunking PDFs from: {raw_dir}")
    t0 = time.perf_counter()
    chunks = chunk_pdfs(raw_dir)
    print(f"[indexer] produced {len(chunks)} chunk(s) in {time.perf_counter() - t0:.2f}s")
    if not chunks:
        raise RuntimeError(f"No chunks produced from {raw_dir} — is the folder empty?")

    docs = _stamp_collection(chunks, BASE_COLLECTION_TAG)

    print("[indexer] building BM25 index...")
    t0 = time.perf_counter()
    bm25_index = _build_bm25(docs)
    print(f"[indexer] BM25 built in {time.perf_counter() - t0:.2f}s")

    print(f"[indexer] embedding chunks with {EMBEDDING_MODEL} → {chroma_dir}")
    t0 = time.perf_counter()
    chroma = _build_chroma(docs, chroma_dir)
    print(f"[indexer] Chroma collection '{COLLECTION_NAME}' built in {time.perf_counter() - t0:.2f}s")

    return bm25_index, chroma


def load_index(chroma_dir: Optional[Path] = None) -> Tuple[BM25Index, Chroma]:
    """
    Load a previously persisted ChromaDB collection from disk and rebuild BM25
    from the same chunks (no pickle — sub-second for our chunk count).

    Args:
        chroma_dir: persistence directory used by a prior `build_index` call.

    Returns:
        `(bm25_index, chroma_collection)`.

    Raises:
        FileNotFoundError: if `chroma_dir` does not exist.
        RuntimeError: if the collection exists but is empty.
    """
    chroma_dir = chroma_dir or CHROMA_DIR
    if not chroma_dir.is_dir():
        raise FileNotFoundError(
            f"Chroma persistence not found at {chroma_dir}. Run build_index() first."
        )

    chroma = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_make_embeddings(),
        persist_directory=str(chroma_dir),
    )

    # Fetch chunks back from Chroma to rebuild BM25. Faster than re-parsing PDFs
    # (no pdfplumber pass, no chunking) and means `load_index` doesn't require
    # the raw PDFs to still be on disk.
    fetched = chroma.get(include=["documents", "metadatas"])
    documents = [
        Document(page_content=text, metadata=metadata)
        for text, metadata in zip(fetched["documents"], fetched["metadatas"])
    ]
    if not documents:
        raise RuntimeError(
            f"Chroma collection at {chroma_dir} is empty. Run build_index() first."
        )

    bm25_index = _build_bm25(documents)
    return bm25_index, chroma


if __name__ == "__main__":
    print(f"[indexer] PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"[indexer] RAW_DIR:      {RAW_DIR}")
    print(f"[indexer] CHROMA_DIR:   {CHROMA_DIR}")
    print()

    bm25_index, chroma = build_index()

    print()
    bm25_count = len(bm25_index.documents)
    chroma_count = len(chroma.get(include=[])["ids"])

    print(f"[indexer] BM25 corpus size:       {bm25_count}")
    print(f"[indexer] Chroma collection size: {chroma_count}")
    print(f"[indexer] Chroma collection name: {COLLECTION_NAME}")
    print(f"[indexer] persisted at:           {CHROMA_DIR}")

    assert chroma_count == bm25_count, (
        f"Mismatch: BM25 has {bm25_count} chunks but Chroma has {chroma_count}"
    )

    # Sanity check both indices respond to a plausible query.
    query = "confidentiality and non-disclosure"
    print(f"\n[indexer] smoke query against both indices: {query!r}")

    query_tokens = _tokenize(query)
    scores = bm25_index.bm25.get_scores(query_tokens)
    top_bm25 = sorted(range(len(scores)), key=lambda i: -scores[i])[:3]
    print("  BM25 top 3:")
    for i in top_bm25:
        d = bm25_index.documents[i]
        snippet = d.page_content[:120].replace("\n", " ")
        print(
            f"    score={scores[i]:.3f} | {d.metadata['source']} "
            f"p.{d.metadata['page']} :: {snippet}"
        )

    print("  Chroma (vector) top 3:")
    for d in chroma.similarity_search(query, k=3):
        snippet = d.page_content[:120].replace("\n", " ")
        print(
            f"    {d.metadata['source']} p.{d.metadata['page']} :: {snippet}"
        )
