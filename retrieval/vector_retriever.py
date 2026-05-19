"""Top-k dense vector retrieval over a persisted ChromaDB collection (works for base or session)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Union

# Allow direct script invocation (`python retrieval/vector_retriever.py`) by
# bootstrapping the project root onto sys.path before first-party imports.
# No-op when run as `python -m retrieval.vector_retriever` or imported.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Silence chromadb's anonymized telemetry posts; must be set before chromadb is
# imported (langchain_chroma re-exports it).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Reuse the indexer's embedding-model identifier and the canonical collection
# name so query-time embeddings MUST match index-time embeddings. A mismatch
# here would produce meaningless similarity scores (different vector spaces).
from ingestion.indexer import COLLECTION_NAME, EMBEDDING_MODEL


class VectorRetriever:
    """
    Top-k dense vector retrieval over a persisted ChromaDB collection,
    embedding queries with BAAI/bge-small-en-v1.5.

    Two construction modes:

    1. **From persist_dir + collection_name** (default) — instantiates a fresh
       Chroma client and its own HuggingFaceEmbeddings. Used for the base
       collection at app startup, or whenever you want a standalone retriever.

    2. **From an existing Chroma instance** (`chroma=` kwarg) — wraps an
       already-built Chroma (e.g. the one returned by
       `session_store.get_session_index` or `indexer.load_index`). Avoids
       re-loading the BGE model (~133MB) for every session query. Used by
       `retrieval/pipeline.py` to keep cold-start time bounded.

    Per the mode-switch design (.cursorrules): one retriever instance per
    collection — base and session never share. The orchestration layer picks
    which retriever to invoke based on whether the session has uploaded docs.
    """

    def __init__(
        self,
        persist_dir: Optional[Union[str, Path]] = None,
        collection_name: str = COLLECTION_NAME,
        chroma: Optional[Chroma] = None,
    ) -> None:
        if chroma is not None:
            # Pre-built Chroma path — caller already loaded the collection and
            # its embedder, just wrap it.
            self.chroma: Chroma = chroma
            return

        if persist_dir is None:
            raise ValueError(
                "VectorRetriever requires either `persist_dir` or `chroma`."
            )

        persist_dir = Path(persist_dir)
        if not persist_dir.is_dir():
            raise FileNotFoundError(
                f"Chroma persistence not found at {persist_dir}. "
                f"Run `python ingestion/indexer.py` first to build the index."
            )

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.chroma = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Document]:
        """
        Embed `query` and return up to `top_k` Documents in descending similarity
        order, with `similarity_score` (float) added to each Document's metadata.

        ChromaDB's default metric is L2 (Euclidean) distance. With BGE's
        `normalize_embeddings=True`, vectors are unit-length so L2 distance lives
        in [0, 2] where 0 = identical and 2 = opposite direction. We convert
        to `similarity = 1.0 - distance / 2.0`, giving roughly [0, 1] where
        higher = more similar.

        Returned Documents are fresh objects — the `similarity_score` annotation
        does NOT mutate the shared docs inside the Chroma store. Same pattern
        as `bm25_retriever.py`.

        Returns:
            `[]` if the collection is empty, the query is empty/whitespace, or
            ChromaDB returns no results.
        """
        if not query or not query.strip():
            return []

        results_with_distance = self.chroma.similarity_search_with_score(query, k=top_k)
        if not results_with_distance:
            return []

        retrieved: List[Document] = []
        for doc, distance in results_with_distance:
            similarity = 1.0 - float(distance) / 2.0
            retrieved.append(
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "similarity_score": similarity},
                )
            )
        return retrieved


if __name__ == "__main__":
    from ingestion.indexer import CHROMA_DIR

    print(f"[vector_retriever] loading Chroma collection {COLLECTION_NAME!r} from: {CHROMA_DIR}")
    try:
        retriever = VectorRetriever(CHROMA_DIR)
    except FileNotFoundError as e:
        print(f"[vector_retriever] {e}")
        sys.exit(1)

    query = "what are the termination conditions?"
    print(f"[vector_retriever] query: {query!r}\n")

    hits = retriever.retrieve(query, top_k=3)
    if not hits:
        print("[vector_retriever] no hits (collection empty or no results)")
        sys.exit(0)

    print(f"[vector_retriever] top {len(hits)} hit(s):")
    for rank, d in enumerate(hits, start=1):
        snippet = d.page_content[:220].replace("\n", " ")
        clause = d.metadata.get("clause_number") or "(no clause detected)"
        print(
            f"  #{rank} similarity_score={d.metadata['similarity_score']:.3f} | "
            f"{d.metadata['source']} p.{d.metadata['page']} | "
            f"clause={clause!r}"
        )
        print(f"      {snippet}")
        print()
