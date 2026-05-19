"""Cohere `rerank-english-v3.0` cross-encoder reranker with graceful fallback to input order on failure."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

# sys.path bootstrap for direct script invocation.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document


DEFAULT_MODEL = "rerank-english-v3.0"
DEFAULT_TOP_K = 5


class CohereReranker:
    """
    Re-score fused retrieval candidates with Cohere's cross-encoder reranker.

    Unlike the upstream retrievers (which embed query and corpus independently),
    the reranker reads `(query, chunk)` as a single input pair and produces a
    relevance score that's far more discriminating than cosine similarity. This
    is what separates "actually about termination" chunks from "merely mentions
    breach" chunks that BGE-small can't reliably distinguish.

    Initialization is lazy and tolerant:
      - If `api_key` is None, reads `COHERE_API_KEY` from the environment.
      - If no key is found, or the `cohere` package isn't importable, or the
        client constructor raises, the reranker enters "fallback mode": calls
        to `.rerank()` return `docs[:top_k]` in input order with no
        `rerank_score` annotation. The pipeline keeps working — useful for
        offline demos, missing-credential cases, and Cohere outages.

    Check `.is_available()` to know which mode you're in.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        self._client = None
        self._init_error: Optional[str] = None

        if not self.api_key:
            self._init_error = "no COHERE_API_KEY found in environment"
            return

        try:
            import cohere
        except ImportError as e:
            self._init_error = f"`cohere` package not installed: {e}"
            return

        try:
            self._client = cohere.ClientV2(api_key=self.api_key)
        except Exception as e:
            self._init_error = f"failed to construct Cohere client: {type(e).__name__}: {e}"

    def is_available(self) -> bool:
        """True iff a usable Cohere client was constructed at init time."""
        return self._client is not None

    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Document]:
        """
        Re-sort `docs` by Cohere relevance score; return up to `top_k` Documents
        with `rerank_score: float` (range [0, 1]) added to each Document's
        metadata. Returns fresh Documents — input docs are not mutated, and
        upstream stage scores (`rrf_score`, `bm25_score`, `similarity_score`)
        survive through the merge.

        Fallback behavior (no crash, ever):
          - empty `docs` → returns `[]`
          - `top_k <= 0` → returns `[]`
          - Cohere unavailable (no key / no package / construct error)
            → returns `docs[:top_k]` in input order, no rerank_score
          - Cohere API call raises (network, rate-limit, auth)
            → returns `docs[:top_k]` in input order, no rerank_score

        Failures print a one-line diagnostic so they're visible in logs without
        bringing the pipeline down.
        """
        if not docs or top_k <= 0:
            return []

        if self._client is None:
            print(
                f"[reranker] Cohere unavailable ({self._init_error}); "
                f"returning input order as-is."
            )
            return docs[:top_k]

        try:
            response = self._client.rerank(
                model=self.model,
                query=query,
                documents=[d.page_content for d in docs],
                top_n=min(top_k, len(docs)),
            )
        except Exception as e:
            print(
                f"[reranker] Cohere API call failed "
                f"({type(e).__name__}: {e}); returning input order as-is."
            )
            return docs[:top_k]

        reranked: List[Document] = []
        for result in response.results:
            original = docs[result.index]
            reranked.append(
                Document(
                    page_content=original.page_content,
                    metadata={
                        **original.metadata,
                        "rerank_score": float(result.relevance_score),
                    },
                )
            )
        return reranked


# ---------------------------------------------------------------------------
# Test block: full pipeline (BM25 + vector → RRF → rerank). Prints before/after.
# ---------------------------------------------------------------------------


def _format_hit(rank: int, doc: Document) -> str:
    meta = doc.metadata
    parts = []
    if "rerank_score" in meta:
        parts.append(f"rerank={meta['rerank_score']:.3f}")
    if "rrf_score" in meta:
        parts.append(f"rrf={meta['rrf_score']:.5f}")
    scores_str = " | ".join(parts) if parts else "(no scores)"
    clause = meta.get("clause_number") or "(no clause)"
    snippet = doc.page_content[:160].replace("\n", " ")
    return (
        f"  #{rank} {scores_str} | {meta['source']} p.{meta['page']} | clause={clause!r}\n"
        f"      {snippet}"
    )


if __name__ == "__main__":
    # Load .env so COHERE_API_KEY is picked up when running standalone.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    from ingestion.indexer import CHROMA_DIR, load_index
    from retrieval.bm25_retriever import BM25Retriever
    from retrieval.rrf_fusion import rrf_fuse
    from retrieval.vector_retriever import VectorRetriever

    print("[reranker] loading indices...")
    try:
        bm25_index, _chroma = load_index()
    except FileNotFoundError as e:
        print(f"[reranker] {e}")
        print("[reranker] run `python ingestion/indexer.py` first.")
        sys.exit(1)

    bm25 = BM25Retriever(index=bm25_index)
    vector = VectorRetriever(CHROMA_DIR)
    reranker = CohereReranker()

    print(f"[reranker] Cohere available: {reranker.is_available()}")
    if not reranker.is_available():
        print(f"[reranker] reason: {reranker._init_error}")
        print("[reranker] (test will demonstrate the fallback path)")
    print()

    query = "what are the termination conditions?"
    print(f"[reranker] query: {query!r}\n")

    bm25_hits = bm25.retrieve(query, top_k=10)
    vector_hits = vector.retrieve(query, top_k=10)
    fused = rrf_fuse(bm25_hits, vector_hits, top_k=10)

    print("=" * 72)
    print(f"BEFORE rerank — RRF top {min(5, len(fused))}")
    print("=" * 72)
    for rank, doc in enumerate(fused[:5], start=1):
        print(_format_hit(rank, doc))
        print()

    reranked = reranker.rerank(query, fused, top_k=5)

    print("=" * 72)
    print(f"AFTER rerank — top {len(reranked)}")
    print("=" * 72)
    for rank, doc in enumerate(reranked, start=1):
        print(_format_hit(rank, doc))
        print()

    # Edge case: empty input always returns []
    assert reranker.rerank(query, [], top_k=5) == [], "expected [] for empty input"
    print("[reranker] empty-input edge case: ✓ returns []")
