"""Reciprocal Rank Fusion (RRF) merging BM25 + vector retriever outputs into a single ranked list."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Set

# sys.path bootstrap for direct script invocation.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document


def rrf_fuse(
    bm25_results: List[Document],
    vector_results: List[Document],
    k: int = 60,
    top_k: int = 10,
) -> List[Document]:
    """
    Reciprocal Rank Fusion of two ranked Document lists.

    For each unique chunk (keyed by `page_content`), the RRF score is the sum
    over the lists where the chunk appears of `1 / (k + rank)` (1-based rank
    within that list). The default `k=60` is the standard damping constant
    from the original RRF paper (Cormack et al., 2009); larger values flatten
    the contribution of top ranks vs lower ranks.

    Either input list may be empty — fusion degrades gracefully to the
    non-empty list's RRF-scored ordering. If both are empty, returns `[]`.

    Within each input list, duplicate page_contents are dropped (first/best
    rank wins) as a defensive safety net. Cross-list duplicates are merged —
    that's the whole point of RRF.

    Metadata handling on the returned Documents:
      - canonical fields (source, page, clause_number, contract_type, collection)
        are carried through (first observation wins; these are identical across
        the two retrievers anyway since both index the same chunks)
      - retriever stage scores are preserved: `bm25_score` from BM25 input
        survives if present, `similarity_score` from vector input survives
        if present
      - `rrf_score` is added by this function
      - cursorrules convention: these stage scores get stripped before
        generation; the LLM only sees the canonical metadata

    Args:
        bm25_results: top-k Documents from `BM25Retriever.retrieve`.
        vector_results: top-k Documents from `VectorRetriever.retrieve`.
        k: RRF damping constant. Defaults to 60 (the canonical value).
        top_k: maximum number of fused Documents to return. Defaults to 10.

    Returns:
        A fresh `List[Document]` sorted by `rrf_score` descending, capped at
        `top_k`. Input Documents are not mutated.
    """
    if not bm25_results and not vector_results:
        return []

    aggregated: Dict[str, Dict[str, Any]] = {}

    for source_list in (bm25_results, vector_results):
        seen_in_list: Set[str] = set()
        for rank, doc in enumerate(source_list, start=1):
            content = doc.page_content
            if content in seen_in_list:
                continue  # within-list dedup: only count the best rank per list
            seen_in_list.add(content)

            contribution = 1.0 / (k + rank)

            if content not in aggregated:
                aggregated[content] = {
                    "rrf_score": 0.0,
                    "metadata": dict(doc.metadata),
                    "page_content": content,
                }
            else:
                # Cross-list merge: keep existing keys (canonical fields are
                # identical) and add this list's stage score (bm25_score or
                # similarity_score) that the first list didn't have.
                existing_meta = aggregated[content]["metadata"]
                for key, val in doc.metadata.items():
                    existing_meta.setdefault(key, val)

            aggregated[content]["rrf_score"] += contribution

    ranked = sorted(aggregated.values(), key=lambda x: -x["rrf_score"])[:top_k]

    return [
        Document(
            page_content=entry["page_content"],
            metadata={**entry["metadata"], "rrf_score": entry["rrf_score"]},
        )
        for entry in ranked
    ]


# ---------------------------------------------------------------------------
# Test block: realistic end-to-end fusion test plus the BM25-empty edge case.
# ---------------------------------------------------------------------------


def _format_hit(rank: int, doc: Document) -> str:
    """Pretty-print a fused hit with all preserved scores and a short snippet."""
    meta = doc.metadata
    parts = [f"rrf={meta['rrf_score']:.5f}"]
    if "bm25_score" in meta:
        parts.append(f"bm25={meta['bm25_score']:.3f}")
    if "similarity_score" in meta:
        parts.append(f"sim={meta['similarity_score']:.3f}")
    scores_str = " | ".join(parts)
    clause = meta.get("clause_number") or "(no clause)"
    snippet = doc.page_content[:160].replace("\n", " ")
    return (
        f"  #{rank} {scores_str} | {meta['source']} p.{meta['page']} "
        f"| clause={clause!r}\n      {snippet}"
    )


if __name__ == "__main__":
    from ingestion.indexer import CHROMA_DIR, load_index
    from retrieval.bm25_retriever import BM25Retriever
    from retrieval.vector_retriever import VectorRetriever

    print("[rrf_fusion] loading indices...")
    try:
        bm25_index, _chroma = load_index()
    except FileNotFoundError as e:
        print(f"[rrf_fusion] {e}")
        print("[rrf_fusion] run `python ingestion/indexer.py` first.")
        sys.exit(1)

    bm25 = BM25Retriever(index=bm25_index)
    vector = VectorRetriever(CHROMA_DIR)

    query = "what are the termination conditions?"
    print(f"[rrf_fusion] query: {query!r}\n")

    bm25_hits = bm25.retrieve(query, top_k=10)
    vector_hits = vector.retrieve(query, top_k=10)
    print(
        f"[rrf_fusion] retrieved {len(bm25_hits)} BM25 hit(s) and "
        f"{len(vector_hits)} vector hit(s) (top_k=10 per retriever)\n"
    )

    # ---------- Case 1: both lists populated ----------
    print("=" * 72)
    print("CASE 1: both lists populated — full hybrid fusion")
    print("=" * 72)
    fused = rrf_fuse(bm25_hits, vector_hits, top_k=5)
    print(f"[rrf_fusion] top {len(fused)} fused hit(s):\n")
    for rank, doc in enumerate(fused, start=1):
        print(_format_hit(rank, doc))
        print()

    # ---------- Case 2: BM25 empty (degrades to vector-only RRF ordering) ----------
    print("=" * 72)
    print("CASE 2: BM25 list empty — fusion degrades to vector-only")
    print("=" * 72)
    fused_v_only = rrf_fuse([], vector_hits, top_k=5)
    print(f"[rrf_fusion] top {len(fused_v_only)} fused hit(s):\n")
    for rank, doc in enumerate(fused_v_only, start=1):
        print(_format_hit(rank, doc))
        print()

    # ---------- Case 3: both empty (sanity) ----------
    assert rrf_fuse([], []) == [], "expected [] when both inputs are empty"
    print("=" * 72)
    print("CASE 3: both empty → returns [] (sanity check passed)")
    print("=" * 72)
