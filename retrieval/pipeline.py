"""Full RAG orchestration with mode-switch retrieval (base XOR session), RRF fusion, Cohere rerank, Claude generation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# sys.path bootstrap for direct script invocation.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_chroma import Chroma
from langchain_core.documents import Document

from generation.llm import ClaudeGenerator
from ingestion.indexer import BM25Index, load_index
from ingestion.session_store import get_session_index
from retrieval.bm25_retriever import BM25Retriever
from retrieval.reranker import CohereReranker
from retrieval.rrf_fusion import rrf_fuse
from retrieval.vector_retriever import VectorRetriever


# ---------------------------------------------------------------------------
# Pipeline knobs. Defaults kept here so the UI can call `answer()` without
# threading these through every call site. Override per-call if needed.
# ---------------------------------------------------------------------------
DEFAULT_RETRIEVER_TOP_K = 10  # per-retriever candidates AND RRF input cap
DEFAULT_RERANK_TOP_K = 5      # candidates the reranker keeps for the LLM

# Diagnostic floor for `low_confidence` flag. Calibrated from the e2e_smoke
# results: every in-corpus query had max rerank ≥ 0.425; the out-of-corpus
# (tax rate) query had max ≈ 0.001. 0.30 cleanly separates the two regimes.
LOW_CONFIDENCE_FLOOR = 0.30


# ---------------------------------------------------------------------------
# Module-level cache. Each is lazy-loaded on first use; subsequent calls hit
# the cache. Critical for Streamlit, which would otherwise rebuild everything
# on every script rerun.
# ---------------------------------------------------------------------------
_BASE_BM25: Optional[BM25Index] = None
_BASE_CHROMA: Optional[Chroma] = None
_RERANKER: Optional[CohereReranker] = None
_GENERATOR: Optional[ClaudeGenerator] = None


def _ensure_base_loaded() -> Tuple[BM25Index, Chroma]:
    """Load the base BM25 + Chroma once per process; subsequent calls return the cached pair."""
    global _BASE_BM25, _BASE_CHROMA
    if _BASE_BM25 is None or _BASE_CHROMA is None:
        _BASE_BM25, _BASE_CHROMA = load_index()
    return _BASE_BM25, _BASE_CHROMA


def _ensure_reranker() -> CohereReranker:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CohereReranker()
    return _RERANKER


def _ensure_generator() -> ClaudeGenerator:
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = ClaudeGenerator()
    return _GENERATOR


def reset_cache() -> None:
    """Drop all cached module-level state. Useful for tests; the UI should never need this."""
    global _BASE_BM25, _BASE_CHROMA, _RERANKER, _GENERATOR
    _BASE_BM25 = None
    _BASE_CHROMA = None
    _RERANKER = None
    _GENERATOR = None


def prewarm() -> None:
    """Eagerly populate all module-level caches: base index, BGE embeddings, Cohere client, Claude client.

    Call once at app startup (e.g. inside a Streamlit spinner) so the first
    user query doesn't pay the ~3s cold-start cost. Idempotent — subsequent
    calls return immediately because the underlying _ensure_* helpers short
    -circuit on already-cached state.
    """
    _ensure_base_loaded()
    _ensure_reranker()
    _ensure_generator()


# ---------------------------------------------------------------------------
# Mode-switch logic.
#
# Per .cursorrules: each query targets ONE collection.
#   - session_id provided AND session has docs → query session only
#   - otherwise (no id, or id given but session gone)  → query base only
# Never both. Merging would dilute relevance with unrelated template content.
# ---------------------------------------------------------------------------
def _select_indices(session_id: Optional[str]) -> Tuple[BM25Index, Chroma, str]:
    """Return `(bm25, chroma, mode)` for the collection this query will hit."""
    if session_id:
        session = get_session_index(session_id)
        if session is not None:
            bm25, chroma = session
            return bm25, chroma, "session"
        # Caller asked for session but it doesn't exist / was cleared.
        # Fall through to base; the `mode` field in the result tells the UI
        # what actually happened so it can show "session expired, querying base".
        print(
            f"[pipeline] session_id={session_id!r} requested but no live "
            f"session found; falling back to base mode."
        )

    bm25, chroma = _ensure_base_loaded()
    return bm25, chroma, "base"


def _source_files(docs: List[Document]) -> List[str]:
    """Unique source filenames across reranked docs, in first-seen order. UI uses this for 'Searched: a.pdf, b.pdf' badges."""
    seen: set = set()
    out: List[str] = []
    for d in docs:
        src = (d.metadata or {}).get("source")
        if src and src not in seen:
            seen.add(src)
            out.append(src)
    return out


# ---------------------------------------------------------------------------
# Public API: the one function the UI calls.
# ---------------------------------------------------------------------------
def answer(
    query: str,
    session_id: Optional[str] = None,
    retriever_top_k: int = DEFAULT_RETRIEVER_TOP_K,
    rerank_top_k: int = DEFAULT_RERANK_TOP_K,
) -> Dict[str, Any]:
    """
    Run the full RAG pipeline end-to-end on a single query.

    Mode (per .cursorrules): if `session_id` is provided AND that session has
    uploaded docs, search the session collection ONLY. Otherwise search base
    ONLY. The two collections never merge — see the rules file for rationale.

    Args:
        query: natural-language question.
        session_id: per-user session identifier; None for base mode.
        retriever_top_k: candidates each of BM25 and vector returns. Also the
            cap on RRF input. Defaults to 10.
        rerank_top_k: candidates the reranker keeps for the LLM. Defaults to 5.

    Returns: dict shaped like `ClaudeGenerator.generate`'s return plus four
    pipeline-level fields:
        {
            "answer": str,                  # natural-language answer with [N] citations
            "citations": [                  # parsed from [N] refs in the answer
                {"ref": int, "source": str, "page": ..., "clause_number": ...},
                ...
            ],
            "usage": {"input_tokens": int|None, "output_tokens": int|None},
            "mode": "base" | "session",     # which collection was queried
            "source_files": [str, ...],     # unique PDF names fed to the LLM
            "low_confidence": bool,         # True iff max rerank_score < 0.30
            "chunks": [Document, ...],      # the reranked chunks fed to the LLM,
                                            # in rerank order; carry rerank_score
                                            # in metadata for UI to display
        }
    """
    bm25_index, chroma, mode = _select_indices(session_id)

    bm25 = BM25Retriever(index=bm25_index)
    # Pass `chroma=` so VectorRetriever wraps the already-loaded collection
    # instead of re-loading the BGE model — important for session queries.
    vector = VectorRetriever(chroma=chroma)

    bm25_hits = bm25.retrieve(query, top_k=retriever_top_k)
    vector_hits = vector.retrieve(query, top_k=retriever_top_k)
    fused = rrf_fuse(bm25_hits, vector_hits, top_k=retriever_top_k)

    reranker = _ensure_reranker()
    reranked = reranker.rerank(query, fused, top_k=rerank_top_k)

    # Confidence diagnostic — UI surfaces this as a badge / disclaimer.
    rerank_scores = [
        d.metadata.get("rerank_score")
        for d in reranked
        if d.metadata.get("rerank_score") is not None
    ]
    max_rerank = max(rerank_scores) if rerank_scores else None
    low_confidence = bool(
        max_rerank is not None and max_rerank < LOW_CONFIDENCE_FLOOR
    )

    generator = _ensure_generator()
    result = generator.generate(query, reranked)

    # Pipeline-level fields (LLM doesn't see these — they're for the UI).
    result["mode"] = mode
    result["source_files"] = _source_files(reranked)
    result["low_confidence"] = low_confidence
    result["chunks"] = reranked
    return result


# ---------------------------------------------------------------------------
# Test block: exercise all three execution paths (base, session, fallback).
# ---------------------------------------------------------------------------
def _print_result(label: str, result: Dict[str, Any]) -> None:
    print(f"--- {label} ---")
    print(f"  mode:            {result['mode']}")
    print(f"  source_files:    {result['source_files']}")
    print(f"  low_confidence:  {result['low_confidence']}")
    print(f"  citations:       {len(result['citations'])}")
    usage = result.get("usage") or {}
    if usage.get("input_tokens") is not None:
        print(f"  tokens:          input={usage['input_tokens']} output={usage['output_tokens']}")
    print(f"  answer:")
    for line in result["answer"].split("\n"):
        print(f"    {line}")
    print()


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    from ingestion.session_store import clear_session, index_session_docs

    # ------------------------------------------------------------------
    # TEST 1: base mode (no session_id) — should query base corpus.
    # ------------------------------------------------------------------
    print("=" * 72)
    print("TEST 1: base mode (session_id=None)")
    print("=" * 72)
    base_query = "What are the termination conditions in the employment agreement?"
    result_base = answer(base_query)
    _print_result("RESULT", result_base)
    assert result_base["mode"] == "base", f"expected base mode, got {result_base['mode']!r}"
    print("  ✓ mode is 'base'")

    # ------------------------------------------------------------------
    # TEST 2: session mode — upload a PDF and query it in isolation.
    # The query is about royalty so the session-only restriction should be
    # clearly demonstrated (no base contracts get cited).
    # ------------------------------------------------------------------
    print("=" * 72)
    print("TEST 2: session mode (upload Royalty PDF then query)")
    print("=" * 72)
    session_id = "pipeline_test_001"
    sample_pdf = (
        Path(__file__).resolve().parent.parent
        / "tests"
        / "sample_pdfs"
        / "Royalty_Agreement.pdf"
    )
    if not sample_pdf.is_file():
        print(f"[pipeline] {sample_pdf} not found; skipping session test")
    else:
        clear_session(session_id)
        with sample_pdf.open("rb") as fh:
            pdf_bytes = fh.read()
        index_session_docs([(pdf_bytes, sample_pdf.name)], session_id)

        result_session = answer(
            "How is the royalty payment calculated?",
            session_id=session_id,
        )
        _print_result("RESULT", result_session)

        assert result_session["mode"] == "session", "expected session mode"
        # The whole point of mode-switch: session mode must NOT cite base files.
        for src in result_session["source_files"]:
            assert src == sample_pdf.name, (
                f"session-mode cited a non-uploaded file: {src!r} "
                f"(expected only {sample_pdf.name!r})"
            )
        print(f"  ✓ mode is 'session'")
        print(f"  ✓ source_files contains ONLY the uploaded file (no base leakage)")
        clear_session(session_id)

    # ------------------------------------------------------------------
    # TEST 3: graceful fallback — pass session_id that doesn't exist;
    # the pipeline should silently route to base and tell us via `mode`.
    # ------------------------------------------------------------------
    print("=" * 72)
    print("TEST 3: graceful fallback (bogus session_id → base mode)")
    print("=" * 72)
    result_fallback = answer(
        base_query,
        session_id="this_session_does_not_exist_xyz",
    )
    print(f"  mode after fallback: {result_fallback['mode']}")
    assert result_fallback["mode"] == "base", "expected silent fallback to base"
    print(f"  ✓ silent fallback to base when session_id is unknown")

    print("\n[pipeline] all assertions passed ✓")
