"""End-to-end smoke test: load_index → BM25 + vector → RRF → Cohere rerank → Claude generate, run against the real corpus.

Manual script (not a pytest test). Useful for:
  - Spot-checking retrieval quality after tuning changes.
  - Reproducing the rerank issue we saw earlier (EMPLOYMENT 11.1 dropping out
    of top 5 on termination queries).
  - Sanity-checking the no-hallucination guardrail on out-of-corpus queries.
  - Serving as a fixture for the Streamlit UI in step 8.

Run:
    python tests/e2e_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# sys.path bootstrap for direct script invocation.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from generation.llm import NOT_FOUND_SENTINEL, ClaudeGenerator
from ingestion.indexer import CHROMA_DIR, load_index
from retrieval.bm25_retriever import BM25Retriever
from retrieval.reranker import CohereReranker
from retrieval.rrf_fusion import rrf_fuse
from retrieval.vector_retriever import VectorRetriever


# Queries chosen to stress different parts of the pipeline.
# Each query is paired with a brief note explaining what aspect it exercises;
# the note is printed alongside the query so output is self-documenting.
QUERIES: List[tuple[str, str]] = [
    (
        "What are the termination conditions in the employment agreement?",
        "the known weak query — rerank previously dropped EMPLOYMENT clause 11.1 out of top 5",
    ),
    (
        "How long is the non-compete obligation in effect after employment ends?",
        "tests the non-compete PDF; 'non-compete' is a multi-word proper-noun-like phrase that should favor BM25",
    ),
    (
        "How is the royalty payment calculated under the royalty agreement?",
        "tests the royalty PDF; 'calculated' is a semantic match (concept), not a literal token in the doc",
    ),
    (
        "What confidentiality obligations apply to the employee?",
        "tests the employment PDF; 'confidentiality' has both exact-token and semantic matches across the corpus",
    ),
    (
        "What is the corporate income tax rate in India?",
        "out-of-corpus question — should trigger the no-hallucination guardrail",
    ),
]

# Pipeline knobs. Kept as constants here so we can A/B them without code edits.
RETRIEVER_TOP_K = 10  # passed to both bm25.retrieve and vector.retrieve, and to rrf_fuse
RERANK_TOP_K = 5      # how many chunks the reranker keeps for the LLM
LOW_CONFIDENCE_FLOOR = 0.30  # rerank score below this = LLM may struggle


def _print_chunks_summary(chunks: List[Document], n: int = 5) -> None:
    """Print top-n chunks with their stage scores + first 140 chars of content."""
    for i, doc in enumerate(chunks[:n], start=1):
        meta = doc.metadata
        scores = []
        if "rerank_score" in meta:
            scores.append(f"rerank={meta['rerank_score']:.3f}")
        if "rrf_score" in meta:
            scores.append(f"rrf={meta['rrf_score']:.5f}")
        scores_str = " | ".join(scores) if scores else "(no scores)"
        clause = meta.get("clause_number") or "(no clause)"
        snippet = doc.page_content[:140].replace("\n", " ")
        print(f"    #{i} {scores_str} | {meta['source']} p.{meta['page']} | clause={clause!r}")
        print(f"        {snippet}...")


def run_pipeline(
    query: str,
    note: str,
    bm25: BM25Retriever,
    vector: VectorRetriever,
    reranker: CohereReranker,
    generator: ClaudeGenerator,
) -> None:
    """Run one query end-to-end and print stage-by-stage diagnostics + final answer."""
    print()
    print("=" * 80)
    print(f"QUERY: {query}")
    print(f"NOTE:  {note}")
    print("=" * 80)

    # Stage 1 — parallel retrieval
    bm25_hits = bm25.retrieve(query, top_k=RETRIEVER_TOP_K)
    vector_hits = vector.retrieve(query, top_k=RETRIEVER_TOP_K)
    print(f"\n[stage 1] retrieved {len(bm25_hits)} bm25 + {len(vector_hits)} vector hits")

    # Stage 2 — RRF fusion
    fused = rrf_fuse(bm25_hits, vector_hits, top_k=RETRIEVER_TOP_K)
    print(f"[stage 2] fused → {len(fused)} candidates")

    # Stage 3 — Cohere rerank
    reranked = reranker.rerank(query, fused, top_k=RERANK_TOP_K)
    print(f"[stage 3] reranked → {len(reranked)} chunks fed to LLM:")
    _print_chunks_summary(reranked, n=RERANK_TOP_K)

    # Confidence floor diagnostic — flag low-quality retrievals before reading the answer.
    rerank_scores = [d.metadata.get("rerank_score") for d in reranked]
    rerank_scores = [s for s in rerank_scores if s is not None]
    max_rerank = max(rerank_scores) if rerank_scores else None
    if max_rerank is not None and max_rerank < LOW_CONFIDENCE_FLOOR:
        print(
            f"\n[signal] ⚠ low retrieval confidence "
            f"(max rerank={max_rerank:.3f} < floor={LOW_CONFIDENCE_FLOOR}) "
            f"— LLM may struggle or refuse to answer"
        )

    # Stage 4 — LLM generation
    result = generator.generate(query, reranked)
    print(f"\n[stage 4] LLM answer:")
    indented = "    " + "\n    ".join(result["answer"].split("\n"))
    print(indented)

    print(f"\n[stage 4] parsed citations ({len(result['citations'])}):")
    if not result["citations"]:
        print("    (none)")
    for c in result["citations"]:
        clause = c.get("clause_number")
        clause_str = f", clause {clause}" if clause else ""
        print(f"    - {c['source']}, page {c['page']}{clause_str}")

    usage = result.get("usage") or {}
    if usage.get("input_tokens") is not None:
        print(f"\n[stage 4] tokens: input={usage['input_tokens']}, output={usage['output_tokens']}")

    # Quality classifier — gives the reader a one-line judgment per query.
    if NOT_FOUND_SENTINEL in result["answer"]:
        verdict = "guardrail fired (LLM refused to answer)"
    elif not result["citations"]:
        verdict = "⚠ answer present but no [N] citations parsed — investigate"
    else:
        verdict = f"answer with {len(result['citations'])} citation(s)"
    print(f"\n[signal] verdict: {verdict}")


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    print("[e2e] loading indices...")
    try:
        bm25_index, _chroma = load_index()
    except FileNotFoundError as e:
        print(f"[e2e] {e}")
        print("[e2e] run `python ingestion/indexer.py` first.")
        sys.exit(1)

    bm25 = BM25Retriever(index=bm25_index)
    vector = VectorRetriever(CHROMA_DIR)
    reranker = CohereReranker()
    generator = ClaudeGenerator()

    print(f"[e2e] Cohere available:  {reranker.is_available()}")
    print(f"[e2e] Claude available:  {generator.is_available()}")
    if not reranker.is_available() or not generator.is_available():
        print("[e2e] WARNING: one or more services unavailable — running with fallbacks")
    print(f"[e2e] config: retriever_top_k={RETRIEVER_TOP_K}, rerank_top_k={RERANK_TOP_K}")
    print(f"[e2e] running {len(QUERIES)} queries through full pipeline...")

    for query, note in QUERIES:
        run_pipeline(query, note, bm25, vector, reranker, generator)

    print()
    print("=" * 80)
    print(f"[e2e] done — ran {len(QUERIES)} queries through full pipeline")
    print("=" * 80)
