"""Top-k BM25 keyword retrieval over an in-memory `BM25Index` built by `indexer.load_index`/`build_index`."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Allow direct script invocation (`python retrieval/bm25_retriever.py`) by
# bootstrapping the project root onto sys.path before first-party imports.
# No-op when run via `python -m retrieval.bm25_retriever` or imported.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

# Reuse the indexer's tokenizer (private name, deliberate single source of truth)
# so query terms are normalized identically to the corpus at index time.
# If this ever changes, only one file gets edited.
from ingestion.indexer import BM25Index, _tokenize


@dataclass
class BM25Retriever:
    """
    Top-k BM25 retrieval over an in-memory `BM25Index`.

    Construction takes the `BM25Index` dataclass (`bm25: BM25Okapi` +
    `documents: List[Document]` in corpus order) that `indexer.build_index`
    and `indexer.load_index` return — no pickle, BM25 is rebuilt on startup.
    """

    index: BM25Index

    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        """
        Score every chunk against `query` and return up to `top_k` Documents
        in descending BM25 score order, with `bm25_score` (float) added to
        each Document's metadata.

        Zero / negative scored chunks are filtered out — a chunk with no term
        overlap is not a useful hit regardless of rank. If no chunk has a
        positive score (or the query has no tokens), returns `[]`.

        The returned Documents are fresh objects (not aliases of the corpus
        Documents) so the `bm25_score` annotation doesn't leak into the
        shared in-memory corpus or the Chroma store.
        """
        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self.index.bm25.get_scores(tokens)
        # Rank all chunks by descending score; slice to top_k.
        ranked_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]

        results: List[Document] = []
        for i in ranked_indices:
            score = float(scores[i])
            if score <= 0.0:
                continue
            original = self.index.documents[i]
            results.append(
                Document(
                    page_content=original.page_content,
                    metadata={**original.metadata, "bm25_score": score},
                )
            )
        return results


if __name__ == "__main__":
    print("[bm25_retriever] loading persisted indices via indexer.load_index()...")
    try:
        from ingestion.indexer import load_index

        bm25_index, _chroma = load_index()
    except FileNotFoundError as e:
        print(f"[bm25_retriever] {e}")
        print("[bm25_retriever] run `python ingestion/indexer.py` first to build the index.")
        sys.exit(1)

    retriever = BM25Retriever(index=bm25_index)

    query = "what are the termination conditions?"
    print(f"[bm25_retriever] query: {query!r}")
    print(f"[bm25_retriever] corpus size: {len(bm25_index.documents)} chunks\n")

    hits = retriever.retrieve(query, top_k=3)
    if not hits:
        print("[bm25_retriever] no hits (all scores were zero — no keyword overlap)")
        sys.exit(0)

    print(f"[bm25_retriever] top {len(hits)} hit(s):")
    for rank, d in enumerate(hits, start=1):
        snippet = d.page_content[:220].replace("\n", " ")
        clause = d.metadata.get("clause_number") or "(no clause detected)"
        print(
            f"  #{rank} bm25_score={d.metadata['bm25_score']:.3f} | "
            f"{d.metadata['source']} p.{d.metadata['page']} | "
            f"clause={clause!r}"
        )
        print(f"      {snippet}")
        print()
