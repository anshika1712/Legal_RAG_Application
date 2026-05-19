"""Page-then-chunk splitter: per-page Documents from pdf_parser → 300-token chunks with 50 overlap."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional, Union

# Allow running this file directly (`python ingestion/chunker.py ...`) by putting
# the project root on sys.path before any first-party imports. When invoked via
# `python -m ingestion.chunker` or imported as a package, this is a no-op.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tiktoken
from langchain_core.documents import Document
# Import the submodule directly to avoid `langchain_text_splitters/__init__.py`,
# which eagerly loads the HTML splitter (pulls in nltk → sklearn) that we don't need.
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

from ingestion.pdf_parser import parse_pdfs


# Per `.cursorrules`: 200-400 tokens, 50 overlap. We pick the middle of the band
# so most chunks land cleanly inside it without hitting the cap.
CHUNK_SIZE_TOKENS = 300
CHUNK_OVERLAP_TOKENS = 50

# tiktoken's cl100k_base is OpenAI's encoding; not exact for Claude but typically
# within ~10%. Plenty accurate for chunk-sizing decisions.
_TIKTOKEN_ENCODING = "cl100k_base"
_ENCODER = tiktoken.get_encoding(_TIKTOKEN_ENCODING)


def _token_len(text: str) -> int:
    return len(_ENCODER.encode(text))


# Best-effort clause/section detection. Patterns are anchored at the START of the
# chunk's first non-blank line — never matched inside body text — to avoid picking
# up section *references* like "as provided under Clause 3". Tried in order; first
# match wins. Precision > recall: a wrong clause_number creates misleading
# citations, so we only fire on patterns that look unambiguously heading-like.
_CLAUSE_PATTERNS: List[re.Pattern[str]] = [
    # "Section 8.2", "Sec. 8.2", "Article IV", "Art. 4", "Clause 12.3"
    # at the start of the line only.
    re.compile(
        r"(?:Section|Sec\.?|Article|Art\.?|Clause)\s+"
        r"(?:[IVXLCM]+|\d+(?:\.\d+)*[A-Za-z]?)\b",
        re.IGNORECASE,
    ),
    # Numbered heading: requires EITHER a multi-level dotted number ("11.1",
    # "14.3", "8.5.2") OR an integer with a trailing period ("4.", "14."),
    # followed by a space and a capital letter. This excludes bare integers like
    # "1 Rice" or "4 Buyers" which are list quantities, not headings, while
    # keeping every real numeric heading we've seen in the corpus.
    re.compile(r"(?:\d+\.\d+(?:\.\d+)*\.?|\d+\.)(?=\s+[A-Z])"),
]


def _detect_clause_number(text: str) -> Optional[str]:
    """Best-effort extract a clause/section label from the start of the chunk's first non-blank line."""
    head: Optional[str] = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            head = stripped[:200]
            break
    if head is None:
        return None

    for pattern in _CLAUSE_PATTERNS:
        match = pattern.match(head)  # anchored at start; no body-text references
        if match:
            return match.group(0).strip().rstrip(".")
    return None


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split per-page Documents (from `pdf_parser.parse_pdfs`) into ~300 token chunks
    with 50 token overlap. Each chunk inherits `source`, `page`, `contract_type`
    from its parent page; `clause_number` is best-effort regex (or None).

    `collection` is intentionally NOT set here — the indexer (`base`) and
    session_store (`session`) stamp it at ingest time.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=_TIKTOKEN_ENCODING,
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Document] = []
    for page_doc in documents:
        sub_texts = splitter.split_text(page_doc.page_content)
        for idx, sub in enumerate(sub_texts):
            metadata = {
                "source": page_doc.metadata.get("source"),
                "page": page_doc.metadata.get("page"),
                "contract_type": page_doc.metadata.get("contract_type"),
                "clause_number": _detect_clause_number(sub),
            }
            # Disambiguate sub-chunks of the same page; absent for single-chunk pages.
            if len(sub_texts) > 1:
                metadata["chunk_index"] = idx
            chunks.append(Document(page_content=sub, metadata=metadata))
    return chunks


def chunk_pdfs(source: Union[str, Path, Iterable[Any]]) -> List[Document]:
    """
    Convenience wrapper: parse PDFs from a folder or uploads, then chunk.

    Args:
        source: a folder path containing `*.pdf` files, or an iterable of uploads
            where each item is `(filename, bytes)` or a file-like object exposing
            `.name` and `.read()` (e.g. Streamlit's `UploadedFile`).

    Returns:
        Flat `List[Document]` of ~300 token chunks with metadata
        `{source, page, contract_type, clause_number}` plus optional `chunk_index`.
    """
    page_docs = parse_pdfs(source)
    return chunk_documents(page_docs)


if __name__ == "__main__":
    from collections import Counter

    # Usage:
    #   python ingestion/chunker.py                         # defaults to tests/sample_pdfs/
    #   python ingestion/chunker.py path/to/folder          # chunk all PDFs in a folder
    #   python ingestion/chunker.py path/to/contract.pdf    # chunk a single PDF via the upload path
    arg = sys.argv[1] if len(sys.argv) > 1 else "tests/sample_pdfs"
    target = Path(arg)

    print(f"[chunker] target: {target}")

    if target.is_file() and target.suffix.lower() == ".pdf":
        with target.open("rb") as fh:
            docs = chunk_pdfs([(target.name, fh.read())])
    else:
        docs = chunk_pdfs(target)

    print(f"[chunker] produced {len(docs)} chunk(s)")
    if not docs:
        print(
            "[chunker] no chunks produced — drop a PDF into "
            f"{target if target.is_dir() else target.parent} and re-run."
        )
        sys.exit(0)

    titled = sum(1 for d in docs if d.metadata.get("clause_number"))
    print(f"[chunker] chunks with detected clause_number: {titled}/{len(docs)}")

    by_source = Counter(d.metadata["source"] for d in docs)
    print("[chunker] chunks per source:")
    for src, n in by_source.most_common():
        print(f"  {src}: {n}")

    token_lens = [_token_len(d.page_content) for d in docs]
    print(
        f"[chunker] token length: min={min(token_lens)} | "
        f"median={sorted(token_lens)[len(token_lens) // 2]} | "
        f"max={max(token_lens)} | avg={sum(token_lens) // len(token_lens)}"
    )

    print("\n[chunker] sample chunks (first 3):")
    for i, d in enumerate(docs[:3]):
        print("-" * 72)
        print(
            f"#{i + 1} source={d.metadata['source']} | "
            f"page={d.metadata['page']} | "
            f"contract_type={d.metadata['contract_type']}"
        )
        print(f"   clause_number: {d.metadata['clause_number']!r}")
        if "chunk_index" in d.metadata:
            print(f"   chunk_index:   {d.metadata['chunk_index']}")
        print(f"   tokens:        {_token_len(d.page_content)}")
        snippet = d.page_content[:300].replace("\n", " ")
        print(f"   text[:300]:    {snippet}")
