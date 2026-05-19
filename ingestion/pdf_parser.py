"""Parse PDFs (from a folder or uploaded bytes) into per-page langchain Documents."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, List, Tuple, Union

import pdfplumber
from langchain_core.documents import Document


# Filename keyword -> normalized contract_type.
# Order matters only when multiple keywords could match; we return the first hit.
_CONTRACT_TYPE_KEYWORDS: Tuple[Tuple[str, str], ...] = (
    ("nda", "nda"),
    ("non_disclosure", "nda"),
    ("non-disclosure", "nda"),
    ("nondisclosure", "nda"),
    ("employment", "employment"),
    ("non_compete", "non_compete"),
    ("non-compete", "non_compete"),
    ("noncompete", "non_compete"),
    ("royalty", "royalty"),
)


def infer_contract_type(filename: str) -> str:
    """Guess contract_type from a filename via keyword matching; fall back to 'unknown'."""
    name = Path(filename).stem.lower()
    for keyword, contract_type in _CONTRACT_TYPE_KEYWORDS:
        if keyword in name:
            return contract_type
    return "unknown"


def _extract_pages(pdf_source: Any, source_name: str) -> List[Document]:
    """Open one PDF (path or file-like) and emit one Document per non-empty page."""
    contract_type = infer_contract_type(source_name)
    docs: List[Document] = []

    with pdfplumber.open(pdf_source) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source_name,
                        "page": page_num,
                        "contract_type": contract_type,
                    },
                )
            )
    return docs


def parse_pdfs(source: Union[str, Path, Iterable[Any]]) -> List[Document]:
    """
    Parse PDFs into per-page langchain `Document`s.

    Args:
        source: one of
            - a folder path (str / Path) containing `*.pdf` files, or
            - an iterable of uploads where each item is either
              a `(filename, bytes)` tuple, or a file-like object exposing
              `.name` and `.read()` (e.g. Streamlit's `UploadedFile`).

    Returns:
        A flat list of Documents, one per non-empty page, each with metadata
        `{source, page, contract_type}`. `clause_number` and `collection`
        are intentionally left to the chunker and indexer / session_store.

    Raises:
        ValueError: if `source` is a path that is not an existing directory.
        TypeError: if an upload item is neither a `(filename, bytes)` tuple
            nor a file-like object with `.name` and `.read()`.
    """
    documents: List[Document] = []

    if isinstance(source, (str, Path)):
        folder = Path(source)
        if not folder.is_dir():
            raise ValueError(f"Folder not found or not a directory: {folder}")
        for pdf_path in sorted(folder.glob("*.pdf")):
            documents.extend(_extract_pages(str(pdf_path), pdf_path.name))
        return documents

    for item in source:
        if isinstance(item, tuple) and len(item) == 2:
            filename, raw_bytes = item
            if not isinstance(raw_bytes, (bytes, bytearray)):
                raise TypeError(
                    f"Upload tuple for {filename!r} must be (str, bytes); "
                    f"got bytes of type {type(raw_bytes).__name__}."
                )
            buffer = BytesIO(bytes(raw_bytes))
        elif hasattr(item, "name") and hasattr(item, "read"):
            filename = item.name
            buffer = BytesIO(item.read())
        else:
            raise TypeError(
                "Each upload must be a (filename, bytes) tuple or a file-like "
                "object with `.name` and `.read()` (e.g. Streamlit UploadedFile)."
            )
        documents.extend(_extract_pages(buffer, filename))

    return documents


if __name__ == "__main__":
    import sys

    # Usage:
    #   python ingestion/pdf_parser.py                          # defaults to tests/sample_pdfs/
    #   python ingestion/pdf_parser.py path/to/folder           # parse all PDFs in a folder
    #   python ingestion/pdf_parser.py path/to/contract.pdf     # parse a single PDF via the upload path
    arg = sys.argv[1] if len(sys.argv) > 1 else "tests/sample_pdfs"
    target = Path(arg)

    print(f"[pdf_parser] target: {target}")

    if target.is_file() and target.suffix.lower() == ".pdf":
        # Exercise the upload-bytes code path on a single file.
        with target.open("rb") as fh:
            docs = parse_pdfs([(target.name, fh.read())])
    else:
        docs = parse_pdfs(target)

    print(f"[pdf_parser] extracted {len(docs)} page-document(s)")

    if not docs:
        print(
            "[pdf_parser] no documents produced — drop a sample PDF into "
            f"{target if target.is_dir() else target.parent} and re-run."
        )
        sys.exit(0)

    for d in docs[:3]:
        print("-" * 72)
        print(
            f"source={d.metadata['source']} | "
            f"page={d.metadata['page']} | "
            f"contract_type={d.metadata['contract_type']}"
        )
        snippet = d.page_content[:300].replace("\n", " ")
        print(f"text[:300]: {snippet}")
