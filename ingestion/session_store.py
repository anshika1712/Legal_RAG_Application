"""Per-session PDF upload → BM25 + Chroma indexing, isolated by session_id (no disk persistence beyond Chroma; cleared on session end)."""

from __future__ import annotations

import os
import re
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# sys.path bootstrap for direct script invocation.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Silence chromadb's anonymized telemetry posts on every startup; must be set
# before chromadb is imported (langchain_chroma re-exports it).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain_chroma import Chroma
from langchain_core.documents import Document

from ingestion.chunker import chunk_documents
from ingestion.indexer import (
    CHROMA_DIR,
    BM25Index,
    _build_bm25,
    _chunk_id,
    _make_embeddings,
    _stamp_collection,
)
from ingestion.pdf_parser import parse_pdfs


# ---------------------------------------------------------------------------
# Limits (per .cursorrules: max 10MB/file, max 5 files/session).
# ---------------------------------------------------------------------------
MAX_FILES_PER_SESSION = 5
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Chroma collection naming.
SESSION_COLLECTION_PREFIX = "session_"
SESSION_COLLECTION_TAG = "session"

# session_id format. Combined with the prefix, the resulting Chroma collection
# name must satisfy Chroma's rules: 3-63 chars, alphanumeric/underscore/hyphen,
# starts and ends with an alphanumeric char. Our regex below enforces a session
# _id of 1-55 chars starting with alphanumeric — yields collection names of
# length 9-63, all valid.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,54}$")


# ---------------------------------------------------------------------------
# Module-level state. Shared across all sessions in a single process.
# `_LOCK` guards mutations to `_SESSION_BM25`; Chroma manages its own
# persistent-storage concurrency, so we don't lock around Chroma operations.
# ---------------------------------------------------------------------------
_SESSION_BM25: Dict[str, BM25Index] = {}
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _validate_session_id(session_id: str) -> None:
    if not isinstance(session_id, str) or not _SESSION_ID_RE.match(session_id):
        raise ValueError(
            f"Invalid session_id: must be 1-55 chars, start with alphanumeric, "
            f"contain only [A-Za-z0-9_-]; got {session_id!r}"
        )


def _validate_uploads(files: List[Tuple[bytes, str]]) -> None:
    if not files:
        raise ValueError("Cannot index session with zero uploaded files.")
    if len(files) > MAX_FILES_PER_SESSION:
        raise ValueError(
            f"Session upload limit exceeded: {len(files)} files "
            f"(max {MAX_FILES_PER_SESSION} per session)."
        )
    for i, item in enumerate(files):
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(
                f"files[{i}] must be a (bytes, filename) tuple; got {type(item).__name__}."
            )
        raw_bytes, filename = item
        if not isinstance(raw_bytes, (bytes, bytearray)):
            raise TypeError(
                f"files[{i}] ({filename!r}): expected bytes, "
                f"got {type(raw_bytes).__name__}."
            )
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"files[{i}] missing a valid filename; got {filename!r}.")
        if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File {filename!r} is {len(raw_bytes) / 1024 / 1024:.1f}MB; "
                f"limit is {MAX_FILE_SIZE_MB}MB."
            )


def _session_collection_name(session_id: str) -> str:
    return f"{SESSION_COLLECTION_PREFIX}{session_id}"


# ---------------------------------------------------------------------------
# Low-level Chroma collection management. We go through chromadb's
# PersistentClient directly because langchain_chroma's Chroma wrapper doesn't
# expose a clean "delete collection" API. Importing inside the helpers keeps
# the import cost out of the module-load path.
# ---------------------------------------------------------------------------
def _get_chroma_client(persist_dir: Path):
    import chromadb

    return chromadb.PersistentClient(path=str(persist_dir))


def _chroma_collection_exists(collection_name: str, persist_dir: Path) -> bool:
    if not persist_dir.is_dir():
        return False
    client = _get_chroma_client(persist_dir)
    try:
        client.get_collection(name=collection_name)
        return True
    except Exception:
        return False


def _delete_chroma_collection_if_exists(
    collection_name: str, persist_dir: Path
) -> bool:
    """Returns True iff a collection was actually deleted."""
    if not persist_dir.is_dir():
        return False
    client = _get_chroma_client(persist_dir)
    try:
        client.delete_collection(name=collection_name)
        return True
    except Exception:
        return False


def _build_session_chroma(
    documents: List[Document],
    session_id: str,
    persist_dir: Path,
) -> Chroma:
    """Build a per-session Chroma collection. Shares persist_dir with the base collection but uses a unique session-scoped collection name."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    collection_name = _session_collection_name(session_id)
    ids = [_chunk_id(d) for d in documents]
    return Chroma.from_documents(
        documents=documents,
        embedding=_make_embeddings(),
        collection_name=collection_name,
        persist_directory=str(persist_dir),
        ids=ids,
    )


# ---------------------------------------------------------------------------
# Public API: index / get / clear
# ---------------------------------------------------------------------------
def index_session_docs(
    files: List[Tuple[bytes, str]],
    session_id: str,
    persist_dir: Optional[Path] = None,
) -> Tuple[BM25Index, Chroma]:
    """
    Parse, chunk, and index uploaded PDFs into a per-session BM25 + Chroma pair.

    Tuple ordering note: `files` is `(bytes, filename)` per spec. This is the
    inverse of `pdf_parser.parse_pdfs`'s `(filename, bytes)` order — we flip
    internally. The bytes-first ordering is more natural at the Streamlit
    call site: `[(f.getvalue(), f.name) for f in st.file_uploader(...)]`.

    Args:
        files: list of `(bytes, filename)` tuples from the file uploader.
        session_id: per-user identifier; becomes the suffix of the Chroma
            collection name `session_{session_id}`. Must match
            `[A-Za-z0-9][A-Za-z0-9_-]{0,54}`.
        persist_dir: where the Chroma collection lives. Defaults to the same
            directory as the base collection — they're isolated by name.

    Returns:
        `(bm25_index, chroma_collection)`. Retrieval can query both via the
        BM25Retriever / VectorRetriever wrappers used for the base collection.

    Raises:
        ValueError: invalid session_id, zero files, too many files, or any file
            over the size limit.
        TypeError: a file element isn't `(bytes, str)`.
        RuntimeError: no extractable text in any of the uploaded PDFs.

    Re-indexing semantics: calling with the same `session_id` REPLACES any
    existing session — the old Chroma collection is deleted before rebuild.
    """
    _validate_session_id(session_id)
    _validate_uploads(files)

    persist_dir = persist_dir or CHROMA_DIR

    # parse_pdfs wants (filename, bytes); flip the tuple order here.
    parser_input = [(filename, raw_bytes) for raw_bytes, filename in files]

    print(f"[session_store] session={session_id!r}: parsing {len(files)} file(s)...")
    page_docs = parse_pdfs(parser_input)
    print(f"[session_store] session={session_id!r}: extracted {len(page_docs)} page(s)")

    chunks = chunk_documents(page_docs)
    print(f"[session_store] session={session_id!r}: chunked into {len(chunks)} chunk(s)")

    if not chunks:
        raise RuntimeError(
            f"Session {session_id!r}: no chunks produced — the uploaded PDF(s) "
            f"may have been image-only or empty."
        )

    # Stamp `collection: "session"` so the retrieval layer can distinguish
    # base vs session results when both are merged in mixed result sets.
    docs = _stamp_collection(chunks, SESSION_COLLECTION_TAG)

    # Idempotent re-upload: wipe any stale collection from a previous call
    # with the same session_id before rebuilding.
    collection_name = _session_collection_name(session_id)
    _delete_chroma_collection_if_exists(collection_name, persist_dir)

    chroma = _build_session_chroma(docs, session_id, persist_dir)
    bm25_index = _build_bm25(docs)

    with _LOCK:
        _SESSION_BM25[session_id] = bm25_index

    print(
        f"[session_store] session={session_id!r}: indexed {len(docs)} chunk(s) → "
        f"Chroma collection {collection_name!r} + in-memory BM25"
    )
    return bm25_index, chroma


def get_session_index(
    session_id: str,
    persist_dir: Optional[Path] = None,
) -> Optional[Tuple[BM25Index, Chroma]]:
    """
    Look up an existing session's indices.

    Returns `None` if the session is not "live" — meaning either:
      - no BM25 in the in-memory dict (process restarted, or never indexed), OR
      - no Chroma collection on disk (cleared, or never built).

    If we find a stale BM25 (in memory) without its matching Chroma collection
    on disk, the orphan BM25 entry is removed lazily as a side effect. This is
    self-healing in the presence of inconsistent state.
    """
    _validate_session_id(session_id)
    persist_dir = persist_dir or CHROMA_DIR

    with _LOCK:
        bm25_index = _SESSION_BM25.get(session_id)
    if bm25_index is None:
        return None

    collection_name = _session_collection_name(session_id)
    if not _chroma_collection_exists(collection_name, persist_dir):
        # Lazy self-heal: in-memory BM25 without on-disk Chroma is a broken
        # session — drop the stale entry so callers get a clean None.
        with _LOCK:
            _SESSION_BM25.pop(session_id, None)
        return None

    chroma = Chroma(
        collection_name=collection_name,
        embedding_function=_make_embeddings(),
        persist_directory=str(persist_dir),
    )
    return bm25_index, chroma


def clear_session(session_id: str, persist_dir: Optional[Path] = None) -> None:
    """
    Delete a session's Chroma collection and remove its BM25 from memory.
    Idempotent: calling on a non-existent session is a no-op (just logs).
    """
    _validate_session_id(session_id)
    persist_dir = persist_dir or CHROMA_DIR

    with _LOCK:
        removed = _SESSION_BM25.pop(session_id, None)

    collection_name = _session_collection_name(session_id)
    deleted = _delete_chroma_collection_if_exists(collection_name, persist_dir)

    if removed is None and not deleted:
        print(f"[session_store] session={session_id!r}: nothing to clear (already gone)")
    else:
        print(
            f"[session_store] session={session_id!r}: cleared "
            f"(bm25_removed={removed is not None}, chroma_deleted={deleted})"
        )


# ---------------------------------------------------------------------------
# Test block: simulates a Streamlit file upload end-to-end on one real PDF.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_dir = Path(__file__).resolve().parent.parent / "tests" / "sample_pdfs"
    candidates = sorted(sample_dir.glob("*.pdf"))
    if not candidates:
        print(f"[session_store] no sample PDFs in {sample_dir}; aborting test.")
        sys.exit(1)

    sample_pdf = candidates[0]
    print(f"[session_store] using sample PDF: {sample_pdf.name}\n")

    with sample_pdf.open("rb") as fh:
        pdf_bytes = fh.read()

    session_id = "test_session_123"
    files: List[Tuple[bytes, str]] = [(pdf_bytes, sample_pdf.name)]

    # ------------------------------------------------------------------
    # Phase 1: clean slate (handles leftover from a previous test run)
    # ------------------------------------------------------------------
    print("=" * 72)
    print("PHASE 1: clear_session (clean slate, may be a no-op)")
    print("=" * 72)
    clear_session(session_id)

    # ------------------------------------------------------------------
    # Phase 2: index_session_docs
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("PHASE 2: index_session_docs")
    print("=" * 72)
    bm25, chroma = index_session_docs(files, session_id)

    chunk_count = len(bm25.documents)
    chroma_count = len(chroma.get(include=[])["ids"])
    assert chunk_count > 0, "BM25 should have at least 1 chunk after indexing"
    assert chunk_count == chroma_count, (
        f"chunk count mismatch: BM25={chunk_count}, Chroma={chroma_count}"
    )
    print(f"  ✓ {chunk_count} chunk(s) in both BM25 and Chroma")

    first = bm25.documents[0]
    assert first.metadata.get("collection") == "session", "metadata.collection should be 'session'"
    assert first.metadata.get("source") == sample_pdf.name, "metadata.source should match upload filename"
    print(f"  ✓ chunks stamped with collection='session', source={first.metadata['source']!r}")

    # ------------------------------------------------------------------
    # Phase 3: get_session_index — round-trip lookup
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("PHASE 3: get_session_index (round-trip lookup)")
    print("=" * 72)
    retrieved = get_session_index(session_id)
    assert retrieved is not None, "should find the session we just indexed"
    retrieved_bm25, retrieved_chroma = retrieved
    assert retrieved_bm25 is bm25, "get_session_index should return the same BM25 instance"
    print(f"  ✓ get_session_index returned the live session ({len(retrieved_bm25.documents)} chunks)")

    # Smoke-query both retrievers on the session collection (no base interference).
    sample_query = "termination"
    hits = retrieved_chroma.similarity_search(sample_query, k=2)
    print(f"  ✓ vector similarity_search({sample_query!r}, k=2) → {len(hits)} hit(s):")
    for h in hits:
        snippet = h.page_content[:120].replace("\n", " ")
        print(f"      {h.metadata['source']} p.{h.metadata['page']} :: {snippet}")

    # ------------------------------------------------------------------
    # Phase 4: clear_session + verify gone
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("PHASE 4: clear_session + verify get_session_index returns None")
    print("=" * 72)
    clear_session(session_id)
    assert get_session_index(session_id) is None, "session should be gone after clear"
    print(f"  ✓ get_session_index returns None after clear")

    # ------------------------------------------------------------------
    # Phase 5: idempotency check
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("PHASE 5: clear_session idempotency (second call should be a no-op)")
    print("=" * 72)
    clear_session(session_id)  # must not raise
    print(f"  ✓ clear_session on an already-cleared session is a no-op")

    # ------------------------------------------------------------------
    # Phase 6: validation errors fire as expected
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("PHASE 6: validation error paths")
    print("=" * 72)
    try:
        index_session_docs([], session_id)
    except ValueError as e:
        print(f"  ✓ empty file list → ValueError: {e}")

    try:
        index_session_docs(files, "bad/session/id")
    except ValueError as e:
        print(f"  ✓ invalid session_id → ValueError: {e}")

    too_many = files * (MAX_FILES_PER_SESSION + 1)
    try:
        index_session_docs(too_many, session_id)
    except ValueError as e:
        print(f"  ✓ {len(too_many)} files → ValueError: {e}")

    print("\n[session_store] all assertions passed ✓")
