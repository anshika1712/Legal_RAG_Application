"""Streamlit chat UI for the Legal Contract Lookup RAG system.

Run with: `streamlit run app.py` from the project root.

UI layout:
  - Sidebar: file uploader, mode indicator, session actions, about section.
  - Main: chat history + chat input. On submit, runs the full RAG pipeline
    and renders the answer with citations and chunk previews.

State management:
  - `st.session_state.session_id`     — unique per browser session (uuid4).
  - `st.session_state.messages`       — chat history, list of message dicts.
  - `st.session_state.indexed_files`  — filenames currently in this session.
  - `st.session_state.uploader_key`   — bumped to reset the file_uploader widget.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

# Load .env BEFORE importing the pipeline (it reads ANTHROPIC_API_KEY +
# COHERE_API_KEY at import time via the lazy-init client constructors).
load_dotenv()

from generation.llm import NOT_FOUND_SENTINEL
from ingestion.session_store import (
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    MAX_FILES_PER_SESSION,
    clear_session,
    get_session_index,
    index_session_docs,
)
from retrieval.pipeline import answer, prewarm


# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call).
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Legal Contract Lookup",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
BASE_PDFS_DIR = PROJECT_ROOT / "data" / "raw_contracts"

CHUNK_PREVIEW_CHARS = 280   # length of inline preview in "Sources cited"


# ---------------------------------------------------------------------------
# Resource warming. `@st.cache_resource` ensures the heavy load (BGE model,
# base index, API clients) happens once per process. The decorated function
# returns a sentinel — we only care about the side effect.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models and base corpus (~5 seconds, one-time)...")
def _ensure_warmed() -> bool:
    prewarm()
    return True


# ---------------------------------------------------------------------------
# State initialization. Called once per script rerun; cheap guards keep it
# idempotent.
# ---------------------------------------------------------------------------
def init_state() -> None:
    if "session_id" not in st.session_state:
        # st_ prefix + 16 hex chars = 19 chars, well within Chroma's
        # collection-name length budget once `session_` is prepended.
        st.session_state.session_id = f"st_{uuid.uuid4().hex[:16]}"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.header("Document Upload")
        st.caption(
            f"Drop in up to {MAX_FILES_PER_SESSION} PDFs ({MAX_FILE_SIZE_MB}MB each). "
            f"Files are indexed automatically — no extra click required. While "
            f"your files are indexed, queries search ONLY your files (the base "
            f"demo corpus is ignored)."
        )

        uploaded_files = st.file_uploader(
            "Select PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
        )

        # Client-side validation — instant feedback instead of waiting for
        # session_store.py to raise during auto-indexing.
        too_many = bool(uploaded_files) and len(uploaded_files) > MAX_FILES_PER_SESSION
        if too_many:
            st.error(
                f"Too many files ({len(uploaded_files)}). "
                f"Max is {MAX_FILES_PER_SESSION} per session."
            )

        oversized: List[str] = []
        if uploaded_files:
            for f in uploaded_files:
                if f.size > MAX_FILE_SIZE_BYTES:
                    oversized.append(f"{f.name} ({f.size / 1024 / 1024:.1f}MB)")
        if oversized:
            st.error(
                f"File(s) over {MAX_FILE_SIZE_MB}MB limit:\n"
                + "\n".join(f"- {x}" for x in oversized)
            )

        # ----- Auto-index when the uploader contents change ---------------
        # The whole point: it should be impossible to "forget to index" what
        # you uploaded. We compare the set of filenames currently in the
        # widget against the set we've already indexed; if they differ, we
        # re-index (replacing any existing session — session_store's
        # index_session_docs is idempotent on re-call).
        valid_to_index = bool(uploaded_files) and not too_many and not oversized
        if valid_to_index:
            current_set = sorted(f.name for f in uploaded_files)
            previous_set = sorted(st.session_state.indexed_files)
            if current_set != previous_set:
                files_payload = [(f.getvalue(), f.name) for f in uploaded_files]
                try:
                    with st.spinner(
                        f"Indexing {len(files_payload)} file(s)..."
                    ):
                        bm25, _chroma = index_session_docs(
                            files_payload, st.session_state.session_id
                        )
                    st.session_state.indexed_files = [f.name for f in uploaded_files]
                    st.success(
                        f"Indexed {len(bm25.documents)} chunk(s). "
                        f"Now querying your uploaded files only."
                    )
                except (ValueError, TypeError, RuntimeError) as e:
                    st.error(f"Auto-indexing failed: {e}")

        # ----- Auto-clear when the user removes all files from the widget -
        # If the uploader is empty BUT we still have an active session,
        # the user has deleted all their files via the widget's X buttons.
        # Drop the session collection so they go back to base mode cleanly.
        if not uploaded_files and st.session_state.indexed_files:
            clear_session(st.session_state.session_id)
            st.session_state.indexed_files = []
            st.info("Uploads removed. Switched back to base corpus.")

        st.divider()

        # ----- Current mode + source files -----
        st.header("Current Mode")
        session_active = get_session_index(st.session_state.session_id) is not None
        if session_active:
            st.success("Querying YOUR uploaded files")
            for fname in st.session_state.indexed_files:
                st.markdown(f"- `{fname}`")
        else:
            st.info("Querying base demo corpus")
            if BASE_PDFS_DIR.is_dir():
                base_files = sorted(p.name for p in BASE_PDFS_DIR.glob("*.pdf"))
                for fname in base_files:
                    st.markdown(f"- `{fname}`")
            else:
                st.caption(f"(base corpus dir not found at {BASE_PDFS_DIR})")

        st.divider()

        # ----- Session actions -----
        st.header("Session Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "Clear uploads",
                disabled=not session_active,
                use_container_width=True,
                help="Delete the session index and switch back to base mode. Chat history is preserved.",
            ):
                clear_session(st.session_state.session_id)
                st.session_state.indexed_files = []
                st.session_state.uploader_key += 1
                st.success("Uploads cleared. Now querying base corpus.")
                st.rerun()
        with col2:
            if st.button(
                "Reset chat",
                disabled=not st.session_state.messages,
                use_container_width=True,
                help="Clear chat history. Uploaded files are kept.",
            ):
                st.session_state.messages = []
                st.rerun()

        st.divider()

        with st.expander("About this system", expanded=False):
            st.markdown(
                """
**Legal Contract Lookup** is a retrieval-augmented chat system for legal
contracts. Every answer cites the specific document, page, and clause it
came from. The system will refuse to answer if it can't find supporting
evidence in the indexed documents.

**Stack:** LangChain · ChromaDB · BM25 (rank_bm25) · BGE embeddings ·
Cohere reranker · Claude

**Per-query pipeline:**
1. Parallel BM25 (keyword) + dense vector retrieval over one collection.
2. Reciprocal Rank Fusion merges both result lists.
3. Cohere `rerank-english-v3.0` re-scores the top candidates.
4. Claude generates an answer citing chunks `[1]`, `[2]`, ...
                """
            )

        st.caption(f"Session ID: `{st.session_state.session_id}`")


# ---------------------------------------------------------------------------
# Assistant message renderer — used both for new responses and when
# re-rendering chat history on script rerun.
# ---------------------------------------------------------------------------
def render_assistant_message(result: Dict[str, Any]) -> None:
    # ----- Low-confidence banner -----
    # Only show when retrieval scored low AND the LLM still attempted a
    # substantive answer. If the LLM correctly returned the NOT_FOUND_SENTINEL,
    # that refusal message is itself the user-visible signal — stacking a
    # yellow banner on top of it is redundant noise.
    answered = NOT_FOUND_SENTINEL not in result.get("answer", "")
    if result.get("low_confidence") and answered:
        st.warning(
            "**Low retrieval confidence.** The indexed documents don't strongly "
            "match your question; the answer below may be partial — read citations "
            "carefully."
        )

    # ----- The answer (markdown — supports the LLM's bold/lists/headers) -----
    st.markdown(result["answer"])

    # ----- Mode + sources searched caption -----
    mode = result.get("mode", "base")
    sources = result.get("source_files", [])
    mode_label = "your uploads" if mode == "session" else "base corpus"
    sources_str = ", ".join(sources) if sources else "(none)"
    st.caption(f"Searched **{mode_label}** ({len(sources)} file(s)): {sources_str}")

    # ----- Sources cited: previews for chunks the LLM actually referenced -----
    #
    # Design rationale: we explicitly do NOT show "other considered" chunks
    # (top-k by rerank but uncited). For someone using this to answer a legal
    # question, what the retrieval pipeline considered-but-discarded is noise;
    # only the evidence backing the actual claims matters. Order matches
    # first-appearance in the answer so the reader's eye stays anchored — if
    # they read "...[3]..." in the answer, the first preview here is [3].
    citations = result.get("citations", [])
    chunks = result.get("chunks", [])
    cited_chunks = []
    for c in citations:
        ref = c.get("ref")
        # Defensive: only include refs that map to a real reranked chunk.
        if ref is not None and 1 <= ref <= len(chunks):
            cited_chunks.append((c, chunks[ref - 1]))

    if cited_chunks:
        with st.expander(f"Sources cited ({len(cited_chunks)})", expanded=False):
            for i, (citation, doc) in enumerate(cited_chunks):
                meta = doc.metadata
                ref = citation["ref"]
                clause = meta.get("clause_number") or ""
                clause_str = f" · clause `{clause}`" if clause else ""

                st.markdown(
                    f"**[{ref}]** `{meta.get('source', 'unknown')}` · "
                    f"page {meta.get('page', '?')}{clause_str}"
                )
                score = meta.get("rerank_score")
                if score is not None:
                    # Rerank score lives in [0, 1]; render as a percentage for
                    # legibility ("84% relevance" reads better than "0.84").
                    st.caption(f"Relevance: {score * 100:.0f}%")

                preview = doc.page_content[:CHUNK_PREVIEW_CHARS].replace("\n", " ")
                truncated = len(doc.page_content) > CHUNK_PREVIEW_CHARS
                if truncated:
                    preview += "..."
                st.markdown(f"> {preview}")

                if truncated:
                    with st.expander("Show full chunk"):
                        st.text(doc.page_content)

                if i < len(cited_chunks) - 1:
                    st.divider()


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
def render_main() -> None:
    # ----- Frozen header (title + caption + mode badge) -----
    # Everything here lives OUTSIDE the scrollable chat container below, so
    # it stays anchored at the top of the page no matter how long the
    # conversation gets. This is the "sticky header" the user asked for —
    # the mode badge in particular must always be visible so the user
    # knows what scope their next question will search.
    st.title("Legal Contract Lookup")
    st.caption(
        "Ask questions about legal contracts. Every answer cites the document, "
        "page, and clause it came from."
    )

    session_active = get_session_index(st.session_state.session_id) is not None
    if session_active:
        files = st.session_state.indexed_files
        files_str = ", ".join(f"`{f}`" for f in files)
        st.info(
            f"**SESSION MODE** — your next question will search ONLY your "
            f"uploaded file(s): {files_str}. The base demo corpus is ignored."
        )
    else:
        st.info(
            "**BASE MODE** — your next question will search the demo corpus "
            "(5 sample contracts). Upload PDFs in the sidebar to query your "
            "own files instead."
        )

    # ----- Scrollable chat container -----
    # `st.container(height=N)` creates a fixed-height region with internal
    # overflow:auto. Only the chat history scrolls inside it; the header
    # above and the `st.chat_input` below remain fixed. 500px fits the
    # typical laptop viewport (~800px) after subtracting browser chrome,
    # the header (~200px), and the chat input (~80px).
    chat_container = st.container(height=500, border=False)

    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(msg["content"])
                else:
                    # Assistant messages persist the full result dict so reruns
                    # re-render citations + chunk previews without re-running
                    # the pipeline.
                    render_assistant_message(msg["result"])

    # ----- Chat input — pinned to the viewport bottom by Streamlit -----
    prompt = st.chat_input("Ask about the contracts...")
    if not prompt:
        return

    # New turn: append to history, then render echo + response INSIDE the
    # same scrollable container so the new exchange joins the conversation
    # in-place (and the container auto-scrolls to show it).
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Searching contracts and generating answer..."):
                result = answer(prompt, session_id=st.session_state.session_id)
            render_assistant_message(result)

    # Persist for replay. Store both the rendered text and the full result dict.
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "result": result}
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    _ensure_warmed()  # idempotent; first call shows spinner, rest are instant
    init_state()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
