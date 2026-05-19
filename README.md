# Legal Contract Lookup — RAG System

A chat interface for querying legal contracts (NDAs, employment agreements,
non-compete, royalty agreements) with cited clause-level answers. Users can
also upload their own contracts and ask questions against them in a
session-scoped collection.

## Stack
- Python 3.11
- LangChain (orchestration)
- ChromaDB (vector storage; `base_contracts` + `session_contracts`)
- BM25 via `rank_bm25` (keyword retrieval)
- Cohere `rerank-english-v3.0` (reranking)
- Anthropic Claude (answer generation)
- RAGAS (evaluation)
- Streamlit (UI)

## Project layout
```
RAG_Application/
├── app.py                  # Streamlit chat + uploader
├── config.py               # paths, collection names, model ids, env loading
├── ingestion/              # pdf_parser, chunker, indexer
├── retrieval/              # bm25, vector, session_store, rrf_fusion, reranker
├── generation/             # Claude generator with citations
├── evaluation/             # RAGAS pipeline + practitioner questions
├── data/
│   ├── raw_contracts/      # base contracts (PDFs) live here
│   └── chroma_db/          # persistent Chroma store
└── tests/
    └── sample_pdfs/        # fixtures for component tests
```

## Build order
Implement strictly in this sequence (per `.cursorrules`):
1. Ingestion pipeline (`pdf_parser`, `chunker`, `indexer`)
2. BM25 retrieval
3. Vector retrieval (base collection)
4. Session store (user upload + session collection)
5. RRF fusion
6. Reranker
7. LLM generation with citations
8. Streamlit UI with file uploader
9. RAGAS eval pipeline

## Setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in API keys
```

## Quick check — PDF parser
Drop a sample PDF into `tests/sample_pdfs/` (or any folder) and run:
```bash
python ingestion/pdf_parser.py tests/sample_pdfs/
```
You should see one Document per non-empty page, each with `source`,
`page`, and `contract_type` metadata.
