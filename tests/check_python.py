"""One-shot inspection: print every chunk that got a non-None clause_number, so we can eyeball regex quality."""

import sys
from pathlib import Path

# Project root on sys.path so `from ingestion.chunker import ...` resolves
# regardless of where this script is run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.chunker import chunk_pdfs

SAMPLE_DIR = PROJECT_ROOT / "tests" / "sample_pdfs"

docs = chunk_pdfs(str(SAMPLE_DIR))

hits = [d for d in docs if d.metadata.get("clause_number")]
print(f"{len(hits)}/{len(docs)} chunks have a detected clause_number\n")

for d in hits:
    src = d.metadata["source"]
    page = d.metadata["page"]
    clause = d.metadata["clause_number"]
    snippet = d.page_content[:160].replace("\n", " ")
    print(f"{src} p.{page} :: {clause!r}")
    print(f"  {snippet}")
    print()
