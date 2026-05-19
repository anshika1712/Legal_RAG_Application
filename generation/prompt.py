"""Citation-enforced prompt template and context formatter for the Claude generation layer."""

from __future__ import annotations

from typing import Dict, List

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# System prompt: rules the LLM must follow when generating an answer.
# ---------------------------------------------------------------------------
# Design notes:
# - Rules are numbered so the LLM can self-check against them.
# - Citation format is [N], matching the chunk numbering in `format_context`.
#   This is a parseable format (cf. `_parse_citations` in llm.py) — the LLM
#   doesn't need to emit JSON, just bracketed integers.
# - The "could not find" string is verbatim per .cursorrules requirements;
#   downstream code may match on it to flag low-confidence answers in the UI.
SYSTEM_PROMPT = """You are a legal contract analysis assistant. Your role is to answer questions about legal contracts using ONLY the context chunks provided below.

CRITICAL RULES — these must be followed exactly:

1. Use ONLY information from the numbered context chunks. Do not draw on outside knowledge of contract law, common practice, or other documents that aren't shown to you.

2. Every factual claim in your answer must be followed by a citation in the format [N], where N is the number of the chunk that supports it. Place the citation immediately after the claim it supports, before any sentence-ending punctuation.
   Example: "The agreement permits termination upon thirty days' written notice [1]."
   If multiple chunks support a single claim, cite all of them: [1][3].

3. If the answer is not present in any of the context chunks, respond with exactly this sentence and nothing else:
   "I could not find this in the provided contracts."
   Do not guess, infer from related clauses, or supplement with general legal knowledge.

4. When quoting clause text, quote it accurately (verbatim). When summarizing, the summary must be directly supported by the cited chunk(s) — do not paraphrase in a way that changes meaning.

5. If the question is only partially answered by the context, answer the part you can and explicitly note what is missing.
   Example: "The agreement specifies a thirty-day notice period [1], but does not address termination for cause."

6. Keep answers concise and focused on the question. Do not include unrelated information from the context just because it appears in the chunks."""


# ---------------------------------------------------------------------------
# User message template: context + question, in that order.
# ---------------------------------------------------------------------------
USER_TEMPLATE = """Context:

{context}

---

Question: {question}"""


# Inter-chunk separator used inside `format_context`. Long enough to be
# unambiguously visible in the model's input, short enough not to bloat tokens.
_CHUNK_SEPARATOR = "\n\n---\n\n"


def format_context(docs: List[Document]) -> str:
    """
    Render retrieved Documents as numbered, header-tagged chunks the LLM can
    cite by index ([1], [2], ...).

    Per .cursorrules conventions, the LLM only sees canonical metadata
    (source, page, clause_number) — stage scores like rerank_score, rrf_score,
    bm25_score, similarity_score are intentionally stripped here.

    Empty input returns an empty string, which the prompt template will render
    visibly so the model sees "no context" and triggers rule 3.
    """
    if not docs:
        return ""

    rendered: List[str] = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        source = meta.get("source", "unknown source")
        page = meta.get("page", "?")
        clause = meta.get("clause_number") or ""
        clause_suffix = f", clause {clause}" if clause else ""

        header = f"[{i}] Source: {source}, page {page}{clause_suffix}"
        body = doc.page_content.strip()
        rendered.append(f"{header}\n{body}")

    return _CHUNK_SEPARATOR.join(rendered)


def build_messages(query: str, docs: List[Document]) -> List[Dict[str, str]]:
    """
    Build the `messages` payload for `anthropic.Anthropic().messages.create()`.

    Note: the system prompt is NOT in this list — Anthropic's API takes it via
    a separate `system=` parameter. Callers should pass SYSTEM_PROMPT directly.
    """
    user_content = USER_TEMPLATE.format(
        context=format_context(docs),
        question=query.strip(),
    )
    return [{"role": "user", "content": user_content}]
