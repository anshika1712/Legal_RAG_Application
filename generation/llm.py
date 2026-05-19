"""Claude-backed answer generator: formats reranked chunks with prompt.py, calls the Anthropic API, returns answer + parsed citations."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# sys.path bootstrap for direct script invocation.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from generation.prompt import SYSTEM_PROMPT, build_messages


# claude-sonnet-4-6: current Sonnet as of May 2026. Anthropic moved to a
# dateless model-id format with the 4.6 generation (the id pins a specific
# snapshot, it's just no longer date-suffixed). Good price/quality balance
# for legal Q&A — swap to `claude-opus-4-7` for higher-stakes eval runs.
DEFAULT_MODEL = "claude-sonnet-4-6"

# 1024 tokens ≈ 750 English words — comfortably fits a multi-clause legal
# answer with citations without truncation, and avoids paying for runaway
# generations on simple questions.
DEFAULT_MAX_TOKENS = 1024

# Deterministic answers given identical context. Critical for legal accuracy
# (no creative interpretation) and for reproducible RAGAS evals at step 9.
DEFAULT_TEMPERATURE = 0.0

# The exact "could not find" sentinel from prompt.py rule 3. Kept here as a
# constant so UI/eval code can match on it to flag low-confidence answers.
NOT_FOUND_SENTINEL = "I could not find this in the provided contracts."

# Citation pattern: matches [1], [2], [12], etc. Greedy on digits only.
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class ClaudeGenerator:
    """
    Generate grounded answers from reranked retrieval results using Claude.

    Like CohereReranker, this is lazy-init and tolerant: missing API key,
    unimportable `anthropic` package, or failed client construction all push
    the instance into "fallback mode" instead of crashing the pipeline.

    In fallback mode, `.generate()` returns an answer string that explains
    what went wrong and lists the retrieved chunks verbatim, plus a citations
    list built directly from the input docs. This keeps the demo runnable
    without credentials and gives the user something useful even when the LLM
    is unreachable.

    Check `.is_available()` to know which mode you're in.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client: Optional[Any] = None
        self._init_error: Optional[str] = None

        if not self.api_key:
            self._init_error = "no ANTHROPIC_API_KEY found in environment"
            return

        try:
            import anthropic
        except ImportError as e:
            self._init_error = f"`anthropic` package not installed: {e}"
            return

        try:
            self._client = anthropic.Anthropic(api_key=self.api_key)
        except Exception as e:
            self._init_error = (
                f"failed to construct Anthropic client: {type(e).__name__}: {e}"
            )

    def is_available(self) -> bool:
        """True iff an Anthropic client was successfully constructed."""
        return self._client is not None

    def generate(self, query: str, docs: List[Document]) -> Dict[str, Any]:
        """
        Generate an answer to `query` grounded in `docs` (typically the top-k
        results from the Cohere reranker).

        Returns a dict:
            {
                "answer": str,                       # natural-language answer with [N] citations
                "citations": [                       # parsed from [N] refs in `answer`
                    {"ref": int,                     # the [N] number from the answer
                     "source": str,
                     "page": int | str,
                     "clause_number": str | None},
                    ...
                ],
                "usage": {                           # token counts (None in fallback)
                    "input_tokens": int | None,
                    "output_tokens": int | None,
                },
            }

        Behavior:
            - Empty `docs` → still calls Claude with empty context; per prompt
              rule 3 the model should return NOT_FOUND_SENTINEL.
            - Claude unavailable / API failure → returns a structured fallback
              answer that names the error and dumps the retrieved chunks for
              the user to read manually.
        """
        if self._client is None:
            return self._fallback_response(
                docs,
                reason=f"Claude unavailable ({self._init_error})",
            )

        messages = build_messages(query, docs)

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
        except Exception as e:
            return self._fallback_response(
                docs,
                reason=f"Claude API call failed ({type(e).__name__}: {e})",
            )

        answer_text = self._extract_text(response)
        citations = _parse_citations(answer_text, docs)

        return {
            "answer": answer_text,
            "citations": citations,
            "usage": {
                "input_tokens": getattr(response.usage, "input_tokens", None),
                "output_tokens": getattr(response.usage, "output_tokens", None),
            },
        }

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Concatenate text blocks from a Claude messages.create response."""
        parts: List[str] = []
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts).strip()

    @staticmethod
    def _fallback_response(
        docs: List[Document], reason: str
    ) -> Dict[str, Any]:
        """
        Build a structured response when the LLM is unavailable. Surfaces the
        retrieved chunks verbatim so the user can read them directly and
        cites every chunk in the citations list (since the LLM didn't pick).
        """
        print(f"[llm] {reason}; returning fallback response.")

        if not docs:
            body = "(no chunks retrieved)"
        else:
            lines = []
            for i, doc in enumerate(docs, start=1):
                meta = doc.metadata or {}
                src = meta.get("source", "unknown")
                page = meta.get("page", "?")
                lines.append(f"[{i}] {src}, page {page}\n{doc.page_content.strip()}")
            body = "\n\n".join(lines)

        return {
            "answer": (
                f"[LLM unavailable: {reason}]\n\n"
                f"Showing retrieved chunks instead:\n\n{body}"
            ),
            "citations": [
                {
                    "source": (doc.metadata or {}).get("source", "unknown"),
                    "page": (doc.metadata or {}).get("page"),
                    "clause_number": (doc.metadata or {}).get("clause_number") or None,
                }
                for doc in docs
            ],
            "usage": {"input_tokens": None, "output_tokens": None},
        }


def _parse_citations(answer: str, docs: List[Document]) -> List[Dict[str, Any]]:
    """
    Extract [N] citation references from the LLM's answer, dedupe (preserving
    first-mention order), and resolve each to its source+page+clause from the
    input docs.

    Each citation includes the original `ref` (the integer N from `[N]`) so
    UIs can render citations labeled to match the inline references in the
    answer text — without `ref`, the order in this list reflects first
    appearance, not the [N] value, which can confuse readers when the LLM
    cites out of numerical order (e.g. "...[3]...[1]...[2]...").

    Out-of-range indices (e.g. LLM hallucinates [99] when only 5 chunks were
    provided) are silently dropped — better than crashing the UI.
    """
    citations: List[Dict[str, Any]] = []
    seen: set[int] = set()

    for match in _CITATION_PATTERN.finditer(answer):
        idx = int(match.group(1))
        if idx in seen or idx < 1 or idx > len(docs):
            continue
        seen.add(idx)
        meta = docs[idx - 1].metadata or {}
        citations.append(
            {
                "ref": idx,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page"),
                "clause_number": meta.get("clause_number") or None,
            }
        )
    return citations


# ---------------------------------------------------------------------------
# Test block: 3 hand-crafted mock chunks, two queries (positive + negative).
# ---------------------------------------------------------------------------


def _print_result(label: str, result: Dict[str, Any]) -> None:
    print(f"--- {label} ---")
    print("ANSWER:")
    print(result["answer"])
    print()
    print("CITATIONS:")
    if not result["citations"]:
        print("  (none)")
    for c in result["citations"]:
        clause = c.get("clause_number")
        clause_str = f", clause {clause}" if clause else ""
        print(f"  - {c['source']}, page {c['page']}{clause_str}")
    print()
    usage = result.get("usage", {})
    if usage.get("input_tokens") is not None:
        print(f"USAGE: input={usage['input_tokens']} output={usage['output_tokens']}")
    print()


if __name__ == "__main__":
    # Load .env so ANTHROPIC_API_KEY is picked up when running standalone.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    mock_chunks = [
        Document(
            page_content=(
                "10.1 Either party may terminate this Agreement at any time by "
                "giving the other party thirty (30) days' prior written notice. "
                "Such notice shall be delivered in accordance with the notice "
                "provisions set forth in Section 14 of this Agreement."
            ),
            metadata={
                "source": "EMPLOYMENT_AGREEMENT Sample.pdf",
                "page": 5,
                "clause_number": "10.1",
                "contract_type": "employment",
                "collection": "base",
                "rerank_score": 0.91,  # should be stripped from LLM input
            },
        ),
        Document(
            page_content=(
                "10.2 Notwithstanding Section 10.1, the Company may terminate "
                "this Agreement immediately, without notice or payment in lieu, "
                "if the Employee: (a) commits an act of gross misconduct; "
                "(b) materially breaches any provision of this Agreement and "
                "fails to cure such breach within fifteen (15) days of written "
                "notice; or (c) is convicted of a criminal offence involving "
                "dishonesty or moral turpitude."
            ),
            metadata={
                "source": "EMPLOYMENT_AGREEMENT Sample.pdf",
                "page": 5,
                "clause_number": "10.2",
                "contract_type": "employment",
                "collection": "base",
                "rerank_score": 0.84,
            },
        ),
        Document(
            page_content=(
                "11.1 Upon termination of this Agreement for any reason, the "
                "Employee shall promptly return to the Company all property "
                "belonging to the Company, including but not limited to "
                "documents, equipment, keys, and access credentials. The "
                "Employee's confidentiality obligations under Section 9 shall "
                "survive termination of this Agreement."
            ),
            metadata={
                "source": "EMPLOYMENT_AGREEMENT Sample.pdf",
                "page": 6,
                "clause_number": "11.1",
                "contract_type": "employment",
                "collection": "base",
                "rerank_score": 0.72,
            },
        ),
    ]

    generator = ClaudeGenerator()
    print(f"[llm] Claude available: {generator.is_available()}")
    if not generator.is_available():
        print(f"[llm] reason: {generator._init_error}")
        print("[llm] (test will demonstrate the fallback path)")
    print(f"[llm] model: {generator.model}, max_tokens: {generator.max_tokens}, temperature: {generator.temperature}")
    print()

    print("=" * 72)
    print("TEST 1: question answerable from the context")
    print("=" * 72)
    q1 = "Under what conditions can the company terminate the employment agreement?"
    print(f"QUERY: {q1}\n")
    _print_result("RESULT", generator.generate(q1, mock_chunks))

    print("=" * 72)
    print("TEST 2: question NOT answerable — expect NOT_FOUND_SENTINEL")
    print("=" * 72)
    q2 = "What is the dispute resolution mechanism under this agreement?"
    print(f"QUERY: {q2}\n")
    result2 = generator.generate(q2, mock_chunks)
    _print_result("RESULT", result2)
    # Guardrail check is only meaningful when Claude actually responded — if
    # the call failed and the fallback path fired, `usage.input_tokens` is
    # None and the answer is the fallback string, not anything Claude said.
    came_from_claude = result2.get("usage", {}).get("input_tokens") is not None
    if not came_from_claude:
        print("[llm] no-hallucination guardrail: ⊘ skipped (Claude did not respond; see error above)")
    elif NOT_FOUND_SENTINEL in result2["answer"]:
        print("[llm] no-hallucination guardrail: ✓ fired correctly")
    else:
        print("[llm] no-hallucination guardrail: ✗ did NOT fire — Claude answered anyway")
        print("       (this is a real finding — may need to tighten the prompt)")
