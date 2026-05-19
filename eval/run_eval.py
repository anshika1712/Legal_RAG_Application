"""RAGAS evaluation harness for the Legal Contract Lookup RAG pipeline.

What this does
--------------
1. Loads `eval/golden_set.json` — a hand-authored set of question/reference pairs
   grounded in the 5 base contracts.
2. For each question, runs `retrieval.pipeline.answer(query)` in BASE mode (eval
   never touches user sessions) and captures the model answer plus the reranked
   chunks that were fed to the LLM (used as `retrieved_contexts`).
3. Builds a RAGAS `EvaluationDataset` and scores it with five metrics:
     - faithfulness         : does the answer stick to the retrieved context?
     - answer_relevancy     : does the answer address the question?
     - context_precision    : are the retrieved chunks relevant?
     - context_recall       : did we retrieve everything needed to support the reference?
     - answer_correctness   : how close is the answer to the reference?
4. Writes a CSV with per-question scores AND a console summary grouped by
   category (bm25_friendly, semantic, multi_doc, out_of_corpus, edge).

Notes
-----
- Judge LLM = Claude (whatever `DEFAULT_MODEL` is in `generation.llm`). Using
  Claude for both generation AND judging is "marking your own homework" — fine
  for tracking pipeline regressions, but treat the absolute numbers with the
  appropriate grain of salt. Out-of-the-box RAGAS expects OpenAI; we override
  via `langchain_anthropic.ChatAnthropic`.
- Judge embeddings = BAAI/bge-small-en-v1.5, same model used by retrieval. Keeps
  the eval consistent with what the system actually indexed.
- Out-of-corpus questions (`expected_sources == []`) have a reference of
  "I could not find this in the provided contracts." — answer_correctness
  rewards correct refusal; context_recall/precision may be undefined for these
  and RAGAS may emit NaN, which we drop from category aggregates.

Usage
-----
    python eval/run_eval.py                # full set (15 questions, ~$0.30-0.60 in Claude calls)
    python eval/run_eval.py --subset 2     # smoke test (2 questions)
    python eval/run_eval.py --out path.csv # custom output location
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---- Path bootstrap so `python eval/run_eval.py` works from project root. ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# RAGAS 0.4.x is mid-migration: their new `metrics.collections` namespace only
# supports the OpenAI-based `InstructorLLM`, while the older `ragas.metrics`
# classes still accept `LangchainLLMWrapper` (which is what we need to wire
# Claude). We deliberately use the older API and silence its deprecation
# warnings — switching to the new API would force an OpenAI dependency, which
# defeats the "Claude as judge" design decision.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*Langchain(LLM|Embeddings)Wrapper is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*Importing .* from 'ragas.metrics' is deprecated.*",
)

# Load .env BEFORE any module that reads ANTHROPIC_API_KEY / COHERE_API_KEY at
# import time (the pipeline's lazy-init clients do this).
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import pandas as pd  # noqa: E402
from langchain_anthropic import ChatAnthropic  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    AnswerCorrectness,
    AnswerRelevancy,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

from generation.llm import DEFAULT_MODEL as JUDGE_MODEL  # noqa: E402
from retrieval.pipeline import answer, prewarm  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GOLDEN_SET_PATH = PROJECT_ROOT / "eval" / "golden_set.json"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
JUDGE_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# RAGAS metric column names in the result DataFrame. Pinned here so the
# summarizer doesn't silently break if a metric class is swapped.
# `llm_context_precision_with_reference` is RAGAS's emitted name for the
# `LLMContextPrecisionWithReference` metric class — we rename it to plain
# `context_precision` after evaluation for readability.
RAW_METRIC_COL_MAP = {
    "llm_context_precision_with_reference": "context_precision",
}
METRIC_COLS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]

# RAGAS adds these duplicate columns to the result DataFrame (its internal
# schema renames of our `question/contexts/answer/ground_truth`). Drop them
# before writing the CSV so the output is clean.
RAGAS_DUPLICATE_COLS = ["user_input", "retrieved_contexts", "response", "reference"]


# ---------------------------------------------------------------------------
# Golden set loader
# ---------------------------------------------------------------------------
def load_golden_set() -> List[Dict[str, Any]]:
    with GOLDEN_SET_PATH.open("r") as f:
        payload = json.load(f)
    questions = payload.get("questions", [])
    if not questions:
        raise RuntimeError(f"No questions found in {GOLDEN_SET_PATH}")
    return questions


# ---------------------------------------------------------------------------
# Stage 1: run the pipeline over every golden question.
# ---------------------------------------------------------------------------
def run_pipeline_over_set(
    golden_set: List[Dict[str, Any]],
    subset: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """For each golden Q, call `pipeline.answer` and capture everything we need.

    We deliberately call with `session_id=None` so every eval run targets the
    base corpus — the golden set is grounded in those PDFs.
    """
    questions = golden_set if subset is None else golden_set[:subset]
    items: List[Dict[str, Any]] = []

    print(f"\nRunning pipeline over {len(questions)} question(s)...")
    print("-" * 80)

    for i, entry in enumerate(questions, 1):
        q = entry["question"]
        category = entry["category"]
        print(f"[{i}/{len(questions)}] ({category}) {q}")

        t0 = time.time()
        result = answer(q, session_id=None)
        latency = time.time() - t0

        # Reranked chunks the LLM actually saw = our `retrieved_contexts`.
        contexts = [d.page_content for d in result.get("chunks", [])]

        items.append(
            {
                "id": entry["id"],
                "category": category,
                "question": q,
                "answer": result["answer"],
                "contexts": contexts,
                "ground_truth": entry["ground_truth"],
                "expected_sources": entry["expected_sources"],
                "actual_sources": result.get("source_files", []),
                "low_confidence": result.get("low_confidence", False),
                "latency_s": round(latency, 2),
            }
        )
        print(
            f"   -> {latency:.2f}s, sources={result.get('source_files', [])}, "
            f"low_conf={result.get('low_confidence', False)}"
        )

    return items


# ---------------------------------------------------------------------------
# Stage 2: score with RAGAS.
# ---------------------------------------------------------------------------
def _build_eval_dataset(items: List[Dict[str, Any]]) -> EvaluationDataset:
    """Translate our internal item dicts into RAGAS's `SingleTurnSample` schema."""
    samples = [
        SingleTurnSample(
            user_input=it["question"],
            retrieved_contexts=it["contexts"],
            response=it["answer"],
            reference=it["ground_truth"],
        )
        for it in items
    ]
    return EvaluationDataset(samples=samples)


def score_with_ragas(items: List[Dict[str, Any]]) -> pd.DataFrame:
    """Run RAGAS over the captured items; return per-question metrics DataFrame."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY missing — RAGAS judge cannot run. "
            "Put it in your .env file at the project root."
        )

    dataset = _build_eval_dataset(items)

    # Wire Claude as judge. temperature=0 for reproducibility; max_tokens
    # generous enough for RAGAS's multi-step prompts.
    judge_llm = LangchainLLMWrapper(
        ChatAnthropic(model=JUDGE_MODEL, temperature=0, max_tokens=2048)
    )
    # BGE-small for any embedding-based metric step (e.g., AnswerRelevancy).
    # Same model used by the retrieval pipeline; keeps eval self-consistent.
    judge_embed = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=JUDGE_EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    )

    # Instantiate metrics. LLMContextPrecisionWithReference is the variant
    # that uses the gold reference (rather than the LLM's own answer) to
    # judge whether each retrieved chunk was relevant — strictly more
    # reliable when we have a reference, which we do.
    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embed),
        LLMContextPrecisionWithReference(llm=judge_llm),
        LLMContextRecall(llm=judge_llm),
        AnswerCorrectness(llm=judge_llm, embeddings=judge_embed),
    ]

    print(f"\nScoring {len(items)} item(s) with RAGAS ({len(metrics)} metrics)...")
    print("Judge LLM:", JUDGE_MODEL)
    print("Judge embeddings:", JUDGE_EMBEDDING_MODEL)
    print("-" * 80)

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embed,
        raise_exceptions=False,  # tolerate per-question metric failures (e.g. NaN on out-of-corpus)
    )
    metrics_df = result.to_pandas()
    # Normalize RAGAS-emitted column names to the shorter names we use in
    # METRIC_COLS, so the summary table is consistent regardless of which
    # metric class variant we picked.
    return metrics_df.rename(columns=RAW_METRIC_COL_MAP)


# ---------------------------------------------------------------------------
# Stage 3: summarize.
# ---------------------------------------------------------------------------
def _source_recall(items: List[Dict[str, Any]]) -> List[Optional[float]]:
    """Fraction of expected_sources that appear in actual_sources.

    Returns None for out-of-corpus questions (no expected sources => the
    fraction is undefined). We track this separately from RAGAS metrics — it
    sanity-checks the retriever's source selection, which RAGAS doesn't
    measure directly.
    """
    out: List[Optional[float]] = []
    for it in items:
        expected = set(it["expected_sources"])
        actual = set(it["actual_sources"])
        if not expected:
            out.append(None)
        else:
            out.append(len(expected & actual) / len(expected))
    return out


def summarize(metrics_df: pd.DataFrame, items: List[Dict[str, Any]]) -> None:
    df = metrics_df.copy()
    df["category"] = [it["category"] for it in items]
    df["source_recall"] = _source_recall(items)
    df["id"] = [it["id"] for it in items]

    # Only summarize metric columns that actually came back from RAGAS — older
    # versions sometimes name them differently.
    available_metric_cols = [c for c in METRIC_COLS if c in df.columns]

    print("\n" + "=" * 80)
    print("RAGAS SUMMARY")
    print("=" * 80)

    if available_metric_cols:
        print("\nPer-category mean scores (NaN dropped within each cell):")
        by_cat = df.groupby("category")[available_metric_cols].mean(numeric_only=True)
        print(by_cat.round(3).to_string())

        print("\nOverall mean scores:")
        overall = df[available_metric_cols].mean(numeric_only=True).round(3)
        print(overall.to_string())
    else:
        print("\n(no recognized metric columns in RAGAS output — check column names)")
        print("Columns present:", list(df.columns))

    # Source-recall diagnostic (excludes out-of-corpus rows where it's undefined).
    src_df = df.dropna(subset=["source_recall"])
    if not src_df.empty:
        print("\nPer-category source recall (cited file ⊇ expected file set):")
        print(src_df.groupby("category")["source_recall"].mean().round(3).to_string())

    # Highlight any per-question failures so the eye lands on them.
    if available_metric_cols:
        print("\nLowest-scoring rows (composite mean across metrics):")
        df["_mean"] = df[available_metric_cols].mean(axis=1, numeric_only=True)
        worst = df.nsmallest(min(3, len(df)), "_mean")[
            ["id", "category", "_mean"] + available_metric_cols
        ]
        print(worst.round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS eval over the golden set.")
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Run only the first N questions (smoke-test mode).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path. Default: eval/results/results_<timestamp>.csv",
    )
    args = parser.parse_args()

    print("Warming pipeline (loads BGE model, base index, API clients)...")
    prewarm()
    print("...warmed.")

    golden_set = load_golden_set()
    items = run_pipeline_over_set(golden_set, subset=args.subset)
    metrics_df = score_with_ragas(items)

    # Persist a wide CSV: per-question pipeline output + per-question metrics.
    pipeline_df = pd.DataFrame(items)
    # Drop RAGAS's internal-schema duplicate columns before merging — they
    # carry the same content as our pipeline_df columns under different names.
    metrics_df_clean = metrics_df.drop(
        columns=[c for c in RAGAS_DUPLICATE_COLS if c in metrics_df.columns],
        errors="ignore",
    )
    full_df = pd.concat(
        [pipeline_df.reset_index(drop=True), metrics_df_clean.reset_index(drop=True)],
        axis=1,
    )

    if args.out is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"results_{ts}.csv"
    else:
        out_path = args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)

    full_df.to_csv(out_path, index=False)
    print(f"\nFull results written to: {out_path}")

    summarize(metrics_df, items)


if __name__ == "__main__":
    main()
