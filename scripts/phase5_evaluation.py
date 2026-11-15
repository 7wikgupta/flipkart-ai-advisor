"""Phase 5 evaluation corpus runner (M5).

Runs a fixed set of test questions through the live retrieval + grounded-answer
pipeline (src.hybrid.make_retriever + src.rag_pipeline.answer_question) and
writes one row per question to data/evaluation/results.csv with retrieval@5
success and the LLM's answer, for manual pass/fail review.

Usage::

    set -a; . ./.env; set +a
    python scripts/phase5_evaluation.py
"""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hybrid import make_retriever
from src.rag_pipeline import LLMUnavailableError, answer_question

OUT_PATH = Path("data/evaluation/results.csv")


@dataclass
class TestQuestion:
    qid: str
    qtype: str  # factual | comparison | multipart | unsupported
    question: str
    expected_product_ids: tuple[str, ...] = ()
    notes: str = ""


QUESTIONS: list[TestQuestion] = [
    # --- Factual lookup: a specific, real product should be retrieved and named ---
    TestQuestion("F1", "factual", "Tell me about the HP 15-ac116TX laptop.",
                 ("P-85aec4bf3185da7ebcc45003429cbfe2",)),
    TestQuestion("F2", "factual", "What is the price of the Sony HT-GT1 home theatre system?",
                 ("P-7c0ff0a4b4e32a55edc8544a065c5cec",), "Expect INR 18,990, labeled historical."),
    TestQuestion("F3", "factual", "Do you have the Technotech Q3 wireless mouse?",
                 ("P-1a304ee48c38d3bf40eecfb50b410404",)),
    TestQuestion("F4", "factual", "Tell me about the HP v220w pen drive.",
                 ("P-650e27dff8f645649353743f64e03f77",)),
    TestQuestion("F5", "factual", "What are the specifications of the RoQ Slim Multimedia keyboard?",
                 ("P-b4d35b6536b623c83b293eb3a1c02bba",)),

    # --- Comparison: both named products should be retrieved and both used ---
    TestQuestion("C1", "comparison", "Compare the two HP laptops you have.",
                 ("P-85aec4bf3185da7ebcc45003429cbfe2", "P-bcb8e456728013a90ba66ab3b1b783ad")),
    TestQuestion("C2", "comparison", "Compare the Technotech Q3 and Ocean Q3 wireless mice.",
                 ("P-1a304ee48c38d3bf40eecfb50b410404", "P-1ae68e922242e296b3f374362cea696c"),
                 "Near-identical names/specs from different brands - checks the retriever doesn't collapse them."),
    TestQuestion("C3", "comparison", "Compare the TP-LINK TL-WR841N and the iBall 150M router.",
                 (), "Example prompt from the app; already spot-checked manually."),
    TestQuestion("C4", "comparison", "Compare the Sony HT-GT1 and the LG DH313OS home theatre systems.",
                 ("P-7c0ff0a4b4e32a55edc8544a065c5cec", "P-fdb9d53b85041d6b9590fd6e054a86e0")),

    # --- Multi-part: budget + a second constraint/question in one message ---
    TestQuestion("M1", "multipart", "Suggest a power bank under 1500 and tell me which has the highest capacity."),
    TestQuestion("M2", "multipart", "I need a home theatre system under 5000 with good power output, what do you suggest?"),
    TestQuestion("M3", "multipart", "Which pen drive has the most storage, and what does it cost?",
                 ("P-a63eecfbb02004863fa1f29a0b8105fc",), "Expect the 32 GB Shopizone drive at INR 999."),
    TestQuestion("M4", "multipart", "Suggest wireless headphones under 2000 and say if any are foldable."),

    # --- Unsupported: no evidence should exist; the bot must refuse, not invent ---
    TestQuestion("U1", "unsupported", "What is the capital of France?",
                 (), "Already observed live: correctly refused."),
    TestQuestion("U2", "unsupported", "Does Flipkart offer same-day delivery on these products?",
                 (), "Policy question; not in the historical product dataset."),
    TestQuestion("U3", "unsupported", "Can I return the Sony HT-GT1 if I don't like it?",
                 (), "Returns policy is not in the dataset, even though the product is."),
    TestQuestion("U4", "unsupported", "Do you sell any smartphones?",
                 (), "Catalogue has 0 phones (README/app sidebar both say so) - must not invent one."),
]


def main() -> None:
    retriever = make_retriever()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for tq in QUESTIONS:
        t0 = time.time()
        try:
            result = answer_question(tq.question, history=None, retriever=retriever)
        except LLMUnavailableError as exc:
            elapsed = time.time() - t0
            rows.append({
                "qid": tq.qid, "type": tq.qtype, "question": tq.question,
                "expected_product_ids": ";".join(tq.expected_product_ids),
                "retrieved_product_ids_top5": "", "retrieval_success_at_5": "",
                "reason": "llm_error", "grounded": False, "cited_product_ids": "",
                "invalid_citations": "", "answer": f"[LLM error] {exc}",
                "latency_s": round(elapsed, 2), "notes": tq.notes,
            })
            print(f"{tq.qid:4s} [{tq.qtype:10s}] {tq.question[:55]:55s} -> LLM_ERROR: {exc} ({elapsed:.1f}s)")
            continue
        elapsed = time.time() - t0
        retrieved_ids = [s.product_id for s in result.sources]
        hit = any(pid in retrieved_ids for pid in tq.expected_product_ids) if tq.expected_product_ids else ""
        rows.append({
            "qid": tq.qid,
            "type": tq.qtype,
            "question": tq.question,
            "expected_product_ids": ";".join(tq.expected_product_ids),
            "retrieved_product_ids_top5": ";".join(retrieved_ids),
            "retrieval_success_at_5": hit,
            "reason": result.reason,
            "grounded": result.grounded,
            "cited_product_ids": ";".join(result.cited_product_ids),
            "invalid_citations": ";".join(result.invalid_citations),
            "answer": result.answer.replace("\n", " / "),
            "latency_s": round(elapsed, 2),
            "notes": tq.notes,
        })
        print(f"{tq.qid:4s} [{tq.qtype:10s}] {tq.question[:55]:55s} -> {result.reason:16s} "
              f"hit={hit} grounded={result.grounded} ({elapsed:.1f}s)")

    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
