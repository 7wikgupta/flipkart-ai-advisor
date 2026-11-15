"""Compare semantic and hybrid retrieval on the real index and write docs/hybrid_evaluation.md.

Usage::

    python scripts/hybrid_eval.py

A. Category relevance: the 14 Phase 2 sample queries, scored as in ``retrieval_report.py``
   (expected-category hits in the top five, budget compliance), unfiltered and with the
   filters that ``understand()`` applies in the app.
B. Named-product lookup: 30 catalogue products chosen with a fixed seed, each queried by
   brand and model code. Hit@1, hit@5 and mean reciprocal rank of the exact product.
C. Off-topic questions: the best score must stay under ``MIN_EVIDENCE_SCORE``.
D. Latency per search.

Decision rule, fixed before running: adopt hybrid if it improves named lookup without
losing more than one category hit in total across A, and adds under 50 ms per search.
"""

from __future__ import annotations

import random
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval_report import SAMPLE_QUERIES  # noqa: E402
from src import settings  # noqa: E402
from src.chunking import load_products  # noqa: E402
from src.hybrid import HybridRetriever  # noqa: E402
from src.query_understanding import understand  # noqa: E402
from src.retrieval import Retriever  # noqa: E402

OUTPUT = ROOT / "docs" / "hybrid_evaluation.md"
OFF_TOPIC = ["what is the capital of France", "how do I bake a chocolate cake", "who won the cricket world cup",
             "best smartphone camera", "Sony PlayStation 5 games"]
_MODEL_CODE = re.compile(r"^(?=.*\d)(?=.*[a-z])[a-z0-9\-]{4,}$", re.I)


def named_queries(n: int = 30, seed: int = 0) -> list[tuple[str, str, str]]:
    """(query, product_id, name) for products whose name has a model code such as TL-WR841N."""
    rows = []
    for p in load_products(settings.PROCESSED_CSV):
        code = next((w for w in p.name.split() if _MODEL_CODE.match(w)), None)
        if code:
            rows.append((f"{p.brand} {code}".strip(), p.product_id, p.name))
    random.Random(seed).shuffle(rows)
    return rows[:n]


def timed_search(retriever, query, **kw):
    t = time.perf_counter()
    results = retriever.search(query, settings.DEFAULT_TOP_K, **kw)
    return results, (time.perf_counter() - t) * 1000


def category_test(retriever, filtered: bool, latencies: list[float]) -> list[dict]:
    rows = []
    for qid, query, category, min_hits, max_price in SAMPLE_QUERIES:
        if filtered:
            c = understand(query)
            kw = {"max_price": c.max_price, "min_price": c.min_price, "categories": c.categories or None}
        else:
            kw = {"max_price": max_price}
        results, ms = timed_search(retriever, query, **kw)
        latencies.append(ms)
        hits = sum(r.category == category for r in results)
        limit = kw.get("max_price")
        budget_ok = all(r.price is not None and r.price <= limit for r in results) if limit else True
        rows.append({"id": qid, "query": query, "hits": hits, "min_hits": min_hits, "budget_ok": budget_ok,
                     "passed": bool(results) and budget_ok and hits >= min_hits})
    return rows


def named_test(retriever, queries, latencies: list[float]) -> list[int | None]:
    ranks = []
    for query, pid, _ in queries:
        results, ms = timed_search(retriever, query)
        latencies.append(ms)
        ranks.append(next((r.rank for r in results if r.product_id == pid), None))
    return ranks


def main() -> int:
    semantic = Retriever()
    hybrid = HybridRetriever(collection=semantic.collection, embedder=semantic.embedder)
    named = named_queries()
    res = {}
    for label, r in (("semantic", semantic), ("hybrid", hybrid)):
        lat: list[float] = []
        res[label] = {
            "unfiltered": category_test(r, False, lat),
            "filtered": category_test(r, True, lat),
            "named": named_test(r, named, lat),
            "off_topic": [max((x.score for x in r.search(q, settings.DEFAULT_TOP_K)), default=0.0) for q in OFF_TOPIC],
            "latency": lat,
        }

    def summary(label):
        d = res[label]
        ranks = d["named"]
        return {
            "unfiltered_pass": sum(x["passed"] for x in d["unfiltered"]),
            "unfiltered_hits": sum(x["hits"] for x in d["unfiltered"]),
            "filtered_pass": sum(x["passed"] for x in d["filtered"]),
            "filtered_hits": sum(x["hits"] for x in d["filtered"]),
            "budget_ok": all(x["budget_ok"] for x in d["unfiltered"] + d["filtered"]),
            "hit1": sum(r == 1 for r in ranks), "hit5": sum(r is not None for r in ranks),
            "mrr": sum(1 / r for r in ranks if r) / len(ranks),
            "off_topic_gated": sum(s < settings.MIN_EVIDENCE_SCORE for s in d["off_topic"]),
            "latency_median": statistics.median(d["latency"]), "latency_max": max(d["latency"]),
        }

    s, h = summary("semantic"), summary("hybrid")
    lost_hits = (s["unfiltered_hits"] + s["filtered_hits"]) - (h["unfiltered_hits"] + h["filtered_hits"])
    extra_ms = h["latency_median"] - s["latency_median"]
    off_topic_worse = any(b > a + 1e-9 for a, b in zip(res["semantic"]["off_topic"], res["hybrid"]["off_topic"]))
    adopt = h["mrr"] > s["mrr"] and lost_hits <= 1 and extra_ms < 50 and h["budget_ok"] and not off_topic_worse

    n, q = len(named), len(SAMPLE_QUERIES)
    out = [
        "# Phase 4 Hybrid RAG evaluation", "",
        f"Generated by `python scripts/hybrid_eval.py` on the Phase 2 index ({settings.COLLECTION_NAME}). "
        f"Semantic search is the Phase 2 baseline. Hybrid fuses it with BM25 keyword search (k1={settings.BM25_K1}, "
        f"b={settings.BM25_B}) by Reciprocal Rank Fusion (k={settings.RRF_K}) over the top "
        f"{settings.DEFAULT_TOP_K * settings.CANDIDATE_MULTIPLIER} products of each. No LLM is involved.", "",
        "**Decision rule, fixed before running:** adopt hybrid if named-product MRR improves, at most one category hit "
        "is lost across both category tests, median latency rises by under 50 ms, budgets hold and no off-topic "
        "question scores higher than with semantic search (the evidence gate must behave the same). The semantic "
        "baseline already lets two off-topic questions past the gate; Phase 3's prompt rule 3 rejects those.", "",
        "| Test | Semantic | Hybrid |", "| --- | --- | --- |",
        f"| A1. Category, unfiltered (Phase 2 rule): queries passed / category hits | {s['unfiltered_pass']}/{q} · {s['unfiltered_hits']} | {h['unfiltered_pass']}/{q} · {h['unfiltered_hits']} |",
        f"| A2. Category, app filters from `understand()`: passed / hits | {s['filtered_pass']}/{q} · {s['filtered_hits']} | {h['filtered_pass']}/{q} · {h['filtered_hits']} |",
        f"| A. Every budget result within its limit | {'yes' if s['budget_ok'] else 'NO'} | {'yes' if h['budget_ok'] else 'NO'} |",
        f"| B. Named product at rank 1 | {s['hit1']}/{n} | {h['hit1']}/{n} |",
        f"| B. Named product in top 5 | {s['hit5']}/{n} | {h['hit5']}/{n} |",
        f"| B. Mean reciprocal rank | {s['mrr']:.3f} | {h['mrr']:.3f} |",
        f"| C. Off-topic questions under the {settings.MIN_EVIDENCE_SCORE} gate | {s['off_topic_gated']}/{len(OFF_TOPIC)} | {h['off_topic_gated']}/{len(OFF_TOPIC)} |",
        f"| D. Median / max search latency | {s['latency_median']:.0f} / {s['latency_max']:.0f} ms | {h['latency_median']:.0f} / {h['latency_max']:.0f} ms |",
        "", f"**Result: {'adopt hybrid' if adopt else 'keep semantic'}.** Category hits lost: {lost_hits}. "
        f"Extra median latency: {extra_ms:.0f} ms.", "",
        "## A. Category queries (hits in top 5)", "",
        "| ID | Query | Needs | Semantic unfiltered | Hybrid unfiltered | Semantic filtered | Hybrid filtered |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, (qid, query, *_rest) in enumerate(SAMPLE_QUERIES):
        cells = [res[m][t][i] for t in ("unfiltered",) for m in ("semantic", "hybrid")] + \
                [res[m]["filtered"][i] for m in ("semantic", "hybrid")]
        out.append(f"| {qid} | {query} | {cells[0]['min_hits']} | " +
                   " | ".join(f"{c['hits']}{'' if c['passed'] else ' ✗'}" for c in cells) + " |")
    out += ["", "## B. Named-product queries (rank of the exact product, - = not in top 5)", "",
            "| Query | Product | Semantic | Hybrid |", "| --- | --- | --- | --- |"]
    for (query, _, name), a, b in zip(named, res["semantic"]["named"], res["hybrid"]["named"]):
        out.append(f"| {query} | {name[:60]} | {a or '-'} | {b or '-'} |")
    out += ["", "## C. Off-topic questions (best score; gate is "
            f"{settings.MIN_EVIDENCE_SCORE})", "", "| Question | Semantic | Hybrid |", "| --- | --- | --- |"]
    for qtext, a, b in zip(OFF_TOPIC, res["semantic"]["off_topic"], res["hybrid"]["off_topic"]):
        out.append(f"| {qtext} | {a:.3f} | {b:.3f} |")
    OUTPUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[7:20]))
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
