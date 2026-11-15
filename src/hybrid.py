"""Phase 4 Hybrid RAG: BM25 keyword search fused with the semantic search by Reciprocal Rank Fusion.

    from src.hybrid import HybridRetriever
    HybridRetriever().search("TP-LINK TL-WR841N", 5, max_price=1500)

Semantic search finds meaning ("earphones" ~ "headset"); BM25 finds exact terms such as
model numbers. Both rank unique products under the same price and category filters, and
RRF adds 1 / (RRF_K + rank) from each list. Each result keeps its cosine similarity as
``score`` (computed for keyword-only hits) so the pipeline's evidence gate means the same
thing in both modes. The BM25 index is built in memory from the ChromaDB collection.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from src import settings
from src.retrieval import RetrievedChunk, Retriever

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _passes(meta: dict, max_price: float | None, min_price: float | None, categories: list[str] | None) -> bool:
    """Same rules as ``retrieval.build_where``: a price filter requires a known price."""
    if max_price is not None or min_price is not None:
        if not meta.get("price_known"):
            return False
        if max_price is not None and meta["price"] > max_price:
            return False
        if min_price is not None and meta["price"] < min_price:
            return False
    return not categories or meta["category"] in categories


class BM25:
    """Okapi BM25 over chunks."""

    def __init__(self, docs: list[list[str]], k1: float = settings.BM25_K1, b: float = settings.BM25_B):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.avg_length = sum(self.lengths) / len(docs) if docs else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, length in zip(self.tf, self.lengths):
            s = 0.0
            for t in query:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg_length))
            out.append(s)
        return out


class HybridRetriever(Retriever):
    """Drop-in ``Retriever`` whose ``search`` fuses semantic and BM25 rankings."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data = self.collection.get(include=["documents", "metadatas"])
        self._ids, self._docs, self._metas = data["ids"], data["documents"], data["metadatas"]
        self._bm25 = BM25([tokenize(d) for d in self._docs])

    def keyword_search(self, query: str, k: int, *, max_price: float | None = None, min_price: float | None = None,
                       categories: list[str] | None = None) -> list[int]:
        """Positions of the best-scoring chunk for the top ``k`` products, best first."""
        scores = self._bm25.scores(tokenize(query))
        order = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: -scores[i])
        picked, seen = [], set()
        for i in order:
            meta = self._metas[i]
            if meta["product_id"] in seen or not _passes(meta, max_price, min_price, categories):
                continue
            seen.add(meta["product_id"])
            picked.append(i)
            if len(picked) == k:
                break
        return picked

    def search(self, query: str, k: int = settings.DEFAULT_TOP_K, *, max_price: float | None = None,
               min_price: float | None = None, categories: list[str] | None = None,
               unique_products: bool = True) -> list[RetrievedChunk]:
        pool = k * settings.CANDIDATE_MULTIPLIER
        semantic = super().search(query, pool, max_price=max_price, min_price=min_price, categories=categories)
        if not unique_products:
            return semantic[:k]
        keyword = self.keyword_search(query, pool, max_price=max_price, min_price=min_price, categories=categories)

        fused: dict[str, float] = {}
        for rank, r in enumerate(semantic, 1):
            fused[r.product_id] = fused.get(r.product_id, 0.0) + 1 / (settings.RRF_K + rank)
        for rank, i in enumerate(keyword, 1):
            pid = self._metas[i]["product_id"]
            fused[pid] = fused.get(pid, 0.0) + 1 / (settings.RRF_K + rank)
        top = sorted(fused, key=lambda p: -fused[p])[:k]

        by_product = {r.product_id: r for r in semantic}
        missing = [i for i in keyword if self._metas[i]["product_id"] in top and self._metas[i]["product_id"] not in by_product]
        if missing:  # keyword-only hits: score them by cosine similarity like the semantic results
            ids = [self._ids[i] for i in missing]
            emb = self.collection.get(ids=ids, include=["embeddings"])
            q = self.embedder.embed_query(query)
            for chunk_id, vec in zip(emb["ids"], emb["embeddings"]):
                i = self._ids.index(chunk_id)
                by_product[self._metas[i]["product_id"]] = self._chunk(i, sum(a * b for a, b in zip(q, vec)))

        return [RetrievedChunk(**{**by_product[p].to_dict(), "rank": n}) for n, p in enumerate(top, 1)]

    def _chunk(self, i: int, score: float) -> RetrievedChunk:
        meta = self._metas[i]
        return RetrievedChunk(
            rank=0, score=round(float(score), 4), chunk_id=self._ids[i], product_id=meta["product_id"],
            name=meta["name"], category=meta["category"], brand=meta.get("brand", ""),
            price=meta.get("price") if meta.get("price_known") else None, source_ref=meta["source_ref"],
            source_url=meta["source_url"], source_row=int(meta["source_row"]),
            chunk_index=int(meta["chunk_index"]), chunk_count=int(meta["chunk_count"]), text=self._docs[i])


def make_retriever(mode: str | None = None) -> Retriever:
    """``settings.RETRIEVAL_MODE`` (env ``RETRIEVAL_MODE``) chooses ``semantic`` or ``hybrid``."""
    mode = (mode or settings.retrieval_mode()).lower()
    if mode not in ("semantic", "hybrid"):
        raise ValueError(f"RETRIEVAL_MODE must be 'semantic' or 'hybrid', not {mode!r}.")
    return HybridRetriever() if mode == "hybrid" else Retriever()
