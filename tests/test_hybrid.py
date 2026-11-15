"""Phase 4 Hybrid RAG checks with an in-memory fake collection. No index or model needed."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import settings  # noqa: E402
from src.hybrid import BM25, HybridRetriever, make_retriever, tokenize  # noqa: E402

# Unit vectors: the query embeds to "x", so cosine is the first component.
PRODUCTS = [  # id, name, category, price, embedding
    ("P1", "Generic wireless router fast internet", "Routers & Networking", 900.0, [0.9, 0.1]),
    ("P2", "TP-LINK TL-WR841N router", "Routers & Networking", 1118.0, [0.3, 0.95]),
    ("P3", "Soft bluetooth speaker", "Speakers", 1500.0, [0.6, 0.8]),
    ("P4", "Unknown price router", "Routers & Networking", None, [0.8, 0.6]),
]


def meta(pid, name, category, price):
    return {"product_id": pid, "name": name, "category": category, "brand": "B", "price": price or 0.0,
            "price_known": price is not None, "source_ref": f"ref-{pid}", "source_url": "", "source_row": 1,
            "chunk_index": 0, "chunk_count": 1}


class FakeCollection:
    def __init__(self):
        self.ids = [f"{p[0]}-0" for p in PRODUCTS]
        self.docs = [f"Product: {p[1]} | Category: {p[2]}" for p in PRODUCTS]
        self.metas = [meta(*p[:4]) for p in PRODUCTS]
        self.vecs = [p[4] for p in PRODUCTS]

    def count(self):
        return len(self.ids)

    def get(self, ids=None, include=()):
        idx = [self.ids.index(i) for i in ids] if ids else range(len(self.ids))
        out = {"ids": [self.ids[i] for i in idx]}
        if "documents" in include:
            out["documents"] = [self.docs[i] for i in idx]
        if "metadatas" in include:
            out["metadatas"] = [self.metas[i] for i in idx]
        if "embeddings" in include:
            out["embeddings"] = [self.vecs[i] for i in idx]
        return out

    def query(self, query_embeddings, n_results, where=None, include=()):
        """Semantic ranking by the first component; only the category filter is modelled."""
        q = query_embeddings[0]
        order = sorted(range(len(self.ids)), key=lambda i: -sum(a * b for a, b in zip(q, self.vecs[i])))
        if where and "category" in where:
            order = [i for i in order if self.metas[i]["category"] == where["category"]["$eq"]]
        order = order[:n_results]
        return {"ids": [[self.ids[i] for i in order]], "documents": [[self.docs[i] for i in order]],
                "metadatas": [[self.metas[i] for i in order]],
                "distances": [[1 - sum(a * b for a, b in zip(q, self.vecs[i])) for i in order]]}


class FakeEmbedder:
    model_name = settings.EMBEDDING_MODEL

    def embed_query(self, text):
        return [1.0, 0.0]


def hybrid():
    return HybridRetriever(collection=FakeCollection(), embedder=FakeEmbedder())


class BM25Tests(unittest.TestCase):
    def test_tokenize_splits_model_codes(self):
        self.assertEqual(tokenize("TP-LINK TL-WR841N!"), ["tp", "link", "tl", "wr841n"])

    def test_rare_exact_term_scores_highest(self):
        scores = BM25([tokenize(d) for d in ["router fast", "tl wr841n router", "speaker"]]).scores(["wr841n"])
        self.assertEqual(max(range(3), key=scores.__getitem__), 1)
        self.assertEqual(scores[2], 0.0)


class HybridTests(unittest.TestCase):
    def test_model_number_is_lifted_by_keyword_search(self):
        r = hybrid().search("TL-WR841N", 2)
        self.assertEqual([x.product_id for x in r][:1], ["P2"])
        self.assertEqual([x.rank for x in r], [1, 2])

    def test_scores_stay_cosine_similarity(self):
        r = {x.product_id: x.score for x in hybrid().search("TL-WR841N", 4)}
        self.assertAlmostEqual(r["P2"], 0.3, places=3)
        self.assertAlmostEqual(r["P1"], 0.9, places=3)

    def test_keyword_hits_respect_price_and_category_filters(self):
        h = hybrid()
        self.assertEqual(h.keyword_search("router", 5, max_price=1000), [0])  # P2 too dear, P4 price unknown
        self.assertEqual(h.keyword_search("router speaker", 5, categories=["Speakers"]), [2])

    def test_results_are_unique_products(self):
        ids = [x.product_id for x in hybrid().search("router", 4)]
        self.assertEqual(len(ids), len(set(ids)))


class ModeTests(unittest.TestCase):
    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            make_retriever("fuzzy")

    def test_mode_comes_from_environment(self):
        with mock.patch.dict(os.environ, {settings.RETRIEVAL_MODE_ENV: "semantic"}):
            self.assertEqual(settings.retrieval_mode(), "semantic")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(settings.retrieval_mode(), settings.DEFAULT_RETRIEVAL_MODE)


if __name__ == "__main__":
    unittest.main()
