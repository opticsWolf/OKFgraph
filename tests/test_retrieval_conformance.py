"""Retrieval conformance set: fixture bundle + pinned query expectations.

`tests/fixtures/retrieval_bundle/` is a 10-document mini-corpus with
distinctive vocabularies per cluster, a hub structure (savanna ← badger,
elephant; photosynthesis ← mycorrhizae, nitrogen-cycle), and one long
document with a buried term. These tests pin end-to-end retrieval
behavior — the vector/hybrid path the PPR conformance tests in
test_ppr_search.py deliberately do not cover.

Determinism notes: winners below lead by ~2x RRF margin (no near-ties);
the no-match case pins the RRF-floor behavior (dregs, not empty); the PPR
case stubs the encoder to prove zero model load.
"""
from pathlib import Path

import pytest

from okfgraph.router import OKFRouter

FIX = Path(__file__).parent / "fixtures" / "retrieval_bundle"


@pytest.fixture(scope="module")
def retr_router(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("retrieval")
    r = OKFRouter(
        db_path=str(tmp_dir / "test_retrieval.db"),
        bundle_root=str(FIX),
        embedding_dim=512,
        chunk_size=50,
        chunk_overlap=10,
        enable_chunking=True,
        device="cpu",
    )
    ids = r.import_mgr.import_bundle(FIX)
    assert len(ids) == 10, f"expected 10 concepts, got {ids}"
    yield r
    r.close()


def _ids(results):
    return [r["id"] for r in results]


def _score(r):
    return r.get("relevance_score", r.get("score", 0))


class TestRetrievalConformance:
    def test_hybrid_top1_wildlife(self, retr_router):
        res = retr_router.search_hybrid("honey badger hive raiding mustelid")
        assert _ids(res)[0] == "honey-badger"

    def test_hybrid_top1_biology(self, retr_router):
        res = retr_router.search_hybrid("mycorrhizal hyphae symbiosis")
        assert _ids(res)[0] == "mycorrhizae"

    def test_hybrid_top1_programming(self, retr_router):
        res = retr_router.search_hybrid("borrow checker lifetimes affine")
        assert _ids(res)[0] == "rust-borrowck"

    def test_tag_filter_narrows(self, retr_router):
        res = retr_router.search_hybrid("bread baking", tags=["food"])
        assert _ids(res)[0] == "sourdough"
        assert all("food" in r["tags"] for r in res)

    def test_type_filter_narrows(self, retr_router):
        res = retr_router.search_hybrid("threads", concept_type="note")
        assert _ids(res)[0] == "concurrency"
        assert all(r["type"] == "note" for r in res)

    def test_chunk_search_finds_buried_term(self, retr_router):
        chunks = retr_router.search_engine.search_chunks("autolyse")
        assert chunks, "expected chunk hits"
        top = chunks[0]
        assert top["parent_doc_id"] == "sourdough"
        assert "autolyse" in top["chunk_text"]

    def test_hub_rank_lifts_linked_hub(self, retr_router):
        res = retr_router.search_hybrid("africa grassland", rank="hub")
        assert _ids(res)[0] == "savanna"

    def test_ppr_surfaces_hub_without_model(self, retr_router, monkeypatch):
        """Stubbed encoder raises — a hit proves PPR never loads the model."""

        def _boom(text, task="Document"):
            raise AssertionError("embedder must not load for rank=ppr")

        monkeypatch.setattr(retr_router.embed_engine, "_encode", _boom)
        monkeypatch.setattr(retr_router.embed_engine, "_encode_batch", _boom)
        res = retr_router.search_engine.search_with_ppr("grassland grazers")
        assert _ids(res)[0] == "savanna"
        assert "honey-badger" in _ids(res)[:3]

    def test_no_match_returns_rrf_dregs(self, retr_router):
        """Unknown vocabulary matches nothing: hybrid returns RRF-floor
        dregs (1/(60+k)), never an error and never a confident hit."""
        res = retr_router.search_hybrid("xyzzy platypus interferometer")
        assert res, "hybrid always returns the fused list"
        assert max(_score(r) for r in res) < 0.02

    def test_budgeted_read_respects_cap(self, retr_router):
        out = retr_router.search_engine.read_with_budget(
            "sourdough", include="context", max_tokens=200
        )
        assert out["concept_id"] == "sourdough"
        assert out["used"] <= 200
