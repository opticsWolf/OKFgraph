"""Live PPR search + budgeted read tests (one class-scoped router).

Phase 1 acceptance: PPR retrieval never touches the embedder (proven by a
stub that raises), hub blending works, and the default hybrid path is
unchanged. Phase 2: budgeted reads fit the cap; uncapped reads are untouched.
"""

import pytest
from pathlib import Path

from okfgraph.router import OKFRouter

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def ppr_router(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("ppr")
    r = OKFRouter(
        db_path=str(tmp_dir / "test_ppr.db"),
        bundle_root=str(FIX / "ppr_bundle"),
        embedding_dim=512,
        chunk_size=50,
        chunk_overlap=10,
        enable_chunking=True,
        device="cpu",
    )
    ids = r.import_mgr.import_bundle(FIX / "ppr_bundle")
    assert len(ids) == 4, f"expected 4 concepts, got {ids}"
    yield r
    r.close()


@pytest.fixture(scope="module")
def router(ppr_router):
    return ppr_router


class TestPprSearch:
    def test_ppr_finds_topic_without_model(self, router, monkeypatch):
        """The embedder stub raises — a PPR hit proves the model is untouched."""

        def _boom(text, task="Document"):
            raise AssertionError("embedder must not load for rank=ppr")

        monkeypatch.setattr(router.embed_engine, "_encode", _boom)
        monkeypatch.setattr(router.embed_engine, "_encode_batch", _boom)
        results = router.search_engine.search_with_ppr("honey badger defense")
        assert results, "expected PPR hits"
        # Single seed (honey-badger) but the hub bridge it points at
        # accumulates more stationary mass — PPR lifts structure over seeds.
        assert results[0]["id"] == "savanna"
        assert "honey-badger" in [r["id"] for r in results[:2]]
        assert all("relevance_score" in r for r in results)

    def test_ppr_hub_bridge_beats_orphan(self, router, monkeypatch):
        """'savanna' seeds savanna.md; PPR lifts its linked hub neighbours."""
        monkeypatch.setattr(
            router.embed_engine, "_encode",
            lambda text, task="Document": (_ for _ in ()).throw(AssertionError("model touched")),
        )
        results = router.search_engine.search_with_ppr("savanna food web")
        ids = [r["id"] for r in results]
        assert ids[0] == "savanna"
        # Linked docs share the seed mass; the unlinked orphan gets none.
        assert {"honey-badger", "lion"} <= set(ids)
        assert "orphan" not in ids

    def test_search_hybrid_rank_ppr_matches_direct(self, router):
        direct = router.search_engine.search_with_ppr("honey badger")
        via_hybrid = router.search_hybrid("honey badger", rank="ppr")
        assert [r["id"] for r in via_hybrid] == [r["id"] for r in direct]

    def test_search_hybrid_rank_hub_blends(self, router):
        plain = router.search_hybrid("savanna")
        hubbed = router.search_hybrid("savanna", rank="hub")
        assert {r["id"] for r in hubbed} == {r["id"] for r in plain}
        assert all("hub_score" in r for r in hubbed)
        # 'savanna' has the most incoming links -> hub weight lifts it to top.
        assert hubbed[0]["id"] == "savanna"

    def test_search_hybrid_default_unchanged(self, router):
        results = router.search_hybrid("honey badger")
        assert results and "hub_score" not in results[0]
        assert all("relevance_score" in r for r in results)

    def test_search_hybrid_bad_rank_raises(self, router):
        with pytest.raises(ValueError):
            router.search_hybrid("honey badger", rank="bogus")


class TestBudgetedRead:
    def test_body_budget_truncates(self, router):
        reading = router.search_engine.read_with_budget(
            "honey-badger", include="body", max_tokens=10,
        )
        assert reading["concept_id"] == "honey-badger"
        assert reading["used"] <= 10
        assert reading["truncated"] is True
        assert len(reading["sections"]) == 1

    def test_generous_budget_keeps_everything(self, router):
        reading = router.search_engine.read_with_budget(
            "orphan", include="body", max_tokens=100000,
        )
        assert reading["truncated"] is False
        full = router.get_by_id("orphan").body
        assert reading["sections"][0]["text"] == full

    def test_context_orders_self_first_then_neighbours(self, router):
        reading = router.search_engine.read_with_budget(
            "savanna", include="context", max_tokens=100000,
        )
        kinds = [s["kind"] for s in reading["sections"]]
        assert kinds[0] == "body"
        neighbour_ids = [s["id"] for s in reading["sections"] if s["kind"] == "neighbour"]
        # PPR from savanna reaches both linked docs, never the orphan.
        assert set(neighbour_ids) == {"honey-badger", "lion"}
        assert "orphan" not in neighbour_ids

    def test_context_budget_cuts_at_cap(self, router):
        reading = router.search_engine.read_with_budget(
            "savanna", include="context", max_tokens=60,
        )
        assert reading["used"] <= 60
        assert reading["truncated"] is True

    def test_unknown_concept_raises(self, router):
        with pytest.raises(KeyError):
            router.search_engine.read_with_budget("nope", max_tokens=100)

    def test_bad_include_raises(self, router):
        with pytest.raises(ValueError):
            router.search_engine.read_with_budget("savanna", include="bogus", max_tokens=100)
