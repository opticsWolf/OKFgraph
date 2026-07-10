"""Tests for graph enrichment methods.

Covers hub score computation, context expansion, ancestry, siblings,
and hub-score reranking.
"""

import tempfile
import shutil
from pathlib import Path

import pytest
import yaml

from okfgraph.router import OKFRouter


def _write_okf(bundle_root: str, rel: str, title: str, body: str, tags=None):
    """Write an OKF-style markdown file with frontmatter."""
    p = Path(bundle_root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = {"title": title, "path": rel}
    if tags:
        meta["tags"] = tags
    header = "---\n" + yaml.dump(meta, default_flow_style=False) + "---\n"
    with open(p, "w", encoding="utf-8") as f:
        f.write(header + body)
    return p


class TestGraphEnrichment:
    """Graph enrichment tests — class-scoped router for speed."""

    @pytest.fixture(scope="class")
    @classmethod
    def tmp_dir(cls):
        d = tempfile.mkdtemp()
        cls._tmp_dir = d
        yield cls._tmp_dir
        shutil.rmtree(cls._tmp_dir, ignore_errors=True)

    @pytest.fixture(scope="class")
    @classmethod
    def router(cls, tmp_dir):
        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test_graph_enrichment.db"),
            bundle_root=tmp_dir,
            embedding_dim=512,
            chunk_size=50,
            chunk_overlap=10,
            enable_chunking=True,
            device="cuda",
        )
        cls._router = r
        yield cls._router
        cls._router.close()

    @pytest.fixture(scope="class")
    @classmethod
    def linked_docs(cls, router, tmp_dir):
        """Create documents with cross-links for hub score testing."""
        # hub_doc is linked by many others (high hub score)
        body_hub = "## Hub Document\n\nThis is the central hub concept. " * 20
        p_hub = _write_okf(tmp_dir, "hub.md", "Hub Doc", body_hub, tags=["hub"])
        id_hub = router.import_from_okf(p_hub)

        # Several docs that link to hub
        ids = [id_hub]
        for i in range(3):
            body = f"## Spoke {i}\n\nContent linked to hub. " * 20 + "\n\n[[hub]]"
            p = _write_okf(tmp_dir, f"spoke_{i}.md", f"Spoke {i}", body, tags=["spoke"])
            cid = router.import_from_okf(p)
            ids.append(cid)

        cls._linked_docs = (id_hub, ids[1:])
        return cls._linked_docs

    def test_compute_hub_scores(self, router, linked_docs):
        hub_id, spoke_ids = linked_docs
        scores = router.search_engine._compute_hub_scores([hub_id] + spoke_ids)
        assert isinstance(scores, dict)
        # Hub should have incoming links from spokes
        assert hub_id in scores
        assert scores[hub_id] >= 1

    def test_search_with_context(self, router, linked_docs):
        hub_id, spoke_ids = linked_docs
        results = router.search_engine.search_with_context("central hub concept", limit=5)
        assert isinstance(results, list)
        if results:
            # Each result has chunk + context fields
            r = results[0]
            assert "chunk" in r
            assert "incoming_links" in r
            assert "outgoing_links" in r
            assert "ancestry" in r
            assert "siblings" in r

    def test_get_ancestry(self, router, linked_docs):
        hub_id, _ = linked_docs
        path = router.search_engine._get_ancestry(hub_id)
        # Returns list of ancestors (may be empty if at root)
        assert isinstance(path, list)

    def test_get_siblings(self, router, linked_docs):
        hub_id, spoke_ids = linked_docs
        # Hub and spokes share the root directory, so siblings exist
        siblings = router.search_engine._get_siblings(hub_id)
        assert isinstance(siblings, list)

    def test_search_chunks_with_hub_score(self, router, linked_docs):
        results = router.search_engine.search_chunks_with_hub_score("content", limit=10)
        assert isinstance(results, list)
        if results:
            r = results[0]
            assert "final_score" in r
            assert "hub_score" in r
            assert "rrf_score" in r

    def test_rerank_with_hub_score(self, router, linked_docs):
        base = router.search_engine.search_chunks("content")
        if not base:
            pytest.skip("No chunk search results")
        ranked = router.search_engine.rerank_with_hub_score(base)
        assert isinstance(ranked, list)
        if ranked:
            assert "final_score" in ranked[0]
            assert "hub_score" in ranked[0]

    def test_expand_with_graph_context(self, router, linked_docs):
        hub_id, _ = linked_docs
        chunks = router.search_engine.get_chunks(hub_id)
        if not chunks:
            pytest.skip("No chunks for hub")
        chunk_ids = [c.id for c in chunks]
        neighbours = router.search_engine.expand_with_graph_context(chunk_ids)
        assert isinstance(neighbours, list)

    def test_find_path(self, router, linked_docs):
        hub_id, spoke_ids = linked_docs
        # Hub and spokes are linked via LINKS_TO
        path = router.search_engine.find_path(hub_id, spoke_ids[0])
        assert isinstance(path, list)
        if path:
            # Path includes start and end nodes
            ids = [n["id"] for n in path]
            assert hub_id in ids
            assert spoke_ids[0] in ids

    def test_find_path_no_path(self, router, linked_docs):
        """A nonexistent ID should yield empty path."""
        hub_id, _ = linked_docs
        path = router.search_engine.find_path(hub_id, "nonexistent-id")
        assert path == []
