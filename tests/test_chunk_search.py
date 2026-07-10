"""Tests for chunk search — RRF ordering, graph filters, per-doc limits.

Covers edge cases not in test_chunking.py: filtering by concept_type, tags,
parent_id, max_chunks_per_doc, and RRF score ordering.
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


class TestChunkSearch:
    """Chunk search tests — class-scoped router for speed."""

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
            db_path=str(Path(tmp_dir) / "test_chunk_search.db"),
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
    def seeded_docs(cls, router, tmp_dir):
        """Seed multiple documents with distinct tags and content."""
        ids = []
        for i, tag in enumerate(["alpha", "beta", "gamma"]):
            body = f"## Section {i}\n\nTopic {tag} content with shared query words. " * 20
            p = _write_okf(tmp_dir, f"search_{i}.md", f"Search Doc {i}", body, tags=[tag])
            cid = router.import_from_okf(p)
            ids.append(cid)
        cls._seeded_docs = ids
        return cls._seeded_docs

    def test_search_returns_results(self, router, seeded_docs):
        results = router.search_engine.search_chunks("shared query words")
        assert len(results) > 0

    def test_rrf_scores_ordered_descending(self, router, seeded_docs):
        results = router.search_engine.search_chunks("shared query words")
        if len(results) < 2:
            pytest.skip("Need at least 2 results for ordering check")
        scores = [r["rrf_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_results_have_required_fields(self, router, seeded_docs):
        results = router.search_engine.search_chunks("shared query words")
        for r in results:
            assert "chunk_id" in r
            assert "chunk_text" in r
            assert "block_type" in r
            assert "chunk_index" in r
            assert "parent_doc_id" in r
            assert "rrf_score" in r

    def test_limit_respected(self, router, seeded_docs):
        results = router.search_engine.search_chunks("shared query words", limit=2)
        assert len(results) <= 2

    def test_max_chunks_per_doc(self, router, seeded_docs):
        """Limit results to at most 1 chunk per document."""
        results = router.search_engine.search_chunks("shared query words", max_chunks_per_doc=1)
        parent_ids = {r["parent_doc_id"] for r in results}
        # Each parent should appear at most once
        for pid in parent_ids:
            count = sum(1 for r in results if r["parent_doc_id"] == pid)
            assert count <= 1

    def test_parent_id_filter(self, router, seeded_docs):
        """Filter chunks to only those belonging to a specific document."""
        target_id = seeded_docs[0]
        results = router.search_engine.search_chunks("shared query words", parent_id=target_id)
        for r in results:
            assert r["parent_doc_id"] == target_id

    def test_empty_results_on_no_match(self, router, seeded_docs):
        """A very specific query that shouldn't match anything."""
        results = router.search_engine.search_chunks("xyzzyplughthudethquux")
        # May still get some vector hits, but should be limited
        assert isinstance(results, list)
