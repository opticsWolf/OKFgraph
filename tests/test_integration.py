"""End-to-end integration tests.

Imports a full bundle, searches, enriches, traverses, and reconstructs —
exercising the complete pipeline.
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


class TestIntegration:
    """Full pipeline integration tests — class-scoped router for speed."""

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
            db_path=str(Path(tmp_dir) / "test_integration.db"),
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
    def full_bundle(cls, router, tmp_dir):
        """Import a bundle with cross-links and run full pipeline."""
        bundle_dir = Path(tmp_dir) / "bundle"
        bundle_dir.mkdir(exist_ok=True)

        # Write docs with cross-links
        files = [
            ("overview.md", "Overview",
             "## Overview\n\nThis is the overview document covering all topics. " * 20
             + "\n\n[[details]]\n\n[[reference]]",
             ["overview", "index"]),
            ("details.md", "Details",
             "## Details\n\nDetailed information about the core subject matter. " * 20
             + "\n\n[[overview]]",
             ["details"]),
            ("reference.md", "Reference",
             "## Reference\n\nReference material and external resources. " * 20
             + "\n\n[[overview]]\n\n[[details]]",
             ["reference"]),
        ]
        ids = {}
        for fname, title, body, tags in files:
            p = _write_okf(str(bundle_dir), fname, title, body, tags=tags)
            cid = router.import_from_okf(p)
            ids[title.lower()] = cid

        cls._full_bundle = ids
        return cls._full_bundle

    def test_bundle_import_creates_concepts(self, router, full_bundle):
        for title, cid in full_bundle.items():
            concept = router.get_by_id(cid)
            assert concept is not None
            assert concept.id == cid

    def test_bundle_has_chunks(self, router, full_bundle):
        for title, cid in full_bundle.items():
            chunks = router.get_chunks(cid)
            assert len(chunks) >= 1, f"Expected chunks for {title}"

    def test_hybrid_search_finds_concepts(self, router, full_bundle):
        results = router.search_hybrid("core subject matter")
        assert len(results) > 0

    def test_chunk_search_finds_content(self, router, full_bundle):
        results = router.search_chunks("detailed information")
        assert len(results) > 0
        assert all("rrf_score" in r for r in results)

    def test_search_with_graph_filters(self, router, full_bundle):
        results = router.search_hybrid("overview", concept_type="Concept")
        assert isinstance(results, list)

    def test_traverse_part_of_returns_chunks(self, router, full_bundle):
        cid = full_bundle["overview"]
        results = router.traverse(cid, "PART_OF", "OUTGOING", 1)
        assert len(results) >= 1

    def test_reconstruct_document(self, router, full_bundle):
        cid = full_bundle["details"]
        text = router.reconstruct_document(cid)
        assert text is not None
        assert len(text) > 0
        assert "Detailed information" in text or "detailed information" in text.lower()

    def test_search_with_context_returns_enriched(self, router, full_bundle):
        results = router.search_with_context("overview", limit=3)
        if results:
            r = results[0]
            assert "chunk" in r
            assert "incoming_links" in r

    def test_hybrid_search_include_chunks(self, router, full_bundle):
        results = router.search_hybrid("overview", include_chunks=True)
        if results:
            r = results[0]
            assert "matched_chunks" in r

    def test_reindex_preserves_chunks(self, router, full_bundle):
        """Reindexing should not lose chunk data."""
        cid = full_bundle["reference"]
        chunks_before = len(router.get_chunks(cid))
        router.reindex(force=True)
        chunks_after = len(router.get_chunks(cid))
        assert chunks_after == chunks_before

    def test_find_path_between_linked_docs(self, router, full_bundle):
        overview_id = full_bundle["overview"]
        details_id = full_bundle["details"]
        path = router.find_path(overview_id, details_id)
        assert isinstance(path, list)
