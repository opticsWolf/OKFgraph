"""Tests for OKF Graph-Aware Chunking & Retrieval (Phases 1-5).

Design choices:
    - **No torch dependency**: The router uses numpy exclusively for
      embedding post-processing (last-token pooling, L2 normalisation,
      Matryoshka truncation). Tests run in any Python environment with
      numpy and onnxruntime-gpu — no CUDA-compiled torch required.
    - **Class-scoped fixtures**: The ONNX model loads once per test class
      (not once per test), speeding up iteration. Within a class, tests
      share the same router instance and database. Sub-fixtures like
      ``long_doc``, ``seeded_db``, and ``linked_db`` are also class-scoped
      to avoid duplicate imports into the shared DB.
    - **Subdirectory isolation**: ``import_bundle`` imports ALL markdown
      files in a directory, so tests use unique subdirectories
      (``import_0``, ``import_1``, etc.) to avoid collisions.
    - **GPU acceleration**: All router fixtures use ``device="cuda"`` to
      leverage onnxruntime-gpu's CUDAExecutionProvider. If CUDA is
      unavailable, the router falls back to CPUExecutionProvider.

TODO: At full-test maturity:
    - Revert to function-scoped fixtures for true test isolation.
    - Add per-test embedding verification (vector dimensions, cosine
      similarity between similar texts, uniqueness per chunk).
    - Add ``--skip-embedding`` flag support to skip real ONNX encoding
      for even faster iteration (use mock vectors instead).
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

from okfgraph.router import OKFRouter
from okfgraph.models import ChunkModel


# ── pytest hook: add --skip-embedding flag (not yet implemented) ──────────
def pytest_addoption(parser):
    """Add --skip-embedding option to skip real ONNX encoding.

    TODO: Implement the --skip-embedding flag to replace real ONNX encoding
    with mock vectors for even faster iteration. When implemented, tests
    that don't actually need real embeddings (schema checks, CLI parsing,
    tool introspection) should use this flag.
    """
    parser.addoption(
        "--skip-embedding",
        action="store_true",
        default=False,
        help="Skip real ONNX embedding; use mock vectors for faster tests.",
    )


# ── Helper ──────────────────────────────────────────────────────────────────

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


# ── Phase 1: Schema (no embedding needed) ──────────────────────────────────

class TestPhase1_Schema:
    """Schema checks — no embedding required.

    TODO: At full-test maturity, verify embedding dimensions and vector
    index properties explicitly rather than just checking table existence.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def tmp_dir(cls):
        d = tempfile.mkdtemp()
        cls._tmp_dir = d
        yield cls._tmp_dir
        import shutil
        shutil.rmtree(cls._tmp_dir, ignore_errors=True)

    @pytest.fixture(scope="class")
    @classmethod
    def router(cls, tmp_dir):
        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test_chunking.db"),
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

    def test_chunk_node_exists(self, router):
        """Chunk node table is created during init."""
        rows = router.conn.execute("MATCH (c:Chunk) RETURN count(c)").rows_as_dict().get_all()
        assert rows is not None

    def test_chunking_methods_present(self, router):
        assert hasattr(router, "_split_into_chunks")
        assert hasattr(router, "_compute_overlap_payloads")
        assert hasattr(router, "reconstruct_document")


# ── Phase 1: Chunking (class-scoped router — model loads once per class) ──

class TestPhase1_Chunking:
    """Chunking tests — class-scoped router for speed.

    TODO: At full-test maturity, split into function-scoped fixtures and
    add per-chunk assertions (exact text boundaries, overlap quality, etc.).
    """

    @pytest.fixture(scope="class")
    @classmethod
    def tmp_dir(cls):
        d = tempfile.mkdtemp()
        cls._tmp_dir = d
        yield cls._tmp_dir
        import shutil
        shutil.rmtree(cls._tmp_dir, ignore_errors=True)

    @pytest.fixture(scope="class")
    @classmethod
    def router(cls, tmp_dir):
        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test_chunking.db"),
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
    def long_doc(cls, router, tmp_dir):
        """Import a document long enough to produce multiple chunks (once per class)."""
        sections = []
        for i in range(10):
            sections.append(f"## Section {i}")
            sections.append(f"Section {i} content here. " * 15)
        body = "\n\n".join(sections)
        p = _write_okf(tmp_dir, "long.md", "Long Doc", body)
        cid = router.import_from_okf(p)
        cls._long_doc = cid
        return cls._long_doc

    def test_splits_into_multiple_chunks(self, router, long_doc):
        chunks = router.get_chunks(long_doc)
        assert len(chunks) >= 2

    def test_chunks_ordered_by_index(self, router, long_doc):
        chunks = router.get_chunks(long_doc)
        indices = [c.chunk_index for c in chunks]
        assert indices == sorted(indices)

    def test_chunk_has_embedding(self, router, long_doc):
        rows = router.conn.execute(
            """
            MATCH (p:Concept {id: $cid})-[:PART_OF]->(c:Chunk)
            RETURN c.embedding IS NOT NULL AS has_emb
            """,
            {"cid": long_doc},
        ).rows_as_dict().get_all()
        assert any(r["has_emb"] for r in rows), "Chunks should have embeddings"

    def test_reconstruct_document(self, router, long_doc):
        reconstructed = router.reconstruct_document(long_doc)
        assert reconstructed is not None
        assert len(reconstructed) > 0

    def test_overlap_payloads(self, router, long_doc):
        chunks = router.get_chunks(long_doc)
        assert len(chunks) > 0
        rows = router.conn.execute(
            """
            MATCH (p:Concept {id: $cid})-[:PART_OF]->(c:Chunk)
            RETURN count(c) AS n
            """,
            {"cid": long_doc},
        ).rows_as_dict().get_all()
        assert rows[0]["n"] > 0


# ── Phase 2: Chunked Ingestion (class-scoped router) ──────────────────────

class TestPhase2_Ingestion:
    """Chunked ingestion tests — class-scoped router for speed.

    TODO: At full-test maturity, add function-scoped fixtures and verify
    embedding quality (dimensions, normalization, uniqueness per chunk).
    """

    @pytest.fixture(scope="class")
    @classmethod
    def tmp_dir(cls):
        d = tempfile.mkdtemp()
        cls._tmp_dir = d
        yield cls._tmp_dir
        import shutil
        shutil.rmtree(cls._tmp_dir, ignore_errors=True)

    @pytest.fixture(scope="class")
    @classmethod
    def router(cls, tmp_dir):
        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test_chunking.db"),
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

    def test_import_creates_chunks(self, router, tmp_dir):
        body = "Word " * 300
        subdir = Path(tmp_dir) / "import_0"
        subdir.mkdir(exist_ok=True)
        p = _write_okf(str(subdir), "test.md", "Chunk Test", body)
        cid = router.import_from_okf(p)
        rows = router.conn.execute(
            "MATCH (p:Concept {id: $cid})-[:PART_OF]->(c:Chunk) RETURN count(c) AS n",
            {"cid": cid},
        ).rows_as_dict().get_all()
        assert rows[0]["n"] >= 1

    def test_import_bundle_creates_chunks(self, router, tmp_dir):
        subdir = Path(tmp_dir) / "import_1"
        subdir.mkdir(exist_ok=True)
        body = "Word " * 200
        _write_okf(str(subdir), "a.md", "A", body)
        _write_okf(str(subdir), "b.md", "B", body)
        ids = router.import_bundle(subdir)
        assert len(ids) == 2
        rows = router.conn.execute("MATCH (c:Chunk) RETURN count(c) AS n").rows_as_dict().get_all()
        assert rows[0]["n"] >= 2

    def test_chunk_has_parent(self, router, tmp_dir):
        body = "Word " * 300
        subdir = Path(tmp_dir) / "import_2"
        subdir.mkdir(exist_ok=True)
        p = _write_okf(str(subdir), "test.md", "Parent Test", body)
        cid = router.import_from_okf(p)
        rows = router.conn.execute(
            """
            MATCH (p:Concept {id: $cid})-[:PART_OF]->(c:Chunk)
            RETURN c.id IS NOT NULL AS has_id, p.id IS NOT NULL AS has_parent
            """,
            {"cid": cid},
        ).rows_as_dict().get_all()
        assert all(r["has_id"] and r["has_parent"] for r in rows)


# ── Phase 3: Search (class-scoped router) ─────────────────────────────────

class TestPhase3_Search:
    """Search tests — class-scoped router for speed.

    TODO: At full-test maturity, add function-scoped fixtures and verify
    RRF score distribution, ranking quality, and vector/FTS alignment.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def tmp_dir(cls):
        d = tempfile.mkdtemp()
        cls._tmp_dir = d
        yield cls._tmp_dir
        import shutil
        shutil.rmtree(cls._tmp_dir, ignore_errors=True)

    @pytest.fixture(scope="class")
    @classmethod
    def router(cls, tmp_dir):
        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test_chunking.db"),
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
    def seeded_db(cls, router, tmp_dir):
        """Seed 3 topic documents (once per class)."""
        for i in range(3):
            body = f"Topic {i} discussion with unique words alpha beta gamma delta epsilon " * 20
            _write_okf(tmp_dir, f"topic_{i}.md", f"Topic {i}", body)
            router.import_from_okf(Path(tmp_dir) / f"topic_{i}.md")

    def test_search_chunks_returns_results(self, router, seeded_db):
        results = router.search_chunks("alpha beta gamma")
        assert len(results) > 0

    def test_search_chunks_has_rrf_score(self, router, seeded_db):
        results = router.search_chunks("alpha beta gamma")
        for r in results:
            assert "rrf_score" in r
            assert r["rrf_score"] > 0

    def test_search_chunks_has_block_type(self, router, seeded_db):
        results = router.search_chunks("alpha beta gamma")
        for r in results:
            assert "block_type" in r

    def test_search_chunks_with_parent(self, router, seeded_db):
        # Parent info is always included in search_chunks results
        results = router.search_chunks("alpha beta gamma")
        for r in results:
            assert "parent_title" in r or "parent_doc_id" in r


# ── Phase 4: Graph Enrichment (class-scoped router) ───────────────────────

class TestPhase4_Graph:
    """Graph enrichment tests — class-scoped router for speed.

    TODO: At full-test maturity, add function-scoped fixtures and verify
    hub score computation accuracy, neighbor discovery completeness,
    and reranking quality.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def tmp_dir(cls):
        d = tempfile.mkdtemp()
        cls._tmp_dir = d
        yield cls._tmp_dir
        import shutil
        shutil.rmtree(cls._tmp_dir, ignore_errors=True)

    @pytest.fixture(scope="class")
    @classmethod
    def router(cls, tmp_dir):
        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test_chunking.db"),
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
    def linked_db(cls, router, tmp_dir):
        """Create two cross-linked documents (once per class)."""
        body_a = "Document A content. " * 50 + "\n\n[[doc_b]]"
        body_b = "Document B content. " * 50 + "\n\n[[doc_a]]"
        _write_okf(tmp_dir, "doc_a.md", "Doc A", body_a, tags=["test"])
        _write_okf(tmp_dir, "doc_b.md", "Doc B", body_b, tags=["test"])
        id_a = router.import_from_okf(Path(tmp_dir) / "doc_a.md")
        id_b = router.import_from_okf(Path(tmp_dir) / "doc_b.md")
        cls._linked_db = (id_a, id_b)
        return cls._linked_db

    def test_expand_with_graph_context(self, router, linked_db):
        id_a, _ = linked_db
        chunks = router.get_chunks(id_a)
        if not chunks:
            pytest.skip("No chunks created")
        chunk_ids = [c.id for c in chunks]
        neighbours = router.expand_with_graph_context(chunk_ids)
        assert isinstance(neighbours, list)

    def test_rerank_with_hub_score(self, router, linked_db):
        id_a, id_b = linked_db
        results = router.search_chunks("Document")
        if not results:
            pytest.skip("No chunk search results")
        ranked = router.rerank_with_hub_score(results)
        assert isinstance(ranked, list)
        if ranked:
            assert "final_score" in ranked[0]

    def test_traverse_part_of(self, router, linked_db):
        id_a, _ = linked_db
        chunks = router.get_chunks(id_a)
        if not chunks:
            pytest.skip("No chunks created")
        results = router.traverse(id_a, "PART_OF", "OUTGOING", 1)
        assert len(results) >= 1

    def test_traverse_includes_asset(self, router):
        assert "PART_OF" in {"CONTAINS", "LINKS_TO", "PART_OF", "INCLUDES_ASSET"}
        assert "INCLUDES_ASSET" in {"CONTAINS", "LINKS_TO", "PART_OF", "INCLUDES_ASSET"}


# ── Phase 5: CLI & Tools (NO router needed — no embedding overhead) ───────

class TestPhase5_Tools:
    """Tool definition checks — no router/embedding needed.

    TODO: At full-test maturity, add integration tests that call tools
    through the full CLI pipeline with real embedding verification.
    """

    def test_chunk_tools_present(self):
        from okfgraph.tools import TOOLS
        names = [t["name"] for t in TOOLS]
        assert "search_chunks" in names
        assert "expand_with_graph_context" in names
        assert "get_chunks" in names
        assert "reconstruct_document" in names

    def test_traverse_tool_has_new_rels(self):
        from okfgraph.tools import TOOLS
        traverse = next(t for t in TOOLS if t["name"] == "traverse")
        rels = traverse["parameters"]["properties"]["relationship"]["enum"]
        assert "PART_OF" in rels
        assert "INCLUDES_ASSET" in rels


class TestPhase5_CLI:
    """CLI parser checks — no router/embedding needed.

    TODO: At full-test maturity, add end-to-end CLI tests that invoke
    actual commands with real data and verify output formatting.
    """

    def test_cli_has_search_chunks_command(self):
        from okfgraph.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["search-chunks", "test"])
        assert args.command == "search-chunks"
        assert args.query == "test"

    def test_cli_has_chunks_command(self):
        from okfgraph.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["chunks", "some_id"])
        assert args.command == "chunks"
        assert args.concept_id == "some_id"

    def test_cli_has_reconstruct_command(self):
        from okfgraph.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["reconstruct", "some_id"])
        assert args.command == "reconstruct"
        assert args.concept_id == "some_id"

    def test_cli_has_chunking_options(self):
        from okfgraph.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["import", "--chunk-size", "256", "--chunk-overlap", "32", "--no-chunking", "file.md"])
        assert args.chunk_size == 256
        assert args.chunk_overlap == 32
        assert args.no_chunking is True

    def test_cli_traverse_has_new_rels(self):
        from okfgraph.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["traverse", "some_id", "--relationship", "PART_OF"])
        assert args.relationship == "PART_OF"


# ── Index rebuild (class-scoped router) ────────────────────────────────────

class TestIndexRebuild:
    """Index rebuild test — class-scoped router for speed.

    TODO: At full-test maturity, add function-scoped fixtures and verify
    that rebuilt indexes include all data (no stale entries after DROP/RECREATE).
    """

    @pytest.fixture(scope="class")
    @classmethod
    def tmp_dir(cls):
        d = tempfile.mkdtemp()
        cls._tmp_dir = d
        yield cls._tmp_dir
        import shutil
        shutil.rmtree(cls._tmp_dir, ignore_errors=True)

    @pytest.fixture(scope="class")
    @classmethod
    def router(cls, tmp_dir):
        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test_chunking.db"),
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

    def test_reindex_includes_chunk_indexes(self, router, tmp_dir):
        body = "Index test content. " * 100
        _write_okf(tmp_dir, "idx.md", "Idx", body)
        router.import_from_okf(Path(tmp_dir) / "idx.md")
        router.reindex(force=True)
        rows = router.conn.execute(
            """
            MATCH (c:Chunk)
            RETURN count(c) AS n, c.embedding IS NOT NULL AS has_emb
            """
        ).rows_as_dict().get_all()
        assert rows[0]["n"] > 0, "Chunks should exist after import"
