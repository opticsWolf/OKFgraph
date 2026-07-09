"""Tests for OKFRouter and ConceptModel."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from okfgraph.models import ConceptModel


# ------------------------------------------------------------------
# ConceptModel tests
# ------------------------------------------------------------------


class TestConceptModel:
    def test_minimal_concept(self):
        concept = ConceptModel(id="test", type="note")
        assert concept.id == "test"
        assert concept.type == "note"
        assert concept.body == ""
        assert concept.tags == []

    def test_full_concept(self):
        concept = ConceptModel(
            id="docs/intro",
            type="chapter",
            title="Introduction",
            description="Welcome to the docs",
            tags=["guide", "intro"],
            timestamp="2024-01-15T10:00:00Z",
            body="# Hello World",
        )
        assert concept.title == "Introduction"
        assert isinstance(concept.timestamp, datetime)
        assert concept.tags == ["guide", "intro"]

    def test_timestamp_iso_format(self):
        concept = ConceptModel(id="t", type="note", timestamp="2024-06-01T12:00:00+00:00")
        assert isinstance(concept.timestamp, datetime)

    def test_timestamp_none(self):
        concept = ConceptModel(id="t", type="note", timestamp=None)
        assert concept.timestamp is None

    def test_extra_fields_preserved(self):
        concept = ConceptModel(
            id="x", type="note", custom_key="custom_value", another=42
        )
        assert concept.custom_key == "custom_value"
        assert concept.another == 42

    def test_model_dump_includes_extra(self):
        concept = ConceptModel(id="x", type="note", author="Alice")
        dump = concept.model_dump()
        assert "author" in dump
        assert dump["author"] == "Alice"

    def test_embedding_field(self):
        embedding = [0.1] * 384
        concept = ConceptModel(id="e", type="note", embedding=embedding)
        assert len(concept.embedding) == 384

    def test_concept_id_format(self):
        # Concept IDs always use forward slashes (cross-platform consistency)
        concept = ConceptModel(id="path/to/concept", type="note")
        assert concept.id == "path/to/concept"


# ------------------------------------------------------------------
# OKFRouter smoke tests (no real DB — mocked)
# ------------------------------------------------------------------


class TestOKFRouterSmoke:
    """Smoke tests that verify the OKFRouter imports and basic structure
    without needing a real LadybugDB instance."""

    def test_import_okf_router(self):
        from okfgraph.router import OKFRouter
        assert OKFRouter is not None

    def test_encode_method_exists(self):
        from okfgraph.router import OKFRouter
        assert hasattr(OKFRouter, "_encode")

    def test_search_hybrid_method_exists(self):
        from okfgraph.router import OKFRouter
        assert hasattr(OKFRouter, "search_hybrid")

    def test_traverse_method_exists(self):
        from okfgraph.router import OKFRouter
        assert hasattr(OKFRouter, "traverse")

    def test_list_directory_method_exists(self):
        from okfgraph.router import OKFRouter
        assert hasattr(OKFRouter, "list_directory")

    def test_get_by_id_method_exists(self):
        from okfgraph.router import OKFRouter
        assert hasattr(OKFRouter, "get_by_id")

    def test_import_from_okf_method_exists(self):
        from okfgraph.router import OKFRouter
        assert hasattr(OKFRouter, "import_from_okf")

    def test_export_to_okf_method_exists(self):
        from okfgraph.router import OKFRouter
        assert hasattr(OKFRouter, "export_to_okf")

    def test_list_broken_links_method_exists(self):
        from okfgraph.router import OKFRouter
        assert hasattr(OKFRouter, "list_broken_links")

    def test_repair_links_method_exists(self):
        from okfgraph.router import OKFRouter
        assert hasattr(OKFRouter, "repair_links")


# ------------------------------------------------------------------
# LLM tools tests
# ------------------------------------------------------------------


class TestCacheManagement:
    """Tests for model cache management features."""

    def test_default_cache_dir(self):
        from okfgraph.router import OKFRouter
        default = OKFRouter.default_cache_dir()
        assert default is not None
        assert "huggingface" in default

    def test_model_info_returns_dict(self):
        from okfgraph.router import OKFRouter
        info = OKFRouter.model_info()
        assert isinstance(info, dict)
        assert "model_id" in info
        assert "cache_dir" in info
        assert "cached" in info
        assert "disk_usage_bytes" in info

    def test_model_info_custom_cache_dir(self):
        from okfgraph.router import OKFRouter
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            info = OKFRouter.model_info(cache_dir=tmp)
            assert info["cache_dir"] == tmp
            assert info["cached"] is False  # empty dir

    def test_model_info_model_id(self):
        from okfgraph.router import OKFRouter
        info = OKFRouter.model_info(model_id="jinaai/jina-embeddings-v5-text-small-retrieval")
        assert info["model_id"] == "jinaai/jina-embeddings-v5-text-small-retrieval"


class TestDeviceSelection:
    """Tests for device/provider selection with CUDA fallback."""

    def test_device_cpu_default(self, tmp_path):
        from okfgraph.router import OKFRouter
        r = OKFRouter(db_path=str(tmp_path / "test.db"), bundle_root=str(tmp_path), device="cpu")
        assert r.device == "cpu"
        assert r._cuda_fallback is False

    def test_device_cuda_fallback(self, tmp_path, caplog):
        import logging
        caplog.set_level(logging.WARNING)
        from okfgraph.router import OKFRouter
        r = OKFRouter(db_path=str(tmp_path / "test.db"), bundle_root=str(tmp_path), device="cuda")
        assert r.device == "cuda"
        # If CUDA is available on this machine, no fallback occurs (happy path).
        # If CUDA is unavailable, fallback to CPU is triggered.
        if r._cuda_fallback:
            assert "CUDA unavailable" in caplog.text
            assert "CPUExecutionProvider" in r.embedder.session.get_providers()[0]
        else:
            # CUDA is available — verify it's the primary provider
            assert "CUDAExecutionProvider" in r.embedder.session.get_providers()

    def test_device_cuda_warning_only_once(self, tmp_path, caplog):
        import logging
        caplog.set_level(logging.WARNING)
        from okfgraph.router import OKFRouter
        r = OKFRouter(db_path=str(tmp_path / "test.db"), bundle_root=str(tmp_path), device="cuda")
        cuda_warnings = [rec for rec in caplog.records if "CUDA" in rec.message]
        # If CUDA is available: 0 warnings. If unavailable: exactly 1 warning.
        assert len(cuda_warnings) <= 1


class TestTools:
    def test_tools_export(self):
        from okfgraph.tools import TOOLS
        assert len(TOOLS) == 16  # 13 original + ingest_md + ingest_thoughts + ingest_pdf

    def test_tool_names(self):
        from okfgraph.tools import TOOLS
        names = [t["name"] for t in TOOLS]
        assert "search_hybrid" in names
        assert "traverse" in names
        assert "get_by_id" in names
        assert "list_directory" in names
        assert "ingest_md" in names
        assert "ingest_thoughts" in names

    def test_search_hybrid_has_query_param(self):
        from okfgraph.tools import TOOLS
        tool = next(t for t in TOOLS if t["name"] == "search_hybrid")
        assert "query" in tool["parameters"]["properties"]
        assert "query" in tool["parameters"]["required"]

    def test_traverse_has_start_id_param(self):
        from okfgraph.tools import TOOLS
        tool = next(t for t in TOOLS if t["name"] == "traverse")
        assert "start_id" in tool["parameters"]["properties"]
        assert "start_id" in tool["parameters"]["required"]

    def test_ingest_md_tool_parameters(self):
        from okfgraph.tools import TOOLS
        tool = next(t for t in TOOLS if t["name"] == "ingest_md")
        assert "md_path" in tool["parameters"]["required"]
        assert "auto_import" not in tool["parameters"]["properties"]  # not exposed to LLM
        assert tool["parameters"]["properties"]["mode"]["enum"] == ["text", "optional", "omni"]

    def test_ingest_thoughts_tool_parameters(self):
        from okfgraph.tools import TOOLS
        tool = next(t for t in TOOLS if t["name"] == "ingest_thoughts")
        assert set(tool["parameters"]["required"]) == {"thoughts", "topic"}
        assert "topic" in tool["parameters"]["properties"]

    def test_write_tools_reference_each_other(self):
        """Write tools should reference each other in descriptions."""
        from okfgraph.tools import TOOLS
        ingest_md = next(t for t in TOOLS if t["name"] == "ingest_md")
        ingest_thoughts = next(t for t in TOOLS if t["name"] == "ingest_thoughts")
        # ingest_md references ingest_thoughts
        assert "ingest_thoughts" in ingest_md["description"].lower()
        # ingest_thoughts references ingest_md
        assert "ingest_md" in ingest_thoughts["description"].lower()

    def test_read_tools_mention_write_operations(self):
        """Read-only tools should mention write operations in descriptions."""
        from okfgraph.tools import TOOLS
        search = next(t for t in TOOLS if t["name"] == "search_hybrid")
        assert "ingest_md" in search["description"].lower()
        traverse = next(t for t in TOOLS if t["name"] == "traverse")
        assert "ingest_thoughts" in traverse["description"].lower()


class TestIngestMd:
    """Tests for OKFRouter.ingest_md()."""

    def test_import_existing_file(self, tmp_path):
        """Import a valid markdown file."""
        from okfgraph.router import OKFRouter

        md_path = tmp_path / "test.md"
        md_path.write_text(
            "---\ntitle: Test\n---\n\nHello world.",
            encoding="utf-8",
        )

        r = OKFRouter(
            db_path=str(tmp_path / "test.db"),
            bundle_root=str(tmp_path),
            device="cpu",
        )
        result = r.ingest_md(md_path)

        assert "concept_id" in result
        assert result["title"] == "Test"
        assert result["lint_issues"]["error_count"] == 0
        r.close()

    def test_import_with_linting(self, tmp_path):
        """Linting auto-fixes fixable issues."""
        from okfgraph.router import OKFRouter

        md_path = tmp_path / "test.md"
        md_path.write_text(
            "---\ntitle: Test\n---\n\nHello  \n\nWorld",
            encoding="utf-8",
        )  # trailing spaces (MD009)

        r = OKFRouter(
            db_path=str(tmp_path / "test.db"),
            bundle_root=str(tmp_path),
            device="cpu",
        )
        result = r.ingest_md(md_path)

        assert result["lint_issues"]["fixed_count"] > 0
        r.close()

    def test_import_nonexistent_file(self, tmp_path):
        """Importing a non-existent file raises FileNotFoundError."""
        from okfgraph.router import OKFRouter

        r = OKFRouter(
            db_path=str(tmp_path / "test.db"),
            bundle_root=str(tmp_path),
            device="cpu",
        )
        with pytest.raises(FileNotFoundError):
            r.ingest_md("/nonexistent/path.md")
        r.close()

    def test_import_with_explicit_metadata(self, tmp_path):
        """Explicit metadata overrides frontmatter."""
        from okfgraph.router import OKFRouter

        md_path = tmp_path / "test.md"
        md_path.write_text(
            "---\ntitle: Frontmatter\n---\n\nHello world.",
            encoding="utf-8",
        )

        r = OKFRouter(
            db_path=str(tmp_path / "test.db"),
            bundle_root=str(tmp_path),
            device="cpu",
        )
        result = r.ingest_md(
            md_path,
            title="Override",
            tags=["custom", "test"],
        )

        assert result["title"] == "Override"
        assert "custom" in result["tags"]
        assert "test" in result["tags"]
        r.close()


class TestIngestThoughts:
    """Tests for OKFRouter.ingest_thoughts()."""

    def test_store_reasoning(self, tmp_path):
        """Store reasoning as a searchable concept."""
        from okfgraph.router import OKFRouter

        r = OKFRouter(
            db_path=str(tmp_path / "test.db"),
            bundle_root=str(tmp_path),
            device="cpu",
        )
        result = r.ingest_thoughts(
            thoughts="I think we should use X because Y and Z.",
            topic="architecture",
        )

        assert "concept_id" in result
        assert result["topic"] == "architecture"
        assert "thought" in result["tags"]
        assert "reasoning" in result["tags"]

        # Verify it's stored as a concept
        concept = r.get_by_id(result["concept_id"])
        assert concept is not None
        assert concept.type == "thought"
        r.close()

    def test_searchable_as_concept(self, tmp_path):
        """Stored thoughts are searchable via graph queries."""
        from okfgraph.router import OKFRouter

        r = OKFRouter(
            db_path=str(tmp_path / "test.db"),
            bundle_root=str(tmp_path),
            device="cpu",
        )
        result = r.ingest_thoughts(
            thoughts="The best approach is to use a graph database.",
            topic="database",
        )

        # Search should find it
        results = r.search_hybrid("graph database")
        ids = [r["id"] for r in results]
        assert result["concept_id"] in ids
        r.close()

    def test_explicit_concept_id(self, tmp_path):
        """Explicit concept_id is used as-is."""
        from okfgraph.router import OKFRouter

        r = OKFRouter(
            db_path=str(tmp_path / "test.db"),
            bundle_root=str(tmp_path),
            device="cpu",
        )
        result = r.ingest_thoughts(
            thoughts="Test reasoning.",
            topic="test",
            concept_id="my_custom_id",
        )

        assert result["concept_id"] == "my_custom_id"

    def test_thoughts_linting_applied(self, tmp_path):
        """ingest_thoughts lints the generated markdown in-memory."""
        from okfgraph.router import OKFRouter

        r = OKFRouter(
            db_path=str(tmp_path / "test.db"),
            bundle_root=str(tmp_path),
            device="cpu",
        )
        # Thoughts with trailing whitespace and extra blank lines
        bad_thoughts = "   This has trailing spaces.   \n\n\n\n\nParagraph two.   "
        result = r.ingest_thoughts(
            thoughts=bad_thoughts,
            topic="linting_test",
        )
        assert result["concept_id"].startswith("thought_")
        # Lint result should be present
        assert "lint_issues" in result
        lint = result["lint_issues"]
        assert isinstance(lint, dict)
        assert "fixed_count" in lint
        # The fixed markdown should have trailing spaces removed
        assert lint["fixed"] is True
        assert lint["fixed_count"] > 0
        r.close()
