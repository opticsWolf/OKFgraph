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
        assert r._cuda_fallback is True
        assert "CUDA unavailable" in caplog.text
        assert "CPUExecutionProvider" in r.embedder.session.get_providers()[0]

    def test_device_cuda_warning_only_once(self, tmp_path, caplog):
        import logging
        caplog.set_level(logging.WARNING)
        from okfgraph.router import OKFRouter
        r = OKFRouter(db_path=str(tmp_path / "test.db"), bundle_root=str(tmp_path), device="cuda")
        # Should be exactly one warning, not two
        cuda_warnings = [r for r in caplog.records if "CUDA" in r.message]
        assert len(cuda_warnings) == 1


class TestTools:
    def test_tools_export(self):
        from okfgraph.tools import TOOLS
        assert len(TOOLS) == 4

    def test_tool_names(self):
        from okfgraph.tools import TOOLS
        names = [t["name"] for t in TOOLS]
        assert "search_hybrid" in names
        assert "traverse" in names
        assert "get_by_id" in names
        assert "list_directory" in names

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
