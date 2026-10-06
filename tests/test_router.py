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
        # Encoding moved to the EmbeddingEngine component (Phase 1 refactor);
        # the facade exposes it via router.embed_engine / the bridge.
        from okfgraph.components import EmbeddingEngine
        assert hasattr(EmbeddingEngine, "_encode")

    def test_search_method_exists(self):
        from okfgraph.router import OKFRouter
        assert hasattr(OKFRouter, "search")  # canonical search op

    def test_traverse_method_exists(self):
        from okfgraph.router import OKFRouter
        assert hasattr(OKFRouter, "traverse")

    def test_list_directory_method_exists(self):
        from okfgraph.router import OKFRouter
        assert hasattr(OKFRouter, "list_directory")

    def test_get_by_id_method_exists(self):
        from okfgraph.router import OKFRouter
        assert hasattr(OKFRouter, "get_by_id")

    def test_import_file_op_exists(self):
        from okfgraph.router import OKFRouter
        assert hasattr(OKFRouter, "import_file")
    def test_export_ops_method_exists(self):
        from okfgraph.router import OKFRouter
        assert hasattr(OKFRouter, "export_concept")
        assert hasattr(OKFRouter, "export_bundle")

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
        for key in ("model_id", "repo", "files", "cache_dir", "cached",
                    "snapshot_path", "disk_usage_bytes"):
            assert key in info

    def test_model_info_custom_cache_dir(self):
        from okfgraph.router import OKFRouter
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            info = OKFRouter.model_info(cache_dir=tmp)
            assert info["cache_dir"] == tmp
            assert info["cached"] is False  # empty dir (offline walker)
            assert info["files"] and all(v is None
                                         for v in info["files"].values())
            assert info["snapshot_path"] is None

    def test_model_info_model_id(self):
        from okfgraph.router import OKFRouter
        info = OKFRouter.model_info(
            model_id="jinaai/jina-embeddings-v5-text-small-retrieval")
        assert info["model_id"] == "jinaai/jina-embeddings-v5-text-small-retrieval"

    def test_model_info_bare_legacy_id_refused(self):
        from okfgraph.router import OKFRouter
        from okfgraph.errors import OKFError
        import pytest
        with pytest.raises(OKFError) as exc:
            OKFRouter.model_info(model_id="auto")
        assert exc.value.code == "BAD_VALUE"

    def test_model_info_default_cache_dir_matches_hub_resolution(self):
        # The display helper must agree with the dir embroider really
        # resolves (HF_HUB_CACHE → … → ~/.cache/huggingface/hub). Both
        # sides read the same env, so this holds on any machine — warm
        # or cold cache, CI or workstation. Compared normpath-normalized:
        # hf-hub 1.0 builds its default with string-concatenated '/'s
        # (mixed separators on Windows) — same directory, different
        # spelling; normpath sees through that.
        import embroider
        import os
        from okfgraph.components.embedding import EmbeddingEngine
        expected = embroider.cache_info(
            "jinaai/jina-embeddings-v5-text-nano-retrieval")["cache_dir"]
        assert os.path.normpath(EmbeddingEngine.default_cache_dir()) == \
            os.path.normpath(expected)

    def test_model_info_unset_cache_dir_uses_effective_default(self):
        # Regression: model_info used to force the suffix-less display dir
        # as an override, reporting cached=False on a warm cache. Unset
        # cache_dir must echo embroider's effective dir instead
        # (normpath: see the hf-hub separator quirk above).
        import embroider
        import os
        from okfgraph.router import OKFRouter
        info = OKFRouter.model_info(precision="fp32")
        expected = embroider.cache_info(
            "jinaai/jina-embeddings-v5-text-small-retrieval",
            precision="fp32")["cache_dir"]
        assert os.path.normpath(info["cache_dir"]) == os.path.normpath(expected)

    def test_model_info_precision_follows_landed_device(self, monkeypatch):
        # The engine degrades CUDA-requested-but-unusable to fp32 weights;
        # model_info must inspect the repo that would really be opened.
        from okfgraph.components import embedding as emb
        from okfgraph.router import OKFRouter
        for usable, want in ((False, "fp32"), (True, "fp16")):
            monkeypatch.setattr(emb, "ort_info",
                                lambda u=usable: {"cuda_usable": u})
            for device in ("auto", "cuda"):
                assert OKFRouter.model_info(device=device)["precision"] == want
            assert OKFRouter.model_info(device="cpu")["precision"] == "fp32"
            assert OKFRouter.model_info(device="cuda",
                                        precision="fp32")["precision"] == "fp32"

    def test_model_info_paths_normalized(self):
        import os
        from okfgraph.router import OKFRouter
        info = OKFRouter.model_info()
        assert info["cache_dir"] == os.path.normpath(info["cache_dir"])

    def test_model_info_old_embroider_refused(self, monkeypatch):
        import embroider
        from okfgraph.errors import OKFError
        from okfgraph.router import OKFRouter
        import pytest
        monkeypatch.delattr(embroider, "cache_info", raising=False)
        with pytest.raises(OKFError) as exc:
            OKFRouter.model_info()
        assert exc.value.code == "EMBROIDER_TOO_OLD"


class TestDeviceSelection:
    """Tests for device selection with CUDA fallback (Rust backend)."""

    def test_device_cpu_default(self, tmp_path):
        from okfgraph.router import OKFRouter
        r = OKFRouter(db_path=str(tmp_path / "test.db"), bundle_root=str(tmp_path), device="cpu")
        assert r.device == "cpu"
        assert r.encoder.used_cuda is False

    def test_device_cuda_uses_gpu_when_available(self, tmp_path, caplog):
        # The fallback warning travels via logging (bound to the real stderr
        # fd at handler creation), so capsys can never see it — assert on the
        # log record instead. Deterministic on GPU and CPU machines alike.
        import logging
        from okfgraph.router import OKFRouter
        with caplog.at_level(logging.WARNING, logger="okfgraph.router"):
            r = OKFRouter(db_path=str(tmp_path / "test.db"), bundle_root=str(tmp_path), device="cuda")
        assert r.device == "cuda"
        if r.encoder.used_cuda:
            # Happy path: GPU active, no fallback warning.
            assert not any("CUDA" in m for m in caplog.messages)
        else:
            # CPU fallback: the router warns about the missing EP.
            assert any("no CUDA execution provider" in m for m in caplog.messages)

    def test_device_cuda_warning_only_once(self, tmp_path, capsys):
        from okfgraph.router import OKFRouter
        r = OKFRouter(db_path=str(tmp_path / "test.db"), bundle_root=str(tmp_path), device="cuda")
        err = capsys.readouterr().err
        # One construction warns at most once.
        assert err.count("falling back to CPU") <= 1


class TestIngestMd:
    """Tests for OKFRouter..ingest("md")."""

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
        result = r.ingest_mgr.ingest("md", md_path=md_path)

        assert "concept_id" in result
        assert result["title"] == "Test"
        assert result["lint_issues"]["error_count"] == 0
        r.close()

    def test_import_with_linting(self, tmp_path):
        """md ingest reports fixable lint issues but never rewrites the source."""
        from okfgraph.router import OKFRouter

        md_path = tmp_path / "test.md"
        original = b"---\ntitle: Test\n---\n\nHello  \n\nWorld"  # MD009 + MD047
        md_path.write_bytes(original)

        r = OKFRouter(
            db_path=str(tmp_path / "test.db"),
            bundle_root=str(tmp_path),
            device="cpu",
        )
        result = r.ingest_mgr.ingest("md", md_path=md_path)

        assert result["lint_issues"]["fixable_count"] > 0
        assert result["lint_issues"]["fixed_count"] == 0
        assert md_path.read_bytes() == original
        r.close()

    def test_md_ingest_image_count(self, tmp_path):
        """image_count counts linked images, not the stats dict's keys (was 7)."""
        from okfgraph.router import OKFRouter

        (tmp_path / "pic.png").write_bytes(
            b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        no_img = tmp_path / "plain.md"
        no_img.write_text("---\ntitle: Plain\n---\n\nSee https://example.org/x.png\n",
                          encoding="utf-8")
        one_img = tmp_path / "one.md"
        one_img.write_text("---\ntitle: One\n---\n\n![a pic](pic.png)\n",
                           encoding="utf-8")
        r = OKFRouter(
            db_path=str(tmp_path / "test.db"),
            bundle_root=str(tmp_path),
            device="cpu",
        )
        try:
            assert r.ingest_mgr.ingest("md", md_path=no_img)["image_count"] == 0
            assert r.ingest_mgr.ingest("md", md_path=one_img, mode="text")["image_count"] == 1
        finally:
            r.close()

    def test_import_nonexistent_file(self, tmp_path):
        """Importing a non-existent file raises OKFError(FILE_NOT_FOUND)."""
        from okfgraph.errors import OKFError
        from okfgraph.router import OKFRouter

        r = OKFRouter(
            db_path=str(tmp_path / "test.db"),
            bundle_root=str(tmp_path),
            device="cpu",
        )
        with pytest.raises(OKFError) as err:
            r.ingest_mgr.ingest("md", md_path="/nonexistent/path.md")
        assert err.value.code == "FILE_NOT_FOUND"
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
        result = r.ingest_mgr.ingest("md", md_path=md_path,
            title="Override",
            tags=["custom", "test"],)

        assert result["title"] == "Override"
        assert "custom" in result["tags"]
        assert "test" in result["tags"]
        r.close()


class TestIngestThoughts:
    """Tests for OKFRouter..ingest("thoughts")."""

    def test_store_reasoning(self, tmp_path):
        """Store reasoning as a searchable concept."""
        from okfgraph.router import OKFRouter

        r = OKFRouter(
            db_path=str(tmp_path / "test.db"),
            bundle_root=str(tmp_path),
            device="cpu",
        )
        result = r.ingest_mgr.ingest("thoughts", thoughts="I think we should use X because Y and Z.",
            topic="architecture",)

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
        result = r.ingest_mgr.ingest("thoughts", thoughts="The best approach is to use a graph database.",
            topic="database",)

        # Search should find it
        results = r.search("graph database")
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
        result = r.ingest_mgr.ingest("thoughts", thoughts="Test reasoning.",
            topic="test",
            concept_id="my_custom_id",)

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
        result = r.ingest_mgr.ingest("thoughts", thoughts=bad_thoughts,
            topic="linting_test",)
        assert result["concept_id"].startswith("thoughts/linting_test/")
        # Lint result should be present
        assert "lint_issues" in result
        lint = result["lint_issues"]
        assert isinstance(lint, dict)
        assert "fixed_count" in lint
        # The fixed markdown should have trailing spaces removed
        assert lint["fixed"] is True
        assert lint["fixed_count"] > 0
        r.close()
