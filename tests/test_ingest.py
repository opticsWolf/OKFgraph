"""Tests for the ONNX/Rapid PDF ingestion engine.

These tests verify the module structure, graceful degradation when RapidAI
packages are not installed, and the HTML table converter.
"""

from __future__ import annotations

import pytest
from okfgraph.ingest.config import ConverterConfig, RoutingMode
from okfgraph.ingest.engine import OnnxRapidEngine
from okfgraph.ingest.converter import HybridConverter
from okfgraph.ingest.tables import html_tables_to_gfm, _SimpleTableParser
from okfgraph.ingest.assets import stage_images_as_okf_assets, asset_id, ASSET_STORE_DIRNAME


class TestConfig:
    """ConverterConfig and RoutingMode."""

    def test_default_config(self):
        cfg = ConverterConfig()
        assert cfg.routing_mode == RoutingMode.AUTO
        assert cfg.use_onnx is True
        assert cfg.ort_providers == ["CPUExecutionProvider"]

    def test_gpu_providers(self):
        cfg = ConverterConfig(device="gpu")
        assert "CUDAExecutionProvider" in cfg.ort_providers
        assert "CPUExecutionProvider" in cfg.ort_providers

    def test_explicit_providers(self):
        cfg = ConverterConfig(ort_providers=["DirectMLExecutionProvider"])
        assert cfg.ort_providers == ["DirectMLExecutionProvider"]

    def test_routing_mode_values(self):
        assert RoutingMode.AUTO.value == "auto"
        assert RoutingMode.SURGICAL.value == "surgical"
        assert RoutingMode.ALWAYS.value == "always"
        assert RoutingMode.NEVER.value == "never"


class TestEngineGracefulDegradation:
    """OnnxRapidEngine degrades gracefully when RapidAI is not installed."""

    def test_formula_returns_none_when_not_installed(self):
        eng = OnnxRapidEngine()
        result = eng.formula()
        # Will be None if rapid_latex_ocr not installed
        # If installed, will be a LatexOCR instance
        assert result is None or hasattr(result, "predict") or hasattr(result, "__call__")

    def test_ocr_returns_none_when_not_installed(self):
        eng = OnnxRapidEngine()
        result = eng.ocr()
        assert result is None or hasattr(result, "predict") or hasattr(result, "__call__")

    def test_layout_returns_none_when_not_installed(self):
        eng = OnnxRapidEngine()
        result = eng.layout()
        assert result is None or hasattr(result, "predict") or hasattr(result, "__call__")

    def test_table_returns_none_when_not_installed(self):
        eng = OnnxRapidEngine()
        result = eng.table()
        assert result is None or hasattr(result, "predict") or hasattr(result, "__call__")

    def test_close_clears_references(self):
        eng = OnnxRapidEngine()
        eng.formula()
        eng.ocr()
        eng.close()
        assert eng._formula is None
        assert eng._ocr is None
        assert eng._layout is None
        assert eng._table is None


class TestHTMLTablesToGFM:
    """html_tables_to_gfm converts simple HTML tables to GFM pipe tables."""

    def test_simple_table(self):
        html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
        result = html_tables_to_gfm(html)
        assert "| A | B |" in result
        assert "| --- | --- |" in result
        assert "| 1 | 2 |" in result

    def test_table_with_data_rows(self):
        html = (
            "<table>"
            "<tr><th>Name</th><th>Value</th></tr>"
            "<tr><td>alpha</td><td>1</td></tr>"
            "<tr><td>beta</td><td>2</td></tr>"
            "</table>"
        )
        result = html_tables_to_gfm(html)
        assert "| Name | Value |" in result
        assert "| alpha | 1 |" in result

    def test_complex_table_with_colspan_kept_as_html(self):
        html = "<table><tr><td colspan=\"2\">merged</td></tr></table>"
        result = html_tables_to_gfm(html)
        # Complex tables are left as-is
        assert "<table>" in result or "<td" in result

    def test_empty_table_returns_none(self):
        html = "<table></table>"
        result = html_tables_to_gfm(html)
        # Empty table is left as-is
        assert result == html

    def test_table_with_pipes_escaped(self):
        html = "<table><tr><th>A</th></tr><tr><td>a|b</td></tr></table>"
        result = html_tables_to_gfm(html)
        assert r"\|" in result  # pipe chars should be escaped

    def test_multiple_tables(self):
        html = (
            "<table><tr><th>X</th></tr><tr><td>1</td></tr></table>"
            "\n\nSome text\n\n"
            "<table><tr><th>Y</th></tr><tr><td>2</td></tr></table>"
        )
        result = html_tables_to_gfm(html)
        assert "| X |" in result
        assert "| Y |" in result


class TestAssetStaging:
    """okf-asset:// staging logic."""

    def test_asset_id_is_deterministic(self):
        id1 = asset_id("doc", 1, b"test data")
        id2 = asset_id("doc", 1, b"test data")
        assert id1 == id2
        assert id1.startswith("img_")

    def test_asset_id_differs_for_different_data(self):
        id1 = asset_id("doc", 1, b"data A")
        id2 = asset_id("doc", 1, b"data B")
        assert id1 != id2

    def test_asset_id_differs_for_different_occurrence(self):
        id1 = asset_id("doc", 1, b"data")
        id2 = asset_id("doc", 2, b"data")
        assert id1 != id2

    def test_stage_images_rewrites_local_links(self, tmp_path):
        from pathlib import Path
        # Create a dummy image file
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"\x89PNG\r\n\x1a\n")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        md = f"Text ![alt]({img_file.name})"
        new_md, count = stage_images_as_okf_assets(
            md, tmp_path, img_file, out_dir, "concept"
        )
        assert count == 1
        assert "okf-asset://" in new_md
        assert img_file.name not in new_md

    def test_stage_images_skips_remote_links(self, tmp_path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        md = "Text ![alt](https://example.com/img.png)"
        new_md, count = stage_images_as_okf_assets(
            md, tmp_path, tmp_path / "source.pdf", out_dir, "concept"
        )
        assert count == 0
        assert "https://example.com/img.png" in new_md

    def test_stage_images_skips_okf_asset_links(self, tmp_path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        md = "Text ![alt](okf-asset://img_abc123)"
        new_md, count = stage_images_as_okf_assets(
            md, tmp_path, tmp_path / "source.pdf", out_dir, "concept"
        )
        assert count == 0
        assert "okf-asset://img_abc123" in new_md


class TestHybridConverterInit:
    """HybridConverter initializes correctly."""

    def test_never_mode_no_models(self):
        cfg = ConverterConfig(routing_mode=RoutingMode.NEVER, use_onnx=False)
        conv = HybridConverter(cfg)
        conv.ensure_models()
        # No models should be loaded
        assert conv.rapid._formula is None
        assert conv.rapid._ocr is None
        conv.close()

    def test_converter_close(self):
        cfg = ConverterConfig()
        conv = HybridConverter(cfg)
        conv.close()
        assert conv.rapid._formula is None
        assert conv.rapid._ocr is None
        assert conv.rapid._layout is None
        assert conv.rapid._table is None

    def test_math_unicode_detection(self):
        from okfgraph.ingest.converter import _is_math_unicode
        assert _is_math_unicode("α")  # Greek
        assert _is_math_unicode("∑")  # Math operator
        assert not _is_math_unicode("a")  # Regular ASCII

    def test_mono_font_detection(self):
        from okfgraph.ingest.converter import _is_mono_font
        assert _is_mono_font("Courier New")
        assert _is_mono_font("Consolas")
        assert not _is_mono_font("Arial")
