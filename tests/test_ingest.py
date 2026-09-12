"""Tests for IngestManager.ingest_pdf() (bobine engine).

Legacy okfgraph.ingest unit tests were removed with the engine
(PDF conversion now lives in the bobine wheel).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from okfgraph.components.converters import BobineConverter


def _never(**kwargs):
    """BobineConverter pinned to the pdf_oxide fast path (no ONNX)."""
    return BobineConverter(routing_mode="never", **kwargs)


class TestIngestPdfMethod:
    """Gap #5b — Router method ingest_pdf() for programmatic use."""

    @pytest.fixture(scope="function")
    def test_router(self, tmp_path):
        from okfgraph.router import OKFRouter
        db_path = str(tmp_path / "test.db")
        bundle_path = Path(__file__).parent / "fixtures" / "bundle"
        router = OKFRouter(
            db_path=db_path,
            bundle_root=str(bundle_path),
            embedding_dim=512,
            device="cpu",
            enable_chunking=False,
        )
        yield router
        router.close()

    def test_ingest_pdf_method_exists(self, test_router):
        """OKFRouter has an ingest_pdf() method."""
        assert hasattr(test_router.ingest_mgr, "ingest_pdf")
        assert callable(test_router.ingest_mgr.ingest_pdf)

    def test_ingest_pdf_returns_result_dict(self, test_router, tmp_path):
        """ingest_pdf returns a dict with expected keys."""
        # Create a minimal test PDF
        pdf_path = tmp_path / "test.pdf"
        pdf_content = b"""%PDF-1.0
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>stream
BT /F1 12 Tf 100 700 Td (Test) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Hyper/FirstChar 0/LastChar 255/Widths[333 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]>>endobj
xref
0 6
trailer<</Size 6/Root 1 0 R>>
startxref
0
%%EOF
"""
        pdf_path.write_bytes(pdf_content)

        result = test_router.ingest_mgr.ingest_pdf(
            pdf_path,
            auto_import=False,
            converter=_never(),
        )

        assert isinstance(result, dict)
        assert "md_path" in result
        assert "concept_ids" in result
        assert "image_dir" in result
        assert "page_count" in result
        assert result["concept_ids"] == []

    def test_ingest_pdf_output_only_writes_md(self, test_router, tmp_path):
        """ingest_pdf with auto_import=False writes markdown to disk."""
        pdf_path = tmp_path / "test.pdf"
        pdf_content = b"""%PDF-1.0
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>stream
BT /F1 12 Tf 100 700 Td (Test) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Hyper/FirstChar 0/LastChar 255/Widths[333 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]>>endobj
xref
0 6
trailer<</Size 6/Root 1 0 R>>
startxref
0
%%EOF
"""
        pdf_path.write_bytes(pdf_content)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = test_router.ingest_mgr.ingest_pdf(
            pdf_path,
            auto_import=False,
            output_dir=str(output_dir),
            converter=_never(),
        )

        assert Path(result["md_path"]).exists()
        md_content = Path(result["md_path"]).read_text(encoding="utf-8")
        assert len(md_content) > 0

    def test_ingest_pdf_returns_page_count(self, test_router, tmp_path):
        """ingest_pdf returns accurate page count."""
        pdf_path = tmp_path / "test.pdf"
        pdf_content = b"""%PDF-1.0
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>stream
BT /F1 12 Tf 100 700 Td (Page) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Hyper/FirstChar 0/LastChar 255/Widths[333 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]>>endobj
xref
0 6
trailer<</Size 6/Root 1 0 R>>
startxref
0
%%EOF
"""
        pdf_path.write_bytes(pdf_content)

        result = test_router.ingest_mgr.ingest_pdf(
            pdf_path,
            auto_import=False,
            converter=_never(),
        )

        assert result["page_count"] >= 1


MINIMAL_PDF = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Hello World from PDF) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000359 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
434
%%EOF
"""


class TestIngestPdfBobinePaths:
    """Bobine wiring: paths, callbacks, error order, work-dir hygiene."""

    @pytest.fixture(scope="function")
    def test_router(self, tmp_path):
        from okfgraph.router import OKFRouter
        db_path = str(tmp_path / "test.db")
        bundle_path = Path(__file__).parent / "fixtures" / "bundle"
        router = OKFRouter(
            db_path=db_path,
            bundle_root=str(bundle_path),
            embedding_dim=64,
            device="cpu",
            enable_chunking=False,
        )
        yield router
        router.close()

    @staticmethod
    def _pdf(tmp_path, name="doc.pdf"):
        pdf_path = tmp_path / name
        pdf_path.write_bytes(MINIMAL_PDF)
        return pdf_path

    def test_missing_bobine_raises_runtime_error(self, test_router, tmp_path, monkeypatch):
        """Without bobine, ingest_pdf fails fast with RuntimeError (not ImportError)."""
        import sys
        monkeypatch.setitem(sys.modules, "bobine", None)
        pdf_path = self._pdf(tmp_path)
        with pytest.raises(RuntimeError, match="bobine is required"):
            test_router.ingest_mgr.ingest_pdf(pdf_path, auto_import=False)

    def test_invalid_routing_mode_rejected_at_construction(self, tmp_path):
        """Bad routing_mode → ValueError before any work is done."""
        with pytest.raises(ValueError, match="routing_mode"):
            BobineConverter(routing_mode="bogus")
        assert list(tmp_path.glob("*.md")) == []

    def test_output_dir_is_respected(self, test_router, tmp_path):
        """auto_import=False + output_dir → <stem>.md + assets under output_dir."""
        pdf_path = self._pdf(tmp_path)
        out = tmp_path / "custom_out"
        result = test_router.ingest_mgr.ingest_pdf(
            pdf_path, auto_import=False, output_dir=out, converter=_never(),
        )
        assert Path(result["md_path"]) == out / "doc.md"
        assert Path(result["md_path"]).exists()
        assert Path(result["image_dir"]).parent == out
        assert Path(result["image_dir"]).exists()
        assert result["concept_ids"] == []
        assert result["page_count"] == 1

    def test_default_output_dir_is_pdf_parent(self, test_router, tmp_path):
        """No output_dir → markdown lands next to the PDF."""
        pdf_path = self._pdf(tmp_path)
        result = test_router.ingest_mgr.ingest_pdf(
            pdf_path, auto_import=False, converter=_never(),
        )
        assert Path(result["md_path"]).parent == tmp_path
        assert Path(result["md_path"]).exists()

    def test_on_page_callback_invoked(self, test_router, tmp_path):
        """Progress callback receives 0-based (page_index, page_total)."""
        pdf_path = self._pdf(tmp_path)
        calls = []
        test_router.ingest_mgr.ingest_pdf(
            pdf_path, auto_import=False, converter=_never(),
            on_page=lambda idx, total: calls.append((idx, total)),
        )
        assert calls == [(0, 1)]

    def test_extract_images_false_returns_full_dict(self, test_router, tmp_path):
        """extract_images=False still honors the result contract."""
        pdf_path = self._pdf(tmp_path)
        result = test_router.ingest_mgr.ingest_pdf(
            pdf_path, auto_import=False, converter=_never(extract_images=False),
        )
        assert set(result) == {"md_path", "concept_ids", "image_dir", "page_count"}
        assert result["page_count"] == 1

    def test_bundle_root_restored_on_import_failure(self, test_router, tmp_path, monkeypatch):
        """A failing import_bundle must not leak the temp work_dir as bundle_root."""
        pdf_path = self._pdf(tmp_path)
        before = test_router.ingest_mgr.bundle_root

        def _boom(*a, **k):
            raise RuntimeError("simulated import failure")

        monkeypatch.setattr(
            test_router.ingest_mgr.import_mgr, "import_bundle", _boom,
        )
        with pytest.raises(RuntimeError, match="simulated import failure"):
            test_router.ingest_mgr.ingest_pdf(
                pdf_path, auto_import=True, converter=_never(),
            )
        mgr = test_router.ingest_mgr
        assert mgr.bundle_root == before
        assert mgr.import_mgr.bundle_root == before
        assert mgr.delta_mgr.bundle_root == before

    def test_auto_import_returns_transient_md_path(self, test_router, tmp_path):
        """auto_import=True: content lands in the graph; md_path was temp-only."""
        pdf_path = self._pdf(tmp_path)
        result = test_router.ingest_mgr.ingest_pdf(
            pdf_path, auto_import=True, converter=_never(),
        )
        assert len(result["concept_ids"]) >= 1
        # Conversion ran in a TemporaryDirectory — the returned md_path
        # documents *what* was imported, not a durable file.
        assert not Path(result["md_path"]).exists()


class TestCustomConverter:
    """The DocumentConverter seam: a non-bobine pipeline plugs in cleanly."""

    @pytest.fixture(scope="function")
    def test_router(self, tmp_path):
        from okfgraph.router import OKFRouter
        db_path = str(tmp_path / "test.db")
        bundle_path = Path(__file__).parent / "fixtures" / "bundle"
        router = OKFRouter(
            db_path=db_path,
            bundle_root=str(bundle_path),
            embedding_dim=64,
            device="cpu",
            enable_chunking=False,
        )
        yield router
        router.close()

    @staticmethod
    def _stub_converter():
        from okfgraph.components.converters import ConvertedDocument

        class StubConverter:
            """Minimal third-party pipeline: writes its own markdown."""

            def __init__(self):
                self.calls = []

            def convert(self, pdf_path, output_dir, *, on_page=None):
                from pathlib import Path as _P
                self.calls.append((_P(pdf_path), _P(output_dir)))
                out = _P(output_dir)
                out.mkdir(parents=True, exist_ok=True)
                md = out / (_P(pdf_path).stem + ".md")
                md.write_text("# Stub pipeline\n\nConverted without bobine.\n", encoding="utf-8")
                assets = out / "_assets"
                assets.mkdir(exist_ok=True)
                if on_page:
                    on_page(0, 1)
                return ConvertedDocument(md_path=md, image_dir=assets, page_count=7)

        return StubConverter()

    def test_per_call_converter_override(self, test_router, tmp_path):
        stub = self._stub_converter()
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(MINIMAL_PDF)
        result = test_router.ingest_mgr.ingest_pdf(
            pdf_path, auto_import=False, converter=stub,
        )
        assert result["page_count"] == 7
        assert Path(result["md_path"]).read_text(encoding="utf-8").startswith("# Stub pipeline")
        assert len(stub.calls) == 1

    def test_per_call_converter_with_auto_import(self, test_router, tmp_path):
        stub = self._stub_converter()
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(MINIMAL_PDF)
        result = test_router.ingest_mgr.ingest_pdf(
            pdf_path, auto_import=True, converter=stub,
        )
        assert len(result["concept_ids"]) >= 1
        assert result["page_count"] == 7

    def test_router_level_converter_injection(self, tmp_path):
        from okfgraph.router import OKFRouter
        stub = self._stub_converter()
        bundle_path = Path(__file__).parent / "fixtures" / "bundle"
        router = OKFRouter(
            db_path=str(tmp_path / "test.db"),
            bundle_root=str(bundle_path),
            embedding_dim=64,
            device="cpu",
            enable_chunking=False,
            converter=stub,
        )
        try:
            pdf_path = tmp_path / "doc.pdf"
            pdf_path.write_bytes(MINIMAL_PDF)
            result = router.ingest_mgr.ingest_pdf(pdf_path, auto_import=False)
            assert result["page_count"] == 7
            assert len(stub.calls) == 1
        finally:
            router.close()

    def test_default_resolves_to_bobine(self, test_router):
        from okfgraph.components.converters import BobineConverter
        resolved = test_router.ingest_mgr._resolve_converter(None)
        assert isinstance(resolved, BobineConverter)
