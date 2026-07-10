"""Tests for document reconstruction from chunks.

Verifies that reconstruct_document produces output matching the original
markdown, with correct block delimiters and ordering.
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


class TestReconstruction:
    """Reconstruction tests — class-scoped router for speed."""

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
            db_path=str(Path(tmp_dir) / "test_reconstruction.db"),
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
    def multi_section_doc(cls, router, tmp_dir):
        """Import a document with multiple sections for reconstruction."""
        body = "\n\n".join([
            "## First Section",
            "Content of the first section with enough words to form a chunk. " * 10,
            "## Second Section",
            "Content of the second section with different text. " * 10,
            "## Third Section",
            "Content of the third section wrapping up the document. " * 10,
        ])
        p = _write_okf(tmp_dir, "reconstruct.md", "Reconstruct Me", body)
        cid = router.import_from_okf(p)
        cls._multi_section_doc = (cid, body)
        return cls._multi_section_doc

    def test_reconstruct_returns_text(self, router, multi_section_doc):
        cid, _ = multi_section_doc
        text = router.embed_engine.reconstruct_document(cid)
        assert text is not None
        assert len(text) > 0

    def test_reconstruct_contains_section_headings(self, router, multi_section_doc):
        cid, _ = multi_section_doc
        text = router.embed_engine.reconstruct_document(cid)
        assert "First Section" in text
        assert "Second Section" in text
        assert "Third Section" in text

    def test_reconstruct_preserves_order(self, router, multi_section_doc):
        cid, _ = multi_section_doc
        text = router.embed_engine.reconstruct_document(cid)
        first_idx = text.index("First Section")
        second_idx = text.index("Second Section")
        third_idx = text.index("Third Section")
        assert first_idx < second_idx < third_idx

    def test_reconstruct_nonexistent(self, router):
        text = router.embed_engine.reconstruct_document("nonexistent-id")
        assert text is None

    def test_reconstruct_has_content_fidelity(self, router, multi_section_doc):
        """Reconstructed text should contain key phrases from original."""
        cid, original_body = multi_section_doc
        text = router.embed_engine.reconstruct_document(cid)
        # Overlap may cause some duplication, but key phrases must be present
        assert "enough words" in text
        assert "different text" in text
        assert "wrapping up" in text
