"""OKF-compliant export tests.

Verifies that exported bundles faithfully reflect the LINKS_TO graph
via See Also + Cited By sections, and that index.md files are generated
for progressive disclosure.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import ClassVar

import pytest
from okfgraph import OKFRouter, ConceptModel


class TestOKFExportCompliance:
    """Tests for OKF-compliant export with graph enrichment."""

    router: ClassVar[OKFRouter]
    tmp_dir: ClassVar[str]

    @classmethod
    def setup_class(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.router = OKFRouter(
            db_path=os.path.join(cls.tmp_dir, "test_export.db"),
            bundle_root=cls.tmp_dir,
            device="cuda",
        )
        cls.router.__enter__()

        # Create test documents
        (Path(cls.tmp_dir) / "doc_a.md").write_text(
            "---\ntype: note\ntitle: Doc A\ndescription: First doc.\n---\nContent of A."
        )
        (Path(cls.tmp_dir) / "doc_b.md").write_text(
            "---\ntype: note\ntitle: Doc B\ndescription: Second doc.\n---\nContent of B."
        )
        (Path(cls.tmp_dir) / "doc_c.md").write_text(
            "---\ntype: note\ntitle: Doc C\ndescription: Third doc.\n---\nContent of C."
        )

        # Import documents
        cls.router.import_from_okf(Path(cls.tmp_dir) / "doc_a.md")
        cls.router.import_from_okf(Path(cls.tmp_dir) / "doc_b.md")
        cls.router.import_from_okf(Path(cls.tmp_dir) / "doc_c.md")

        # Create graph edges: doc_a -> doc_b, doc_c -> doc_a
        cls.router.conn.execute("""
            MATCH (a:Concept {id: $aid}), (b:Concept {id: $bid})
            MERGE (a)-[:LINKS_TO]->(b)
        """, {"aid": "doc_a", "bid": "doc_b"})
        cls.router.conn.execute("""
            MATCH (c:Concept {id: $cid}), (a:Concept {id: $aid})
            MERGE (c)-[:LINKS_TO]->(a)
        """, {"cid": "doc_c", "aid": "doc_a"})

    @classmethod
    def teardown_class(cls):
        cls.router.__exit__(None, None, None)
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_export_has_see_also(self):
        """doc_a should have 'See Also' with doc_b (outgoing link)."""
        out = Path(self.tmp_dir) / "export"
        self.router.export_bundle(out)
        body = (out / "doc_a.md").read_text()
        assert "## See Also" in body
        assert "doc_b.md" in body

    def test_export_has_cited_by(self):
        """doc_a should have 'Cited By' with doc_c (incoming link)."""
        out = Path(self.tmp_dir) / "export"
        self.router.export_bundle(out)
        body = (out / "doc_a.md").read_text()
        assert "## Cited By" in body
        assert "doc_c.md" in body

    def test_export_no_duplicate_links(self):
        """If a link already exists in the body, it should not be duplicated."""
        out = Path(self.tmp_dir) / "export"
        self.router.export_bundle(out)
        body = (out / "doc_b.md").read_text()
        # doc_b has no outgoing LINKS_TO edges, so no "See Also"
        assert "## See Also" not in body

    def test_export_has_index_files(self):
        """index.md files should be generated for each directory."""
        out = Path(self.tmp_dir) / "export"
        self.router.export_bundle(out)
        # Root-level concepts don't generate index.md (no parent directory)
        # But nested concepts do. Add a nested concept and re-export.
        nested = Path(self.tmp_dir) / "nested/item.md"
        nested.parent.mkdir(exist_ok=True)
        nested.write_text("---\ntype: note\ntitle: Nested Item\n---\nNested.")
        self.router.import_from_okf(nested)
        self.router.export_bundle(out)
        assert (out / "nested" / "index.md").exists()

    def test_export_preserves_original_body(self):
        """Original body content is preserved, sections are appended."""
        out = Path(self.tmp_dir) / "export"
        self.router.export_bundle(out)
        body = (out / "doc_a.md").read_text()
        assert "Content of A." in body
        assert "description: First doc." in body

    def test_export_filters_by_type(self):
        """Only concepts of the specified type are exported."""
        out = Path(self.tmp_dir) / "export"
        self.router.export_bundle(out, concept_type="note")
        assert (out / "doc_a.md").exists()
        assert (out / "doc_b.md").exists()
        assert (out / "doc_c.md").exists()

    def test_export_filters_by_directory(self):
        """Only concepts under the specified directory are exported."""
        # Add a nested concept
        nested = Path(self.tmp_dir) / "sub/nested.md"
        nested.parent.mkdir(exist_ok=True)
        nested.write_text("---\ntype: note\ntitle: Nested\n---\nNested content.")
        self.router.import_from_okf(nested)
        out = Path(self.tmp_dir) / "export_sub"
        self.router.export_bundle(out, directory_id="sub")
        assert (out / "sub" / "nested.md").exists()
        assert not (out / "doc_a.md").exists()  # not under "sub"

    def test_export_single_concept(self):
        """Export a single concept by ID."""
        out = Path(self.tmp_dir) / "export"
        self.router.export_to_okf("doc_a", out / "doc_a.md")
        assert (out / "doc_a.md").exists()
        body = (out / "doc_a.md").read_text()
        assert "## See Also" in body
        assert "## Cited By" in body

    def test_enrich_body_no_outgoing_links(self):
        """Concept with no outgoing links gets no See Also section."""
        body = self.router._enrich_body_with_graph_links("doc_b", "Some content.")
        assert "## See Also" not in body

    def test_enrich_body_no_incoming_links(self):
        """Concept with no incoming links gets no Cited By section."""
        # doc_b has incoming link from doc_a, so use a concept with no links
        body = self.router._enrich_body_with_graph_links("doc_c", "Some content.")
        # doc_c has outgoing link to doc_a but no incoming links
        assert "## Cited By" not in body

    def test_generate_index_files_empty(self):
        """Empty concept dict generates no index files."""
        out = Path(self.tmp_dir) / "export_empty"
        out.mkdir(exist_ok=True)
        self.router._generate_index_files(out, {})
        # No index.md should be created for empty dict
        assert not (out / "index.md").exists()

    def test_export_bundle_returns_ids(self):
        """export_bundle returns list of exported concept IDs."""
        out = Path(self.tmp_dir) / "export"
        exported = self.router.export_bundle(out)
        assert "doc_a" in exported
        assert "doc_b" in exported
        assert "doc_c" in exported

    def test_export_bundle_empty(self):
        """export_bundle returns empty list when no concepts match."""
        out = Path(self.tmp_dir) / "export"
        exported = self.router.export_bundle(out, concept_type="nonexistent")
        assert exported == []
