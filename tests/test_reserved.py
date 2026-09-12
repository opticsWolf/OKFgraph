"""Reserved filenames: generated index.md files are graph noise, not knowledge.

Bulk import, diff, and delta share the is_concept_file() predicate;
explicit single-file import of an index.md still works.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import ClassVar, Set

from okfgraph import OKFRouter
from okfgraph.components.import_ import RESERVED_FILENAMES, is_concept_file

NOTE = "---\ntype: note\ntitle: {title}\n---\n{body}\n"


def _write(bundle: Path, rel: str, title: str, body: str = "Body.") -> Path:
    fp = bundle / rel
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(NOTE.format(title=title, body=body), encoding="utf-8")
    return fp


class TestReservedPredicate:
    def test_predicate(self, tmp_path):
        md = tmp_path / "a.md"
        md.write_text("x")
        assert is_concept_file(md)
        idx = tmp_path / "Index.md"  # case-insensitive (macOS/Win checkouts)
        idx.write_text("x")
        assert not is_concept_file(idx)
        txt = tmp_path / "n.txt"
        txt.write_text("x")
        assert is_concept_file(txt)
        png = tmp_path / "i.png"
        png.write_text("x")
        assert not is_concept_file(png)
        assert "index.md" in RESERVED_FILENAMES

    def test_diff_agrees(self, tmp_path):
        from okfgraph.components.diff import state_of_dir

        _write(tmp_path, "a.md", "A")
        _write(tmp_path, "index.md", "Idx", "See [A](a.md).")
        state = state_of_dir(tmp_path)
        assert set(state.concepts) == {"a"}


class TestReservedLive:
    """Live router: bulk import skips index.md; export→re-import is stable."""

    router: ClassVar[OKFRouter]
    tmp_dir: ClassVar[str]

    @classmethod
    def setup_class(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.router = OKFRouter(
            db_path=os.path.join(cls.tmp_dir, "reserved.db"),
            bundle_root=cls.tmp_dir,
            device="cuda",
        )
        cls.router.__enter__()

    @classmethod
    def teardown_class(cls):
        try:
            cls.router.close()
        finally:
            shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def _ids(self) -> Set[str]:
        rows = self.router.conn.execute(
            "MATCH (c:Concept) RETURN c.id"
        ).rows_as_dict().get_all()
        return {r["c.id"] for r in rows}

    def test_bulk_import_skips_index(self):
        bundle = Path(self.tmp_dir) / "t1"
        _write(bundle, "t1_a.md", "T1 A")
        _write(bundle, "sub/t1_b.md", "T1 B")
        # Mimic export output: navigation files linking the real concepts.
        _write(bundle, "index.md", "T1 Index", "See [A](t1_a.md).")
        _write(bundle, "sub/index.md", "T1 Sub", "See [B](t1_b.md).")
        ids = set(self.router.import_mgr.import_bundle(bundle))
        assert ids == {"t1_a", "sub/t1_b"}
        assert not {i for i in self._ids() if i.split("/")[-1] == "index"}

    def test_explicit_single_import_still_works(self):
        bundle = Path(self.tmp_dir) / "t2"
        fp = _write(bundle, "index.md", "T2 Index", "Hand-written, explicitly imported.")
        cid = self.router.import_from_okf(fp)
        assert cid == "t2/index"
        assert "t2/index" in self._ids()

    def test_export_reimport_stable(self):
        # Import, export (writes index.md files), re-import elsewhere:
        # concept set identical, drift clean.
        src = Path(self.tmp_dir) / "t3"
        _write(src, "t3_a.md", "T3 A", "Links [B](sub/t3_b.md).")
        _write(src, "sub/t3_b.md", "T3 B")
        # NOTE: the class router also holds t1/t2 concepts, so export
        # contains more than t3 — assert containment, not equality.
        before = set(self.router.import_mgr.import_bundle(src))
        out = Path(self.tmp_dir) / "t3_out"
        self.router.export_mgr.export_bundle(out)
        # Export writes navigation files for non-root dirs (sub/index.md).
        assert (out / "sub" / "index.md").exists()

        fresh_dir = tempfile.mkdtemp()
        try:
            fresh = OKFRouter(
                db_path=os.path.join(fresh_dir, "fresh.db"),
                bundle_root=str(out),
                device="cuda",
            )
            fresh.__enter__()
            try:
                after = set(fresh.import_mgr.import_bundle(out))
                assert before <= after
                assert not {i for i in after if i.split("/")[-1] == "index"}
                assert fresh.diff_db_dir(out)["identical"] is True
            finally:
                fresh.close()
        finally:
            shutil.rmtree(fresh_dir, ignore_errors=True)
