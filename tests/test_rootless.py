"""File-free mode: OKFRouter without bundle_root.

The thoughts-in/search-out loop needs no bundle: ingest, search, read,
traverse, doctor, and export are graph-side. File-side calls fail closed
with a clear ValueError (naming the missing root) instead of crashing on
None. Explicit paths keep working (single file → bare-stem ID, explicit
bundle dir → adopted by the primary detector).
"""

import pytest

from okfgraph.errors import UsageError
from okfgraph.router import OKFRouter


def _rootless(tmp_path, name="r.db"):
    return OKFRouter(db_path=str(tmp_path / name), device="cpu")


class TestFileFreeLoop:
    def test_thoughts_search_read_traverse_doctor_export(self, tmp_path):
        r = _rootless(tmp_path)
        try:
            cid = r.ingest_mgr.ingest("thoughts", thoughts="the falcon cannot hear the falconer",
                topic="Rootless Loop")["concept_id"]
            assert cid.startswith("thoughts/rootless_loop/")
            # read
            got = r.search_engine.get_by_id(cid)
            assert got is not None and "falconer" in got.body
            # search (vector-backed, proves encode path is unaffected)
            hits = r.search_engine.search_hybrid("falconer", limit=3)
            assert any(h["id"] == cid for h in hits)
            # traverse (graph-side)
            assert r.search_engine.get_by_id(cid) is not None
            # doctor (root status must not crash on a None primary)
            report = r.doctor_mgr.diagnose()
            assert report["score"] >= 0
            # export (explicit output dir, no bundle needed)
            exported = r.export_mgr.export_bundle(tmp_path / "out")
            assert cid in exported
            assert (tmp_path / "out" / (cid + ".md")).is_file()
        finally:
            r.close()


class TestFailClosed:
    def test_import_bundle_default_refused(self, tmp_path):
        r = _rootless(tmp_path)
        try:
            with pytest.raises(ValueError, match="bundle_root"):
                r.import_mgr.import_bundle()
        finally:
            r.close()

    def test_diff_no_side_refused(self, tmp_path):
        r = _rootless(tmp_path)
        try:
            # The op refuses any side-less diff without a bundle root.
            with pytest.raises(UsageError, match="bundle"):
                r.diff()
        finally:
            r.close()

    def test_detach_no_root_refused(self, tmp_path):
        r = _rootless(tmp_path)
        try:
            with pytest.raises(ValueError, match="bundle_root"):
                r.import_mgr.detach()
        finally:
            r.close()


class TestExplicitPaths:
    def test_single_file_import_bare_stem(self, tmp_path):
        md = tmp_path / "note.md"
        md.write_text("---\ntitle: Lone\ntype: note\n---\n\nBody.\n",
                      encoding="utf-8")
        r = _rootless(tmp_path)
        try:
            cid = r.import_mgr.import_from_okf(md)
            assert cid == "note"
            assert r.search_engine.get_by_id("note") is not None
        finally:
            r.close()

    def test_explicit_bundle_dir_adopted(self, tmp_path):
        bundle = tmp_path / "kb"
        bundle.mkdir()
        (bundle / "a.md").write_text(
            "---\ntitle: A\ntype: note\n---\n\nSee [B](b.md).\n",
            encoding="utf-8")
        (bundle / "b.md").write_text(
            "---\ntitle: B\ntype: note\n---\n\nBody B.\n", encoding="utf-8")
        r = _rootless(tmp_path)
        try:
            ids = r.import_mgr.import_bundle(bundle)
            assert sorted(ids) == ["a", "b"]
            # links resolve against the adopted tree
            assert r.list_broken_links() == []
            # second run is delta-clean (detector tracks the adopted tree)
            assert r.import_mgr.import_bundle(bundle) == []
            # drift diff against the adopted tree is identical
            assert r.diff(new=bundle)["identical"] is True
        finally:
            r.close()

    def test_ingest_md_explicit_file(self, tmp_path):
        md = tmp_path / "doc.md"
        md.write_text("---\ntitle: Doc\n---\n\nContent.\n", encoding="utf-8")
        r = _rootless(tmp_path)
        try:
            out = r.ingest_mgr.ingest("md", md_path=str(md))
            assert out["concept_id"] == "doc"
        finally:
            r.close()


class TestNamedRootsWithoutPrimary:
    def test_import_bundle_serves_named_roots(self, tmp_path):
        extra = tmp_path / "extra"
        extra.mkdir()
        (extra / "x.md").write_text(
            "---\ntitle: X\ntype: note\n---\n\nBody X.\n", encoding="utf-8")
        r = OKFRouter(
            db_path=str(tmp_path / "r.db"),
            roots={"side": str(extra)},
            device="cpu",
        )
        try:
            assert r.bundle_root is None
            ids = r.import_mgr.import_bundle()
            assert ids == ["@side/x"]
        finally:
            r.close()
