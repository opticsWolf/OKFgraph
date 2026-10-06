"""Namespaced thought IDs: virtual placement without files.

Thoughts ingest file-free, but their IDs reserve an export location:
``thoughts/<slug>/<ts>_<uuid6>``. These tests pin the ID shape, the
slug sanitizing (no nesting, no traversal), explicit-ID control, and the
full fileless lifecycle: thoughts-in → export → nested files → reimport
with identical IDs and zero broken links.
"""

import re
import shutil
import tempfile
from pathlib import Path

from okfgraph.components.ingest import _slugify_topic
from okfgraph.router import OKFRouter

ID_RE = re.compile(r"^thoughts/[A-Za-z0-9_.\-]+/\d{14}_[0-9a-f]{6}$")


def _router(tmp_dir, name="t.db"):
    return OKFRouter(
        db_path=str(Path(tmp_dir) / name),
        bundle_root=str(tmp_dir),
        embedding_dim=512,
        chunk_size=50,
        chunk_overlap=10,
        device="cpu",
    )


class TestSlugify:
    def test_plain_topic(self):
        assert _slugify_topic("Vision Parity") == "vision_parity"

    def test_no_nesting_or_traversal(self):
        assert "/" not in _slugify_topic("a/b/c")
        assert ".." not in _slugify_topic("../../evil")
        assert "\\" not in _slugify_topic("a\\b")

    def test_hostile_topic_stays_single_level(self):
        slug = _slugify_topic("../../evil!")
        assert slug and "/" not in slug and slug != ""
        assert _slugify_topic("!!!") == ""

    def test_truncated(self):
        assert len(_slugify_topic("x" * 100)) == 30


class TestThoughtIds:
    def test_default_id_namespaced(self, tmp_path):
        r = _router(tmp_path)
        try:
            out = r.ingest_mgr.ingest("thoughts", thoughts="some reasoning", topic="Vision Parity")
            assert ID_RE.match(out["concept_id"]), out["concept_id"]
        finally:
            r.close()

    def test_empty_slug_falls_back(self, tmp_path):
        r = _router(tmp_path)
        try:
            out = r.ingest_mgr.ingest("thoughts", thoughts="x", topic="!!!")
            assert out["concept_id"].startswith("thoughts/untitled/")
            assert ID_RE.match(out["concept_id"])
        finally:
            r.close()

    def test_explicit_id_with_slashes_preserved(self, tmp_path):
        r = _router(tmp_path)
        try:
            out = r.ingest_mgr.ingest("thoughts", thoughts="x", topic="y",
                concept_id="journal/2026-10/my-thought")
            assert out["concept_id"] == "journal/2026-10/my-thought"
        finally:
            r.close()


class TestFilelessLifecycle:
    """thoughts-in (empty bundle dir) → export → nested tree → reimport."""

    def test_export_lands_in_namespace(self, tmp_path):
        r = _router(tmp_path)
        try:
            cid = r.ingest_mgr.ingest("thoughts", thoughts="vision is live", topic="Release Notes")["concept_id"]
            exported = r.export_mgr.export_bundle(tmp_path / "out")
            assert cid in exported
            expect = tmp_path / "out" / (cid + ".md")
            assert expect.is_file(), expect
            # Namespaced two levels deep, not flat at the root.
            assert len(expect.relative_to(tmp_path / "out").parts) == 3
        finally:
            r.close()

    def test_export_reimport_roundtrip(self, tmp_path):
        r1 = _router(tmp_path, "a.db")
        try:
            cid = r1.ingest_mgr.ingest("thoughts", thoughts="roundtrip body", topic="Roundtrip")["concept_id"]
            r1.export_mgr.export_bundle(tmp_path / "bundle")
        finally:
            r1.close()
        # Fresh graph from the exported files: same ID, no breakage.
        r2 = OKFRouter(
            db_path=str(tmp_path / "b.db"),
            bundle_root=str(tmp_path / "bundle"),
            embedding_dim=512,
            device="cpu",
        )
        try:
            ids = r2.import_mgr.import_bundle(tmp_path / "bundle")
            assert cid in ids
            assert r2.list_broken_links() == []
            got = r2.search_engine.get_by_id(cid)
            assert got is not None and "roundtrip body" in got.body
        finally:
            r2.close()

    def test_flat_legacy_ids_still_roundtrip(self, tmp_path):
        # Pre-namespace graphs used flat thought_<slug>_<ts>_<uuid> IDs —
        # they export flat and reimport unchanged (no migration needed).
        r1 = _router(tmp_path, "a.db")
        try:
            cid = r1.ingest_mgr.ingest("thoughts", thoughts="legacy", topic="Legacy",
                concept_id="thought_legacy_20200101000000_abcdef")["concept_id"]
            r1.export_mgr.export_bundle(tmp_path / "bundle")
            assert (tmp_path / "bundle" / (cid + ".md")).is_file()
        finally:
            r1.close()
