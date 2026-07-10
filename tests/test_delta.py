"""Delta detection tests — file-level hash skip for import_bundle.

Validates that unchanged files are skipped on re-import, saving ONNX
encoding and DB upsert costs.
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


class TestDeltaDetection:
    """File-level hash skip in import_bundle — class-scoped router for speed."""

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
            db_path=str(Path(tmp_dir) / "test_delta.db"),
            bundle_root=tmp_dir,
            embedding_dim=512,
            chunk_size=50,
            chunk_overlap=10,
            enable_chunking=True,
            device="cpu",
        )
        cls._router = r
        yield cls._router
        cls._router.close()

    @pytest.fixture(scope="class")
    @classmethod
    def seeded_bundle(cls, tmp_dir, router):
        """Create three files in separate subdirs, import them, return file paths.

        Directory-level hashing groups files by parent directory. Placing each
        file in its own subdirectory ensures that modifying one file only changes
        that directory's hash, so only that file is re-imported.
        """
        files = {
            "dir_a/a.md": ("Alpha", "This is the first concept. It talks about alpha."),
            "dir_b/b.md": ("Beta", "This is the second concept. It talks about beta."),
            "dir_c/c.md": ("Gamma", "This is the third concept. It talks about gamma."),
        }
        paths = {}
        for rel, (title, body) in files.items():
            paths[rel] = _write_okf(tmp_dir, rel, title, body)

        # First import — all files are new
        ids = router.import_mgr.import_bundle()
        assert len(ids) == 3, f"Expected 3 imported IDs, got {len(ids)}"
        cls._file_paths = paths
        return paths

    # -- Tests -----------------------------------------------------------

    def test_first_import_imports_all(self, seeded_bundle, router):
        """All three files are new → all three are imported."""
        concepts = router.conn.execute(
            "MATCH (c:Concept) RETURN count(c) AS n"
        ).rows_as_dict().get_all()
        assert concepts[0]["n"] == 3

    def test_reimport_unchanged_returns_empty(self, router):
        """Re-import with no changes → empty list (nothing to do)."""
        ids = router.import_mgr.import_bundle()
        assert ids == [], f"Expected no changes, got {len(ids)} IDs"

    def test_deleted_file_detected(self, seeded_bundle, tmp_dir, router):
        """Deleting a file from disk → it appears in the deleted list."""
        # Delete dir_b/b.md from disk
        b_path = Path(tmp_dir) / "dir_b" / "b.md"
        b_path.unlink()

        changed, deleted = router.delta_mgr._changed_files(
            [fp for fp in Path(tmp_dir).rglob("*")
             if fp.is_file() and fp.suffix.lower() in (".md", ".txt", ".markdown")]
        )
        # Use native path separator for comparison
        expected_deleted = str((Path(tmp_dir) / "dir_b" / "b.md").relative_to(tmp_dir))
        assert expected_deleted in deleted, f"Expected {expected_deleted} in deleted, got {deleted}"

    def test_purge_nonexistent_concept_returns_false(self, router):
        """Purging a non-existent concept returns False."""
        result = router.purge_mgr._purge_concept("nonexistent")
        assert result is False

    def test_reimport_unchanged_preserves_count(self, router):
        """Re-import with no changes → concept count unchanged."""
        concepts = router.conn.execute(
            "MATCH (c:Concept) RETURN count(c) AS n"
        ).rows_as_dict().get_all()
        assert concepts[0]["n"] == 3

    def test_modify_one_file_imports_only_that_file(self, seeded_bundle, tmp_dir, router):
        """Edit one file → only that file is re-imported."""
        # Modify dir_b/b.md
        _write_okf(
            tmp_dir,
            "dir_b/b.md",
            "Beta Updated",
            "This is the updated second concept. It now talks about beta and delta.",
        )

        ids = router.import_mgr.import_bundle()
        assert len(ids) == 1, f"Expected 1 changed file, got {len(ids)}: {ids}"
        assert ids[0] == "dir_b/b", f"Expected 'dir_b/b', got {ids[0]}"

    def test_modified_file_has_new_embedding(self, seeded_bundle, tmp_dir, router):
        """Modified file gets a new embedding vector."""
        # Grab old embedding for dir_b/b
        old_row = router.conn.execute(
            "MATCH (c:Concept {id: 'dir_b/b'}) RETURN c.embedding AS emb"
        ).rows_as_dict().get_all()
        old_emb = old_row[0]["emb"]

        # Modify dir_b/b.md
        _write_okf(
            tmp_dir,
            "dir_b/b.md",
            "Beta v2",
            "Completely different content for beta now.",
        )
        router.import_mgr.import_bundle()

        # Grab new embedding for dir_b/b
        new_row = router.conn.execute(
            "MATCH (c:Concept {id: 'dir_b/b'}) RETURN c.embedding AS emb"
        ).rows_as_dict().get_all()
        new_emb = new_row[0]["emb"]

        assert old_emb != new_emb, "Embedding should change when content changes"

    def test_unmodified_files_keep_same_embedding(self, seeded_bundle, tmp_dir, router):
        """Files that weren't modified keep their original embeddings."""
        # Grab old embeddings for dir_a/a and dir_c/c
        old_embs = {}
        for cid in ("dir_a/a", "dir_c/c"):
            row = router.conn.execute(
                "MATCH (c:Concept {id: $id}) RETURN c.embedding AS emb",
                {"id": cid},
            ).rows_as_dict().get_all()
            old_embs[cid] = row[0]["emb"]

        # Modify only dir_b/b.md
        _write_okf(
            tmp_dir,
            "dir_b/b.md",
            "Beta v3",
            "Yet another change to beta.",
        )
        router.import_mgr.import_bundle()

        # Verify dir_a/a and dir_c/c embeddings are unchanged
        for cid in ("dir_a/a", "dir_c/c"):
            row = router.conn.execute(
                "MATCH (c:Concept {id: $id}) RETURN c.embedding AS emb",
                {"id": cid},
            ).rows_as_dict().get_all()
            assert row[0]["emb"] == old_embs[cid], f"Embedding for {cid} should not change"

    def test_new_file_is_imported(self, seeded_bundle, tmp_dir, router):
        """Adding a new file → it's detected as changed and imported."""
        _write_okf(
            tmp_dir,
            "dir_d/d.md",
            "Delta",
            "This is a brand new concept about delta.",
        )

        ids = router.import_mgr.import_bundle()
        assert len(ids) == 1
        assert ids[0] == "dir_d/d"

        # Verify it's in the DB
        concepts = router.conn.execute(
            "MATCH (c:Concept) RETURN count(c) AS n"
        ).rows_as_dict().get_all()
        assert concepts[0]["n"] == 4

    def test_file_hashes_stored_in_filehash(self, router):
        """File hash mapping is persisted in the FileHash table."""
        rows = router.conn.execute(
            "MATCH (f:FileHash) RETURN f.path AS p, f.hash AS h, f.concept_id AS c"
        ).rows_as_dict().get_all()
        assert len(rows) > 0, "FileHash table should have entries"
        for r in rows:
            assert len(r["h"]) == 64  # SHA-256 hex digest
            assert r["c"] is not None, "concept_id should be stored"
            assert len(r["c"]) > 0, "concept_id should not be empty"

    def test_file_hash_concept_id_mapping(self, router):
        """concept_id mapping is stored and loadable."""
        cid_map = router.delta_mgr._load_file_hash_concept_ids()
        assert len(cid_map) > 0, "Should have concept_id mappings"
        # Verify mapping: path without extension → concept_id (forward slashes)
        for path, cid in cid_map.items():
            expected = str(Path(path).with_suffix("")).replace("\\", "/")
            assert cid == expected, f"Expected {expected}, got {cid}"

    def test_combined_hash_sentinel_stored(self, router):
        """Write epoch is bumped after import (proves data was written)."""
        rows = router.conn.execute(
            "MATCH (m:Meta {key: 'write_epoch'}) RETURN m.value AS v"
        ).rows_as_dict().get_all()
        assert len(rows) == 1
        assert rows[0]["v"] > 0

    def test_file_hash_is_deterministic(self, tmp_dir):
        """_file_hash returns the same value for the same content."""
        p = Path(tmp_dir) / "determinism_test.md"
        p.write_text("Hello, world!", encoding="utf-8")
        r = OKFRouter(
            db_path=str(Path(tmp_dir) / "test_hash.db"),
            bundle_root=tmp_dir,
            device="cpu",
        )
        h1 = r.delta_mgr._file_hash(p)
        h2 = r.delta_mgr._file_hash(p)
        r.close()
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest


class TestPurgeDeleted:
    """Purge tests — own seeded bundle to control delete timing."""

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
            db_path=str(Path(tmp_dir) / "test_purge.db"),
            bundle_root=tmp_dir,
            embedding_dim=512,
            chunk_size=50,
            chunk_overlap=10,
            enable_chunking=True,
            device="cpu",
        )
        cls._router = r
        yield cls._router
        cls._router.close()

    @pytest.fixture(scope="class")
    @classmethod
    def seeded_bundle(cls, tmp_dir, router):
        """Create three files in separate subdirs, import them.

        Directory-level hashing groups files by parent directory. Placing each
        file in its own subdirectory ensures that modifying one file only changes
        that directory's hash, so only that file is re-imported.
        """
        files = {
            "dir_x/x.md": ("X-Ray", "Content about x-ray imaging."),
            "dir_y/y.md": ("Yield", "Content about yield strength."),
            "dir_z/z.md": ("Zenith", "Content about zenith angle."),
        }
        for rel, (title, body) in files.items():
            _write_okf(tmp_dir, rel, title, body)
        ids = router.import_mgr.import_bundle()
        assert len(ids) == 3

    def test_purge_end_to_end_via_import_bundle(self, seeded_bundle, tmp_dir, router):
        """Full flow: add file, import, delete, purge via import_bundle."""
        # Add a new file in its own directory
        _write_okf(tmp_dir, "dir_w/w.md", "Wave", "Content about wave functions.")
        ids = router.import_mgr.import_bundle()
        assert len(ids) == 1
        assert ids[0] == "dir_w/w"

        # Delete it from disk
        (Path(tmp_dir) / "dir_w" / "w.md").unlink()

        # Purge via import_bundle
        ids = router.import_mgr.import_bundle(purge_deleted=True)
        assert ids == []  # no changed files

        # Verify w is gone
        row = router.conn.execute(
            "MATCH (c:Concept {id: 'dir_w/w'}) RETURN count(c) AS n"
        ).rows_as_dict().get_all()
        assert row[0]["n"] == 0, "w should be purged"

    def test_without_purge_concept_persists(self, seeded_bundle, tmp_dir, router):
        """Delete dir_y/y.md, re-import without purge → y still exists."""
        (Path(tmp_dir) / "dir_y" / "y.md").unlink()
        ids = router.import_mgr.import_bundle(purge_deleted=False)
        assert ids == []  # no changed files

        row = router.conn.execute(
            "MATCH (c:Concept {id: 'dir_y/y'}) RETURN count(c) AS n"
        ).rows_as_dict().get_all()
        assert row[0]["n"] == 1, "y should persist without purge"

    def test_with_purge_y_removed(self, seeded_bundle, tmp_dir, router):
        """dir_y/y.md already deleted. Purge via _purge_concept directly."""
        result = router.purge_mgr._purge_concept("dir_y/y")
        assert result is True

        row = router.conn.execute(
            "MATCH (c:Concept {id: 'dir_y/y'}) RETURN count(c) AS n"
        ).rows_as_dict().get_all()
        assert row[0]["n"] == 0, "y should be purged"

    def test_purge_removes_chunks(self, seeded_bundle, tmp_dir, router):
        """Purge removes chunks linked to the deleted concept."""
        chunks = router.conn.execute(
            "MATCH (ch:Chunk {parent_doc_id: 'dir_y/y'}) RETURN count(ch) AS n"
        ).rows_as_dict().get_all()
        assert chunks[0]["n"] == 0, "Chunks for y should be gone"

    def test_purge_removes_filehash_entry(self, seeded_bundle, tmp_dir, router):
        """Purge removes the FileHash entry for the deleted concept."""
        fh = router.conn.execute(
            "MATCH (f:FileHash {concept_id: 'dir_y/y'}) RETURN count(f) AS n"
        ).rows_as_dict().get_all()
        assert fh[0]["n"] == 0, "FileHash for y should be purged"

    def test_purge_preserves_other_concepts(self, seeded_bundle, tmp_dir, router):
        """Purging y leaves x and z intact."""
        concepts = router.conn.execute(
            "MATCH (c:Concept) RETURN count(c) AS n"
        ).rows_as_dict().get_all()
        assert concepts[0]["n"] == 2, "x and z should remain"


class TestChangedFilesEmptyBundle:
    """Edge case: _changed_files with no source files."""

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
            db_path=str(Path(tmp_dir) / "test_empty.db"),
            bundle_root=tmp_dir,
            device="cpu",
        )
        cls._router = r
        yield cls._router
        cls._router.close()

    def test_no_files_returns_empty_tuple(self, router):
        changed, deleted = router.delta_mgr._changed_files([])
        assert changed == []
        assert deleted == []

    def test_purge_nonexistent_returns_false(self, router):
        """Purging a non-existent concept returns False."""
        result = router.purge_mgr._purge_concept("does_not_exist")
        assert result is False
