"""Directory-level hash aggregation tests — skip entire subtrees when unchanged.

Validates that directory-level hashing correctly identifies unchanged directories
and skips all files within them, while still detecting changes at the file level
when a directory's contents change.
"""

import tempfile
import shutil
from pathlib import Path

import pytest
import yaml

from okfgraph.router import OKFRouter


def _write_okf(bundle_root: str, rel: str, title: str, body: str):
    """Write an OKF-style markdown file with frontmatter."""
    p = Path(bundle_root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = {"title": title, "path": rel}
    header = "---\n" + yaml.dump(meta, default_flow_style=False) + "---\n"
    with open(p, "w", encoding="utf-8") as f:
        f.write(header + body)
    return p


class TestDirectoryHash:
    """Directory-level hash aggregation — function-scoped fixtures for isolation."""

    def _make_router(self, tmp_dir):
        """Helper to create a fresh router."""
        return OKFRouter(
            db_path=str(Path(tmp_dir) / "test_dirhash.db"),
            bundle_root=tmp_dir,
            embedding_dim=512,
            chunk_size=50,
            chunk_overlap=10,
            enable_chunking=True,
            device="cpu",
        )

    def test_directory_hash_computed(self, tmp_path):
        """Directory hashes are stored after import."""
        router = self._make_router(str(tmp_path))
        _write_okf(str(tmp_path), "dir_a/a1.md", "Alpha1", "Body of alpha 1.")
        _write_okf(str(tmp_path), "dir_a/a2.md", "Alpha2", "Body of alpha 2.")
        _write_okf(str(tmp_path), "dir_b/b1.md", "Beta1", "Body of beta 1.")

        ids = router.import_mgr.import_bundle()
        assert len(ids) == 3

        dir_hashes = router.delta_mgr._load_directory_hashes()
        assert "dir_a" in dir_hashes
        assert "dir_b" in dir_hashes
        assert len(dir_hashes["dir_a"]["hash"]) == 64  # SHA-256 hex digest length
        assert len(dir_hashes["dir_b"]["hash"]) == 64
        router.close()

    def test_unchanged_directory_skipped(self, tmp_path):
        """Re-importing unchanged files skips all files via directory hash."""
        router = self._make_router(str(tmp_path))
        _write_okf(str(tmp_path), "a.md", "Alpha", "Body of alpha.")
        _write_okf(str(tmp_path), "b.md", "Beta", "Body of beta.")

        # First import
        ids = router.import_mgr.import_bundle()
        assert len(ids) == 2

        # Second import — no files changed
        ids = router.import_mgr.import_bundle()
        assert ids == [], f"Expected empty list for unchanged bundle, got {ids}"
        router.close()

    def test_modified_file_triggers_directory_import(self, tmp_path):
        """Modifying one file triggers re-import of all files in that directory."""
        router = self._make_router(str(tmp_path))
        _write_okf(str(tmp_path), "dir_a/a1.md", "Alpha1", "Body of alpha 1.")
        _write_okf(str(tmp_path), "dir_a/a2.md", "Alpha2", "Body of alpha 2.")

        # First import
        ids = router.import_mgr.import_bundle()
        assert len(ids) == 2

        # Modify a file
        _write_okf(str(tmp_path), "dir_a/a1.md", "Alpha1 Updated", "Modified body of alpha 1.")

        # Import — dir_a files should be re-imported
        ids = router.import_mgr.import_bundle()
        assert len(ids) == 2, f"Expected 2 imported IDs (dir_a files), got {len(ids)}"
        router.close()

    def test_new_file_triggers_directory_import(self, tmp_path):
        """Adding a new file to an existing directory triggers re-import."""
        router = self._make_router(str(tmp_path))
        _write_okf(str(tmp_path), "dir_a/a1.md", "Alpha1", "Body of alpha 1.")

        # First import
        ids = router.import_mgr.import_bundle()
        assert len(ids) == 1

        # Add a new file to dir_a
        _write_okf(str(tmp_path), "dir_a/a2.md", "Alpha2", "Body of alpha 2.")

        # Import — dir_a should be re-imported (now has 2 files)
        ids = router.import_mgr.import_bundle()
        assert len(ids) == 2, f"Expected 2 imported IDs (dir_a files), got {len(ids)}"
        router.close()

    def test_new_directory_triggers_import(self, tmp_path):
        """Adding a new directory triggers import of its files."""
        router = self._make_router(str(tmp_path))
        _write_okf(str(tmp_path), "dir_a/a1.md", "Alpha1", "Body of alpha 1.")

        # First import
        ids = router.import_mgr.import_bundle()
        assert len(ids) == 1

        # Add a new directory with a file
        _write_okf(str(tmp_path), "dir_b/b1.md", "Beta1", "Body of beta 1.")

        # Import — dir_b should be imported
        ids = router.import_mgr.import_bundle()
        assert len(ids) == 1, f"Expected 1 imported ID (dir_b file), got {len(ids)}"
        assert any("dir_b/b1" in cid for cid in ids)
        router.close()

    def test_deleted_directory_triggers_purge(self, tmp_path):
        """Deleting a directory triggers purge of its concepts."""
        router = self._make_router(str(tmp_path))
        _write_okf(str(tmp_path), "dir_a/a1.md", "Alpha1", "Body of alpha 1.")
        _write_okf(str(tmp_path), "dir_b/b1.md", "Beta1", "Body of beta 1.")

        # First import
        ids = router.import_mgr.import_bundle()
        assert len(ids) == 2

        # Delete dir_b
        shutil.rmtree(Path(tmp_path) / "dir_b")

        # Import with purge — dir_b concepts should be purged
        ids = router.import_mgr.import_bundle(purge_deleted=True)
        assert ids == [], f"Expected empty list (dir_b deleted, dir_a unchanged), got {ids}"

        # Verify dir_b concepts are purged (1 concept remains: dir_a/a1)
        concepts = router.conn.execute(
            "MATCH (c:Concept) RETURN count(c) AS cnt"
        ).rows_as_dict().get_all()
        assert concepts[0]["cnt"] == 1, f"Expected 1 concept remaining, got {concepts[0]['cnt']}"
        router.close()

    def test_directory_hash_is_deterministic(self, tmp_path):
        """Directory hash is deterministic for the same content."""
        r = OKFRouter(
            db_path=str(tmp_path / "test_deth.db"),
            bundle_root=str(tmp_path),
            device="cpu",
        )
        # Create a directory with files
        d = tmp_path / "test_dir"
        d.mkdir()
        (d / "file1.md").write_text("content1", encoding="utf-8")
        (d / "file2.md").write_text("content2", encoding="utf-8")

        h1 = r.delta_mgr._compute_directory_hash(d)
        h2 = r.delta_mgr._compute_directory_hash(d)
        r.close()

        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest length

    def test_directory_hash_changes_on_file_modification(self, tmp_path):
        """Directory hash changes when a file's content changes."""
        r = OKFRouter(
            db_path=str(tmp_path / "test_deth2.db"),
            bundle_root=str(tmp_path),
            device="cpu",
        )
        d = tmp_path / "test_dir_mod"
        d.mkdir(exist_ok=True)
        (d / "file1.md").write_text("content1", encoding="utf-8")

        h1 = r.delta_mgr._compute_directory_hash(d)

        # Modify file
        (d / "file1.md").write_text("modified content1", encoding="utf-8")
        h2 = r.delta_mgr._compute_directory_hash(d)

        r.close()

        assert h1 != h2, "Directory hash should change when file content changes"

    def test_directory_hash_changes_on_file_addition(self, tmp_path):
        """Directory hash changes when a new file is added."""
        r = OKFRouter(
            db_path=str(tmp_path / "test_deth3.db"),
            bundle_root=str(tmp_path),
            device="cpu",
        )
        d = tmp_path / "test_dir_add"
        d.mkdir(exist_ok=True)
        (d / "file1.md").write_text("content1", encoding="utf-8")

        h1 = r.delta_mgr._compute_directory_hash(d)

        # Add new file
        (d / "file2.md").write_text("content2", encoding="utf-8")
        h2 = r.delta_mgr._compute_directory_hash(d)

        r.close()

        assert h1 != h2, "Directory hash should change when a new file is added"

    def test_directory_hash_independent_of_file_order(self, tmp_path):
        """Directory hash is independent of file ordering (uses sorted paths)."""
        r = OKFRouter(
            db_path=str(tmp_path / "test_deth4.db"),
            bundle_root=str(tmp_path),
            device="cpu",
        )
        d = tmp_path / "test_dir_order"
        d.mkdir(exist_ok=True)
        # Create files in different order
        (d / "z_file.md").write_text("content_z", encoding="utf-8")
        (d / "a_file.md").write_text("content_a", encoding="utf-8")

        h1 = r.delta_mgr._compute_directory_hash(d)

        # Recreate directory with files in different order
        shutil.rmtree(d)
        d.mkdir(exist_ok=True)
        (d / "a_file.md").write_text("content_a", encoding="utf-8")
        (d / "z_file.md").write_text("content_z", encoding="utf-8")

        h2 = r.delta_mgr._compute_directory_hash(d)

        r.close()

        assert h1 == h2, "Directory hash should be independent of file creation order"
