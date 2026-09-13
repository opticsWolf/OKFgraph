"""Phase 0 (0.2.15) — import crash-consistency, deletion tombstones,
work-dir delta isolation, and wedge repair.

Uses real OKFRouter instances (same as the existing suite) with small
bundles; no vector values are asserted, so the ONNX encoder is just a
slow-ish upsert path, not a test dependency.
"""

import logging
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from okfgraph.router import OKFRouter


def _write_okf(bundle_root, rel, title, body, tags=None):
    """Write an OKF-style markdown file with frontmatter."""
    p = Path(bundle_root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = {"title": title, "path": rel}
    if tags:
        meta["tags"] = tags
    header = "---\n" + yaml.dump(meta, default_flow_style=False) + "---\n"
    p.write_text(header + body, encoding="utf-8")
    return p


def _concepts(conn):
    rows = conn.execute(
        "MATCH (c:Concept) RETURN c.id AS id"
    ).rows_as_dict().get_all()
    return {r["id"] for r in rows}


def _filehash_rows(conn):
    return conn.execute(
        "MATCH (f:FileHash) RETURN f.path AS p, f.hash AS h, f.concept_id AS c"
    ).rows_as_dict().get_all()


def _dirhash_keys(conn):
    rows = conn.execute(
        "MATCH (d:DirHash) RETURN d.path AS p"
    ).rows_as_dict().get_all()
    return {r["p"] for r in rows}


def _tombstones(conn):
    rows = conn.execute(
        "MATCH (d:DeletedPath) RETURN d.path AS p"
    ).rows_as_dict().get_all()
    return {r["p"] for r in rows}


@pytest.fixture()
def env():
    d = tempfile.mkdtemp()
    r = OKFRouter(
        db_path=str(Path(d) / "phase0.db"),
        bundle_root=d,
        embedding_dim=512,
        chunk_size=50,
        chunk_overlap=10,
        enable_chunking=True,
        device="cpu",
    )
    yield {"tmp": d, "router": r}
    r.close()
    shutil.rmtree(d, ignore_errors=True)


def _native(*parts):
    return str(Path(*parts))


class TestCrashConsistency:
    """Hashes must never describe state newer than the graph (0.1)."""

    def test_crash_before_commit_leaves_no_hashes(self, env):
        tmp, router = env["tmp"], env["router"]
        _write_okf(tmp, "a.md", "A", "content about apples and orchards and harvest season fruit trees.")
        _write_okf(tmp, "b.md", "B", "content about bananas and tropical plantations and yellow fruit bunches.")

        orig = router.import_mgr._batch_upsert_concepts

        def boom(parsed, embeddings):
            raise RuntimeError("simulated crash before commit")

        router.import_mgr._batch_upsert_concepts = boom
        try:
            with pytest.raises(RuntimeError, match="simulated crash"):
                router.import_mgr.import_bundle()
        finally:
            router.import_mgr._batch_upsert_concepts = orig

        # Rolled back: no concepts, and crucially no hashes either.
        assert _concepts(router.conn) == set()
        assert _filehash_rows(router.conn) == []
        assert _dirhash_keys(router.conn) == set()

        # Next run simply redoes the work and settles.
        assert sorted(router.import_mgr.import_bundle()) == ["a", "b"]
        assert _concepts(router.conn) == {"a", "b"}
        assert len(_filehash_rows(router.conn)) == 2
        assert router.import_mgr.import_bundle() == []

    def test_crash_after_commit_recovers_without_wedge(self, env):
        tmp, router = env["tmp"], env["router"]
        _write_okf(tmp, "a.md", "A", "content about apples and orchards and harvest season fruit trees.")
        _write_okf(tmp, "b.md", "B", "content about bananas and tropical plantations and yellow fruit bunches.")

        orig_store = router.delta_mgr._store_file_hashes
        state = {"fail": True}

        def flaky(hashes):
            if state["fail"] and hashes:
                raise RuntimeError("simulated crash after commit")
            return orig_store(hashes)

        router.delta_mgr._store_file_hashes = flaky
        try:
            with pytest.raises(RuntimeError, match="simulated crash"):
                router.import_mgr.import_bundle()
        finally:
            router.delta_mgr._store_file_hashes = orig_store

        # Good crash direction: concepts committed, hashes absent (never
        # the reverse — that would be the silent wedge).
        assert _concepts(router.conn) == {"a", "b"}
        assert _filehash_rows(router.conn) == []

        # Next run redoes the files (idempotent upserts) and settles.
        assert sorted(router.import_mgr.import_bundle()) == ["a", "b"]
        assert len(_filehash_rows(router.conn)) == 2
        assert router.import_mgr.import_bundle() == []


class TestParseFailureRetry:
    """Failed parses keep no hash state and retry next run (0.2)."""

    def test_parse_failure_retries(self, env):
        tmp, router = env["tmp"], env["router"]
        _write_okf(tmp, "dir_good/good.md", "Good", "solid content here about reliable systems and steady engineering practice.")
        bad = Path(tmp) / "dir_bad" / "bad.md"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text(
            "---\ntitle: Bad\ntags: [unclosed\n---\nbody text\n",
            encoding="utf-8",
        )

        ids = router.import_mgr.import_bundle()
        assert ids == ["dir_good/good"]
        assert _concepts(router.conn) == {"dir_good/good"}

        # Failed file left no hash state...
        assert all(
            r["p"] != _native("dir_bad", "bad.md")
            for r in _filehash_rows(router.conn)
        )
        # ...and its directory was dropped so the next run re-walks it.
        assert "dir_bad" not in _dirhash_keys(router.conn)

        bad.write_text("---\ntitle: Bad\n---\nnow valid body\n", encoding="utf-8")
        assert router.import_mgr.import_bundle() == ["dir_bad/bad"]
        assert _concepts(router.conn) == {"dir_good/good", "dir_bad/bad"}
        assert router.import_mgr.import_bundle() == []


class TestDeletionTombstones:
    """Per-file deletions persist across no-purge runs; purge is honest (0.3)."""

    def test_single_file_delete_tombstone_and_purge(self, env, caplog):
        tmp, router = env["tmp"], env["router"]
        _write_okf(tmp, "docs/a.md", "A", "alpha content words about the first letter and its ancient history.")
        _write_okf(tmp, "docs/b.md", "B", "beta content words about the second letter and its classical usage.")
        assert sorted(router.import_mgr.import_bundle()) == ["docs/a", "docs/b"]

        (Path(tmp) / "docs" / "b.md").unlink()
        gone = _native("docs", "b.md")

        # No-purge run: sibling re-imported, concept persists, tombstone kept.
        with caplog.at_level(logging.INFO):
            assert router.import_mgr.import_bundle(purge_deleted=False) == ["docs/a"]
        assert "docs/b" in _concepts(router.conn)
        assert gone in _tombstones(router.conn)

        # Purge run: nothing left to re-import (sibling was re-upserted
        # above), honest count, concept gone, tombstone consumed.
        with caplog.at_level(logging.INFO):
            assert router.import_mgr.import_bundle(purge_deleted=True) == []
        assert "purged 1 deleted concept" in caplog.text
        assert "docs/b" not in _concepts(router.conn)
        assert _tombstones(router.conn) == set()

        # Settled: quiet re-imports, no phantom deletions.
        assert router.import_mgr.import_bundle() == []
        assert router.import_mgr.import_bundle(purge_deleted=True) == []
        assert _tombstones(router.conn) == set()


class TestWorkDirIsolation:
    """Temp-dir imports must not touch the shared delta baseline (0.5)."""

    def test_workdir_import_leaves_no_delta_rows(self, env, tmp_path):
        tmp, router = env["tmp"], env["router"]
        _write_okf(tmp, "real/keep.md", "Keep", "real bundle content here about keeping things stable.")
        assert router.import_mgr.import_bundle() == ["real/keep"]
        before_dirs = _dirhash_keys(router.conn)
        before_files = {r["p"] for r in _filehash_rows(router.conn)}
        assert before_dirs and before_files

        # Work dir lives OUTSIDE the bundle root (like the real TemporaryDirectory).
        work = tmp_path / "work_pdf"
        work.mkdir()
        (work / "report.md").write_text(
            "---\ntitle: Report\n---\nconverted pdf body text about quarterly figures and tables.\n",
            encoding="utf-8",
        )
        ids = router.ingest_mgr._import_work_dir(
            work, 32, "text", False, Path(tmp) / "report.pdf"
        )
        assert ids == ["report"]
        assert "report" in _concepts(router.conn)

        # No temp keys leaked into the baseline...
        assert _dirhash_keys(router.conn) == before_dirs
        assert {r["p"] for r in _filehash_rows(router.conn)} == before_files
        # ...so the real bundle still settles quietly (top-level key intact).
        assert router.import_mgr.import_bundle() == []


class TestWedgeRepair:
    """doctor reports orphan hashes and --fix clears them."""

    def test_doctor_orphan_hash_reports_and_fixes(self, env):
        tmp, router = env["tmp"], env["router"]
        _write_okf(tmp, "ghost.md", "Ghost", "a file that will wedge the baseline with orphan hash rows.")
        # Craft the wedge directly: hashes committed, concepts never written.
        router.conn.execute(
            "MERGE (f:FileHash {path: $p}) SET f.hash = $h, f.concept_id = $c",
            {"p": "ghost.md", "h": "0" * 64, "c": "ghost"},
        )
        router.conn.execute(
            "MERGE (d:DirHash {path: $p}) SET d.hash = $h, d.files = $f",
            {"p": ".", "h": "0" * 64, "f": '["ghost.md"]'},
        )

        report = router.diagnose()
        orphan = [f for f in report["findings"] if f["rule"] == "orphan_hash"]
        assert len(orphan) == 1
        assert orphan[0]["path"] == "ghost.md"
        assert orphan[0]["severity"] == "error"

        fixed = router.doctor_fix()
        assert fixed["cleared_orphan_hashes"] == 1
        assert fixed["cleared_dir_hashes"] == 1
        assert _filehash_rows(router.conn) == []
        assert _dirhash_keys(router.conn) == set()

        report2 = router.diagnose()
        assert not [f for f in report2["findings"] if f["rule"] == "orphan_hash"]

        # Repair un-wedged the baseline: the file imports normally.
        assert router.import_mgr.import_bundle() == ["ghost"]
        assert "ghost" in _concepts(router.conn)
