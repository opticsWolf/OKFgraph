"""Phase 1 (0.2.16) — `okf detach`: schema v7, fidelity verify, refusal
semantics, re-attach, and continued reads after source deletion.

Uses real OKFRouter instances (same as the existing suite) with small
bundles; bodies are long enough to stay clear of the short-doc chunk
quirk (see test_phase0).
"""

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


def _hash_counts(conn):
    out = {}
    for table in ("FileHash", "DirHash", "DeletedPath"):
        rows = conn.execute(
            f"MATCH (n:{table}) RETURN count(n) AS n"
        ).rows_as_dict().get_all()
        out[table] = rows[0]["n"] if rows else 0
    return out


def _export_tree(router, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    router.export_mgr.export_bundle(output_dir=out)
    return {
        str(p.relative_to(out)): p.read_bytes()
        for p in sorted(out.rglob("*"))
        if p.is_file()
    }


@pytest.fixture()
def env():
    d = tempfile.mkdtemp()
    r = OKFRouter(
        db_path=str(Path(d) / "detach.db"),
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


def _seed(router, tmp):
    _write_okf(tmp, "a.md", "Alpha",
               "Alpha content about the first letter and its long ancient history.")
    _write_okf(tmp, "b.md", "Beta",
               "Beta content about the second letter and its classical Greek usage.")
    return sorted(router.import_mgr.import_bundle())


class TestSchemaV7:
    def test_fresh_db_is_v7_with_sourceroot(self, env):
        router = env["router"]
        assert router.schema_mgr._get_meta("schema_version") == 8
        rows = router.conn.execute(
            "CALL TABLE_INFO('SourceRoot') RETURN *"
        ).rows_as_dict().get_all()
        assert rows, "SourceRoot table should exist"
        assert not router.import_mgr.delta_mgr.is_detached()

    def test_v6_db_migrates_to_v7(self, tmp_path):
        import ladybug as lb

        db_path = str(tmp_path / "v6.db")
        conn = lb.Connection(lb.Database(db_path))
        conn.execute(
            "CREATE NODE TABLE Meta (key STRING PRIMARY KEY, value INT64)"
        )
        conn.execute(
            "CREATE (m:Meta {key: 'schema_version', value: 6})"
        )
        del conn
        r = OKFRouter(db_path=db_path, bundle_root=str(tmp_path),
                       embedding_dim=512, enable_chunking=False, device="cpu")
        try:
            assert r.schema_mgr._get_meta("schema_version") == 8
            rows = r.conn.execute(
                "CALL TABLE_INFO('SourceRoot') RETURN *"
            ).rows_as_dict().get_all()
            assert rows, "migration should create SourceRoot"
            # Migrated DB is fully usable.
            (tmp_path / "m.md").write_text(
                "# M\n\nMigrated database body text here.\n", encoding="utf-8")
            assert r.import_mgr.import_bundle(tmp_path) == ["m"]
        finally:
            r.close()

    def test_v7_db_gains_dir_backfill(self, tmp_path):
        """A 0.2.16–0.3.0 DB (version 7, mirror rows without `dir`)
        migrates to v8 with the legacy parent math backfilled."""
        import ladybug as lb

        db_path = str(tmp_path / "v7.db")
        conn = lb.Connection(lb.Database(db_path))
        conn.execute(
            "CREATE NODE TABLE Meta (key STRING PRIMARY KEY, value INT64)"
        )
        conn.execute(
            "CREATE (m:Meta {key: 'schema_version', value: 7})"
        )
        conn.execute("""
            CREATE NODE TABLE FileHash (
                path STRING PRIMARY KEY, hash STRING, concept_id STRING
            )
        """)
        conn.execute(
            "CREATE (f:FileHash {path: 'sub/g.md', hash: 'h', "
            "concept_id: 'sub/g'})"
        )
        conn.execute("""
            CREATE NODE TABLE DeletedPath (
                path STRING PRIMARY KEY, detected_at INT64
            )
        """)
        conn.execute(
            "CREATE (d:DeletedPath {path: 'old.md', detected_at: 1})"
        )
        del conn
        r = OKFRouter(db_path=db_path, bundle_root=str(tmp_path),
                       embedding_dim=512, enable_chunking=False, device="cpu")
        try:
            assert r.schema_mgr._get_meta("schema_version") == 8
            fh = r.conn.execute(
                "MATCH (f:FileHash) RETURN f.path AS p, f.dir AS d"
            ).rows_as_dict().get_all()
            assert [(x["p"], x["d"]) for x in fh] == [("sub/g.md", "sub")]
            dp = r.conn.execute(
                "MATCH (d:DeletedPath) RETURN d.path AS p, d.dir AS d"
            ).rows_as_dict().get_all()
            assert [(x["p"], x["d"]) for x in dp] == [("old.md", ".")]
        finally:
            r.close()


class TestDetachClean:
    def test_detach_clean_graph(self, env, tmp_path):
        tmp, router = env["tmp"], env["router"]
        assert _seed(router, tmp) == ["a", "b"]

        report = router.import_mgr.detach()
        assert report["verified"] is True
        assert report["file_count"] == 2
        assert report["mismatched"] == []
        assert report["untracked"] == []
        assert report["non_source_files"] == []
        assert report["provenance"]["path"] == str(Path(tmp).resolve())
        assert report["provenance"]["file_count"] == 2

        assert router.import_mgr.delta_mgr.is_detached()
        assert router.schema_mgr._get_meta("detached") == 1
        roots = router.conn.execute(
            "MATCH (s:SourceRoot) RETURN s.alias AS a, s.path AS p"
        ).rows_as_dict().get_all()
        assert len(roots) == 1 and roots[0]["p"] == str(Path(tmp).resolve())
        # Mirror baseline dropped; graph content untouched.
        assert _hash_counts(router.conn) == {
            "FileHash": 0, "DirHash": 0, "DeletedPath": 0}
        assert _concepts(router.conn) == {"a", "b"}

    def test_detach_twice_refuses(self, env):
        tmp, router = env["tmp"], env["router"]
        _seed(router, tmp)
        router.import_mgr.detach()
        with pytest.raises(RuntimeError, match="already detached"):
            router.import_mgr.detach()

    def test_reads_unaffected_after_source_delete(self, env, tmp_path):
        tmp, router = env["tmp"], env["router"]
        _write_okf(tmp, "a.md", "Alpha",
                   "Alpha content about the first letter. See [b](b.md) for more.")
        _write_okf(tmp, "b.md", "Beta",
                   "Beta content about the second letter. Back to [[Alpha]] again.")
        assert len(router.import_mgr.import_bundle()) == 2

        before_tree = _export_tree(router, tmp_path / "before")
        before_search = [r["id"] for r in router.search_hybrid("Alpha", limit=5)]
        assert before_search and before_search[0] == "a"

        router.import_mgr.detach()
        assert router.diagnose()["detached"] is not None
        # Delete every source file; the DB is the artifact now.
        for p in Path(tmp).rglob("*.md"):
            p.unlink()

        after_tree = _export_tree(router, tmp_path / "after")
        assert after_tree == before_tree
        after_search = [r["id"] for r in router.search_hybrid("Alpha", limit=5)]
        assert after_search == before_search
        assert _concepts(router.conn) == {"a", "b"}


class TestDetachVerify:
    def _messy_bundle(self, tmp, router):
        _write_okf(tmp, "good.md", "Good",
                   "Good content that stays exactly as imported here.")
        _write_okf(tmp, "dirty.md", "Dirty",
                   "Dirty content in its original imported wording here.")
        assert len(router.import_mgr.import_bundle()) == 2
        # Post-import changes: edit one, add an untracked one + originals.
        _write_okf(tmp, "dirty.md", "Dirty",
                   "Dirty content REWRITTEN after import, diverging now.")
        _write_okf(tmp, "new.md", "New",
                   "New content never imported into the graph at all.")
        (Path(tmp) / "data.pdf").write_bytes(b"%PDF-1.4 fake")
        (Path(tmp) / "img.png").write_bytes(b"\x89PNG fake")

    def test_verify_lists_every_category(self, env):
        tmp, router = env["tmp"], env["router"]
        self._messy_bundle(tmp, router)
        with pytest.raises(RuntimeError, match="detach refused") as exc:
            router.import_mgr.detach()
        msg = str(exc.value)
        assert "1 mismatched" in msg
        assert "1 untracked" in msg
        assert "2 source-only" in msg
        # Nothing was recorded; the mirror is intact.
        assert not router.import_mgr.delta_mgr.is_detached()
        assert _concepts(router.conn) == {"dirty", "good"}
        rows = router.conn.execute(
            "MATCH (c:Concept {id: 'new'}) RETURN count(c) AS n"
        ).rows_as_dict().get_all()
        assert rows[0]["n"] == 0

    def test_force_acknowledges_and_detaches(self, env):
        tmp, router = env["tmp"], env["router"]
        self._messy_bundle(tmp, router)
        report = router.import_mgr.detach(force=True)
        assert report["verified"] is True
        assert len(report["mismatched"]) == 1
        assert report["mismatched"][0]["path"] == _native("dirty.md")
        assert "body" in report["mismatched"][0]["fields"]
        assert len(report["untracked"]) == 1
        assert len(report["non_source_files"]) == 2
        assert router.import_mgr.delta_mgr.is_detached()

    def test_verify_missing_bundle_refuses(self, env):
        tmp, router = env["tmp"], env["router"]
        _seed(router, tmp)
        with pytest.raises(RuntimeError, match="--no-verify"):
            router.import_mgr.detach(bundle_path=Path(tmp) / "gone")

    def test_no_verify_missing_bundle_records_path(self, env):
        tmp, router = env["tmp"], env["router"]
        _seed(router, tmp)
        gone = Path(tmp) / "gone"
        report = router.import_mgr.detach(bundle_path=gone, verify=False)
        assert report["verified"] is False
        assert report["provenance"]["path"] == str(gone.resolve())
        assert router.import_mgr.delta_mgr.is_detached()


class TestRefusalAndReattach:
    def test_import_refuses_when_detached(self, env):
        tmp, router = env["tmp"], env["router"]
        _seed(router, tmp)
        router.import_mgr.detach()
        with pytest.raises(RuntimeError, match="detached"):
            router.import_mgr.import_bundle()
        with pytest.raises(RuntimeError, match="detached"):
            router.import_mgr.import_bundle(purge_deleted=True)

    def test_rootless_writes_refuse_without_force(self, env, tmp_path):
        tmp, router = env["tmp"], env["router"]
        _seed(router, tmp)
        router.import_mgr.detach()
        outside = tmp_path / "outside.md"
        outside.write_text("---\ntitle: Outside\n---\nOutside bundle body text here.\n",
                           encoding="utf-8")
        with pytest.raises(RuntimeError, match="detached"):
            router.ingest_mgr.ingest_md(md_path=outside)
        with pytest.raises(RuntimeError, match="detached"):
            router.ingest_mgr.ingest_thoughts("some reasoning", topic="t")
        with pytest.raises(RuntimeError, match="detached"):
            router.import_mgr.import_from_okf(outside)

    def test_rootless_force_does_not_reattach(self, env, tmp_path):
        tmp, router = env["tmp"], env["router"]
        _seed(router, tmp)
        router.import_mgr.detach()
        outside = tmp_path / "outside.md"
        outside.write_text("---\ntitle: Outside\n---\nOutside bundle body text here.\n",
                           encoding="utf-8")
        result = router.ingest_mgr.ingest_md(md_path=outside, force=True)
        assert result["concept_id"] == "outside"
        # Addressed write allowed; mirror still detached.
        assert router.import_mgr.delta_mgr.is_detached()

    def test_force_reattach_matching_root(self, env):
        tmp, router = env["tmp"], env["router"]
        _seed(router, tmp)
        router.import_mgr.detach()
        ids = router.import_mgr.import_bundle(force=True)
        assert sorted(ids) == ["a", "b"]
        assert not router.import_mgr.delta_mgr.is_detached()
        assert router.schema_mgr._get_meta("detached", 0) == 0
        assert _hash_counts(router.conn)["FileHash"] == 2
        assert router.import_mgr.import_bundle() == []
        assert router.diagnose()["detached"] is None

    def test_force_reattach_mismatched_root(self, env, tmp_path):
        tmp, router = env["tmp"], env["router"]
        _seed(router, tmp)
        router.import_mgr.detach()
        other = tmp_path / "other"
        other.mkdir()
        _write_okf(other, "z.md", "Zed",
                   "Zed content living in a different tree entirely.")
        with pytest.raises(RuntimeError, match="different source tree"):
            router.import_mgr.import_bundle(other, force=True)
        assert router.import_mgr.delta_mgr.is_detached()

    def test_deleted_commands_work_detached(self, env):
        tmp, router = env["tmp"], env["router"]
        _seed(router, tmp)
        router.import_mgr.detach()
        assert router.purge_mgr._soft_delete_concept("a") is True
        listed = router.purge_mgr.list_deleted_concepts()
        assert any(r["concept_id"] == "a" for r in listed)
        assert router.purge_mgr._recover_concept("a") is True
        assert "a" in _concepts(router.conn)


def _native(*parts):
    return str(Path(*parts))
