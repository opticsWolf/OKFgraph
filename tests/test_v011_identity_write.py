"""0.11.0 acceptance: frontmatter identity (O3), search filters (O7),
deferred index step (O8), typed idempotent ingest (O9), producer cursors
(O10) and the write ops update / supersede / delete (O11).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import frontmatter
import pytest

from okfgraph import OKFRouter
from okfgraph.errors import OKFError

from tests._v011 import (  # noqa: F401  (fixture import)
    concept_ids, edges_from, links_into, note, stub_encoder, write,
)


@pytest.fixture
def kb(tmp_path):
    return tmp_path / "kb"


def _router(tmp_path, kb, **kw):
    return OKFRouter(db_path=str(tmp_path / "kb.db"), bundle_root=str(kb), **kw)


def _ids(hits):
    return [h["id"] for h in hits]


# ---------------------------------------------------------------------------
# O3 — stable identity from frontmatter (opt-in)
# ---------------------------------------------------------------------------

def test_o3_move_keeps_id_links_and_leaves_no_tombstone(tmp_path, kb, stub_encoder):
    """identity="frontmatter": move a file with id: finding/x to another
    directory — same concept id, new source_path, links kept, no tombstone."""
    write(kb, "evidence/x.md", note("X", "the finding", id="finding/x"))
    write(kb, "docs/a.md", note("A", "path [x](../evidence/x.md) and wiki [[finding/x]]"))
    write(kb, "docs/b.md", note("B", "b", relations={"supersedes": ["finding/x"]}))
    r = _router(tmp_path, kb, identity="frontmatter")
    try:
        r.import_bundle()
        assert concept_ids(r) == ["docs/a", "docs/b", "finding/x"]
        before = links_into(r, "finding/x")
        assert before == [("docs/a", "mention"), ("docs/b", "supersedes")]
        assert r.read("finding/x")["source_path"] == "evidence/x.md"

        (kb / "decisions").mkdir()
        (kb / "evidence/x.md").rename(kb / "decisions/x.md")
        res = r.import_bundle(prune_missing=True)
        assert res["concept_ids"] == ["finding/x"]
        assert res["moved"] == ["evidence/x.md"]
        assert concept_ids(r) == ["docs/a", "docs/b", "finding/x"]
        assert r.read("finding/x")["source_path"] == "decisions/x.md"
        assert links_into(r, "finding/x") == before
        tomb = r.conn.execute("MATCH (d:DeletedPath) RETURN count(d) AS n").rows_as_dict().get_all()
        assert tomb[0]["n"] == 0
    finally:
        r.close()


def test_o3_two_files_one_id_refuses_the_second(tmp_path, kb, stub_encoder):
    write(kb, "a/x.md", note("First", "one", id="finding/x"))
    write(kb, "b/x.md", note("Second", "two", id="finding/x"))
    r = _router(tmp_path, kb, identity="frontmatter")
    try:
        res = r.import_bundle()
        assert res["concept_ids"] == ["finding/x"]
        assert res["duplicates"] == [str(kb / "b/x.md")]
        assert r.read("finding/x")["title"] == "First"
        # Later, a third file claiming the id while the first still exists.
        write(kb, "c/x.md", note("Third", "three", id="finding/x"))
        res = r.import_bundle()
        assert str(kb / "c/x.md") in res["duplicates"]
        assert r.read("finding/x")["title"] == "First"
    finally:
        r.close()
    from okfgraph.components.lint import lint_bundle
    dups = [e for e in lint_bundle(kb, identity="frontmatter")["errors"]
            if e["rule"] == "duplicate_id"]
    assert len(dups) == 2


def test_o3_files_without_id_keep_path_ids_and_bad_ids_are_refused(tmp_path, kb, stub_encoder):
    write(kb, "plain/n.md", note("Plain", "no id here"))
    write(kb, "bad/n.md", note("Bad", "escape", id="../outside"))
    r = _router(tmp_path, kb, identity="frontmatter")
    try:
        r.import_bundle()
        assert concept_ids(r) == ["plain/n"]
    finally:
        r.close()


def test_o3_path_identity_is_unchanged_by_default(tmp_path, kb, stub_encoder):
    """Default identity: an id: is only a wikilink alias (pre-0.11 behaviour)."""
    write(kb, "evidence/x.md", note("X", "x", id="finding/x"))
    r = _router(tmp_path, kb)
    try:
        r.import_bundle()
        assert concept_ids(r) == ["evidence/x"]
        assert r.read("evidence/x")["uid"] == "finding/x"
    finally:
        r.close()


def test_o3_export_writes_id_back(tmp_path, kb, stub_encoder):
    write(kb, "evidence/x.md", note("X", "x", id="finding/x"))
    r = _router(tmp_path, kb, identity="frontmatter")
    try:
        r.import_bundle()
        out = tmp_path / "out"
        r.export_bundle(str(out))
        files = list(out.rglob("*.md"))
        metas = [frontmatter.load(f).metadata for f in files]
        assert any(m.get("id") == "finding/x" for m in metas)
        assert not any("source_path" in m for m in metas)
    finally:
        r.close()


# ---------------------------------------------------------------------------
# O7 — filters on extras, time and excluded tags
# ---------------------------------------------------------------------------

@pytest.fixture
def filtered_kb(tmp_path, kb, stub_encoder):
    write(kb, "f/e1.md", note("Zebra evidence one", "zebra stripes",
                              jev_category="evidence", timestamp="2026-01-10T00:00:00"))
    write(kb, "f/e2.md", note("Zebra evidence two", "zebra herd",
                              jev_category="evidence", timestamp="2026-03-10T00:00:00",
                              tags=["superseded"]))
    write(kb, "f/c1.md", note("Zebra claim", "zebra claim",
                              jev_category="claim", timestamp="2026-02-10T00:00:00"))
    return kb


def test_o7_where_since_until_and_exclude_tags(tmp_path, filtered_kb):
    r = _router(tmp_path, filtered_kb)
    try:
        r.import_bundle()
        for rank in ("none", "ppr"):
            hits = r.search("zebra", limit=10, rank=rank, where={"jev_category": "evidence"})
            assert sorted(_ids(hits)) == ["f/e1", "f/e2"], rank
            hits = r.search("zebra", limit=10, rank=rank, since="2026-02-01", until="2026-03-01")
            assert _ids(hits) == ["f/c1"], rank
            hits = r.search("zebra", limit=10, rank=rank, exclude_tags=["superseded"])
            assert sorted(_ids(hits)) == ["f/c1", "f/e1"], rank
        with pytest.raises(OKFError) as e:
            r.search("zebra", since="last tuesday")
        assert e.value.code == "BAD_VALUE"
    finally:
        r.close()


def test_o7_configured_exclude_tags_is_a_default_a_call_can_override(tmp_path, filtered_kb):
    r = _router(tmp_path, filtered_kb, exclude_tags=["superseded"])
    try:
        r.import_bundle()
        assert "f/e2" not in _ids(r.search("zebra", limit=10))
        assert "f/e2" in _ids(r.search("zebra", limit=10, exclude_tags=[]))
    finally:
        r.close()


# ---------------------------------------------------------------------------
# O8 — fresh results without a rebuild per import
# ---------------------------------------------------------------------------

def test_o8_deferred_index_step_still_finds_the_new_file(tmp_path, kb, stub_encoder):
    """import_files([one_new_file], rebuild="defer"): a search finds it
    before any rebuild — Ladybug 0.21.2 maintains both indexes in place."""
    for i in range(5):
        write(kb, f"f/n{i}.md", note(f"Note {i}", f"ordinary note {i}"))
    r = _router(tmp_path, kb)
    try:
        r.import_bundle()
        new = write(kb, "f/fresh.md", note("Quokka finding", "quokka quokka"))
        res = r.import_files([new], rebuild="defer")
        assert res["concept_ids"] == ["f/fresh"]
        assert _ids(r.search("quokka", limit=3))[0] == "f/fresh"
        # Chunk index too.
        hits = r.search("quokka", target="chunks", limit=3)
        assert hits and hits[0]["parent_doc_id"] == "f/fresh"
        assert r.reindex()["rebuilt"] is True  # force: drop + recreate works
        assert _ids(r.search("quokka", limit=3))[0] == "f/fresh"
    finally:
        r.close()


def test_o8_import_files_refuses_missing_files_before_writing(tmp_path, kb, stub_encoder):
    write(kb, "f/a.md", note("A", "a"))
    r = _router(tmp_path, kb)
    try:
        with pytest.raises(OKFError) as e:
            r.import_files([kb / "f/a.md", kb / "f/missing.md"])
        assert e.value.code == "FILE_NOT_FOUND"
        assert concept_ids(r) == []
        with pytest.raises(OKFError):
            r.import_bundle(rebuild="sometimes")
    finally:
        r.close()


# ---------------------------------------------------------------------------
# O9 — typed, idempotent ingest that can write the file
# ---------------------------------------------------------------------------

def test_o9_same_thoughts_call_twice_is_one_concept_with_type_and_file(tmp_path, kb, stub_encoder):
    kb.mkdir()
    r = _router(tmp_path, kb)
    try:
        for _ in range(2):
            out = r.ingest(kind="thoughts", thoughts="Use UUIDv7 for row ids.",
                           topic="ids", concept_id="decisions/uuid7",
                           type="decision", extra={"jev_category": "decision"},
                           materialize=True)
        assert out["concept_id"] == "decisions/uuid7"
        assert concept_ids(r) == ["decisions/uuid7"]
        c = r.read("decisions/uuid7")
        assert c["type"] == "decision" and c["jev_category"] == "decision"
        f = kb / "decisions/uuid7.md"
        assert f.is_file() and frontmatter.load(f).metadata["type"] == "decision"
        # The bundle is the source of truth: a bundle import changes nothing.
        stub_encoder.reset()
        assert r.import_bundle()["concept_ids"] == []
        assert stub_encoder.count == 0
    finally:
        r.close()


def test_o9_thoughts_without_materialize_upsert_by_id_and_keep_links(tmp_path, kb, stub_encoder):
    write(kb, "d/ref.md", note("Ref", "see [[thoughts/t/1]]"))
    r = _router(tmp_path, kb)
    try:
        r.import_bundle()
        r.ingest(kind="thoughts", thoughts="first", topic="t", concept_id="thoughts/t/1")
        r.ingest(kind="thoughts", thoughts="second", topic="t", concept_id="thoughts/t/1")
        assert [c for c in concept_ids(r) if c.startswith("thoughts/")] == ["thoughts/t/1"]
        assert links_into(r, "thoughts/t/1") == [("d/ref", "mention")]
        assert r.read("thoughts/t/1")["type"] == "thought"
    finally:
        r.close()


def test_o9_md_ingest_keeps_all_frontmatter_and_refuses_reserved_extra(tmp_path, kb, stub_encoder):
    src = write(tmp_path / "outside", "n.md",
                note("Outside", "body", relations={"supports": ["d/x"]}, jev_category="evidence"))
    write(kb, "d/x.md", note("X", "x"))
    r = _router(tmp_path, kb)
    try:
        r.import_bundle()
        out = r.ingest(kind="md", md_path=str(src), concept_id="findings/n",
                       type="finding", materialize=True)
        assert out["path"] == str(kb / "findings/n.md")
        c = r.read("findings/n")
        assert c["type"] == "finding" and c["jev_category"] == "evidence"
        assert edges_from(r, "findings/n") == [("d/x", "supports")]
        with pytest.raises(OKFError) as e:
            r.ingest(kind="thoughts", thoughts="x", topic="t", extra={"title": "nope"})
        assert e.value.code == "BAD_VALUE"
        with pytest.raises(OKFError) as e:
            r.ingest(kind="thoughts", thoughts="x", topic="t",
                     concept_id="../escape", materialize=True)
        assert e.value.code == "BAD_VALUE"
    finally:
        r.close()


# ---------------------------------------------------------------------------
# O10 — producer plugins with a cursor
# ---------------------------------------------------------------------------

def _sqlite(path: Path, rows: int) -> Path:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY, name TEXT)")
    con.execute("DELETE FROM runs")
    con.executemany("INSERT INTO runs (name) VALUES (?)", [(f"run {i}",) for i in range(rows)])
    con.commit()
    con.close()
    return path


def test_o10_sync_imports_what_was_written_and_stores_the_cursor(tmp_path, kb, stub_encoder):
    kb.mkdir()
    src = _sqlite(tmp_path / "runs.sqlite", 3)
    r = _router(tmp_path, kb)
    try:
        first = r.sync("sqlite", str(src))
        assert first["written"] and first["concept_ids"]
        assert first["cursor"]
        again = r.sync("sqlite", str(src))
        assert again == {"producer": "sqlite", "cursor": first["cursor"],
                         "written": [], "concept_ids": []}
        _sqlite(src, 5)
        third = r.sync("sqlite", str(src))
        assert third["cursor"] != first["cursor"] and third["written"]
    finally:
        r.close()


def test_o10_entry_point_producers_are_discovered_and_cannot_shadow_builtins(monkeypatch):
    import importlib.metadata as md

    from okfgraph.components import producers as P

    class Fake:
        name = "fake"

        def produce(self, source, output_dir, *, prefix=None, overwrite=False):
            raise AssertionError("not called")

    class EP:
        def __init__(self, name, obj):
            self.name, self._obj = name, obj

        def load(self):
            return self._obj

    class Broken(EP):
        def load(self):
            raise ImportError("plugin is broken")

    monkeypatch.setattr(md, "entry_points", lambda group: [
        EP("silk-findings", Fake), EP("sqlite", Fake), Broken("bad", None)])
    found = P.available_producers()
    assert found["silk-findings"] is Fake
    assert found["sqlite"] is P.SQLiteProducer
    assert "bad" not in found
    # A producer without a cursor parameter is still called correctly.
    seen = {}

    class Old:
        def produce(self, source, output_dir, *, prefix=None, overwrite=False):
            seen.update(prefix=prefix, overwrite=overwrite)
            return "bundle"

    assert P.run_producer(Old(), "s", "o", prefix="p", cursor="c") == "bundle"
    assert seen == {"prefix": "p", "overwrite": False}


# ---------------------------------------------------------------------------
# O11 — update, supersede, delete (write the file, then import it)
# ---------------------------------------------------------------------------

def test_o11_update_rewrites_frontmatter_and_imports(tmp_path, kb, stub_encoder):
    path = write(kb, "f/a.md", note("Alpha", "the body stays", tags=["draft"]))
    r = _router(tmp_path, kb)
    try:
        r.import_bundle()
        res = r.update("f/a", title="Alpha v2", add_tags=["reviewed"],
                       remove_tags=["draft"], extra={"jev_category": "evidence"})
        assert res["path"] == str(path)
        assert set(res["changed"]) == {"title", "tags", "jev_category"}
        meta = frontmatter.load(path)
        assert meta["title"] == "Alpha v2" and meta["tags"] == ["reviewed"]
        assert meta.content.strip() == "the body stays"
        c = r.read("f/a")
        assert c["title"] == "Alpha v2" and c["jev_category"] == "evidence"
        assert r.update("f/a", title="Alpha v2")["changed"] == []
        with pytest.raises(OKFError) as e:
            r.update("f/a", extra={"id": "x"})
        assert e.value.code == "BAD_VALUE"
        with pytest.raises(OKFError) as e:
            r.update("nope", title="x")
        assert e.value.code == "UNKNOWN_CONCEPT"
    finally:
        r.close()


def test_o11_supersede_links_new_tags_old_and_search_hides_it(tmp_path, kb, stub_encoder):
    write(kb, "f/old.md", note("Old zebra finding", "zebra old"))
    write(kb, "f/new.md", note("New zebra finding", "zebra new"))
    r = _router(tmp_path, kb, exclude_tags=["superseded"])
    try:
        r.import_bundle()
        res = r.supersede("f/old", "f/new")
        assert res["old_path"] == str(kb / "f/old.md")
        assert ("f/old", "supersedes") in edges_from(r, "f/new")
        old = r.read("f/old")
        assert "superseded" in old["tags"] and old["superseded_by"] == "f/new"
        assert _ids(r.search("zebra", limit=5)) == ["f/new"]
        assert frontmatter.load(kb / "f/new.md")["relations"] == {"supersedes": ["f/old"]}
        r.supersede("f/old", "f/new")  # idempotent
        assert frontmatter.load(kb / "f/new.md")["relations"] == {"supersedes": ["f/old"]}
    finally:
        r.close()


def test_o11_delete_with_remove_file_trashes_and_recover_restores(tmp_path, kb, stub_encoder):
    path = write(kb, "f/gone.md", note("Gone", "soon gone"))
    write(kb, "f/ref.md", note("Ref", "[g](gone.md)"))
    r = _router(tmp_path, kb)
    try:
        r.import_bundle()
        with pytest.raises(OKFError):
            r.delete("f/gone")  # backed by a file: refused without remove_file
        res = r.delete("f/gone", remove_file=True)
        assert not path.exists()
        trash = Path(res["trashed_files"][0])
        assert trash.is_file() and ".okf-trash" in trash.parts
        assert "f/gone" not in concept_ids(r)
        # The trash is never walked: a bundle import does not resurrect it.
        assert r.import_bundle()["concept_ids"] == []
        rec = r.recover_deleted("f/gone")
        assert rec["restored_files"] == [str(path)] and path.is_file()
        assert links_into(r, "f/gone") == [("f/ref", "mention")]
        assert r.import_bundle()["concept_ids"] == []
    finally:
        r.close()


def test_o11_graph_only_concept_is_materialized_on_update(tmp_path, kb, stub_encoder):
    kb.mkdir()
    r = _router(tmp_path, kb)
    try:
        r.ingest(kind="thoughts", thoughts="graph only", topic="t", concept_id="thoughts/t/x")
        res = r.update("thoughts/t/x", add_tags=["kept"])
        assert res["path"] == str(kb / "thoughts/t/x.md")
        assert Path(res["path"]).is_file()
        assert "kept" in r.read("thoughts/t/x")["tags"]
    finally:
        r.close()
