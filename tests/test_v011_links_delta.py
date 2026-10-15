"""0.11.0 acceptance: links survive re-import (O1), per-file change detection
(O2), typed relations (O6) and ambiguous wikilinks (O12).

Spec: the "OKFgraph 0.11" findings-store changes. Each test names the
acceptance criterion it pins.
"""
from __future__ import annotations

import json

import pytest

from okfgraph import OKFRouter
from okfgraph.errors import OKFError

from tests._v011 import (  # noqa: F401  (fixture import)
    broken, concept_ids, edges_from, links_into, note, stub_encoder, write,
)


@pytest.fixture
def kb(tmp_path):
    return tmp_path / "kb"


def _router(tmp_path, kb, **kw):
    return OKFRouter(db_path=str(tmp_path / "kb.db"), bundle_root=str(kb), **kw)


# ---------------------------------------------------------------------------
# O1 — incoming links survive a re-import of their target
# ---------------------------------------------------------------------------

def test_o1_cross_directory_link_survives_reimport_and_md_ingest(tmp_path, kb, stub_encoder):
    """ev/t.md and dec/a.md, a links to ../ev/t.md. Edit t, re-import, then
    ingest(kind="md") it: the link dec/a -> ev/t exists after each step."""
    write(kb, "ev/t.md", note("T", "version one"))
    write(kb, "dec/a.md", note("A", "see [t](../ev/t.md)"))
    r = _router(tmp_path, kb)
    try:
        r.import_bundle()
        assert links_into(r, "ev/t") == [("dec/a", "mention")]

        write(kb, "ev/t.md", note("T", "version two"))
        res = r.import_bundle()
        assert res["concept_ids"] == ["ev/t"], "only the edited file re-imports (O2)"
        assert links_into(r, "ev/t") == [("dec/a", "mention")]

        write(kb, "ev/t.md", note("T", "version three"))
        r.ingest(kind="md", md_path=str(kb / "ev/t.md"))
        assert links_into(r, "ev/t") == [("dec/a", "mention")]
        assert r.read("ev/t")["body"].strip() == "version three"
    finally:
        r.close()


def test_o1_root_relative_and_same_directory_links_survive(tmp_path, kb, stub_encoder):
    write(kb, "ev/t.md", note("T", "target"))
    write(kb, "ev/sib.md", note("Sib", "[t](t.md)"))            # same dir, relative
    write(kb, "dec/root.md", note("Root", "[t](/ev/t.md)"))      # root-relative
    r = _router(tmp_path, kb)
    try:
        r.import_bundle()
        expected = [("dec/root", "mention"), ("ev/sib", "mention")]
        assert links_into(r, "ev/t") == expected
        write(kb, "ev/t.md", note("T", "target, edited"))
        r.import_bundle()
        assert links_into(r, "ev/t") == expected
    finally:
        r.close()


def test_o1_deleted_target_becomes_broken_link_and_recreate_repairs(tmp_path, kb, stub_encoder):
    """Delete t: a BrokenLink from dec/a appears. Recreate t: the link is back."""
    write(kb, "ev/t.md", note("T", "target"))
    write(kb, "dec/a.md", note("A", "see [t](../ev/t.md)"))
    r = _router(tmp_path, kb)
    try:
        r.import_bundle()
        (kb / "ev/t.md").unlink()
        r.import_bundle(prune_missing=True)
        assert "ev/t" not in concept_ids(r)
        assert broken(r) == [{"source": "dec/a", "target": "ev/t", "kind": "mention",
                              "reason": "missing", "candidates": []}]
        report = r.doctor()["report"]
        assert any(f["rule"] == "broken_link" and f["path"] == "dec/a"
                   for f in report["findings"])

        write(kb, "ev/t.md", note("T", "back again"))
        r.import_bundle()
        assert links_into(r, "ev/t") == [("dec/a", "mention")]
        assert broken(r) == []
    finally:
        r.close()


def test_o1_link_written_before_its_target_resolves_without_reimporting_source(
        tmp_path, kb, stub_encoder):
    """A link whose target arrives in a later import resolves then (repair on
    every import), so link state does not depend on import order."""
    write(kb, "dec/a.md", note("A", "see [t](../ev/t.md)"))
    r = _router(tmp_path, kb)
    try:
        r.import_bundle()
        assert [b["target"] for b in broken(r)] == ["ev/t"]
        write(kb, "ev/t.md", note("T", "arrives later"))
        res = r.import_bundle()
        assert res["concept_ids"] == ["ev/t"]
        assert links_into(r, "ev/t") == [("dec/a", "mention")]
        assert broken(r) == []
    finally:
        r.close()


# ---------------------------------------------------------------------------
# O2 — per-file change detection
# ---------------------------------------------------------------------------

def test_o2_editing_one_of_fifty_embeds_only_that_file(tmp_path, kb, stub_encoder):
    """50 files in one directory. Edit one: the import embeds exactly that
    file's concept and chunks. A no-op import embeds nothing."""
    for i in range(50):
        write(kb, f"ev/f{i:02d}.md", note(f"F{i}", f"finding number {i}"))
    r = _router(tmp_path, kb)
    try:
        r.import_bundle()
        stub_encoder.reset()
        r.import_bundle()
        assert stub_encoder.count == 0, "no-op import embeds nothing"

        write(kb, "ev/f17.md", note("F17", "finding number 17, revised"))
        stub_encoder.reset()
        res = r.import_bundle()
        assert res["concept_ids"] == ["ev/f17"]
        chunks = r.conn.execute(
            "MATCH (ch:Chunk {parent_doc_id: 'ev/f17'}) RETURN count(ch) AS n"
        ).rows_as_dict().get_all()[0]["n"]
        assert stub_encoder.count == 1 + chunks, "the concept plus its chunks, nothing else"

        stub_encoder.reset()
        r.import_bundle()
        assert stub_encoder.count == 0
    finally:
        r.close()


def test_o2_single_file_ingest_then_bundle_import_embeds_nothing(tmp_path, kb, stub_encoder):
    """ingest(kind="md") on one file, then a bundle import: the second step
    embeds nothing (the ingest recorded the file's hash)."""
    for i in range(5):
        write(kb, f"ev/f{i}.md", note(f"F{i}", f"body {i}"))
    r = _router(tmp_path, kb)
    try:
        r.import_bundle()
        write(kb, "ev/f3.md", note("F3", "body 3, edited"))
        r.ingest(kind="md", md_path=str(kb / "ev/f3.md"))
        stub_encoder.reset()
        res = r.import_bundle()
        assert res["concept_ids"] == []
        assert stub_encoder.count == 0
    finally:
        r.close()


def test_o2_concept_lost_from_the_graph_reimports_even_with_matching_hash(
        tmp_path, kb, stub_encoder):
    """Unchanged = same bytes AND the concept still exists: a concept removed
    behind the delta's back must come back, never be skipped forever."""
    for i in range(3):
        write(kb, f"ev/f{i}.md", note(f"F{i}", f"body {i}"))
    write(kb, "ev/f0.md", note("F0", "body 0 edited"))  # dir changed vs nothing yet
    r = _router(tmp_path, kb)
    try:
        r.import_bundle()
        r.conn.execute("MATCH (c:Concept {id: 'ev/f1'}) DETACH DELETE c")
        write(kb, "ev/f2.md", note("F2", "body 2 edited"))  # marks the dir changed
        res = r.import_bundle()
        assert sorted(res["concept_ids"]) == ["ev/f1", "ev/f2"]
    finally:
        r.close()


# ---------------------------------------------------------------------------
# O6 — typed relations in frontmatter
# ---------------------------------------------------------------------------

def test_o6_relations_create_typed_edges_and_traverse_follows_a_kind(tmp_path, kb, stub_encoder):
    write(kb, "f/old.md", note("Old finding", "old"))
    write(kb, "d/ledger.md", note("Ledger design", "design"))
    write(kb, "f/new.md", note("New finding", "see [old](old.md)",
                               relations={"supersedes": ["f/old"],
                                          "supports": "d/ledger"}))
    r = _router(tmp_path, kb)
    try:
        r.import_bundle()
        assert edges_from(r, "f/new") == [
            ("d/ledger", "supports"), ("f/old", "mention"), ("f/old", "supersedes")]
        hops = r.traverse("f/new", relationship="LINKS_TO", link_kinds=["supersedes"])
        assert [h["id"] for h in hops] == ["f/old"]
        # Without link_kinds: one row per neighbour, not per edge.
        hops = r.traverse("f/new", relationship="LINKS_TO")
        assert sorted(h["id"] for h in hops) == ["d/ledger", "f/old"]
        with pytest.raises(OKFError) as e:
            r.traverse("f/new", relationship="CONTAINS", link_kinds=["supersedes"])
        assert e.value.code == "BAD_VALUE"
    finally:
        r.close()


def test_o6_export_round_trips_relations(tmp_path, kb, stub_encoder):
    write(kb, "f/old.md", note("Old", "old"))
    write(kb, "f/new.md", note("New", "new", relations={"supersedes": ["f/old"]}))
    r = _router(tmp_path, kb)
    try:
        r.import_bundle()
        out = tmp_path / "export"
        r.export_bundle(str(out))
        import frontmatter
        meta = frontmatter.load(out / "f/new.md").metadata
        assert meta["relations"] == {"supersedes": ["f/old"]}
    finally:
        r.close()


def test_o6_unresolved_relation_is_a_broken_link_with_its_kind(tmp_path, kb, stub_encoder):
    write(kb, "f/new.md", note("New", "x", relations={"depends-on": ["missing/thing"],
                                                      "mention": ["f/x"]}))
    r = _router(tmp_path, kb)
    try:
        r.import_bundle()
        assert broken(r) == [{"source": "f/new", "target": "missing/thing",
                              "kind": "depends-on", "reason": "missing",
                              "candidates": []}]
        # ``mention`` cannot be declared: lint reports it, import skips it.
        from okfgraph.components.lint import lint_bundle
        rules = {e["rule"] for e in lint_bundle(kb)["errors"]}
        assert {"bad_relation", "dangling_relation"} <= rules
    finally:
        r.close()


def test_o6_migration_v8_to_v9_marks_existing_edges_as_mention(tmp_path):
    """The schema rung: a pre-0.11 LINKS_TO table gains ``kind`` and every
    existing edge reads as ``mention``; BrokenLink gains its new columns."""
    import ladybug as lb
    from okfgraph.components.schema import SchemaManager

    db = lb.Database(str(tmp_path / "v8.db"))
    conn = lb.Connection(db)
    try:
        conn.execute("CREATE NODE TABLE Concept (id STRING PRIMARY KEY)")
        conn.execute("CREATE REL TABLE LINKS_TO (FROM Concept TO Concept)")
        conn.execute("CREATE NODE TABLE BrokenLink (id STRING PRIMARY KEY, "
                     "source_id STRING, target_id STRING, timestamp TIMESTAMP)")
        conn.execute("CREATE (:Concept {id: 'a'}), (:Concept {id: 'b'})")
        conn.execute("MATCH (a:Concept {id:'a'}), (b:Concept {id:'b'}) "
                     "CREATE (a)-[:LINKS_TO]->(b)")
        mgr = SchemaManager(conn, 512, None)
        mgr._migrate_v8_to_v9()
        mgr._migrate_v8_to_v9()  # idempotent: a re-run after a crash is clean
        kinds = conn.execute(
            "MATCH (:Concept)-[l:LINKS_TO]->(:Concept) RETURN l.kind AS k"
        ).rows_as_dict().get_all()
        assert [k["k"] for k in kinds] == ["mention"]
        cols = {c["name"] for c in conn.execute(
            "CALL TABLE_INFO('BrokenLink') RETURN *").rows_as_dict().get_all()}
        assert {"kind", "reason", "candidates"} <= cols
        conn.execute("MATCH (p:ProducerCursor) RETURN count(p)")
    finally:
        conn.close()
        db.close()


# ---------------------------------------------------------------------------
# O12 — ambiguous wikilinks are reported, not just logged
# ---------------------------------------------------------------------------

def test_o12_ambiguous_wikilink_is_a_broken_link_with_candidates(tmp_path, kb, stub_encoder):
    """Two notes titled "Use UUIDv7": [[Use UUIDv7]] resolves to nothing and
    is recorded with reason ambiguous and both candidate ids; doctor shows it."""
    write(kb, "d/u1.md", note("Use UUIDv7", "one"))
    write(kb, "d/u2.md", note("Use UUIDv7", "two"))
    write(kb, "d/ref.md", note("Ref", "see [[Use UUIDv7]]"))
    r = _router(tmp_path, kb)
    try:
        r.import_bundle()
        assert broken(r) == [{"source": "d/ref", "target": "Use UUIDv7",
                              "kind": "mention", "reason": "ambiguous",
                              "candidates": ["d/u1", "d/u2"]}]
        findings = r.doctor()["report"]["findings"]
        amb = [f for f in findings if f["rule"] == "ambiguous_link"]
        assert amb and "d/u1" in amb[0]["message"] and "d/u2" in amb[0]["message"]

        # Rename one: the name becomes unique and the link repairs on import.
        write(kb, "d/u2.md", note("Prefer ULID", "two"))
        r.import_bundle()
        assert edges_from(r, "d/ref") == [("d/u1", "mention")]
        assert broken(r) == []
    finally:
        r.close()


def test_o12_lint_names_the_candidates(tmp_path, kb):
    write(kb, "d/u1.md", note("Use UUIDv7", "one"))
    write(kb, "d/u2.md", note("Use UUIDv7", "two"))
    write(kb, "d/ref.md", note("Ref", "see [[Use UUIDv7]]"))
    from okfgraph.components.lint import lint_bundle
    errs = [e for e in lint_bundle(kb)["errors"] if e["rule"] == "dangling_wikilink"]
    assert errs and errs[0]["reason"] == "ambiguous"
    assert errs[0]["candidates"] == ["d/u1", "d/u2"]
    assert json.dumps(errs[0])  # report stays JSON-serialisable
