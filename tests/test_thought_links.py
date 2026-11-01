"""Thoughts link to what they are about.

A ``[[Title]]`` in a thought's text is an ordinary ``mention`` link, and
``related=[ids]`` adds typed ``related`` links (O6), so traverse can walk
from a thought and hub/PPR ranking see it. Unresolved ``related`` targets
are reported and repaired once the target arrives.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from okfgraph.cli import build_parser
from okfgraph.errors import OKFError
from okfgraph.router import OKFRouter


@pytest.fixture
def router():
    tmp = tempfile.mkdtemp()
    r = OKFRouter(db_path=str(Path(tmp) / "t.db"), bundle_root=tmp,
                  embedding_dim=512, chunk_size=50, chunk_overlap=10,
                  device="cpu")
    try:
        yield r
    finally:
        r.close()
        shutil.rmtree(tmp, ignore_errors=True)


def _note(router, cid, title):
    return router.ingest("thoughts", thoughts=f"{title} is a design note.",
                         topic="notes", concept_id=cid, title=title)


def _links(router, source):
    rows = router.conn.execute(
        "MATCH (a:Concept {id: $s})-[l:LINKS_TO]->(b:Concept) "
        "RETURN b.id AS id, l.kind AS kind ORDER BY b.id, l.kind",
        {"s": source}).rows_as_dict().get_all()
    return [(r["id"], r["kind"]) for r in rows]


def test_wikilink_in_thought_text_is_a_mention(router):
    _note(router, "design/delta", "Delta detection")
    out = router.ingest("thoughts", topic="perf",
                        thoughts="[[Delta detection]] re-embeds too much.")
    assert _links(router, out["concept_id"]) == [("design/delta", "mention")]


def test_related_writes_typed_links(router):
    _note(router, "design/delta", "Delta detection")
    _note(router, "design/search", "Search ranking")
    out = router.ingest("thoughts", topic="perf", thoughts="Per-file hashing helps.",
                        related=["design/delta", "Search ranking"])
    cid = out["concept_id"]
    assert _links(router, cid) == [("design/delta", "related"),
                                   ("design/search", "related")]
    assert out["unresolved_related"] == []
    walked = router.traverse(cid, relationship="LINKS_TO", link_kinds=["related"])
    assert sorted(r["id"] for r in walked) == ["design/delta", "design/search"]


def test_related_merges_with_extra_relations(router):
    _note(router, "design/delta", "Delta detection")
    _note(router, "design/old", "Old idea")
    out = router.ingest("thoughts", topic="perf", thoughts="Builds on both.",
                        extra={"relations": {"supersedes": ["design/old"],
                                             "related": "design/old"}},
                        related=["design/delta", "design/old"])
    assert _links(router, out["concept_id"]) == [
        ("design/delta", "related"), ("design/old", "related"),
        ("design/old", "supersedes")]
    assert "related:\n  - design/old\n  - design/delta" in out["markdown"]


def test_unresolved_related_is_reported_then_repaired(router):
    out = router.ingest("thoughts", topic="perf", thoughts="About a later note.",
                        related=["design/later"])
    cid = out["concept_id"]
    assert out["unresolved_related"] == ["design/later"]
    assert _links(router, cid) == []
    _note(router, "design/later", "Later note")
    assert _links(router, cid) == [("design/later", "related")]


@pytest.mark.parametrize("bad", ["design/x", [""], [3]])
def test_bad_related_is_refused(router, bad):
    with pytest.raises(OKFError) as e:
        router.ingest("thoughts", topic="t", thoughts="x.", related=bad)
    assert e.value.code == "BAD_VALUE"


def test_related_is_thoughts_only(router, tmp_path):
    md = tmp_path / "n.md"
    md.write_text("# N\n\nbody\n", encoding="utf-8")
    with pytest.raises(OKFError) as e:
        router.ingest("md", md_path=str(md), related=["x"])
    assert e.value.code == "BAD_VALUE"
    assert "related" in str(e.value)


def test_cli_related_is_repeatable():
    args = build_parser().parse_args(
        ["ingest", "--kind", "thoughts", "--thoughts", "x", "--topic", "t",
         "--related", "a/b", "--related", "Some title"])
    assert args.related == ["a/b", "Some title"]
