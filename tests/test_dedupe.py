"""``search(target="chunks", dedupe=True)`` (0.11.0, plan §10.2): a chunk
whose stored embedding repeats a higher-ranked one is dropped and listed
on the kept row as ``collapsed``."""

import hashlib

import pytest

from okfgraph.cli import build_parser
from okfgraph.errors import OKFError
from okfgraph.router import OKFRouter


def _vec(text, dim):
    h = hashlib.sha256((text or "").encode()).digest()
    v = [((h[i % len(h)] / 255.0) - 0.5) for i in range(dim)]
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


@pytest.fixture
def env(tmp_path, monkeypatch):
    from okfgraph.components import embedding as E

    dim = lambda self: int(getattr(self, "_truncate_dim", 512) or 512)  # noqa: E731
    monkeypatch.setattr(E.LazyRustEncoder, "encode",
                        lambda self, text, task="Document": _vec(text, dim(self)))
    monkeypatch.setattr(E.LazyRustEncoder, "encode_batch",
                        lambda self, texts, task="Document": [_vec(t, dim(self)) for t in texts])
    monkeypatch.setattr(E.LazyRustEncoder, "count_tokens",
                        lambda self, text: len((text or "").split()))
    monkeypatch.setattr(E.LazyRustEncoder, "is_loaded", property(lambda self: True))
    kb = tmp_path / "kb"
    kb.mkdir()
    for name, body in {
        "plan": "Hash each file, not each directory, to find what changed.",
        "plan-v2": "Hash each file, not each directory, to find what changed!",
        "plan-v3": "Hash each file, not each directory, to find what changed?",
        "other": "Directory hashing re-embeds whole folders after one edit.",
    }.items():
        (kb / f"{name}.md").write_text(f"---\ntitle: {name}\n---\n\n{body}\n",
                                       encoding="utf-8")
    r = OKFRouter(db_path=str(tmp_path / "kb.db"), bundle_root=str(kb),
                  embedding_dim=512, device="cpu")
    r.import_bundle(str(kb))
    # Make plan-v2 / plan-v3 near-identical to plan: copy plan's vector
    # (a real model puts copied passages at cosine ~0.99).
    ids = {row["p"]: row["id"] for row in r.conn.execute(
        "MATCH (ch:Chunk) RETURN ch.parent_doc_id AS p, ch.id AS id"
    ).rows_as_dict().get_all()}
    for twin in ("plan-v2", "plan-v3"):
        r.conn.execute(
            "MATCH (a:Chunk {id: $a}), (b:Chunk {id: $b}) SET b.embedding = a.embedding",
            {"a": ids["plan"], "b": ids[twin]})
    yield r, ids
    r.close()


def _chunks(r, **kw):
    return r.search("hash each file directory changed", target="chunks", **kw)


def test_off_by_default_keeps_every_copy(env):
    r, _ = env
    out = _chunks(r)
    assert {x["concept_id"] for x in out} >= {"plan", "plan-v2", "plan-v3"}
    assert not any("collapsed" in x for x in out)


def test_dedupe_keeps_the_best_ranked_copy(env):
    r, ids = env
    plain = [x["concept_id"] for x in _chunks(r)]
    copies = [c for c in plain if c.startswith("plan")]
    out = _chunks(r, dedupe=True)
    kept = [x for x in out if x["concept_id"].startswith("plan")]
    assert [x["concept_id"] for x in kept] == copies[:1]
    assert sorted(kept[0]["collapsed"]) == sorted(ids[c] for c in copies[1:])
    assert "other" in [x["concept_id"] for x in out]
    assert [x for x in out if x is not kept[0] and "collapsed" in x] == []


def test_dedupe_fills_the_page(env):
    r, _ = env
    assert len(_chunks(r, dedupe=True, limit=2)) == 2


def test_dedupe_on_hub_path(env):
    r, _ = env
    out = _chunks(r, hub_rerank=True, dedupe=True)
    assert sum(x["concept_id"].startswith("plan") for x in out) == 1


def test_collapse_unit_threshold_and_missing_embeddings(env):
    r, ids = env
    eng = r.search_engine
    rows = [{"chunk_id": ids[c], "id": c} for c in ("plan-v3", "other", "plan")]
    out = eng.collapse_near_duplicates(rows)
    assert [x["id"] for x in out] == ["plan-v3", "other"]
    assert out[0]["collapsed"] == [ids["plan"]]
    # Unknown ids (no stored embedding) never collapse.
    ghost = [{"chunk_id": "nope", "id": "a"}, {"chunk_id": "nope2", "id": "b"}]
    assert eng.collapse_near_duplicates(ghost) == ghost
    # Threshold above 1: nothing is a duplicate.
    rows = [{"chunk_id": ids[c], "id": c} for c in ("plan", "plan-v2")]
    assert len(eng.collapse_near_duplicates(rows, threshold=1.01)) == 2


@pytest.mark.parametrize("kw,needle", [
    ({"dedupe": True}, "ignores: dedupe"),
    ({"dedupe": True, "rank": "ppr"}, "ignores: dedupe"),
    ({"dedupe": True, "target": "images"}, "ignores: dedupe"),
    ({"dedupe": True, "target": "chunks", "expand": True}, "ignores: dedupe"),
    ({"dedupe": 1, "target": "chunks"}, "dedupe must be"),
])
def test_dedupe_refused_elsewhere(env, kw, needle):
    r, _ = env
    with pytest.raises(OKFError) as exc:
        r.search("hash", **kw)
    assert exc.value.code == "BAD_VALUE" and needle in str(exc.value)


def test_cli_and_mcp(capsys):
    from okfgraph.cli import _render_chunks
    from okfgraph.mcp_server import create_mcp_server
    from okfgraph.settings import Settings

    assert build_parser().parse_args(
        ["search", "q", "--target", "chunks", "--dedupe"]).dedupe is True
    _render_chunks([{"rrf_score": 0.1, "block_type": "p", "chunk_index": 0,
                     "chunk_id": "a#0", "chunk_text": "x", "collapsed": ["b#0"]}])
    assert "also in: b#0" in capsys.readouterr().out
    mcp = create_mcp_server(Settings(db_path=":memory:"))
    search = next(t for t in mcp._tool_manager.list_tools() if t.name == "search")
    assert search.parameters["properties"]["dedupe"]["default"] is False
