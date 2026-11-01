"""``search(efs=N)`` (0.11.0, B3): HNSW search breadth passed to
``QUERY_VECTOR_INDEX`` as ``efs := $efs``."""

import hashlib

import pytest

from okfgraph.cli import build_parser
from okfgraph.errors import OKFError
from okfgraph.router import OKFRouter
from okfgraph.settings import Settings

N_DOCS = 300


def _vec(text, dim):
    h = hashlib.sha256((text or "").encode()).digest()
    v = [((h[i % len(h)] / 255.0) - 0.5) for i in range(dim)]
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    from okfgraph.components import embedding as E

    mp = pytest.MonkeyPatch()
    dim = lambda self: int(getattr(self, "_truncate_dim", 512) or 512)  # noqa: E731
    mp.setattr(E.LazyRustEncoder, "encode",
               lambda self, text, task="Document": _vec(text, dim(self)))
    mp.setattr(E.LazyRustEncoder, "encode_batch",
               lambda self, texts, task="Document": [_vec(t, dim(self)) for t in texts])
    mp.setattr(E.LazyRustEncoder, "count_tokens",
               lambda self, text: len((text or "").split()))
    mp.setattr(E.LazyRustEncoder, "is_loaded", property(lambda self: True))
    tmp = tmp_path_factory.mktemp("efs")
    kb = tmp / "kb"
    kb.mkdir()
    for i in range(N_DOCS):
        (kb / f"d{i:03d}.md").write_text(
            f"---\ntitle: Doc {i}\n---\n\nNote {i} on topic {i % 17} and item {i * 7}.\n",
            encoding="utf-8")
    r = OKFRouter(db_path=str(tmp / "kb.db"), bundle_root=str(kb),
                  embedding_dim=512, device="cpu")
    r.import_bundle(str(kb))
    yield r
    r.close()
    mp.undo()


@pytest.fixture
def spy(graph, monkeypatch):
    calls = []
    real = graph.search_engine._index_rows

    def record(cypher, params):
        if "QUERY_VECTOR_INDEX" in cypher:
            calls.append((cypher, dict(params)))
        return real(cypher, params)

    monkeypatch.setattr(graph.search_engine, "_index_rows", record)
    return calls


def _efs_seen(calls):
    return [p.get("efs") for c, p in calls if ("efs := $efs" in c) == ("efs" in p)]


@pytest.mark.parametrize("kw", [
    {}, {"rank": "hub"}, {"target": "chunks"}, {"target": "chunks", "hub_rerank": True},
])
def test_efs_reaches_the_vector_query(graph, spy, kw):
    graph.search("topic 3", efs=64, **kw)
    assert spy and _efs_seen(spy) == [64] * len(spy)


def test_no_efs_by_default(graph, spy):
    graph.search("topic 3")
    assert spy and all("efs" not in c and "efs" not in p for c, p in spy)


def test_router_default_and_per_call_override(graph, spy, monkeypatch):
    monkeypatch.setattr(graph.search_engine, "efs", 200)
    graph.search("topic 3")
    graph.search("topic 3", efs=32)
    assert _efs_seen(spy) == [200, 32]


@pytest.mark.parametrize("kw,needle", [
    ({"efs": 15}, "16..1000"),
    ({"efs": 1001}, "16..1000"),
    ({"efs": True}, "16..1000"),
    ({"efs": "64"}, "16..1000"),
    ({"efs": 64, "rank": "ppr"}, "ignores: efs"),
    ({"efs": 64, "target": "images"}, "ignores: efs"),
    ({"efs": 64, "target": "chunks", "expand": True}, "ignores: efs"),
    ({"efs": 64, "near_id": "d001"}, "near_id"),
])
def test_bad_efs(graph, kw, needle):
    with pytest.raises(OKFError) as exc:
        graph.search("topic 3", **kw)
    assert exc.value.code == "BAD_VALUE" and needle in str(exc.value)


def test_higher_efs_never_lowers_recall(graph):
    """Vector-only recall@10 against a brute-force cosine oracle."""
    eng = graph.search_engine
    for q in ("topic 1", "item 70", "note on topic 9", "doc 250", "unrelated words"):
        vec = _vec(q, 512)
        oracle = {row["id"] for row in graph.conn.execute(
            "MATCH (c:Concept) WHERE c.embedding IS NOT NULL RETURN c.id AS id, "
            "array_cosine_similarity(c.embedding, CAST($v AS FLOAT[512])) AS s "
            "ORDER BY s DESC LIMIT 10", {"v": vec}).rows_as_dict().get_all()}
        recalls = []
        for efs in (16, 64, 200, 1000):
            rows = eng._index_rows(
                "CALL QUERY_VECTOR_INDEX('Concept', 'concept_embedding', $vec, $k, "
                "efs := $efs) RETURN node", {"vec": vec, "k": 10, "efs": efs})
            recalls.append(len({r["node"]["id"] for r in rows} & oracle))
        assert recalls == sorted(recalls), (q, recalls)
        assert recalls[-1] == 10, (q, recalls)


def test_settings_and_router_bounds(tmp_path):
    assert Settings().efs is None and Settings(efs=64).router_kwargs()["efs"] == 64
    errors, _ = Settings(efs=8).checks()
    assert any("search.efs" in e for e in errors)
    with pytest.raises(ValueError):
        OKFRouter(db_path=str(tmp_path / "x.db"), embedding_dim=512, device="cpu",
                  efs=5000)


def test_cli_and_mcp():
    from okfgraph.mcp_server import create_mcp_server

    assert build_parser().parse_args(["search", "q", "--efs", "128"]).efs == 128
    mcp = create_mcp_server(Settings(db_path=":memory:"))
    search = next(t for t in mcp._tool_manager.list_tools() if t.name == "search")
    assert "efs" in search.parameters["properties"]
