"""0.11.0 §10.3: ``find_path`` is one ``SHORTEST`` query (B8), and doctor
reports the search indexes (B7)."""

import time

import pytest

from okfgraph.errors import OKFError
from okfgraph.router import OKFRouter

CHAIN = 1000


@pytest.fixture(scope="module")
def router(tmp_path_factory):
    base = tmp_path_factory.mktemp("path")
    r = OKFRouter(db_path=str(base / "kb.db"), bundle_root=str(base),
                  embedding_dim=64, device="cpu")
    r.conn.execute("UNWIND range(0, $n - 1) AS i "
                   "CREATE (:Concept {id: 'n' + CAST(i AS STRING), "
                   "title: 'N' + CAST(i AS STRING), type: 'note'})", {"n": CHAIN})
    r.conn.execute("MATCH (a:Concept), (b:Concept) "
                   "WHERE a.id STARTS WITH 'n' AND b.id STARTS WITH 'n' "
                   "AND CAST(substring(b.id, 2, 10) AS INT64) = "
                   "CAST(substring(a.id, 2, 10) AS INT64) + 1 "
                   "CREATE (a)-[:LINKS_TO {kind: 'mention'}]->(b)")
    # A shortcut through a directory: n0 and n5 share a parent.
    r.conn.execute("CREATE (:Directory {id: 'dir'})")
    r.conn.execute("MATCH (d:Directory {id: 'dir'}), (c:Concept) "
                   "WHERE c.id IN ['n0', 'n5'] CREATE (d)-[:CONTAINS]->(c)")
    r.conn.execute("CREATE (:Concept {id: 'island', title: 'Island', type: 'note'})")
    yield r
    r.close()


def _ids(rows):
    return [x["id"] for x in rows]


def test_shortest_path_in_order(router):
    assert _ids(router.traverse("n0", target="n3")) == ["n0", "n1", "n2", "n3"]
    # Undirected, and directories count as hops.
    assert _ids(router.traverse("n7", target="n0")) == ["n7", "n6", "n5", "dir", "n0"]
    assert router.traverse("n0", target="n3")[1]["title"] == "N1"


def test_no_path_within_the_cap_is_empty(router):
    assert router.traverse("n0", target="n40", max_path_length=6) == []
    assert router.traverse("n0", target="island") == []


def test_unknown_end_is_an_error(router):
    for start, target, field in (("nope", "n1", "start_id"), ("n1", "nope", "target")):
        with pytest.raises(OKFError) as exc:
            router.traverse(start, target=target)
        assert exc.value.code == "UNKNOWN_CONCEPT"
        assert field in (exc.value.fields or {})


def test_long_chain_is_one_fast_query(router):
    t = time.perf_counter()
    rows = router.search_engine.find_path("n100", "n110", max_path_length=10)
    assert _ids(rows) == [f"n{i}" for i in range(100, 111)]
    assert time.perf_counter() - t < 5.0


def test_doctor_lists_indexes_and_flags_a_missing_one(router):
    schema = router.schema_mgr
    if not schema._search_available:
        pytest.skip("vector/fts extensions not loaded")
    schema._build_search_indexes(rebuild=False)
    report = router.doctor()["report"]
    lines = [i["message"] for i in report["info"] if i["rule"] == "search_index"]
    assert any("Concept.concept_embedding: HNSW (embedding; cosine, dim 64)" in m
               for m in lines)
    assert any("Chunk.chunk_fts: FTS" in m for m in lines)
    assert "missing_index" not in report["summary"]
    before = report["score"]

    router.conn.execute("CALL DROP_FTS_INDEX('Chunk', 'chunk_fts')")
    try:
        report = router.doctor()["report"]
        missing = [f for f in report["findings"] if f["rule"] == "missing_index"]
        assert len(missing) == 1 and "Chunk.chunk_fts" in missing[0]["message"]
        # Report-only: the score ignores it.
        assert report["score"] == before
    finally:
        schema._build_search_indexes(rebuild=False)
