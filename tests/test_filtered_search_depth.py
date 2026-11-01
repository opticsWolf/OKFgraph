"""WS5.4: a filtered search fills ``limit`` even when its matches rank low.

200 concepts, the 3 with ``type: decision`` ranked last by the (stubbed)
index fusion. The unfiltered window (``limit * 3``) never reaches them; the
filtered window is wider and deepens once, so all 3 come back.
"""

import hashlib

import pytest

from okfgraph.components import search as S
from okfgraph.router import OKFRouter

N = 200
DECISIONS = [f"n{i:03d}" for i in range(N - 3, N)]


def _vec(text, dim):
    h = hashlib.sha256((text or "").encode()).digest()
    v = [((h[i % len(h)] / 255.0) - 0.5) for i in range(dim)]
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


@pytest.fixture(scope="module")
def router(tmp_path_factory):
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

    base = tmp_path_factory.mktemp("depth")
    kb = base / "kb"
    kb.mkdir()
    for i in range(N):
        cid = f"n{i:03d}"
        kind = "decision" if cid in DECISIONS else "note"
        (kb / f"{cid}.md").write_text(
            f"---\ntitle: Note {i}\ntype: {kind}\n---\n\nquokka note {i}\n",
            encoding="utf-8")
    r = OKFRouter(db_path=str(base / "kb.db"), bundle_root=str(kb),
                  embedding_dim=512, device="cpu")
    r.import_bundle()
    yield r
    r.close()
    mp.undo()


@pytest.fixture
def ranked(router, monkeypatch):
    """Fusion returns every concept (or its chunks) in id order, so the
    decisions rank last; records the ``k`` of each call."""
    calls = []

    def fake(table, vec_index, fts_index, query_vec, query, k):
        calls.append(k)
        if table == "Concept":
            ids = [f"n{i:03d}" for i in range(N)]
        else:
            rows = router.conn.execute(
                "MATCH (ch:Chunk) RETURN ch.id AS id, ch.parent_doc_id AS p "
                "ORDER BY p, ch.chunk_index").rows_as_dict().get_all()
            ids = [r["id"] for r in rows]
        top = ids[:k]
        return [(cid, 1.0 / (60 + i)) for i, cid in enumerate(top)], len(ids) >= k

    monkeypatch.setattr(router.search_engine, "_rrf_candidates", fake)
    return calls


def test_fetch_window_rules():
    assert S.fetch_k(10, filtered=False) == 30
    assert S.fetch_k(3, filtered=True) == 50
    assert S.fetch_k(20, filtered=True) == 200
    assert S.fetch_k(500, filtered=True) == 1500  # never below the unfiltered window
    assert S.deeper_k(50, found=0, limit=3, full=True) == 200
    assert S.deeper_k(50, found=3, limit=3, full=True) is None   # filled
    assert S.deeper_k(50, found=0, limit=3, full=False) is None  # index exhausted
    assert S.deeper_k(400, found=0, limit=3, full=True) == S.FETCH_CAP


def test_concepts_filter_reaches_low_ranked_matches(router, ranked):
    hits = router.search_engine.search_hybrid("quokka", concept_type="decision", limit=3)
    assert sorted(h["id"] for h in hits) == DECISIONS
    assert ranked == [50, 200]  # widened window, then one deepening retry


def test_chunks_filter_reaches_low_ranked_matches(router, ranked):
    hits = router.search_engine.search_chunks("quokka", concept_type="decision", limit=3)
    assert sorted(h["id"] for h in hits) == DECISIONS
    assert len(ranked) == 2 and ranked[0] == 50


def test_unfiltered_search_keeps_the_narrow_window(router, ranked):
    hits = router.search_engine.search_hybrid("quokka", limit=5)
    assert len(hits) == 5
    assert ranked == [15]
