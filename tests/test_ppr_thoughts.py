"""WS5.1 (R31-C1): ``--rank ppr`` surfaces a thought that matches the query.

Probe: 30 md notes linked in a ring, all matching the query weakly, and one
thought matching it exactly. The thought is the top lexical seed; it has no
links. Before the fix PPR spread its mass into the ring and it fell out of
the top 10; now it keeps its seed mass and ranks first.
"""

import hashlib

import pytest

from okfgraph.components.ranking import seeds
from okfgraph.router import OKFRouter

QUERY = "ladybug storage upgrade one-way"


def _vec(text, dim):
    h = hashlib.sha256((text or "").encode()).digest()
    v = [((h[i % len(h)] / 255.0) - 0.5) for i in range(dim)]
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


@pytest.fixture
def router(tmp_path, monkeypatch):
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
    for i in range(30):
        (kb / f"m{i:02d}.md").write_text(
            f"---\ntitle: Ladybug storage note {i}\n---\n\n"
            f"About ladybug storage. See [next](m{(i + 1) % 30:02d}.md).\n",
            encoding="utf-8")
    r = OKFRouter(db_path=str(tmp_path / "kb.db"), bundle_root=str(kb),
                  embedding_dim=512, device="cpu")
    r.import_bundle()
    yield r
    r.close()


def test_ppr_ranks_a_matching_thought_first(router):
    tid = router.ingest(
        kind="thoughts", topic="crash",
        thoughts="The ladybug storage upgrade is one-way: version 49 cannot "
                 "be read by 0.21.2.")["concept_id"]
    # The thought is in the seed corpus and is the strongest lexical seed.
    top_seed = seeds(router.search_engine._seed_corpus(), QUERY)[0][0]
    assert top_seed == tid
    hits = router.search_engine.search_with_ppr(QUERY, limit=10)
    assert hits[0]["id"] == tid
