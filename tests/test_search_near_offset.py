"""``near_id`` + ``hops`` limit a search to a graph neighbourhood, and
``offset`` pages through one deterministic ranking.

300 filler notes share the query word with a 4-concept neighbourhood
(center → a → b, d → center). The neighbourhood is scored exactly, never
through the index window, so it is never starved.
"""

import hashlib

import pytest

from okfgraph.cli import _search_filters, build_parser
from okfgraph.components import search as S
from okfgraph.errors import OKFError
from okfgraph.router import OKFRouter

FILLERS = 300
DOCS = {
    "center": "Center note about quokka habitats. See [A](a.md).",
    "a": "A note on quokka diet. Next: [B](b.md).",
    "b": "B note on quokka predators.",
    "d": "D note, quokka survey, cites [the center](center.md).",
}


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

    base = tmp_path_factory.mktemp("near")
    kb = base / "kb"
    kb.mkdir()
    for cid, body in DOCS.items():
        (kb / f"{cid}.md").write_text(
            f"---\ntitle: {cid.upper()}\ntype: note\n---\n\n{body}\n", encoding="utf-8")
    for i in range(FILLERS):
        (kb / f"f{i:03d}.md").write_text(
            f"---\ntitle: Filler {i}\ntype: note\n---\n\nquokka filler {i}\n",
            encoding="utf-8")
    r = OKFRouter(db_path=str(base / "kb.db"), bundle_root=str(kb),
                  embedding_dim=512, device="cpu")
    r.import_bundle()
    yield r
    r.close()
    mp.undo()


def _ids(rows):
    return {r["id"] for r in rows}


def test_near_one_hop_is_the_direct_neighbourhood(router):
    rows = router.search("quokka", near_id="center", hops=1, limit=10)
    assert _ids(rows) == {"center", "a", "d"}


def test_near_defaults_to_two_hops(router):
    rows = router.search("quokka", near_id="center", limit=10)
    assert _ids(rows) == {"center", "a", "b", "d"}


def test_near_never_uses_the_index_window(router, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("near_id must not post-filter the index window")
    monkeypatch.setattr(router.search_engine, "_rrf_candidates", boom)
    rows = router.search("quokka", near_id="b", hops=1, limit=2)
    assert _ids(rows) == {"a", "b"}
    chunks = router.search("quokka", target="chunks", near_id="b", hops=1, limit=10)
    assert {r["concept_id"] for r in chunks} == {"a", "b"}


def test_near_with_ppr_and_hub(router):
    for rank in ("ppr", "hub"):
        rows = router.search("quokka", near_id="center", hops=1, rank=rank, limit=10)
        assert _ids(rows) <= {"center", "a", "d"}, rank
        assert rows, rank


@pytest.mark.parametrize("kw", [
    {}, {"rank": "hub"}, {"rank": "ppr"}, {"target": "chunks"},
    {"near_id": "center"},
])
def test_offset_pages_are_slices_of_one_ranking(router, kw):
    key = "chunk_id" if kw.get("target") == "chunks" else "id"
    whole = [r[key] for r in router.search("quokka", limit=4, **kw)]
    page1 = [r[key] for r in router.search("quokka", limit=2, **kw)]
    page2 = [r[key] for r in router.search("quokka", limit=2, offset=2, **kw)]
    assert page1 + page2 == whole


def test_offset_past_the_end_is_empty(router):
    assert router.search("quokka", near_id="center", hops=1, offset=50) == []


@pytest.mark.parametrize("kw, code", [
    ({"hops": 2 + 1, "near_id": "center"}, "BAD_VALUE"),
    ({"hops": 0, "near_id": "center"}, "BAD_VALUE"),
    ({"hops": 1}, "BAD_VALUE"),
    ({"offset": -1}, "BAD_VALUE"),
    ({"near_id": "nope"}, "UNKNOWN_CONCEPT"),
    ({"near_id": "center", "target": "chunks", "expand": True}, "BAD_VALUE"),
])
def test_bad_near_and_offset_are_refused(router, kw, code):
    if kw.get("hops") == 3:
        kw = {**kw, "hops": S.MAX_HOPS + 1}
    with pytest.raises(OKFError) as e:
        router.search("quokka", **kw)
    assert e.value.code == code


def test_rrf_ties_break_by_id():
    fused = S._rrf_fuse({"b": 0.5, "a": 0.5, "c": 0.9}, {})
    assert [cid for cid, _ in fused] == ["c", "a", "b"]


def test_cli_flags_map_to_search_params():
    args = build_parser().parse_args(
        ["search", "quokka", "--near", "center", "--hops", "1", "--offset", "10"])
    assert _search_filters(args) == {"near_id": "center", "hops": 1, "offset": 10}
