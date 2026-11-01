"""Cross-encoder rerank (0.11.0): opt-in ``rerank=True`` (chunks) and
``rank='cross'`` (concepts) over a stub session — no model weights.

The live test at the bottom runs only with ``OKFGRAPH_LIVE_RERANK=1``.
"""

import hashlib
import os
import threading

import pytest

from okfgraph.cli import build_parser
from okfgraph.components.embedding import LazyReranker
from okfgraph.errors import OKFError
from okfgraph.router import OKFRouter
from okfgraph.settings import Settings


def _vec(text, dim):
    h = hashlib.sha256((text or "").encode()).digest()
    v = [((h[i % len(h)] / 255.0) - 0.5) for i in range(dim)]
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


class FakeSession:
    """Scores a pair by how often the query's words occur in the doc."""

    def __init__(self, max_length=512):
        self.max_length = max_length
        self.used_cuda = False
        self.precision = "fp32"
        self.batches = []
        self.long_docs = []

    @staticmethod
    def _score(query, doc):
        words = query.lower().split()
        return float(sum(doc.lower().split().count(w) for w in words))

    def count_tokens(self, text):
        return len(text.split())

    def score_batch(self, pairs):
        self.batches.append(list(pairs))
        return [self._score(q, d) for q, d in pairs]

    def score_long(self, query, doc, window=None, stride=None):
        self.long_docs.append((doc, window, stride))
        return self._score(query, doc)


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
    filler = " ".join(f"filler{i}" for i in range(120))
    docs = {
        "short": ("Short note", "zebracorn appears once here."),
        # The query word sits past the 500-char cut of the result row.
        "long": ("Long note", filler + " zebracorn zebracorn zebracorn"),
        "other": ("Other note", "nothing relevant, only zebracorn adjacent words."),
    }
    for name, (title, body) in docs.items():
        (kb / f"{name}.md").write_text(
            f"---\ntitle: {title}\ndescription: {title} about things\n---\n\n{body}\n",
            encoding="utf-8")
    r = OKFRouter(db_path=str(tmp_path / "kb.db"), bundle_root=str(kb),
                  embedding_dim=512, device="cpu")
    r.import_bundle(str(kb))
    session = FakeSession()
    calls = []

    def factory():
        calls.append(1)
        return session

    r.reranker._session_factory = factory
    yield r, session, calls
    r.close()


# ── settings ────────────────────────────────────────────────────────────

def test_settings_defaults_and_router_kwargs():
    s = Settings()
    assert s.rerank_model_id == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert (s.rerank_device, s.rerank_precision, s.rerank_top_n) == ("auto", "auto", 30)
    kw = s.router_kwargs()
    assert kw["rerank_top_n"] == 30 and kw["rerank_window"] is None
    assert s.checks()[0] == []


@pytest.mark.parametrize("field,value", [
    ("rerank_top_n", 0), ("rerank_top_n", 201), ("rerank_precision", "fp16"),
    ("rerank_device", "mps"), ("rerank_model_id", "nomodel"),
    ("rerank_window", 8), ("rerank_stride", 0),
])
def test_settings_bounds(field, value):
    errors, _ = Settings(**{field: value}).checks()
    assert any(f"search.{field}" in e for e in errors), errors


def test_settings_stride_bounded_by_window():
    errors, _ = Settings(rerank_window=128, rerank_stride=200).checks()
    assert any("rerank_stride" in e for e in errors)


@pytest.mark.parametrize("kw", [
    {"rerank_device": "tpu"}, {"rerank_precision": "int8"}, {"rerank_top_n": 0},
])
def test_router_refuses_bad_rerank_args(tmp_path, kw):
    with pytest.raises(ValueError):
        OKFRouter(db_path=str(tmp_path / "x.db"), embedding_dim=512, device="cpu", **kw)


# ── lazy session ────────────────────────────────────────────────────────

def test_lazy_session_fails_once_and_caches():
    calls = []

    def factory():
        calls.append(1)
        raise RuntimeError("no weights")

    lazy = LazyReranker(model_id="a/b", device="cpu", session_factory=factory)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="no weights"):
            lazy.score_batch([("q", "d")])
    assert len(calls) == 1 and not lazy.is_loaded


def test_lazy_session_does_not_cache_cancellation():
    calls = []

    def factory():
        calls.append(1)
        if len(calls) == 1:
            raise KeyboardInterrupt
        return FakeSession()

    lazy = LazyReranker(model_id="a/b", device="cpu", session_factory=factory)
    with pytest.raises(KeyboardInterrupt):
        lazy.count_tokens("x")
    assert lazy.count_tokens("x y") == 2 and len(calls) == 2


def test_lazy_session_opens_once_under_threads():
    calls, opened = [], []
    gate = threading.Event()

    def factory():
        calls.append(1)
        gate.wait(1)
        return FakeSession()

    lazy = LazyReranker(model_id="a/b", device="cpu", session_factory=factory,
                        on_open=opened.append)
    threads = [threading.Thread(target=lambda: lazy.score_batch([("q", "q")]))
               for _ in range(4)]
    for t in threads:
        t.start()
    gate.set()
    for t in threads:
        t.join()
    assert len(calls) == 1 and len(opened) == 1 and lazy.is_loaded


def test_router_factory_passes_settings(tmp_path, monkeypatch):
    import embroider

    seen = {}

    def fake_open(model_id, **kw):
        seen.update(kw, model_id=model_id)
        return FakeSession()

    monkeypatch.setattr(embroider.Reranker, "open", staticmethod(fake_open))
    r = OKFRouter(db_path=str(tmp_path / "x.db"), embedding_dim=512, device="cpu",
                  rerank_model_id="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                  rerank_device="cpu", rerank_precision="q8")
    try:
        assert not r.reranker.is_loaded
        r.ort_dylib = r.ort_dylib or "stub"
        assert r.reranker.max_length == 512
        assert seen["model_id"] == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
        assert (seen["device"], seen["precision"]) == ("cpu", "q8")
    finally:
        r.close()


# ── engine ──────────────────────────────────────────────────────────────

def test_plain_search_never_opens_the_reranker(env):
    r, _, calls = env
    r.search("zebracorn", target="chunks")
    r.search("zebracorn")
    assert calls == [] and not r.reranker.is_loaded


def test_chunk_rerank_scores_the_whole_chunk(env):
    r, session, calls = env
    out = r.search("zebracorn", target="chunks", rerank=True)
    assert calls == [1]
    assert out[0]["concept_id"] == "long"
    # The row still shows the cut text; the scored text was whole.
    assert len(out[0]["chunk_text"]) <= 503
    scored = [d for batch in session.batches for _, d in batch]
    assert any(d.startswith("Long note: ") and d.endswith("zebracorn") for d in scored)
    assert all("cross_score" in row and "rrf_score" in row for row in out)
    assert [row["cross_score"] for row in out] == sorted(
        (row["cross_score"] for row in out), reverse=True)


def test_concept_rank_cross(env):
    r, _, calls = env
    out = r.search("zebracorn", rank="cross")
    assert calls == [1]
    assert out[0]["id"] == "long" and "cross_score" in out[0]
    assert "relevance_score" in out[0]


def test_hub_rerank_then_cross(env):
    r, _, _ = env
    out = r.search("zebracorn", target="chunks", hub_rerank=True, rerank=True)
    assert out[0]["concept_id"] == "long"
    assert "hub_score" in out[0] and "cross_score" in out[0]


def test_long_text_goes_through_score_long(env):
    r, session, _ = env
    session.max_length = 32
    r.search_engine.rerank_window = 32
    r.search_engine.rerank_stride = 8
    r.search("zebracorn", target="chunks", rerank=True)
    longs = [d for d, _, _ in session.long_docs]
    assert any(d.startswith("Long note: ") for d in longs)
    assert all((w, s) == (32, 8) for _, w, s in session.long_docs)
    batched = [d for batch in session.batches for _, d in batch]
    assert not any(d.startswith("Long note: ") for d in batched)


def test_top_n_caps_rescored_rows_and_keeps_the_tail(env):
    r, session, _ = env
    rows = [{"id": f"c{i}", "chunk_id": f"x{i}", "chunk_text": t,
             "parent_title": None, "rrf_score": 1.0 - i / 10}
            for i, t in enumerate(["a", "b q", "q q", "q q q"])]
    out = r.search_engine.rerank_with_cross_encoder(rows, "q", top_n=3)
    assert [row["id"] for row in out] == ["c2", "c1", "c0", "c3"]
    assert "cross_score" not in out[3]
    assert sum(len(b) for b in session.batches) == 3


def test_ties_break_by_id(env):
    r, _, _ = env
    rows = [{"id": i, "title": None, "description": "same"} for i in ("b", "a", "c")]
    out = r.search_engine.rerank_with_cross_encoder(rows, "q")
    assert [row["id"] for row in out] == ["a", "b", "c"]


def test_paging_slices_one_reranked_order(env):
    r, _, _ = env
    full = r.search("zebracorn", target="chunks", rerank=True, limit=3)
    page2 = r.search("zebracorn", target="chunks", rerank=True, limit=1, offset=1)
    assert [x["chunk_id"] for x in page2] == [full[1]["chunk_id"]]


# ── dispatch ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kw,needle", [
    ({"rerank": True}, "rank='cross'"),
    ({"target": "chunks", "rank": "cross"}, "rank is concepts-only"),
    ({"target": "chunks", "rerank": True, "recency": 30}, "recency"),
    ({"rank": "cross", "recency": 30}, "recency"),
    ({"target": "chunks", "rerank": True, "expand": True}, "ignores: rerank"),
    ({"target": "images", "rerank": True}, "ignores: rerank"),
    ({"target": "chunks", "rerank": "yes"}, "rerank must be"),
    ({"rank": "cross", "hub_weight": 0.5}, "ignores: hub_weight"),
])
def test_bad_rerank_combinations(env, kw, needle):
    r, _, calls = env
    with pytest.raises(OKFError) as exc:
        r.search("zebracorn", **kw)
    assert exc.value.code == "BAD_VALUE" and needle in str(exc.value)
    assert calls == []


def test_cli_flags():
    args = build_parser().parse_args(
        ["search", "q", "--target", "chunks", "--rerank", "--rerank-top-n", "12"])
    assert args.rerank is True and args.rerank_top_n == 12
    assert build_parser().parse_args(["search", "q", "--rank", "cross"]).rank == "cross"


def test_cli_renders_cross_score(capsys):
    from okfgraph.cli import _render_chunks, _render_concepts

    _render_concepts([{"relevance_score": 0.5, "cross_score": 3.25, "title": "T",
                       "type": "concept", "id": "t"}])
    _render_chunks([{"rrf_score": 0.1, "block_type": "p", "chunk_index": 0,
                     "chunk_id": "c", "chunk_text": "x"}])
    out = capsys.readouterr().out
    assert "cross=3.250" in out and out.count("cross=") == 1


# ── live (downloads the default cross-encoder on first run) ─────────────

@pytest.mark.skipif(os.environ.get("OKFGRAPH_LIVE_RERANK") != "1",
                    reason="set OKFGRAPH_LIVE_RERANK=1 to run against real weights")
def test_live_cross_encoder_orders_relevant_first(tmp_path):
    r = OKFRouter(db_path=str(tmp_path / "live.db"), embedding_dim=512, device="cpu")
    try:
        rows = [
            {"id": "b", "title": "Cooking", "description": "How to bake sourdough bread."},
            {"id": "a", "title": "Hashing", "description":
                "Per-file content hashing detects which files changed since the last import."},
        ]
        out = r.search_engine.rerank_with_cross_encoder(
            rows, "how does the importer detect changed files?")
        assert out[0]["id"] == "a" and out[0]["cross_score"] > out[1]["cross_score"]
    finally:
        r.close()


def test_mcp_search_has_rerank():
    from okfgraph.mcp_server import create_mcp_server

    mcp = create_mcp_server(Settings(db_path=":memory:"))
    search = next(t for t in mcp._tool_manager.list_tools() if t.name == "search")
    props = search.parameters["properties"]
    assert props["rerank"]["default"] is False
    assert "cross" in props["rank"]["enum"]
