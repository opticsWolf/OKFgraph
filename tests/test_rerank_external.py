# -*- coding: utf-8 -*-
"""External rerank engine tests (plan-embedder-loader W9 + 0.12.0, offline).

A stdlib HTTP server stands in for a llama.cpp ``--rerank`` host serving
BOTH ``/v1/embeddings`` (so the graph's embed lane can be external too)
and ``/v1/rerank``; no crate, no GPU, no network beyond loopback.
Contracts under test: the rerank wire (scored order + full index
coverage), query grouping in ``score_batch``, the windowing one-shot
warning, the first-failure→``down`` session contract (the search answers
unranked, later searches skip the lane, no take-down of retrieval), the
settings validation around ``[search] rerank_backend='external'``, and
the 0.12.0 hardening: the TEI-shape fallback (bare list, ``texts``, the
host root's ``/rerank`` — probe once, the answering dialect latches) and
the gated ``server-tokenize`` counter (exact counts from the host's
``/tokenize`` for the ambiguous middle band only; texts far from the cap
never pay a call; failures latch to ``heuristic-fallback``).
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from okfgraph import OKFRouter
from okfgraph.components.external_reranker import (
    ExternalRerankEngine,
    ExternalRerankError,
    RerankDownError,
    rerank_engine_from_settings,
)
from okfgraph.settings import Settings


def _det(text: str, dim: int):
    v = [((sum(map(ord, text)) + i * 47) % 1000 - 500) / 500.0 for i in range(dim)]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


class _ZHandler(BaseHTTPRequestHandler):
    """Loopback double endpoint: /v1/embeddings (stub embeds) and
    /v1/rerank (deterministic scores keyed by (query, doc) hashes).
    The 0.12.0 flags drive both dialect modes and the /tokenize route."""
    fail_rerank = False            # 500 every rerank call
    broken_index = False           # non-covering rerank rows
    tei_mode = False               # 404 /v1/rerank; root /rerank = TEI shape
    score_alias = False            # standard rows carry 'score' instead
    tokenize_ok = True             # POST /tokenize answers (False → 500)
    rerank_calls = []              # request bodies
    rerank_paths = []              # request paths (parallel to rerank_calls)
    embed_calls = []
    tokenize_calls = []            # 'content' posts

    def _json(self, status, payload):
        p = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(p)))
        self.end_headers()
        self.wfile.write(p)

    @staticmethod
    def _scores(q, docs, key):
        return [{"index": i,
                 key: ((sum(map(ord, q)) * 7 + sum(map(ord, d))) % 100) / 10.0}
                for i, d in enumerate(docs)]

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if self.path.endswith("/embeddings"):
            _ZHandler.embed_calls.append(body)
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            rows = [{"embedding": _det(t, 512)} for t in inputs]
            self._json(200, {"data": rows})
        elif self.path.endswith("/tokenize"):
            _ZHandler.tokenize_calls.append(body.get("content"))
            if _ZHandler.tokenize_ok:
                # a pseudo-BPE: ~2 chars per token (≠ the len/4 heuristic)
                self._json(200, {"tokens": ["t"] * (len(body.get("content", "")) // 2 + 1)})
            else:
                self._json(500, {"error": "no counting here"})
        elif self.path.endswith("/rerank"):
            _ZHandler.rerank_calls.append(body)
            _ZHandler.rerank_paths.append(self.path)
            if _ZHandler.fail_rerank:
                self._json(500, {"error": "kaboom"})
                return
            if _ZHandler.tei_mode:
                if self.path.endswith("/v1/rerank"):
                    self._json(404, {"error": "no /v1 here"})
                    return
                assert "texts" in body, f"TEI request must speak 'texts': {body}"
                q, docs = body["query"], body["texts"]
                self._json(200, self._scores(q, docs, "score"))   # bare list
            else:
                q, docs = body["query"], body["documents"]
                if _ZHandler.broken_index:
                    rows = [{"index": 0, "relevance_score": 1.0}
                            for _ in docs]
                else:
                    rows = self._scores(q, docs,
                                        "score" if _ZHandler.score_alias
                                        else "relevance_score")
                self._json(200, {"results": rows})
        else:
            self.send_error(404)
            return

    def log_message(self, *args):
        pass


SERVER = None
PORT = None


def setup_module(module):
    global SERVER, PORT
    _ZHandler.rerank_calls = []
    _ZHandler.rerank_paths = []
    _ZHandler.embed_calls = []
    _ZHandler.tokenize_calls = []
    _ZHandler.fail_rerank = False
    _ZHandler.tei_mode = False
    _ZHandler.score_alias = False
    _ZHandler.tokenize_ok = True
    SERVER = ThreadingHTTPServer(("127.0.0.1", 0), _ZHandler)
    PORT = SERVER.server_address[1]
    threading.Thread(target=SERVER.serve_forever, daemon=True).start()


def teardown_module(module):
    if SERVER is not None:
        SERVER.shutdown()


def _engine(model_id="stub/bge-v2-m3"):
    return ExternalRerankEngine(
        endpoint_url=f"http://127.0.0.1:{PORT}",
        model_id=model_id, family="bge-v2-m3", max_tokens=8192)


def _boot(db, *, files=None, fail=False):
    _ZHandler.rerank_calls = []
    _ZHandler.rerank_paths = []
    _ZHandler.embed_calls = []
    _ZHandler.tokenize_calls = []
    _ZHandler.tei_mode = False
    _ZHandler.tokenize_ok = True
    _ZHandler.fail_rerank = fail
    r = OKFRouter(
        db_path=str(db),
        model_id="stub/nomic-a",
        embedding_dim=512,
        embed_backend="external",
        endpoint_url=f"http://127.0.0.1:{PORT}",
        embed_family="nomic-v1.5", embed_prefixes="nomic-v1.5",
        embed_counter="heuristic",
        rerank_backend="external",
        rerank_endpoint_url=f"http://127.0.0.1:{PORT}",
        rerank_family="bge-v2-m3",
        rerank_max_tokens=8192,
    )
    for name, text in (files or {}).items():
        p = db.parent / name
        p.write_text(text, encoding="utf-8")
        r.ingest("md", md_path=str(p))
    _ZHandler.rerank_calls = []     # scope the smoke's rerank assertions
    return r


MD_ONE = "# alpha note\ncollision geometry and hit-testing in alpha."
MD_TWO = "# beta note\ncompletely unrelated tidal forces in beta."


def test_score_batch_groups_by_query_and_preserves_order():
    _ZHandler.rerank_calls = []
    eng = _engine()
    pairs = [("cats", "doc one"), ("dogs", "doc two"),
             ("cats", "doc three")]
    scores = eng.score_batch(pairs)
    # two queries -> two HTTP calls; input order intact
    assert len(scores) == 3
    assert isinstance(scores[0], float)
    assert [c["query"] for c in _ZHandler.rerank_calls] == ["cats", "dogs"]
    # group-2 docs arrive in input order
    assert _ZHandler.rerank_calls[0]["documents"] == ["doc one", "doc three"]
    assert _ZHandler.rerank_calls[1]["documents"] == ["doc two"]
    # paired-by-index scores reconstruct the original pairing
    assert _ZHandler.rerank_calls[0]["top_n"] == 2


def test_score_long_scores_one_pair_and_warns_once_about_windowing():
    _ZHandler.rerank_calls = []
    eng = _engine()
    s = eng.score_long("query", "document text", window=512, stride=256)
    assert isinstance(s, float)
    assert len(_ZHandler.rerank_calls) == 1
    assert _ZHandler.rerank_calls[0]["top_n"] == 1


def test_broken_index_coverage_raises_and_trips_down():
    _ZHandler.rerank_calls = []
    _ZHandler.broken_index = True
    try:
        eng = _engine()
        # a malformed host: the rows do not cover the docs exactly once
        with pytest.raises((ExternalRerankError, RerankDownError)):
            eng.score_batch([("q", "d1"), ("q", "d2")])
        assert eng.down and eng.down_reason
        with pytest.raises(RerankDownError):
            eng.score_batch([("q", "d3")])      # fast path, no HTTP call
    finally:
        _ZHandler.broken_index = False


def test_first_failure_degrades_search_and_disables_session(tmp_path):
    r = _boot(tmp_path / "graph.db",
              files={"a.md": MD_ONE, "b.md": MD_TWO}, fail=True)
    hits = r.search("tidal forces", limit=3, rank="cross")
    assert hits                                  # the search still answers
    assert all("cross_score" not in h for h in hits)
    assert r.reranker.down and r.reranker.down_reason
    calls_after_first = len(_ZHandler.rerank_calls or [])
    hits2 = r.search("alpha geometry", limit=3, rank="cross")
    assert hits2
    assert len(_ZHandler.rerank_calls) == calls_after_first   # skipped now


def test_healthy_external_rerank_orders_with_cross_score(tmp_path):
    r = _boot(tmp_path / "graph.db",
              files={"a.md": MD_ONE, "b.md": MD_TWO}, fail=False)
    hits = r.search("tidal forces", limit=3, rank="cross")
    assert hits and all("cross_score" in h for h in hits)
    assert [h["cross_score"] for h in hits] == \
           sorted((h["cross_score"] for h in hits), reverse=True)


def test_settings_checks_rerank_external():
    s = Settings()
    s.db_path = "t.db"
    s.rerank_backend = "external"
    s.rerank_identity = "declared"     # 'auto' lets an embroider host name it
    errs, _ = s.checks()
    assert any("rerank_endpoint_url" in e for e in errs)
    assert any("rerank_family" in e for e in errs)
    s2 = Settings()
    s2.db_path = "t.db"
    s2.rerank_family = "bge-v2-m3"               # on the crate backend
    errs2, _ = s2.checks()
    assert any("rerank_backend='external'" in e for e in errs2)


def test_rerank_engine_from_settings_factory():
    with pytest.raises(ValueError):
        rerank_engine_from_settings(
            backend="embroider", model_id="m", family="f",
            endpoint_url="http://x")
    eng = rerank_engine_from_settings(
        backend="external", model_id="stub/bge-v2-m3", family="bge-v2-m3",
        endpoint_url=f"http://127.0.0.1:{PORT}", max_tokens=8192)
    assert eng.max_length == 8192
    assert eng.used_cuda is False
    rep = eng.spec_report()
    assert rep["family"] == "bge-v2-m3" and rep["backend"] == "external"


# ── 0.12.0: the TEI-shape fallback and the gated exact counting ───────────

def test_tei_shape_fallback_latches_and_speaks_texts():
    """Root-only TEI host: /v1/rerank 404s, the probe falls back to the
    TEI shape, the dialect latches, later groups skip the /v1 attempt."""
    _ZHandler.tei_mode = True
    _ZHandler.rerank_calls = []
    _ZHandler.rerank_paths = []
    try:
        eng = ExternalRerankEngine(
            endpoint_url=f"http://127.0.0.1:{PORT}",
            model_id="stub/bge-v2-m3", family="bge-v2-m3", max_tokens=8192)
        scores = eng.score_batch([("cats", "doc one"), ("cats", "doc two")])
        assert scores and all(isinstance(s, float) for s in scores)
        assert eng.spec_report()["dialect"] == "tei"
        # one refused /v1 attempt (the probe), the rest plain /rerank
        v1 = [p for p in _ZHandler.rerank_paths if p.endswith("/v1/rerank")]
        root = [p for p in _ZHandler.rerank_paths if p.endswith("/rerank")
                and not p.endswith("/v1/rerank")]
        assert len(v1) == 1 and len(root) == 1
        before = len(_ZHandler.rerank_paths)
        assert eng.score_batch([("dogs", "doc three")])
        assert len(_ZHandler.rerank_paths) == before + 1   # no re-probe
        assert _ZHandler.rerank_paths[-1].endswith("/rerank")
        assert not _ZHandler.rerank_paths[-1].endswith("/v1/rerank")
        assert "texts" in _ZHandler.rerank_calls[-1]
        assert "documents" not in _ZHandler.rerank_calls[-1]
    finally:
        _ZHandler.tei_mode = False


def test_standard_rows_accept_score_alias():
    _ZHandler.score_alias = True
    try:
        eng = _engine()
        scores = eng.score_batch([("cats", "doc one")])
        assert scores and isinstance(scores[0], float)
    finally:
        _ZHandler.score_alias = False


def test_server_tokenize_gate_short_mid_long():
    """The gate: far-from-cap texts pay zero calls; only the ambiguous
    middle band asks the host (probe once, then exact)."""
    _ZHandler.tokenize_calls = []
    eng = ExternalRerankEngine(
        endpoint_url=f"http://127.0.0.1:{PORT}",
        model_id="stub/bge-v2-m3", family="bge-v2-m3",
        max_tokens=512, count="server-tokenize")
    # 80 chars ≤ (512-4)//2 = 254 → guaranteed short, zero calls
    assert eng.count_tokens("a" * 80) == 20
    assert _ZHandler.tokenize_calls == []
    # 300 chars sits in the band → probe + exact count (2 calls)
    mid = eng.count_tokens("b" * 300)
    assert mid == 151                        # the host's pseudo-BPE, exact
    assert len(_ZHandler.tokenize_calls) == 2
    # a second band text skips the probe (1 call)
    assert eng.count_tokens("c" * 400) == 201
    assert len(_ZHandler.tokenize_calls) == 3
    # 700 chars > cap → over-cautiously long, zero calls
    assert eng.count_tokens("d" * 700) == 700
    assert len(_ZHandler.tokenize_calls) == 3
    # the caller can shrink the cap (rerank window): the gate follows
    assert eng.count_tokens("e" * 40, cap=64) == 21
    assert len(_ZHandler.tokenize_calls) == 4


def test_server_tokenize_degrades_loudly_once():
    _ZHandler.tokenize_ok = False
    try:
        eng = ExternalRerankEngine(
            endpoint_url=f"http://127.0.0.1:{PORT}",
            model_id="stub/bge-v2-m3", family="bge-v2-m3",
            max_tokens=512, count="server-tokenize")
        # first band text: the probe fails → heuristic, one latch
        first = eng.count_tokens("b" * 300)
        assert first == 75                    # the len/4 heuristic
        assert eng.counter == "heuristic-fallback"
        calls = len(_ZHandler.tokenize_calls)
        # latched: later band texts never ask again
        assert eng.count_tokens("f" * 350) == 87
        assert len(_ZHandler.tokenize_calls) == calls
        # fast bands are untouched by the latch (and never asked anyway)
        assert eng.count_tokens("a" * 80) == 20
    finally:
        _ZHandler.tokenize_ok = True


def test_rerank_count_rows_and_factory():
    s = Settings()
    s.db_path = "t.db"
    s.rerank_count = "server-tokenize"
    errs, _ = s.checks()
    assert any("rerank_backend='external'" in e for e in errs)
    s2 = Settings()
    s2.db_path = "t.db"
    s2.rerank_backend = "external"
    s2.rerank_endpoint_url = f"http://127.0.0.1:{PORT}"
    s2.rerank_family = "bge-v2-m3"
    s2.rerank_count = "server-tokenize"
    errs2, _ = s2.checks()
    assert not any("rerank_count" in e for e in errs2)
    with pytest.raises(ValueError):
        rerank_engine_from_settings(            # bad counting source
            backend="external", model_id="m", family="f",
            endpoint_url=f"http://127.0.0.1:{PORT}", count="greedy")
    eng = rerank_engine_from_settings(
        backend="external", model_id="stub/bge-v2-m3", family="bge-v2-m3",
        endpoint_url=f"http://127.0.0.1:{PORT}", count="server-tokenize")
    assert eng.counter == "server-tokenize"
    assert eng.spec_report()["count_declared"] == "server-tokenize"
