# -*- coding: utf-8 -*-
"""Embed-op tests (plan-embedder-loader W5, offline).

Same shape as test_reembed.py: a stdlib /embeddings stub on loopback and
throwaway Ladybug files. Contracts under test for ``OKFRouter.embed``
(the inverse lane — Silk's ledger riding the graph's pinned encoder):
the prefixed Document/Query wire, the per-call Matryoshka dim
(slice + renormalise on both an honouring and an ignoring stub), the
provenance fields (renormalised / honours_dimensions / counter), and
the BAD_VALUE refusals (empty/blank texts, caps, task, ladder).
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from okfgraph import OKFRouter


class _EHandler(BaseHTTPRequestHandler):
    native_dim = 768
    honour_dimensions = True
    requests_captured = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        _EHandler.requests_captured.append(body)
        if self.path.endswith("/embeddings"):
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            honour = body.get("dimensions") if _EHandler.honour_dimensions else None
            dims = int(honour) if honour is not None else self.native_dim
            rows = [{"embedding": _det(t, dims)} for t in inputs]
            payload = json.dumps({"data": rows}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass


def _det(text: str, dim: int):
    v = [((sum(map(ord, text)) + i * 43) % 1000 - 500) / 500.0 for i in range(dim)]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


SERVER = None
PORT = None


def setup_module(module):
    global SERVER, PORT
    _EHandler.requests_captured = []
    _EHandler.honour_dimensions = True
    SERVER = ThreadingHTTPServer(("127.0.0.1", 0), _EHandler)
    PORT = SERVER.server_address[1]
    threading.Thread(target=SERVER.serve_forever, daemon=True).start()


def teardown_module(module):
    if SERVER is not None:
        SERVER.shutdown()


def _router(tmp_path):
    return OKFRouter(
        db_path=str(tmp_path / "graph.db"),
        model_id="stub/nomic-a",
        embedding_dim=512,
        embed_backend="external",
        endpoint_url=f"http://127.0.0.1:{PORT}",
        embed_family="nomic-v1.5",       # native 768, ladder ...512 768
        embed_prefixes="nomic-v1.5",
        embed_counter="heuristic",
    )


def test_embed_sends_prefixed_task_and_returns_vectors(tmp_path):
    r = _router(tmp_path)
    out = r.embed(texts=["ledger line one", "ledger line two"],
                  task="Document")
    assert len(out["vectors"]) == 2
    assert out["dim"] == 512
    assert out["model_id"] == "stub/nomic-a"
    assert out["renormalised"] is False
    assert out["honours_dimensions"] is True
    assert out["counter"] == "heuristic"
    sent = [r2 for b in _EHandler.requests_captured for r2 in b["input"]]
    assert sent[0] == "search_document: ledger line one"
    assert sent[1] == "search_document: ledger line two"
    vec = out["vectors"][0]
    assert len(vec) == 512
    assert abs(sum(x * x for x in vec) - 1.0) < 1e-6   # unit norm


def test_embed_query_task_uses_query_prefix(tmp_path):
    r = _router(tmp_path)
    r.embed(texts=["find the shard"], task="Query")
    last = _EHandler.requests_captured[-1]["input"][0]
    assert last.startswith("search_query: find the shard")


def test_per_call_dim_slices_and_renormalises(tmp_path):
    r = _router(tmp_path)
    out = r.embed(texts=["compact fingerprint"], task="Document", dim=256)
    assert out["dim"] == 256
    assert out["renormalised"] is True
    assert len(out["vectors"][0]) == 256
    assert abs(sum(x * x for x in out["vectors"][0]) - 1.0) < 1e-6
    # The engine always requests its session dim; the op slices the
    # returned vectors client-side (no server cooperation needed).
    assert _EHandler.requests_captured[-1]["dimensions"] == 512


def test_per_call_dim_when_endpoint_ignores_dimensions(tmp_path):
    _EHandler.honour_dimensions = False
    try:
        r = _router(tmp_path)
        out = r.embed(texts=["ignorant endpoint"], task="Document", dim=256)
        assert out["dim"] == 256
        assert out["renormalised"] is True
        assert out["honours_dimensions"] is False   # the stub fell back
        assert len(out["vectors"][0]) == 256
    finally:
        _EHandler.honour_dimensions = True


def test_embed_reports_the_effective_counter_source(tmp_path):
    """W4: the provenance `counter` is the effective source — a declared
    `server-tokenize` probe that fails (LM Studio / TEI serve no /tokenize)
    reports 'heuristic-fallback', not the wish."""
    r = OKFRouter(
        db_path=str(tmp_path / "graph.db"),
        model_id="stub/nomic-a",
        embedding_dim=512,
        embed_backend="external",
        endpoint_url=f"http://127.0.0.1:{PORT}",
        embed_family="nomic-v1.5",
        embed_prefixes="nomic-v1.5",
        embed_counter="server-tokenize",      # the stub has no /tokenize
    )
    out = r.embed(texts=["degraded counting"], task="Document")
    assert out["counter"] == "heuristic-fallback"
    assert out["honours_dimensions"] is True


def test_budget_fences_hold_under_heuristic_counter(tmp_path):
    """W6 tail: the greedy token bins respect the budget even when the
    counting source is the external engine's heuristic (len/4, floor 8)."""
    r = _router(tmp_path)
    eng = r.encoder                                  # external, heuristic
    texts = [("token %d " % i) * (1 + (i * 37) % 60) for i in range(50)]
    counts = [eng.count_tokens(t) for t in texts]
    assert min(counts) >= 8                          # the declared floor
    from okfgraph.components.import_ import _token_bucketed_encode
    bins = []
    def encode_fn(bucket):
        bins.append(list(bucket))
        return [_det(t, 64) for t in bucket]
    out = _token_bucketed_encode(
        texts, 4, encode_fn, None, 512, eng.count_tokens)
    assert len(out) == len(texts)
    assert all(v is not None for v in out)           # every row embedded
    assert sorted(b for idx in bins for b in idx) == sorted(texts)
    pos = {t: i for i, t in enumerate(texts)}
    for idx in bins:
        assert len(idx) <= 4                         # batch cap
        assert sum(counts[pos[t]] for t in idx) <= 512    # budget fence holds


def test_refusals(tmp_path):
    r = _router(tmp_path)
    with pytest.raises(ValueError):                     # no empty list
        r.embed(texts=[])
    with pytest.raises(ValueError):                     # no blank strings
        r.embed(texts=["ok", "   "])
    with pytest.raises(ValueError):                     # text cap
        r.embed(texts=[f"x{i}" for i in range(r.EMBED_MAX_TEXTS + 1)])
    with pytest.raises(ValueError):                     # unknown task
        r.embed(texts=["t"], task="Frobnicate")
    with pytest.raises(ValueError):                     # off-ladder dim
        r.embed(texts=["t"], dim=100)
    with pytest.raises(ValueError):                     # above session dim
        r.embed(texts=["t"], dim=768)
