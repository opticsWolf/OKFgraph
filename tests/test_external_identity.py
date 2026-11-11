# -*- coding: utf-8 -*-
"""The external embedder's identity probe (0.13.0, offline).

Two stdlib servers on 127.0.0.1: an embroider host (``GET /v1/models``
answers with ``engine: embroider``, ``/tokenize`` with a ``count``) and a
llama.cpp-shaped server (its own ``/v1/models`` row, ``/tokenize`` token
lists). Contracts under test: the probe tells them apart; an embroider
host supplies the spec (family, width, ceiling, precision, space), gets
``task`` and no prefixes, counts exactly, and the graph pins precision and
space; declarations must agree with it; anything else keeps the declared
0.12 contract; and one graph moves between lanes only while the pins hold.
"""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from okfgraph import OKFRouter
from okfgraph.components.embedding import SPACE_META_KEY
from okfgraph.components.external_embedder import (
    ExternalEmbeddingError,
    engine_from_settings,
    probe_identity,
)
from okfgraph.components.external_reranker import rerank_engine_from_settings
from okfgraph.errors import OKFError
from okfgraph.settings import Settings

MODEL = "owner/text-model"
SHA = "c" * 40


def _space(precision="fp32", revision=SHA):
    return f"embroider/{MODEL}@{revision}:{precision}:512:2048:text_embed:v1"


def _det(text: str, dim: int):
    v = [((sum(map(ord, text)) + i * 41) % 1000 - 500) / 500.0 for i in range(dim)]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


class _Host(BaseHTTPRequestHandler):
    """Shared wire; subclasses differ in who they say they are."""

    requests = []

    def _send(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            self._send(200, {"object": "list", "data": [self.identity()]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        type(self).requests.append((self.path, body))
        if self.path.endswith("/embeddings"):
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            self._send(200, {"data": [{"embedding": _det(t, 512)} for t in inputs]})
        elif self.path == "/tokenize":
            self._send(200, self.tokenize(body.get("content", "")))
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, *args):
        pass


class _Embroider(_Host):
    requests = []
    precision = "fp32"
    revision = SHA

    def identity(self):
        return {"id": MODEL, "object": "model", "kind": "embed",
                "engine": "embroider", "engine_version": "0.15.0",
                "contract": "0.15", "revision": self.revision,
                "device": "cpu", "precision": self.precision,
                "max_length": 2048, "family": "text-model", "dim": 512,
                "native_dim": 1024, "ladder": [128, 256, 512, 1024],
                "tuning": "text_embed",
                "space": _space(self.precision, self.revision)}

    def tokenize(self, text):
        # A wrong-length id list: only `count` is the model's own count.
        return {"tokens": [0], "count": 7}


class _Llama(_Host):
    requests = []

    def identity(self):
        return {"id": MODEL, "object": "model", "owned_by": "llamacpp"}

    def tokenize(self, text):
        return {"tokens": list(range(3))}


class _NoModels(_Host):
    requests = []

    def do_GET(self):
        self.send_error(404)


RERANK = "owner/rerank-model"


class _EmbroiderRerank(_Host):
    requests = []

    def identity(self):
        return {"id": RERANK, "object": "model", "kind": "rerank",
                "engine": "embroider", "engine_version": "0.15.0",
                "contract": "0.15", "revision": SHA, "device": "cpu",
                "precision": "fp32", "max_length": 1024,
                "family": "rerank-model"}

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        type(self).requests.append((self.path, body))
        if self.path == "/v1/rerank":
            self._send(200, {"results": [
                {"index": i, "relevance_score": float(len(d))}
                for i, d in enumerate(body["documents"])]})
        elif self.path == "/tokenize":
            self._send(200, {"tokens": [0], "count": 300})
        else:
            self._send(404, {"error": "not found"})


SERVERS = {}


def setup_module(module):
    for name, handler in (("embroider", _Embroider), ("llama", _Llama),
                          ("bare", _NoModels), ("rerank", _EmbroiderRerank)):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        SERVERS[name] = server


def teardown_module(module):
    for server in SERVERS.values():
        server.shutdown()
        server.server_close()


def _url(name):
    return f"http://127.0.0.1:{SERVERS[name].server_address[1]}"


@pytest.fixture(autouse=True)
def _reset():
    _Embroider.precision, _Embroider.revision = "fp32", SHA
    for handler in (_Embroider, _Llama, _NoModels, _EmbroiderRerank):
        handler.requests = []


def _engine(host, **kw):
    kw.setdefault("embedding_dim", 512)
    return engine_from_settings(
        backend="external", model_id=kw.pop("model_id", MODEL),
        endpoint_url=_url(host), **kw)


# ── the probe ────────────────────────────────────────────────────────────────
def test_probe_tells_hosts_apart():
    assert probe_identity(_url("embroider"))["space"] == _space()
    assert probe_identity(_url("llama")) is None
    assert probe_identity(_url("bare")) is None
    with pytest.raises(ExternalEmbeddingError, match="unreachable"):
        probe_identity("http://127.0.0.1:9", timeout=2.0)


# ── an embroider host ────────────────────────────────────────────────────────
def test_embroider_host_supplies_the_spec():
    eng = _engine("embroider")       # no family, no prefixes, no counter
    assert eng.identity_source == "embroider"
    assert eng.family == "text-model"
    assert eng.precision == "fp32" and eng.space == _space()
    assert eng.spec.max_tokens == 2048 and eng.spec.prefixes is None

    eng.encode("find the shard config", task="Query")
    path, body = _Embroider.requests[-1]
    assert body["input"] == ["find the shard config"]     # no prefix
    assert body["task"] == "Query" and "dimensions" not in body
    assert eng.count_tokens("any text") == 7               # the host's count
    report = eng.spec_report()
    assert report["identity"] == "embroider"
    assert report["space"] == _space() and report["counter_effective"] == "server-tokenize"


@pytest.mark.parametrize("kw, match", [
    ({"embed_family": "nomic-v1.5"}, "serves family 'text-model'"),
    ({"embed_prefixes": "nomic-v1.5"}, "drop embed_prefixes"),
    ({"model_id": "owner/other"}, "serves owner/text-model"),
    ({"embedding_dim": 256}, "serves 512-wide"),
])
def test_declarations_must_agree_with_the_host(kw, match):
    with pytest.raises(ValueError, match=match):
        _engine("embroider", **kw)


def test_agreeing_family_is_fine():
    assert _engine("embroider", embed_family="text-model").family == "text-model"


def test_declared_mode_skips_the_probe():
    eng = _engine("embroider", embed_identity="declared",
                  embed_family="text-model")
    assert eng.identity_source == "declared" and eng.space is None
    eng.encode("x", task="Query")
    assert "task" not in _Embroider.requests[-1][1]


# ── anything else: the declared contract ─────────────────────────────────────
def test_other_server_keeps_the_declared_contract():
    with pytest.raises(ValueError, match="embed_family"):
        _engine("llama")
    eng = _engine("llama", embed_family="nomic-v1.5",
                  embed_prefixes="nomic-v1.5", embed_counter="server-tokenize")
    assert eng.identity_source == "declared" and eng.precision == "external"
    eng.encode("q", task="Query")
    body = _Llama.requests[-1][1]
    assert body["input"] == ["search_query: q"] and "task" not in body
    assert eng.count_tokens("q") == 3                       # len(tokens)


def test_unreachable_host_needs_a_family(caplog):
    url = "http://127.0.0.1:9"
    with pytest.raises(ValueError, match="no embed_family is declared"):
        engine_from_settings(backend="external", model_id=MODEL,
                             embedding_dim=512, endpoint_url=url)
    with caplog.at_level(logging.WARNING):
        eng = engine_from_settings(backend="external", model_id=MODEL,
                                   embedding_dim=512, endpoint_url=url,
                                   embed_family="text-model")
    assert eng.identity_source == "declared"
    assert any("identity probe failed" in m for m in caplog.messages)


def test_settings_family_optional_only_when_probing():
    s = Settings()
    s.db_path = "test.db"
    s.backend, s.endpoint_url = "external", "http://127.0.0.1:9"
    errors, _ = s.checks()
    assert not any("embed_family" in e for e in errors)
    s.embed_identity = "declared"
    errors, _ = s.checks()
    assert any("embed_family" in e for e in errors)
    assert Settings().router_kwargs()["embed_identity"] == "auto"


# ── the rerank lane ──────────────────────────────────────────────────────────
def _reranker(host, **kw):
    return rerank_engine_from_settings(
        backend="external", model_id=kw.pop("model_id", RERANK),
        family=kw.pop("family", ""), endpoint_url=_url(host) + "/v1/rerank",
        **kw)


def test_embroider_rerank_host_supplies_family_and_ceiling():
    eng = _reranker("rerank")
    assert eng.identity_source == "embroider"
    assert eng.spec.family == "rerank-model" and eng.max_length == 1024
    assert eng.counter == "server-tokenize"
    assert eng.score_batch([("q", "a"), ("q", "bbb")]) == [1.0, 3.0]
    # The ambiguous band asks the host, which answers its own count.
    assert eng.count_tokens("x" * 700) == 300
    assert eng.spec_report()["identity"] == "embroider"


@pytest.mark.parametrize("host, kw, match", [
    ("rerank", {"family": "bge-v2-m3"}, "serves family 'rerank-model'"),
    ("rerank", {"model_id": "owner/other"}, "serves owner/rerank-model"),
    ("embroider", {"model_id": MODEL}, "not a reranker"),
])
def test_rerank_declarations_must_agree(host, kw, match):
    with pytest.raises(ValueError, match=match):
        _reranker(host, **kw)


def test_rerank_other_server_keeps_the_declared_contract():
    with pytest.raises(ValueError, match="rerank_family"):
        _reranker("llama")
    eng = _reranker("llama", family="bge-v2-m3", max_tokens=4096)
    assert eng.identity_source == "declared" and eng.max_length == 4096
    with pytest.raises(ValueError, match="no rerank_family is declared"):
        rerank_engine_from_settings(
            backend="external", model_id=RERANK, family="",
            endpoint_url="http://127.0.0.1:9")


def test_rerank_settings_family_optional_only_when_probing():
    s = Settings()
    s.db_path = "test.db"
    s.rerank_backend = "external"
    s.rerank_endpoint_url = "http://127.0.0.1:9"
    errors, _ = s.checks()
    assert not any("rerank_family" in e for e in errors)
    s.rerank_identity = "declared"
    errors, _ = s.checks()
    assert any("rerank_family" in e for e in errors)


# ── one graph, several lanes ─────────────────────────────────────────────────
def _router(tmp_path, host, **kw):
    return OKFRouter(db_path=str(tmp_path / "graph.db"), model_id=MODEL,
                     embedding_dim=512, embed_backend="external",
                     endpoint_url=_url(host), **kw)


def _ingest(router, tmp_path):
    p = tmp_path / "a.md"
    p.write_text("# alpha note\ncollision geometry notes.", encoding="utf-8")
    router.ingest("md", md_path=str(p))


def _stored_space(router):
    rows = router.conn.execute(
        f"MATCH (m:MetaText {{key: '{SPACE_META_KEY}'}}) RETURN m.value AS v"
    ).rows_as_dict().get_all()
    return rows[0]["v"] if rows else None


def test_graph_pins_the_host_space_and_refuses_another(tmp_path):
    r = _router(tmp_path, "embroider")
    try:
        _ingest(r, tmp_path)
        assert _stored_space(r) == _space()
        assert r.pin_info()["pinned"]["precision"] == "fp32"
        assert r.search("collision", limit=3)
    finally:
        r.close()

    r = _router(tmp_path, "embroider")      # same space: reopens
    r.close()

    _Embroider.revision = "d" * 40          # another snapshot
    with pytest.raises(OKFError, match="pinned revision=c+, landed d+"):
        _router(tmp_path, "embroider")

    _Embroider.revision, _Embroider.precision = SHA, "fp16"
    with pytest.raises(OKFError, match="precision"):
        _router(tmp_path, "embroider")


def test_declared_graph_is_not_blessed_by_an_embroider_host(tmp_path, caplog):
    # A graph a declared engine filled has no precision pin: the first
    # embroider host to open it must not adopt a space for those vectors.
    r = _router(tmp_path, "llama", embed_family="text-model")
    try:
        _ingest(r, tmp_path)
    finally:
        r.close()
    r = _router(tmp_path, "embroider")
    try:
        assert _stored_space(r) is None
        assert r.pinned_space is None
    finally:
        r.close()


def test_lanes_name_who_serves_each_model(tmp_path):
    r = _router(tmp_path, "embroider", rerank_backend="external",
                rerank_model_id=RERANK,
                rerank_endpoint_url=_url("rerank") + "/v1/rerank")
    try:
        _ingest(r, tmp_path)
        lanes = r.lanes()
        assert lanes["embed"] == {
            "backend": "external", "model_id": MODEL,
            "endpoint": _url("embroider") + "/v1", "identity": "embroider",
            "family": "text-model", "precision": "fp32",
            "space": _space(), "pinned_space": _space(),
        }
        assert lanes["rerank"]["identity"] == "embroider"
        assert lanes["rerank"]["family"] == "rerank-model"
        assert lanes["rerank"]["precision"] == "fp32"
        info = {i["rule"]: i["message"]
                for i in r.doctor()["report"]["info"]}
        assert "identity=embroider" in info["embed_lane"]
        assert f"space={_space()}" in info["embed_lane"]
        assert info["rerank_lane"].startswith(f"{RERANK} via ")
    finally:
        r.close()


def test_lanes_report_declared_and_cold_in_process(tmp_path):
    r = _router(tmp_path, "llama", embed_family="text-model")
    try:
        lanes = r.lanes()
        assert lanes["embed"]["identity"] == "declared"
        assert lanes["embed"]["precision"] is None
        assert lanes["embed"]["space"] is None
        # The default crate reranker stays cold: lanes() never opens it.
        assert lanes["rerank"]["identity"] == "in-process"
        assert lanes["rerank"]["endpoint"] is None
        assert not r.reranker.is_loaded
    finally:
        r.close()
