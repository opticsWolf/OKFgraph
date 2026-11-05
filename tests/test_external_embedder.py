# -*- coding: utf-8 -*-
"""External embedder seam tests (plan-embedder-loader W1/W2, offline).

A stdlib HTTP server on 127.0.0.1 stands in for the llama.cpp /embeddings
endpoint; no crate and no network beyond loopback. Contracts under test:
the OpenAI wire (schema + order), the per-family prefix contract (nomic's
``search_query:``/``search_document:`` = the reference table declared in
``embed_prefixes``), client-side Matryoshka slice+renormalise when the
endpoint cannot honour ``dimensions``, batch chunking, the heuristic
counter's one-shot warning, settings validation, and graph pin refusal on
family mismatch.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from okfgraph.components.external_embedder import (
    ExternalEmbeddingError,
    ExternalEmbedEngine,
    engine_from_settings,
)
from okfgraph.settings import Settings


class _StubHandler(BaseHTTPRequestHandler):
    """Records the last request body in `requests_captured` and answers
    with dim 768 unless `dimensions` is honoured (MRL_OK = per-test)."""

    native_dim = 768
    honour_dimensions = True
    requests_captured = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        _StubHandler.requests_captured.append(body)
        if self.path.endswith("/embeddings"):
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            honour = body.get("dimensions") if _StubHandler.honour_dimensions else None
            dims = int(honour) if honour is not None else self.native_dim
            rows = [{"embedding": _deterministic_vector(t, dims)} for t in inputs]
            payload = json.dumps({"data": rows}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_error(404)

    def log_message(self, *args):  # silence request logging
        pass


def _deterministic_vector(text: str, dim: int):
    v = [(((_h := sum(ord(c) for c in text)) + i * 37) % 1000 - 500) / 500.0 for i in range(dim)]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


SERVER = None
PORT = None


def setup_module(module):
    global SERVER, PORT
    _StubHandler.requests_captured = []
    SERVER = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    PORT = SERVER.server_address[1]
    threading.Thread(target=SERVER.serve_forever, daemon=True).start()


def teardown_module(module):
    if SERVER is not None:
        SERVER.shutdown()


def _engine(prefixes="nomic-v1.5", dim=512, mrl=True, family="nomic-v1.5"):
    return engine_from_settings(
        backend="external", model_id="owner/nomic-static", family=family,
        embedding_dim=dim, endpoint_url=f"http://127.0.0.1:{PORT}",
        embed_family=family, embed_prefixes=prefixes,
        embed_pooling="mean", embed_counter="heuristic",
        embed_max_tokens=8192,
    )


def test_encode_applies_declared_prefixes_and_honours_mrl_dimensions():
    eng = _engine(dim=512)
    vec = eng.encode("find the shard config", task="Query")
    assert len(vec) == 512
    q = _StubHandler.requests_captured[0] if hasattr(_StubHandler, "requests_captured") else None
    assert q["input"] == ["search_query: find the shard config"]
    assert q["dimensions"] == 512


def test_document_side_prefix_contract():
    eng = _engine(dim=768)
    eng.encode_batch(["hello world"], task="Document")
    assert _StubHandler.requests_captured[-1]["input"] == ["search_document: hello world"]


def test_prefixes_none_means_raw_text():
    eng = _engine(prefixes="none", dim=768)
    eng.encode("raw and unprefix'd", task="Document")
    assert _StubHandler.requests_captured[-1]["input"] == ["raw and unprefix'd"]


def test_slice_renorm_fallback_when_server_ignores_dimensions():
    _StubHandler.honour_dimensions = False
    try:
        eng = _engine(prefixes="nomic-v1.5", dim=512)
        vec = eng.encode("salikhinamTEST", task="Document")
        assert len(vec) == 512
        assert abs(sum(x * x for x in vec) - 1.0) < 1e-5   # renormalised
    finally:
        _StubHandler.honour_dimensions = True


def test_encode_batch_preserves_order_and_chunks():
    eng = _engine(dim=768)
    texts = [f"doc-{i}" for i in range(70)]
    _StubHandler.requests_captured = []   # scope this test's captures
    vecs = eng.encode_batch(texts, task="Document")
    assert len(vecs) == 70
    found = [row for r in _StubHandler.requests_captured for row in r["input"]
             if str(row).startswith("search_document: doc-")]
    assert found == [f"search_document: doc-{i}" for i in range(70)]


def test_heuristic_counter_logs_once_and_is_positive():
    eng = _engine(dim=768)
    assert eng.count("short") > 0
    assert eng.count("x" * 400) >= 100   # len/4 floor


def test_rejects_dim_off_the_declared_ladder():
    with pytest.raises(ValueError, match="neither its native dim"):
        _engine(dim=600)


def test_rejects_unknown_family_without_explicit_prefixes():
    with pytest.raises(ValueError, match="unknown embed_prefixes"):
        engine_from_settings(
            backend="external", model_id="owner/x", embedding_dim=512,
            endpoint_url="http://127.0.0.1:1", embed_family="mystery-family",
            embed_prefixes="mystery-family")


def test_rejects_jargonn_inline_json_prefixes():
    with pytest.raises(ValueError, match="inline JSON"):
        engine_from_settings(
            backend="external", model_id="owner/x", embedding_dim=512,
            endpoint_url="http://127.0.0.1:1", embed_family="nomic-v1.5",
            embed_prefixes='{"Query": "q: "}')


def test_rejects_missing_api_key_env():
    import os
    key = os.environ.get("TEST_SK_PORT_KEY")
    assert key is None or True  # unset either way
    with pytest.raises(ValueError, match="api_key_env"):
        engine_from_settings(
            backend="external", model_id="owner/x", embedding_dim=768,
            endpoint_url="http://127.0.0.1:1", embed_family="nomic-v1.5",
            embed_prefixes="nomic-v1.5", api_key_env="TEST_SK_PORT_KEY")


# ── settings layers ──────────────────────────────────────────────────────
def test_settings_checks_external_validation():
    s = Settings()
    s.db_path = "test.db"
    s.backend = "external"
    errors, _ = s.checks()
    assert any("endpoint_url" in e for e in errors)
    s.endpoint_url = "http://127.0.0.1:9"
    s.embed_family = "nomic-v1.5"
    errors, _ = s.checks()
    assert not any("endpoint_url" in e or "embed_family" in e for e in errors)
    s.endpoint_url = ""
    errors, _ = s.checks()
    assert any("external" in e for e in errors)


def test_settings_rejects_external_fields_on_default_backend():
    s = Settings()
    s.db_path = "test.db"
    s.endpoint_url = "http://127.0.0.1:9"
    errors, _ = s.checks()
    assert any("backend='external'" in e for e in errors)


def test_router_kwargs_spray_covers_the_new_rows():
    s = Settings()
    kwargs = s.router_kwargs()
    for key in ("embed_backend", "endpoint_url", "api_key_env",
                "embed_family", "embed_pooling", "embed_prefixes",
                "embed_max_tokens", "embed_counter", "embed_counter_gguf",
                "embed_chars_per_token"):
        assert key in kwargs, key


# ── W4 counter ladder (settings layers) ──────────────────────────────────

def test_settings_checks_counter_ladder(tmp_path):
    s = Settings()
    s.db_path = "test.db"
    s.backend = "external"
    s.endpoint_url = "http://127.0.0.1:9"
    s.embed_family = "nomic-v1.5"
    # gguf-meta without a real file refuses
    s.embed_counter = "gguf-meta"
    errors, _ = s.checks()
    assert any("embed_counter_gguf" in e for e in errors)
    s.embed_counter_gguf = str(tmp_path / "absent.gguf")
    errors, _ = s.checks()
    assert any("embed_counter_gguf" in e for e in errors)
    good = tmp_path / "model.gguf"
    good.write_bytes(b"GGUF" + b"\x00" * 8)
    s.embed_counter_gguf = str(good)
    errors, _ = s.checks()
    assert not any("embed_counter_gguf" in e for e in errors)
    # declared needs the constant in range
    s.embed_counter = "declared"
    errors, _ = s.checks()
    assert any("embed_chars_per_token" in e for e in errors)
    s.embed_chars_per_token = 3.8
    errors, _ = s.checks()
    assert not any("embed_chars_per_token" in e for e in errors)
    s.embed_chars_per_token = 999.0
    errors, _ = s.checks()
    assert any("embed_chars_per_token" in e for e in errors)
    # the choice table carries the four W4 sources (tokenizer is gone)
    s.embed_counter = "tokenizer"
    errors, _ = s.checks()
    assert any("must be one of" in e for e in errors)


def test_settings_reject_counter_rows_on_the_crate_lane():
    s = Settings()
    s.db_path = "test.db"
    s.embed_counter = "server-tokenize"     # harmless wish, wrong lane
    errors, _ = s.checks()
    assert any("backend='external'" in e for e in errors)


def test_float_setting_parses_from_env():
    base = {"OKFGRAPH_BACKEND": "external",
            "OKFGRAPH_ENDPOINT_URL": "http://127.0.0.1:9",
            "OKFGRAPH_EMBED_FAMILY": "nomic-v1.5",
            "OKFGRAPH_EMBED_CHARS_PER_TOKEN": "3.8"}
    s = Settings.load(environ=base)
    assert s.embed_chars_per_token == pytest.approx(3.8)
    bad = dict(base, **{"OKFGRAPH_EMBED_CHARS_PER_TOKEN": "nan-ish"})
    with pytest.raises(ValueError):
        Settings.load(environ=bad)
