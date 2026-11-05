# -*- coding: utf-8 -*-
"""Reembed op tests (plan-embedder-loader W3, offline).

Everything runs on throwaway Ladybug files in tmp_path against the
stdlib stub embeddings server — no crate, no GPU, no network beyond
loopback. Contracts under test: ``OKFRouter.reembed`` (dry_run reports
the plan without writing), execute (+ the re-pin of the
``embedding_model`` MetaText row when the active engine's model id
differs, with both vector searches still answering), and the
single-tenant refusal when the graph carries ImageAsset rows.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from okfgraph import OKFRouter


class _RHandler(BaseHTTPRequestHandler):
    """Loopback /embeddings stub: deterministic unit vectors keyed by the
    exact (prefixed) input text."""

    native_dim = 512
    requests_captured = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        _RHandler.requests_captured.append(body)
        if self.path.endswith("/embeddings"):
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            rows = [{"embedding": _det(t, self.native_dim)} for t in inputs]
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
    v = [((sum(map(ord, text)) + i * 41) % 1000 - 500) / 500.0 for i in range(dim)]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


SERVER = None
PORT = None


def setup_module(module):
    global SERVER, PORT
    _RHandler.requests_captured = []
    SERVER = ThreadingHTTPServer(("127.0.0.1", 0), _RHandler)
    PORT = SERVER.server_address[1]
    threading.Thread(target=SERVER.serve_forever, daemon=True).start()


def teardown_module(module):
    if SERVER is not None:
        SERVER.shutdown()


MD_ONE = "# alpha note\ncollision geometry and hit-testing notes for alpha."
MD_TWO = "# beta note\nunrelated content about tidal forces in beta."


def _router_files(tmp_path, model_id, **kw):
    """Fresh graph; ingest two small md files (returns the router)."""
    files = {}
    for name, text in (("a.md", MD_ONE), ("b.md", MD_TWO)):
        p = tmp_path / name
        p.write_text(text, encoding="utf-8")
        files[name] = p
    r = OKFRouter(
        db_path=str(tmp_path / "graph.db"),
        model_id=model_id,
        embedding_dim=512,
        embed_backend="external",
        endpoint_url=f"http://127.0.0.1:{PORT}",
        embed_family="nomic-v1.5",
        embed_prefixes="nomic-v1.5",
        embed_counter="heuristic",
        enable_chunking=True,
        chunk_size=200,
        **kw,
    )
    for name, p in files.items():
        r.ingest("md", md_path=str(p))
    return r


def _pin(router):
    from okfgraph.components.embedding import MODEL_META_KEY
    rows = router.conn.execute(
        f"MATCH (m:MetaText {{key: '{MODEL_META_KEY}'}}) RETURN m.value AS v"
    ).rows_as_dict().get_all()
    return rows[0]["v"] if rows else None





def test_dry_run_reports_plan_without_writing(tmp_path):
    r = _router_files(tmp_path, "stub/nomic-a")
    before = r.get_by_id("b") or {}
    pinned = _pin(r)
    plan = r.reembed(dry_run=True)
    assert plan["dry_run"] is True
    assert plan["concepts"] == 2
    assert plan["chunks"] >= 1
    assert plan["dimension"] == 512
    assert _pin(r) == pinned                 # nothing re-pinned
    assert (r.get_by_id("b") or {}) == before


def test_execute_reembeds_keeps_pin_until_model_changes(tmp_path):
    r = _router_files(tmp_path, "stub/old-weights")
    assert _pin(r) == "stub/old-weights"
    assert r.search("tidal forces", limit=2)          # pre-switch sanity
    out = r.reembed()
    assert out["reembedded"]["concepts"] == 2
    assert out["reembedded"]["chunks"] >= 1
    assert isinstance(out["seconds"], float)
    assert _pin(r) == "stub/old-weights"              # same engine, same pin

    # settings update simulation: reopen the SAME file with the new model
    # id and the reembed escape (a pinned graph opened under a different
    # model on purpose, because a reembed is about to run) — otherwise
    # the boot refuses with MODEL_PIN_MISMATCH and reembed is unreachable.
    import os
    os.environ["OKFGRAPH_ALLOW_REPIN"] = "1"
    r.close()
    r2 = OKFRouter(
        db_path=str(tmp_path / "graph.db"),
        model_id="stub/new-weights",
        embedding_dim=512,
        embed_backend="external",
        endpoint_url=f"http://127.0.0.1:{PORT}",
        embed_family="nomic-v1.5",
        embed_prefixes="nomic-v1.5",
        embed_counter="heuristic",
    )
    os.environ.pop("OKFGRAPH_ALLOW_REPIN", None)
    assert _pin(r2) == "stub/old-weights"             # still the old pin
    report = r2.reembed()
    assert report["model_id"] == "stub/new-weights"
    os.environ.pop("OKFGRAPH_ALLOW_REPIN", None)
    assert _pin(r2) == "stub/new-weights"
    hits = r2.search("tidal forces", limit=3)
    assert hits and any((h.get("id") or "") == "b" for h in hits)


def test_pin_info_reports_pin_and_mismatch(tmp_path):
    """The node-facing pins flow: pin_info names what is pinned, flags a
    configured-model mismatch via the re-embed escape, and clears after a
    reembed re-pins the graph."""
    import os
    r = _router_files(tmp_path, "stub/model-a")
    info = r.pin_info()
    assert info["pinned"]["model_id"] == "stub/model-a"
    assert info["pinned"]["embedding_dim"] == 512
    assert info["mismatch"] == []
    assert info["concepts"] >= 1

    r.close()
    os.environ["OKFGRAPH_ALLOW_REPIN"] = "1"
    try:
        r2 = OKFRouter(
            db_path=str(tmp_path / "graph.db"),
            model_id="stub/model-b",
            embedding_dim=512,
            embed_backend="external",
            endpoint_url=f"http://127.0.0.1:{PORT}",
            embed_family="nomic-v1.5",
            embed_prefixes="nomic-v1.5",
            embed_counter="heuristic",
        )
    finally:
        os.environ.pop("OKFGRAPH_ALLOW_REPIN", None)
    info = r2.pin_info()
    assert info["pinned"]["model_id"] == "stub/model-a"
    assert info["mismatch"] == ["model_id"]
    assert "embedding_dim" not in info["mismatch"]    # width never re-pins
    report = r2.reembed()
    assert report["pins_after"]["mismatch"] == []      # re-embed re-pins
    assert r2.pin_info()["mismatch"] == []


def test_refuses_graph_with_image_rows(tmp_path):
    r = _router_files(tmp_path, "stub/nomic-a")
    # add an ImageAsset row directly — the schema lane created the table
    # at router boot (no vision engine involved)
    r.conn.execute(
        "CREATE (:ImageAsset {id: 'i1', file_name: 'x.png', mime_type: 'image/png'})")
    with pytest.raises(Exception) as ei:
        r.reembed()
    assert "ImageAsset" in str(ei.value) or "image" in str(ei.value).lower()
