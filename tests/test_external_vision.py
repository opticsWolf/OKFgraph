# -*- coding: utf-8 -*-
"""The external vision lane (0.14.0, plan O3 / step B).

Two stdlib servers: an embroider *vision* host and an embroider *text*
host, so a whole graph can be built and searched in a process where
neither ``onnxruntime`` nor ``embroider`` can be imported — which is the
lane's reason to exist (Weave runs okfgraph that way).

Contracts under test: the settings pair; the identity probe and every
refusal it can raise; base64 of the ORIGINAL bytes (the host owns the
resize); batching; the image space pin; the caption fallback for a
refused image but never for a downed host; and reuse across imports.
"""
from __future__ import annotations

import base64
import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from okfgraph import OKFRouter
from okfgraph.components.embedding import (
    IMAGE_MODEL_META_KEY,
    IMAGE_SPACE_META_KEY,
    SPACE_META_KEY,
)
from okfgraph.components.external_vision import (
    ExternalVisionEngine,
    ExternalVisionError,
    build_external_vision,
)
from okfgraph.errors import OKFError
from okfgraph.settings import Settings

TEXT_MODEL = "owner/text-model"
IMAGE_MODEL = "google/vision-model"
SHA = "d" * 40


def _space(model=TEXT_MODEL, precision="fp32"):
    return f"embroider/{model}@{SHA}:{precision}:512:2048:text_embed:v1"


def _image_space(precision="fp32"):
    return f"embroider/{IMAGE_MODEL}@{SHA}:{precision}:512:0:image_embed:v1"


def _det(payload: bytes, dim: int = 512):
    seed = sum(payload[:512]) if payload else 1
    v = [((seed + i * 37) % 1000 - 500) / 500.0 for i in range(dim)]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


class _Host(BaseHTTPRequestHandler):
    requests: list = []

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
        if self.path.endswith("/images/embeddings"):
            self.images(body)
        elif self.path.endswith("/embeddings"):
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            self._send(200, {"data": [{"embedding": _det(t.encode())} for t in inputs]})
        elif self.path == "/tokenize":
            self._send(200, {"tokens": [0, 1], "count": 4})
        else:
            self._send(404, {"error": "not found"})

    def images(self, body):
        rows = [{"embedding": _det(base64.b64decode(b))} for b in body["image"]]
        self._send(200, {"data": rows})

    def log_message(self, *args):
        pass


class _Vision(_Host):
    requests: list = []
    kind = "vision"
    model_id = IMAGE_MODEL
    precision = "fp32"
    dim = 512
    partner = TEXT_MODEL
    space = _image_space()
    refuse = False

    def identity(self):
        row = {"id": self.model_id, "object": "model", "kind": self.kind,
               "engine": "embroider", "engine_version": "0.15.0",
               "contract": "0.15", "revision": SHA, "device": "cpu",
               "precision": self.precision, "dim": self.dim}
        if self.partner:
            row["text_partner"] = self.partner
        if self.space:
            row["space"] = self.space
        return row

    def images(self, body):
        if self.refuse:
            self._send(400, {"error": "undecodable image"})
            return
        super().images(body)


class _Text(_Host):
    requests: list = []

    def identity(self):
        return {"id": TEXT_MODEL, "object": "model", "kind": "embed",
                "engine": "embroider", "engine_version": "0.15.0",
                "contract": "0.15", "revision": SHA, "device": "cpu",
                "precision": "fp32", "max_length": 2048,
                "family": "text-model", "dim": 512, "native_dim": 512,
                "ladder": [128, 256, 512], "tuning": "text_embed",
                "space": _space()}


class _Foreign(_Host):
    """A non-embroider host: answers /v1/models in its own shape."""

    requests: list = []

    def identity(self):
        return {"id": IMAGE_MODEL, "object": "model", "owned_by": "other"}


SERVERS: dict = {}


def setup_module(module):
    for name, handler in (("vision", _Vision), ("text", _Text),
                          ("foreign", _Foreign)):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        SERVERS[name] = server
        threading.Thread(target=server.serve_forever, daemon=True).start()


def teardown_module(module):
    for server in SERVERS.values():
        server.shutdown()


def _url(name):
    return f"http://127.0.0.1:{SERVERS[name].server_address[1]}"


def _reset():
    for handler in (_Vision, _Text, _Foreign):
        handler.requests = []
    _Vision.kind, _Vision.model_id = "vision", IMAGE_MODEL
    _Vision.precision, _Vision.dim = "fp32", 512
    _Vision.partner, _Vision.space = TEXT_MODEL, _image_space()
    _Vision.refuse = False


@pytest.fixture(autouse=True)
def _clean():
    _reset()
    yield
    _reset()


def _png(size=(320, 200), color=(20, 90, 200)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _router(tmp_path, **kw):
    """A router with BOTH lanes external: no ONNX, no crate, anywhere."""
    kw.setdefault("image_backend", "external")
    kw.setdefault("image_endpoint_url", _url("vision"))
    return OKFRouter(
        db_path=str(tmp_path / "graph.db"), model_id=TEXT_MODEL,
        embedding_dim=512, embed_backend="external",
        endpoint_url=kw.pop("text_url", _url("text")), **kw)


# ── settings ────────────────────────────────────────────────────────────────
def test_settings_need_a_url_for_the_external_lane():
    s = Settings()
    s.db_path = "t.db"
    s.image_backend = "external"
    errs, _ = s.checks()
    assert any("image_endpoint_url" in e for e in errs)

    s2 = Settings()
    s2.db_path = "t.db"
    s2.image_endpoint_url = "http://127.0.0.1:9"
    errs2, _ = s2.checks()
    assert any("image_backend='external'" in e for e in errs2)


def test_router_refuses_a_mixed_pair():
    with pytest.raises(ValueError, match="image_endpoint_url is an external"):
        OKFRouter(db_path=":memory:", image_endpoint_url="http://x:1")
    with pytest.raises(ValueError, match="image_backend"):
        OKFRouter(db_path=":memory:", image_backend="declared")


# ── the probe and its refusals ───────────────────────────────────────────────
def test_foreign_host_is_refused():
    with pytest.raises(ValueError, match="did not describe itself"):
        build_external_vision(endpoint_url=_url("foreign"), embedding_dim=512)


def test_an_embed_host_is_refused():
    _Vision.kind = "embed"
    with pytest.raises(ValueError, match="not an image one"):
        build_external_vision(endpoint_url=_url("vision"), embedding_dim=512)


def test_model_id_must_agree_with_the_host():
    with pytest.raises(ValueError, match="serves google/vision-model"):
        build_external_vision(endpoint_url=_url("vision"), embedding_dim=512,
                              image_model_id="google/other-vision")


def test_precision_must_agree_with_the_host():
    with pytest.raises(ValueError, match="landed fp32"):
        build_external_vision(endpoint_url=_url("vision"), embedding_dim=512,
                              image_precision="fp16")


def test_dim_must_agree_with_the_graph():
    with pytest.raises(ValueError, match="--dim 384"):
        build_external_vision(endpoint_url=_url("vision"), embedding_dim=384)


def test_a_host_without_a_text_partner_is_refused():
    _Vision.partner = ""
    with pytest.raises(OKFError) as excinfo:
        build_external_vision(endpoint_url=_url("vision"), embedding_dim=512)
    assert excinfo.value.code == "IMAGE_PIN_MISMATCH"


def test_engine_surface_matches_the_crate_lane():
    eng = build_external_vision(endpoint_url=_url("vision"),
                                embedding_dim=512)
    assert (eng.model_id, eng.dim, eng.precision) == (IMAGE_MODEL, 512, "fp32")
    assert eng.text_partner == TEXT_MODEL and eng.space == _image_space()
    assert eng.used_cuda is False and eng.is_loaded is True
    assert eng.identity_source == "embroider"
    report = eng.spec_report()
    assert report["backend"] == "external"
    assert report["space"] == _image_space()


# ── the wire ────────────────────────────────────────────────────────────────
def test_posts_base64_of_the_original_bytes():
    eng = build_external_vision(endpoint_url=_url("vision"),
                                embedding_dim=512)
    data = _png()
    vec = eng.encode_image_bytes(data)
    assert len(vec) == 512
    path, body = _Vision.requests[-1]
    assert path == "/v1/images/embeddings"
    assert base64.b64decode(body["image"][0]) == data      # not a resized RGB
    assert body["model"] == IMAGE_MODEL


def test_batches_split_by_count():
    eng = build_external_vision(endpoint_url=_url("vision"),
                                embedding_dim=512)
    png = _png((64, 64))
    vectors = eng.encode_images_bytes([png] * 9)
    assert len(vectors) == 9
    sizes = [len(body["image"]) for _p, body in _Vision.requests]
    assert sizes == [8, 1]


def test_a_short_answer_is_a_refusal():
    class _Short(_Vision):
        def images(self, body):
            self._send(200, {"data": []})

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Short)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        eng = build_external_vision(
            endpoint_url=f"http://127.0.0.1:{server.server_address[1]}",
            embedding_dim=512)
        with pytest.raises(ExternalVisionError, match="answered 0 embeddings"):
            eng.encode_image_bytes(_png())
    finally:
        server.shutdown()


# ── the lane end to end, with no ONNX and no crate importable ────────────────
def test_import_with_images_needs_neither_onnx_nor_embroider(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    monkeypatch.setitem(sys.modules, "embroider", None)
    monkeypatch.setitem(sys.modules, "embroider._native", None)

    doc = tmp_path / "doc.md"
    doc.write_text("# alpha\n\nnotes\n\n![](pic.png)\n", encoding="utf-8")
    (tmp_path / "pic.png").write_bytes(_png())

    r = _router(tmp_path)
    try:
        assert r.encoder.identity_source == "embroider"   # the text host
        cid = r.import_mgr.import_from_okf(doc, mode="omni")
        images = r.image_mgr.list_images(cid)
        assert len(images) == 1 and images[0]["embed_route"] == "vision-onnx"

        stored = {row["key"]: row["v"] for row in r.conn.execute(
            "MATCH (m:MetaText) RETURN m.key AS key, m.value AS v"
        ).rows_as_dict().get_all()}
        assert stored[IMAGE_MODEL_META_KEY] == IMAGE_MODEL
        assert stored[IMAGE_SPACE_META_KEY] == _image_space()
        assert stored[SPACE_META_KEY] == _space()
        # the lane is in the reports
        lane = r.lanes()["image"]
        assert lane["backend"] == "external" and lane["identity"] == "embroider"
        assert lane["space"] == _image_space()
        assert r.pin_info()["image"]["pinned_model"] == IMAGE_MODEL
    finally:
        r.close()


def test_reimport_reuses_the_vectors(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text("# alpha\n\nnotes\n\n![](pic.png)\n", encoding="utf-8")
    (tmp_path / "pic.png").write_bytes(_png())

    r = _router(tmp_path)
    try:
        r.import_mgr.import_from_okf(doc, mode="omni")
        first = len([p for p in _Vision.requests if p[0].endswith("/images/embeddings")])
        assert first == 1
        r.import_mgr.import_from_okf(doc, mode="omni", force=True)
        second = len([p for p in _Vision.requests if p[0].endswith("/images/embeddings")])
        assert second == first          # same hash: no second POST
    finally:
        r.close()


def test_a_downed_host_is_one_typed_error(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text("# alpha\n\n![](a.png)![](b.png)\n", encoding="utf-8")
    (tmp_path / "a.png").write_bytes(_png())
    (tmp_path / "b.png").write_bytes(_png((64, 64)))

    # A dedicated host: this test kills it, and the shared one must live
    # on for the rest of the module.
    dead = ThreadingHTTPServer(("127.0.0.1", 0), _Vision)
    threading.Thread(target=dead.serve_forever, daemon=True).start()
    r = _router(tmp_path, image_endpoint_url=(
        f"http://127.0.0.1:{dead.server_address[1]}"))
    try:
        dead.shutdown()                       # the lane dies mid-import
        dead.server_close()                   # ... and refuses connections
        with pytest.raises(OKFError) as excinfo:
            r.image_mgr._ingest_concept_images(
                "c1", doc.read_text(encoding="utf-8"), tmp_path, "omni")
        assert excinfo.value.code == "EXTERNAL_UNAVAILABLE"
        assert "endpoint" in (excinfo.value.fields or {})
    finally:
        r.close()


def test_a_refused_image_falls_back_to_its_caption(tmp_path):
    _Vision.refuse = True
    doc = tmp_path / "doc.md"
    doc.write_text("# alpha\n\n![](pic.png)\n", encoding="utf-8")
    (tmp_path / "pic.png").write_bytes(_png())

    r = _router(tmp_path)
    try:
        r.conn.execute(
            "MERGE (c:Concept {id: 'c1'}) SET c.title='alpha', c.body=$b",
            {"b": doc.read_text(encoding="utf-8")})
        stats = r.image_mgr._ingest_concept_images(
            "c1", doc.read_text(encoding="utf-8"), tmp_path, "omni")
        assert stats["vision"] == 0 and stats["text"] == 1
        images = r.image_mgr.list_images("c1")
        assert images[0]["embed_route"] == "text"
    finally:
        r.close()


# ── pins ────────────────────────────────────────────────────────────────────
def test_a_graph_pinned_to_another_text_model_refuses_the_vision_host(tmp_path):
    r = _router(tmp_path)
    try:
        r.conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS MetaText (key STRING PRIMARY KEY, "
            "value STRING)")
        r.conn.execute(
            "MERGE (m:MetaText {key: 'embedding_model'}) SET m.value = 'other/model'")
        with pytest.raises(OKFError) as excinfo:
            r.vision_encoder._get_encoder()
        assert excinfo.value.code == "VISION_INCOMPATIBLE"
        assert TEXT_MODEL in str(excinfo.value)
    finally:
        r.close()


def test_the_image_space_pin_refuses_another_space(tmp_path):
    from okfgraph.components.embedding import enforce_image_space_pin

    r = _router(tmp_path)
    try:
        conn = r.conn
        conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS MetaText (key STRING PRIMARY KEY, "
            "value STRING)")
        conn.execute(
            "MERGE (m:MetaText {key: $k}) SET m.value = $v",
            {"k": IMAGE_SPACE_META_KEY,
             "v": _image_space().replace("fp32", "fp16")})
        conn.execute(
            "CREATE (i:ImageAsset {id: 'i1', embed_route: 'vision-onnx'})")
        with pytest.raises(OKFError) as excinfo:
            enforce_image_space_pin(conn, _image_space(), adopt=False)
        assert excinfo.value.code == "IMAGE_PIN_MISMATCH"
        assert "precision" in str(excinfo.value)

        # a graph without vision vectors just re-pins
        conn.execute("MATCH (i:ImageAsset {id: 'i1'}) DELETE i")
        assert enforce_image_space_pin(conn, _image_space(),
                                       adopt=False) == _image_space()
    finally:
        r.close()


def test_doctor_names_the_image_lane(tmp_path):
    r = _router(tmp_path)
    try:
        rules = [row["rule"] for row in r.doctor()["report"]["info"]]
        assert "image_lane" in rules
        assert "embed_lane" in rules and "rerank_lane" in rules
    finally:
        r.close()
