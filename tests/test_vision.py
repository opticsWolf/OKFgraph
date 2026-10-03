"""Vision-route tests (0.8.0): refusals, pins, stub-encoder ingest, doctor.

Hermetic by design: the real `JinaV5Vision` session is never opened here.
Compat refusals run against a monkeypatched `available_models` registry;
ingest runs against a stub vision encoder; pin tests run on a cold router
conn. No network, no ORT dylib, no torch.
"""

import base64
import shutil
import tempfile
from pathlib import Path

import pytest

from okfgraph.components import embedding as E
from okfgraph.images import EmbedRoute
from okfgraph.router import OKFRouter

PNG_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

NANO_TEXT = "jinaai/jina-embeddings-v5-text-nano-retrieval"
VISION_ID = "jina-v5-omni-nano-retrieval-vision"

FAKE_REGISTRY = [
    {"id": NANO_TEXT, "text_partner": ""},
    {"id": VISION_ID, "text_partner": NANO_TEXT},
]


@pytest.fixture()
def registry(monkeypatch):
    """embroider wheel with the vision contract (0.3+ shape)."""
    import embroider

    monkeypatch.setattr(embroider, "available_models", lambda: FAKE_REGISTRY)
    return embroider


@pytest.fixture()
def router():
    d = tempfile.mkdtemp()
    r = OKFRouter(
        db_path=str(Path(d) / "v.db"),
        bundle_root=d,
        model_id=NANO_TEXT,  # the vision text_partner: compat passes
        embedding_dim=512,
        enable_chunking=False,
        device="cpu",
    )
    yield r
    r.close()
    shutil.rmtree(d, ignore_errors=True)


class StubVision:
    """Stand-in for LazyVisionEncoder: fixed vectors, no session."""

    model_id = VISION_ID
    _precision_cfg = "auto"
    _device = "cpu"
    is_loaded = False
    used_cuda = False

    def __init__(self):
        self.calls = []

    def encode_image(self, rgb, h, w):
        self.calls.append((bytes(rgb), h, w))
        return [0.5] * 512


# -- compat refusals --------------------------------------------------------
# Direct `enforce_vision_compat` calls: the factory adds only the ORT
# pre-check and the session open, so these run with no runtime at all.

def test_vision_refused_on_text_small_graph(router, registry):
    with pytest.raises(RuntimeError, match="needs a .* graph"):
        E.enforce_vision_compat(
            router.conn,
            text_model_id="jinaai/jina-embeddings-v5-text-small-retrieval",
            embedding_dim=512,
            image_model_id=VISION_ID,
        )


def test_vision_refused_above_native_dim(router, registry):
    with pytest.raises(RuntimeError, match="embedding_dim<=768"):
        E.enforce_vision_compat(
            router.conn,
            text_model_id=NANO_TEXT,
            embedding_dim=1024,
            image_model_id=VISION_ID,
        )


def test_vision_compat_passes_for_partner(router, registry):
    assert E.enforce_vision_compat(
        router.conn,
        text_model_id=NANO_TEXT,
        embedding_dim=512,
        image_model_id=VISION_ID,
    ) == NANO_TEXT


def test_vision_unknown_image_model_refused(registry):
    with pytest.raises(RuntimeError, match="unknown image model"):
        E.vision_text_partner("someone/else")


def test_router_rejects_int8_image_precision():
    d = tempfile.mkdtemp()
    try:
        with pytest.raises(ValueError, match="image_precision"):
            OKFRouter(
                db_path=str(Path(d) / "v.db"),
                bundle_root=d,
                image_precision="int8",
            )
    finally:
        shutil.rmtree(d, ignore_errors=True)


# -- pins -------------------------------------------------------------------

def test_image_model_pin_refuses_second_model(router):
    assert E.enforce_image_model_pin(router.conn, VISION_ID) == VISION_ID
    # Empty graph re-pins silently (adoption).
    assert E.enforce_image_model_pin(router.conn, "other-vision") == "other-vision"


def test_image_model_pin_fires_on_populated_graph(router):
    # Any Concept row (no encodes needed) makes the graph non-empty.
    router.conn.execute("CREATE (c:Concept {id: 'x'})")
    E.enforce_image_model_pin(router.conn, VISION_ID)
    with pytest.raises(RuntimeError, match="pinned to image model"):
        E.enforce_image_model_pin(router.conn, "other-vision")


def test_image_precision_pin_refuses_mix(router):
    router.conn.execute("CREATE (c:Concept {id: 'x'})")
    assert E.enforce_image_precision_pin(router.conn, "fp32") == "fp32"
    with pytest.raises(RuntimeError, match="pinned to image precision"):
        E.enforce_image_precision_pin(router.conn, "fp16")


def test_vision_hash_covers_spec():
    from okfgraph.images import VISION_CONTRACT

    h1 = E.vision_content_hash(VISION_ID, "fp32", b"img")
    assert h1 == E.vision_content_hash(VISION_ID, "fp32", b"img")
    assert h1 != E.vision_content_hash(VISION_ID, "fp16", b"img")
    assert h1 != E.vision_content_hash("other-model", "fp32", b"img")
    assert h1 != E.vision_content_hash(VISION_ID, "fp32", b"other-img")
    assert VISION_CONTRACT == "smart32/min262144/max1310720/bicubic-rgb"


# -- stub-encoder ingest ------------------------------------------------------

def _import_vision_doc(router, monkeypatch):
    """Import one alt-less image in omni mode against the stub encoder."""
    tmp = Path(router.bundle_root)
    (tmp / "pic.png").write_bytes(PNG_1x1)
    (tmp / "doc.md").write_text(
        "---\ntitle: Doc\n---\nSee ![](pic.png).\n", encoding="utf-8"
    )
    stub = StubVision()
    monkeypatch.setattr(router.embed_engine, "vision_encoder", stub)
    # Resize prep bypassed: the stub records whatever it is given.
    monkeypatch.setattr(
        router.image_mgr, "_prepare_vision", lambda img: (b"rgb", 32, 32)
    )
    ids = router.import_mgr.import_from_okf(tmp / "doc.md", mode="omni")
    assert ids
    return stub


def test_omni_mode_ingest_mints_vision_route(router, monkeypatch):
    stub = _import_vision_doc(router, monkeypatch)
    rows = router.conn.execute(
        "MATCH (i:ImageAsset) RETURN i.embed_route AS route, i.caption AS caption"
    ).rows_as_dict().get_all()
    assert len(rows) == 1
    assert rows[0]["route"] == EmbedRoute.VISION.value == "vision-onnx"
    assert rows[0]["caption"] == ""
    assert stub.calls and stub.calls[0][1:] == (32, 32)


def test_text_hash_stable_across_upgrade(router, monkeypatch):
    """Caption rows keep the historical hash — no re-embed churn."""
    from okfgraph.images import fallback_caption

    h = router.image_mgr._content_hash(
        EmbedRoute.TEXT, b"pic.png (image 1 in doc)"
    )
    import hashlib

    want = hashlib.sha256(b"text|pic.png (image 1 in doc)").hexdigest()
    assert h == want
    assert fallback_caption("pic.png", 1, "doc") == "pic.png (image 1 in doc)"


def test_doctor_reports_image_routes(router, monkeypatch):
    _import_vision_doc(router, monkeypatch)
    report = router.diagnose()
    image_info = [i for i in report["info"] if i["rule"] == "image_routes"]
    assert image_info, report["info"]
    assert "vision-onnx=1" in image_info[0]["message"]
