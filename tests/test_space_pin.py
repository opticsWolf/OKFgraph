"""The embedding_space pin (0.13.0): embroider's vector-space identity.

No model, no ORT session: the pin is driven directly on a graph's
connection, and through the router with a stubbed `embroider`.
"""

import logging
import sys
import types

import pytest

from okfgraph.components.embedding import (
    SPACE_META_KEY,
    enforce_precision_pin,
    enforce_space_pin,
    parse_space,
)
from okfgraph.errors import OKFError

MODEL = "google/embeddinggemma-2"
SHA = "a" * 40


def _space(precision="fp32", *, revision=SHA, dim=512, max_length=2048,
           model=MODEL):
    return (f"embroider/{model}@{revision}:{precision}:{dim}:{max_length}:"
            "text_embed:v1")


def _router(tmp_path, name="space.db", **kwargs):
    from okfgraph.router import OKFRouter

    return OKFRouter(db_path=str(tmp_path / name), bundle_root=str(tmp_path),
                     **kwargs)


def _stored(router):
    rows = router.conn.execute(
        f"MATCH (m:MetaText {{key: '{SPACE_META_KEY}'}}) RETURN m.value AS v"
    ).rows_as_dict().get_all()
    return rows[0]["v"] if rows else None


def _add_concept(router):
    router.conn.execute(
        "CREATE (c:Concept {id: 't', embedding: $vec})", {"vec": [0.0] * 512})


# ---- parse_space -------------------------------------------------------------

def test_parse_space_reads_every_field():
    assert parse_space(_space("fp16")) == {
        "model": MODEL, "revision": SHA, "precision": "fp16", "dim": "512",
        "max_length": "2048", "tuning": "text_embed", "contract": "v1"}


@pytest.mark.parametrize("space", [
    None, "", "external", _space(revision="?"),
    "embroider/m@abc:fp32:512", "openai/m@abc:fp32:512:2048:t:v1"])
def test_parse_space_refuses_what_cannot_pin(space):
    assert parse_space(space) is None


# ---- enforce_space_pin -------------------------------------------------------

def test_first_open_records_the_space(tmp_path):
    r = _router(tmp_path)
    try:
        assert enforce_space_pin(r.conn, _space(), adopt=False) == _space()
        assert _stored(r) == _space()
        assert enforce_space_pin(r.conn, _space(), adopt=False) == _space()
    finally:
        r.close()


def test_max_length_never_counts(tmp_path):
    # Stored vectors are never truncated (the chunker caps at max_length).
    r = _router(tmp_path)
    try:
        enforce_space_pin(r.conn, _space(max_length=8192), adopt=False)
        _add_concept(r)
        assert enforce_space_pin(r.conn, _space(max_length=512),
                                 adopt=False) == _space(max_length=8192)
    finally:
        r.close()


def test_other_space_is_refused_with_the_difference(tmp_path):
    r = _router(tmp_path)
    try:
        enforce_space_pin(r.conn, _space("fp16"), adopt=False)
        _add_concept(r)
        with pytest.raises(OKFError, match="another space") as err:
            enforce_space_pin(r.conn, _space("fp32"), adopt=False)
        assert err.value.code == "SPACE_PIN_MISMATCH"
        text = str(err.value)
        assert "pinned precision=fp16, landed fp32" in text
        assert "CUDA" in text
        assert _stored(r) == _space("fp16")  # refusal leaves the pin
        with pytest.raises(OKFError, match="pinned revision="):
            enforce_space_pin(r.conn, _space("fp16", revision="b" * 40),
                              adopt=False)
    finally:
        r.close()


def test_empty_graph_repins(tmp_path):
    r = _router(tmp_path)
    try:
        enforce_space_pin(r.conn, _space("fp16"), adopt=False)
        assert enforce_space_pin(r.conn, _space("fp32"), adopt=False) == _space("fp32")
        assert _stored(r) == _space("fp32")
    finally:
        r.close()


def test_old_graph_adopts_only_when_the_pins_agreed(tmp_path):
    r = _router(tmp_path)
    try:
        _add_concept(r)
        assert enforce_space_pin(r.conn, _space(), adopt=False) is None
        assert _stored(r) is None
        assert enforce_space_pin(r.conn, _space(), adopt=True) == _space()
        assert _stored(r) == _space()
    finally:
        r.close()


def test_session_without_a_space_checks_nothing(tmp_path, caplog):
    r = _router(tmp_path)
    try:
        enforce_space_pin(r.conn, _space(), adopt=False)
        _add_concept(r)
        with caplog.at_level(logging.WARNING):
            assert enforce_space_pin(r.conn, None, adopt=True) == _space()
            assert enforce_space_pin(r.conn, _space(revision="?"),
                                     adopt=True) == _space()
        assert any("can't name its own" in m for m in caplog.messages)
        assert _stored(r) == _space()
    finally:
        r.close()


# ---- through the router's crate lane ----------------------------------------

def _stub_embroider(monkeypatch, space, precision="fp32"):
    class _Session:
        used_cuda = False

        def __init__(self):
            self.precision = precision

        def encode(self, text, task="Document"):
            return [0.0] * 512

        def encode_batch(self, texts, task="Document"):
            return [self.encode(t) for t in texts]

    class _Model:
        def __init__(self, kind, model_id, native, options):
            assert kind == "embed"
            self.native = native
            self.space = space

        def __getattr__(self, name):
            return getattr(self.native, name)

    stub = types.SimpleNamespace(
        JinaV5=types.SimpleNamespace(open=lambda *a, **k: _Session()),
        Tokenizer=types.SimpleNamespace(open=lambda *a, **k: None),
        Model=_Model,
        MAX_LENGTH=8192,
    )
    monkeypatch.setitem(sys.modules, "embroider", stub)
    monkeypatch.setitem(sys.modules, "onnxruntime",
                        types.ModuleType("onnxruntime"))
    monkeypatch.delenv("OKFGRAPH_ALLOW_REPIN", raising=False)


def test_crate_lane_pins_the_session_space(tmp_path, monkeypatch):
    _stub_embroider(monkeypatch, _space("fp32"))
    r = _router(tmp_path, model_id=MODEL, device="cpu")
    try:
        r.embed_engine._encode("hello")
        assert _stored(r) == _space("fp32")
        assert r.pinned_space == _space("fp32")
        info = r.pin_info()
        assert info["pinned"]["space"] == _space("fp32")
        assert info["configured"]["space"] == _space("fp32")
        assert "space" not in info["mismatch"]
        _add_concept(r)
    finally:
        r.close()

    # Same model and precision, another snapshot: refused at first encode.
    _stub_embroider(monkeypatch, _space("fp32", revision="b" * 40))
    r = _router(tmp_path, model_id=MODEL, device="cpu")
    try:
        with pytest.raises(OKFError, match="pinned revision=a+, landed b+"):
            r.embed_engine._encode("hello")
    finally:
        r.close()


def test_pre_space_graph_adopts_on_first_open(tmp_path, monkeypatch):
    _stub_embroider(monkeypatch, _space("fp32"))
    r = _router(tmp_path, model_id=MODEL, device="cpu")
    try:
        # A 0.12 graph: precision and model pinned, vectors stored.
        enforce_precision_pin(r.conn, "fp32")
        _add_concept(r)
        r.embed_engine._encode("hello")
        assert _stored(r) == _space("fp32")
    finally:
        r.close()
