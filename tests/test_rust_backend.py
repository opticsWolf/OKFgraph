"""Rust embedding backend wiring: EmbeddingEngine + okf_embed integration.

Needs okf_embed wheel + mordant. The numeric parity itself lives in
test_parity.py; here we pin the *wiring* (backend branch, token counting,
ORT dylib bootstrap).
"""
import math
import os

import pytest

okf_embed = pytest.importorskip("okf_embed")
pytest.importorskip("mordant")

from okfgraph.components.embedding import EmbeddingEngine, resolve_ort_dylib

MODEL = "jinaai/jina-embeddings-v5-text-small-retrieval"


@pytest.fixture(scope="module")
def encoder():
    resolve_ort_dylib()
    return okf_embed.JinaV5.open(MODEL, truncate_dim=64, device="cpu")


@pytest.fixture(scope="module")
def engine(encoder):
    return EmbeddingEngine(
        encoder, 64, "cpu", None, MODEL,
        "jinaai/jina-embeddings-v5-omni-small-retrieval", None,
        512, 40, True, None,
    )


def test_encode_delegates_to_rust(engine, encoder):
    for task in ("Document", "Query"):
        a = engine._encode("delegation check", task=task)
        b = encoder.encode("delegation check", task=task)
        assert a == b
        assert len(a) == 64
        assert abs(math.sqrt(sum(x * x for x in a)) - 1.0) < 1e-5


def test_encode_batch_delegates(engine):
    vecs = engine._encode_batch(["one", "two", "three"])
    assert len(vecs) == 3 and all(len(v) == 64 for v in vecs)
    assert engine._encode_batch([]) == []


def test_count_tokens_matches_transformers(encoder):
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    for text in ["hello world", "", "Übergröße — 数学 formula", "x" * 500]:
        assert encoder.count_tokens(text) == len(
            tok.encode(text, add_special_tokens=False)
        )


def test_resolve_ort_dylib(monkeypatch):
    monkeypatch.delenv("ORT_DYLIB_PATH", raising=False)
    got = resolve_ort_dylib()
    assert got is None or got.endswith("onnxruntime.dll")
    if got is not None:
        assert os.environ["ORT_DYLIB_PATH"] == got
        assert os.path.exists(got)


def test_import_manager_accepts_counter_kwargs():
    im = pytest.importorskip("okfgraph.components.import_")
    import inspect

    sig = inspect.signature(im.ImportManager.__init__)
    assert "token_counter" in sig.parameters
    assert "context_window" in sig.parameters
    assert sig.parameters["context_window"].default == 8192
    assert "tokenizer" not in sig.parameters
