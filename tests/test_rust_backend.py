"""Rust embedding backend wiring: EmbeddingEngine + okf_embed integration.

Needs okf_embed wheel + mordant. The numeric parity itself lives in
test_parity.py; here we pin the *wiring* (backend branch, token counting,
ORT dylib bootstrap).
"""
import math
import os
from pathlib import Path

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
    assert got is None or got.endswith(("onnxruntime.dll", "libonnxruntime.dylib", ".so")) or ".so." in got
    if got is not None:
        assert os.environ["ORT_DYLIB_PATH"] == got
        assert os.path.exists(got)


def test_open_files_matches_hf_acquisition():
    """Explicit local files produce the same vectors as HF acquisition.

    Uses the already-cached snapshot (no new download): the same bytes must
    yield the same session, proving the air-gapped path is numerically the
    default path."""
    from okfgraph.components.embedding import EmbeddingEngine

    hub = Path(EmbeddingEngine.default_cache_dir()) / "hub"
    cached = hub / ("models--" + MODEL.replace("/", "--")) / "snapshots"
    snapshots = sorted(cached.iterdir())
    assert snapshots, f"model not cached locally: {MODEL}"
    snap = snapshots[-1]
    onnx_path = snap / "onnx" / "model.onnx"
    tok_path = snap / "tokenizer.json"
    assert onnx_path.is_file() and tok_path.is_file()

    via_hub = okf_embed.JinaV5.open(MODEL, truncate_dim=64, device="cpu")
    via_files = okf_embed.JinaV5.open_files(
        str(onnx_path), str(tok_path), truncate_dim=64, device="cpu"
    )
    assert via_files.used_cuda is False
    for text in ["hello world", "Übergröße — 数学 formula"]:
        a = via_hub.encode(text, task="Document")
        b = via_files.encode(text, task="Document")
        assert a == pytest.approx(b, abs=1e-6)
    tok = okf_embed.JinaTokenizer.open_files(str(tok_path))
    assert tok.count_tokens("hello world") == 2


def test_import_manager_accepts_counter_kwargs():
    im = pytest.importorskip("okfgraph.components.import_")
    import inspect

    sig = inspect.signature(im.ImportManager.__init__)
    assert "token_counter" in sig.parameters
    assert "context_window" in sig.parameters
    assert sig.parameters["context_window"].default == 8192
    assert "tokenizer" not in sig.parameters
