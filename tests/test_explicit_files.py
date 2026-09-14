"""Explicit local model-file tests — no downloads, no ORT session.

Validation (pairing, existence) fires at router construction. Routing to
`open_files` vs `open` is proven with a stubbed `embroider` module; numeric
equivalence of the two acquisition paths is pinned in test_rust_backend.py.
"""

import sys
import types

import pytest

from okfgraph.router import OKFRouter


def _stub_embed(monkeypatch, calls):
    class _Session:
        def __init__(self, via):
            self.via = via
            self.used_cuda = False
            self.precision = "fp32"  # explicit files always report fp32

        def encode(self, text, task="Document"):
            calls["encode"] += 1
            return [float(len(text))]

        def encode_batch(self, texts, task="Document"):
            return [self.encode(t, task=task) for t in texts]

    class _Tokenizer:
        def __init__(self, via):
            self.via = via

        def count_tokens(self, text):
            return len(text.split())

    stub = types.SimpleNamespace(
        JinaV5=types.SimpleNamespace(
            open=lambda *a, **k: (_ for _ in ()).throw(
                AssertionError(f"unexpected open: {a} {k}")
            ),
            open_files=lambda onnx, tok, **k: (
                calls["open_files"].append((onnx, tok, k)),
                _Session("files"),
            )[1],
        ),
        JinaTokenizer=types.SimpleNamespace(
            open=lambda *a, **k: (_ for _ in ()).throw(
                AssertionError(f"unexpected tokenizer open: {a} {k}")
            ),
            open_files=lambda tok: (
                calls["tok_files"].append(tok),
                _Tokenizer("files"),
            )[1],
        ),
        MAX_LENGTH=8192,
    )
    monkeypatch.setitem(sys.modules, "embroider", stub)
    return stub


def _router(tmp_path, **kwargs):
    return OKFRouter(
        db_path=str(tmp_path / "explicit.db"),
        bundle_root=str(tmp_path),
        device="cpu",
        **kwargs,
    )


def test_paths_must_be_given_together(tmp_path):
    with pytest.raises(ValueError, match="must be given together"):
        _router(tmp_path, model_path="model.onnx")
    with pytest.raises(ValueError, match="must be given together"):
        _router(tmp_path, tokenizer_path="tokenizer.json")


def test_missing_files_fail_fast(tmp_path):
    with pytest.raises(FileNotFoundError, match="model_path not found"):
        _router(
            tmp_path,
            model_path=str(tmp_path / "nope.onnx"),
            tokenizer_path=str(tmp_path / "tok.json"),
        )


def test_encode_routes_to_open_files(tmp_path, monkeypatch):
    calls = {"open_files": [], "tok_files": [], "encode": 0}
    _stub_embed(monkeypatch, calls)
    onnx = tmp_path / "model.onnx"
    tok = tmp_path / "tokenizer.json"
    onnx.write_bytes(b"fake")
    tok.write_text("{}", encoding="utf-8")

    router = _router(tmp_path, model_path=str(onnx), tokenizer_path=str(tok))
    try:
        assert router.encoder.is_loaded is False
        assert router.embed_engine._encode("hi") == [2.0]
        assert router.embed_engine.count_tokens("one two") == 2
    finally:
        router.close()

    assert calls["open_files"] == [
        (str(onnx), str(tok), {"truncate_dim": 512, "device": "cpu",
                               "max_length": 8192, "cpu_arena": False})
    ]
    assert calls["tok_files"] == [str(tok)]
    assert calls["encode"] == 1


def test_default_path_still_uses_open(tmp_path, monkeypatch):
    import embroider as real_embed

    opened = {}

    class _Session:
        used_cuda = False
        precision = "fp32"

        def encode(self, text, task="Document"):
            return [1.0]

        def encode_batch(self, texts, task="Document"):
            return [[1.0] for _ in texts]

    stub = types.SimpleNamespace(
        JinaV5=types.SimpleNamespace(
            open=lambda *a, **k: (opened.setdefault("open", (a, k)), _Session())[1],
            open_files=lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("open_files must not be used by default")
            ),
        ),
        JinaTokenizer=types.SimpleNamespace(
            open=lambda *a, **k: types.SimpleNamespace(
                count_tokens=lambda t: 1
            ),
            open_files=lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("tokenizer open_files must not be used by default")
            ),
        ),
        MAX_LENGTH=real_embed.MAX_LENGTH,
    )
    monkeypatch.setitem(sys.modules, "embroider", stub)

    router = _router(tmp_path)
    try:
        assert router.embed_engine._encode("hi") == [1.0]
    finally:
        router.close()
    assert opened["open"][1]["device"] == "cpu"
