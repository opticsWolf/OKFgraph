"""Lazy encoder lifecycle tests — no model download, no ORT session.

Fake factories prove router construction never opens the session, the first
encode opens exactly once (thread-safe), open failures are cached, and token
counting stays on the tokenizer-only path.
"""

import sys
import threading
import types

from okfgraph.components.embedding import LazyRustEncoder


class _FakeEncoder:
    def __init__(self, calls, used_cuda=False):
        self._calls = calls
        self.used_cuda = used_cuda

    def encode(self, text, task="Document"):
        self._calls["encode"] += 1
        return [0.5, 0.5]

    def encode_batch(self, texts, task="Document"):
        self._calls["encode_batch"] += 1
        return [[0.5, 0.5] for _ in texts]


class _FakeTokenizer:
    def __init__(self, calls):
        self._calls = calls

    def count_tokens(self, text):
        self._calls["count"] += 1
        return len(text.split())


def _make_proxy(**overrides):
    calls = {
        "session": 0,
        "tokenizer": 0,
        "encode": 0,
        "encode_batch": 0,
        "count": 0,
        "on_open": 0,
    }

    def session_factory():
        calls["session"] += 1
        return _FakeEncoder(calls)

    def tokenizer_factory():
        calls["tokenizer"] += 1
        return _FakeTokenizer(calls)

    def on_open(encoder):
        calls["on_open"] += 1
        assert isinstance(encoder, _FakeEncoder)

    kwargs = {
        "model_id": "test/model",
        "truncate_dim": 2,
        "device": "cpu",
        "session_factory": session_factory,
        "tokenizer_factory": tokenizer_factory,
        "on_open": on_open,
    }
    kwargs.update(overrides)
    return LazyRustEncoder(**kwargs), calls


def test_construction_opens_nothing():
    proxy, calls = _make_proxy()
    assert proxy.is_loaded is False
    assert calls == {
        "session": 0,
        "tokenizer": 0,
        "encode": 0,
        "encode_batch": 0,
        "count": 0,
        "on_open": 0,
    }
    assert proxy.model_id == "test/model"
    assert proxy.dim == 2
    assert "cold" in repr(proxy)


def test_first_encode_opens_once():
    proxy, calls = _make_proxy()
    assert proxy.encode("hello") == [0.5, 0.5]
    assert proxy.encode("world", task="Query") == [0.5, 0.5]
    assert proxy.encode_batch(["a", "b"]) == [[0.5, 0.5], [0.5, 0.5]]
    assert calls["session"] == 1
    assert calls["on_open"] == 1
    assert proxy.is_loaded is True
    assert "loaded" in repr(proxy)


def test_used_cuda_triggers_open():
    proxy, calls = _make_proxy()
    assert proxy.used_cuda is False
    assert calls["session"] == 1


def test_count_tokens_never_opens_session():
    proxy, calls = _make_proxy()
    assert proxy.count_tokens("one two three") == 3
    assert proxy.count_tokens("one two three") == 3
    assert calls["count"] == 2
    assert calls["tokenizer"] == 1
    assert calls["session"] == 0
    assert proxy.is_loaded is False


def test_session_failure_is_cached():
    def boom():
        raise RuntimeError("no model here")

    proxy, _ = _make_proxy(session_factory=boom)
    for _ in range(2):
        try:
            proxy.encode("x")
        except RuntimeError as exc:
            assert "no model here" in str(exc)
        else:
            raise AssertionError("encode should have raised")
    assert proxy.is_loaded is False


def test_session_factory_called_once_on_failure():
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise RuntimeError("flaky backend")

    proxy, _ = _make_proxy(session_factory=boom)
    for _ in range(3):
        try:
            proxy.encode_batch(["x"])
        except RuntimeError:
            pass
    assert calls["n"] == 1


def test_concurrent_first_encode_opens_once():
    import time

    inner_calls = {"session": 0}

    def slow_factory():
        inner_calls["session"] += 1
        time.sleep(0.05)
        return _FakeEncoder({"encode": 0, "encode_batch": 0})

    proxy, _ = _make_proxy(session_factory=slow_factory)
    errors = []

    def worker():
        try:
            proxy.encode("race")
        except Exception as exc:  # pragma: no cover - safety net
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert inner_calls["session"] == 1
    assert proxy.is_loaded is True


def test_router_construction_stays_cold(tmp_path, monkeypatch):
    """OKFRouter builds fully (DB + schema + managers) without opening the
    session — and model-free maintenance works while the session is booby-
    trapped."""
    from okfgraph.router import OKFRouter

    def boom(*args, **kwargs):
        raise AssertionError("session/tokenizer must stay cold")

    stub = types.SimpleNamespace(
        JinaV5=types.SimpleNamespace(open=boom),
        JinaTokenizer=types.SimpleNamespace(open=boom),
        MAX_LENGTH=8192,
    )
    monkeypatch.setitem(sys.modules, "embroider", stub)

    router = OKFRouter(
        db_path=str(tmp_path / "cold.db"),
        bundle_root=str(tmp_path),
        device="cpu",
    )
    try:
        assert router.encoder.is_loaded is False
        report = router.doctor(stale_days=365)["report"]
        assert isinstance(report, dict)
        assert router.encoder.is_loaded is False
    finally:
        router.close()


class _RustPanic(BaseException):
    """Stands in for pyo3 PanicException: derives from BaseException, not
    Exception (verified: a real ORT-load panic propagates straight through
    `except Exception`)."""


def test_base_exception_failure_is_cached():
    """A Rust-side open failure (BaseException) is cached like any error —
    it must not retry the poisoned ORT init on every encode."""
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise _RustPanic("Failed to load ONNX Runtime dylib")

    proxy, _ = _make_proxy(session_factory=boom)
    for _ in range(2):
        try:
            proxy.encode("x")
        except _RustPanic:
            pass
        else:
            raise AssertionError("encode should have raised")
    assert calls["n"] == 1


def test_keyboard_interrupt_not_cached():
    """Cancellations pass through and are never cached as open failures."""
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise KeyboardInterrupt

    proxy, _ = _make_proxy(session_factory=boom)
    for _ in range(2):
        try:
            proxy.encode("x")
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("encode should have raised")
    assert calls["n"] == 2


def test_no_ort_fails_before_backend(tmp_path, monkeypatch):
    """Step 3 gate: with no ORT installed, the first encode fails fast with
    the install hint without ever touching the backend (a failed ORT init
    poisons ort's global lock and aborts the process at teardown)."""
    import builtins
    from okfgraph.router import OKFRouter

    real_import = builtins.__import__

    def _no_ort(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_ort)
    monkeypatch.delenv("ORT_DYLIB_PATH", raising=False)

    calls = {"open": 0}

    def panic_open(*args, **kwargs):
        calls["open"] += 1
        raise _RustPanic("Failed to load ONNX Runtime dylib")

    stub = types.SimpleNamespace(
        JinaV5=types.SimpleNamespace(open=panic_open),
        JinaTokenizer=types.SimpleNamespace(open=panic_open),
        MAX_LENGTH=8192,
    )
    monkeypatch.setitem(sys.modules, "embroider", stub)

    router = OKFRouter(
        db_path=str(tmp_path / "hint.db"),
        bundle_root=str(tmp_path),
        device="cpu",
    )
    try:
        assert router.ort_dylib is None
        try:
            router.encoder.encode("hello")
        except RuntimeError as exc:
            assert "okfgraph[cpu]" in str(exc) and "okfgraph[gpu]" in str(exc)
            assert "ORT_DYLIB_PATH" in str(exc)
        else:
            raise AssertionError("encode should have raised the install hint")
        assert calls["open"] == 0, "backend must never be touched without ORT"
    finally:
        router.close()
