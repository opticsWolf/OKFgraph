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
    monkeypatch.setitem(sys.modules, "okf_embed", stub)

    router = OKFRouter(
        db_path=str(tmp_path / "cold.db"),
        bundle_root=str(tmp_path),
        device="cpu",
    )
    try:
        assert router.encoder.is_loaded is False
        report = router.diagnose()
        assert isinstance(report, dict)
        assert router.encoder.is_loaded is False
    finally:
        router.close()
