"""CPU-vs-GPU runtime guidance: GPU probe, doctor note, auto-fallback warning.

All probes are faked: no nvidia-smi call, no ORT session, no model.
"""
import logging
import subprocess
import types

import pytest

from okfgraph.components import embedding as emb

CPU_ORT = {"installed": ["onnxruntime"], "cuda_usable": False,
           "import_error": None, "version": "1.29.0", "dylib_path": None}


@pytest.fixture(autouse=True)
def _fresh_probe_cache():
    emb.nvidia_gpu_name.cache_clear()
    yield
    emb.nvidia_gpu_name.cache_clear()


class TestNvidiaProbe:
    def test_no_nvidia_smi(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: None)
        assert emb.nvidia_gpu_name() is None

    def test_parses_first_gpu(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: "nvidia-smi")
        monkeypatch.setattr(subprocess, "run", lambda *a, **k:
                            types.SimpleNamespace(
                                returncode=0,
                                stdout="NVIDIA GeForce RTX 3090\nOther\n"))
        assert emb.nvidia_gpu_name() == "NVIDIA GeForce RTX 3090"

    def test_failure_is_silent(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: "nvidia-smi")

        def _boom(*a, **k):
            raise subprocess.TimeoutExpired("nvidia-smi", 5)
        monkeypatch.setattr(subprocess, "run", _boom)
        assert emb.nvidia_gpu_name() is None

    def test_nonzero_exit(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: "nvidia-smi")
        monkeypatch.setattr(subprocess, "run", lambda *a, **k:
                            types.SimpleNamespace(returncode=9, stdout=""))
        assert emb.nvidia_gpu_name() is None


class TestHint:
    def test_gpu_plus_cpu_runtime_warns(self, monkeypatch):
        monkeypatch.setattr(emb, "nvidia_gpu_name", lambda: "RTX 3090")
        hint = emb.cpu_runtime_with_gpu_hint(CPU_ORT)
        assert "RTX 3090" in hint and "okfgraph[gpu]" in hint
        assert "CPU/fp32" in hint

    @pytest.mark.parametrize("ort", [
        dict(CPU_ORT, cuda_usable=True),       # already on CUDA
        dict(CPU_ORT, import_error="boom"),    # broken: other hint owns it
    ])
    def test_silent_when_not_applicable(self, monkeypatch, ort):
        monkeypatch.setattr(emb, "nvidia_gpu_name", lambda: "RTX 3090")
        assert emb.cpu_runtime_with_gpu_hint(ort) is None

    def test_silent_without_gpu(self, monkeypatch):
        monkeypatch.setattr(emb, "nvidia_gpu_name", lambda: None)
        assert emb.cpu_runtime_with_gpu_hint(CPU_ORT) is None

    def test_missing_runtime_hint_recommends_gpu(self, monkeypatch):
        missing = {"installed": [], "import_error": "No module"}
        monkeypatch.setattr(emb, "nvidia_gpu_name", lambda: "RTX 3090")
        assert "pick [gpu]" in emb.ort_missing_hint(missing)
        monkeypatch.setattr(emb, "nvidia_gpu_name", lambda: None)
        assert "[cpu] otherwise" in emb.ort_missing_hint(missing)


class _StubResult:
    def rows_as_dict(self):
        return self

    def get_all(self):
        return []


class _StubConn:
    def execute(self, *a, **k):
        return _StubResult()


def test_doctor_notes_unused_gpu_without_scoring(monkeypatch, tmp_path):
    from okfgraph.components.doctor import DoctorManager
    monkeypatch.setattr(emb, "ort_info", lambda: dict(CPU_ORT))
    monkeypatch.setattr(emb, "nvidia_gpu_name", lambda: "RTX 3090")
    hub = tmp_path / "hub"
    hub.mkdir()
    rep = DoctorManager(_StubConn(), object()).diagnose(cache_dir=str(hub))
    by_rule = {i["rule"]: i["message"] for i in rep["info"]}
    assert "RTX 3090" in by_rule["ort_gpu_unused"]
    assert rep["findings"] == [] and rep["score"] == 100


def test_doctor_quiet_without_gpu(monkeypatch, tmp_path):
    from okfgraph.components.doctor import DoctorManager
    monkeypatch.setattr(emb, "ort_info", lambda: dict(CPU_ORT))
    monkeypatch.setattr(emb, "nvidia_gpu_name", lambda: None)
    hub = tmp_path / "hub"
    hub.mkdir()
    rep = DoctorManager(_StubConn(), object()).diagnose(cache_dir=str(hub))
    assert "ort_gpu_unused" not in {i["rule"] for i in rep["info"]}


def test_auto_fallback_logs_warning(monkeypatch, tmp_path, caplog):
    """A real router open on this (CPU-runtime) venv with a faked GPU."""
    if emb.ort_info().get("cuda_usable"):
        pytest.skip("CUDA runtime installed: auto lands on GPU")
    from okfgraph.router import OKFRouter
    monkeypatch.setattr(emb, "nvidia_gpu_name", lambda: "RTX 3090")
    r = OKFRouter(db_path=str(tmp_path / "g.db"), device="auto")
    try:
        with caplog.at_level(logging.WARNING, logger="okfgraph.router"):
            r.encoder.encode("hello")
        assert any("RTX 3090" in m and "okfgraph[gpu]" in m
                   for m in caplog.messages)
    finally:
        r.close()
