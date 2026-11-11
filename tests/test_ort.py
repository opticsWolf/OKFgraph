"""ONNX Runtime discovery tests.

okfgraph finds the runtime through embroider's locator (0.13); the
per-platform library search is embroider's and tested there. These cover
the delegation: pure filesystem coverage, no model download, no ORT shared
library load, no encoder initialization.
"""

import os
import sys
from pathlib import Path

import pytest

from okfgraph.components import embedding as emb

_MISSING = object()


@pytest.fixture(autouse=True)
def _isolate_ort_dylib_path():
    """Save/restore `ORT_DYLIB_PATH` around every test.

    `locate_ort_runtime()` assigns `os.environ` directly (that is its job -
    ort reads the variable at load), which `monkeypatch.delenv` cannot
    reliably undo when the variable starts absent. Without this fixture a
    fake runtime path leaks into later tests in the same process and real
    session opens die with an ORT `Dlopen` panic.
    """
    old = os.environ.get("ORT_DYLIB_PATH", _MISSING)
    os.environ.pop("ORT_DYLIB_PATH", None)
    try:
        yield
    finally:
        if old is _MISSING:
            os.environ.pop("ORT_DYLIB_PATH", None)
        else:
            os.environ["ORT_DYLIB_PATH"] = old


# The library name embroider looks for on this host.
if os.name == "nt":
    _LIBRARY = "onnxruntime.dll"
elif sys.platform == "darwin":
    _LIBRARY = "libonnxruntime.dylib"
else:
    _LIBRARY = "libonnxruntime.so.1.31.1"


def _site_packages(monkeypatch, tmp_path, *, library=True,
                   dist_info="onnxruntime_gpu-1.31.1.dist-info"):
    """A site-packages folder holding an ``onnxruntime`` package, first on
    ``sys.path`` so ``find_spec`` resolves it. Its ``__init__`` raises: the
    locator must find the library without importing the module."""
    site = tmp_path / "site"
    capi = site / "onnxruntime" / "capi"
    capi.mkdir(parents=True)
    (site / "onnxruntime" / "__init__.py").write_text(
        "raise ImportError('the locator imported onnxruntime')\n")
    if library:
        (capi / _LIBRARY).write_bytes(b"fake-runtime")
    if dist_info:
        (site / dist_info).mkdir()
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setattr(os, "add_dll_directory", lambda path: None,
                        raising=False)
    return capi


def test_explicit_env_path_always_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("ORT_DYLIB_PATH", str(tmp_path / "explicit.so"))
    _site_packages(monkeypatch, tmp_path)
    got = emb.locate_ort_runtime()
    assert got["dylib"] == str(tmp_path / "explicit.so")
    assert got["source"] == "env"


def test_pip_runtime_found_without_importing_it(monkeypatch, tmp_path):
    capi = _site_packages(monkeypatch, tmp_path)

    got = emb.locate_ort_runtime()

    assert Path(got["dylib"]) == capi / _LIBRARY
    assert got["source"] == "pip:onnxruntime-gpu"
    assert got["version"] == "1.31.1"
    assert os.environ["ORT_DYLIB_PATH"] == got["dylib"]
    assert "onnxruntime" not in sys.modules


def test_missing_runtime_reports_os(monkeypatch, tmp_path):
    _site_packages(monkeypatch, tmp_path, library=False, dist_info=None)

    got = emb.locate_ort_runtime()

    assert got["dylib"] is None and got["source"] == "os"
    assert "ORT_DYLIB_PATH" not in os.environ


def test_without_embroider_only_the_env_counts(monkeypatch, tmp_path):
    # backend='external' needs no wheel; a stand-in without the locator
    # behaves like a missing one.
    monkeypatch.setitem(sys.modules, "embroider", object())
    assert emb.locate_ort_runtime() == {
        "dylib": None, "source": "os", "version": None}
    monkeypatch.setenv("ORT_DYLIB_PATH", str(tmp_path / "explicit.so"))
    assert emb.locate_ort_runtime()["source"] == "env"


def test_resolve_ort_dylib_is_a_deprecated_shim(monkeypatch, tmp_path):
    capi = _site_packages(monkeypatch, tmp_path)
    with pytest.warns(DeprecationWarning, match="locate_ort_runtime"):
        got = emb.resolve_ort_dylib(warm_gpu=False, os_name="nt",
                                    sys_platform="win32")
    assert Path(got) == capi / _LIBRARY


def test_ort_info_reports_distributions(monkeypatch):
    """ort_info() names the installed ORT distribution(s) without a session."""
    import importlib.util
    info = emb.ort_info()
    assert set(info) == {"dylib_path", "installed", "version",
                         "providers", "cuda_usable", "both_installed",
                         "import_error"}
    has_ort = importlib.util.find_spec("onnxruntime") is not None
    if has_ort:
        assert info["import_error"] is None
    else:
        # ORT-less env (dev-only venv, CI): the cause must be kept,
        # not silently swallowed.
        assert info["import_error"] is not None
        assert "ModuleNotFoundError" in info["import_error"]
        assert info["providers"] == []
    # This venv has exactly one ORT (cpu XOR gpu — never both).
    assert len(info["installed"]) <= 1
    assert info["both_installed"] is False


def test_ort_info_no_runtime(monkeypatch):
    """With neither distribution present, installed is empty (actionable)."""
    import builtins
    from importlib import metadata as _metadata
    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    def _missing(name):
        raise _metadata.PackageNotFoundError(name)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    monkeypatch.setattr(_metadata, "version", _missing)
    info = emb.ort_info()
    assert info["installed"] == []
    assert info["version"] is None
    assert info["cuda_usable"] is False
    assert "no ONNX Runtime installed" in emb.ort_missing_hint(info)


def test_ort_hint_installed_but_broken(monkeypatch):
    """An installed ORT that fails to import is reported as broken, with the
    import error, not as missing (the misleading pre-fix hint)."""
    import builtins
    real_import = builtins.__import__

    def _broken(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("DLL load failed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _broken)
    info = emb.ort_info()
    if not info["installed"]:
        pytest.skip("no ORT distribution in this venv")
    hint = emb.ort_missing_hint(info)
    assert "installed" in hint and "fails to import" in hint
    assert "DLL load failed" in hint
    assert "no ONNX Runtime installed" not in hint
