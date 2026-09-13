"""Cross-platform ONNX Runtime discovery tests.

Pure filesystem/import-behavior coverage: no model download, no ORT shared
library load, no encoder initialization.
"""

import os
import sys
import types

import pytest

from okfgraph.components import embedding as emb

_MISSING = object()


@pytest.fixture(autouse=True)
def _isolate_ort_dylib_path():
    """Save/restore `ORT_DYLIB_PATH` around every test.

    `resolve_ort_dylib()` assigns `os.environ` directly (that is its job -
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


def _make_package(tmp_path, dirname="ortpkg", files=()):
    pkg = tmp_path / dirname
    capi = pkg / "capi"
    capi.mkdir(parents=True, exist_ok=True)
    for name in files:
        (capi / name).write_bytes(b"fake-runtime")
    return pkg


def _fake_module(monkeypatch, name, package_dir, providers=(), preload=None):
    module = types.ModuleType(name)
    module.__file__ = str(package_dir / "__init__.py")
    calls = {"preload": 0}

    def _providers():
        return list(providers)

    def _preload():
        calls["preload"] += 1
        if isinstance(preload, Exception):
            raise preload

    module.get_available_providers = _providers
    module.preload_dlls = _preload
    monkeypatch.setitem(sys.modules, name, module)
    return calls


def test_candidate_names_cover_supported_platforms():
    assert emb._candidate_ort_library_names("nt", "win32") == ("onnxruntime.dll",)
    assert emb._candidate_ort_library_names("posix", "darwin") == (
        "libonnxruntime.dylib",
    )
    assert emb._candidate_ort_library_names("posix", "linux") == (
        "libonnxruntime.so",
    )


def test_find_runtime_windows_exact(tmp_path):
    pkg = _make_package(tmp_path, files=("onnxruntime.dll",))
    found = emb._find_runtime_in_package(pkg, "nt", "win32")
    assert found == pkg / "capi" / "onnxruntime.dll"


def test_find_runtime_macos_exact(tmp_path):
    pkg = _make_package(tmp_path, files=("libonnxruntime.dylib",))
    found = emb._find_runtime_in_package(pkg, "posix", "darwin")
    assert found == pkg / "capi" / "libonnxruntime.dylib"


def test_find_runtime_linux_prefers_versioned(tmp_path):
    pkg = _make_package(
        tmp_path,
        files=("libonnxruntime.so", "libonnxruntime.so.1", "libonnxruntime.so.2"),
    )
    found = emb._find_runtime_in_package(pkg, "posix", "linux")
    assert found == pkg / "capi" / "libonnxruntime.so.2"


def test_find_runtime_linux_unversioned_fallback(tmp_path):
    pkg = _make_package(tmp_path, files=("libonnxruntime.so",))
    found = emb._find_runtime_in_package(pkg, "posix", "linux")
    assert found == pkg / "capi" / "libonnxruntime.so"


def test_find_runtime_missing_returns_none(tmp_path):
    pkg = tmp_path / "empty-pkg"
    pkg.mkdir()
    assert emb._find_runtime_in_package(pkg, "nt", "win32") is None


def test_explicit_env_path_always_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("ORT_DYLIB_PATH", str(tmp_path / "explicit.so"))
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    monkeypatch.delitem(sys.modules, "onnxruntime-gpu", raising=False)
    assert emb.resolve_ort_dylib() == str(tmp_path / "explicit.so")


def test_resolve_windows_package(monkeypatch, tmp_path):
    monkeypatch.delenv("ORT_DYLIB_PATH", raising=False)
    pkg = _make_package(tmp_path, files=("onnxruntime.dll",))
    preload_calls = _fake_module(monkeypatch, "onnxruntime", pkg)
    monkeypatch.delitem(sys.modules, "onnxruntime-gpu", raising=False)
    dll_dirs = []
    monkeypatch.setattr(
        os, "add_dll_directory", lambda path: dll_dirs.append(path), raising=False
    )

    got = emb.resolve_ort_dylib(os_name="nt", sys_platform="win32")

    assert got == str(pkg / "capi" / "onnxruntime.dll")
    assert os.environ["ORT_DYLIB_PATH"] == got
    # DLL-directory setup is Windows-only; elsewhere it must stay a no-op.
    assert dll_dirs == ([str(pkg / "capi")] if os.name == "nt" else [])
    assert preload_calls["preload"] == 0


def test_resolve_macos_package(monkeypatch, tmp_path):
    monkeypatch.delenv("ORT_DYLIB_PATH", raising=False)
    pkg = _make_package(tmp_path, files=("libonnxruntime.dylib",))
    preload_calls = _fake_module(monkeypatch, "onnxruntime", pkg)
    monkeypatch.delitem(sys.modules, "onnxruntime-gpu", raising=False)

    got = emb.resolve_ort_dylib(os_name="posix", sys_platform="darwin")

    assert got == str(pkg / "capi" / "libonnxruntime.dylib")
    assert os.environ["ORT_DYLIB_PATH"] == got
    assert preload_calls["preload"] == 0


def test_resolve_linux_gpu_preload_failure_is_nonfatal(monkeypatch, tmp_path):
    monkeypatch.delenv("ORT_DYLIB_PATH", raising=False)
    pkg = _make_package(
        tmp_path,
        files=("libonnxruntime.so.1", "libonnxruntime.so.2"),
    )
    preload_calls = _fake_module(
        monkeypatch,
        "onnxruntime",
        pkg,
        providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
        preload=RuntimeError("missing CUDA DLL"),
    )
    monkeypatch.delitem(sys.modules, "onnxruntime-gpu", raising=False)

    got = emb.resolve_ort_dylib(os_name="posix", sys_platform="linux")

    assert got == str(pkg / "capi" / "libonnxruntime.so.2")
    assert os.environ["ORT_DYLIB_PATH"] == got
    assert preload_calls["preload"] == 1


def test_resolve_falls_through_to_gpu_package(monkeypatch, tmp_path):
    monkeypatch.delenv("ORT_DYLIB_PATH", raising=False)
    # First package has no runtime under capi/ — resolver must continue to
    # onnxruntime-gpu instead of importing the real installed package.
    empty = tmp_path / "no-runtime"
    empty.mkdir()
    _fake_module(monkeypatch, "onnxruntime", empty)
    pkg = _make_package(tmp_path, dirname="ortgpu", files=("onnxruntime.dll",))
    _fake_module(monkeypatch, "onnxruntime-gpu", pkg)
    monkeypatch.setattr(os, "add_dll_directory", lambda path: None, raising=False)

    got = emb.resolve_ort_dylib(os_name="nt", sys_platform="win32")

    assert got == str(pkg / "capi" / "onnxruntime.dll")
    assert os.environ["ORT_DYLIB_PATH"] == got


def test_resolve_missing_runtime_returns_none(monkeypatch, tmp_path):
    monkeypatch.delenv("ORT_DYLIB_PATH", raising=False)
    pkg = tmp_path / "no-capi"
    pkg.mkdir()
    _fake_module(monkeypatch, "onnxruntime", pkg)
    monkeypatch.delitem(sys.modules, "onnxruntime-gpu", raising=False)

    assert emb.resolve_ort_dylib() is None
    assert "ORT_DYLIB_PATH" not in os.environ
