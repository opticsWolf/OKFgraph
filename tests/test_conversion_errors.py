# -*- coding: utf-8 -*-
"""Typed conversion failures (0.14.0, plan step D).

A corrupt PDF used to surface as ``INTERNAL`` and write a crash log, which
reads like an OKFgraph bug. bobine collapses every internal failure into
``PyRuntimeError`` with a prefixed message (checked in its
``py_bindings.rs``: ``to_py_err`` maps ``BobineError`` one way), so the
converter maps exactly that type onto ``CONVERSION_FAILED`` — and leaves
everything else (a pyo3 panic, a callback error) alone.
"""
from __future__ import annotations

import sys
import types

import pytest

from okfgraph.components.converters import BobineConverter
from okfgraph.errors import OKFError


class _FakeRoutingMode:
    Auto = "auto"
    Surgical = "surgical"
    Always = "always"
    Never = "never"


class _FakeConfig:
    def __init__(self, **kw):
        self.kw = kw


def _fake_bobine(monkeypatch, *, raises):
    module = types.ModuleType("bobine")
    module.RoutingMode = _FakeRoutingMode
    module.ConverterConfig = _FakeConfig

    def ingest_document(path, out, config, **kw):
        if raises is not None:
            raise raises
        return types.SimpleNamespace(md_path=f"{out}/x.md", image_dir=out,
                                     page_count=1)

    module.ingest_document = ingest_document
    monkeypatch.setitem(sys.modules, "bobine", module)
    return module


def test_a_bobine_failure_is_conversion_failed(tmp_path, monkeypatch):
    src = tmp_path / "broken.pdf"
    src.write_bytes(b"%PDF-1.4 not really")
    _fake_bobine(monkeypatch, raises=RuntimeError(
        "pdf_oxide error: malformed xref table"))
    with pytest.raises(OKFError) as excinfo:
        BobineConverter().convert(src, tmp_path / "out")
    err = excinfo.value
    assert err.code == "CONVERSION_FAILED"
    assert err.fields == {"path": str(src), "converter": "bobine"}
    assert "malformed xref table" in err.message
    assert "another reader" in (err.remedy or "")


def test_a_missing_file_stays_file_not_found(tmp_path, monkeypatch):
    _fake_bobine(monkeypatch, raises=AssertionError("must not be called"))
    with pytest.raises(OKFError) as excinfo:
        BobineConverter().convert(tmp_path / "absent.pdf", tmp_path / "out")
    assert excinfo.value.code == "FILE_NOT_FOUND"
    assert excinfo.value.fields["path"].endswith("absent.pdf")


def test_other_failures_are_not_masked(tmp_path, monkeypatch):
    """A non-RuntimeError (a pyo3 panic is a BaseException) stays raw."""
    src = tmp_path / "x.pdf"
    src.write_bytes(b"%PDF-1.4")
    _fake_bobine(monkeypatch, raises=KeyError("unexpected"))
    with pytest.raises(KeyError):
        BobineConverter().convert(src, tmp_path / "out")


def test_a_successful_conversion_is_unchanged(tmp_path, monkeypatch):
    src = tmp_path / "ok.pdf"
    src.write_bytes(b"%PDF-1.4")
    _fake_bobine(monkeypatch, raises=None)
    doc = BobineConverter().convert(src, tmp_path / "out")
    assert doc.page_count == 1 and doc.md_path.name == "x.md"


@pytest.mark.pdf
def test_a_real_corrupt_pdf_is_conversion_failed(tmp_path):
    """No fake: bobine really refuses a truncated PDF."""
    pytest.importorskip("bobine")
    src = tmp_path / "corrupt.pdf"
    src.write_bytes(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog")   # truncated
    with pytest.raises(OKFError) as excinfo:
        BobineConverter(routing_mode="never").convert(src, tmp_path / "out")
    assert excinfo.value.code == "CONVERSION_FAILED"
