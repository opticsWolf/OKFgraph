# -*- coding: utf-8 -*-
"""PDF converter knobs and the warm session lane (0.14.1, plan steps E/F).

E — ``pdf_device`` / ``pdf_precision`` reach bobine's ``ConverterConfig``,
validated against the same vocabulary bobine exports, and the resolved plan
is reported from ``describe()`` (never from ``model_status()``, which is
cache status only).

F — one warm ``HybridConverter`` per settings key: many files on one
converter build the layout/OCR sessions once, ``close()`` releases them and
the next conversion rebuilds, an idle converter releases after
``pdf_idle_seconds`` (0 = after every call), and concurrent conversions
serialise because bobine raises rather than waiting.

Every test here uses a fake ``bobine`` module (no models, no ONNX); the
real-converter behaviour is covered by the ``pdf``-marked tests.
"""
from __future__ import annotations

import sys
import threading
import time
import types

import pytest

from okfgraph.components.converters import (
    BobineConverter,
    converter_plan,
    converter_status,
)
from okfgraph.errors import OKFError
from okfgraph.settings import DEVICES, PRECISIONS, SETTINGS, Settings


# ---- fake bobine -----------------------------------------------------------

class _FakeRoutingMode:
    Auto = "auto"
    Surgical = "surgical"
    Always = "always"
    Never = "never"


class _FakeConfig:
    def __init__(self, **kw):
        self.kw = kw


class _FakeHybrid:
    """Warm converter stand-in: records config, documents and overlap."""

    instances: list = []
    document_delay = 0.0
    active = 0
    max_active = 0
    raises = None

    def __init__(self, config, cache_dir):
        self.config = config
        self.cache_dir = cache_dir
        self.documents: list = []
        self.closed = False
        type(self).instances.append(self)

    def ingest_document(self, path, out, **kw):
        cls = type(self)
        cls.active += 1
        cls.max_active = max(cls.max_active, cls.active)
        try:
            if cls.document_delay:
                time.sleep(cls.document_delay)
            self.documents.append((path, out, kw))
            if cls.raises is not None:
                raise cls.raises
            return types.SimpleNamespace(
                md_path=f"{out}/x.md", image_dir=out, page_count=1)
        finally:
            cls.active -= 1

    def describe(self):
        return {
            "device_requested": self.config.kw.get("device", "auto"),
            "precision_requested": self.config.kw.get("precision", "auto"),
            "passes": {"layout": {"providers": None, "policy": "auto-cuda",
                                  "precision": "fp16", "landed": False}},
        }

    def close(self):
        self.closed = True


def _fake_bobine(monkeypatch, *, model_status=None):
    module = types.ModuleType("bobine")
    module.RoutingMode = _FakeRoutingMode
    module.ConverterConfig = _FakeConfig
    module.HybridConverter = _FakeHybrid
    module.instances = _FakeHybrid.instances
    if model_status is not None:
        module.model_status = lambda cache_dir=None: model_status
    _FakeHybrid.instances = []
    _FakeHybrid.document_delay = 0.0
    _FakeHybrid.active = 0
    _FakeHybrid.max_active = 0
    _FakeHybrid.raises = None
    monkeypatch.setitem(sys.modules, "bobine", module)
    return module


@pytest.fixture()
def pdf(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4 stub")
    return path


# ---- E: settings -----------------------------------------------------------

def test_pdf_settings_defaults_and_vocabulary():
    rows = {row.name: row for row in SETTINGS}
    assert rows["pdf_device"].section == "import"
    assert rows["pdf_device"].choices == DEVICES == ("auto", "cpu", "cuda")
    assert rows["pdf_precision"].choices == PRECISIONS == (
        "auto", "fp32", "fp16", "int8")
    # A long-running host option: no per-command flag.
    assert rows["pdf_idle_seconds"].cli is False

    s = Settings()
    assert (s.pdf_device, s.pdf_precision) == ("auto", "auto")
    assert s.pdf_idle_seconds == 300
    kwargs = s.router_kwargs()
    assert kwargs["pdf_device"] == "auto"
    assert kwargs["pdf_precision"] == "auto"
    assert kwargs["pdf_idle_seconds"] == 300


def test_pdf_settings_from_toml(tmp_path):
    (tmp_path / "okfgraph.toml").write_text(
        "[import]\npdf_device = \"cpu\"\npdf_precision = \"fp16\"\n"
        "pdf_idle_seconds = 0\n",
        encoding="utf-8")
    s = Settings.load(bundle_root=str(tmp_path), environ={})
    assert (s.pdf_device, s.pdf_precision, s.pdf_idle_seconds) == (
        "cpu", "fp16", 0)


def test_pdf_settings_refuse_garbage():
    with pytest.raises(ValueError, match="pdf_device"):
        Settings.load(environ={"OKFGRAPH_PDF_DEVICE": "gpu2"})
    with pytest.raises(ValueError, match="pdf_precision"):
        Settings.load(environ={"OKFGRAPH_PDF_PRECISION": "q8"})
    with pytest.raises(ValueError, match="pdf_idle_seconds"):
        Settings.load(environ={"OKFGRAPH_PDF_IDLE_SECONDS": "-1"})


def test_vocabulary_never_drifts_from_bobine():
    bobine = pytest.importorskip("bobine")
    if not hasattr(bobine, "DEVICES"):  # bobine < 0.9
        pytest.skip("bobine has no DEVICES/PRECISIONS (needs >=0.9)")
    assert tuple(bobine.DEVICES) == DEVICES
    assert tuple(bobine.PRECISIONS) == PRECISIONS


# ---- E: the converter ------------------------------------------------------

def test_bad_knobs_are_refused_at_construction():
    with pytest.raises(OKFError) as excinfo:
        BobineConverter(device="gpu2")
    assert excinfo.value.code == "BAD_VALUE"
    assert "auto, cpu, cuda" in excinfo.value.message

    with pytest.raises(OKFError) as excinfo:
        BobineConverter(precision="q8")
    assert excinfo.value.code == "BAD_VALUE"
    # The stack has no q8 artifact; int8 is the compact formula pair.
    assert "int8" in excinfo.value.message


def test_converter_config_gets_words_and_cache_dir(tmp_path, monkeypatch):
    _fake_bobine(monkeypatch)
    src = tmp_path / "doc.pdf"
    src.write_bytes(b"%PDF-1.4 stub")
    conv = BobineConverter(device="cuda", precision="fp16",
                           cache_dir=str(tmp_path / "cache"))
    conv.convert(src, tmp_path / "out")
    hybrid = _FakeHybrid.instances[0]
    assert hybrid.config.kw["device"] == "cuda"
    assert hybrid.config.kw["precision"] == "fp16"
    assert hybrid.cache_dir == str(tmp_path / "cache")
    assert hybrid.documents[0][0] == str(src)


def test_converter_status_reports_the_plan(monkeypatch):
    _fake_bobine(monkeypatch, model_status=[])
    status = converter_status(device="cpu", precision="fp32")
    assert status["available"] is True
    assert status["plan"]["device_requested"] == "cpu"
    assert status["plan"]["precision_requested"] == "fp32"
    assert status["plan"]["passes"]["layout"]["policy"] == "auto-cuda"


def test_converter_status_without_bobine_has_no_plan(monkeypatch):
    monkeypatch.setitem(sys.modules, "bobine", None)  # import -> ImportError
    assert converter_plan() is None
    status = converter_status()
    assert status["available"] is False
    assert status["plan"] is None
    assert "bobine not installed" in status["note"]


# ---- F: warm sessions ------------------------------------------------------

def test_many_files_build_the_converter_once(tmp_path, monkeypatch):
    _fake_bobine(monkeypatch)
    conv = BobineConverter()
    for n in range(3):
        src = tmp_path / f"doc{n}.pdf"
        src.write_bytes(b"%PDF-1.4 stub")
        conv.convert(src, tmp_path / f"out{n}")
    assert len(_FakeHybrid.instances) == 1
    assert len(_FakeHybrid.instances[0].documents) == 3
    assert conv.held is True


def test_close_releases_and_the_next_convert_rebuilds(tmp_path, monkeypatch):
    _fake_bobine(monkeypatch)
    src = tmp_path / "doc.pdf"
    src.write_bytes(b"%PDF-1.4 stub")
    conv = BobineConverter()
    conv.convert(src, tmp_path / "out")
    first = _FakeHybrid.instances[0]

    conv.close()
    assert first.closed is True
    assert conv.held is False

    conv.convert(src, tmp_path / "out2")
    assert len(_FakeHybrid.instances) == 2
    assert conv.held is True


def test_idle_release_uses_an_injected_clock(tmp_path, monkeypatch):
    _fake_bobine(monkeypatch)
    src = tmp_path / "doc.pdf"
    src.write_bytes(b"%PDF-1.4 stub")
    conv = BobineConverter(idle_seconds=60)
    conv.convert(src, tmp_path / "out")
    hybrid = _FakeHybrid.instances[0]
    last = conv._last_used

    assert conv.release_if_idle(now=last + 59) is False
    assert conv.held is True

    assert conv.release_if_idle(now=last + 60) is True
    assert hybrid.closed is True
    assert conv.held is False
    assert conv.release_if_idle(now=last + 120) is False  # already released


def test_idle_zero_releases_right_after_the_call(tmp_path, monkeypatch):
    _fake_bobine(monkeypatch)
    src = tmp_path / "doc.pdf"
    src.write_bytes(b"%PDF-1.4 stub")
    conv = BobineConverter(idle_seconds=0)
    conv.convert(src, tmp_path / "out")

    deadline = time.monotonic() + 5
    while conv.held and time.monotonic() < deadline:
        time.sleep(0.01)
    assert conv.held is False
    assert _FakeHybrid.instances[0].closed is True


def test_concurrent_converts_serialise(tmp_path, monkeypatch):
    _fake_bobine(monkeypatch)
    _FakeHybrid.document_delay = 0.05
    conv = BobineConverter()
    errors: list = []

    def worker(n):
        try:
            src = tmp_path / f"doc{n}.pdf"
            src.write_bytes(b"%PDF-1.4 stub")
            conv.convert(src, tmp_path / f"out{n}")
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert _FakeHybrid.max_active == 1  # bobine would have raised otherwise
    assert len(_FakeHybrid.instances[0].documents) == 2


def test_callbacks_are_forwarded_to_the_warm_converter(tmp_path, monkeypatch):
    _fake_bobine(monkeypatch)
    src = tmp_path / "doc.pdf"
    src.write_bytes(b"%PDF-1.4 stub")
    conv = BobineConverter()
    seen: list = []

    def on_page(index, total):
        seen.append((index, total))

    conv.convert(src, tmp_path / "out", on_page=on_page)
    kwargs = _FakeHybrid.instances[0].documents[0][2]
    assert kwargs["on_page"] is on_page
    assert callable(kwargs["should_continue"])


# ---- F: manager and router ownership ---------------------------------------

def test_manager_reuses_one_converter_per_key(monkeypatch):
    _fake_bobine(monkeypatch)
    from okfgraph.components.ingest import IngestManager

    mgr = IngestManager(None, None, "auto", None, None,
                        pdf_device="cpu", pdf_precision="fp32")
    first = mgr._resolve_converter(None)
    assert mgr._resolve_converter(None) is first

    # A different routing closes the warm one instead of holding two sets.
    surgical = mgr._resolve_converter(None, routing_mode="surgical")
    assert surgical is not first
    assert first.held is False
    assert surgical.routing_mode == "surgical"
    assert surgical.device == "cpu"

    mgr.close_converter()
    assert surgical.held is False


def test_injected_converter_is_never_replaced():
    from okfgraph.components.ingest import IngestManager

    class Mine:
        pass

    mine = Mine()
    mgr = IngestManager(None, None, "auto", None, None, converter=mine)
    assert mgr._resolve_converter(None) is mine
    mgr.close_converter()  # nothing warm to release


def test_router_close_releases_the_warm_converter(tmp_path, monkeypatch):
    _fake_bobine(monkeypatch)
    from okfgraph.router import OKFRouter

    router = OKFRouter(db_path=str(tmp_path / "kb.db"),
                       bundle_root=str(tmp_path),
                       pdf_device="cpu", pdf_precision="fp32")
    try:
        warm = router.ingest_mgr._resolve_converter(None)
        assert warm.held is False  # built, no conversion yet
        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF-1.4 stub")
        warm.convert(src, tmp_path / "out")
        assert warm.held is True
    finally:
        router.close()
    assert warm.held is False
    assert _FakeHybrid.instances[0].closed is True


def test_doctor_reports_the_pdf_converter(tmp_path, monkeypatch):
    _fake_bobine(monkeypatch, model_status=[])
    from okfgraph.router import OKFRouter

    router = OKFRouter(db_path=str(tmp_path / "kb.db"),
                       bundle_root=str(tmp_path),
                       pdf_device="cpu", pdf_precision="fp32")
    try:
        report = router.doctor()["report"]
        row = next(r for r in report["info"] if r["rule"] == "pdf_converter")
        assert "released" in row["message"]
        assert "device=cpu" in row["message"]
        assert "precision=fp32" in row["message"]

        warm = router.ingest_mgr._resolve_converter(None)
        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF-1.4 stub")
        warm.convert(src, tmp_path / "out")

        report = router.doctor()["report"]
        row = next(r for r in report["info"] if r["rule"] == "pdf_converter")
        assert "warm" in row["message"]
        assert "routing=auto" in row["message"]
        assert "1 conversion(s)" in row["message"]
    finally:
        router.close()


# ---- F: real bobine (no models; the fast path needs no ONNX) ---------------

def _min_pdf(text: str) -> bytes:
    body = f"BT /F1 12 Tf 100 700 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%*s\nendstream" % (len(body), len(body), body),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % n + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += (b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objects) + 1, xref))
    return bytes(out)


@pytest.mark.pdf
def test_real_bobine_keeps_sessions_warm(tmp_path, monkeypatch):
    """Two PDFs, one HybridConverter: the sessions survive between files."""
    bobine = pytest.importorskip("bobine")
    if not hasattr(bobine, "HybridConverter"):
        pytest.skip("bobine < 0.9 has no warm converter")

    real = bobine.HybridConverter
    built: list = []

    class Counting:
        def __init__(self, config, cache_dir):
            built.append(config)
            self._inner = real(config, cache_dir)

        def ingest_document(self, *a, **kw):
            return self._inner.ingest_document(*a, **kw)

        def describe(self):
            return self._inner.describe()

        def close(self):
            self._inner.close()

    monkeypatch.setattr(bobine, "HybridConverter", Counting)

    paths = []
    for n in range(2):
        path = tmp_path / f"p{n}.pdf"
        path.write_bytes(_min_pdf(f"Warm page {n}"))
        paths.append(path)

    conv = BobineConverter(routing_mode="never")
    docs = [conv.convert(p, tmp_path / f"out{n}") for n, p in enumerate(paths)]

    assert len(built) == 1, "the converter was rebuilt per file"
    assert [d.page_count for d in docs] == [1, 1]
    assert all(d.md_path.is_file() for d in docs)

    status = conv.status()
    assert status["held"] is True
    assert status["conversions"] == 2
    assert status["device"] == "auto"

    conv.close()
    assert conv.held is False


def test_a_failed_conversion_still_arms_the_idle_timer(tmp_path, monkeypatch):
    """A host that only sees bad files must not hold sessions forever."""
    module = _fake_bobine(monkeypatch)
    _FakeHybrid.raises = RuntimeError("pdf_oxide error: malformed xref")

    conv = BobineConverter(idle_seconds=60)
    src = tmp_path / "broken.pdf"
    src.write_bytes(b"%PDF-1.4 not really")
    with pytest.raises(OKFError) as excinfo:
        conv.convert(src, tmp_path / "out")
    assert excinfo.value.code == "CONVERSION_FAILED"
    assert conv.held is True
    assert conv.status()["conversions"] == 0
    last = conv._last_used
    assert last is not None
    assert conv.release_if_idle(now=last + 60) is True
    assert conv.held is False


def test_doctor_marks_an_injected_pipeline_as_custom(tmp_path, monkeypatch):
    """A custom DocumentConverter is not the bobine lane; say so."""
    _fake_bobine(monkeypatch, model_status=[])
    from okfgraph.router import OKFRouter

    class Mine:
        def convert(self, pdf_path, output_dir, *, on_page=None):
            raise AssertionError("not called")

    router = OKFRouter(db_path=str(tmp_path / "kb.db"),
                       bundle_root=str(tmp_path), converter=Mine())
    try:
        report = router.doctor()["report"]
        row = next(r for r in report["info"] if r["rule"] == "pdf_converter")
        assert "custom pipeline" in row["message"]
    finally:
        router.close()
