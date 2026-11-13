"""Document-converter interface for PDF ingestion.

The knowledge graph never talks to a conversion engine directly — it talks
to a :class:`DocumentConverter`. Bobine is just the default implementation;
any pipeline that turns a PDF into markdown plus staged images can plug in
by implementing the one-method protocol::

    class MyConverter:
        def convert(self, pdf_path, output_dir, *, on_page=None):
            ...
            return ConvertedDocument(md_path=..., image_dir=..., page_count=...)

The ONNX runtime itself is still chosen at ORT level (``ORT_DYLIB_PATH``);
the device and weight precision a converter *asks* for are its own knobs
(0.14.1, ``pdf_device`` / ``pdf_precision``), so a machine can keep the
embedding lane on one device and conversion on another.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol

from okfgraph.errors import OKFError
from okfgraph.settings import DEVICES as PDF_DEVICES, PRECISIONS as PDF_PRECISIONS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConvertedDocument:
    """The only thing the graph needs from a conversion pipeline.

    Attributes:
        md_path: The converted ``<stem>.md`` file inside ``output_dir``.
            Transient when produced in a temp dir (auto-import) — the graph
            keeps the content, not the file.
        image_dir: Directory holding the staged images (may be empty).
        page_count: Number of pages in the source PDF.
    """

    md_path: Path
    image_dir: Path
    page_count: int


class DocumentConverter(Protocol):
    """One-method protocol every ingestion pipeline must satisfy."""

    def convert(
        self,
        pdf_path: str | Path,
        output_dir: str | Path,
        *,
        on_page: Callable[[int, int], None] | None = None,
    ) -> ConvertedDocument:
        """Convert ``pdf_path`` into ``output_dir``.

        Must write ``<stem>.md`` plus any staged images into ``output_dir``
        and return their locations. ``on_page`` receives 0-based
        ``(page_index, page_total)`` progress callbacks.
        """
        ...  # pragma: no cover - protocol stub


# Bobine routing names are Capitalized; the public API takes lowercase.
_BOBINE_ROUTING = {
    "auto": "Auto",
    "surgical": "Surgical",
    "always": "Always",
    "never": "Never",
}


def _hub_cache_dir(cache_dir: Optional[str]) -> str:
    """Resolve the hub-cache root without touching the network.

    ``cache_dir=None`` means "the standard hub cache" for every lane; bobine's
    ``HybridConverter`` wants the path itself, so resolve it the same way the
    embedding engine does instead of duplicating hf-hub's rules.
    """
    if cache_dir:
        return str(cache_dir)
    from okfgraph.components.embedding import EmbeddingEngine  # local: slow import

    return EmbeddingEngine.default_cache_dir()


def converter_plan(
    device: str = "auto",
    precision: str = "auto",
    cache_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """What a converter with these words resolves to, without loading models.

    bobine's ``describe()`` answers before the first conversion (requested
    words filled, ``landed: false``), so a status screen can show the plan a
    machine *would* use. Returns ``None`` when bobine cannot answer (not
    installed, or older than 0.9); callers render that as "unknown", never as
    an error.
    """
    try:
        import bobine  # noqa: PLC0415 — optional dependency
    except ImportError:
        return None
    if not hasattr(bobine, "HybridConverter"):
        return None
    try:
        converter = bobine.HybridConverter(
            bobine.ConverterConfig(device=device, precision=precision),
            _hub_cache_dir(cache_dir),
        )
        try:
            return converter.describe()
        finally:
            converter.close()
    except Exception as exc:  # a status probe must never raise
        logger.debug("converter_plan: bobine.describe failed: %s", exc)
        return None


def converter_status(
    cache_dir: Optional[str] = None,
    *,
    device: str = "auto",
    precision: str = "auto",
) -> Dict[str, Any]:
    """Converter-model cache status without downloading anything.

    Aggregates ``bobine.model_status()`` (bobine >= 0.6.0) into the same
    report shape okfgraph's own ``model_info`` uses, so one screen can
    answer "is this machine ready for PDF ingest offline?". Never
    raises: every outcome — missing bobine, too-old bobine, unexpected
    failure — is data, so doctor and CLI can render it unconditionally.

    Returns ``{"available", "reports", "note", "plan"}``: ``reports`` is
    the verbatim ``bobine.model_status()`` list when ``available``;
    otherwise ``note`` carries the one-line reason. ``plan`` is the
    converter's resolved per-pass plan for ``device``/``precision``
    (:func:`converter_plan`), or ``None`` when bobine cannot answer it —
    never from ``model_status``, which is cache status only.
    """
    plan = converter_plan(device=device, precision=precision, cache_dir=cache_dir)
    try:
        import bobine  # noqa: PLC0415 — optional dependency, lazy import
    except ImportError:
        return {
            "available": False,
            "reports": [],
            "note": ("converters: bobine not installed "
                       "(PDF ingest unavailable; pip install \"okfgraph[pdf]\")"),
            "plan": plan,
        }
    if not hasattr(bobine, "model_status"):
        try:
            from importlib.metadata import version as _version
            ver = _version("bobine")
        except Exception:  # pragma: no cover - metadata edge
            ver = "unknown"
        return {
            "available": False,
            "reports": [],
            "note": (f"converters: bobine {ver} has no model_status "
                       "(upgrade to >=0.6.0 for converter cache status)"),
            "plan": plan,
        }
    try:
        reports = bobine.model_status(cache_dir=cache_dir)
    except Exception as exc:  # offline lookup; failures are local (perms)
        logger.warning("converter_status: bobine.model_status failed: %s", exc)
        return {
            "available": False,
            "reports": [],
            "note": f"converters: status lookup failed ({exc})",
            "plan": plan,
        }
    return {"available": True, "reports": list(reports), "note": None,
            "plan": plan}


class BobineConverter:
    """The bobine (Rust) implementation of :class:`DocumentConverter`.

    Args:
        routing_mode: "auto" | "surgical" | "always" | "never". Controls
            when bobine invokes ONNX models ("never" = pdf_oxide fast path).
        extract_images: Whether to extract embedded images from the PDF.
        device: "auto" | "cpu" | "cuda" — which device the converter's ONNX
            passes target (``pdf_device``, 0.14.1).
        precision: "auto" | "fp32" | "fp16" | "int8" — converter weight
            precision (``pdf_precision``, 0.14.1).
        cache_dir: Converter-model cache root (default: the hub cache).

    Raises:
        ValueError: On an unknown routing_mode/device/precision (at
            construction — no work done).
        RuntimeError: When converting without bobine installed.
    """

    def __init__(
        self,
        routing_mode: str = "auto",
        extract_images: bool = True,
        device: str = "auto",
        precision: str = "auto",
        cache_dir: Optional[str] = None,
        idle_seconds: float = 300.0,
    ):
        if routing_mode not in _BOBINE_ROUTING:
            raise OKFError(
                "BAD_VALUE",
                f"routing_mode must be one of {sorted(_BOBINE_ROUTING)}, "
                f"got '{routing_mode}'"
            )
        if device not in PDF_DEVICES:
            raise OKFError(
                "BAD_VALUE",
                f"device must be one of {', '.join(PDF_DEVICES)}, "
                f"got '{device}'"
            )
        if precision not in PDF_PRECISIONS:
            raise OKFError(
                "BAD_VALUE",
                f"precision must be one of {', '.join(PDF_PRECISIONS)}, "
                f"got '{precision}'"
            )
        self.routing_mode = routing_mode
        self.extract_images = extract_images
        self.device = device
        self.precision = precision
        self.cache_dir = cache_dir
        # Warm sessions (0.14.1): the native HybridConverter is built once,
        # reused across files, and released by close() or idly. The lock is
        # load-bearing — bobine allows one document at a time and raises on a
        # concurrent call instead of waiting (it cannot block on a native
        # mutex while holding the GIL), so callers serialise here.
        self.idle_seconds = float(idle_seconds)
        self._lock = threading.RLock()
        self._hybrid = None
        self._last_used: Optional[float] = None
        self._timer: Optional[threading.Timer] = None
        self._conversions = 0

    # -- warm sessions ----------------------------------------------------

    @property
    def held(self) -> bool:
        """Whether this converter currently holds open ONNX sessions."""
        with self._lock:
            return self._hybrid is not None

    def _ensure_hybrid(self, bobine):
        hybrid = self._hybrid
        if hybrid is not None:
            return hybrid
        mode = getattr(bobine.RoutingMode, _BOBINE_ROUTING[self.routing_mode])
        config = bobine.ConverterConfig(
            routing_mode=mode,
            extract_images=self.extract_images,
            device=self.device,
            precision=self.precision,
        )
        hybrid = bobine.HybridConverter(config, _hub_cache_dir(self.cache_dir))
        self._hybrid = hybrid
        return hybrid

    def _cancel_timer(self) -> None:
        timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()

    def _arm_timer(self) -> None:
        """Release the sessions after ``idle_seconds`` without a conversion.

        One daemon thread per held converter, cancelled and re-armed on every
        conversion: no thread runs while nothing is held, and ``0`` releases
        right after each call (``okf serve`` keeps them only as long as the
        setting says).
        """
        self._cancel_timer()
        if self.idle_seconds < 0:
            return
        timer = threading.Timer(self.idle_seconds, self.release_if_idle)
        timer.daemon = True
        self._timer = timer
        timer.start()

    def release_if_idle(self, *, now: Optional[float] = None) -> bool:
        """Release the warm sessions when nothing converted for a while.

        Returns True when it released. ``now`` is injectable so tests never
        sleep; the timer passes nothing and reads ``time.monotonic()``.
        """
        with self._lock:
            hybrid = self._hybrid
            if hybrid is None:
                return False
            current = time.monotonic() if now is None else now
            last = self._last_used if self._last_used is not None else current
            idle = current - last
            if idle < self.idle_seconds:
                return False
            self._cancel_timer()
            self._hybrid = None
            try:
                hybrid.close()
            except Exception as exc:  # close must never raise at a timer
                logger.debug("closing idle warm converter failed: %s", exc)
            logger.debug("released warm PDF converter after %.1fs idle", idle)
            return True

    def close(self) -> None:
        """Release the warm ONNX sessions (idempotent).

        Bobine's closed converter refuses work; this wrapper does not — the
        next :meth:`convert` builds fresh sessions, which is what idle
        eviction and ``with`` both need.
        """
        with self._lock:
            self._cancel_timer()
            hybrid, self._hybrid = self._hybrid, None
            if hybrid is not None:
                hybrid.close()

    def __enter__(self) -> "BobineConverter":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False

    def status(self) -> Dict[str, Any]:
        """Warm-session state for ``okf doctor`` (never raises)."""
        with self._lock:
            held = self._hybrid is not None
            plan = None
            if held:
                try:
                    plan = self._hybrid.describe()
                except Exception:  # pragma: no cover - describe must answer
                    plan = None
            else:
                plan = converter_plan(self.device, self.precision,
                                      self.cache_dir)
            idle = (time.monotonic() - self._last_used
                    if self._last_used is not None else None)
            return {
                "held": held,
                "idle_seconds": round(idle, 1) if idle is not None else None,
                "release_after": self.idle_seconds,
                "conversions": self._conversions,
                "routing_mode": self.routing_mode,
                "device": self.device,
                "precision": self.precision,
                "plan": plan,
            }

    # -- conversion --------------------------------------------------------

    def convert(
        self,
        pdf_path: str | Path,
        output_dir: str | Path,
        *,
        on_page: Callable[[int, int], None] | None = None,
    ) -> ConvertedDocument:
        try:
            import bobine  # noqa: PLC0415 — optional dependency, lazy import
        except ImportError:
            raise RuntimeError(
                "bobine is required for PDF and Office ingestion: pip install \"okfgraph[pdf]\" (or pip install bobine)"
            ) from None

        # A missing file is a usage problem the caller can name exactly; it
        # must not read as a converter bug (0.14.0).
        source = Path(pdf_path)
        if not source.exists():
            raise OKFError(
                "FILE_NOT_FOUND",
                f"file not found: {source}",
                fields={"path": str(source), "converter": "bobine"},
            )

        # Serialise: bobine raises on a concurrent call, and two documents on
        # one converter would race the per-file engine state.
        with self._lock:
            self._cancel_timer()
            hybrid = self._ensure_hybrid(bobine)
            try:
                doc = hybrid.ingest_document(
                    str(pdf_path),
                    str(output_dir),
                    should_continue=lambda: True,
                    on_page=on_page
                    or (lambda idx, total: logger.info("page %d/%d",
                                                       idx + 1, total)),
                )
            except OKFError:
                raise
            except RuntimeError as exc:
                # bobine collapses every internal failure (a corrupt PDF, an
                # unsupported file, a pdf_oxide/office_oxide refusal) into
                # PyRuntimeError with a prefixed message. That is a conversion
                # outcome about this file, not an OKFgraph bug: type it so it
                # stops writing a crash log (0.14.0).
                raise OKFError(
                    "CONVERSION_FAILED",
                    f"bobine could not convert {source}: {exc}",
                    op="ingest",
                    fields={"path": str(source), "converter": "bobine"},
                    remedy="check the file opens in another reader, or route it "
                           "through a different converter",
                ) from exc
            finally:
                # A failed file still opened the sessions, so the idle clock
                # has to start and the timer has to be re-armed — otherwise a
                # host that only ever sees bad files would hold them forever.
                self._last_used = time.monotonic()
                self._arm_timer()
            self._conversions += 1
            return ConvertedDocument(
                md_path=Path(doc.md_path),
                image_dir=Path(doc.image_dir),
                page_count=doc.page_count,
            )


__all__ = ["ConvertedDocument", "DocumentConverter", "BobineConverter",
           "converter_status", "converter_plan"]
