"""Document-converter interface for PDF ingestion.

The knowledge graph never talks to a conversion engine directly — it talks
to a :class:`DocumentConverter`. Bobine is just the default implementation;
any pipeline that turns a PDF into markdown plus staged images can plug in
by implementing the one-method protocol::

    class MyConverter:
        def convert(self, pdf_path, output_dir, *, on_page=None):
            ...
            return ConvertedDocument(md_path=..., image_dir=..., page_count=...)

Device selection for ONNX-backed converters is ORT-level
(``ORT_DYLIB_PATH``) and intentionally not part of this interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol

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


class BobineConverter:
    """The bobine (Rust) implementation of :class:`DocumentConverter`.

    Args:
        routing_mode: "auto" | "surgical" | "always" | "never". Controls
            when bobine invokes ONNX models ("never" = pdf_oxide fast path).
        extract_images: Whether to extract embedded images from the PDF.

    Raises:
        ValueError: On an unknown routing_mode (at construction — no work done).
        RuntimeError: When converting without bobine installed.
    """

    def __init__(self, routing_mode: str = "auto", extract_images: bool = True):
        if routing_mode not in _BOBINE_ROUTING:
            raise ValueError(
                f"routing_mode must be one of {sorted(_BOBINE_ROUTING)}, "
                f"got '{routing_mode}'"
            )
        self.routing_mode = routing_mode
        self.extract_images = extract_images

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
                "bobine is required for PDF ingestion: pip install bobine"
            ) from None

        mode = getattr(bobine.RoutingMode, _BOBINE_ROUTING[self.routing_mode])
        config = bobine.ConverterConfig(
            routing_mode=mode,
            extract_images=self.extract_images,
        )
        doc = bobine.ingest_document(
            str(pdf_path),
            str(output_dir),
            config,
            should_continue=lambda: True,
            on_page=on_page
            or (lambda idx, total: logger.info("page %d/%d", idx + 1, total)),
        )
        return ConvertedDocument(
            md_path=Path(doc.md_path),
            image_dir=Path(doc.image_dir),
            page_count=doc.page_count,
        )


__all__ = ["ConvertedDocument", "DocumentConverter", "BobineConverter"]
