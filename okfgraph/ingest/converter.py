"""HybridConverter — pdf_oxide fast path + ONNX/Rapid heavy passes.

Core, Qt-independent pipeline. Three routing modes control when ONNX models
are invoked:

* **NEVER**   — fast path only, no ONNX.
* **AUTO**    — flag math/scanned pages and run the *whole* page through
                RapidLayout + RapidOCR.
* **ALWAYS**  — every page through RapidLayout + RapidOCR.
* **SURGICAL** — minimize the neural surface: locate formula boxes from
                 pdf_oxide's own character geometry (no layout network), crop
                 just those regions, batch them through RapidLaTeXOCR, and
                 splice the LaTeX back into pdf_oxide's fast markdown.
                 Genuine text-less scans still fall back to the full
                 RapidLayout + RapidOCR pipeline, loaded lazily only when
                 the first scan page is hit.

Output contract: a single ``.md`` string containing inline/display LaTeX,
fenced code blocks, GFM tables, and ``okf-asset://`` image links.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Optional

from okfgraph.ingest.config import ConverterConfig, RoutingMode
from okfgraph.ingest.engine import OnnxRapidEngine
from okfgraph.ingest.tables import html_tables_to_gfm

# ── Optional native deps (always guarded) ────────────────────────────────
try:
    from pdf_oxide import PdfDocument
except ImportError:
    PdfDocument = None  # type: ignore[assignment]

try:
    from office_oxide import Document as OfficeDocument
except ImportError:
    OfficeDocument = None  # type: ignore[assignment]

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None  # type: ignore[assignment]

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

# ── Math font / unicode heuristics ──────────────────────────────────────
_MATH_FONT_KEYWORDS = (
    "cmmi", "cmsy", "cmex", "msam", "msbm", "math", "symbol",
    "mathjax", "stix", "xits", "asana", "euclid", "cmr10",
)

_MATH_UNICODE_RANGES = (
    (0x0370, 0x03FF),   # Greek
    (0x2200, 0x22FF),   # Mathematical Operators
    (0x2A00, 0x2AFF),   # Supplemental Mathematical Operators
    (0x27C0, 0x27EF),   # Misc Mathematical Symbols-A
    (0x2980, 0x29FF),   # Misc Mathematical Symbols-B
    (0x1D400, 0x1D7FF), # Mathematical Alphanumeric Symbols
)

_MONO_FONT_KEYWORDS = (
    "mono", "courier", "consol", "menlo", "inconsolata",
    "sourcecode", "dejavu sans mono", "fixed", "terminal",
)


def _is_math_unicode(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _MATH_UNICODE_RANGES)


def _is_mono_font(font_name: str) -> bool:
    fn = font_name.lower()
    return any(k in fn for k in _MONO_FONT_KEYWORDS)


# ── Core pipeline ───────────────────────────────────────────────────────
class HybridConverter:
    """PDF → Markdown converter with ONNX/Rapid heavy passes."""

    def __init__(self, config: ConverterConfig, log: Callable[[str], None] = print) -> None:
        self.cfg = config
        self.log = log
        self.rapid = OnnxRapidEngine(
            log_fn=self.log,
            ort_providers=config.ort_providers,
        )

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def ensure_models(self) -> None:
        """Load models required by the current routing mode."""
        if not self.cfg.use_onnx:
            return
        mode = self.cfg.routing_mode
        if mode == RoutingMode.NEVER:
            return
        if mode == RoutingMode.SURGICAL:
            self._ensure_formula()
        else:
            self._ensure_full_structure()

    def _ensure_formula(self) -> None:
        """Load RapidLaTeXOCR (light, single-purpose)."""
        self.rapid.formula()

    def _ensure_full_structure(self) -> None:
        """Load RapidOCR + RapidLayout (heavy)."""
        self.rapid.ocr()
        self.rapid.layout()

    def close(self) -> None:
        """Release all ONNX model references."""
        self.rapid.close()

    # ------------------------------------------------------------------
    # pdf_oxide fast path
    # ------------------------------------------------------------------

    def _fast_page_markdown(self, page) -> str:
        """Extract markdown from a single page via pdf_oxide."""
        for attempt in (
            lambda: page.markdown(detect_headings=self.cfg.detect_headings),
            lambda: page.markdown(),
            lambda: page.plain_text(),
            lambda: getattr(page, "text", "") or "",
        ):
            try:
                out = attempt()
                if out is not None:
                    return out
            except (AttributeError, TypeError):
                continue
        return ""

    def _extract_page_images(self, doc, page, index: int, img_dir: Path) -> list[Path]:
        """Extract embedded images from a page into ``img_dir``."""
        paths: list[Path] = []
        try:
            images = doc.extract_image_bytes(index)
            for n, im in enumerate(images or []):
                data = im.get("data") if isinstance(im, dict) else getattr(im, "data", None)
                fmt = (im.get("format") if isinstance(im, dict)
                       else getattr(im, "format", None)) or "png"
                fmt = str(fmt).lower().lstrip(".")
                if not data:
                    continue
                p = img_dir / f"p{index}_img{n}.{fmt}"
                p.write_bytes(data)
                paths.append(p)
            return paths
        except AttributeError:
            pass
        except Exception as e:
            self.log(f"   ⚠️  image extraction failed on page {index + 1}: {e}")
        return paths

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render_raw(self, doc, page, dpi: int):
        """Return whatever the pdf_oxide render API produces, or None."""
        scale = dpi / 72.0
        idx = getattr(page, "index", 0)
        for attempt in (
            lambda: page.render(dpi=dpi),
            lambda: page.render(scale=scale),
            lambda: page.render(),
            lambda: page.render_to_image(dpi=dpi),
            lambda: doc.render(idx, dpi=dpi),
        ):
            try:
                obj = attempt()
                if obj is not None:
                    return obj
            except (AttributeError, TypeError):
                continue
            except Exception:
                continue
        return None

    def _obj_to_pil(self, obj):
        """Convert a render result to a PIL Image, or None."""
        if obj is None or PILImage is None:
            return None
        try:
            if hasattr(obj, "crop") and hasattr(obj, "save"):
                return obj
            if isinstance(obj, (bytes, bytearray)):
                return PILImage.open(io.BytesIO(bytes(obj)))
            if isinstance(obj, dict) and obj.get("data"):
                return PILImage.open(io.BytesIO(obj["data"]))
            if np is not None and isinstance(obj, np.ndarray):
                return PILImage.fromarray(obj)
        except Exception:
            return None
        return None

    def _render_page_to_pil(self, doc, page, index: int, dpi: int):
        """Return ``(PIL.Image RGB, page_width_pts, page_height_pts)`` or None."""
        obj = self._render_raw(doc, page, dpi)
        img = self._obj_to_pil(obj)
        if img is None:
            return None
        try:
            img = img.convert("RGB")
        except Exception:
            return None
        w = float(getattr(page, "width", 0) or 0)
        h = float(getattr(page, "height", 0) or 0)
        if h <= 0:
            return None
        return img, w, h

    def _render_page_to_png(self, doc, page, index: int, out_png: Path, dpi: int = 0) -> bool:
        """Render a page to a PNG file on disk."""
        dpi = dpi or self.cfg.render_dpi
        obj = self._render_raw(doc, page, dpi)
        if obj is None:
            return False
        try:
            if hasattr(obj, "save"):
                obj.save(str(out_png))
            elif isinstance(obj, (bytes, bytearray)):
                out_png.write_bytes(bytes(obj))
            elif isinstance(obj, dict) and obj.get("data"):
                out_png.write_bytes(obj["data"])
            elif np is not None and isinstance(obj, np.ndarray):
                if PILImage is None:
                    return False
                PILImage.fromarray(obj).save(str(out_png))
            else:
                return False
        except Exception as e:
            self.log(f"   ⚠️  could not save render of page {index + 1}: {e}")
            return False
        return out_png.is_file()

    # ------------------------------------------------------------------
    # Routing signals
    # ------------------------------------------------------------------

    def _page_math_signal(self, page) -> tuple[int, int]:
        """Count math chars vs total chars on a page."""
        total = 0
        math = 0
        chars = getattr(page, "chars", None)
        if chars:
            for c in chars:
                total += 1
                fn = (getattr(c, "font_name", "") or "").lower()
                ch = getattr(c, "char", "") or ""
                if any(k in fn for k in _MATH_FONT_KEYWORDS) or (ch and _is_math_unicode(ch)):
                    math += 1
            return math, total
        text = getattr(page, "text", "") or ""
        total = len(text)
        math = sum(1 for ch in text if _is_math_unicode(ch))
        for sp in getattr(page, "spans", []) or []:
            fn = (getattr(sp, "font_name", "") or "").lower()
            if any(k in fn for k in _MATH_FONT_KEYWORDS):
                math += len(getattr(sp, "text", "") or "")
        return math, total

    def _is_scanned(self, page) -> bool:
        """Heuristic: few chars + images present → scanned page."""
        try:
            chars = getattr(page, "chars", None)
            total = len(chars) if chars is not None else len(getattr(page, "text", "") or "")
            n_images = len(getattr(page, "images", []) or [])
            return total < self.cfg.scanned_text_threshold and n_images > 0
        except Exception:
            return False

    def _needs_paddle(self, page) -> bool:
        """Should this page go through the full ONNX pipeline?"""
        if self.cfg.routing_mode == RoutingMode.NEVER:
            return False
        if self.cfg.routing_mode == RoutingMode.ALWAYS:
            return True
        try:
            math, total = self._page_math_signal(page)
            if math > self.cfg.math_char_threshold and (total == 0 or math / total > 0.02):
                return True
            if self._is_scanned(page):
                return True
        except Exception as e:
            self.log(f"   ⚠️  routing heuristic failed ({e}); using fast path.")
        return False

    # ------------------------------------------------------------------
    # SURGICAL: formula-box detection from char geometry (no model)
    # ------------------------------------------------------------------

    def _math_boxes_from_chars(self, page) -> list[tuple[float, float, float, float]]:
        """Detect formula regions from pdf_oxide character geometry."""
        chars = getattr(page, "chars", None)
        if not chars:
            return []
        raw: list[list[float]] = []
        for c in chars:
            bbox = getattr(c, "bbox", None)
            if not bbox or len(bbox) < 4:
                continue
            fn = (getattr(c, "font_name", "") or "").lower()
            ch = getattr(c, "char", "") or ""
            if any(k in fn for k in _MATH_FONT_KEYWORDS) or (ch and _is_math_unicode(ch)):
                x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
                raw.append([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), 1])
        if not raw:
            return []

        HGAP, VGAP = 14.0, 8.0

        def _overlaps(a, b):
            return not (b[2] < a[0] - HGAP or b[0] > a[2] + HGAP
                        or b[3] < a[1] - VGAP or b[1] > a[3] + VGAP)

        def _merge_into(dst, b):
            dst[0] = min(dst[0], b[0]); dst[1] = min(dst[1], b[1])
            dst[2] = max(dst[2], b[2]); dst[3] = max(dst[3], b[3]); dst[4] += b[4]

        raw.sort(key=lambda b: (b[1], b[0]))
        boxes: list[list[float]] = []
        for b in raw:
            for u in boxes:
                if _overlaps(u, b):
                    _merge_into(u, b)
                    break
            else:
                boxes.append(list(b))

        # Coalesce boxes that became adjacent after the first greedy pass
        changed = True
        while changed:
            changed = False
            out: list[list[float]] = []
            for b in boxes:
                for u in out:
                    if _overlaps(u, b):
                        _merge_into(u, b)
                        changed = True
                        break
                else:
                    out.append(b)
            boxes = out

        return [(b[0], b[1], b[2], b[3]) for b in boxes
                if b[4] >= self.cfg.min_formula_math_chars]

    def _crop_pil(self, img, box, page_h: float, dpi: int):
        """Crop a PIL image to a point-space box."""
        try:
            s = dpi / 72.0
            x0, y0, x1, y1 = box[:4]
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            pad = self.cfg.formula_pad_pts
            x0 -= pad; y0 -= pad; x1 += pad; y1 += pad
            # pdf_oxide: points, origin bottom-left, y up → flip for image (top-left).
            left = max(0, int(x0 * s))
            right = int(x1 * s)
            top = max(0, int((page_h - y1) * s))
            bottom = int((page_h - y0) * s)
            if right <= left or bottom <= top:
                return None
            return img.crop((left, top, right, bottom))
        except Exception:
            return None

    def _region_text(self, doc, page, index: int, box) -> str:
        """Extract text for a rectangular region via pdf_oxide."""
        x0, y0, x1, y1 = box[:4]
        try:
            reg = doc.within(index, (x0, y0, x1 - x0, y1 - y0))
            t = reg.extract_text()
            if t:
                return t
        except Exception:
            pass
        # Fallback: reconstruct from chars inside the box
        try:
            items = []
            for c in getattr(page, "chars", []) or []:
                bb = getattr(c, "bbox", None)
                if not bb or len(bb) < 4:
                    continue
                cx = (bb[0] + bb[2]) / 2.0
                cy = (bb[1] + bb[3]) / 2.0
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    items.append((round(cy, 1), cx, getattr(c, "char", "") or ""))
            items.sort(key=lambda t: (-t[0], t[1]))
            return "".join(ch for _, _, ch in items)
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # SURGICAL: formula recognition + splicing
    # ------------------------------------------------------------------

    def _latex_wrap(self, latex: str, box, page_line_height: float) -> str:
        """Decide inline ``$…$`` vs display ``$$…$$`` based on box dimensions."""
        x0, y0, x1, y1 = box
        is_display = (y1 - y0) > 1.6 * page_line_height or (x1 - x0) > self.cfg.formula_inline_max_width_pts
        return f"$$\n{latex}\n$$" if is_display else f"${latex}$"

    def _surgical_page_markdown(self, doc, page, index: int, work_dir: Path) -> str:
        """Fast path + surgical formula pass via RapidLaTeXOCR."""
        fast_md = self._fast_page_markdown(page)
        formula_eng = self.rapid.formula()
        if formula_eng is None:
            return fast_md

        boxes = self._math_boxes_from_chars(page)
        if not boxes:
            return fast_md

        rendered = self._render_page_to_pil(doc, page, index, self.cfg.formula_dpi)
        if rendered is None:
            self.log(f"   ⚠️  page {index + 1}: render unavailable; formulas left as text.")
            return fast_md
        img, _w, page_h = rendered

        # Estimate line height from page geometry
        page_line_height = page_h / max(1, len(getattr(page, "chars", []) or [1]))

        crop_paths, kept = [], []
        for j, box in enumerate(boxes):
            crop = self._crop_pil(img, box, page_h, self.cfg.formula_dpi)
            if crop is None or crop.width < 4 or crop.height < 4:
                continue
            p = work_dir / f"_formula_p{index}_{j}.png"
            try:
                crop.save(str(p))
            except Exception:
                continue
            crop_paths.append(str(p))
            kept.append(box)
        if not crop_paths:
            return fast_md

        self.log(f"   ∑ page {index + 1}: {len(crop_paths)} formula region(s) → RapidLaTeXOCR")

        # Recognize formulas (one at a time — RapidLaTeXOCR is single-crop)
        latexes = [self.rapid.recognize_formula(p) for p in crop_paths]

        # Splice LaTeX back into fast markdown
        repls = []
        for box, latex in zip(kept, latexes):
            if not latex:
                continue
            needle = self._region_text(doc, page, index, box)
            wrapped = self._latex_wrap(latex, box, page_line_height)
            repls.append((needle, wrapped))
        return self._splice(fast_md, repls)

    # ------------------------------------------------------------------
    # Full ONNX structure page (AUTO/ALWAYS, and SURGICAL scans)
    # ------------------------------------------------------------------

    def _reading_order(self, items: list[tuple]) -> list[tuple]:
        """Sort items by reading order (top-to-bottom, left-to-right)."""
        if not items:
            return items
        # Simple single-column: sort by y then x
        return sorted(items, key=lambda t: (t[0][1] if isinstance(t[0], (list, tuple)) else 0,
                                            t[0][0] if isinstance(t[0], (list, tuple)) else 0))

    def _full_structure_page_markdown(self, doc, page, index: int, work_dir: Path) -> Optional[str]:
        """Full ONNX pipeline: layout → OCR / formula / table per region."""
        rendered = self._render_page_to_pil(doc, page, index, self.cfg.render_dpi)
        if rendered is None:
            return None
        img, _w, page_h = rendered

        if np is None:
            self.log("   ⚠️  numpy not available for ONNX pipeline; using fast path.")
            return None

        page_np = np.asarray(img)

        regions = self.rapid.layout_regions(page_np)
        if not regions:
            # Layout missing → OCR whole page as text
            lines = self.rapid.ocr_lines(page_np)
            return "\n\n".join(t for _b, t, _s in self._reading_order(lines)) or None

        blocks = []
        for box, label, _score in self._reading_order(regions):
            crop = self._crop_pil(img, box, page_h, self.cfg.render_dpi)
            crop_np = np.asarray(crop) if crop else None
            lab = (label or "").lower()

            if lab in ("table",):
                html = self.rapid.table_html(crop_np)
                if html:
                    blocks.append(html_tables_to_gfm(html) if self.cfg.convert_html_tables else html)
            elif lab in ("formula", "equation", "isolate_formula"):
                p = work_dir / f"_reg_f_{len(blocks)}.png"
                crop.save(str(p))
                latex = self.rapid.recognize_formula(str(p))
                if latex:
                    blocks.append(f"$$\n{latex}\n$$")
            elif lab in ("figure", "image"):
                p = work_dir / f"_reg_fig_{len(blocks)}.png"
                crop.save(str(p))
                blocks.append(f"![]({p.name})")
            else:
                lines = self.rapid.ocr_lines(crop_np)
                text = " ".join(t for _b, t, _s in lines).strip()
                if lab == "title":
                    text = f"## {text}"
                if text:
                    blocks.append(text)

        md = "\n\n".join(b for b in blocks if b)
        return md or None

    # ------------------------------------------------------------------
    # Splice helpers
    # ------------------------------------------------------------------

    def _ws_replace(self, md: str, needle: str, block: str) -> Optional[str]:
        """Whitespace-tolerant replacement."""
        tokens = [re.escape(t) for t in needle.split() if t]
        if not tokens:
            return None
        m = re.search(r"\s+".join(tokens), md)
        if not m:
            return None
        return md[:m.start()] + block + md[m.end():]

    def _splice(self, md: str, repls: list[tuple[str, str]]) -> str:
        """Apply a list of (needle, replacement) to ``md``."""
        leftovers = []
        for needle, block in repls:
            needle = (needle or "").strip()
            if needle and needle in md:
                md = md.replace(needle, block, 1)
                continue
            new = self._ws_replace(md, needle, block) if needle else None
            if new is not None:
                md = new
            else:
                leftovers.append(block)
        if leftovers:
            md = md.rstrip() + "\n\n" + "\n\n".join(leftovers)
        return md

    # ------------------------------------------------------------------
    # Per-page routing
    # ------------------------------------------------------------------

    def _route_page(self, doc, page, index: int, work_dir: Path) -> str:
        """Route a single page through the appropriate pipeline."""
        mode = self.cfg.routing_mode
        if not self.cfg.use_onnx or mode == RoutingMode.NEVER:
            return self._fast_page_markdown(page)

        if mode == RoutingMode.SURGICAL:
            if self._is_scanned(page):
                self._ensure_full_structure()
                md = self._full_structure_page_markdown(doc, page, index, work_dir)
                if md and md.strip():
                    self.log(f"   🖼  page {index + 1}: scanned → ONNX layout+OCR")
                    return md
                return self._fast_page_markdown(page)
            return self._surgical_page_markdown(doc, page, index, work_dir)

        # AUTO / ALWAYS
        if self.rapid.ocr() is not None and self._needs_paddle(page):
            md = self._full_structure_page_markdown(doc, page, index, work_dir)
            if md and md.strip():
                self.log(f"   🧠 page {index + 1} → ONNX layout+OCR (full pipeline)")
                return md
            self.log(f"   ⚠️  page {index + 1}: render unavailable; using fast path.")
        return self._fast_page_markdown(page)

    # ------------------------------------------------------------------
    # Code block detection (fast path enhancement)
    # ------------------------------------------------------------------

    def _wrap_code_blocks(self, page) -> list[str]:
        """Detect monospace runs and wrap them in fenced code blocks."""
        if not self.cfg.detect_code_blocks:
            return []
        chars = getattr(page, "chars", None)
        if not chars:
            return []

        # Group consecutive monospace chars into lines
        lines: dict[float, list[str]] = {}
        for c in chars:
            bbox = getattr(c, "bbox", None)
            if not bbox or len(bbox) < 4:
                continue
            fn = (getattr(c, "font_name", "") or "").lower()
            ch = getattr(c, "char", "") or ""
            if _is_mono_font(fn):
                y = round(bbox[1], 1)
                lines.setdefault(y, []).append(ch)

        blocks: list[str] = []
        for y in sorted(lines.keys()):
            text = "".join(lines[y])
            if len(text) > 4:  # skip noise
                blocks.append(f"```\n{text}\n```")
        return blocks

    # ------------------------------------------------------------------
    # Top-level conversion
    # ------------------------------------------------------------------

    def convert_pdf(
        self,
        path: Path,
        work_dir: Path,
        should_continue: Callable[[], bool],
        on_page: Callable[[int, int], None],
    ) -> str:
        """Convert a PDF to markdown.

        Args:
            path: Path to the PDF file.
            work_dir: Temporary directory for intermediate renders/crops.
            should_continue: Callback that returns False to cancel.
            on_page: Callback(page_index, page_total) for progress.

        Returns:
            Complete markdown string.
        """
        if PdfDocument is None:
            raise RuntimeError("pdf_oxide is not installed (pip install pdf_oxide).")

        md_blocks: list[str] = []
        with PdfDocument(str(path)) as doc:
            try:
                n_pages = len(doc)
            except TypeError:
                n_pages = doc.page_count()

            for i, page in enumerate(doc):
                if not should_continue():
                    break
                on_page(i, n_pages)

                page_imgs: list[Path] = []
                if self.cfg.extract_images:
                    page_imgs = self._extract_page_images(doc, page, i, work_dir)

                page_md = self._route_page(doc, page, i, work_dir)

                if (self.cfg.extract_images and page_imgs
                        and self.cfg.append_unreferenced_images
                        and not re.search(r"!\[.*?\]\(.*?\)", page_md)):
                    gallery = "\n".join(f"![]({p.name})" for p in page_imgs)
                    page_md = f"{page_md}\n\n{gallery}"

                md_blocks.append(page_md)

        return "\n\n---\n\n".join(md_blocks)

    def convert_office(self, path: Path) -> str:
        """Convert a DOCX/XLSX/PPTX to markdown via office_oxide."""
        if OfficeDocument is None:
            raise RuntimeError("office_oxide is not installed; cannot convert Office files.")
        with OfficeDocument.open(str(path)) as doc:
            return doc.to_markdown()
