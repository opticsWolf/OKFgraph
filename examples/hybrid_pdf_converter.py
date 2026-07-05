#!/usr/bin/env python3
"""
Hybrid Document-to-Markdown Converter (PySide6 + pdf_oxide + PaddleOCR)
======================================================================

Fast native extraction (pdf_oxide, Rust core) for clean text pages, with an
optional AI pass routed only where it earns its keep. Three routing modes:

* NEVER    — fast path only, no AI.
* AUTO     — flag math/scanned pages and run the *whole* page through
             PaddleOCR PP-StructureV3.
* ALWAYS   — every page through PP-StructureV3.
* SURGICAL — minimize the neural surface: locate formula boxes from pdf_oxide's
             own character geometry (no layout network), crop just those regions,
             batch them through the light PP-FormulaNet model, and splice the
             LaTeX back into pdf_oxide's fast markdown. Genuine text-less scans
             still fall back to the full PP-StructureV3 pipeline, which is loaded
             lazily only when the first scan page is hit.

Why SURGICAL is fast: rasterizing a page is cheap; running ~6 neural nets across
the whole page is not. SURGICAL runs one small model over a few hundred pixels
per equation instead of the full pipeline over millions of page pixels, and
batches every crop in the document into as few predict() calls as possible.

Design notes
------------
* PaddleOCR 3.x removed `PPStructure`; `PPStructureV3` + `FormulaRecognition`
  (PP-FormulaNet) are used. Formula results carry LaTeX at `res["rec_formula"]`.
* pdf_oxide API: `page.render(...)`, `doc.extract_image_bytes(i)`,
  `page.markdown(detect_headings=...)`, `page.chars`, `doc.within(i,(x,y,w,h))`.
* Every external call is defensive: version drift degrades to the fast path with
  a log line instead of crashing the batch.
* Core pipeline (`HybridConverter`) is Qt-independent and unit-testable.

Optional runtime deps (all guarded):
    pip install pdf_oxide                        # fast native pass (MIT)
    pip install paddlepaddle paddleocr           # AI passes        (Apache-2.0)
    pip install pillow numpy                      # render / cropping for AI passes
    pip install office_oxide                      # docx/xlsx/pptx -> markdown
"""

from __future__ import annotations

import io
import re
import sys
import shutil
import hashlib
import warnings
from dataclasses import dataclass
from enum import Enum
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Optional

# ── PySide6 ─────────────────────────────────────────────────────────────────
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QLabel, QTextBrowser, QWidget, QFileDialog,
    QPushButton, QProgressBar, QFrame, QCheckBox, QGroupBox,
    QComboBox, QLineEdit, QSpinBox, QFormLayout,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent

# ── Optional heavy/native deps (always guarded) ─────────────────────────────
try:
    from pdf_oxide import PdfDocument
except ImportError:
    PdfDocument = None

try:
    from office_oxide import Document as OfficeDocument
except ImportError:
    OfficeDocument = None

# PaddleOCR 3.x -> PPStructureV3 (full pipeline). Keep a fallback name for 2.x.
_PADDLE_IS_V3 = False
try:
    from paddleocr import PPStructureV3 as _PaddleStructure  # noqa: N814
    _PADDLE_IS_V3 = True
except ImportError:
    try:
        from paddleocr import PPStructure as _PaddleStructure  # legacy 2.x
    except ImportError:
        _PaddleStructure = None

# Light, single-purpose formula model for SURGICAL mode.
try:
    from paddleocr import FormulaRecognition
except ImportError:
    FormulaRecognition = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from okfgraph.router import OKFRouter
except ImportError:
    OKFRouter = None


SUPPORTED_EXTS = {".pdf", ".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt"}
OFFICE_EXTS = SUPPORTED_EXTS - {".pdf"}

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


def _is_math_unicode(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _MATH_UNICODE_RANGES)


# ── okf-asset:// staging convention ──────────────────────────────────────────
ASSET_STORE_DIRNAME = "_assets"
_IMG_LINK_RE = re.compile(r"!\[(?P<alt>.*?)\]\((?P<src>.*?)\)")


def _split_src_and_title(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return raw
    m = re.match(r"^(<[^>]+>|\S+)", raw)
    part = m.group(1) if m else raw
    if part.startswith("<") and part.endswith(">"):
        part = part[1:-1]
    return part


def _asset_id(concept_stem: str, occurrence: int, img_bytes: bytes) -> str:
    h = hashlib.sha256()
    h.update(f"{concept_stem}|{occurrence}|".encode("utf-8"))
    h.update(img_bytes)
    return f"img_{h.hexdigest()[:16]}"


def _stage_images_as_okf_assets(
    md: str, image_src_dir: Path, source_path: Path, out_dir: Path, concept_stem: str
) -> tuple[str, int]:
    """Rewrite ![](local) links to ![](okf-asset://<id>) and copy bytes into _assets/."""
    assets_dir = out_dir / ASSET_STORE_DIRNAME
    occurrence = {"n": 0}

    def _repl(match: "re.Match") -> str:
        alt = match.group("alt").strip()
        src = _split_src_and_title(match.group("src"))
        low = src.lower()
        if (not src
                or low.startswith(("http://", "https://", "okf-asset://", "data:"))
                or src.startswith(f"{ASSET_STORE_DIRNAME}/")):
            return match.group(0)

        cand = image_src_dir / Path(src).name
        if not cand.is_file():
            cand = source_path.parent / src
        if not cand.is_file() and Path(src).is_absolute():
            cand = Path(src)
        if not cand.is_file():
            return match.group(0)

        occurrence["n"] += 1
        data = cand.read_bytes()
        asset_id = _asset_id(concept_stem, occurrence["n"], data)
        suffix = cand.suffix.lower() or ".bin"
        assets_dir.mkdir(parents=True, exist_ok=True)
        dest = assets_dir / f"{asset_id}{suffix}"
        if not dest.exists():
            shutil.copy2(cand, dest)
        return f"![{alt}](okf-asset://{asset_id})"

    new_md = _IMG_LINK_RE.sub(_repl, md)
    return new_md, occurrence["n"]


# ── Minimal, dependency-free HTML-table -> GFM converter ─────────────────────
class _SimpleTableParser(HTMLParser):
    """Parses ONE <table> into rows of cell-text. Bails (complex=True) on row/colspans."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.header_flags: list[bool] = []
        self._cur: list[str] = []
        self._buf: list[str] = []
        self._in_cell = False
        self._row_has_th = False
        self.complex = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "tr":
            self._cur = []
            self._row_has_th = False
        elif tag in ("td", "th"):
            if attrs.get("colspan") not in (None, "1") or attrs.get("rowspan") not in (None, "1"):
                self.complex = True
            self._in_cell = True
            self._buf = []
            if tag == "th":
                self._row_has_th = True
        elif tag == "br":
            if self._in_cell:
                self._buf.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            text = re.sub(r"\s+", " ", "".join(self._buf)).strip().replace("|", r"\|")
            self._cur.append(text)
            self._in_cell = False
        elif tag == "tr":
            if self._cur:
                self.rows.append(self._cur)
                self.header_flags.append(self._row_has_th)

    def handle_data(self, data):
        if self._in_cell:
            self._buf.append(data)


def _one_table_to_gfm(html: str) -> Optional[str]:
    p = _SimpleTableParser()
    try:
        p.feed(html)
    except Exception:
        return None
    if p.complex or not p.rows:
        return None
    ncols = max(len(r) for r in p.rows)
    rows = [r + [""] * (ncols - len(r)) for r in p.rows]
    header_idx = next((i for i, f in enumerate(p.header_flags) if f), 0)
    header = rows[header_idx]
    body = [r for i, r in enumerate(rows) if i != header_idx]
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(["---"] * ncols) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


def html_tables_to_gfm(md: str) -> str:
    """Replace simple <table>…</table> blocks with GFM pipe tables; leave complex ones as-is."""
    def _repl(m: "re.Match") -> str:
        gfm = _one_table_to_gfm(m.group(0))
        return f"\n\n{gfm}\n\n" if gfm else m.group(0)
    return re.sub(r"<table\b.*?</table>", _repl, md, flags=re.IGNORECASE | re.DOTALL)


# ── Configuration ────────────────────────────────────────────────────────────
class RoutingMode(str, Enum):
    AUTO = "auto"          # heuristics per page -> full PP-StructureV3 on flagged pages
    SURGICAL = "surgical"  # formula crops via PP-FormulaNet; full pipeline only for scans
    ALWAYS = "always"      # every page -> full PP-StructureV3
    NEVER = "never"        # fast path only (no AI)


@dataclass
class ConverterConfig:
    extract_images: bool = True
    use_paddle: bool = True
    routing_mode: RoutingMode = RoutingMode.AUTO
    render_dpi: int = 300              # for scanned-page OCR (needs resolution)
    ocr_lang: str = "en"
    device: str = "cpu"               # "cpu" | "gpu"
    detect_headings: bool = True
    convert_html_tables: bool = True
    append_unreferenced_images: bool = True
    # AUTO / scan detection
    math_char_threshold: int = 30
    scanned_text_threshold: int = 50
    # SURGICAL formula pass
    formula_model_name: str = "PP-FormulaNet-S"  # -S = fast; _plus-M / _plus-L = accurate
    formula_dpi: int = 200            # crops of digital formulas need less than scans
    formula_batch_size: int = 8
    formula_pad_pts: float = 4.0      # padding around a formula box (points)
    min_formula_math_chars: int = 3   # skip stray single math glyphs in body text


# ── Core, Qt-independent pipeline ────────────────────────────────────────────
class HybridConverter:
    def __init__(self, config: ConverterConfig, log: Callable[[str], None] = print):
        self.cfg = config
        self.log = log
        self.paddle = None    # full PP-StructureV3 pipeline (heavy)
        self.formula = None   # PP-FormulaNet single-purpose model (light)

    # --- model lifecycle -----------------------------------------------------
    def ensure_models(self) -> None:
        if not self.cfg.use_paddle:
            return
        mode = self.cfg.routing_mode
        if mode == RoutingMode.NEVER:
            return
        if mode == RoutingMode.SURGICAL:
            self._ensure_formula()  # light; full pipeline stays lazy until a scan
        else:
            self._ensure_full_structure()

    def _ensure_full_structure(self) -> None:
        if self.paddle is not None or _PaddleStructure is None:
            return
        self.log("⚙️  Loading PaddleOCR PP-StructureV3 (full pipeline)…")
        try:
            if _PADDLE_IS_V3:
                self.paddle = _PaddleStructure(
                    lang=self.cfg.ocr_lang,
                    device="gpu" if self.cfg.device == "gpu" else "cpu",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            else:  # legacy 2.x signature
                self.paddle = _PaddleStructure(
                    show_log=False, recovery=True, lang=self.cfg.ocr_lang,
                    use_gpu=(self.cfg.device == "gpu"),
                )
            self.log("✅ PP-StructureV3 ready.")
        except Exception as e:  # noqa: BLE001
            self.paddle = None
            self.log(f"⚠️  Could not initialize PP-StructureV3 ({e}). Full pipeline disabled.")

    def _ensure_formula(self) -> None:
        if self.formula is not None:
            return
        if FormulaRecognition is None:
            self.log("⚠️  paddleocr FormulaRecognition unavailable; math will pass through as text.")
            return
        self.log(f"⚙️  Loading {self.cfg.formula_model_name} (surgical formula pass)…")
        try:
            self.formula = FormulaRecognition(model_name=self.cfg.formula_model_name)
            self.log("✅ Formula model ready.")
        except Exception as e:  # noqa: BLE001
            self.formula = None
            self.log(f"⚠️  Could not load formula model ({e}); math will pass through as text.")

    def close(self) -> None:
        for attr in ("paddle", "formula"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    close = getattr(obj, "close", None)
                    if callable(close):
                        close()
                except Exception:  # noqa: BLE001
                    pass
                setattr(self, attr, None)

    # --- pdf_oxide fast path -------------------------------------------------
    def _fast_page_markdown(self, page) -> str:
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
        except Exception as e:  # noqa: BLE001
            self.log(f"   ⚠️  image extraction failed on page {index + 1}: {e}")
            return paths
        try:
            for n, im in enumerate(getattr(page, "images", []) or []):
                p = img_dir / f"p{index}_img{n}.png"
                if hasattr(im, "save"):
                    im.save(str(p))
                    paths.append(p)
                elif getattr(im, "data", None):
                    p.write_bytes(im.data)
                    paths.append(p)
        except Exception as e:  # noqa: BLE001
            self.log(f"   ⚠️  fallback image extraction failed on page {index + 1}: {e}")
        return paths

    # --- rendering -----------------------------------------------------------
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
            except Exception:  # noqa: BLE001
                continue
        return None

    def _obj_to_pil(self, obj):
        if obj is None or PILImage is None:
            return None
        try:
            if hasattr(obj, "crop") and hasattr(obj, "save"):     # already PIL
                return obj
            if isinstance(obj, (bytes, bytearray)):
                return PILImage.open(io.BytesIO(bytes(obj)))
            if isinstance(obj, dict) and obj.get("data"):
                return PILImage.open(io.BytesIO(obj["data"]))
            if np is not None and isinstance(obj, np.ndarray):
                return PILImage.fromarray(obj)
        except Exception:  # noqa: BLE001
            return None
        return None

    def _render_page_to_png(self, doc, page, index: int, out_png: Path, dpi: int = 0) -> bool:
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
        except Exception as e:  # noqa: BLE001
            self.log(f"   ⚠️  could not save render of page {index + 1}: {e}")
            return False
        return out_png.is_file()

    def _render_page_to_pil(self, doc, page, index: int, dpi: int):
        """Return (PIL.Image RGB, page_width_pts, page_height_pts) or None."""
        obj = self._render_raw(doc, page, dpi)
        img = self._obj_to_pil(obj)
        if img is None:
            return None
        try:
            img = img.convert("RGB")
        except Exception:  # noqa: BLE001
            return None
        w = float(getattr(page, "width", 0) or 0)
        h = float(getattr(page, "height", 0) or 0)
        if h <= 0:
            return None  # cannot map point-space boxes to pixels without page height
        return img, w, h

    # --- routing signals -----------------------------------------------------
    def _page_math_signal(self, page) -> tuple[int, int]:
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
        try:
            chars = getattr(page, "chars", None)
            total = len(chars) if chars is not None else len(getattr(page, "text", "") or "")
            n_images = len(getattr(page, "images", []) or [])
            return total < self.cfg.scanned_text_threshold and n_images > 0
        except Exception:  # noqa: BLE001
            return False

    def _needs_paddle(self, page) -> bool:
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
        except Exception as e:  # noqa: BLE001
            self.log(f"   ⚠️  routing heuristic failed ({e}); using fast path.")
        return False

    # --- SURGICAL: formula-box detection from char geometry (no model) -------
    def _math_boxes_from_chars(self, page) -> list[tuple[float, float, float, float]]:
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

        # coalesce boxes that became adjacent after the first greedy pass
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
        try:
            s = dpi / 72.0
            x0, y0, x1, y1 = box[:4]
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            pad = self.cfg.formula_pad_pts
            x0 -= pad; y0 -= pad; x1 += pad; y1 += pad
            # pdf_oxide: points, origin bottom-left, y up -> flip for image (top-left).
            left = max(0, int(x0 * s))
            right = int(x1 * s)
            top = max(0, int((page_h - y1) * s))
            bottom = int((page_h - y0) * s)
            if right <= left or bottom <= top:
                return None
            return img.crop((left, top, right, bottom))
        except Exception:  # noqa: BLE001
            return None

    def _region_text(self, doc, page, index: int, box) -> str:
        """pdf_oxide's own text for a rectangle — best chance of matching its markdown."""
        x0, y0, x1, y1 = box[:4]
        try:
            reg = doc.within(index, (x0, y0, x1 - x0, y1 - y0))
            t = reg.extract_text()
            if t:
                return t
        except Exception:  # noqa: BLE001
            pass
        # fallback: reconstruct from chars inside the box
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
            items.sort(key=lambda t: (-t[0], t[1]))  # top-to-bottom (y up), left-to-right
            return "".join(ch for _, _, ch in items)
        except Exception:  # noqa: BLE001
            return ""

    def _extract_rec_formula(self, res) -> Optional[str]:
        try:
            if hasattr(res, "get"):
                v = res.get("rec_formula")
                if v:
                    return v
        except Exception:  # noqa: BLE001
            pass
        for holder in (getattr(res, "json", None), res if isinstance(res, dict) else None):
            if isinstance(holder, dict):
                inner = holder.get("res", holder)
                if isinstance(inner, dict) and inner.get("rec_formula"):
                    return inner["rec_formula"]
        return None

    def _recognize_formulas(self, crop_paths: list[str]) -> list[Optional[str]]:
        if self.formula is None:
            return [None] * len(crop_paths)
        results = []
        try:
            results = list(self.formula.predict(crop_paths,
                                                batch_size=self.cfg.formula_batch_size))
        except Exception as e:  # noqa: BLE001
            self.log(f"   ⚠️  batched formula predict failed ({e}); retrying per-crop.")
            results = []
            for p in crop_paths:
                try:
                    results.extend(list(self.formula.predict(p, batch_size=1)))
                except Exception:  # noqa: BLE001
                    results.append(None)
        latexes = [self._extract_rec_formula(r) if r is not None else None for r in results]
        if len(latexes) < len(crop_paths):
            latexes += [None] * (len(crop_paths) - len(latexes))
        return latexes[:len(crop_paths)]

    def _ws_replace(self, md: str, needle: str, block: str) -> Optional[str]:
        tokens = [re.escape(t) for t in needle.split() if t]
        if not tokens:
            return None
        m = re.search(r"\s+".join(tokens), md)
        if not m:
            return None
        return md[:m.start()] + block + md[m.end():]

    def _splice(self, md: str, repls: list[tuple[str, str]]) -> str:
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

    def _surgical_page_markdown(self, doc, page, index: int, work_dir: Path) -> str:
        fast_md = self._fast_page_markdown(page)
        if self.formula is None:
            return fast_md
        boxes = self._math_boxes_from_chars(page)
        if not boxes:
            return fast_md
        rendered = self._render_page_to_pil(doc, page, index, self.cfg.formula_dpi)
        if rendered is None:
            self.log(f"   ⚠️  page {index + 1}: render unavailable; formulas left as text.")
            return fast_md
        img, _w, page_h = rendered

        crop_paths, kept = [], []
        for j, box in enumerate(boxes):
            crop = self._crop_pil(img, box, page_h, self.cfg.formula_dpi)
            if crop is None or crop.width < 4 or crop.height < 4:
                continue
            p = work_dir / f"_formula_p{index}_{j}.png"
            try:
                crop.save(str(p))
            except Exception:  # noqa: BLE001
                continue
            crop_paths.append(str(p))
            kept.append(box)
        if not crop_paths:
            return fast_md

        self.log(f"   ∑ page {index + 1}: {len(crop_paths)} formula region(s) → PP-FormulaNet")
        latexes = self._recognize_formulas(crop_paths)
        repls = []
        for box, latex in zip(kept, latexes):
            if not latex:
                continue
            needle = self._region_text(doc, page, index, box)
            repls.append((needle, f"$$\n{latex.strip()}\n$$"))
        return self._splice(fast_md, repls)

    # --- full PP-StructureV3 page (AUTO/ALWAYS, and SURGICAL scans) -----------
    def _result_markdown(self, res, img_dir: Path) -> str:
        info = getattr(res, "markdown", None)
        txt = ""
        images = {}
        if isinstance(info, str):
            txt = info
        elif info is not None and hasattr(info, "get"):
            raw = info.get("markdown_texts", "")
            txt = raw if isinstance(raw, str) else str(raw)
            images = info.get("markdown_images", {}) or {}
        elif info is not None:
            txt = str(info)
        for relpath, pil in images.items():
            try:
                pil.save(str(img_dir / Path(relpath).name))
            except Exception:  # noqa: BLE001
                pass
        return txt

    def _paddle_page_markdown(self, png_path: Path, img_dir: Path) -> str:
        try:
            output = self.paddle.predict(str(png_path))
        except Exception as e:  # noqa: BLE001
            self.log(f"   ⚠️  PP-StructureV3 predict failed: {e}")
            return ""
        parts = [self._result_markdown(res, img_dir) for res in output]
        md = "\n\n".join(p for p in parts if p)
        if self.cfg.convert_html_tables and md:
            md = html_tables_to_gfm(md)
        return md

    def _full_structure_page_markdown(self, doc, page, index: int, work_dir: Path) -> Optional[str]:
        if self.paddle is None:
            return None
        png = work_dir / f"_render_p{index}.png"
        if not self._render_page_to_png(doc, page, index, png):
            return None
        md = self._paddle_page_markdown(png, work_dir)
        return md or None

    # --- per-page routing ----------------------------------------------------
    def _route_page(self, doc, page, index: int, work_dir: Path) -> str:
        mode = self.cfg.routing_mode
        if not self.cfg.use_paddle or mode == RoutingMode.NEVER:
            return self._fast_page_markdown(page)

        if mode == RoutingMode.SURGICAL:
            if self._is_scanned(page):
                self._ensure_full_structure()  # lazy: only now do we pay the heavy load
                md = self._full_structure_page_markdown(doc, page, index, work_dir)
                if md and md.strip():
                    self.log(f"   🖼  page {index + 1}: scanned → PP-StructureV3 OCR")
                    return md
                return self._fast_page_markdown(page)
            return self._surgical_page_markdown(doc, page, index, work_dir)

        # AUTO / ALWAYS
        if self.paddle is not None and self._needs_paddle(page):
            md = self._full_structure_page_markdown(doc, page, index, work_dir)
            if md and md.strip():
                self.log(f"   🧠 page {index + 1} → PP-StructureV3 (full pipeline)")
                return md
            self.log(f"   ⚠️  page {index + 1}: render unavailable; using fast path.")
        return self._fast_page_markdown(page)

    # --- top-level per-file conversion --------------------------------------
    def convert_pdf(
        self,
        path: Path,
        work_dir: Path,
        should_continue: Callable[[], bool],
        on_page: Callable[[int, int], None],
    ) -> str:
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
                        and not _IMG_LINK_RE.search(page_md)):
                    gallery = "\n".join(f"![]({p.name})" for p in page_imgs)
                    page_md = f"{page_md}\n\n{gallery}"

                md_blocks.append(page_md)

        return "\n\n---\n\n".join(md_blocks)

    def convert_office(self, path: Path) -> str:
        if OfficeDocument is None:
            raise RuntimeError("office_oxide is not installed; cannot convert Office files.")
        with OfficeDocument.open(str(path)) as doc:
            return doc.to_markdown()


# ── Qt worker ────────────────────────────────────────────────────────────────
class ConversionWorker(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int, int)        # file index, file total
    page_signal = Signal(int, int)            # page index, page total (current file)
    finished_signal = Signal(int, int, int)   # success, fail, skipped

    def __init__(self, files, seen_hashes, config: ConverterConfig,
                 ingest_into_okf=False, db_path="okfgraph.db",
                 bundle_root=".", ingest_mode="text"):
        super().__init__()
        self.files = files
        self.seen_hashes = seen_hashes
        self.cfg = config
        self._running = True

        self.ingest_into_okf = ingest_into_okf
        self.db_path = db_path
        self.bundle_root = bundle_root
        self.ingest_mode = ingest_mode

        self.router = None
        self.converter = HybridConverter(config, log=self.log_signal.emit)

    @staticmethod
    def _compute_hash(filepath: Path) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(1 << 16):
                h.update(chunk)
        return h.hexdigest()

    def _ensure_router(self) -> None:
        if self.ingest_into_okf and self.router is None and OKFRouter is not None:
            self.log_signal.emit(f"⚙️  Loading okfgraph model (device={self.cfg.device})…")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.router = OKFRouter(
                    db_path=self.db_path, bundle_root=self.bundle_root,
                    embedding_dim=512,
                    device="cuda" if self.cfg.device == "gpu" else "cpu",
                )
            self.log_signal.emit("✅ Embedding model ready.")

    def run(self):
        success = fail = skipped = ingest_ok = ingest_fail = 0
        total = len(self.files)
        try:
            self._ensure_router()
            self.converter.ensure_models()
            for idx, (file_path, out_dir) in enumerate(self.files, 1):
                if not self._running:
                    break
                self.progress_signal.emit(idx, total)
                result = self._convert_one(Path(file_path), Path(out_dir))
                if result == "success":
                    success += 1
                elif result == "skipped":
                    skipped += 1
                elif result == "ingest-ok":
                    success += 1
                    ingest_ok += 1
                elif result == "ingest-fail":
                    success += 1
                    ingest_fail += 1
                else:
                    fail += 1
        except Exception as e:  # noqa: BLE001
            self.log_signal.emit(f"❌ Fatal worker error: {e}")
        finally:
            self.converter.close()
            if self.router is not None:
                try:
                    if self.router.reindex(force=False):
                        self.log_signal.emit("🔁 Rebuilt search indexes for the batch.")
                except Exception as e:  # noqa: BLE001
                    self.log_signal.emit(f"⚠️  Index rebuild failed: {e}")
                try:
                    self.router.close()
                except Exception:  # noqa: BLE001
                    pass
                self.router = None

        if ingest_fail:
            self.log_signal.emit(f"⚠️  {ingest_fail} file(s) converted but failed okfgraph import.")
        self.finished_signal.emit(success, fail, skipped)

    def _convert_one(self, path: Path, out_dir: Path) -> str:
        ext = path.suffix.lower()
        self.log_signal.emit(f"➡️ <b>{path.name}</b>")

        try:
            file_hash = self._compute_hash(path)
        except Exception as e:  # noqa: BLE001
            self.log_signal.emit(f"❌ {path.name}: cannot read file ({e})")
            return "fail"
        if file_hash in self.seen_hashes:
            self.log_signal.emit(f"⏩ Skipped identical file: {path.name}")
            return "skipped"
        self.seen_hashes.add(file_hash)

        tmpctx = TemporaryDirectory() if self.cfg.extract_images else None
        render_ctx = tmpctx or TemporaryDirectory()
        work_dir = Path(render_ctx.name)

        try:
            if ext == ".pdf":
                md_output = self.converter.convert_pdf(
                    path, work_dir,
                    should_continue=lambda: self._running,
                    on_page=lambda i, n: self.page_signal.emit(i, n),
                )
            elif ext in OFFICE_EXTS:
                md_output = self.converter.convert_office(path)
            else:
                self.log_signal.emit(f"❌ {path.name}: unsupported extension {ext}")
                return "fail"

            if not self._running:
                return "skipped"

            base_name = path.stem
            md_path = out_dir / f"{base_name}.md"
            counter = 1
            while md_path.exists():
                md_path = out_dir / f"{base_name}_v{counter}.md"
                counter += 1

            img_count = 0
            if self.cfg.extract_images:
                md_output, img_count = _stage_images_as_okf_assets(
                    md_output, work_dir, path, out_dir, md_path.stem
                )

            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(md_output, encoding="utf-8")
            extra = f" — {img_count} image(s)" if img_count else ""
            self.log_signal.emit(
                f"✅ <a href='{md_path.absolute().as_uri()}'>{md_path.name}</a>{extra}"
            )

            if self.ingest_into_okf and self.router:
                try:
                    cid = self.router.import_from_okf(
                        md_path, mode=self.ingest_mode, rebuild_indexes=False
                    )
                    self.log_signal.emit(f"📈 Imported into okfgraph as '{cid}'")
                    return "ingest-ok"
                except Exception as e:  # noqa: BLE001
                    self.log_signal.emit(f"⚠️  okfgraph import failed for {path.name}: {e}")
                    return "ingest-fail"
            return "success"

        except Exception as e:  # noqa: BLE001
            self.log_signal.emit(f"❌ {path.name}: {e}")
            return "fail"
        finally:
            render_ctx.cleanup()

    def stop(self):
        self._running = False
        self.wait(3000)


# ── Drag & drop area ─────────────────────────────────────────────────────────
class DropArea(QFrame):
    files_dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(130)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self._idle_style = ("DropArea { background-color: #ecf0f1; border: 3px dashed #bdc3c7; "
                            "border-radius: 15px; } QLabel { color: #7f8c8d; font-size: 15px; "
                            "font-weight: bold; }")
        self._active_style = ("DropArea { background-color: #eafaf1; border: 3px dashed #2ecc71; "
                             "border-radius: 15px; } QLabel { color: #27ae60; font-size: 15px; "
                             "font-weight: bold; }")
        self.setStyleSheet(self._idle_style)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        self.label = QLabel("📂 Drop files here\nPDF, DOCX, XLSX, PPTX")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            self.setStyleSheet(self._active_style)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet(self._idle_style)

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet(self._idle_style)
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()

    def mousePressEvent(self, event):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Documents", "",
            "Documents (*.pdf *.docx *.xlsx *.pptx);;All Files (*)"
        )
        if files:
            self.files_dropped.emit(files)


# ── Main window ──────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hybrid AI Document Converter")
        self.resize(700, 760)
        self.worker = None
        self.pending_files = []
        self.seen_hashes = set()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.drop_area = DropArea()
        self.drop_area.files_dropped.connect(self.add_files)
        layout.addWidget(self.drop_area)

        # -- AI pass options --
        paddle_box = QGroupBox("🧠 AI pass (PaddleOCR)")
        paddle_form = QFormLayout(paddle_box)
        self.paddle_cb = QCheckBox("Enable AI pass for math / scanned pages")
        self.paddle_cb.setChecked(True)
        paddle_form.addRow(self.paddle_cb)
        self.routing_combo = QComboBox()
        self.routing_combo.addItems(["surgical", "auto", "always", "never"])
        self.routing_combo.setToolTip(
            "surgical: formula crops only + full pipeline for scans (fastest)\n"
            "auto: whole page → PP-StructureV3 when flagged\n"
            "always: every page → PP-StructureV3\n"
            "never: fast path only"
        )
        paddle_form.addRow("Routing:", self.routing_combo)
        self.formula_combo = QComboBox()
        self.formula_combo.addItems(["PP-FormulaNet-S", "PP-FormulaNet_plus-M", "PP-FormulaNet_plus-L"])
        self.formula_combo.setToolTip("Used by surgical mode. -S is fastest; plus-M/L more accurate.")
        paddle_form.addRow("Formula model:", self.formula_combo)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(300)
        self.dpi_spin.setSingleStep(50)
        self.dpi_spin.setToolTip("Render DPI for scanned-page OCR.")
        paddle_form.addRow("Scan DPI:", self.dpi_spin)
        self.lang_edit = QLineEdit("en")
        paddle_form.addRow("OCR language:", self.lang_edit)
        layout.addWidget(paddle_box)

        self.images_cb = QCheckBox("Extract images as okf-asset:// links")
        self.images_cb.setChecked(True)
        layout.addWidget(self.images_cb)

        self.tables_cb = QCheckBox("Convert HTML tables to Markdown (GFM)")
        self.tables_cb.setChecked(True)
        layout.addWidget(self.tables_cb)

        # -- okfgraph ingest --
        self.ingest_box = QGroupBox("📈 Import into okfgraph (optional)")
        self.ingest_box.setCheckable(True)
        self.ingest_box.setChecked(False)
        ingest_form = QFormLayout(self.ingest_box)
        self.db_path_edit = QLineEdit("okfgraph.db")
        ingest_form.addRow("Database:", self.db_path_edit)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "gpu"])
        ingest_form.addRow("Device:", self.device_combo)
        layout.addWidget(self.ingest_box)

        # -- controls --
        btn_row = QHBoxLayout()
        self.status_lbl = QLabel("Ready")
        btn_row.addWidget(self.status_lbl)
        btn_row.addStretch()
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_all)
        btn_row.addWidget(self.clear_btn)
        self.convert_btn = QPushButton("▶ Convert")
        self.convert_btn.setEnabled(False)
        self.convert_btn.clicked.connect(self.start_conversion)
        btn_row.addWidget(self.convert_btn)
        layout.addLayout(btn_row)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.log_area = QTextBrowser()
        self.log_area.setOpenExternalLinks(True)
        layout.addWidget(self.log_area, stretch=1)

        if PdfDocument is None:
            self.log("⚠️ pdf_oxide not found — PDF conversion disabled. `pip install pdf_oxide`")
        if self.paddle_cb.isChecked() and FormulaRecognition is None and _PaddleStructure is None:
            self.log("⚠️ PaddleOCR not found — AI passes disabled. `pip install paddleocr paddlepaddle`")

    def add_files(self, paths: list):
        for p in paths:
            path = Path(p)
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
                item = (path, path.parent)
                if item not in self.pending_files:
                    self.pending_files.append(item)
        if self.pending_files:
            self.convert_btn.setEnabled(True)
            self.status_lbl.setText(f"{len(self.pending_files)} file(s) queued.")

    def clear_all(self):
        self.pending_files.clear()
        self.log_area.clear()
        self.convert_btn.setEnabled(False)
        self.status_lbl.setText("Ready")

    def log(self, msg: str):
        self.log_area.append(msg)

    def start_conversion(self):
        if not self.pending_files:
            return
        self.convert_btn.setEnabled(False)
        cfg = ConverterConfig(
            extract_images=self.images_cb.isChecked(),
            use_paddle=self.paddle_cb.isChecked(),
            routing_mode=RoutingMode(self.routing_combo.currentText()),
            render_dpi=self.dpi_spin.value(),
            ocr_lang=self.lang_edit.text().strip() or "en",
            device=self.device_combo.currentText(),
            convert_html_tables=self.tables_cb.isChecked(),
            formula_model_name=self.formula_combo.currentText(),
        )
        self.worker = ConversionWorker(
            self.pending_files.copy(), self.seen_hashes, cfg,
            ingest_into_okf=self.ingest_box.isChecked(),
            db_path=self.db_path_edit.text(),
        )
        self.worker.log_signal.connect(self.log)
        self.worker.progress_signal.connect(self._on_file_progress)
        self.worker.page_signal.connect(self._on_page_progress)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def _on_file_progress(self, current: int, total: int):
        self.progress.setValue(int(current / total * 100) if total else 0)

    def _on_page_progress(self, page_idx: int, page_total: int):
        if page_total > 1:
            self.status_lbl.setText(f"Page {page_idx + 1}/{page_total}…")

    def on_finished(self, success, fail, skipped):
        self.convert_btn.setEnabled(True)
        self.progress.setValue(100)
        self.status_lbl.setText(f"Done: {success} ok · {fail} fail · {skipped} skipped")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
