"""OnnxRapidEngine — all ONNX heavy passes behind lazy loaders.

Wraps RapidAI family packages (rapid_latex_ocr, rapidocr, rapid_layout,
rapid_table) so models are loaded on first use. A clean born-digital paper
loads only the tiny formula model, never the OCR/layout/table stack.

Every version-sensitive RapidAI call is flagged ``# VERIFY`` because the
packages move fast and their call signatures differ across minor versions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

# -- lazy imports: all guarded so the module loads without RapidAI installed --

try:
    from rapid_latex_ocr import LatexOCR  # VERIFY: class may be LaTeXOCR in some builds
except ImportError:  # pragma: no cover
    LatexOCR = None  # type: ignore[assignment]

try:
    from rapidocr import RapidOCR  # VERIFY: older wheels: rapidocr_onnxruntime
except ImportError:  # pragma: no cover
    RapidOCR = None  # type: ignore[assignment]

try:
    from rapid_layout import RapidLayout
except ImportError:  # pragma: no cover
    RapidLayout = None  # type: ignore[assignment]

try:
    from rapid_table import RapidTable
except ImportError:  # pragma: no cover
    RapidTable = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


class OnnxRapidEngine:
    """All ONNX heavy passes behind lazy loaders. No PaddlePaddle."""

    def __init__(
        self,
        log_fn: Callable[[str], None] = print,
        ort_providers: Optional[list[str]] = None,
    ) -> None:
        self.log = log_fn
        self.ort_providers = ort_providers
        self._formula: Optional[object] = None
        self._ocr: Optional[object] = None
        self._layout: Optional[object] = None
        self._table: Optional[object] = None

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def formula(self) -> Optional[object]:
        """Return the RapidLaTeXOCR instance (lazy)."""
        if self._formula is None:
            if LatexOCR is None:
                self.log("⚠️  rapid_latex_ocr not installed; math stays as text.")
                return None
            self.log("⚙️  Loading RapidLaTeXOCR (formula → LaTeX)…")
            try:
                self._formula = LatexOCR()  # VERIFY: pass model paths for offline
            except Exception as e:
                self.log(f"⚠️  Could not load RapidLaTeXOCR: {e}")
                return None
        return self._formula

    def ocr(self) -> Optional[object]:
        """Return the RapidOCR instance (lazy)."""
        if self._ocr is None and RapidOCR is not None:
            self.log("⚙️  Loading RapidOCR (text det+rec)…")
            try:
                self._ocr = RapidOCR(
                    use_angle_cls=True,
                    cls_use_cuda=False,
                    det_use_cuda=False,
                    rec_use_cuda=False,
                )  # VERIFY: RapidOCR(providers=self.ort_providers) in some builds
            except Exception as e:
                self.log(f"⚠️  Could not load RapidOCR: {e}")
                return None
        return self._ocr

    def layout(self) -> Optional[object]:
        """Return the RapidLayout instance (lazy)."""
        if self._layout is None and RapidLayout is not None:
            self.log("⚙️  Loading RapidLayout…")
            try:
                self._layout = RapidLayout()
            except Exception as e:
                self.log(f"⚠️  Could not load RapidLayout: {e}")
                return None
        return self._layout

    def table(self) -> Optional[object]:
        """Return the RapidTable instance (lazy)."""
        if self._table is None and RapidTable is not None:
            self.log("⚙️  Loading RapidTable…")
            try:
                self._table = RapidTable()
            except Exception as e:
                self.log(f"⚠️  Could not load RapidTable: {e}")
                return None
        return self._table

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release all model references."""
        self._formula = None
        self._ocr = None
        self._layout = None
        self._table = None

    # ------------------------------------------------------------------
    # Formula: image bytes / path → LaTeX
    # ------------------------------------------------------------------

    def recognize_formula(self, crop_path: str) -> Optional[str]:
        """Run a single crop through RapidLaTeXOCR and return LaTeX, or None."""
        eng = self.formula()
        if eng is None:
            return None
        try:
            data = Path(crop_path).read_bytes()
            res, _elapse = eng(data)  # VERIFY: returns (latex_str, elapse)
            return (res or "").strip() or None
        except Exception as e:
            self.log(f"   ⚠️  formula recog failed: {e}")
            return None

    # ------------------------------------------------------------------
    # OCR: image → [(box, text, score)]
    # ------------------------------------------------------------------

    def ocr_lines(self, img) -> list[tuple]:
        """Run RapidOCR on ``img`` (path, ndarray, or bytes).

        Returns a list of ``(box, text, score)`` tuples.
        """
        eng = self.ocr()
        if eng is None:
            return []
        try:
            result = eng(img)
        except Exception as e:
            self.log(f"   ⚠️  OCR failed: {e}")
            return []

        # Normalize across versions:
        #   v3 -> result.boxes / .txts / .scores
        #   older -> list of [box, text, score]
        if hasattr(result, "txts"):  # VERIFY
            boxes = getattr(result, "boxes", None) or []
            txts = getattr(result, "txts", None) or []
            scores = getattr(result, "scores", None) or []
            return list(zip(boxes, txts, scores))
        if isinstance(result, (list, tuple)) and result and isinstance(result[0], (list, tuple)):
            return [(r[0], r[1], r[2] if len(r) > 2 else 1.0) for r in result]
        return []

    # ------------------------------------------------------------------
    # Layout: image → [(box, label, score)]
    # ------------------------------------------------------------------

    def layout_regions(self, img) -> list[tuple]:
        """Run RapidLayout on ``img``.

        Returns a list of ``(box, label, score)`` tuples.
        """
        eng = self.layout()
        if eng is None:
            return []
        try:
            boxes, scores, labels, _elapse = eng(img)  # VERIFY: 4-tuple, order may vary
            return list(zip(boxes, labels, scores))
        except Exception as e:
            self.log(f"   ⚠️  layout failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Table: region crop (+ OCR) → HTML
    # ------------------------------------------------------------------

    def table_html(self, crop_img) -> Optional[str]:
        """Run RapidTable on ``crop_img`` (with OCR text fill).

        Returns HTML table string, or None on failure.
        """
        eng = self.table()
        if eng is None:
            return None
        try:
            ocr_res = self.ocr_lines(crop_img)
            out = eng(crop_img, ocr_res)  # VERIFY: (html, cell_bboxes, elapse) OR obj.html
            if isinstance(out, tuple):
                html = out[0]
            else:
                html = getattr(out, "pred_html", None) or getattr(out, "html", None)
            return html
        except Exception as e:
            self.log(f"   ⚠️  table recog failed: {e}")
            return None
