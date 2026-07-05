"""Configuration for the ONNX/Rapid PDF ingestion engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RoutingMode(str, Enum):
    """Controls how pages are routed between the fast (pdf_oxide) and ONNX paths.

    =========  ====================================================================
    Mode       Behaviour
    =========  ====================================================================
    NEVER      Fast path only. No ONNX models loaded.
    AUTO       Heuristics per page → full ONNX pipeline only on flagged pages.
    SURGICAL   Formula crops via RapidLaTeXOCR; full pipeline only for scans.
    ALWAYS     Every page through the full ONNX layout + OCR pipeline.
    =========  ====================================================================
    """

    AUTO = "auto"
    SURGICAL = "surgical"
    ALWAYS = "always"
    NEVER = "never"


@dataclass
class ConverterConfig:
    """Tunable knobs for the HybridConverter pipeline.

    Most fields have sensible defaults; override only what you need.
    """

    # -- image extraction ---------------------------------------------------
    extract_images: bool = True
    append_unreferenced_images: bool = True

    # -- ONNX heavy passes --------------------------------------------------
    use_onnx: bool = True
    routing_mode: RoutingMode = RoutingMode.AUTO

    # ONNX Runtime execution providers (None → onnxruntime default)
    ort_providers: list[str] | None = None

    # -- rendering ----------------------------------------------------------
    render_dpi: int = 300        # scanned-page OCR needs resolution
    formula_dpi: int = 200      # crops of digital formulas need less

    # -- fast path ----------------------------------------------------------
    detect_headings: bool = True
    convert_html_tables: bool = True

    # -- AUTO / scan detection thresholds -----------------------------------
    math_char_threshold: int = 30
    scanned_text_threshold: int = 50

    # -- SURGICAL formula pass ----------------------------------------------
    formula_batch_size: int = 8
    formula_pad_pts: float = 4.0
    min_formula_math_chars: int = 3

    # -- inline vs display LaTeX threshold ----------------------------------
    formula_inline_max_width_pts: float = 220.0

    # -- code block detection -----------------------------------------------
    detect_code_blocks: bool = True

    # -- table rescue (fast path) -------------------------------------------
    rescue_bad_tables: bool = False

    # -- OCR language (for RapidOCR) ----------------------------------------
    ocr_lang: str = "en"

    # -- device hint (cuda → CUDAExecutionProvider, cpu → CPU) ---------------
    device: str = "cuda"

    def __post_init__(self) -> None:
        """Coerce device → ort_providers if the user didn't specify providers.

        Accepts ``"cuda"`` or ``"gpu"`` for GPU; anything else falls back to CPU.
        This matches OKFRouter.device semantics.
        """
        if self.ort_providers is None:
            if self.device in ("cuda", "gpu"):
                self.ort_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                self.ort_providers = ["CPUExecutionProvider"]
