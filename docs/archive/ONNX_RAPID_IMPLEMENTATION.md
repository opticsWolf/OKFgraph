> **Historical design record — do not follow.** Describes the pre-0.2.x
> optimum/transformers embedding stack and/or the in-tree RapidAI ingest
> engine, neither of which exists anymore. The authoritative surface is
> `architecture.md` v6.0 (as-built for okfgraph 0.2.12: external
> `embroider` crate, bobine converter seam, 5 MCP tools). Kept for
> archaeology, not guidance.

# Migrating the Hybrid Engine to ONNX / Rapid — Implementation Guide

**Goal.** Replace the PaddleOCR / PaddlePaddle heavy pass with an all-ONNX stack (RapidAI
family) so the engine runs on a single `onnxruntime` wheel — CPU, CUDA, DirectML, CoreML,
TensorRT, or OpenVINO — with no PaddlePaddle/CUDA-version coupling. The fast path stays on
`pdf_oxide`. The deliverable is unchanged: **one Markdown file per document**, containing
inline/display LaTeX, fenced code blocks, GFM tables, and `okf-asset://` image links.

> **API-drift warning.** The Rapid packages move fast and their call signatures and result
> attributes differ across minor versions. **Pin the versions in `requirements.txt`** and
> confirm exact signatures with `help(obj)` on first run. Every version-sensitive line below
> is flagged `# VERIFY`. The architecture is stable; only the leaf calls churn.

---

## 0. At a glance

| Your current (Paddle) piece | ONNX / Rapid replacement | Package |
| --- | --- | --- |
| `FormulaRecognition` (PP-FormulaNet) — **surgical formula pass** | **RapidLaTeXOCR** (LaTeX-OCR → ONNX) | `rapid_latex_ocr` |
| `PPStructureV3` whole-page OCR (scanned fallback) | RapidOCR + RapidLayout | `rapidocr`, `rapid_layout` |
| PP-Structure table recognition | RapidTable | `rapid_table` |
| `paddlepaddle-gpu` runtime | ONNX Runtime execution providers | `onnxruntime[-gpu/-directml]` |

The routing (`fast → surgical → full-structure fallback`) and all assembly logic
(`_stage_images_as_okf_assets`, `html_tables_to_gfm`, the splice) are **kept as-is**.

---

## 1. Target architecture

```
                          ┌─────────────────────────────┐
   PDF ──▶ pdf_oxide ────▶│ page has a usable text layer?│
                          └──────────────┬──────────────┘
                                yes │            │ no  (few chars + images = scanned/old)
                    ┌───────────────▼──┐      ┌──▼────────────────────────────────────┐
                    │ FAST PATH        │      │ FALLBACK (heavy, ONNX)                 │
                    │ pdf_oxide.markdown│     │ render page → RapidLayout regions      │
                    │  + surgical passes│     │  ├ text/title/list → RapidOCR          │
                    │  ├ math boxes →   │     │  ├ table          → RapidTable → GFM    │
                    │  │  RapidLaTeXOCR │     │  ├ formula        → RapidLaTeXOCR       │
                    │  ├ mono runs →    │     │  └ figure         → asset crop          │
                    │  │  code fences   │     │ assemble in reading order              │
                    │  └ tables kept as │     └────────────────────────────────────────┘
                    │    pdf_oxide GFM  │
                    │    (RapidTable    │
                    │     rescue opt.)  │
                    └───────────────────┘
                                │            │
                                └─────┬──────┘
                                      ▼
                        per-page markdown blocks
                                      ▼
             stage images → okf-asset://  •  join pages  •  write ONE .md
```

**Formula extraction is required on both paths.** On the fast path it fires on
char-geometry crops (born-digital equations); inside the fallback it fires on
`formula`/`equation` regions returned by RapidLayout.

---

## 2. Dependencies & install

```bash
# --- fast path (unchanged) ---
pip install pdf_oxide pillow

# --- ONNX heavy passes ---
pip install rapidocr            # text detection + recognition (PP-OCRv4/v5/v6 → ONNX)
pip install rapid_latex_ocr     # formula image → LaTeX  (LaTeX-OCR → ONNX)   ★ required
pip install rapid_layout        # layout region detection (scanned/old fallback)
pip install rapid_table         # table structure → HTML   (fallback rescue)

# --- ONNX Runtime: pick ONE execution provider that matches your hardware ---
pip install onnxruntime             # CPU everywhere
# pip install onnxruntime-gpu       # NVIDIA CUDA (incl. Blackwell via CUDA EP)
# pip install onnxruntime-directml  # any DirectX 12 GPU on Windows (incl. RTX 50-series)
```

Licenses are commercial-friendly: RapidAI packages are Apache-2.0; LaTeX-OCR is MIT.

> `rapid_latex_ocr` downloads its ONNX weights **separately** from the wheel (they exceed
> PyPI's size cap). First construction pulls them; for offline/air-gapped installs, pre-place
> the model files and pass explicit paths (see §9).

---

## 3. Model inventory

| Role | Package | Model files (ONNX) | Needed for | Priority |
| --- | --- | --- | --- | --- |
| **Formula → LaTeX** | `rapid_latex_ocr` | `image_resizer.onnx`, `encoder.onnx`, `decoder.onnx`, `tokenizer.json` | Surgical pass **and** formula regions in fallback | **Required** |
| Text det + rec | `rapidocr` | `*_det_infer.onnx`, `*_rec_infer.onnx` (PP-OCRv5) | Whole-page OCR for scanned/old PDFs | Required (fallback) |
| Text angle cls | `rapidocr` | `*_cls_infer.onnx` | Rotated scans | Optional |
| Layout regions | `rapid_layout` | PP-DocLayout / layout ONNX | Splitting scanned pages into text/table/formula/figure | Recommended |
| Table structure | `rapid_table` | SLANet / unitable ONNX | Rescuing tables pdf_oxide can't read (scanned) | Recommended |

`rapidocr` v3 auto-resolves models via its bundled `default_model.yaml`; the others download
on first use. Nothing here needs PaddlePaddle.

---

## 4. The ONNX engine

Add a single `OnnxRapidEngine` that owns all four models with **lazy** loaders — so a clean
born-digital paper loads only the tiny formula model, never the OCR/layout/table stack.

```python
# onnx_rapid_engine.py
from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional

try:
    from rapid_latex_ocr import LatexOCR            # VERIFY: class may be LaTeXOCR in some builds
except ImportError:
    LatexOCR = None
try:
    from rapidocr import RapidOCR                   # VERIFY: older wheels: rapidocr_onnxruntime
except ImportError:
    RapidOCR = None
try:
    from rapid_layout import RapidLayout
except ImportError:
    RapidLayout = None
try:
    from rapid_table import RapidTable
except ImportError:
    RapidTable = None


class OnnxRapidEngine:
    """All ONNX heavy passes behind lazy loaders. No PaddlePaddle."""

    def __init__(self, log: Callable[[str], None] = print, ort_providers: Optional[list] = None):
        self.log = log
        self.ort_providers = ort_providers      # e.g. ["CUDAExecutionProvider","CPUExecutionProvider"]
        self._formula = None
        self._ocr = None
        self._layout = None
        self._table = None

    # ---- lazy loaders -------------------------------------------------------
    def formula(self):
        if self._formula is None:
            if LatexOCR is None:
                self.log("⚠️  rapid_latex_ocr not installed; math stays as text.")
                return None
            self.log("⚙️  Loading RapidLaTeXOCR (formula → LaTeX)…")
            self._formula = LatexOCR()             # VERIFY: pass model paths for offline (see §9)
        return self._formula

    def ocr(self):
        if self._ocr is None and RapidOCR is not None:
            self.log("⚙️  Loading RapidOCR (text det+rec)…")
            self._ocr = RapidOCR()                 # VERIFY: RapidOCR(providers=self.ort_providers)
        return self._ocr

    def layout(self):
        if self._layout is None and RapidLayout is not None:
            self.log("⚙️  Loading RapidLayout…")
            self._layout = RapidLayout()
        return self._layout

    def table(self):
        if self._table is None and RapidTable is not None:
            self.log("⚙️  Loading RapidTable…")
            self._table = RapidTable()
        return self._table

    def close(self):
        self._formula = self._ocr = self._layout = self._table = None

    # ---- formula: image bytes/path → LaTeX ----------------------------------
    def recognize_formula(self, crop_path: str) -> Optional[str]:
        eng = self.formula()
        if eng is None:
            return None
        try:
            data = Path(crop_path).read_bytes()
            res, _elapse = eng(data)               # VERIFY: returns (latex_str, elapse)
            return (res or "").strip() or None
        except Exception as e:  # noqa: BLE001
            self.log(f"   ⚠️  formula recog failed: {e}")
            return None

    # ---- OCR: image → [(box, text, score)] ----------------------------------
    def ocr_lines(self, img) -> list[tuple]:
        eng = self.ocr()
        if eng is None:
            return []
        try:
            result = eng(img)                      # img = path | np.ndarray | bytes
        except Exception as e:  # noqa: BLE001
            self.log(f"   ⚠️  OCR failed: {e}")
            return []
        # Normalize across versions: v3 -> result.boxes/.txts/.scores ; older -> list of [box,text,score]
        if hasattr(result, "txts"):                # VERIFY
            boxes = getattr(result, "boxes", None) or []
            txts = getattr(result, "txts", None) or []
            scores = getattr(result, "scores", None) or []
            return list(zip(boxes, txts, scores))
        if isinstance(result, (list, tuple)) and result and isinstance(result[0], (list, tuple)):
            return [(r[0], r[1], r[2] if len(r) > 2 else 1.0) for r in result]
        return []

    # ---- layout: image → [(box, label, score)] ------------------------------
    def layout_regions(self, img) -> list[tuple]:
        eng = self.layout()
        if eng is None:
            return []
        try:
            boxes, scores, labels, _elapse = eng(img)   # VERIFY: 4-tuple, order may vary
            return list(zip(boxes, labels, scores))
        except Exception as e:  # noqa: BLE001
            self.log(f"   ⚠️  layout failed: {e}")
            return []

    # ---- table: region crop (+ ocr) → HTML ----------------------------------
    def table_html(self, crop_img) -> Optional[str]:
        eng = self.table()
        if eng is None:
            return None
        try:
            ocr_res = self.ocr_lines(crop_img)          # RapidTable fills cells from OCR text
            out = eng(crop_img, ocr_res)                # VERIFY: (html, cell_bboxes, elapse) OR obj.html
            if isinstance(out, tuple):
                html = out[0]
            else:
                html = getattr(out, "pred_html", None) or getattr(out, "html", None)
            return html
        except Exception as e:  # noqa: BLE001
            self.log(f"   ⚠️  table recog failed: {e}")
            return None
```

---

## 5. Wiring into the existing `HybridConverter`

Only three methods change; the routing, cropping, and splicing you already built are reused
verbatim.

**5.1 — Construct the engine** in `HybridConverter.__init__` and replace the Paddle loaders:

```python
self.rapid = OnnxRapidEngine(log=self.log, ort_providers=self.cfg.ort_providers)

def _ensure_formula(self):          # was: load PP-FormulaNet
    self.rapid.formula()            # lazy; safe to call repeatedly

def _ensure_full_structure(self):   # was: load PPStructureV3
    self.rapid.ocr(); self.rapid.layout()   # table loads on demand
```

**5.2 — Formula recognition** (surgical pass): swap the body of `_recognize_formulas`.

```python
def _recognize_formulas(self, crop_paths: list[str]) -> list[Optional[str]]:
    # RapidLaTeXOCR runs one crop at a time; loop (crops are tiny and few per page).
    return [self.rapid.recognize_formula(p) for p in crop_paths]
```

Everything upstream (`_math_boxes_from_chars`, `_crop_pil`, `_region_text`, `_splice`) is
unchanged — you still get `$$…$$` spliced back into pdf_oxide's fast markdown in place.

**5.3 — Whole-page fallback**: replace `_full_structure_page_markdown` with a layout-driven
ONNX assembler (see §7). `_route_page` keeps its exact shape:

```
NEVER   → fast only
SURGICAL→ scanned? full-structure(ONNX) : surgical formula crops   ← recommended default
AUTO    → flagged? full-structure(ONNX) : fast
ALWAYS  → full-structure(ONNX) every page
```

---

## 6. Producing the single final `.md`

The per-page blocks are concatenated and the images staged exactly as today. Here's how each
required output element is produced.

**Inline vs display LaTeX.** From `_math_boxes_from_chars`, a box that occupies (nearly) its
own line → **display** `$$…$$`; a short box embedded within a text line → **inline** `$…$`.
Decide with a width/line-height ratio:

```python
def _latex_wrap(latex: str, box, page_line_height: float) -> str:
    x0, y0, x1, y1 = box
    is_display = (y1 - y0) > 1.6 * page_line_height or (x1 - x0) > 220  # tune
    return f"$$\n{latex}\n$$" if is_display else f"${latex}$"
```

**Fenced code blocks.** pdf_oxide chars carry `font_name`; monospace fonts imply code. Group
consecutive lines whose dominant font is monospace and wrap them:

```python
_MONO = ("mono", "courier", "consol", "menlo", "inconsolata", "sourcecode", "dejavu sans mono")

def _wrap_code_blocks(page) -> list[str]:
    # returns fenced blocks; integrate at the line level of your fast-path builder
    ...
    # if line looks monospace and previous line was too -> keep buffering
    # on transition out -> emit "```\n" + "\n".join(buffer) + "\n```"
```

Emit as a plain fence (```` ``` ````); leave language tag blank unless you detect one.

**Tables → GFM.** Fast path: keep pdf_oxide's native GFM tables. Fallback / rescue: RapidTable
returns HTML → run it through your existing `html_tables_to_gfm()` (simple tables become pipe
tables; rowspan/colspan tables stay as HTML, which Markdown renderers accept).

**Image assets → `okf-asset://`.** Unchanged. Extract embedded images with
`doc.extract_image_bytes(i)` (fast path) and save RapidLayout `figure` crops (fallback) into
the same working dir, then `_stage_images_as_okf_assets()` rewrites every `![alt](local)` to
`![alt](okf-asset://<sha-id>)` and copies bytes into `_assets/`.

**Links / references.** pdf_oxide's `markdown()` already emits `[text](url)` for hyperlink
annotations; preserve them untouched. For internal cross-references ("see Fig. 3"), leave the
prose as-is — don't try to synthesize anchors unless you also emit heading anchors.

**Page joining.** Concatenate blocks with `"\n\n"` for a continuous document (drop the `---`
separators if you want one seamless flow rather than page-delimited).

**Target output shape (single file):**

````markdown
# Paper Title

Intro paragraph with an inline relation $E = mc^2$ and a reference to [the spec](https://example.org).

## 2. Method

The loss is defined as

$$
\mathcal{L}(\theta) = -\sum_{i} y_i \log \hat{y}_i
$$

```python
def train(model, data):
    return model.fit(data)
```

| Metric | Value |
| --- | --- |
| Accuracy | 0.94 |

![Figure 1: architecture](okf-asset://img_a1b2c3d4e5f6a7b8)
````

---

## 7. Fallback path for old / scanned PDFs

Triggered by `_is_scanned(page)` (few/no chars + images present). Renders the page and rebuilds
markdown from ONNX region analysis:

```python
def _full_structure_page_markdown(self, doc, page, index, work_dir) -> Optional[str]:
    rendered = self._render_page_to_pil(doc, page, index, self.cfg.render_dpi)
    if rendered is None:
        return None
    img, _w, page_h = rendered
    import numpy as np
    page_np = np.asarray(img)                       # RapidOCR/Layout accept ndarray or path

    regions = self.rapid.layout_regions(page_np)    # [(box,label,score)]
    if not regions:                                 # layout missing → OCR whole page as text
        lines = self.rapid.ocr_lines(page_np)
        return "\n\n".join(t for _b, t, _s in self._reading_order(lines)) or None

    blocks = []
    for box, label, _score in self._reading_order(regions):
        crop = self._crop_pil(img, self._px_to_pts(box, page_h), page_h, self.cfg.render_dpi)
        crop_np = np.asarray(crop) if crop else None
        lab = (label or "").lower()
        if lab in ("table",):
            html = self.rapid.table_html(crop_np)
            blocks.append(html_tables_to_gfm(html) if html else "")
        elif lab in ("formula", "equation", "isolate_formula"):
            p = work_dir / f"_reg_f_{len(blocks)}.png"
            crop.save(str(p))
            latex = self.rapid.recognize_formula(str(p))
            blocks.append(f"$$\n{latex}\n$$" if latex else "")
        elif lab in ("figure", "image"):
            p = work_dir / f"_reg_fig_{len(blocks)}.png"     # staged as okf-asset later
            crop.save(str(p))
            blocks.append(f"![]({p.name})")
        else:  # text, title, list, header, footer, reference, caption …
            lines = self.rapid.ocr_lines(crop_np)
            text = " ".join(t for _b, t, _s in lines).strip()
            if lab == "title":
                text = f"## {text}"
            if text:
                blocks.append(text)
    md = "\n\n".join(b for b in blocks if b)
    return md or None
```

Helpers you provide: `_reading_order()` (sort by box top-left; for two-column, bucket by
x-midpoint then y), and `_px_to_pts()` (invert the DPI + y-flip mapping from `_crop_pil`).
Table and whole-page OCR are thus **only** invoked here — the fast path never pays for them.

---

## 8. Config surface

Extend `ConverterConfig`:

```python
ort_providers: list[str] | None = None      # None → onnxruntime default; else explicit EP order
rescue_bad_tables: bool = False             # fast path: send garbled pdf_oxide tables to RapidTable
formula_inline_max_width_pts: float = 220.0 # display vs inline threshold
detect_code_blocks: bool = True             # monospace-run → fenced code
```

Drop the Paddle-only fields (`formula_model_name` etc.) or repurpose `formula_model_name` to
select a RapidLaTeXOCR model directory.

---

## 9. Model download & offline packaging

```python
# Explicit paths make installs air-gapped and reproducible.
from rapid_latex_ocr import LatexOCR
formula = LatexOCR(
    image_resizer_path="models/latexocr/image_resizer.onnx",   # VERIFY kwarg names
    encoder_path="models/latexocr/encoder.onnx",
    decoder_path="models/latexocr/decoder.onnx",
    tokenizer_json="models/latexocr/tokenizer.json",
)
```

- Vendor all `.onnx` files + `tokenizer.json` into a `models/` dir shipped with your app.
- For RapidOCR, point at local det/rec/cls `.onnx` and its keys/dict file rather than relying
  on auto-download.
- Cache dir for auto-downloaded models is under the package install path; set it explicitly in
  Docker to keep layers cacheable.

---

## 10. Execution providers — this is what fixes the CUDA problem

ONNX Runtime decouples you from CUDA toolkit versions entirely:

- **NVIDIA (incl. Blackwell / RTX 50-series):** `onnxruntime-gpu`, providers
  `["CUDAExecutionProvider","CPUExecutionProvider"]`. If a CUDA build lags your driver, fall
  back to **`onnxruntime-directml`** on Windows — DirectML runs on any DX12 GPU regardless of
  CUDA version, which is the pragmatic Blackwell answer.
- **Apple Silicon:** `CoreMLExecutionProvider` (or CPU).
- **CPU-only:** default provider; RapidLaTeXOCR-S and PP-OCR are light enough to be usable.

Verify at runtime:

```python
import onnxruntime as ort
print(ort.get_available_providers())
```

If your provider isn't listed, you installed the wrong `onnxruntime` wheel (they're mutually
exclusive — uninstall others first).

---

## 11. Testing checklist

- [ ] Born-digital paper with display + inline equations → correct `$$`/`$`, spliced in place.
- [ ] Scanned/old PDF (no text layer) → fallback fires; text, tables, formulas recovered.
- [ ] Table-heavy digital PDF → pdf_oxide GFM tables preserved (no RapidTable invoked).
- [ ] Scanned table → RapidTable → GFM (or HTML for rowspan/colspan).
- [ ] Code-heavy PDF (monospace) → fenced ``` blocks.
- [ ] Image-heavy PDF → every image staged as `okf-asset://`, none dropped.
- [ ] Hyperlinks preserved as `[text](url)`.
- [ ] GPU path: `ort.get_available_providers()` shows your EP; CPU fallback works.
- [ ] Offline: with network disabled, explicit model paths load and run.

---

## 12. Known caveats & version pins

- **RapidLaTeXOCR class name / kwargs** differ (`LatexOCR` vs `LaTeXOCR`; model-path kwargs).
  Confirm on install; isolate in `OnnxRapidEngine.formula()`.
- **RapidTable requires OCR results** to fill cell text — always pass `ocr_lines(crop)`; an
  empty OCR result yields an empty table.
- **RapidLayout label sets vary by model** (`formula` vs `isolate_formula`, `figure` vs
  `image`). Keep the `lab in (...)` sets permissive and log unknown labels.
- **Reading order** for multi-column scans needs column bucketing; single-column is just
  top-to-bottom. Bad reading order is the most common fallback-quality issue.
- **RapidOCR v3 config** changed (`default_model.yaml`, `dict_url`); if you upgrade and see a
  `dict_url` error, align `rapidocr>=3.x` with a matching `onnxruntime`.
- Pin, e.g.: `rapidocr==3.x`, `rapid_latex_ocr==<pinned>`, `rapid_layout==<pinned>`,
  `rapid_table==<pinned>`, `onnxruntime==<pinned>`.

---

### Summary

Keep `pdf_oxide` + your routing/splice/asset machinery. Replace exactly three method bodies to
call `OnnxRapidEngine`: formulas via **RapidLaTeXOCR** (the required star model, used on both
paths), and the scanned/old fallback via **RapidLayout + RapidOCR + RapidTable**. The output
contract is identical — a single `.md` with inline/display LaTeX, fenced code, GFM tables, and
`okf-asset://` links — but the runtime is now one portable `onnxruntime` wheel.
