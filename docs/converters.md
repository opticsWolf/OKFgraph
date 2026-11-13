# Document converters (PDF ingestion seam)

`okfgraph` never talks to a conversion engine directly. It talks to a
`DocumentConverter` (`okfgraph/components/converters.py`). Bobine is the
default implementation, not a dependency: nothing imports it at module
level, and the router builds fine without it installed. Only converting a
PDF with the *default* converter raises a `RuntimeError` naming
`pip install "okfgraph[pdf]"`.

A file bobine cannot convert (a corrupt PDF, an unsupported format) is
`CONVERSION_FAILED` (exit 1, since 0.14) with `fields={"path",
"converter": "bobine"}` and bobine's message, not `INTERNAL`. A missing
file is `FILE_NOT_FOUND`, checked before conversion starts. With several
`--pdf-path` files, each failure is reported for its file and the rest
still convert. A folder or `*`/`?` pattern on `--pdf-path` (0.14.2)
expands to the files with bobine's extensions (pdf, docx, xlsx, pptx, doc,
xls, ppt), skipping Office owner files (`~$…`) and hidden files; a folder or
pattern that matches nothing is `FILE_NOT_FOUND` naming it.

## The contract

```python
class DocumentConverter(Protocol):
    def convert(self, pdf_path, output_dir, *, on_page=None) -> ConvertedDocument: ...
```

- Write `<stem>.md` plus staged images into `output_dir`.
- Return `ConvertedDocument(md_path=..., image_dir=..., page_count=...)`.
- `on_page` receives 0-based `(page_index, page_total)` callbacks (optional).
- Everything downstream (mordant lint → chunking → embeddings → import) is
  converter-agnostic.

## Using a different pipeline

```python
from okfgraph.components.converters import ConvertedDocument
from okfgraph.router import OKFRouter

class MyPipeline:
    def convert(self, pdf_path, output_dir, *, on_page=None):
        ...  # your conversion (pymupdf, external service, ...)
        return ConvertedDocument(md_path=..., image_dir=..., page_count=...)

router = OKFRouter(db_path="kb.db", bundle_root="bundle", converter=MyPipeline())
# or per call (0.10: one ingest dispatch):
router.ingest("pdf", pdf_path="doc.pdf", converter=MyPipeline())
```

Bobine-specific knobs (`routing_mode`, `extract_images`, `device`,
`precision`) live on `BobineConverter(...)`, not on `ingest(kind, …)` — each
provider owns its options. `converter`/`on_page` are Python-only advanced
params. The ONNX *binary* is still chosen at ORT level (`ORT_DYLIB_PATH`);
the device and weight precision a converter asks for are its own knobs, so
one machine can embed on CUDA and convert on CPU.

## Device and precision (0.14.1)

`pdf_device` / `pdf_precision` (TOML `[import]`, `OKFGRAPH_PDF_DEVICE` /
`OKFGRAPH_PDF_PRECISION`, or the global `--pdf-device` / `--pdf-precision`
flags) take the same words the other ONNX lanes take:

| Setting | Words | Meaning |
|---|---|---|
| `pdf_device` | `auto` (default), `cpu`, `cuda` | Which device the converter's passes target; `auto` keeps each pass's measured policy, `cuda` moves every pass (including the CPU-pinned table slots) |
| `pdf_precision` | `auto` (default), `fp32`, `fp16`, `int8` | Weight precision; `auto` follows the device, `int8` is the compact formula pair |

The words come from bobine itself (`bobine.DEVICES` / `bobine.PRECISIONS`);
there is no `q8` artifact in the stack, so it is refused by name. The table
validates before work starts, and `okf model-info --converters` prints the
resulting per-pass plan from bobine's `describe()` (requested words first,
landed provider after the first session).

## The warm converter (0.14.1)

`BobineConverter` keeps one native `HybridConverter` per settings key and
calls `ingest_document` on it, so:

- `okf ingest --kind pdf --pdf-path a.pdf b.pdf c.pdf` builds layout and OCR
  once for the whole command;
- `okf serve` / `okf-mcp` keep the sessions between PDF ingests;
- `pdf_idle_seconds` (default 300, `0` = after every conversion) releases
  them when nothing converts, and `OKFRouter.close()` releases them on exit.

Conversions on one converter are serialised by a lock: bobine allows one
document at a time and *raises* on a concurrent call (it cannot block on a
native mutex while holding the GIL), so the lock is load-bearing, not
decorative. A knob change (another routing mode, another device) closes the
warm converter and builds the new one — two sets are never held.

`okf doctor` reports the state as an informational `pdf_converter` line:
`warm (routing=auto, device=cpu, precision=fp32, layout=…)` with the last
conversion and the pending release, or `released` with the words the next
ingest will use.

## Converter cache status (0.10)

`converters.converter_status(cache_dir=None)` aggregates
`bobine.model_status()` (bobine >= 0.6.0) into the same report shape as
`model_info`, so one screen answers "is this machine ready for PDF
ingest offline?". It never raises: missing bobine, bobine < 0.6.0
(no `model_status`), and lookup failures all come back as data
(`{"available", "reports", "note"}`). Surfaced as `okf model-info
--converters` and as the informational `converter_cache` doctor entry
(never a finding, never scored).

## Hub-cache hygiene (0.10)

`EmbeddingEngine.cache_hygiene(cache_dir=None)` walks the hub-cache root
(listings + stat only — never writes, deletes, or downloads) and flags
`models--*` repos none of whose `refs/*` resolve to a populated snapshot.
Offline cache-mode lookups resolve `refs/<rev>` first, so such repos — from
pre-cache-mode seeders, manual copies, interrupted downloads — can never be
reused: deleting them is safe, and anything still needed re-downloads once
in the current layout. An empty `blobs/` alone is *not* legacy (Windows
`huggingface_hub` without symlinks). Surfaced as the informational
`cache_hygiene` doctor entry, which names the repos, the reclaimable bytes,
and the remedy — report-only, never a finding.

## Tests

`tests/test_ingest.py::TestCustomConverter` plugs a stub pipeline through all
three paths (per-call, per-call + import, router-wide) — extend it when
adding a real provider. `tests/test_pdf_converter.py` covers the 0.14.1 lane
with a fake bobine (settings validation, the config words, the plan report,
warm reuse, close/rebuild, idle release with an injected clock, concurrency,
the manager's one-slot registry and the doctor row) plus one `pdf`-marked
test that proves real bobine keeps the sessions across two files.
