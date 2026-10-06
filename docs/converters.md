# Document converters (PDF ingestion seam)

`okfgraph` never talks to a conversion engine directly. It talks to a
`DocumentConverter` (`okfgraph/components/converters.py`). Bobine is the
default implementation, not a dependency: nothing imports it at module
level, and the router builds fine without it installed. Only converting a
PDF with the *default* converter raises `RuntimeError: pip install bobine`.

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

Bobine-specific knobs (`routing_mode`, `extract_images`) live on
`BobineConverter(...)`, not on `ingest(kind, …)` — each provider owns its
options. `converter`/`on_page` are Python-only advanced params. Device
selection for ONNX-backed converters is ORT-level (`ORT_DYLIB_PATH`).

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
adding a real provider.
