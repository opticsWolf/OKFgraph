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
# or per call:
router.ingest_mgr.ingest_pdf("doc.pdf", converter=MyPipeline())
```

Bobine-specific knobs (`routing_mode`, `extract_images`) live on
`BobineConverter(...)`, not on `ingest_pdf` — each provider owns its options.
Device selection for ONNX-backed converters is ORT-level (`ORT_DYLIB_PATH`).

## Tests

`tests/test_ingest.py::TestCustomConverter` plugs a stub pipeline through all
three paths (per-call, per-call + import, router-wide) — extend it when
adding a real provider.
