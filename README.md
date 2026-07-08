# OKFgraph

**Ladybug-backed knowledge graph with ONNX-optimized Jina v5 embeddings, multimodal image ingestion, delta-aware incremental imports, and schema migration.**

**Architecture**: [architecture.md](architecture.md) v5.5 — gap analysis reviewed 15 gaps (14 closed, 1 open).

OKFgraph is a Python library and CLI tool for building, querying, and managing knowledge graphs from Markdown/OKF documents. It combines graph traversal, hybrid semantic search, chunk-level retrieval, and — in v5.1 — delta detection with safe purge of deleted concepts into a single SQLite-backed system.

---

## Features

| Category | Features |
|---|---|
| **Embeddings** | ONNX-optimized Jina v5 text model, SentenceTransformer omni model (lazy-loaded), Matryoshka truncation (32–1024 dims, default 512), numpy-only post-processing (no torch) |
| **Search** | Hybrid RRF fusion (vector + FTS), graph traversal, chunk-level search with RRF, graph-aware reranking (hub scores), context expansion, direct lookup, image search via unified index |
| **Storage** | LadybugDB — graph + vector + full-text search in a single SQLite file |
| **Images** | Three ingestion modes (`text` / `optional` / `omni`), unified vector space for text + image, content-hash dedup, `okf-asset://` protocol |
| **Import/Export** | OKF Markdown round-trip, batch import with single-transaction upsert, **delta detection** (SHA-256 hash skip), **purge deleted concepts** (`--purge`), filtered bulk export with graph enrichment (See Also + Cited By), auto-generated index.md files |
| **PDF Ingestion** | `okfgraph.ingest` sub-module — pdf_oxide fast path + ONNX/Rapid heavy passes (RapidLaTeXOCR, RapidOCR, RapidLayout, RapidTable), four routing modes (NEVER/AUTO/SURGICAL/ALWAYS), Paddle-free |
| **CLI** | 14 commands + interactive REPL, `--mode` for image ingestion, `--device cuda` with auto-fallback, chunking flags (`--chunk-overlap`, `--no-chunking`), `--purge` for stale concept cleanup |
| **LLM Tools** | 13 OpenAI-compatible tool definitions for agent integration (search, traverse, chunks, graph enrichment, path finding, export) |

---

## Quick Start

### Installation

```bash
pip install ladybug pydantic python-frontmatter pyyaml optimum[onnxruntime] transformers torch sentence-transformers Pillow
```

### Programmatic Usage

```python
from okfgraph import OKFRouter, ConceptModel

# Initialize the router
router = OKFRouter(
    db_path="okfgraph.db",
    bundle_root="./my-knowledge-base",
    embedding_dim=512,       # Matryoshka: 32, 64, 128, 256, 512, 768, 1024
    device="cuda",            # or "cpu" — defaults to CUDA when onnxruntime-gpu is available
)

# Import an OKF bundle
ids = router.import_bundle(mode="optional")
print(f"Imported {len(ids)} concepts")

# Hybrid search
results = router.search_hybrid("machine learning basics", limit=5)
for r in results:
    print(f"  [{r['relevance_score']:.3f}] {r['title']} ({r['type']})")

# Chunk-level search (RRF-fused vector + FTS)
chunks = router.search_chunks("neural network architecture", limit=5)
for c in chunks:
    print(f"  [{c['rrf_score']:.3f}] {c['parent_title']} §{c['chunk_index']}")

# Search with graph context
enriched = router.search_with_context("graph neural networks", limit=3)
for r in enriched:
    chunk = r["chunk"]
    print(f"  [{chunk['rrf_score']:.3f}] {chunk['parent_title']}")
    print(f"    ← linked by: {len(r['incoming_links'])} docs")

# Hub-score reranking
ranked = router.search_chunks_with_hub_score("fundamentals", limit=10)
for r in ranked:
    print(f"  [{r['final_score']:.3f}] hub={r['hub_score']:.2f} {r['parent_title']}")

# Reconstruct document from chunks
original = router.reconstruct_document("concepts/intro")

# Image search (text → image via unified index)
images = router.search_images_with_text("diagram of a neural network", limit=10)
for img in images:
    print(f"  [{img['relevance_score']:.3f}] {img['file_name']}")

# Graph traversal
nodes = router.traverse("concepts", relationship="CONTAINS", direction="OUTGOING", depth=2)
```

### CLI Usage

```bash
# Initialize database
okf init --bundle ./my-knowledge-base --dim 512

# Import a single file
okf import ./concepts/intro.md --mode optional

# Import an entire bundle
okf import --all --mode omni --allow-remote-images

# Import with delta detection + purge deleted concepts
okf import --all --purge

# Hybrid search
okf search "graph neural networks" --type section --limit 5

# Hybrid search with matched chunks
okf search "machine learning" --chunks

# Chunk-level search
okf search-chunks "neural network architecture"

# Search with graph context
okf context "graph neural networks"

# Hub-score reranking
okf hub-search "fundamentals"

# Find path between concepts
okf path concepts/intro concepts/advanced

# List chunks for a concept
okf chunks concepts/intro

# Reconstruct document from chunks
okf reconstruct concepts/intro

# Image search
okf search-images "network architecture diagram"

# List directory
okf list concepts

# Interactive shell
okf shell
```

---

## Architecture

### Embedding Engine

OKFgraph uses a **dual-model** approach with a **unified vector space**:

| Model | Role | Framework | Loaded |
|---|---|---|---|
| `jina-embeddings-v5-text-small-retrieval` | Text embeddings for concepts | ONNX Runtime (`optimum`) | Always |
| `jina-embeddings-v5-omni-small-retrieval` | Image embeddings (vision tower) | SentenceTransformer | Lazy (first use) |

Both models output vectors in the **same Matryoshka space**, so text and image embeddings are directly comparable in the `image_omni_idx` index.

### Image Ingestion Modes

| Mode | Image with alt-text | Image without alt-text | Omni model loaded? |
|---|---|---|---|
| `text` | Embed alt-text (lightweight) | Embed `filename + image-number` | Never |
| `optional` | Embed alt-text (lightweight) | Embed image bytes (rich) | Only for images without alt-text |
| `omni` | Embed image bytes (rich) | Embed image bytes (rich) | Always (if bytes available) |

When `omni` is requested but raw bytes are unavailable (e.g. remote URLs not fetched), the system **degrades gracefully** to the text path.

### Database Schema

```
Concept        ImageAsset       Directory       BrokenLink     Chunk        FileHash
├── id         ├── id           ├── id          ├── id         ├── id       ├── path (PK)
├── type       ├── file_name    └── (empty)     ├── source_id  ├── parent_doc_id  ├── hash
├── title      ├── mime_type                 ├── target_id    ├── chunk_index      ├── concept_id
├── body       ├── alt_text                └── timestamp      ├── chunk_text
├── embedding  ├── caption                                  ├── block_type
└── extra      ├── embed_route                          ├── start_offset
               ├── content_hash                         ├── end_offset
               ├── data (BLOB)                          └── embedding
               └── embedding

CONTAINS: Directory → Directory / Concept
LINKS_TO: Concept → Concept
INCLUDES_ASSET: Concept → ImageAsset
PART_OF: Concept → Chunk
```

### Chunking & Graph Retrieval

OKFgraph v5.0 adds **document chunking** with **graph-aware retrieval**:

| Feature | Description |
|---|---|
| **Mordant chunker** | Rust-based Markdown parser that splits documents into semantic blocks (paragraphs, headings, code blocks, lists, tables, blockquotes) |
| **Heading context injection** | Paragraph chunks get parent heading prepended to embedding payloads without mutating stored text |
| **Structural boundaries** | Headings, code blocks, lists, tables, blockquotes, and diagrams enforce hard boundaries — no overlap tails bleed across them |
| **RRF-fused chunk search** | Vector + FTS fusion at chunk granularity with per-doc limits and graph filters |
| **Hub-score reranking** | Chunks from authoritative documents (high incoming link count) rank higher |
| **Graph context expansion** | Search results enriched with incoming/outgoing links, directory ancestry, and sibling concepts |
| **Path finding** | BFS shortest path between any two concepts in the knowledge graph |
| **Document reconstruction** | ~98% fidelity round-trip from chunks back to original Markdown |
| **Hybrid search with chunks** | `search_hybrid(query, include_chunks=True)` attaches matched chunks to concept results |

### PDF Ingestion Engine (`okfgraph.ingest`)

The `okfgraph.ingest` sub-module provides a **Paddle-free** PDF → Markdown conversion pipeline using the RapidAI family of ONNX models. Four routing modes control when ONNX models are invoked:

| Mode | Behaviour |
|---|---|
| **NEVER** | Fast path only (pdf_oxide). No ONNX models loaded. |
| **AUTO** | Heuristics per page → full ONNX pipeline only on flagged pages. |
| **SURGICAL** | Formula crops via RapidLaTeXOCR; full pipeline only for scans. |
| **ALWAYS** | Every page through the full ONNX layout + OCR pipeline. |

Key features:

- **Zero hard dependencies** — all RapidAI imports are guarded; the module loads cleanly without them
- **Lazy loading** — born-digital PDFs never pay for OCR/layout/table models
- **Graceful degradation** — if a model fails to load, the pipeline falls back to the fast path
- **Device → ort_providers coercion** — `device="cuda"` auto-resolves to `["CUDAExecutionProvider", "CPUExecutionProvider"]` (accepts `"gpu"` as alias)
- **Output contract** — single `.md` with inline/display LaTeX, fenced code, GFM tables, and `okf-asset://` links

### Key Design Decisions

- **Last-token pooling** (not mean pooling) — required by Jina v5's training protocol
- **Delete-then-create upserts** — LadybugDB vector indexes block `SET` on indexed properties
- **Sequential encoding** — padded batch tokenization wastes attention compute with variable-length documents
- **Content-hash dedup** — unchanged images are not re-embedded on re-import
- **`okf-asset://` protocol** — markdown references use UUID-based URIs instead of inline Base64

---

## API Reference

### `OKFRouter`

| Method | Description |
|---|---|
| `__init__(db_path, bundle_root, model_id, omni_model_id, embedding_dim, cache_dir, device, allow_remote_images)` | Initialize router |
| `import_from_okf(file_path, mode)` | Import a single OKF file |
| `import_bundle(bundle_path, batch_size, mode, purge_deleted)` | Import entire bundle with batch encoding and delta detection |
| `export_to_okf(concept_id, output_path)` | Export a single concept to OKF markdown |
| `export_bundle(output_dir, directory_id, concept_type, tags)` | Export filtered bundle with graph enrichment |
| `search_hybrid(query, concept_type, tags, parent_id, exclude_reserved, limit)` | Hybrid RRF search |
| `search_images_with_text(text_query, use_text_model, limit)` | Text→image search via unified index |
| `traverse(start_id, relationship, direction, depth, node_type)` | Graph traversal |
| `list_directory(directory_id)` | List children of a directory |
| `get_by_id(concept_id)` | Fetch full concept by ID |
| `list_images(concept_id)` | List image assets attached to a concept |
| `get_image_data(asset_id)` | Fetch image asset with raw bytes |
| `list_broken_links()` | List orphaned links |
| `repair_links()` | Repair broken links |
| `model_info(model_id, cache_dir)` | Inspect model cache status (class method) |
| `search_chunks(query, concept_type, tags, parent_id, limit, max_chunks_per_doc)` | RRF-fused chunk search |
| `search_with_context(query, limit, context_hops)` | Chunk search + graph neighborhood expansion |
| `search_chunks_with_hub_score(query, limit, hub_weight)` | Chunk search reranked by hub score |
| `get_chunks(concept_id)` | List all chunks for a concept |
| `reconstruct_document(document_id)` | Reconstruct original Markdown from chunks |
| `find_path(start_id, end_id, max_length)` | BFS shortest path between concepts |
| `expand_with_graph_context(chunk_ids)` | Discover related concepts via graph edges |
| `rerank_with_hub_score(chunk_results)` | Adjust chunk scores by parent hub score |
| `ingest_pdf(pdf_path, auto_import, output_dir, ...)` | Programmatic PDF ingestion (Gap #5b) |
| `_purge_concept(concept_id)` | Safe cascading delete (concept, chunks, links, orphaned assets) |
| `_changed_files(source_files)` | Delta detection — returns `(changed, deleted)` tuple |
| `_file_hash(path)` | Compute SHA-256 hex digest of a file |

### `IngestMode`

```python
from okfgraph.images import IngestMode

IngestMode.TEXT       # alt-text / filename fallback (no omni)
IngestMode.OPTIONAL   # omni only for images lacking alt-text
IngestMode.OMNI       # every image via omni model
```

Aliases: `text-only`, `alt` → `TEXT`; `hybrid`, `auto` → `OPTIONAL`; `full`, `multimodal` → `OMNI`.

### `ConceptModel`

```python
class ConceptModel(BaseModel):
    id: str
    type: str
    title: Optional[str] = None
    description: Optional[str] = None
    resource: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    timestamp: Optional[datetime] = None
    body: str = ""
    embedding: Optional[List[float]] = None
    # extra frontmatter keys captured via model_config.extra = "allow"
```

---

## CLI Reference

| Command | Description |
|---|---|
| `okf init` | Initialize database and schema |
| `okf model-info` | Show model cache status |
| `okf import <files>` | Import one or more OKF files |
| `okf import --all` | Import entire bundle recursively |
| `okf search <query>` | Hybrid search (with `--chunks` for matched chunks) |
| `okf search-images <query>` | Find images via unified vector index |
| `okf search-chunks <query>` | Chunk-level RRF-fused search |
| `okf context <query>` | Search with graph neighborhood expansion |
| `okf hub-search <query>` | Chunk search reranked by hub score |
| `okf traverse <id>` | Graph traversal |
| `okf list [dir]` | List directory contents |
| `okf get <id>` | Fetch full concept |
| `okf export --all` | Export entire bundle |
| `okf path <id1> <id2>` | Find shortest path between concepts |
| `okf siblings <id>` | List sibling concepts in same directory |
| `okf ancestry <id>` | Show directory hierarchy for a concept |
| `okf chunks <id>` | List chunks for a concept |
| `okf reconstruct <id>` | Reconstruct document from chunks |
| `okf broken-links` | List orphaned links |
| `okf repair-links` | Repair broken links |
| `okf reindex [--if-dirty]` | Rebuild vector + FTS indexes |
| `okf shell` | Interactive REPL |

### Global Options

| Option | Default | Description |
|---|---|---|
| `--db` | `okfgraph.db` | Database file path |
| `--bundle` | `.` | Bundle root directory |
| `--dim` | `512` | Embedding dimension (32–1024) |
| `--cache-dir` | `~/.cache/huggingface` | HuggingFace model cache |
| `--device` | `cuda` | `cpu` or `cuda` |
| `--omni-model-id` | `jinaai/jina-embeddings-v5-omni-small-retrieval` | Multimodal model ID |
| `--verbose, -v` | — | Verbose logging (Gap #10) |
| `--quiet, -q` | — | Suppress non-essential output |
| `--log-file` | — | Write logs to file |
| `--profile` | — | Enable cProfile profiling (Gap #10) |

### Import Options

| Option | Default | Description |
|---|---|---|
| `--mode` | `text` | Image ingestion mode: `text`, `optional`, `omni` |
| `--allow-remote-images` | — | Fetch `http(s)://` image URLs |
| `--batch-size` | `32` | Batch size for encoding |
| `--purge` | — | Also purge concepts whose source files were deleted from disk |

---

## Performance

| Metric | Time | Notes |
|---|---|---|
| Single import (100 concepts) | ~140s | ~1400ms per concept |
| Batch import (100 concepts) | ~128s | **1.1x faster** than single |
| Hybrid search | 107ms mean | 5 repetitions |
| Export (100 concepts) | 77ms | 0.8ms per concept |

**Batch speedup** comes from DB-level optimizations (single transaction, bulk directory/link building), not from ONNX batching — sequential single-pass encoding avoids padding overhead with variable-length documents.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run just image tests (no heavy deps)
python tests/test_images.py

# Run integration tests (requires installed models)
pytest tests/test_integration.py -v
```

| Test Suite | Tests | Status |
|---|---|---|
| `test_ingest.py` | 39 | ✅ All passing (Gap #5b: ingest_pdf method, version checking, engine degradation, asset staging) |
| `test_router_misc.py` | 33 | ✅ All passing (Gap #12a: reindex, repair_links, meta/epoch, error isolation, context window, schema migration) |
| `test_router.py` | 29 | ✅ All passing |
| `test_chunking.py` | 26 | ✅ All passing |
| `test_delta.py` | 21 | ✅ All passing |
| `test_cli.py` | 19 | ✅ All passing |
| `test_export_compliance.py` | 13 | ✅ All passing |
| `test_integration.py` | 11 | ✅ All passing |
| `test_images.py` | 11 | ✅ All passing |
| `test_logging.py` | 10 | ✅ All passing (Gap #10: logging setup, timing instrumentation, profiling) |
| `test_directory_hash.py` | 10 | ✅ All passing (Gap #1b: directory-level hash aggregation, purge of deleted subtrees) |
| `test_search_browser.py` | 9 | ✅ All passing (PySide6 stubbed) |
| `test_graph_enrichment.py` | 9 | ✅ All passing |
| `test_chunk_search.py` | 7 | ✅ All passing |
| `test_reconstruction.py` | 5 | ✅ All passing |
| `test_okf_ingest_tool.py` | 5 | ✅ All passing |
| `test_ingest_tool.py` | 3 | ✅ All passing |
| `test_converter.py` | 2 | ✅ All passing (PySide6 stubbed) |
| **Total** | **262** | **All passing (0 warnings)** |

## Gap Analysis & Production Readiness

OKFgraph has undergone a comprehensive gap analysis ([docs/gap-analysis.md](docs/gap-analysis.md)) assessing production-readiness across 15 areas:

| Category | Status | Notes |
|---|---|---|
| **Delta Detection** | ✅ Closed (v5.1, v5.5) | File-level SHA-256 + directory-level hash aggregation + purge |
| **Schema Migration** | ✅ Closed (v5.3) | Versioned migrations (v1→v2→v3→v4) |
| **PDF Ingestion** | ✅ Closed (v5.3) | CLI `okf ingest` command with `--auto-import` |
| **Error Isolation** | ✅ Closed (v5.1) | Per-concept error isolation in import pipeline |
| **Index Lifecycle** | ✅ Closed (v5.1) | Epoch-based dirty tracking + change-driven rebuild |
| **Context Window** | ✅ Closed (v5.1) | 90% threshold warning for oversized chunks |
| **Missing Tests** | ✅ Closed (v5.6) | 277 tests across 22 files (17 GPU tests) |
| **RapidAI Pinning** | ✅ Closed (v5.4) | Version pins + runtime warning |
| **Observability** | ✅ Closed (v5.4) | Structured logging + profiling hooks |
| **PDF Ingest API** | ✅ Closed (v5.4) | `OKFRouter.ingest_pdf()` programmatic API |
| **Open: Concurrency** | ⚠️ Open | WAL mode + single-writer constraint needed |

See [architecture.md §15](architecture.md#15-open-gaps-production-readiness) for details on open gaps.

---

## Project Structure

```text
okfgraph/
├── okfgraph/
│   ├── __init__.py          # Exports: ConceptModel, OKFRouter, cli_main, ChunkModel
│   ├── models.py            # ConceptModel + ImageAssetModel + ChunkModel
│   ├── router.py            # OKFRouter — ~2700 lines (chunking, graph enrichment, export)
│   ├── cli.py               # CLI + interactive shell — ~800 lines
│   ├── tools.py             # LLM tool definitions (13 tools)
│   ├── images.py            # Image ingestion logic
│   ├── ingest/              # ONNX/Rapid PDF ingestion engine
│   │   ├── __init__.py      # Public API exports
│   │   ├── config.py        # ConverterConfig + RoutingMode
│   │   ├── engine.py        # OnnxRapidEngine (lazy ONNX loaders)
│   │   ├── converter.py     # HybridConverter (core pipeline)
│   │   ├── tables.py        # HTML → GFM pipe-table converter
│   │   ├── assets.py        # okf-asset:// staging logic
│   │   └── versions.py      # RapidAI version pinning + runtime checks (Gap #15)
│   └── docs/
│       ├── chunking-and-graph-retrieval.md  # Chunking specification
│       └── chunking-status.md               # Implementation status
├── tests/
│   ├── test_chunking.py       # 13 chunking unit tests
│   ├── test_reconstruction.py # 9 document reconstruction tests
│   ├── test_chunk_search.py   # 9 chunk search tests
│   ├── test_graph_enrichment.py # 11 graph enrichment tests
│   ├── test_integration.py    # 16 end-to-end tests
│   ├── test_export_compliance.py # 13 OKF export compliance tests
│   ├── test_ingest.py         # 39 ONNX/Rapid ingestion tests (version checking, ingest_pdf, asset staging)
│   ├── test_delta.py          # 21 delta detection & purge tests
│   ├── test_router_misc.py    # 23 router unit tests (reindex, repair_links, meta/epoch, error isolation, context window)
│   ├── test_router.py         # 29 smoke & cache & device tests
│   ├── test_logging.py        # 10 logging & profiling tests
│   ├── test_converter.py      # 2 converter staging tests (PySide6 stubbed)
│   ├── test_search_browser.py # 9 search browser tests (PySide6 stubbed)
│   └── fixtures/bundle/       # Test markdown fixtures
├── benchmarks/
│   └── benchmark_500.py
├── examples/
│   ├── hybrid_pdf_converter.py    # PySide6 UI (legacy Paddle stack)
│   └── oxide_to_markdown_converter.py
├── docs/
│   ├── okf-export-compliance.md   # OKF export specification
│   ├── ONNX_RAPID_IMPLEMENTATION.md # ONNX/Rapid migration guide
│   └── gap-analysis.md           # Production-readiness gap analysis (v3.0)
├── architecture.md        # Full architecture specification (v5.4)
├── pyproject.toml         # Project metadata
└── requirements.txt       # Dependencies
```

---

## Requirements

### Core (required)

- Python 3.10+
- `ladybug>=0.17`
- `pydantic>=2.0`
- `python-frontmatter>=1.0`
- `pyyaml>=6.0`
- `optimum[onnxruntime]>=1.20`
- `transformers>=4.57`
- `numpy>=1.24` (replaces torch for post-processing)
- `mordant>=0.12` (Rust-based Markdown chunker)
- `sentence-transformers>=3.0`
- `Pillow>=10.0`
- `pytest>=8.0` (dev)

### PDF Ingestion (optional — ONNX/Rapid stack)

- `pdf_oxide` — fast native PDF→Markdown (MIT)
- `rapidocr` — text detection + recognition (Apache-2.0)
- `rapid_latex_ocr` — formula image → LaTeX (MIT)
- `rapid_layout` — layout region detection (Apache-2.0)
- `rapid_table` — table structure → HTML (Apache-2.0)
- `onnxruntime-gpu` or `onnxruntime-directml` or `onnxruntime` (pick one for your hardware)

All RapidAI imports are guarded — the ingestion sub-module loads cleanly without them installed.

---

## License

See LICENSE for details.

---

## Contributing

Contributions welcome! Please open an issue or pull request. Key areas for contribution (per [gap analysis](docs/gap-analysis.md)):

- **P1: RapidAI version pinning** — pin exact versions + runtime warning ✅ (closed)
- **P2: Structured logging** — replace `print()` with stdlib logging ✅ (closed)
- **P2: PDF Ingest API** — `OKFRouter.ingest_pdf()` programmatic API ✅ (closed)
- **P2: WAL mode** — enable SQLite WAL for concurrent read access
- **P2: URL allowlist** — restrict `--allow-remote-images` to safe domains
- **P2: TOML config** — add `okfgraph.toml` + env var support
- **P2: Query latency tracking** — log search query latency
- **P2: Embedding cache hit rates** — track ONNX model cache hit/miss
- **P3: LLM tool definition** — add `ingest_pdf` tool to `tools.py`
- GPU performance benchmarking at scale (10k+ concepts)
- Real-world OKF bundle testing
- `--skip-embedding` flag for faster imports without ONNX encoding
- ONNX/Rapid end-to-end PDF tests (see `docs/ONNX_RAPID_IMPLEMENTATION.md` testing checklist)
- Office file conversion via `office_oxide`
- Delta detection: directory-level hash aggregation (skip entire subtrees)
- Delta detection: soft-delete with recovery window (undo purge within N hours)
- `okf index-status` command — report epoch, dirty state, last rebuild time
- `okf schema --version` command — inspect schema version
