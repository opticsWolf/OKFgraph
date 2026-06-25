# OKFgraph

**Ladybug-backed knowledge graph with ONNX-optimized Jina v5 embeddings and multimodal image ingestion.**

OKFgraph is a Python library and CLI tool for building, querying, and managing knowledge graphs from Markdown/OKF documents. It combines graph traversal, hybrid semantic search, and — in v4.0 — unified multimodal image embedding into a single SQLite-backed system.

---

## Features

| Category | Features |
|---|---|
| **Embeddings** | ONNX-optimized Jina v5 text model, SentenceTransformer omni model (lazy-loaded), Matryoshka truncation (32–1024 dims, default 512) |
| **Search** | Hybrid RRF fusion (vector + FTS), graph traversal, direct lookup, image search via unified index |
| **Storage** | LadybugDB — graph + vector + full-text search in a single SQLite file |
| **Images** | Three ingestion modes (`text` / `optional` / `omni`), unified vector space for text + image, content-hash dedup, `okf-asset://` protocol |
| **Import/Export** | OKF Markdown round-trip, batch import with single-transaction upsert, filtered bulk export |
| **CLI** | 14 commands + interactive REPL, `--mode` for image ingestion, `--device cuda` with auto-fallback |
| **LLM Tools** | 5 OpenAI-compatible tool definitions for agent integration |

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
    device="cpu",             # or "cuda" with onnxruntime-gpu
)

# Import an OKF bundle
ids = router.import_bundle(mode="optional")
print(f"Imported {len(ids)} concepts")

# Hybrid search
results = router.search_hybrid("machine learning basics", limit=5)
for r in results:
    print(f"  [{r['relevance_score']:.3f}] {r['title']} ({r['type']})")

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

# Hybrid search
okf search "graph neural networks" --type section --limit 5

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
Concept        ImageAsset       Directory       BrokenLink
├── id         ├── id           ├── id          ├── id
├── type       ├── file_name    └── (empty)     ├── source_id
├── title      ├── mime_type                 ├── target_id
├── body       ├── alt_text                └── timestamp
├── embedding  ├── caption
└── extra      ├── embed_route
               ├── content_hash
               ├── data (BLOB)
               └── embedding

CONTAINS: Directory → Directory / Concept
LINKS_TO: Concept → Concept
INCLUDES_ASSET: Concept → ImageAsset
```

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
| `import_bundle(bundle_path, batch_size, mode)` | Import entire bundle with batch encoding |
| `export_to_okf(concept_id, output_path)` | Export a single concept to OKF markdown |
| `export_bundle(output_dir, directory_id, concept_type, tags)` | Export filtered bundle |
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
| `okf search <query>` | Hybrid search |
| `okf search-images <query>` | Find images via unified vector index |
| `okf traverse <id>` | Graph traversal |
| `okf list [dir]` | List directory contents |
| `okf get <id>` | Fetch full concept |
| `okf export --all` | Export entire bundle |
| `okf broken-links` | List orphaned links |
| `okf repair-links` | Repair broken links |
| `okf shell` | Interactive REPL |

### Global Options

| Option | Default | Description |
|---|---|---|
| `--db` | `okfgraph.db` | Database file path |
| `--bundle` | `.` | Bundle root directory |
| `--dim` | `512` | Embedding dimension (32–1024) |
| `--cache-dir` | `~/.cache/huggingface` | HuggingFace model cache |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--omni-model-id` | `jinaai/jina-embeddings-v5-omni-small-retrieval` | Multimodal model ID |

### Import Options

| Option | Default | Description |
|---|---|---|
| `--mode` | `text` | Image ingestion mode: `text`, `optional`, `omni` |
| `--allow-remote-images` | — | Fetch `http(s)://` image URLs |
| `--batch-size` | `32` | Batch size for encoding |

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
| `test_router.py` | 29 | ✅ All passing |
| `test_cli.py` | 19 | ✅ All passing |
| `test_images.py` | 10 | ✅ All passing |
| `test_integration.py` | 19 steps | ✅ All passing |
| **Total** | **58** | **All passing** |

---

## Project Structure

```
okfgraph/
├── okfgraph/
│   ├── __init__.py          # Exports: ConceptModel, OKFRouter, cli_main
│   ├── models.py            # ConceptModel + ImageAssetModel
│   ├── router.py            # OKFRouter — ~1500 lines
│   ├── cli.py               # CLI + interactive shell — ~400 lines
│   ├── tools.py             # LLM tool definitions (5 tools)
│   └── images.py            # Image ingestion logic — 10 unit tests
├── tests/
│   ├── test_router.py       # 29 unit tests
│   ├── test_cli.py          # 19 CLI tests
│   ├── test_images.py       # 10 image logic tests
│   └── test_integration.py  # Full end-to-end tests
├── benchmarks/
│   └── benchmark_500.py
├── examples/
│   └── oxide_to_markdown_converter.py
├── architecture_v2.md       # Full architecture specification (v4.0)
├── implementation_status.md # Detailed implementation status
├── IMPLEMENTATION_NOTES.md  # v4.0 "Unified Omni" details
├── pyproject.toml           # Project metadata
└── requirements.txt         # Dependencies
```

---

## Requirements

- Python 3.10+
- `ladybug>=0.17`
- `pydantic>=2.0`
- `python-frontmatter>=1.0`
- `pyyaml>=6.0`
- `optimum[onnxruntime]>=1.20`
- `transformers>=4.57`
- `torch>=2.5`
- `sentence-transformers>=3.0`
- `Pillow>=10.0`
- `pytest>=8.0` (dev)

---

## License

See LICENSE for details.

---

## Contributing

Contributions welcome! Please open an issue or pull request. Key areas for contribution:

- GPU performance benchmarking at scale (10k+ concepts)
- Real-world OKF bundle testing
- Concept temporal dual-tracking (`created_date`/`modified_date`)
- `okf-asset://` link rewriting on ingest
- Documentation and examples
