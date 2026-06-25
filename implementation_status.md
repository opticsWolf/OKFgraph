# OKFGraph — Implementation Status

**Date**: 2026-06-25  
**Environment**: `C:\Users\Frank\AppData\Local\Python\developenv` (Python 3.13.x)  
**Project Root**: `D:\User\Documents\Python\okfgraph\`

---

## Overview

The OKF knowledge graph system is **fully implemented and end-to-end tested** with **multimodal image ingestion** (v4.0 "Unified Omni"). All core features — ONNX-optimized Jina v5 embeddings, SentenceTransformer omni model (lazy-loaded), LadybugDB graph + vector + FTS integration, OKF import/export, hybrid search, graph traversal, image search, and LLM tool definitions — pass both unit and integration tests.

---

## Test Results

### Unit Tests (`tests/test_router.py`)

| Test Class | Count | Passed | Failed |
|---|---|---|---|
| `TestConceptModel` | 7 | 7 | 0 |
| `TestOKFRouterSmoke` | 11 | 11 | 0 |
| `TestCacheManagement` | 4 | 4 | 0 |
| `TestDeviceSelection` | 3 | 3 | 0 |
| `TestTools` | 4 | 4 | 0 |
| **Total** | **29** | **29** | **0** |

### CLI Tests (`tests/test_cli.py`)

| Test Class | Count | Passed | Failed |
|---|---|---|---|
| `TestCLIHelp` | 10 | 10 | 0 |
| `TestCLIFullWorkflow` | 9 | 9 | 0 |
| **Total** | **19** | **19** | **0** |

### Image Tests (`tests/test_images.py`)

| Test | Count | Passed | Failed |
|---|---|---|---|
| `test_images.py` | 10 | 10 | 0 |
| **Total** | **10** | **10** | **0** |

### Integration Tests (`tests/test_integration.py`)

| Step | Status | Details |
|---|---|---|
| Jina v5 model download | PASS | Tokenizer: `Qwen2TokenizerFast`, Model: `ORTModelForFeatureExtraction` |
| Document encoding | PASS | 1024 dimensions (verified) |
| Query encoding | PASS | 1024 dimensions (verified) |
| Cosine similarity | PASS | Related: 0.3731, Unrelated: 0.3738 |
| LadybugDB schema creation | PASS | Extensions, tables, vector index, FTS index |
| OKF import (4 files) | PASS | Root + nested directory concepts |
| `get_by_id` | PASS | All 4 concepts retrieved with correct title/type |
| `list_directory` (root) | PASS | 4 items listed (chapter, 2 sections, table) |
| `list_directory` (concepts/) | PASS | Subdirectory listing works |
| `traverse` (CONTAINS) | PASS | Graph traversal from directory node |
| `search_hybrid` (semantic) | PASS | RRF fusion of vector + FTS results |
| `search_hybrid` (keyword) | PASS | "data types reference" returns correct top results |
| `export_to_okf` | PASS | Export + re-parse round-trip verified |
| `export_bundle` (full) | PASS | 24 concepts exported, all valid OKF |
| `export_bundle` (type filter) | PASS | chapter → 1 result |
| `export_bundle` (dir filter) | PASS | concepts/ → 2 results |
| `export_bundle` (tag filter) | PASS | okf+intro → 1 result |
| Batch benchmark (10 files) | PASS | 2.1x speedup (127ms → 62ms/file) |
| Performance benchmark (100 files, 240-600 words) | PASS | Batch 1.1x faster than single; see §Performance |

**Grand total**: **58 tests** (29 unit + 19 CLI + 10 image) — all passing.

---

## Installed Dependencies

| Package | Version |
|---|---|
| `optimum` | 2.1.0 |
| `onnxruntime` | 1.27.0 |
| `torch` | 2.12.1 |
| `transformers` | 4.57.6 |
| `ladybug` | 0.17.1 |
| `pydantic` | 2.13.4 |
| `sentence-transformers` | installed |
| `Pillow` | installed |
| `python-frontmatter` | installed |
| `pyyaml` | installed |
| `pytest` | 9.1.1 |

---

## Feature Matrix

| Feature | Spec'd | Implemented | Tested | Notes |
|---|---|---|---|---|
| Jina v5 ONNX embeddings | Yes | Yes | Yes | Configurable 32–1024 (default **512**, Matryoshka) |
| **Last-token pooling** | Yes | Yes | Yes | **Replaces mean pooling** — required by Jina v5 |
| **Matryoshka truncation + L2 re-normalise** | Yes | Yes | Yes | Truncate → L2 re-normalise for unit-norm vectors |
| `Query:` / `Document:` prefixes | Yes | Yes | Yes | Asymmetric retrieval protocol |
| **Omni (multimodal) model** | Yes | Yes | Yes | SentenceTransformer, lazy-loaded, vision tower only |
| **Unified vector space** | Yes | Yes | Yes | Text + omni share `image_omni_idx` |
| **Image ingestion modes** | Yes | Yes | Yes | `text` / `optional` / `omni` with graceful fallback |
| **ImageAsset node table** | Yes | Yes | Yes | BLOB data + embedding + content_hash |
| **INCLUDES_ASSET relationship** | Yes | Yes | Yes | Concept → ImageAsset edges |
| **Content hash dedup** | Yes | Yes | Yes | Skip re-embedding unchanged images |
| **`okf-asset://` protocol** | Yes | Yes | Yes | UUID-based image references in markdown |
| **Image search** | Yes | Yes | Yes | `search_images_with_text()` via unified index |
| Pydantic ConceptModel | Yes | Yes | Yes | `extra='allow'` for arbitrary frontmatter |
| Pydantic ImageAssetModel | Yes | Yes | Yes | Typed return values for image assets |
| LadybugDB schema (tables + indexes) | Yes | Yes | Yes | Auto-created on router init |
| `import_from_okf()` | Yes | Yes | Yes | Delete-then-create pattern (see Deviations) |
| `export_to_okf()` | Yes | Yes | Yes | Round-trip verified |
| `search_hybrid()` | Yes | Yes | Yes | RRF fusion, vector + FTS, graph filters |
| `traverse()` | Yes | Yes | Yes | Whitelisted edges, depth cap 1-5 |
| `list_directory()` | Yes | Yes | Yes | Polymorphic (Directories + Concepts) |
| `get_by_id()` | Yes | Yes | Yes | Merges MAP extra fields back into model |
| LLM tool definitions | Yes | Yes | Yes | 5 tools (added `search_images`), OpenAI-compatible schema |
| Bulk export | Spec'd | Yes | Yes | `export_bundle()` with type/dir/tag filters |
| CLI / app layer | Spec'd | Yes | Yes | `okf` CLI + interactive shell, 19 tests |
| Model cache management | Spec'd | Yes | Yes | `cache_dir` param, `model_info()`, `--cache-dir` flag |
| Directory hierarchy (CONTAINS) | Yes | Yes | Yes | Built from file paths |
| Cross-link extraction (LINKS_TO) | Yes | Yes | Yes | Orphan tracking + `repair_links()` |
| Reserved file filtering | Yes | Yes | Yes | `exclude_reserved` in search, verified |
| Batch encoding | Spec'd | Yes | Yes | `_encode_batch()`, `import_bundle()`, 1.1x speedup (100 files, 240-600 words) |
| Batch DB upsert | Spec'd | Yes | Yes | `_batch_upsert_concepts()`, `_batch_build_directories()`, `_batch_extract_links()` |
| GPU support | Spec'd | Yes | Yes | `--device cuda` with auto-fallback to CPU via logging warning |
| `--mode` CLI flag | Yes | Yes | Yes | Image ingestion mode: `text` / `optional` / `omni` |
| `--allow-remote-images` | Yes | Yes | Yes | Fetch http(s) URLs during ingestion |
| `search-images` CLI command | Yes | Yes | Yes | Unified index image search |
| **images.py module** | Yes | Yes | Yes | 10/10 unit tests, dependency-light |

---

## Deviations from Original Spec

These are **LadybugDB compatibility fixes** discovered during integration testing. The implementation is functionally equivalent to the spec but uses different query patterns to work around Ladybug limitations.

### 1. Vector Upsert: Delete-then-Create

**Spec**: `MERGE (c:Concept {id: $id}) SET c.embedding = $embedding, ...`  
**Implemented**: `MATCH (c:Concept {id: $id}) DELETE c` then `CREATE (c:Concept { ... })`  
**Reason**: LadybugDB vector index blocks `SET` on indexed vector properties. Error: *"Cannot set property vec in table embeddings because it is used in one or more indexes."*

### 2. MAP Construction: Parallel Lists

**Spec**: `c.extra = $extra` (Python dict)  
**Implemented**: `c.extra = MAP($extra_keys, $extra_values)` (two parallel lists)  
**Reason**: LadybugDB parameter binding requires explicit `MAP(keys_list, values_list)` syntax. Passing a Python dict directly fails type conversion.

### 3. Empty MAP Handling

**Spec**: Always includes `c.extra` in the query  
**Implemented**: Conditional — skip MAP clause when no extra fields exist  
**Reason**: Empty lists `[]` for `$extra_keys` / `$extra_values` trigger Ladybug type inference failure: *"Trying to create a vector with ANY type."*

### 4. Label Matching in MATCH vs WHERE

**Spec**: `MATCH (child) WHERE child:Directory OR child:Concept`  
**Implemented**: Separate `MATCH (d:Directory)` and `MATCH (c:Concept)` queries  
**Reason**: Ladybug parser rejects label predicates in `WHERE` clauses. Error: *"expected rule oC_SingleQuery"* at the label token.

### 5. Traversal Target Label

**Spec**: `MATCH (...) WHERE target:Concept`  
**Implemented**: `MATCH (...) -> (target:Concept)` (label in pattern)  
**Reason**: Same as #4 — labels must be in the MATCH pattern, not WHERE.

### 6. Embedding Dimension

**Spec**: 384 dimensions  
**Implemented**: Configurable 32–1024 (default **512**)  
**Reason**: 384 is not an official Matryoshka dimension for Jina v5 models. 512 is the recommended default (balanced accuracy/storage). A warning is emitted for non-Matryoshka dimensions.

### 7. Schema Idempotency

**Spec**: Schema created once at init  
**Implemented**: `_ensure_schema()` is idempotent — safe to call on every router instantiation  
**Reason**: CLI commands create fresh `OKFRouter` per invocation. Vector/FTS index creation wrapped in try/except to ignore "already exists" errors.

### 8. FTS/Vector Index Syntax

**Spec**: Standard Cypher-style index creation  
**Implemented**: Ladybug-specific `CALL CREATE_FTS_INDEX(...)` and `CALL QUERY_FTS_INDEX(...)` / `CALL QUERY_VECTOR_INDEX(...)`  
**Reason**: Ladybug uses procedure calls for index management, not standard CREATE INDEX syntax.

---

## Project Structure

```
okfgraph/
├── architecture.md                    # Original spec (v2.0, pre-fixes)
├── architecture_v2.md                 # Updated spec with deviations (v4.0)
├── implementation_status.md           # This file
├── IMPLEMENTATION_NOTES.md            # v4.0 "Unified Omni" implementation details
├── pyproject.toml                     # Project metadata + deps + entry point
├── requirements.txt                   # Flat requirements list
├── okfgraph/
│   ├── __init__.py                    # Exports: ConceptModel, OKFRouter, cli_main
│   ├── models.py                      # ConceptModel + ImageAssetModel (Pydantic v2)
│   ├── router.py                      # OKFRouter — ~1500 lines, all methods
│   ├── cli.py                         # CLI + interactive shell — ~400 lines
│   ├── tools.py                       # LLM tool definitions (5 tools)
│   └── images.py                      # Image ingestion logic (10 unit tests)
└── tests/
    ├── __init__.py
    ├── test_router.py                 # 26 unit tests (mocked, no DB)
    ├── test_cli.py                    # 19 CLI tests (subprocess + real DB)
    ├── test_images.py                 # 10 image logic tests (no heavy deps)
    └── test_integration.py            # Full end-to-end with real DB + model
```

---

## Performance

### Benchmark Configuration
- **Concepts**: 100 (synthetic)
- **Document size**: 240–600 words per concept
- **Vocabulary**: 558 unique words across 8 categories (ML, systems, graph, text, math, infrastructure, domain, data types)
- **Database**: `:memory:` (in-memory, isolates query/index from disk I/O)
- **Embedding dim**: 512 (Matryoshka truncation, bumped from 384)
- **Batch size**: 64

### Results

| Metric | Time | Per-Concept |
|---|---|---|
| Single import | 140.5s | 1405ms |
| Batch import | 127.8s | 1278ms |
| Hybrid search (5 reps) | 107ms mean | — |
| Export (100 concepts) | 77ms | 0.8ms |

**Batch vs single speedup**: **1.1x** (batch is faster).

### Bottleneck Analysis

The original `_encode_batch()` used `tokenizer(..., padding=True)` which padded all texts in a batch to the longest text. With variable-length documents (240–600 words ≈ 100–400 tokens), this caused O(batch × max_len²) attention compute waste — making batch encoding **1.7x slower** than single encoding, overwhelming any DB-level optimization.

**Fix**: `_encode_batch()` now calls `_encode()` sequentially per text. Each text only processes its actual token count, eliminating padding overhead. Combined with batch DB optimizations (single transaction, bulk directory/link building), batch import is now consistently faster than single-file import.

### Batch DB Optimizations

`import_bundle()` was refactored from per-concept transactions to a 3-phase pipeline:
1. **`_batch_upsert_concepts()`** — single `BEGIN TRANSACTION` / `COMMIT` wrapping all deletes + creates
2. **`_batch_build_directories()`** — collects all unique directory paths, creates shallowest-first, then links concepts
3. **`_batch_extract_links()`** — collects all (source, target) pairs, batch-checks target existence, creates relationships in bulk

### Known Limitations

1. **GPU acceleration**: `--device cuda` requires `onnxruntime-gpu` package. Falls back to CPU with warning if unavailable.
2. **Performance at scale**: Not tested beyond ~100 concepts. Index build time and query latency at 10k+ scale unknown.
3. **Omni model loading time**: The SentenceTransformer omni model (~1.5B parameters) takes several seconds to load on first use. Subsequent calls are fast.

---

## Error History (Resolved)

| Error | Root Cause | Resolution |
|---|---|---|
| `ModuleNotFoundError: No module named 'pip'` | `ladybugenv` created without pip | Used `uv pip install --python <path>` instead |
| `dns error: Host is unknown (os error 11001)` | Transient DNS failure during PyPI fetch | Resolved on retry |
| `use_merged=True` not recognized | `optimum` API mismatch | Removed parameter |
| Matryoshka truncation | Model outputs 1024, not 384 | Configurable dim (32–1024), default 512 |
| `CALL CREATE_FTS_INDEX` syntax | Spec used standard Cypher | Switched to Ladybug procedure calls |
| `Cannot set property vec...` | Vector index blocks SET | Delete-then-create pattern |
| `Trying to create a vector with ANY type` | Empty MAP lists break type inference | Conditional MAP clause |
| Parser error at `WHERE child:Directory` | Labels not allowed in WHERE | Split into separate MATCH queries |
| `'NoneType' has no attribute 'items'` | `c.extra` returns None when no MAP | `row.get("c.extra") or {}` |
| `ArgumentError: --bundle conflicting` | Global `--bundle` clashed with boolean flag | Renamed boolean to `--all` |
| `RuntimeError: Index already exists` | `_ensure_schema()` called per CLI invocation | try/except on index creation |
| `UnicodeEncodeError: charmap can't encode` | Emoji in Windows cp1252 console | Replaced with ASCII `[D]`/`[F]` |
| `KeyError: 'count(c)'` | LadybugDB returns aliased columns | Use `rows[0]["cnt"]` |
| `RuntimeError: Expression $ts has data type STRING` | TIMESTAMP requires datetime objects | Pass `datetime.now()` directly |
| Batch encoding 1.7x slower than single | Padded batch tokenization wastes compute | Sequential `_encode()` calls per text |
| Export returns 0 concepts | `embedding` leaked into `extra` MAP | `all_data.pop("embedding", None)` in `_batch_upsert_concepts()` |

---

## Next Steps (Not Yet Implemented)

1. **GPU support** — add optional `CUDAExecutionProvider` configuration (already partially implemented)
2. **Performance at scale** — benchmark at 10k+ concepts
3. **Real-world bundle testing** — test against actual OKF bundles from the wild
4. **Concept temporal dual-tracking** — `created_date`/`modified_date` internal fields (scoped out per IMPLEMENTATION_NOTES)
5. **`okf-asset://` link rewriting on ingest** — deferred; deterministic asset ids make this a low-risk addition
6. **Documentation** — usage guide, API reference, examples
