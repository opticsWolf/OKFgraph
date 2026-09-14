# OKF Knowledge Graph — Architecture Specification

**Version**: 6.1 (as-built for okfgraph 0.4.0 — §16 multi-root; supersedes the v5.x design lineage as the authoritative surface)  
**Based on**: Architecture v5.9 (Core Gaps closure, 2026-07-09)  
**Verified against**: LadybugDB v0.20.3, Python 3.11–3.13, `embroider 0.1.3`, `bobine 0.5.11`, `onnxruntime==1.29.0`

> **Scope note.** The v5.x lineage (and `docs/gap-analysis.md`,
> `docs/OKF Graph V6.0.md`, `docs/Okfgraph 6.0 amendment.md`,
> `docs/ONNX_RAPID_IMPLEMENTATION.md`) are **historical design records** —
> they describe an optimum/transformers embedding stack and an in-tree
> RapidAI ingest engine that no longer exist. This document describes the
> shipped system: external `embroider` embedding crate, bobine converter
> seam, MCP ≥ 2.0 with 5 tools, consolidated CLI, model-free PPR
> retrieval. Sections kept verbatim from v5.9 are those whose claims
> still hold against the 0.2.12 tree; every changed claim below was
> re-verified against code.

**Storage**: LadybugDB (v0.20.3) — graph + vector + full-text search.  
**Data Model**: Pydantic v2 with `extra='allow'` — preserves OKF extensibility, maps cleanly to Ladybug's `MAP` and `LIST` columns.  
**Embedding Engine**: Jina v5 text model (`jinaai/jina-embeddings-v5-text-small-retrieval`) via the external **`embroider`** crate (github.com/opticsWolf/embroider — Rust/ORT, no torch, no transformers, no optimum anywhere in core).  
**Multimodal Engine**: SentenceTransformer with `jinaai/jina-embeddings-v5-omni-small-retrieval` (vision tower, lazy-loaded, `omni` extra only).  
**Unified Vector Space**: Both encoders write into one `ImageAsset.embedding` column indexed by `image_omni_idx`.  
**Chunking**: Mordant (Rust-based Markdown parser) with heading context injection and structural block boundaries.  
**Search Modes**: Hybrid (RRF fusion, vector + FTS), chunk-level RRF with matched-chunk attachment, graph traversal, direct ID lookup, image search (text→image via unified index), **model-free PPR retrieval** (works with no embedding model loaded).

---

## Summary of Changes (v5.8 → v5.9)

### Core Gaps Closure (2026-07-09)

| Area | v5.8 | v5.9 | Reason |
|---|---|---|---|
| **Concurrency** | WAL only | **filelock-based write locking** | Multi-process write safety |
| **Security** | URL allowlist only | **path traversal sandboxing + cache verification** | SSRF prevention, supply chain |
| **Config** | TOML + env vars | **schema validation** | Reject invalid values early |
| **PDF Tests** | None | **6 end-to-end tests** | Pipeline verification |
| **Test Coverage** | 300 tests | **335 tests** | Security, config, PDF tests |
| **Version bump** | 5.8 | **5.9** | Core gaps closure |

### Gap #7b: Filelock Write Locking

Added `InterProcessLock` from `fasteners` library for multi-process write safety:

- Lock file created alongside the DB (`okfgraph.db.lock`)
- 5-minute timeout for lock acquisition
- All write operations wrapped: `import_bundle`, `ingest_md`, `ingest_thoughts`, `reindex`, soft-delete methods
- Lock released on router `close()`

### Gap #9b: Path Traversal Sandboxing

Added `okfgraph/security.py` module with:

- `is_path_safe()` — validates file paths are within bundle root
- `is_private_ip()` — blocks private/internal IP ranges
- `validate_image_src()` — comprehensive image source validation
- `ModelCacheVerifier` — SHA-256 hash verification for model cache files

Integrated into `load_image_bytes()` in `images.py`:
- Blocks `file://` URLs (SSRF risk)
- Validates local paths are within bundle root
- Supports `bundle_root` parameter for path validation

### Gap #9d: Model Cache Verification

`ModelCacheVerifier` class for supply chain security:

- Register expected hashes for model files
- Verify files on first load, cache results for speed
- Warn on hash mismatches (first-time loads trusted)

### Gap #11b: TOML Schema Validation

Added `validate()` methods to all config dataclasses:

- `DatabaseConfig`: path non-empty, dim in [32, 1024], recommended Matryoshka dims
- `EmbeddingConfig`: device in [cpu, cuda, mps, auto], cache_dir absolute
- `ImportConfig`: mode in [text, optional, omni], batch_size in [1, 256], chunk_size in [64, 8192]
- `OKFConfig`: aggregates all section validations

Validation warnings logged on config load (non-blocking — CLI args can override).

### Gap #12c: End-to-End PDF Tests

Created `tests/test_pdf_e2e.py` with 6 tests:

- PDF ingestion creates searchable concept
- PDF output-only mode (no auto-import)
- FileNotFoundError for nonexistent PDF
- Progress callback invocation
- Router pipeline includes linting
- Router pipeline includes image staging

---

## Summary of Changes (v5.7 → v5.8)

### Phase 4: Operations (2026-07-09)

| Area | v5.7 | v5.8 | Reason |
|---|---|---|---|
| **Configuration** | CLI args only | **TOML config + env vars** | Persistent settings, team-wide standardisation |
| **WAL Mode** | Not available | **`wal_mode=True` on OKFRouter** | Concurrent reads during writes |
| **URL Allowlist** | No domain filtering | **`allowed_image_domains` + `_domain_allowed()`** | SSRF prevention for remote images |
| **Soft-Delete** | Hard purge only | **DeletedConcept table + recovery window** | Undo purge within 24h |
| **Schema Version** | 4 (DirHash) | **5 (DeletedConcept)** | Soft-delete support |
| **CLI Commands** | 22 commands | **25 commands (+deleted-list, -recover, -purge)** | Soft-delete management |
| **Test Coverage** | 300 tests | **300 tests** | Config, WAL, allowlist, soft-delete tests |
| **Version bump** | 5.7 | **5.8** | Phase 4 completion |

## Summary of Changes (v5.4 → v5.5)

### Directory-Level Hash Aggregation (Gap #1b — 2026-07-08)

| Area | v5.4 | v5.5 | Reason |
|---|---|---|---|
| **Directory-level hash aggregation** | Not present | **DirHash table + `_changed_directories()`** | Skip entire subtrees when unchanged |
| **Purge of deleted directories** | File-level only | **Directory-level with stored file paths** | Purge all concepts in deleted subtrees |
| **Schema migration** | v3 | **v4 (DirHash table)** | Subtree-level delta detection |
| **Version bump** | 5.4 | **5.5** | Gap #1b closure |

## Summary of Changes (v5.3 → v5.4)

### Gap Analysis (Gap Analysis v3.0 — 2026-07-05)

| Area | v5.3 | v5.4 | Reason |
|---|---|---|---|
| **Gap analysis consolidation** | Not present | **15 gaps reviewed, 13 closed, 2 open** | Production-readiness assessment |
| **Open gap documentation** | Not present | **§15 — 2 open gaps documented** | Concurrent access, security |
| **Closed gaps** | — | **#5 PDF→import, #6 error isolation, #8 schema migration, #12 missing tests, #13 index health, #14 chunk limits, #16 LLM tool coverage, #10 observability, #15 RapidAI pinning, #5b ingest_pdf, #5c ingest_md + ingest_thoughts + linting** | Core reliability, feature completeness, observability |
| **Gap #7 (Concurrent Access)** | Not documented | **Documented** — WAL mode + single-writer constraint recommendation (OPEN) | Data integrity under concurrent use |
| **Gap #9 (Security)** | Not documented | **Documented** — SSRF risk, URL allowlist recommendation (OPEN) | Security architecture |
| **Version bump** | 5.3 | **5.4** | Gap analysis artifact |

---

## 1. Dependencies

### Core (required)

```bash
pip install okfgraph   # ladybug, embroider, onnxruntime, mordant, mcp, …
```

```toml
# pyproject.toml (0.2.12) — deliberately few, all pinned or floor-pinned:
ladybug == 0.20.3            # graph + vector + FTS store (newer 0.20.x segfaults index builds — pinned)
embroider >= 0.1, < 0.2      # Jina v5 text embeddings (external Rust/ORT crate, PyPI wheels)
onnxruntime == 1.29.0        # ONE pinned ORT binary, shared by bobine + embroider (load-dynamic)
mordant >= 0.9               # Rust GFM chunking
mcp >= 2.0                   # MCP server surface
pydantic >= 2.0, python-frontmatter, pyyaml, numpy, fasteners
```

> **No torch, no transformers, no optimum in core.** All text-embedding
> numerics (tokenize → last-token pooling → L2 → Matryoshka truncate →
> re-normalise) live in the `embroider` Rust crate; nothing is computed in
> Python. The Python side owns orchestration: dylib resolution
> (`resolve_ort_dylib()` → `ORT_DYLIB_PATH`), lazy session lifecycle
> (`LazyRustEncoder`), and explicit air-gapped model paths.

**Verified versions** (0.2.12 tree):
- `ladybug==0.20.3`, `onnxruntime==1.29.0`, `embroider==0.1.3`, `mordant` (Rust GFM parser)
- Single pinned ORT binary shared by bobine + embroider — do not float.

### PDF conversion (optional — bobine)

```bash
pip install "okfgraph[pdf]"   # bobine>=0.5: PDF/Office/text → Markdown engine
```

PDF conversion runs through the `bobine` Rust engine behind the
`DocumentConverter` plugin seam (`okfgraph/components/converters.py`, default
`BobineConverter`). No converter code lives in okfgraph — see §10 and
`docs/converters.md`.

### Multimodal images (optional — omni)

```bash
pip install "okfgraph[omni]"   # sentence-transformers + Pillow (image embeddings only)
```

Text-only installs never touch torch. The omni model
(`jinaai/jina-embeddings-v5-omni-small-retrieval`, vision tower) loads
lazily on first image encode.

---

## 2. Database Schema (Ladybug)

```cypher
-- Extensions must be installed and loaded
INSTALL vector;
LOAD vector;
INSTALL fts;
LOAD fts;

CREATE NODE TABLE Concept (
    id STRING PRIMARY KEY,
    type STRING,
    title STRING,
    description STRING,
    resource STRING,
    tags STRING[],                     -- native list
    timestamp TIMESTAMP,               -- native timestamp type
    body STRING,
    embedding FLOAT[dim],              -- Jina v5 text model output (Matryoshka: configurable 32-1024, default 512)
    extra MAP(STRING, STRING)          -- arbitrary OKF frontmatter keys
);

CREATE NODE TABLE ImageAsset (
    id STRING PRIMARY KEY,
    file_name STRING,
    mime_type STRING,
    alt_text STRING,
    caption STRING,                    -- provenance: how the image was embedded
    embed_route STRING,                -- "text" | "omni"
    content_hash STRING,               -- change-detection key for re-embedding
    data BLOB,                         -- raw image bytes
    embedding FLOAT[dim]               -- shared vector space (text or omni)
);

CREATE NODE TABLE Directory (id STRING PRIMARY KEY);

CREATE NODE TABLE BrokenLink (
    id STRING PRIMARY KEY,
    source_id STRING,
    target_id STRING,
    timestamp TIMESTAMP
);

CREATE NODE TABLE FileHash (
    path STRING PRIMARY KEY,
    hash STRING,                  -- SHA-256 hex digest of file contents
    concept_id STRING             -- maps file path → Concept.id
);

CREATE NODE TABLE Chunk (
    id STRING PRIMARY KEY,
    parent_doc_id STRING,
    chunk_index INTEGER,
    chunk_text STRING,
    block_type STRING,                -- "paragraph", "heading", "code", "list", "blockquote", "table", "diagram"
    start_offset INTEGER,
    end_offset INTEGER,
    embedding FLOAT[dim]              -- Jina v5 text model output
);

CREATE REL TABLE PART_OF (FROM Concept TO Chunk);

CREATE REL TABLE CONTAINS (
    FROM Directory TO Directory,
    FROM Directory TO Concept
);

CREATE REL TABLE LINKS_TO (FROM Concept TO Concept);

CREATE REL TABLE INCLUDES_ASSET (FROM Concept TO ImageAsset);

-- Vector index for ANN search on Concept.embedding
CALL CREATE_VECTOR_INDEX(
    'Concept', 'concept_embedding', 'embedding',
    mu := 30, ml := 60, metric := 'cosine', efc := 200
);

-- Full-text index on combined search text (title + description + body)
CALL CREATE_FTS_INDEX('Concept', 'concept_fts', ['title', 'description', 'body']);

-- Chunk vector index for ANN search on Chunk.embedding
CALL CREATE_VECTOR_INDEX(
    'Chunk', 'chunk_embedding', 'embedding',
    mu := 30, ml := 60, metric := 'cosine', efc := 200
);

-- Chunk full-text index on chunk_text
CALL CREATE_FTS_INDEX('Chunk', 'chunk_fts', ['chunk_text']);

-- Unified image vector index (shared space: text-model alt-text vectors + omni-model image vectors)
CALL CREATE_VECTOR_INDEX(
    'ImageAsset', 'image_omni_idx', 'embedding',
    mu := 30, ml := 60, metric := 'cosine', efc := 200
);
```

### Matryoshka Dimensions

Both `jinaai/jina-embeddings-v5-text-small-retrieval` and `jinaai/jina-embeddings-v5-omni-small-retrieval` support Matryoshka truncation at these official dimensions:

| Dimension | Use Case |
|---|---|
| 1024 | Maximum fidelity |
| 768 | High fidelity |
| **512** | **Default — balanced accuracy/storage** |
| 256 | Moderate |
| 128 | Compact |
| 64 | Very compact |
| 32 | Minimal |

The default dimension was bumped from 384 → 512 because 384 is not an official Matryoshka dimension for these models.

---

## 2a. Ladybug Query Syntax — Verified Patterns

LadybugDB uses specific syntax that differs from standard Cypher. All patterns below are **verified against Ladybug v0.20.3**.

### Index Management

```cypher
-- Create FTS index
CALL CREATE_FTS_INDEX('Table', 'index_name', ['col1', 'col2', 'col3']);

-- Query FTS index
CALL QUERY_FTS_INDEX('Table', 'index_name', 'search query')
RETURN node, score;

-- Create vector index
CALL CREATE_VECTOR_INDEX('Table', 'index_name', 'column', mu := 30, ml := 60, metric := 'cosine', efc := 200);

-- Query vector index (ANN)
CALL QUERY_VECTOR_INDEX('Table', 'index_name', $vec, $k)
RETURN node, distance;
```

### MAP Construction

Ladybug requires two parallel lists, not a Python dict:

```cypher
-- Correct: parallel key/value lists
CREATE (n:Node { extra: MAP($keys, $values) })

-- WRONG: passing a dict directly
CREATE (n:Node { extra: $extra_dict })  -- type conversion error
```

**Note**: Empty lists `[]` for keys/values trigger type inference failures on vector columns. Use conditional query construction to skip the MAP clause when no extra fields exist.

### Vector Upsert

Indexed vector properties block `SET`. Use delete-then-create:

```cypher
-- Correct upsert pattern
MATCH (c:Concept {id: $id}) DELETE c;
CREATE (c:Concept { id: $id, embedding: $vec, ... });

-- WRONG: MERGE + SET on indexed vector
MERGE (c:Concept {id: $id}) SET c.embedding = $vec;  -- RuntimeError
```

### Label Matching

Labels must appear in the MATCH pattern, not in WHERE:

```cypher
-- Correct: label in MATCH pattern
MATCH (n:Concept) WHERE n.type = 'chapter'

-- WRONG: label in WHERE clause
MATCH (n) WHERE n:Concept AND n.type = 'chapter'  -- Parser error
```

For polymorphic queries (matching multiple labels), split into separate MATCH statements:

```cypher
-- Correct: two queries
MATCH (d:Directory) WHERE ... RETURN d.id, 'Directory' AS type
MATCH (c:Concept)   WHERE ... RETURN c.id, c.type AS type

-- WRONG: label predicates in WHERE
MATCH (child) WHERE child:Directory OR child:Concept  -- Parser error
```

---

## 3. Pydantic Models

### ConceptModel

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Any
from datetime import datetime


class ConceptModel(BaseModel):
    id: str
    type: str
    title: Optional[str] = None
    description: Optional[str] = None
    resource: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    timestamp: Optional[datetime] = None
    body: str = ""
    embedding: Optional[List[float]] = None   # internal use only

    model_config = {"extra": "allow"}

    @field_validator('timestamp', mode='before')
    @classmethod
    def parse_timestamp(cls, v: Any) -> Any:
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v
```

**Key**: Extra fields are stored in Ladybug's `extra` MAP column. On read, they are merged back via `model_validate`.

### ImageAssetModel

```python
class ImageAssetModel(BaseModel):
    """Metadata for an image asset stored in the unified vector index."""

    id: str
    file_name: str = ""
    mime_type: str = "application/octet-stream"
    alt_text: Optional[str] = None
    caption: Optional[str] = None       # provenance: how embedded
    embed_route: Optional[str] = None   # "text" | "omni"
    content_hash: Optional[str] = None  # change-detection key
    embedding: Optional[List[float]] = None

    model_config = {"extra": "allow"}
```

---

## 4. OKFRouter — Verified Implementation

### 4.1. Constructor

```python
class OKFRouter:
    ALLOWED_DIMS = (32, 64, 128, 256, 512, 768, 1024)

    def __init__(
        self,
        db_path: str,
        bundle_root: str,
        model_id: str = "jinaai/jina-embeddings-v5-text-small-retrieval",
        omni_model_id: str = "jinaai/jina-embeddings-v5-omni-small-retrieval",
        embedding_dim: int = 512,
        cache_dir: Optional[str] = None,
        model_path: Optional[str] = None,      # explicit local ONNX …
        tokenizer_path: Optional[str] = None,  # … must be given together
        device: str = "cpu",                 # cpu | cuda (+ "auto"/"mps" aliases)
        allow_remote_images: bool = False,
        chunk_size: int = 512, chunk_overlap: int = 40,
        enable_chunking: bool = True,
    ):
        # ... validates dim, resolves the ORT dylib, imports embroider —
        # the text session itself opens LAZILY on first encode (see §4.3)
```

Router construction stays cheap: importing the `embroider` wheel does not
open any ONNX session, and `resolve_ort_dylib()` fixes the shared runtime
choice (single pinned `onnxruntime==1.29.0`, `ORT_DYLIB_PATH`-overridable)
*before* the native module is first used.

### Auto-Detect Embedding Dimension

When opening an **existing** database, `OKFRouter.__init__()` calls `_adopt_existing_embedding_dim()` which:

1. Queries `CALL TABLE_INFO('Concept')` to read the stored column type (`FLOAT[N]`).
2. Extracts the dimension `N` from the column type string.
3. If the stored dimension differs from the requested `embedding_dim`, logs a **warning** and adopts the stored value.

This prevents users from needing to remember `--dim` on re-init. The `--dim` flag acts as an override only for **new** databases; for existing databases, the on-disk dimension always wins. If the database is brand new (no Concept table yet), the requested dimension is used as-is.

### 4.2. Schema Setup

```python
def _ensure_schema(self) -> None:
    # Creates Concept, ImageAsset, Directory, BrokenLink node tables
    # Creates CONTAINS, LINKS_TO, INCLUDES_ASSET rel tables
    # Creates concept_embedding, concept_fts, image_omni_idx indexes
```

---

## 4.3. Embedding Engine

Text embeddings come from the external **`embroider`** crate
(github.com/opticsWolf/embroider — moved out of this tree in 0.2.12 after
living here as `rust/okf-embed/`). okfgraph holds no embedding numerics:
the Python side (`okfgraph/components/embedding.py`) owns dylib
resolution, the lazy session lifecycle, and path wiring; the crate owns the
Jina v5 contract. Class names and semantics are identical on both sides of
the move — the golden parity tests pin explicit-path ≡ hub-path vectors
byte-for-byte.

### Text Model (embroider — lazily opened)

The Jina v5 contract, enforced in Rust:

```
prefix (Query:/Document:) → tokenize @8192 → last-token pooling →
L2 normalise → Matryoshka truncate (32–1024, default 512) → re-normalise
```

**Last-token pooling (NOT mean)** — required by Jina v5; mean pooling
puts text vectors in a different space than the omni image vectors,
silently breaking the unified index. This was the v4.0 pooling fix and
remains a frozen vector-space invariant: no consumer may change pooling,
prefixes, or truncation order without re-indexing every database.

**Lazy sessions.** `LazyRustEncoder` opens the ONNX session on first
encode, not at router construction — importing a bundle, running doctor,
or opening a DB never pays for a session it doesn't use. `JinaTokenizer`
is a separate session-free handle for exact token counts (used for
token-budgeted reads).

**Explicit local files (air-gapped).** `OKFRouter(model_path=,
tokenizer_path=)` pins both files; they must be given together, must
exist, and the crate performs zero network access from those paths
(the sidecar layout lives next to the ONNX file). Default acquisition
(HF hub into the model cache) and explicit paths produce identical
vectors — pinned by test.

### Matryoshka Truncation

Native dim is 1024; `ALLOWED_DIMS = (32, 64, 128, 256, 512, 768, 1024)`
with 512 the default. Truncation keeps the first N of the L2-normalised
1024-vector and re-normalises — cheap, deterministic, CPU-only-safe.
Opening an existing DB adopts its on-disk dimension (§4.1); mixing
tuning levels inside one index is forbidden (ORT optimisation levels
fuse differently at the 1e-8 level — documented in the embroider README
benchmark table).

### Omni Model (SentenceTransformer — lazy-loaded)

```python
def _get_omni(self):
    """Load the omni model on first use (vision + text towers only)."""
    if self._omni is None:
        from sentence_transformers import SentenceTransformer
        self._omni = SentenceTransformer(
            self.omni_model_id,
            trust_remote_code=True,
            cache_folder=self.cache_dir,
            device=self.device,
            model_kwargs={"modality": "vision"},  # skip the audio tower
        )
    return self._omni

def _encode_image(self, data: bytes) -> List[float]:
    """Embed raw image bytes with the omni model (shared vector space)."""
    from io import BytesIO
    from PIL import Image
    img = Image.open(BytesIO(data))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    model = self._get_omni()
    vec = model.encode(img, truncate_dim=self.embedding_dim,
                       normalize_embeddings=True,
                       show_progress_bar=False)
    return self._truncate_normalize([float(x) for x in list(vec)])

def _encode_omni_text(self, text: str, task: str = "Query") -> List[float]:
    """Embed text with the omni model's text side (for cross-modal queries)."""
    model = self._get_omni()
    encoder = model.encode_query if task == "Query" else model.encode_document
    vec = encoder(text, truncate_dim=self.embedding_dim)
    return self._truncate_normalize([float(x) for x in list(vec)])
```

The omni model is **lazy-loaded** on first actual use — text-only pipelines pay none of its cost.

### Batch Encoding Algorithm

`_encode_batch(texts, task)` encodes multiple texts sequentially — **not** in a single padded ONNX call. With variable-length texts (80–300 words), padding all inputs to the longest in the batch causes **O(batch × max_len²)** attention compute waste. Sequential single-pass encoding processes each text at its actual token count, yielding **O(Σ len²)** total compute — strictly less than padded batching.

| Approach | Compute | Benchmark (100 concepts) |
|---|---|---|
| Padded batch (padding to longest) | O(batch × max_len²) | ~140s |
| Sequential single-pass | O(Σ len²) | **~128s** (1.1x faster) |

The speedup is modest (1.1x) because the ONNX forward pass is already efficient. The real batch speedup comes from **DB-level optimizations** (single transaction, bulk directory/link building), not from ONNX batching. Sequential encoding is the correct choice for variable-length documents.

---

## 4.4. Image Ingestion Modes

Three modes control how images become vectors in the unified `ImageAsset` index:

| Mode | Image **with** alt-text | Image **without** alt-text |
|---|---|---|
| `text` | text-embed(alt-text) | text-embed(`filename + image-number`) |
| `optional` | text-embed(alt-text) | **omni**-embed(image bytes) |
| `omni` | **omni**-embed(image bytes) | **omni**-embed(image bytes) |

Both encoders write into **one** vector space / index (`image_omni_idx` on `ImageAsset.embedding`). This works because jina-embeddings-v5 text-small-retrieval and omni-small-retrieval share a vector space — you can index with text and query with image (or vice-versa) without reindexing.

When `omni` is requested but the raw bytes are unavailable (e.g. an `http(s)://` URL, which is not fetched unless `--allow-remote-images` is set, or a missing local file), the plan **degrades gracefully** to the text path (alt-text, else filename fallback) so ingestion never hard-fails.

### Content Hash Change Detection

Each image asset stores a `content_hash` (SHA-256 of route + payload). On re-import, unchanged images are **not** re-embedded — critical for the costly omni path. Images removed from a document are pruned.

---

## 4.5. Import from OKF

```python
def import_from_okf(self, file_path: Path, mode: str | IngestMode = IngestMode.TEXT) -> str:
    """Parse OKF .md file and create/update concept in the graph.

    Args:
        file_path: Path to the ``.md`` file.
        mode: Image ingestion mode — ``text`` (alt-text / filename fallback,
            no omni model), ``optional`` (omni only for images without
            alt-text), or ``omni`` (omni for every image).

    Returns the concept ID (relative path without .md extension).
    """
```

### Batch Import Pipeline

`import_bundle()` uses a 4-phase pipeline with batched DB operations:

```python
def import_bundle(self, bundle_path: Optional[Path] = None, batch_size: int = 32,
                  mode: str | IngestMode = IngestMode.TEXT,
                  purge_deleted: bool = False) -> List[str]:
    """Walk bundle, parse all files, encode in batches, upsert in bulk.

    Phase 0: Delta detection — skip unchanged files (SHA-256 hash comparison)
    Phase 1: Parse all .md files (frontmatter + body)
    Phase 2: Batch encode search texts via _encode_batch()
    Phase 3: Batch upsert all concepts in single transaction
    Phase 4: Batch build directory hierarchy
    Phase 5: Batch extract cross-links
    Phase 6: Image ingestion (per concept, honouring the selected mode)

    purge_deleted: If True, concepts whose source files were deleted from
        disk are removed from the graph (including chunks, links, and
        orphaned image assets). Orphan check preserves shared assets.
    """
```

**`_batch_upsert_concepts()`**: Bulk deletes existing concepts, then creates all new ones with embeddings in a single transaction. Critically, `all_data.pop("embedding", None)` prevents embedding from leaking into the `extra` MAP column.

**`_batch_build_directories()`**: Collects all unique directory paths from concept IDs, sorts shallowest-first, creates directory nodes and CONTAINS relationships in order.

**`_batch_extract_links()`**: Collects all markdown links, batch-checks target existence, creates LINKS_TO or BrokenLink records in bulk.

### Delta Detection (v5.1)

On each `import_bundle()` call, file-level SHA-256 hashes are compared against the `FileHash` node table:

| Method | Role |
|---|---|
| `_file_hash(path)` | Compute SHA-256 hex digest of a file |
| `_store_file_hashes(paths, concept_ids)` | Upsert `FileHash` entries with `path`, `hash`, `concept_id` |
| `_load_file_hashes()` | Return `{path: hash}` dict from DB |
| `_load_file_hash_concept_ids()` | Return `{path: concept_id}` dict from DB |
| `_changed_files(source_files)` | Return `(changed_paths, deleted_paths)` tuple |

- **Unchanged files** are skipped entirely (no parsing, encoding, or DB writes).
- **Deleted files** appear in the `deleted_paths` list. If `purge_deleted=True`, their concepts are removed via `_purge_concept()`.
- **`concept_id` column** in `FileHash` enables mapping deleted file paths back to the concepts to purge.

### Purge (v5.1)

`_purge_concept(concept_id)` safely deletes a concept and all its dependents:

1. Collect `ImageAsset` IDs referenced by the concept
2. `DETACH DELETE` all `Chunk` nodes (`parent_doc_id = concept_id`)
3. `DETACH DELETE` the `Concept` node (cascades `LINKS_TO`, `CONTAINS`, `INCLUDES_ASSET`)
4. For each collected asset: if `ref_count == 0` → `DELETE` (**orphan check** — shared assets survive)
5. `DELETE` `BrokenLink` entries where concept was source or target
6. `DELETE` `FileHash` entry for this concept

Returns `True` if a concept was found and purged, `False` if not found. All operations are wrapped in a transaction with rollback on failure.

---

## 4.6. Export to OKF (v5.0 — Graph-Enriched)

```python
def export_to_okf(self, concept_id: str, output_path: Path) -> None:
    """Export a concept back to an OKF .md file.
    
    Body is enriched with graph-derived LINKS_TO links:
    - "See Also" section for outgoing links not already in body
    - "Cited By" section for incoming links
    """
    concept = self.get_by_id(concept_id)
    self._write_okf(concept, output_path)

def export_bundle(self, output_dir: Path,
                  directory_id: Optional[str] = None,
                  concept_type: Optional[str] = None,
                  tags: Optional[List[str]] = None) -> List[str]:
    """Export concepts from graph to OKF markdown files.

    Filters: directory_id (subtree), concept_type, tags (AND logic).
    Reconstructs directory hierarchy from concept IDs.
    Generates index.md files for progressive disclosure.
    """
```

### Graph Enrichment

Exported bodies are enriched with `LINKS_TO` relationships:

| Section | Source | Behavior |
|---|---|---|
| **See Also** | Outgoing `LINKS_TO` edges | Appended if target not already linked in body |
| **Cited By** | Incoming `LINKS_TO` edges | Appended if any sources exist |

This ensures exported bundles are **graphs** (linked documents), not just **trees** (files in directories).

### Progressive Disclosure

`index.md` files are auto-generated for every directory in the bundle, listing children (concepts and subdirectories) sorted by title.

---

## 4.7. Hybrid Search (RRF Fusion)

```python
def search_hybrid(
    self,
    query: str,
    concept_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    parent_id: Optional[str] = None,
    exclude_reserved: bool = True,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Hybrid search: RRF fusion of vector + FTS with optional graph filters."""
```

### Rank modes (`--rank` / `rank=`)

| Mode | Behaviour |
|---|---|
| `none` (default) | RRF-fused vector + FTS scores, graph filters applied |
| `hub` | Chunk/concept hits reranked by parent hub score (authoritative docs surface) |
| `ppr` | **Model-free Personalized PageRank** over the graph from the seed hits |

PPR needs no embedding model at all — the seed set comes from the cheap
FTS/vector pass (or an explicit seed), and ranking is pure graph
propagation. It is the retrieval mode that keeps working when no ONNX
session exists (air-gapped boxes without a model cache, doctor-style
triage). Measured details live in `docs/plan-retrieval-roundup.md`;
pinned end-to-end expectations (10 docs, 10 queries over the hybrid +
chunk + hub + PPR paths) live in `tests/test_retrieval_conformance.py`
over `tests/fixtures/retrieval_bundle/`.

---

## 4.8. Image Search (Unified Index)

```python
def search_images_with_text(
    self,
    text_query: str,
    use_text_model: bool = True,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Find image assets from a text query via the unified vector index.

    use_text_model=True (default) encodes the query with the lightweight
    text model — no omni load required, since both models share the vector
    space. Set it to False to route the query through the omni text side.
    """

def list_images(self, concept_id: str) -> List[Dict[str, Any]]:
    """List the image assets attached to a concept (no BLOB payloads)."""

def get_image_data(self, asset_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single image asset including its raw BLOB bytes."""
```

---

## 4.9. Graph Traversal

```python
def traverse(
    self,
    start_id: str,
    relationship: str = "CONTAINS",
    direction: str = "OUTGOING",
    depth: int = 1,
    node_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Navigate graph relationships with whitelisted edges and depth cap."""
```

---

## 4.10. Directory Listing

```python
def list_directory(self, directory_id: str) -> List[Dict[str, Any]]:
    """List immediate children of a directory (polymorphic: Directories + Concepts)."""
```

---

## 4.11. Direct Lookup

```python
def get_by_id(self, concept_id: str) -> Optional[ConceptModel]:
    """Fetch a full concept by ID, merging extra MAP fields back into the model."""
```

---

## 4.12. Broken Link Tracking

```python
def list_broken_links(self) -> List[Dict[str, Any]]:
    """List all tracked broken links (references to concepts not yet imported)."""

def repair_links(self) -> int:
    """Attempt to repair broken links by re-checking if targets now exist.

    Returns: Number of links successfully repaired.
    """
```

**Schema**: `BrokenLink` node table with `id` (STRING PRIMARY KEY), `source_id` (STRING), `target_id` (STRING), and `timestamp` (TIMESTAMP).

**Behavior**: During `_insert_concept()`, when a markdown link references a concept not yet in the graph, instead of silently skipping, a `BrokenLink` record is created. After subsequent imports, `repair_links()` can be called to resolve those orphans.

---

## 4.13. Reserved File Filtering

The `exclude_reserved` flag in `search_hybrid()` filters out concepts whose IDs end with `index` or `log`:

```python
def search_hybrid(self, query: str, ..., exclude_reserved: bool = True) -> List[Dict[str, Any]]:
    # Query applies: NOT c.id ENDS WITH 'index' AND NOT c.id ENDS WITH 'log'
```

**Default**: `exclude_reserved=True` (index/log files excluded from search results).

---

## 4.14. Cache Management

```python
@staticmethod
def default_cache_dir() -> str:
    """Return the HuggingFace default cache directory."""

@classmethod
def model_info(cls, model_id: str = "...", cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """Inspect model cache status without loading the model.

    Returns a dict with cache location, snapshot path, and disk usage.
    """
```

---

## 4.15. Index Lifecycle

Ladybug's vector and FTS indexes are built over a table's **current** contents; rows inserted after an index is created are not returned by search until the index is rebuilt. The system uses a **change-driven rebuild** strategy with two epoch counters stored in the `Meta` node table:

| Key | Meaning |
|---|---|
| `write_epoch` | Bumped on every index-affecting write (concept/image upsert or delete) |
| `indexed_epoch` | Records the `write_epoch` value at the last successful rebuild |

When `write_epoch > indexed_epoch`, the indexes are **dirty** and need rebuilding. Import paths call `_build_search_indexes(rebuild=True)` after data is written; the function checks `_indexes_dirty()` and skips the work if the counters match — so a no-op import or redundant call costs nothing.

### Rebuild Modes

| Mode | Trigger | Behavior |
|---|---|---|
| **Construction** (`rebuild=False`) | `OKFRouter.__init__()` | Creates missing indexes only; never drops, never checks dirty, never stamps. Merely opening a DB should not trigger an O(N) rebuild. |
| **Change-driven** (`rebuild=True, force=False`) | Import paths after data commit | Rebuilds only if `write_epoch > indexed_epoch`. Stamps `indexed_epoch = write_epoch` on success. |
| **Force** (`rebuild=True, force=True`) | `reindex()` CLI command, crash recovery | Rebuilds regardless of dirty state. Used to repair a DB written by an older build whose markers don't exist. |

### Ladybug Index Quirks

- **DROP INDEX leaves stale internal state** — prevents recreation with the same name. The system relies on `CREATE ... IF NOT EXISTS` semantics (Ladybug silently skips if present) instead of drop-and-recreate.
- **Index rebuild is O(N)** — proportional to the number of rows in the table. Change-driven rebuilds amortize this cost by only rebuilding when data actually changed.

### CLI Exposure

The `okf reindex [--if-dirty]` command invokes `reindex(force=True)` to rebuild all indexes. The `--if-dirty` flag would invoke `reindex(force=False)` to only rebuild when needed (not yet implemented — deferred to Gap #2 Option B).

### 4a.1. Chunk Model

```python
class ChunkModel(BaseModel):
    """A semantic chunk of a document, stored with its own vector embedding."""
    id: str
    parent_doc_id: str
    chunk_index: int
    chunk_text: str
    block_type: str              # "paragraph", "heading", "code", "list", "blockquote", "table", "diagram"
    start_offset: int = 0
    end_offset: int = 0
    embedding: Optional[List[float]] = None

    model_config = {"extra": "allow"}
```

### 4a.2. Chunking Pipeline

Documents are chunked during import using **Mordant** (Rust-based Markdown parser):

1. **Parse** — `MarkdownChunker` splits the document into semantic blocks (headings, paragraphs, code blocks, lists, tables, blockquotes, diagrams).
2. **Heading context injection** — Paragraph chunks track `current_heading` as an ephemeral key. This heading is prepended to the embedding payload without mutating the stored `chunk_text`.
3. **Structural boundaries** — The `STRUCTURAL_BLOCKS` tuple (`Heading`, `CodeBlock`, `List`, `Blockquote`, `Table`, `Diagram`) enforces hard semantic breaks. Overlap tails are cleared when hitting any structural block to prevent "chimera" vectors (e.g., code tokens bleeding into prose).
4. **Sliding window overlap** — Default `chunk_overlap=40` words. Tails are only generated from non-structural blocks.
5. **Encoding** — Each chunk is encoded via `_encode()` with `Document:` prefix, last-token pooling, L2 normalization, and Matryoshka truncation.
6. **Storage** — Chunks are stored as `Chunk` nodes with `PART_OF` relationships to the parent `Concept`.

### 4a.3. Chunk Search (RRF Fusion)

```python
def search_chunks(
    self,
    query: str,
    concept_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    parent_id: Optional[str] = None,
    limit: int = 10,
    max_chunks_per_doc: int = 3,
) -> List[Dict[str, Any]]:
    """RRF-fused chunk search: vector + FTS at chunk granularity.
    
    Returns list of dicts with keys:
    - chunk_id, chunk_text, block_type, chunk_index
    - parent_doc_id, parent_title, parent_type, parent_tags
    - rrf_score
    """
```

### 4a.4. Graph-Aware Reranking

```python
def search_with_context(
    self,
    query: str,
    limit: int = 10,
    context_hops: int = 1,
) -> List[Dict[str, Any]]:
    """Chunk search + graph neighborhood expansion.
    
    Returns chunks enriched with incoming_links, outgoing_links,
    directory ancestry, and sibling concepts.
    """

def search_chunks_with_hub_score(
    self,
    query: str,
    limit: int = 10,
    hub_weight: float = 0.5,
) -> List[Dict[str, Any]]:
    """Chunk search reranked by parent hub score.
    
    Hub score = incoming link count. Blended with RRF score:
    final_score = (1 - hub_weight) * rrf_score + hub_weight * normalized_hub
    """

def rerank_with_hub_score(
    self,
    chunk_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Adjust chunk scores by parent hub score."""

def expand_with_graph_context(
    self,
    chunk_ids: List[str],
) -> List[Dict[str, Any]]:
    """Discover related concepts via graph edges from chunk parents."""
```

### 4a.5. Document Reconstruction

```python
def reconstruct_document(self, document_id: str) -> Optional[str]:
    """Reconstruct original Markdown from chunks.
    
    Uses Mordant's `get_delimiter()` to join chunks with proper spacing.
    Returns None for nonexistent IDs.
    Achieves ~98% fidelity round-trip.
    """

def get_chunks(self, concept_id: str) -> List[ChunkModel]:
    """List all chunks for a concept, ordered by chunk_index."""
```

### 4a.6. Path Finding

```python
def find_path(
    self,
    start_id: str,
    end_id: str,
    max_length: int = 10,
) -> Optional[List[Dict[str, Any]]]:
    """BFS shortest path between two concepts.
    
    Returns list of nodes along the path, or None if no path found.
    Uses Ladybug variable-length patterns *1..N with MATCH path = ....
    """
```

### 4a.7. Hybrid Search with Chunks

```python
def search_hybrid(
    self,
    query: str,
    concept_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    parent_id: Optional[str] = None,
    exclude_reserved: bool = True,
    limit: int = 10,
    include_chunks: bool = False,  # NEW in v5.0
) -> List[Dict[str, Any]]:
    """Hybrid search with optional matched chunks attached.
    
    When include_chunks=True, each result dict includes a 'matched_chunks'
    key with top matching chunks for that concept.
    """
```

### 4a.8. Design Decisions

| Decision | Rationale |
|---|---|
| **Rust-side embedding contract** | Jina numerics (last-token pooling, truncation, normalisation) live in the `embroider` crate — Python holds no embedding math, so no vector-space drift is possible from this side |
| **Mordant `get_all_chunks()`** | Includes headings as separate chunks for ~98% reconstruction fidelity |
| **Storage vs embedding text decoupling** | `chunk_text` remains pristine; heading context injected in-memory for encoder |
| **Structural block boundaries** | Prevents chimera vectors (code/table tokens bleeding into prose) |
| **Default `chunk_overlap=40`** | Tighter semantic overlap without excessive redundancy (was 64) |
| **Class-scoped test fixtures** | ~80s run times vs 5-10x slower per-test model loading |
| **`chunk_id` in return dicts** | Explicit key name distinguishes chunk results from concept results |

---

## 5. CLI / App Layer

The `okf` CLI provides command-line and interactive access to all `OKFRouter` operations.

### 5.1. Entry Point

```toml
[project.scripts]
okf = "okfgraph.cli:main"
```

### 5.2. Commands

The CLI is a strict superset of the MCP surface (§6a): the five MCP tools
map 1:1 onto `search`, `read`, `traverse`, `ingest`, `export`, everything
else is library/maintenance surface.

| Command | Description |
|---|---|
| `okf init` | Initialize database and schema |
| `okf model-info` | Show model cache status (location, size, cached/missing) |
| `okf import <files>` | Import one or more OKF files |
| `okf import --all [--purge]` | Import entire bundle recursively, optionally purging deleted concepts |
| `okf search <query>` | Hybrid search over concepts (type/tags/parent/limit filters) |
| `okf search <query> --target chunks\|images` | Chunk-level RRF search, or image search via the unified index |
| `okf search <query> --rank hub\|ppr` | Rerank by hub score, or model-free PPR (§4.7) |
| `okf read <id> [--include body\|chunks\|document\|context]` | Fetch a concept, its chunks, the reconstructed document, or graph context |
| `okf traverse <id>` | Graph traversal (relationship/direction/depth); no id = root listing; two ids = shortest path |
| `okf ingest --kind md\|pdf\|thoughts <path>` | Add one piece of content (PDF goes through bobine, §10) |
| `okf export --all --output <dir>` | Export entire bundle (OKF or Obsidian flavor) |
| `okf diff` | Structural diff: concepts/edges/broken-link deltas |
| `okf doctor` | Health scan: score, findings, safe `--fix` |
| `okf lint` | Validate bundle frontmatter + links before import |
| `okf broken-links` / `okf repair-links` | List / repair links to not-yet-imported concepts |
| `okf reindex` | Rebuild vector + FTS search indexes |
| `okf deleted-list` / `deleted-recover` / `deleted-purge` | Soft-delete lifecycle |
| `okf shell` | Interactive REPL (own inline grammar, §5.5) |

### 5.3. Global Options

| Option | Default | Description |
|---|---|---|
| `--db <path>` | `okfgraph.db` | Database file path |
| `--bundle <path>` | `.` | Bundle root directory |
| `--dim <int>` | `512` | Embedding dimension (32-1024, official Matryoshka) |
| `--cache-dir <path>` | `~/.cache/huggingface` | HuggingFace model cache directory |
| `--device cpu\|cuda` | `cpu` | Inference device (or from okfgraph.toml) |
| `--omni-model-id <id>` | `jinaai/jina-embeddings-v5-omni-small-retrieval` | Multimodal model ID |

### 5.4. Import Options

| Option | Default | Description |
|---|---|---|
| `--mode <mode>` | `text` | Image ingestion mode: `text`, `optional`, `omni` |
| `--allow-remote-images` | — | Fetch `http(s)://` image URLs during ingestion (off by default) |
| `--batch-size <int>` | `32` | Batch size for encoding |
| `--purge` | — | Also purge concepts whose source files were deleted from disk (removes concept, chunks, links, and orphaned image assets) |

### 5.5. Interactive Shell

The `okf shell` command opens a REPL with its **own inline grammar** (not
the `okf <verb>` syntax) — `help` inside the shell lists it:

```
> search chunks:transformer efficiency
> search <query> expand      # chunk hits + graph neighborhood
> search <query> hub         # chunk hits reranked by hub score
> read <id> [chunks|document|context]
> traverse <id1> <id2>       # shortest path
> ingest notes.md --auto-import
> model-info
```

### 5.6. Design Decisions

- **Per-invocation router**: Each CLI command creates a fresh `OKFRouter` — schema must be idempotent.
- **`--all` vs `--bundle`**: Boolean flag for "import/export all" renamed from `--bundle` to avoid collision with global `--bundle <path>`.
- **ASCII icons**: `[D]`/`[F]` instead of emoji for Windows cp1252 compatibility.

---

## 6. LLM Tool Definitions (superseded)

The 16-tool surface this section used to specify (`tools.py`) is
superseded since the MCP migration: the agent surface is the 5-tool
registry in §6a (`search`, `read`, `traverse`, `ingest`,
`export_bundle`), and the CLI in §5 is its strict superset. The old
definitions were removed rather than kept as a second contract to
maintain — history lives in git.

## 6a. MCP Server (`okfgraph.mcp_server`)

The MCP server exposes all OKFgraph tools via the [Model Context Protocol](https://modelcontextprotocol.io/), allowing any MCP-compatible client (Claude Desktop, Cursor, Continue, etc.) to interact with the knowledge graph directly.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MCP Client                           │
│  (Claude Desktop, Cursor, Continue, etc.)               │
└──────────────────────┬──────────────────────────────────┘
                       │ stdio transport
                       ▼
┌─────────────────────────────────────────────────────────┐
│                 FastMCP Server                           │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │              Lifespan Context                     │   │
│  │                                                   │   │
│  │  @asynccontextmanager async def _lifespan(mcp):   │   │
│  │      router = OKFRouter(...)                      │   │
│  │      try:                                         │   │
│  │          yield GraphContext(router=router)        │   │
│  │      finally:                                     │   │
│  │          router.close()                           │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │              Tool Registry (5 tools)              │   │
│  │                                                   │   │
│  │  Read (3):  search, read, traverse               │   │
│  │                                                   │   │
│  │  Write (2): ingest (md|pdf|thoughts),            │   │
│  │              export_bundle                        │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Tool Annotations

All tools carry MCP-compliant annotations (`_RO` read-only / `_WR` write,
all non-destructive, idempotent):

| Tool | Kind | Surface |
|---|---|---|
| `search` | read | Concepts / chunks / images (`target`), filters, `rank` none\|hub\|ppr |
| `read` | read | Concept body / chunks / reconstructed document / graph context (`include`) |
| `traverse` | read | Relationships (CONTAINS/LINKS_TO/PART_OF/INCLUDES_ASSET), direction, depth; two ids = path; empty id = root listing |
| `ingest` | write | One piece of content, `kind` md\|pdf\|thoughts (PDF via bobine) |
| `export_bundle` | write | Whole bundle or filtered export (OKF/Obsidian flavor) |

### Context Injection

Each tool function receives the `OKFRouter` instance via the MCP `Context` parameter:

```python
@mcp.tool()
def search(query: str, ctx: Context) -> str:
    router = _get_router(ctx)  # extracts OKFRouter from lifespan context
    results = router.search_hybrid(query)
    return json.dumps(results, default=str, indent=2)
```

The `_get_router()` helper extracts the `GraphContext` from `ctx.request_context.lifespan_context`.

### CLI Entry Point

```bash
# Start MCP server with stdio transport
okf-mcp --db-path ./my_graph.db

# With GPU acceleration
okf-mcp --db-path ./my_graph.db --device cuda

# With custom embedding dimension
okf-mcp --db-path ./my_graph.db --embedding-dim 512

# Disable chunking
okf-mcp --db-path ./my_graph.db --no-chunking
```

> Embedding dim is **512 by default on every surface** (router, CLI
> `--dim`, `okf-mcp --embedding-dim`, `okfgraph.toml`). Any Matryoshka
> ladder value (32/64/128/256/512/768/1024) is accepted at creation; a
> server opened on an existing DB adopts the on-disk dimension per §4.1,
> so pass the flag explicitly only when creating.

### Programmatic Usage

```python
from okfgraph.mcp_server import create_mcp_server

mcp = create_mcp_server(
    db_path="./my_graph.db",
    bundle_root="./my-knowledge-base",
    device="cpu",
    embedding_dim=1024,
    enable_chunking=True,
)
mcp.run(transport="stdio")
```

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `--db-path` | **(required)** | Path to the Ladybug database file |
| `--bundle-root` | db parent | Root directory for the OKF bundle |
| `--device` | `cpu` | Device for ONNX inference (`cpu` or `cuda`) |
| `--embedding-dim` | `1024` | Dimension of the embedding vectors |
| `--no-chunking` | `False` | Disable document chunking |
| `--log-level` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## 7. OKF Optional Features

| Feature | Status | Notes |
|---|---|---|
| **Citations** | Handled | Markdown links (including citations) extracted as `LINKS_TO` if referencing internal `.md` files |
| **Log files** | Not handled | Can be added as separate concept type |
| **Index files** | On-the-fly | Generated by `list_directory`; no separate storage |
| **`okf_version`** | Read, not enforced | Parsed from root `index.md` frontmatter |
| **`okf-asset://` protocol** | Implemented | Markdown files use `![alt](okf-asset://img_uuid)` instead of inline Base64 |
| **Image ingestion modes** | Implemented | `text`, `optional`, `omni` with graceful fallback |
| **Content hash dedup** | Implemented | Unchanged images skipped on re-import |

---

## 8. Embedding Engine

### Text Model

| Feature | Detail |
|---|---|
| **Model** | `jinaai/jina-embeddings-v5-text-small-retrieval` |
| **Framework** | External `embroider` crate (Rust/ORT — no Python ML stack in core) |
| **Dimensions** | **1024** native (Matryoshka truncation: `ALLOWED_DIMS`, default **512**) |
| **Context Window** | 8192 tokens natively (tokenizer-only counts via `JinaTokenizer`, no session) |
| **Prefix Logic** | `Query:` for search queries, `Document:` for indexed content (frozen contract) |
| **Pooling** | **Last-token pooling** (NOT mean pooling — required by Jina v5) |
| **Normalization** | L2 normalization (cosine similarity), re-normalised after truncation |
| **Session Lifecycle** | Lazy — opens on first encode; router construction stays cheap |
| **Air-gapped** | `model_path` + `tokenizer_path` pin both files (zero network, fail-fast on missing) |
| **Acquisition** | HF hub into the model cache, or explicit paths — identical vectors, pinned by test |
| **GPU Support** | `--device cuda` (auto/cpu/cuda aliases) against the single pinned ORT; CUDA is opportunistic, CPU always works |

### Omni (Multimodal) Model

| Feature | Detail |
|---|---|
| **Model** | `jinaai/jina-embeddings-v5-omni-small-retrieval` |
| **Framework** | `sentence-transformers` (modality="vision" — skips audio tower) |
| **Dimensions** | **1024** (Matryoshka truncation: same as text model) |
| **Pooling** | Model-native (shared vector space with text model) |
| **Normalization** | `normalize_embeddings=True` (shared vector space) |
| **Lazy Loading** | Loaded on first actual use — text-only pipelines pay no cost |
| **Image Input** | Raw image bytes via `Pillow` |
| **Text Query** | `model.encode_query()` / `model.encode_document()` for cross-modal search |

### Unified Vector Space

Both encoders write into the **same** `ImageAsset.embedding` FLOAT[dim] column indexed by `image_omni_idx`. This means:

- Images embedded via alt-text (text model) can be queried by image
- Images embedded via omni model can be queried by text
- No reindexing needed when mixing embeddings from both models

---

## 9. images.py Module

The `okfgraph.images` module handles image extraction, resolution, and embedding planning. It is intentionally dependency-light — `Pillow` is imported lazily, only when raw image bytes actually need to be decoded. This keeps the mode-routing / extraction logic importable and unit-testable without the heavy embedding stack.

### Key Types

| Type | Description |
|---|---|
| `IngestMode` | `TEXT` / `OPTIONAL` / `OMNI` — how images become vectors |
| `EmbedRoute` | `TEXT` / `OMNI` — which encoder produces the embedding |
| `ExtractedImage` | Dataclass: concept_id, index, src, alt_text, filename, mime_type, data, asset_id |

### Key Functions

| Function | Description |
|---|---|
| `extract_image_refs(body)` | Return `(alt_text, src)` pairs for every markdown image, in document order |
| `asset_id_for(concept_id, index, src)` | Deterministic asset id (re-uses `okf-asset://` ids, generates UUID5 otherwise) |
| `build_extracted_images(concept_id, body, search_dirs, allow_remote)` | Extract every image ref and resolve bytes/metadata |
| `plan_embedding(img, mode)` | Decide how a single image should be embedded — returns `(route, caption)` |
| `sniff_mime(data, filename)` | Best-effort MIME detection: magic bytes first, then filename extension |
| `fallback_caption(filename, index, concept_id)` | Caption for text path when image has no alt-text |
| `load_image_bytes(src, search_dirs, allow_remote)` | Resolve raw image bytes for a markdown src (data URIs, local files, optional remote) |
| `filename_from_src(src)` | Derive a human-ish filename from a markdown image target |

### MIME Sniffing

Detects image types from magic bytes: PNG, JPEG, GIF, BMP, TIFF, WEBP, AVIF, HEIC.

### `okf-asset://` Protocol

Markdown files avoid Base64 bloat by using custom URI links:

```markdown
![chart](okf-asset://img_6f6b6661-0000-0000-0000-6f6b66617373)
```

The embedded UUID is deterministic and stable across round-trips.

---

## 10. PDF Conversion — the bobine seam (`okfgraph/components/converters.py`)

okfgraph holds **no converter code**. PDF/Office/text → Markdown conversion
is delegated to the external **`bobine`** engine (github.com/opticsWolf/bobine)
behind the `DocumentConverter` plugin seam; the default `BobineConverter`
ships in the `pdf` extra. The old in-tree ONNX/Rapid stack (`okfgraph.ingest`,
pdf_oxide + RapidAI models) was deleted — its pipeline (fast text-layer
path, TexTeller formula OCR, DocLayout-YOLO regions, RapidOCR text,
SLANet tables, per-slot CUDA/CPU providers) now lives in bobine and is
specified in *its* `docs/architecture.md`, not here.

### The seam

```python
class DocumentConverter(Protocol):
    def convert(self, path, work_dir) -> str: ...   # markdown out
```

Provider owns its options (`routing_mode`, model cache, provider lists);
okfgraph owns orchestration (kind dispatch, staging, lint, import).

### Ingest kinds (explicit, no auto-detect)

| Kind | Path |
|---|---|
| `md` | Read in Python, mordant lint, chunk, embed, upsert |
| `pdf` | `convert()` via the configured converter (default bobine), then the `md` pipeline |
| `thoughts` | Persist LLM reasoning as a searchable concept |

Kinds are explicit on both surfaces (CLI `--kind`, MCP `ingest(kind=)`) —
there is no format sniffing at the okfgraph layer. (bobine itself
content-sniffs its text fallback since v0.5.11: binary files under
text-ish extensions fail fast instead of returning decode garbage.)

### Routing modes

`NEVER / AUTO / SURGICAL / ALWAYS` are **bobine** `ConverterConfig`
concepts (born-digital fast path → full ONNX pipeline per page):
`SURGICAL` is the default — formula crops via TexTeller, full pipeline
only for scans. A missing converter (no `pdf` extra) fails fast with a
clear error; there is no legacy fallback.

## 11. Summary of Changes (v2.2 → v4.0)

### Major Additions

| Area | v2.2 | v4.0 | Reason |
|---|---|---|---|
| **Image ingestion** | Not specified | **Three modes** (`text`/`optional`/`omni`) | Multimodal knowledge graph |
| **ImageAsset node** | Not specified | **Full node table** with BLOB data + embedding | Binary storage of images |
| **INCLUDES_ASSET rel** | Not specified | **Concept → ImageAsset** | Graph edges to images |
| **Unified vector index** | `concept_embedding` only | **`image_omni_idx` on ImageAsset** | Shared text+image vector space |
| **Omni model** | Not specified | **SentenceTransformer, lazy-loaded** | Vision tower for image embeddings |
| **Content hash** | Not specified | **SHA-256 of route+payload** | Change detection, skip re-embedding |
| **Image search** | Not specified | **`search_images_with_text()`** | Text→image via unified index |
| **`okf-asset://` protocol** | Not specified | **UUID-based image references** | Avoid Base64 bloat in markdown |
| **`--mode` CLI flag** | Not specified | **`text`/`optional`/`omni`** | User-selectable image ingestion |
| **`--allow-remote-images`** | Not specified | **Fetch http(s) URLs** | Optional remote image support |
| **LLM tool: search_images** | Not specified | **New tool definition** | Agent-accessible image search |
| **Graceful fallback** | Not specified | **omni → text when bytes missing** | Ingestion never hard-fails |

### Verified Corrections

| Area | v2.2 (Spec) | v4.0 (Verified) | Reason |
|---|---|---|---|
| **Default embedding dimension** | 384 | **512** | 384 is not an official Matryoshka dimension; 512 is |
| **Pooling method** | Mean pooling | **Last-token pooling** | Jina v5 uses last-token; mean breaks unified space |
| **Pooling + truncation** | Mean → truncate | **Last-token → truncate → L2 re-normalise** | Ensures unit-norm truncated vectors |
| **Non-Matryoshka warning** | Not specified | **Warning emitted** | Guides users toward valid dimensions |
| **Schema idempotency** | try/except on index | **Unchanged** | Still needed per CLI invocation pattern |

### Unchanged (Verified Correct)

- Pydantic model structure and validators
- Hybrid search RRF fusion logic
- Directory hierarchy construction
- Cross-link extraction regex
- Export round-trip (export → re-parse)
- Batch encoding optimization (sequential single-pass)
- Batch DB upsert pipeline (3-phase)
- Broken link tracking + repair
- Reserved file filtering
- Label matching patterns
- MAP construction patterns
- Vector upsert (delete-then-create)
- FTS/vector index syntax
- Model cache management
- CLI / app layer design decisions

### Scoped Out (Clean Follow-ups)

| Feature | Reason |
|---|---|
| **Concept temporal dual-tracking** (`created_date`/`modified_date`) | Single-`timestamp` behaviour unchanged; ImageAsset uses `content_hash`-based change detection |
| **`okf-asset://` link rewriting on ingest** | Concept bodies stored verbatim (markdown round-trips); asset ids are deterministic, so rewriting can be added later without data migration |

---

## 10a. Summary of Changes (v4.0 → v5.0)

### Major Additions

| Area | v4.0 | v5.0 | Reason |
|---|---|---|---|
| **Chunk node** | Not specified | **Full node table** with embedding | Chunk-level search and retrieval |
| **PART_OF rel** | Not specified | **Concept → Chunk** | Graph edges to chunks |
| **Chunk vector index** | Not specified | **`chunk_embedding` on Chunk** | ANN search at chunk granularity |
| **Chunk FTS index** | Not specified | **`chunk_fts` on Chunk** | Keyword search at chunk granularity |
| **Mordant chunker** | Not specified | **Rust-based Markdown parser** | Semantic block splitting with heading awareness |
| **Heading context injection** | Not specified | **Ephemeral heading prepended to embedding payloads** | Enriches vectors without mutating storage |
| **Structural block boundaries** | Not specified | **`STRUCTURAL_BLOCKS` tuple** | Prevents chimera vectors (code/table tokens bleeding into prose) |
| **Chunk search (RRF)** | Not specified | **`search_chunks()`** | Vector + FTS fusion at chunk level |
| **Graph context expansion** | Not specified | **`search_with_context()`** | Enriches search results with neighborhood info |
| **Hub-score reranking** | Not specified | **`search_chunks_with_hub_score()`** | Chunks from authoritative docs rank higher |
| **Document reconstruction** | Not specified | **`reconstruct_document()`** | ~98% fidelity round-trip from chunks |
| **Path finding** | Not specified | **`find_path()`** | BFS shortest path between concepts |
| **Hybrid search with chunks** | `include_chunks` absent | **`include_chunks=True`** | Attaches matched chunks to concept results |
| **Numpy-only post-processing** | `torch` dependency | **`numpy` exclusively** | Removes heavy torch dependency |
| **CLI chunking commands** | Not specified | **search-chunks, context, hub-search, path, siblings, ancestry, chunks, reconstruct** | Full CLI coverage |
| **LLM tools** | 5 tools | **13 tools** | Agent-accessible chunking, graph enrichment, and export |
| **Export graph enrichment** | Body written verbatim | **See Also + Cited By sections** | Exported bundles reflect LINKS_TO graph |
| **Index file generation** | Not specified | **Auto-generated index.md files** | Progressive disclosure for OKF consumers |
| **ONNX/Rapid ingestion engine** | PaddleOCR/PaddlePaddle stack | **`okfgraph.ingest` sub-module** | Paddle-free PDF→Markdown via RapidAI ONNX models |
| **Surgical formula pass (ONNX)** | PP-FormulaNet (Paddle) | **RapidLaTeXOCR (ONNX)** | Formula recognition without CUDA-version coupling |
| **Scanned-page fallback (ONNX)** | PP-StructureV3 (Paddle) | **RapidLayout + RapidOCR + RapidTable** | Full layout-driven ONNX assembler for scanned PDFs |
| **HTML → GFM table converter** | Not in core | **`okfgraph.ingest.tables`** | Dependency-free pipe-table converter with rowspan/colspan bail |
| **okf-asset:// staging (ingest)** | Scattered across examples | **`okfgraph.ingest.assets`** | Deterministic asset ids, centralized staging logic |

### Verified Corrections

| Area | v4.0 (Spec) | v5.0 (Verified) | Reason |
|---|---|---|---|
| **`search_chunks` return key** | `"id"` | **`"chunk_id"`** | Distinguishes chunk results from concept results |
| **`reconstruct_document` nonexistent** | Returns `""` | **Returns `None`** | Pythonic sentinel for missing data |
| **Default `chunk_overlap`** | 64 | **40** | Tighter semantic overlap without excessive redundancy |
| **Ladybug `$end` reserved** | Would fail | **`$eid`** | `$end` is a reserved parameter name in Ladybug |
| **Ladybug `WITH` clauses** | Missing aliases | **Explicit `AS` aliases** | Ladybug requires all `WITH` expressions aliased |
| **Ladybug `id(node)`** | Used in `ORDER BY` | **Removed** | Unsupported in Ladybug for ordering |
| **Ladybug parameter stripping** | Silent drops | **`to_json($param)` wrapping** | Pybind backend strips non-JSON/non-wrapped parameters |

### Unchanged (Verified Correct)

- All v4.0 features (image ingestion, unified vector space, hybrid search)
- Pydantic model structure and validators
- ONNX embedding pipeline (last-token pooling, Matryoshka truncation)
- Ladybug schema patterns (vector upsert, MAP construction, label matching)
- CLI per-invocation router pattern
- Batch encoding optimization (sequential single-pass)

### Scoped Out (Clean Follow-ups)

| Feature | Reason |
|---|---|
| **Function-scoped test fixtures** | Class-scoped for ~80s run times; revert at full-test maturity |
| **`--skip-embedding` flag** | Faster imports without ONNX encoding; low priority |
| **Mordant-specific unit tests** | Pure Mordant features; not OKF-specific logic |
| **ONNX/Rapid end-to-end PDF tests** | Require RapidAI packages + test PDFs; tracked in §10 testing checklist |
| **Office file conversion (office_oxide)** | Optional dependency; wired through `HybridConverter.convert_office()` |

---

## 10b. Summary of Changes (v5.0 → v5.1)

### Major Additions

| Area | v5.0 | v5.1 | Reason |
|---|---|---|---|
| **FileHash node** | Not specified | **Full node table** with `path`, `hash`, `concept_id` | Delta detection for incremental imports |
| **`concept_id` in FileHash** | Not tracked | **Maps file path → Concept.id** | Enables purge of deleted files |
| **Delta detection (Phase 0)** | Full re-import | **SHA-256 hash comparison** | Skips unchanged files (no parsing, encoding, or DB writes) |
| **`_changed_files()`** | Not specified | **Returns `(changed, deleted)` tuple** | Detects both modified and deleted files |
| **`_purge_concept()`** | Not specified | **Safe cascading delete** | Removes concept, chunks, links, orphaned assets |
| **Orphan asset check** | Not specified | **Shared assets survive purge** | Two files referencing same `okf-asset://<id>` → asset preserved |
| **`--purge` CLI flag** | Not specified | **`okf import --all --purge`** | User-controllable deletion of stale concepts |
| **`purge_deleted` parameter** | Not specified | **`import_bundle(purge_deleted=True)`** | Programmatic purge control |
| **BrokenLink cleanup on purge** | Not specified | **Auto-clean on concept removal** | No dangling broken links |
| **FileHash cleanup on purge** | Not specified | **Auto-clean on concept removal** | Consistent state between FileHash and Concept tables |
| **Test coverage** | 98 tests | **220 tests** (+21 delta/purge, +23 router misc, +2 converter, +9 search browser, +29 router smoke) |

### Documentation Additions

| Area | v5.1 | v5.2 | Reason |
|---|---|---|---|
| **§4.1 Auto-Detect Embedding Dimension** | Not documented | **Documented** | Explains `_adopt_existing_embedding_dim()` |
| **§4.3 Batch Encoding Algorithm** | Not documented | **Documented** | Sequential vs padded compute analysis |
| **§4.15 Index Lifecycle** | Not documented | **Documented** | Epoch counters, dirty tracking, rebuild modes |
| **§6 LLM Tool: export_bundle** | Missing | **Added** | 13th tool definition |

### Error Handling (v5.2)

| Area | v5.1 | v5.2 | Reason |
|---|---|---|---|
| **Per-concept error isolation** | Implicit (images only) | **Explicit** — `_import_chunks_for_concept()` wrapped in try/except | One bad concept doesn't block the rest of the bundle |
| **Import failure reporting** | `print()` for images | **`logger.warning()`** with aggregate summary | Structured logging, failure counts |
| **Context-window warning** | Silent truncation | **`logging.warning()`** at 90% threshold | Makes silent behavior visible to users |

### Test Coverage (v5.2)

| Area | v5.1 | v5.2 | Reason |
|---|---|---|---|
| **`test_router_misc.py`** | Not present | **23 tests** | Covers reindex, repair_links, meta/epoch, adopt_dim, error isolation, context window |
| **`test_router.py` CUDA tests** | Assumed CPU-only | **Conditional** — verifies CUDA provider if available | Works on GPU machines |
| **`test_router.py` tools count** | 5 | **13** | Matches actual tool definitions |
| **`test_converter.py`** | Collection error | **2 passing** | Added missing PySide6 stubs |
| **`test_search_browser.py`** | Collection error | **9 passing** | Fixed path to `examples/okf_search_browser.py` |
| **Total** | 119 | **220** | Full suite, zero errors, zero warnings |

### Design Decisions

| Decision | Rationale |
|---|---|
| **File-level SHA-256** | Simple, deterministic, zero schema migration cost |
| **`concept_id` in FileHash** | Maps deleted file paths back to concepts for purge |
| **Orphan check for assets** | Shared assets (same `okf-asset://<id>`) survive if any Concept still references them |
| **Chunk deletion before Concept** | Ladybug's `DETACH DELETE` on Concept doesn't cascade to separate Chunk nodes |
| **Transactional purge** | All-or-nothing — rollback on any failure |
| **Default `purge_deleted=False`** | Safe default — user must opt-in to deletions |

---

## 10c. Summary of Changes (v5.2 → v5.3)

### Schema Versioning & Migration (Gap #8 — Option A)

| Area | v5.2 | v5.3 | Reason |
|---|---|---|---|
| **Schema version tracking** | Not present | **`schema_version` in Meta table** | Detects outdated DBs on startup |
| **`SCHEMA_VERSION` constant** | Not present | **= 3** | Current schema version |
| **Migration registry** | Not present | **`_MIGRATIONS` dict** | Maps version → migration function |
| **v1 → v2 migration** | Not present | **Chunk table + PART_OF + Chunk indexes** | Backports v5.0 features for old DBs |
| **v2 → v3 migration** | Not present | **FileHash table with concept_id** | Backports v5.1 features for old DBs |
| **`_run_schema_migrations()`** | Not present | **Auto-runs on startup** | Idempotent, version-stamped, error-reporting |
| **Fresh DB handling** | Implicit | **Version 0 → stamped to current** | No migrations needed (full schema created) |
| **Test coverage** | 130 tests | **+10 migration tests** | Version stamping, idempotency, partial migrations |

### PDF Ingestion → Import Integration (Gap #5 — Option A)

| Area | v5.2 | v5.3 | Reason |
|---|---|---|---|
| **`okf ingest` CLI command** | Not present | **New subcommand** | End-to-end PDF→graph in one command |
| **`--auto-import` flag** | Not present | **Temp dir + import + cleanup** | One-shot workflow |
| **`--routing-mode` flag** | Not present | **auto/surgical/always/never** | Controls ONNX heavy-pass routing |
| **`--mode` flag** | Not present | **text/optional/omni** | Image ingestion mode for auto-import |
| **`--batch-size` flag** | Not present | **Batch size for encoding** | Tunes import performance |
| **`--purge` flag** | Not present | **Purge deleted concepts** | Consistent with `okf import` |
| **`--no-extract-images` flag** | Not present | **Skip image extraction** | Faster for text-only PDFs |
| **Shell support** | Not present | **`ingest <pdf>` in REPL** | Interactive use |
| **Output-only mode** | Not present | **Writes .md + _assets/** | Two-step workflow (convert then import) |

### Design Decisions

| Decision | Rationale |
|---|---|
| **Schema version in Meta table** | Reuses existing key/value store; no new tables |
| **Fresh DB starts at version 0** | Distinguishes "never migrated" from "already at v1" |
| **Idempotent migrations** | Safe to re-run; Ladybug's `IF NOT EXISTS` handles duplicates |
| **Error reporting on migration failure** | Logs version, error, and stops — no silent corruption |
| **Temp dir for auto-import** | Clean resource management; no leftover files on failure |
| **Output-only mode** | Users can inspect converted markdown before importing |
| **`--routing-mode` exposed** | Lets users tune ONNX usage (NEVER for speed, ALWAYS for quality) |

---

## 12. Performance Baseline

**Benchmark**: `benchmarks/benchmark_500.py` — 100 synthetic concepts, in-memory DB.
*(Timings below are from the optimum-era stack; the DB-level insight
still holds — batch speedup comes from single-transaction / bulk
link-building, not from ONNX batching, and sequential per-text encoding
remains the policy in the embroider crate.)*

| Parameter | Value |
|---|---|
| Concepts | 100 |
| Document size | 240-600 words |
| Vocabulary | 558 unique words (8 categories) |
| Database | `:memory:` (isolates query/index from disk I/O) |
| Embedding dim | 512 (Matryoshka truncation, bumped from 384) |
| Batch size | 64 |

| Metric | Time | Per-Concept |
|---|---|
| Single import | ~140s | ~1400ms |
| Batch import | ~128s | ~1280ms |
| Hybrid search (5 reps) | 107ms mean | — |
| Export (100 concepts) | 77ms | 0.8ms |

**Batch vs single**: **1.1x faster** (batch wins).

**Key insight**: Padded batch tokenization (`padding=True`) causes O(batch × max_len²) attention waste with variable-length texts. Sequential single-pass encoding is optimal for variable-length documents. Batch speedup comes from DB-level optimizations (single transaction, bulk directory/link building), not from ONNX batching.

---

This specification is **verified against production LadybugDB v0.20.3**
(okfgraph 0.2.12 tree: `embroider 0.1.3`, `bobine 0.5.11`,
`onnxruntime==1.29.0`). All code patterns have been tested end-to-end
with real data, real model inference, and real database operations.

---

## 15. Closed Gaps & Current Constraints

The v5.x gap program (concurrency/filelock, path-traversal sandboxing,
model-cache verification, TOML schema validation, end-to-end PDF tests —
`docs/gap-analysis.md`) is **closed**; §15's old per-gap text is retired
with it. What remains are standing constraints, not gaps:

| Constraint | Status |
|---|---|
| **Ladybug three-clause MERGE** | Vector upserts must avoid `MERGE … SET` on indexed columns (runtime abort — see the quarantine at §2). Reported upstream; watch `macrame-db` 0.18 |
| **Single pinned ORT** | `onnxruntime==1.29.0` shared by bobine + embroider; a stale system DLL fails session creation with `BadVersion`. `ORT_DYLIB_PATH` overrides; entry points resolve before first use |
| **Frozen vector space** | Jina contract (prefixes, last-token pooling, truncation order) is identical across okfgraph 0.2.x and embroider 0.1.x — enforced by golden parity tests, never by convention alone |
| **Floor-pinned embroider** | `embroider>=0.1,<0.2`: a new embroider minor without an okfgraph release is a *supported* state, and the suite proves the floor still passes (contract fixtures in the embroider repo `fixtures/`, vendored at `tests/fixtures/golden_jina_v5_text_small.json` + `tests/test_golden_vectors.py`; matrix: embroider `COMPAT.md`) |
| **Schema v6** | `DeletedConcept` carries a full-fidelity snapshot (embeddings + metadata survive soft-delete → recover); v5 DBs migrate on open (ALTER existing table or CREATE it — 0.2.x fresh DBs never had it) |
| **Schema v8** | `FileHash.dir` / `DeletedPath.dir` store the exact parent DirHash key (multi-root namespacing makes path-string math ambiguous); v7 DBs migrate on open with legacy-math backfill (exact for all pre-0.4.0 rows) |

---

## 16. Multi-root bundles (0.4.0)

One graph, N live roots, no copies, no ID collisions (plan:
`docs/plan-multi-root-detach.md` Phase 2). The constructor `bundle_root`
stays the primary tree — its files keep **bare IDs** (empty-alias rule, no
migration for legacy graphs). `OKFRouter(roots={alias: path})` adds named
trees whose files mint **`@alias/rel`** IDs (`@` is filename-safe, never a
URI-scheme start; `:` was rejected: Windows-illegal on export and swallowed
by `SCHEME_RE` in wikilinks).

- **Identity**: derivation lives in `parse_source_file(..., alias)`; aliases
  match `^[A-Za-z0-9][A-Za-z0-9_-]*$`, never start with `@`; roots must not
  overlap (validated fail-fast, existence NOT required). Outside-root files
  keep the bare-stem fallback. Per-alias `Directory` nodes fall out of the
hierarchy builder (IDs split on `/`). Legacy trees with top-level `@*`
  entries warn (re-imported exports reproduce exact IDs).
- **Delta per root**: one `DeltaDetector` per namespace sharing the
  connection; FileHash/DirHash/DeletedPath keys are `@alias/`-prefixed
  (`roots.py:prefix_key`), concept IDs derive prefix-safe. Emptied-but-present
  roots fall through detection (absent trees return early — see liveness).
- **Liveness** (the phase invariant): import skips absent roots with a loud
  warning and never tombstones them; `--purge-deleted` refuses unless **all**
  roots are present (unknown must never read as deleted); reads/exports are
  unaffected; doctor reports per-root presence + counts (informational).
- **Links**: `[[@alias/rel]]` resolves via the exact-id probe for free;
  `[[alias/rel]]` rewrites through `roots.qualify_alias_link` (alias folds
  case, rest stays exact). Unqualified names keep uid→alias→title→stem order;
  cross-root collisions become repairable BrokenLinks. Relative `](path)`
  links stay source-unaware (deferred, as before).
- **Surfaces**: CLI `--bundle-root ALIAS=PATH` (repeatable, combines with
  `--bundle`); TOML `[[roots]]` (paths relative to the TOML file); MCP
  `--root ALIAS=PATH` + `create_mcp_server(roots=...)`. Path-based `ingest md`
  resolves longest-prefix-match; PDF work-dir imports mint a stable
  `@pdf-<content-hash12>/...` namespace (same-stem pages from different PDFs
  can no longer overwrite each other); thoughts IDs were already unique.
- **Export/diff/detach**: export writes `<out>/@alias/rel.md` (single-root
  re-import reproduces exact IDs); drift diff unions all present trees
  (absent ≠ drift); detach verifies + records one SourceRoot row per alias
  and `--force` re-attach requires the full configured root set to match.
- **Deviations from the plan draft**: `--bundle`/`--bundle-root` combine
  (no exclusion — none exists); schema went v7→v8 after all (Ladybug's binder
  rejects unknown properties on write, so the `dir` key needed a declared
  column); doctor roots omit `last_seen` (no per-root clock exists while
  attached).

*This specification is a living artifact. Update the version and sections as the tree changes — and mark superseded design docs historical instead of deleting them.*
