# OKF Knowledge Graph — Architecture Specification

**Version**: 5.5 (Directory-Level Hash Aggregation)  
**Based on**: Architecture v5.4 (Gap Analysis Consolidation)  
**Verified against**: LadybugDB v0.17.1, Python 3.13.14

**Gap Analysis Baseline**: [docs/gap-analysis.md](docs/gap-analysis.md) — 15 gaps reviewed, 14 closed, 1 open (v5.5).

**Storage**: LadybugDB (v0.17+) — graph + vector + full-text search.  
**Data Model**: Pydantic v2 with `extra='allow'` — preserves OKF extensibility, maps cleanly to Ladybug's `MAP` and `LIST` columns.  
**Embedding Engine**: ONNX-optimized Jina v5 text model (`jinaai/jina-embeddings-v5-text-small-retrieval`) via `optimum[onnxruntime]`. **Numpy-only post-processing** (no torch).  
**Multimodal Engine**: SentenceTransformer with `jinaai/jina-embeddings-v5-omni-small-retrieval` (vision tower, lazy-loaded).  
**Unified Vector Space**: Both encoders write into one `ImageAsset.embedding` column indexed by `image_omni_idx`.  
**Chunking**: Mordant (Rust-based Markdown parser) with heading context injection and structural block boundaries.  
**Search Modes**: Hybrid (RRF fusion), Traversal (pure graph), Direct (exact ID lookup), Image search (text→image via unified index), **Chunk-level search with graph enrichment**.

---

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
| **Closed gaps** | — | **#5 PDF→import, #6 error isolation, #8 schema migration, #12 missing tests, #13 index health, #14 chunk limits, #16 LLM tool coverage, #10 observability, #15 RapidAI pinning, #5b ingest_pdf** | Core reliability, feature completeness, observability |
| **Gap #7 (Concurrent Access)** | Not documented | **Documented** — WAL mode + single-writer constraint recommendation (OPEN) | Data integrity under concurrent use |
| **Gap #9 (Security)** | Not documented | **Documented** — SSRF risk, URL allowlist recommendation (OPEN) | Security architecture |
| **Version bump** | 5.3 | **5.4** | Gap analysis artifact |

---

## 1. Dependencies

### Core (required)

```bash
pip install optimum[onnxruntime] transformers numpy mordant python-frontmatter pyyaml pydantic ladybug sentence-transformers Pillow
```

> **No torch dependency**. ONNX Runtime returns numpy arrays; all post-processing (last-token pooling, L2 normalization, Matryoshka truncation) uses `numpy` exclusively.

**Installed versions** (verified working):
- `optimum==2.1.0`, `onnxruntime==1.27.0`, `numpy>=1.24`, `transformers==4.57.6`
- `ladybug==0.17.1`, `pydantic==2.13.4`, `mordant>=0.12`

### PDF Ingestion (optional — ONNX/Rapid stack)

```bash
pip install pdf_oxide pillow                          # fast path + rendering
pip install rapidocr                                  # text detection + recognition
pip install rapid_latex_ocr                           # formula image → LaTeX
pip install rapid_layout                              # layout region detection
pip install rapid_table                               # table structure → HTML
pip install onnxruntime-gpu                           # or onnxruntime-directml / onnxruntime
```

All RapidAI packages are **Apache-2.0** (commercial-friendly); LaTeX-OCR is **MIT**. Every RapidAI import is guarded — the ingestion sub-module loads cleanly without them installed. Models download lazily on first use; offline installs can vendor `.onnx` files into a `models/` directory.

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

LadybugDB uses specific syntax that differs from standard Cypher. All patterns below are **verified against Ladybug v0.17.1**.

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
        embedding_dim: int = 512,          # bumped from 384
        cache_dir: Optional[str] = None,
        device: str = "cuda",
        allow_remote_images: bool = False,
    ):
        # ... validates dim, loads text model, lazy-loads omni model
```

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

### Text Model (ONNX — always loaded)

```python
def _encode(self, text: str, task: str = "Document") -> List[float]:
    """Encode text with ONNX Jina v5 model.

    Uses **last-token pooling** (NOT mean pooling).
    Jina v5 was trained with last-token pooling; mean pooling
    produces vectors in a different space that will NOT align
    with the omni model's image embeddings.
    """
    # Apply prefix (Query:/Document:)
    # Tokenize → ONNX forward → last-token pooling → L2 normalize → truncate → L2 re-normalize
    return self._truncate_normalize(vec)
```

**Pooling fix (v4.0)**: The previous implementation used mean pooling, which puts text vectors in a different space than the omni image vectors — silently breaking the unified index. Fixed to use **last-token pooling** followed by L2 re-normalisation.

### Matryoshka Truncation

```python
def _truncate_normalize(self, vec: List[float]) -> List[float]:
    """Truncate to configured dim and L2-renormalise."""
    v = list(vec[:self.embedding_dim])
    if len(v) < self.embedding_dim:
        v = v + [0.0] * (self.embedding_dim - len(v))
    norm = math.sqrt(sum(x * x for x in v))
    if norm > 0:
        v = [x / norm for x in v]
    return v
```

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
| **Numpy-only post-processing** | ONNX model returns numpy arrays; torch adds no value for pooling/normalization |
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

| Command | Description |
|---|---|
| `okf init` | Initialize database and schema |
| `okf model-info` | Show model cache status (location, size, cached/missing) |
| `okf import <files>` | Import one or more OKF files |
| `okf import --all` | Import entire bundle recursively |
| `okf import --all --purge` | Import bundle and purge deleted concepts |
| `okf search <query>` | Hybrid search (type/tags/parent/limit filters) |
| `okf search <query> --chunks` | Hybrid search with matched chunks attached |
| `okf search-images <query>` | Find images via the unified vector index |
| `okf search-chunks <query>` | Chunk-level RRF-fused search |
| `okf context <query>` | Search with graph neighborhood expansion |
| `okf hub-search <query>` | Chunk search reranked by hub score |
| `okf path <id1> <id2>` | Find shortest path between concepts |
| `okf siblings <id>` | List sibling concepts in same directory |
| `okf ancestry <id>` | Show directory hierarchy for a concept |
| `okf chunks <id>` | List chunks for a concept |
| `okf reconstruct <id>` | Reconstruct document from chunks |
| `okf traverse <id>` | Graph traversal (relationship/direction/depth) |
| `okf list [dir]` | List directory contents |
| `okf get <id>` | Fetch full concept (JSON + body) |
| `okf export --all --output <dir>` | Export entire bundle |
| `okf export --concept-id <id> --output <dir>` | Export single concept |
| `okf broken-links` | List broken (orphan) links to not-yet-imported concepts |
| `okf repair-links` | Repair broken links by re-checking if targets now exist |
| `okf shell` | Interactive REPL |

### 5.3. Global Options

| Option | Default | Description |
|---|---|---|
| `--db <path>` | `okfgraph.db` | Database file path |
| `--bundle <path>` | `.` | Bundle root directory |
| `--dim <int>` | `512` | Embedding dimension (32-1024, official Matryoshka) |
| `--cache-dir <path>` | `~/.cache/huggingface` | HuggingFace model cache directory |
| `--device cpu\|cuda` | `cuda` | Inference device (auto-fallback to CPU if CUDA unavailable) |
| `--omni-model-id <id>` | `jinaai/jina-embeddings-v5-omni-small-retrieval` | Multimodal model ID |

### 5.4. Import Options

| Option | Default | Description |
|---|---|---|
| `--mode <mode>` | `text` | Image ingestion mode: `text`, `optional`, `omni` |
| `--allow-remote-images` | — | Fetch `http(s)://` image URLs during ingestion (off by default) |
| `--batch-size <int>` | `32` | Batch size for encoding |
| `--purge` | — | Also purge concepts whose source files were deleted from disk (removes concept, chunks, links, and orphaned image assets) |

### 5.5. Interactive Shell

The `okf shell` command opens a REPL with inline commands:

```
> import ./concepts/basics.md
> search advanced type:section
> search-images a cat
> images concepts/intro
> traverse concepts CONTAINS OUTGOING 2
> export-bundle ./output
```

### 5.6. Design Decisions

- **Per-invocation router**: Each CLI command creates a fresh `OKFRouter` — schema must be idempotent.
- **`--all` vs `--bundle`**: Boolean flag for "import/export all" renamed from `--bundle` to avoid collision with global `--bundle <path>`.
- **ASCII icons**: `[D]`/`[F]` instead of emoji for Windows cp1252 compatibility.

---

## 6. LLM Tool Definitions

```python
TOOLS = [
    {
        "name": "search_hybrid",
        "description": "Semantic + keyword search over concepts. Use for open-ended questions.",
        ...
    },
    {
        "name": "traverse",
        "description": "Navigate relationships (CONTAINS or LINKS_TO) from a concept.",
        ...
    },
    {
        "name": "get_by_id",
        "description": "Fetch the full markdown body of a specific concept.",
        ...
    },
    {
        "name": "list_directory",
        "description": "List contents of a directory for progressive disclosure.",
        ...
    },
    {
        "name": "search_images",
        "description": "Find image assets by a text description via the unified vector index. Works whether images were embedded from alt-text or by the multimodal model, since both share one vector space.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text describing the image(s) to find."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Maximum number of images to return (default 10)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_chunks",
        "description": "Search document chunks with RRF-fused vector + FTS. Returns chunk-level results with parent concept metadata.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                "max_chunks_per_doc": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "expand_with_graph_context",
        "description": "Discover related concepts via graph edges from chunk parents.",
        "parameters": {
            "type": "object",
            "properties": {
                "chunk_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["chunk_ids"],
        },
    },
    {
        "name": "rerank_with_hub_score",
        "description": "Adjust chunk scores by parent hub score (incoming link count).",
        "parameters": {
            "type": "object",
            "properties": {
                "chunk_results": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["chunk_results"],
        },
    },
    {
        "name": "get_chunks",
        "description": "List all chunks for a concept.",
        "parameters": {
            "type": "object",
            "properties": {
                "concept_id": {"type": "string"},
            },
            "required": ["concept_id"],
        },
    },
    {
        "name": "reconstruct_document",
        "description": "Reconstruct original Markdown from chunks (~98% fidelity).",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {"type": "string"},
            },
            "required": ["document_id"],
        },
    },
    {
        "name": "find_path",
        "description": "Find shortest path between two concepts in the knowledge graph.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_id": {"type": "string"},
                "end_id": {"type": "string"},
                "max_length": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["start_id", "end_id"],
        },
    },
    {
        "name": "export_bundle",
        "description": (
            "Export concepts from the graph to an OKF-compliant bundle directory. "
            "Each concept is written as a markdown file with YAML frontmatter. "
            "The body is enriched with graph-derived LINKS_TO links (See Also + Cited By). "
            "index.md files are generated for every directory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "output_dir": {
                    "type": "string",
                    "description": "Output directory for the bundle.",
                },
                "directory_id": {
                    "type": "string",
                    "description": "Optional: only export concepts under this directory.",
                },
                "concept_type": {
                    "type": "string",
                    "description": "Optional: only export concepts of this type.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: only export concepts with ALL these tags.",
                },
            },
            "required": ["output_dir"],
        },
    },
]
```

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
| **Framework** | ONNX Runtime via `optimum[onnxruntime]` |
| **Dimensions** | **1024** (Matryoshka truncation: configurable 32-1024, default **512**) |
| **Context Window** | 32,768 tokens (practically truncated to 8,192) |
| **Prefix Logic** | `Query:` for search queries, `Document:` for indexed content |
| **Pooling** | **Last-token pooling** (NOT mean pooling — required by Jina v5) |
| **Normalization** | L2 normalization (cosine similarity), re-normalised after truncation |
| **Cache Dir** | Optional `cache_dir` param on `OKFRouter.__init__()` |
| **Cache Inspection** | `OKFRouter.model_info()` — location, status, disk usage |
| **GPU Support** | `--device cuda` with auto-fallback to CPU (requires `onnxruntime-gpu`) |

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

## 10. ONNX/Rapid PDF Ingestion Engine (`okfgraph.ingest`)

The `okfgraph.ingest` sub-module provides a **Paddle-free** PDF → Markdown conversion pipeline using the RapidAI family of ONNX models. It replaces the PaddleOCR/PaddlePaddle stack entirely.

### Architecture

```
                          ┌─────────────────────────────┐
   PDF ──▶ pdf_oxide ────▶│ page has a usable text layer?│
                          └──────────────┬──────────────┘
                            yes │            │ no  (few chars + images = scanned)
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

### Sub-Module Structure

| File | Role |
|---|---|
| `config.py` | `ConverterConfig` dataclass + `RoutingMode` enum (NEVER/AUTO/SURGICAL/ALWAYS) |
| `engine.py` | `OnnxRapidEngine` — lazy loaders for RapidLaTeXOCR, RapidOCR, RapidLayout, RapidTable |
| `converter.py` | `HybridConverter` — core pipeline (pdf_oxide fast path + ONNX heavy passes) |
| `tables.py` | `_SimpleTableParser` + `html_tables_to_gfm()` — HTML → GFM pipe-table converter |
| `assets.py` | `stage_images_as_okf_assets()` — okf-asset:// staging for extracted images |

### Routing Modes

| Mode | Behaviour |
|---|---|
| **NEVER** | Fast path only. No ONNX models loaded. |
| **AUTO** | Heuristics per page → full ONNX pipeline only on flagged pages. |
| **SURGICAL** | Formula crops via RapidLaTeXOCR; full pipeline only for scans. |
| **ALWAYS** | Every page through the full ONNX layout + OCR pipeline. |

### Key Design Decisions

- **Zero hard dependencies** — all RapidAI imports are guarded; the module loads cleanly without them
- **Lazy loading** — born-digital PDFs never pay for OCR/layout/table models
- **Graceful degradation** — if a model fails to load, the pipeline falls back to the fast path
- **`# VERIFY` flags** — every version-sensitive RapidAI call is marked for confirmation
- **Device → ort_providers coercion** — `device="cuda"` auto-resolves to `["CUDAExecutionProvider", "CPUExecutionProvider"]` (accepts `"gpu"` as alias)
- **Inline vs display LaTeX** — `_latex_wrap()` decides based on box dimensions vs threshold
- **Output contract unchanged** — single `.md` with inline/display LaTeX, fenced code, GFM tables, and `okf-asset://` links

### Execution Providers

ONNX Runtime decouples from CUDA toolkit versions:

| Hardware | Package | Providers |
|---|---|---|
| NVIDIA (incl. RTX 50-series) | `onnxruntime-gpu` | `CUDAExecutionProvider`, `CPUExecutionProvider` |
| Windows DirectX 12 GPU | `onnxruntime-directml` | `DirectMLExecutionProvider`, `CPUExecutionProvider` |
| Apple Silicon | `onnxruntime` | `CoreMLExecutionProvider` (or CPU) |
| CPU-only | `onnxruntime` | `CPUExecutionProvider` |

### Testing Checklist

- [ ] Born-digital paper with display + inline equations → correct `$$`/`$`, spliced in place
- [ ] Scanned/old PDF (no text layer) → fallback fires; text, tables, formulas recovered
- [ ] Table-heavy digital PDF → pdf_oxide GFM tables preserved (no RapidTable invoked)
- [ ] Scanned table → RapidTable → GFM (or HTML for rowspan/colspan)
- [ ] Code-heavy PDF (monospace) → fenced ``` blocks
- [ ] Image-heavy PDF → every image staged as `okf-asset://`, none dropped
- [ ] Hyperlinks preserved as `[text](url)`
- [ ] GPU path: `ort.get_available_providers()` shows your EP; CPU fallback works
- [ ] Offline: with network disabled, explicit model paths load and run

---

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

This specification is **verified against production LadybugDB v0.17.1**. All code patterns have been tested end-to-end with real data, real model inference, and real database operations.

---

## 15. Open Gaps (Production-Readiness)

The [gap analysis](docs/gap-analysis.md) (v3.0, 2026-07-05) reviewed 15 gaps between the architecture spec and implementation. **13 are closed** (v5.1–v5.4). **2 remain open** and are documented below.

### Gap #7 — Concurrent Access / Locking (Medium)

**Problem**: Two CLI invocations hitting the same DB simultaneously could corrupt indexes or create duplicate concepts. No locking strategy documented or implemented.

**Recommendation**: **Option A + C** — enable WAL mode if Ladybug supports it, and document the single-writer constraint.

| Aspect | Detail |
|---|---|
| **WAL mode** | SQLite's recommended concurrent mode. Reads can proceed during writes. Writes are serialized by SQLite's internal locking. |
| **Risks** | WAL mode adds `-wal` and `-shm` sidecar files. Ladybug may not expose WAL configuration. |
| **Fallback** | Document single-writer constraint. Multiple readers are safe. |

**Implementation**: Enable `PRAGMA journal_mode = WAL` on the Ladybug connection (if supported). Add a startup check that warns if the DB appears locked.

### Gap #9 — Security (Medium)

**Problem**: No security architecture. Gaps in:
- `--allow-remote-images` SSRF risks
- Untrusted markdown execution
- Database file permissions
- Model cache integrity

**Recommendation**: **Option A + C** — add URL allowlist for remote images and security documentation.

| Aspect | Detail |
|---|---|
| **URL allowlist** | Restrict `--allow-remote-images` to configurable domains. Block `file://`, `http://0.0.0.0`, and internal IP ranges. |
| **Security docs** | Document threat model, recommend file permissions, add warnings for `--allow-remote-images`. |
| **Deferred** | Sandboxed markdown parsing (Gap #9B) — significant effort, low current threat. |
| **Optional** | HuggingFace cache verification (Gap #9D) — pin model revisions, verify hashes on first load. |

**Implementation**: Add domain allowlist check in `load_image_bytes()`. Document threat model in README.

### Gap #10 — Observability (Medium) ✅ **CLOSED** (v5.4)

**Status**: Closed — structured logging + profiling hooks implemented.

**Implemented**:
- `okfgraph/cli.py`: `_setup_logging()` with `--verbose / --quiet / --log-file` flags
- `--profile` flag: on-demand cProfile with `pstats` output
- `okfgraph/router.py`: Timing instrumentation in `import_bundle()` — logs phase durations
- stdlib `logging` used (not loguru) — avoids third-party dependency for CLI tool

**CLI flags**:
| Flag | Description |
|---|---|
| `--verbose / -v` | Enable DEBUG logging |
| `--quiet / -q` | Suppress all logging except errors |
| `--log-file <path>` | Write logs to file with 5MB rotation |
| `--profile` | Enable cProfile for current invocation |

**Timing logs** (import_bundle phases):
| Phase | Log message |
|---|---|
| Phase 0: Delta | `delta: %d changed, %d deleted (%.1fs)` |
| Phase 1: Parse | `parsed %d concept(s)` |
| Phase 2: Encode | `encode: %d texts in %.1fs` |
| Phase 3: Upsert | `upsert: %d concepts in %.1fs` |
| Phase 3.5: Chunk | `chunk: %d concepts in %.1fs` |
| Phase 4: Directories | `directories: %d in %.1fs` |
| Phase 5: Links | `links: %d concepts in %.1fs` |
| Phase 6: Images | `images: %d concepts in %.1fs` |
| Phase 7: Reindex | `reindex: %.1fs` |
| Summary | `import_bundle: %d concept(s) in %.1fs` |

**Test coverage**: 10 tests in `tests/test_logging.py`.

**Remaining follow-ups**: Prometheus metrics (#10b), query latency tracking (#10c), embedding cache hit rates (#10d).

### Gap #11 — Configuration Management (Low)

**Problem**: All config is CLI args or Python defaults. No config file for persistent settings. Users must repeat `--db`, `--dim`, `--device`, etc. on every invocation.

**Recommendation**: **Option A + B** — TOML config file with env var overrides.

| Aspect | Detail |
|---|---|
| **Precedence** | CLI > env var > file > defaults |
| **File location** | `okfgraph.toml` in bundle root or `~/.config/okfgraph/` |
| **Env vars** | `OKFGRAPH_DB`, `OKFGRAPH_DIM`, `OKFGRAPH_DEVICE`, etc. |

```toml
[database]
path = "okfgraph.db"
dim = 512

[embedding]
device = "cuda"
cache_dir = "/mnt/models"

[import]
mode = "optional"
batch_size = 64
```

**Current state**: No config file, no env var support. All settings via CLI args or defaults.

### Gap #15 — RapidAI Version Pinning (Medium) ✅ **CLOSED** (v5.4)

**Status**: Closed — version pinning + runtime warning implemented.

**Implemented**:
- `pyproject.toml` optional-dependencies `pdf-ingest` group with pinned versions
- `okfgraph/ingest/versions.py` — runtime version checking on import
- `OKFGRAPH_INGEST_ALLOW_UNPINNED=1` env var to silence warnings
- 8 tests in `tests/test_ingest.py::TestVersionChecking`

**Pinned versions**:
| Package | Version |
|---|---|
| `rapidocr` | `==1.5.2` |
| `rapid_latex_ocr` | `==1.0.13` |
| `rapid_layout` | `==0.2.0` |
| `rapid_table` | `==1.0.3` |
| `pdf_oxide` | `>=0.2.1` |

**Remaining follow-ups**: Tighten tolerance to exact version (#15b), automated version bump CI (#15c), runtime error for major version mismatches (#15d).

---

### Gap #5b — Router method `ingest_pdf()` (Medium) ✅ **CLOSED** (v5.4)

**Status**: Closed — `OKFRouter.ingest_pdf()` implemented (v5.4).

**Implemented**:
- `OKFRouter.ingest_pdf()` — programmatic API for scripts/notebooks
- Parameters: `pdf_path`, `auto_import`, `output_dir`, `routing_mode`, `mode`, `batch_size`, `purge_deleted`, `extract_images`, `on_page`
- Returns: Dict with `md_path`, `concept_ids`, `image_dir`, `page_count`
- Auto-import mode: converts to temp dir, imports via `import_bundle()`, cleans up
- Output-only mode: converts to disk, stages images as `okf-asset://` URIs
- Test coverage: 4 tests in `tests/test_ingest.py::TestIngestPdfMethod`

**Remaining follow-ups**: LLM tool definition (#5c), progress callbacks (#5d).

---

### Priority Matrix (from Gap Analysis)

| Priority | Gaps | Rationale |
|---|---|---|
| **P1 — Important** | #6 follow-ups, #14 follow-ups, ✅ #15 RapidAI pinning | Reliability, correctness, reproducibility |
| **P2 — Nice-to-have** | #7 Concurrency, #9 Security, ✅ #10 Observability, ✅ #5b ingest_pdf, #11 Config management | Operations, documentation, developer experience |

### Recommended Implementation Order

```
Phase 1 (Documentation + Low-risk)  ✅ COMPLETE
Phase 2 (Core Reliability)  ✅ #6d, #12a, #8a, #5a, #15 COMPLETE
Phase 3 (Feature Completeness)  ✅ #10a, #5b COMPLETE
├── #10a Structured logging (stdlib) + profiling hooks   ✅
├── #5b  Router method ingest_pdf() for programmatic use ✅
├── #12b GPU integration tests ✅
└── #5c  LLM tool definition (follow-up)
Phase 4 (Operations)
├── #11a TOML config file + env var support
├── #9a  URL allowlist for remote images
├── #7a  WAL mode + documentation
└── #12c End-to-end PDF tests
```

---

*This specification is a living artifact. Update the version and sections as gaps are closed.*
