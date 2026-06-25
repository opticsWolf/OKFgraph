# OKF Knowledge Graph — Architecture Specification

**Version**: 4.0 (Unified Omni — Multimodal Image Ingestion)  
**Based on**: Architecture v2.2 (implementation-verified)  
**Verified against**: LadybugDB v0.17.1, Python 3.13.14

**Storage**: LadybugDB (v0.17+) — graph + vector + full-text search.  
**Data Model**: Pydantic v2 with `extra='allow'` — preserves OKF extensibility, maps cleanly to Ladybug's `MAP` and `LIST` columns.  
**Embedding Engine**: ONNX-optimized Jina v5 text model (`jinaai/jina-embeddings-v5-text-small-retrieval`) via `optimum[onnxruntime]`.  
**Multimodal Engine**: SentenceTransformer with `jinaai/jina-embeddings-v5-omni-small-retrieval` (vision tower, lazy-loaded).  
**Unified Vector Space**: Both encoders write into one `ImageAsset.embedding` column indexed by `image_omni_idx`.  
**Search Modes**: Hybrid (RRF fusion), Traversal (pure graph), Direct (exact ID lookup), Image search (text→image via unified index).

---

## 1. Dependencies

```bash
pip install optimum[onnxruntime] transformers torch python-frontmatter pyyaml pydantic ladybug sentence-transformers Pillow
```

> `torch` is required as a backend for ONNX Runtime's tensor operations. Use `torch --index-url https://download.pytorch.org/whl/cpu` for CPU-only (lightweight) builds.

**Installed versions** (verified working):
- `optimum==2.1.0`, `onnxruntime==1.27.0`, `torch==2.12.1`, `transformers==4.57.6`
- `ladybug==0.17.1`, `pydantic==2.13.4`

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
        device: str = "cpu",
        allow_remote_images: bool = False,
    ):
        # ... validates dim, loads text model, lazy-loads omni model
```

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

`import_bundle()` uses a 3-phase pipeline with batched DB operations:

```python
def import_bundle(self, bundle_path: Optional[Path] = None, batch_size: int = 32,
                  mode: str | IngestMode = IngestMode.TEXT) -> List[str]:
    """Walk bundle, parse all files, encode in batches, upsert in bulk.

    Phase 1: Parse all .md files (frontmatter + body)
    Phase 2: Batch encode search texts via _encode_batch()
    Phase 3: Batch upsert all concepts in single transaction
    Phase 4: Batch build directory hierarchy
    Phase 5: Batch extract cross-links
    Phase 6: Image ingestion (per concept, honouring the selected mode)
    """
```

**`_batch_upsert_concepts()`**: Bulk deletes existing concepts, then creates all new ones with embeddings in a single transaction. Critically, `all_data.pop("embedding", None)` prevents embedding from leaking into the `extra` MAP column.

**`_batch_build_directories()`**: Collects all unique directory paths from concept IDs, sorts shallowest-first, creates directory nodes and CONTAINS relationships in order.

**`_batch_extract_links()`**: Collects all markdown links, batch-checks target existence, creates LINKS_TO or BrokenLink records in bulk.

---

## 4.6. Export to OKF

```python
def export_to_okf(self, concept_id: str, output_path: Path) -> None:
    """Export a concept back to an OKF .md file."""
    concept = self.get_by_id(concept_id)
    self._write_okf(concept, output_path)

def export_bundle(self, output_dir: Path,
                  directory_id: Optional[str] = None,
                  concept_type: Optional[str] = None,
                  tags: Optional[List[str]] = None) -> List[str]:
    """Export concepts from graph to OKF markdown files.

    Filters: directory_id (subtree), concept_type, tags (AND logic).
    Reconstructs directory hierarchy from concept IDs.
    """
```

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
| `okf search <query>` | Hybrid search (type/tags/parent/limit filters) |
| `okf search-images <query>` | Find images via the unified vector index |
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
| `--device cpu\|cuda` | `cpu` | Inference device (auto-fallback to CPU if CUDA unavailable) |
| `--omni-model-id <id>` | `jinaai/jina-embeddings-v5-omni-small-retrieval` | Multimodal model ID |

### 5.4. Import Options

| Option | Default | Description |
|---|---|---|
| `--mode <mode>` | `text` | Image ingestion mode: `text`, `optional`, `omni` |
| `--allow-remote-images` | — | Fetch `http(s)://` image URLs during ingestion (off by default) |
| `--batch-size <int>` | `32` | Batch size for encoding |

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

## 10. Summary of Changes (v2.2 → v4.0)

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

## 11. Performance Baseline

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
