# OKF Graph-Aware Chunking & Retrieval — Implementation Specification

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  OKF Graph (LadybugDB)                                      │
│                                                             │
│  Concept (document) ──PART_OF──► Chunk                      │
│  Concept ──LINKS_TO──► Concept                              │
│  Directory ──CONTAINS──► Concept/Directory                  │
│                                                             │
│  Chunk embedding index  │  Chunk FTS index                  │
│  Concept embedding index│  Concept FTS index                │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│  Mordant (Rust chunker)                                     │
│                                                             │
│  MarkdownChunker(text) → iterator of ExtractedChunk         │
│  ExtractedChunk { text, block_type, start_offset, end_offset }│
│                                                             │
│  MarkdownChunker.from_file(path) → same                     │
│  MarkdownChunker.from_file_mmap(path) → same (zero-copy)    │
└─────────────────────────────────────────────────────────────┘
```

## 2. Mordant Contract (What OKF Graph Expects)

OKF graph depends on mordant providing exactly this interface. Everything else is an implementation detail.

### 2.1 `MarkdownChunker` class

```python
# Constructor
chunker = mordant.MarkdownChunker(text: str)

# Class methods
chunker = mordant.MarkdownChunker.from_file(path: str)
chunker = mordant.MarkdownChunker.from_file_mmap(path: str)

# Iterator protocol — yields bare str chunks (no heading prefix)
for chunk_text in chunker:
    # chunk_text is a str — the raw block content
    # Headings are NOT yielded; they update current_header

# Properties
chunker.current_header  # str or None — last top-level heading text
chunker.node_count      # int — number of top-level nodes parsed

# OKF-facing methods
chunker.get_chunks()          # List[ExtractedChunk] — bare chunks, headings skipped
chunker.get_all_chunks()      # List[ExtractedChunk] — all chunks incl. headings
chunker.get_chunks_with_context()  # List[ExtractedChunk] — body chunks with heading prefix
chunker.get_bare_chunks()     # List[str] — all body chunks as str (same as iterating)
mordant.MarkdownChunker.get_delimiter(prev_type, curr_type)  # static str
chunker.compute_overlap_payloads(overlap_words: int)  # List[Dict[str, str]]
```

### 2.2 `ExtractedChunk` class

Returned by `get_chunks()`, `get_all_chunks()`, `get_chunks_with_context()`.

```python
chunk.text        # str — the block content (trim_end applied)
chunk.block_type  # str — one of: "Heading", "Paragraph", "CodeBlock",
                  #            "List", "Table", "Blockquote", "Other"
chunk.start_offset  # int — byte offset in original source (inclusive)
chunk.end_offset    # int — byte offset in original source (exclusive)
```

### 2.3 `BlockType` enum

Internal Rust enum exposed via `ExtractedChunk.block_type`. Deterministic `as_str()` mapping:

| Rust variant | `block_type` string | Description |
|---|---|---|
| `Heading` | `"Heading"` | Top-level markdown heading |
| `Paragraph` | `"Paragraph"` | Regular paragraph text |
| `CodeBlock` | `"CodeBlock"` | Fenced code block |
| `List` | `"List"` | Ordered or unordered list |
| `Table` | `"Table"` | GFM table |
| `Blockquote` | `"Blockquote"` | Blockquote |
| `Other` | `"Other"` | Any other block type |

### 2.4 Chunking semantics

| Node Kind | `__next__` yields? | `current_header` update |
|---|---|---|
| Heading | **No** (not yielded) | Updates to heading text |
| Paragraph | **Yes** (bare str) | No change |
| CodeBlock | **Yes** (bare str) | No change |
| List | **Yes** (bare str) | No change |
| Table | **Yes** (bare str) | No change |
| Blockquote | **Yes** (bare str) | No change |
| ThematicBreak / HtmlBlock / LinkRefDef | **No** (skipped) | No change |

**Bare chunks:** `__next__` yields raw block content with **no heading prefix**. This prevents storage bloat, improves embedding quality, and gives OKF control over context depth and breadcrumb injection at embed time.

**`current_header` tracking:** Even though headings are not yielded, `chunker.current_header` always tracks the last top-level heading seen. Nested headings (inside blockquotes, list items) do **not** update `current_header`.

### 2.5 Mordant guarantees

| Guarantee | Why OKF needs it |
|---|---|
| `__next__` yields `str` only (backward compatible) | OKF can iterate without knowing about `ExtractedChunk` |
| `block_type` is deterministic across runs | Delimiter logic depends on consistent values |
| `start_offset` / `end_offset` are byte-exact into original source | Reconstruction safety net |
| Headings are NOT yielded by `__next__` but update `current_header` | OKF skips them during storage, re-injects at embed time |
| Consecutive list items are grouped into one block | Prevents fragmentation of lists |
| `trim_end()` strips trailing whitespace from each block | OKF uses type-aware delimiters to restore spacing |
| `get_delimiter()` is static and deterministic | OKF can call it without a chunker instance |
| `compute_overlap_payloads()` returns ordered overlap dicts | OKF uses this for embedding context continuity |

### 2.6 What OKF does NOT depend on

- How mordant handles heading context injection (OKF does this itself)
- Memory mapping details (OKF just uses `from_file_mmap`)
- GIL release strategy (irrelevant to OKF)
- Arena lifetime management (internal to mordant)

### 2.3 Mordant guarantees

| Guarantee | Why OKF needs it |
|---|---|
| `block_type` is deterministic across runs | Delimiter logic depends on consistent values |
| `start_offset` / `end_offset` are byte-exact into original source | Reconstruction safety net |
| Headings are yielded as separate `ChunkKind::Heading` blocks | OKF skips them during storage, re-injects at embed time |
| Consecutive list items are grouped into one block | Prevents fragmentation of lists |
| `trim_end()` strips trailing whitespace from each block | OKF uses type-aware delimiters to restore spacing |

### 2.4 What OKF does NOT depend on

- How mordant handles heading context injection (OKF does this itself)
- Overlap behavior (OKF computes it in Python)
- Memory mapping details (OKF just uses `from_file_mmap`)
- GIL release strategy (irrelevant to OKF)

## 3. Database Schema

### 3.1 New tables in `router.py` `_ensure_schema()`

```python
# After existing tables, add:

self.conn.execute(f"""
    CREATE NODE TABLE IF NOT EXISTS Chunk (
        id STRING PRIMARY KEY,
        parent_doc_id STRING,
        chunk_index INT64,
        chunk_text STRING,
        block_type STRING,
        start_offset INT64,
        end_offset INT64,
        embedding FLOAT[{self.embedding_dim}]
    )
""")

self.conn.execute("""
    CREATE REL TABLE IF NOT EXISTS PART_OF (
        FROM Concept TO Chunk
    )
""")
```

### 3.2 Index specs in `_index_specs()`

```python
def _index_specs(self):
    vec = (
        "CALL CREATE_VECTOR_INDEX('{table}', '{name}', 'embedding', "
        "mu := 30, ml := 60, metric := 'cosine', efc := 200)"
    )
    return [
        ("Concept", "concept_embedding", vec.format(table="Concept", name="concept_embedding")),
        ("Concept", "concept_fts",
         "CALL CREATE_FTS_INDEX('Concept', 'concept_fts', ['title', 'description', 'body'])"),
        ("ImageAsset", "image_omni_idx", vec.format(table="ImageAsset", name="image_omni_idx")),
        # NEW:
        ("Chunk", "chunk_embedding", vec.format(table="Chunk", name="chunk_embedding")),
        ("Chunk", "chunk_fts",
         "CALL CREATE_FTS_INDEX('Chunk', 'chunk_fts', ['chunk_text'])"),
    ]
```

### 3.3 Constructor params in `__init__()`

```python
def __init__(
    self,
    db_path: str,
    bundle_root: str,
    model_id: str = "jinaai/jina-embeddings-v5-text-small-retrieval",
    omni_model_id: str = "jinaai/jina-embeddings-v5-omni-small-retrieval",
    embedding_dim: int = 512,
    cache_dir: Optional[str] = None,
    device: str = "cpu",
    allow_remote_images: bool = False,
    chunk_size: int = 512,           # NEW: words per chunk for overlap
    chunk_overlap: int = 64,         # NEW: overlap words between chunks
    enable_chunking: bool = True,    # NEW: toggle chunking
):
    ...
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap
    self.enable_chunking = enable_chunking
```

## 4. Chunking Implementation

### 4.1 Chunking method in `router.py`

Use `chunker.get_chunks()` to get `ExtractedChunk` objects with metadata. The iterator `__next__` yields bare `str` only.

```python
def _split_into_chunks(
    self, body: str, document_id: str
) -> List[Dict[str, Any]]:
    """Split document body into pure blocks using mordant chunker.

    Uses chunker.get_chunks() to get ExtractedChunk objects with
    block_type and byte offsets. Headings are excluded (they are
    context, not content). No overlap is stored.
    Overlap is computed at embed/search time via compute_overlap_payloads().
    """
    chunker = mordant.MarkdownChunker(body)
    chunks: List[Dict[str, Any]] = []
    index = 0

    # get_chunks() returns ExtractedChunk objects (headings skipped)
    for chunk in chunker.get_chunks():
        chunks.append({
            "parent_doc_id": document_id,
            "chunk_text": chunk.text,
            "block_type": chunk.block_type,
            "start_offset": chunk.start_offset,
            "end_offset": chunk.end_offset,
            "chunk_index": index,
        })
        index += 1

    return chunks
```

### 4.2 Overlap computation — use mordant's built-in method

Mordant provides `compute_overlap_payloads(overlap_words)` directly on the chunker.
OKF can use this instead of implementing overlap in Python.

```python
def _split_into_chunks_with_overlap(
    self, body: str, document_id: str
) -> List[Dict[str, Any]]:
    """Split and compute overlap using mordant's built-in method."""
    chunker = mordant.MarkdownChunker(body)

    # Use mordant's compute_overlap_payloads — returns List[Dict]
    payloads = chunker.compute_overlap_payloads(self.chunk_overlap)

    # payloads has keys like "chunk:0", "chunk:1" with overlapped text
    chunks: List[Dict[str, Any]] = []
    for i, payload in enumerate(payloads):
        chunks.append({
            "parent_doc_id": document_id,
            "chunk_text": payload["text"],  # overlapped text for embedding
            "chunk_index": i,
        })

    return chunks
```

**Fallback** — if OKF needs custom overlap logic, the manual implementation:

```python
def _compute_overlap_payloads(
    self, chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Add overlap in memory for embedding or display.

    The tail of chunk N is prepended to chunk N+1. Overlap is never
    stored — it is purely a query-time transformation.
    """
    payloads: List[Dict[str, Any]] = []
    prev_tail = ""

    for chunk in chunks:
        if prev_tail:
            embed_text = f"{prev_tail}\n\n{chunk['chunk_text']}"
        else:
            embed_text = chunk["chunk_text"]

        payloads.append({
            "chunk_id": f"{chunk['parent_doc_id']}#chunk:{chunk['chunk_index']}",
            "text": embed_text,
        })

        # Compute tail from the PURE chunk text (not the overlapped text)
        words = chunk["chunk_text"].split()
        if self.chunk_overlap > 0:
            prev_tail = "  ".join(words[-self.chunk_overlap:])
        else:
            prev_tail = ""

    return payloads
```

### 4.3 Reconstruction method

`get_delimiter()` is a **static method** on `MarkdownChunker` — no instance needed.

```python
def reconstruct_document(self, document_id: str) -> str:
    """Reconstruct original markdown from stored chunks.

    Uses block_type to determine correct delimiters between chunks.
    Approximate byte-exact reconstruction (~98% fidelity).
    """
    result = self.conn.execute("""
        SELECT chunk_text, block_type, chunk_index
        FROM Chunk WHERE parent_doc_id = $id
        ORDER BY chunk_index
    """, {"id": document_id})
    rows = result.rows_as_dict().get_all()

    if not rows:
        return ""

    parts = [rows[0]["chunk_text"]]
    for i in range(1, len(rows)):
        # Use mordant's static get_delimiter for deterministic delimiters
        sep = mordant.MarkdownChunker.get_delimiter(
            rows[i - 1]["block_type"], rows[i]["block_type"]
        )
        parts.append(sep + rows[i]["chunk_text"])

    return "".join(parts)
```

**Delimiter rules (from mordant):**

| prev_type | curr_type | Delimiter | Reason |
|---|---|---|---|
| `List` | `List` | `"\n"` | Items belong together |
| `Blockquote` | `Blockquote` | `"\n> "` | Re-attach quote marker |
| anything else | anything else | `"\n\n"` | Paragraph break |

## 4.4 OKF-facing methods (direct usage)

Mordant provides several methods that OKF can call directly for common workflows:

### `get_chunks()` — bare chunks with metadata

```python
chunker = mordant.MarkdownChunker("# Title\n\nPara one\n\n## Sub\n\nPara two")
for chunk in chunker.get_chunks():
    print(chunk.block_type, chunk.text, chunk.start_offset, chunk.end_offset)
# Paragraph Para one 9 17
# Paragraph Para two 27 35
```

Returns `List[ExtractedChunk]` — body chunks only, headings skipped, no prefix.

### `get_all_chunks()` — all chunks including headings

```python
for chunk in chunker.get_all_chunks():
    print(chunk.block_type, chunk.text)
# Heading # Title
# Paragraph Para one
# Heading ## Sub
# Paragraph Para two
```

Use for structural graph building (heading nodes → concept nodes).

### `get_chunks_with_context()` — body chunks with heading prefix

```python
for chunk in chunker.get_chunks_with_context():
    print(chunk.text)
# # Title\n\nPara one
# ## Sub\n\nPara two
```

Use for display/rendering where heading context is needed.

### `get_bare_chunks()` — all body chunks as str

```python
chunks = chunker.get_bare_chunks()
# Same as list(chunker), but as a list
```

Equivalent to iterating the chunker, but returns a list.

### `compute_overlap_payloads(n)` — embedding context

```python
chunker = mordant.MarkdownChunker("# Title\n\nFirst para second para third para.\n\n## Sub\n\nMore text here.")
payloads = chunker.compute_overlap_payloads(2)
# [{"chunk:0": "First para second para third para."},
#  {"chunk:1": "third  para.\n\nMore text here."}]
```

Returns `List[Dict[str, str]]` with keys like `"chunk:0"`, `"chunk:1"`. The tail of chunk N (n words) is prepended to chunk N+1 for context continuity during embedding.

### `get_delimiter(prev, curr)` — reconstruction

```python
mordant.MarkdownChunker.get_delimiter("List", "List")            # "\n"
mordant.MarkdownChunker.get_delimiter("Blockquote", "Blockquote") # "\n> "
mordant.MarkdownChunker.get_delimiter("Paragraph", "CodeBlock")   # "\n\n"
```

Static method — no chunker instance needed.

## 5. Ingestion Pipeline

### 5.1 Updated `import_from_okf()`

```python
def import_from_okf(
    self,
    file_path: Path,
    mode: "str | IngestMode" = IngestMode.TEXT,
    rebuild_indexes: bool = True,
) -> str:
    mode = IngestMode.coerce(mode)

    # 1. Parse frontmatter/body
    concept, body, concept_id = self._parse_source_file(file_path, self.bundle_root)

    # 2. Chunk the body (NEW)
    chunk_ids: List[str] = []
    if self.enable_chunking:
        # Delete old chunks for this document (re-import)
        self.conn.execute(
            "MATCH (c:Concept {id: $id})-[:PART_OF]->(ch:Chunk) DELETE ch",
            {"id": concept_id},
        )

        chunks = self._split_into_chunks(body, concept_id)
        if chunks:
            # Compute overlap payloads for embedding
            payloads = self._compute_overlap_payloads(chunks)
            texts = [p["text"] for p in payloads]
            embeddings = self._encode_batch(texts, task="Document")

            self.conn.execute("BEGIN TRANSACTION")
            try:
                for payload, emb in zip(payloads, embeddings):
                    chunk_id = payload["chunk_id"]
                    # Find the original chunk for metadata
                    orig_chunk = next(
                        c for c in chunks
                        if c["chunk_index"] == int(chunk_id.split(":")[-1])
                    )
                    self.conn.execute("""
                        CREATE (ch:Chunk {
                            id: $id, parent_doc_id: $doc_id,
                            chunk_index: $idx, chunk_text: $text,
                            block_type: $block_type,
                            start_offset: $start, end_offset: $end,
                            embedding: $emb
                        })
                    """, {
                        "id": chunk_id,
                        "doc_id": concept_id,
                        "idx": orig_chunk["chunk_index"],
                        "text": orig_chunk["chunk_text"],
                        "block_type": orig_chunk["block_type"],
                        "start": orig_chunk["start_offset"],
                        "end": orig_chunk["end_offset"],
                        "emb": emb,
                    })
                    self.conn.execute("""
                        MATCH (d:Concept {id: $doc})
                        MATCH (ch:Chunk {id: $cid})
                        MERGE (d)-[:PART_OF]->(ch)
                    """, {"doc": concept_id, "cid": chunk_id})
                self.conn.execute("COMMIT")
            except Exception:
                try:
                    self.conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            chunk_ids = [p["chunk_id"] for p in payloads]
            self._bump_write_epoch()

    # 3. Full-document embedding (backward compat)
    search_text = f"{concept.title or ''} {concept.description or ''} {concept.body or ''}"
    concept.embedding = self._encode(search_text, task="Document")

    # 4. Insert concept
    self._insert_concept(concept, body, concept_id)

    # 5. Image ingestion
    self._ingest_concept_images(concept_id, body, file_path.parent, mode)

    # 6. Rebuild indexes
    if rebuild_indexes:
        self._build_search_indexes(rebuild=True)

    return concept_id
```

### 5.2 Updated `import_bundle()` — chunking phase

After Phase 2 (batch concept embeddings) and Phase 3 (batch upsert concepts), add:

```python
# Phase 2.5: Chunk all documents
if self.enable_chunking:
    all_chunks: List[Dict[str, Any]] = []
    for p in parsed:
        chunks = self._split_into_chunks(p["body"], p["cid"])
        all_chunks.extend(chunks)

    if all_chunks:
        # Group by parent doc for overlap computation
        chunks_by_doc: Dict[str, List[Dict]] = {}
        for c in all_chunks:
            chunks_by_doc.setdefault(c["parent_doc_id"], []).append(c)

        all_payloads: List[Dict[str, Any]] = []
        for doc_id, doc_chunks in chunks_by_doc.items():
            payloads = self._compute_overlap_payloads(doc_chunks)
            for p in payloads:
                p["_doc_id"] = doc_id  # carry doc_id through
            all_payloads.extend(all_payloads)

        # Batch encode
        texts = [p["text"] for p in all_payloads]
        embeddings = self._encode_batch(texts, task="Document")

        self.conn.execute("BEGIN TRANSACTION")
        try:
            for payload, emb in zip(all_payloads, embeddings):
                doc_id = payload.pop("_doc_id")
                chunk_idx = int(payload["chunk_id"].split(":")[-1])
                orig_chunk = next(
                    c for c in chunks_by_doc[doc_id]
                    if c["chunk_index"] == chunk_idx
                )
                self.conn.execute("""
                    CREATE (ch:Chunk {
                        id: $id, parent_doc_id: $doc_id,
                        chunk_index: $idx, chunk_text: $text,
                        block_type: $block_type,
                        start_offset: $start, end_offset: $end,
                        embedding: $emb
                    })
                """, {
                    "id": payload["chunk_id"],
                    "doc_id": doc_id,
                    "idx": orig_chunk["chunk_index"],
                    "text": orig_chunk["chunk_text"],
                    "block_type": orig_chunk["block_type"],
                    "start": orig_chunk["start_offset"],
                    "end": orig_chunk["end_offset"],
                    "emb": emb,
                })
                self.conn.execute("""
                    MATCH (d:Concept {id: $doc})
                    MATCH (ch:Chunk {id: $cid})
                    MERGE (d)-[:PART_OF]->(ch)
                """, {"doc": doc_id, "cid": payload["chunk_id"]})
            self.conn.execute("COMMIT")
        except Exception:
            try:
                self.conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        self._bump_write_epoch()
```

## 6. Search Implementation

### 6.1 Chunk-level search

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
    """Search at chunk granularity using RRF over vector + FTS.

    Returns chunks ranked by RRF score, with parent document metadata.
    Applies `max_chunks_per_doc` to limit how many chunks from the same
    document appear in results.
    """
    if not getattr(self, "_search_available", False):
        raise RuntimeError("Search unavailable — vector/fts extensions not loaded.")

    query_vec = self._encode(query, task="Query")

    # Vector search over chunks
    vec_results = self.conn.execute(
        "CALL QUERY_VECTOR_INDEX('Chunk', 'chunk_embedding', $vec, $k)",
        {"vec": query_vec, "k": limit * 3}
    )
    vec_scores: Dict[str, Dict[str, Any]] = {}
    for row in vec_results.rows_as_dict().get_all():
        node = row.get("node", {})
        if not isinstance(node, dict):
            continue
        cid = node.get("id")
        if cid:
            vec_scores[cid] = {
                "score": 1 - row.get("distance", 0),
                "node": node,
            }

    # FTS search over chunks
    fts_results = self.conn.execute(
        "CALL QUERY_FTS_INDEX('Chunk', 'chunk_fts', $query)",
        {"query": query}
    )
    fts_scores: Dict[str, float] = {}
    for row in fts_results.rows_as_dict().get_all():
        node = row.get("node", {})
        cid = node.get("id") if isinstance(node, dict) else None
        if cid:
            fts_scores[cid] = row.get("score", 0)

    # RRF fusion
    def _rank_map(scores: Dict[str, float]) -> Dict[str, int]:
        ordered = sorted(scores, key=scores.get, reverse=True)
        return {cid: i + 1 for i, cid in enumerate(ordered)}

    vec_rank = _rank_map({cid: v["score"] for cid, v in vec_scores.items()})
    fts_rank = _rank_map(fts_scores)

    combined: List[Tuple[str, float]] = []
    for cid in set(vec_scores) | set(fts_scores):
        score = 0.0
        if cid in vec_rank:
            score += 1.0 / (60 + vec_rank[cid])
        if cid in fts_rank:
            score += 1.0 / (60 + fts_rank[cid])
        combined.append((cid, score))
    combined.sort(key=lambda x: x[1], reverse=True)

    # Fetch parent doc metadata in bulk
    chunk_ids = [cid for cid, _ in combined]
    meta_by_parent: Dict[str, Dict[str, Any]] = {}
    if chunk_ids:
        results = self.conn.execute("""
            MATCH (c:Concept) WHERE c.id IN $ids
            RETURN c.id, c.title, c.type, c.description, c.tags
        """, {"ids": chunk_ids}).rows_as_dict().get_all()
        for row in results:
            meta_by_parent[row["c.id"]] = row

    # Assemble results, applying per-doc limit and graph filters
    out: List[Dict[str, Any]] = []
    doc_counts: Dict[str, int] = {}  # track chunks per doc

    for cid, rrf_score in combined:
        chunk_node = vec_scores.get(cid, {}).get("node", {})
        parent_id_val = chunk_node.get("parent_doc_id")
        parent_meta = meta_by_parent.get(parent_id_val, {})

        # Apply per-doc limit
        if parent_id_val:
            doc_counts[parent_id_val] = doc_counts.get(parent_id_val, 0) + 1
            if doc_counts[parent_id_val] > max_chunks_per_doc:
                continue

        out.append({
            "chunk_id": cid,
            "chunk_text": chunk_node.get("chunk_text", ""),
            "chunk_title": chunk_node.get("chunk_text", "")[:80],
            "chunk_index": chunk_node.get("chunk_index"),
            "block_type": chunk_node.get("block_type"),
            "parent_doc_id": parent_id_val,
            "parent_title": parent_meta.get("c.title", ""),
            "parent_type": parent_meta.get("c.type", ""),
            "parent_tags": parent_meta.get("c.tags", []),
            "rrf_score": rrf_score,
        })
        if len(out) >= limit:
            break

    return out
```

### 6.2 Updated `search_hybrid()` — backward compatible, adds `include_chunks`

```python
def search_hybrid(
    self,
    query: str,
    concept_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    parent_id: Optional[str] = None,
    exclude_reserved: bool = True,
    limit: int = 10,
    include_chunks: bool = False,  # NEW
) -> List[Dict[str, Any]]:
    """Hybrid search: RRF fusion of vector + FTS with optional chunk results.

    Returns Concept-level results (unchanged behavior). If `include_chunks=True`,
    each result also contains `matched_chunks` — the top chunks from that
    document matching the query.
    """
    # ... existing logic unchanged through Stage 4 ...

    # Stage 4: fetch metadata (and apply graph filters) — UNCHANGED
    where_clauses: List[str] = ["c.id IN $ids"]
    params: Dict[str, Any] = {"ids": candidate_ids}
    if concept_type:
        where_clauses.append("c.type = $type")
        params["type"] = concept_type
    if tags:
        where_clauses.append("ANY(tag IN $tags WHERE tag IN c.tags)")
        params["tags"] = tags
    if parent_id:
        where_clauses.append(
            "EXISTS { MATCH (p:Directory {id: $parent})-[:CONTAINS*1..3]->(c) }"
        )
        params["parent"] = parent_id
    if exclude_reserved:
        where_clauses.append(
            "NOT c.id ENDS WITH 'index' AND NOT c.id ENDS WITH 'log'"
        )

    cypher = f"""
    MATCH (c:Concept)
    WHERE {" AND ".join(where_clauses)}
    RETURN c.id, c.title, c.type, c.description, c.tags
    """
    rows = self.conn.execute(cypher, params).rows_as_dict().get_all()
    meta_by_id = {row["c.id"]: row for row in rows}

    # Assemble in RRF order
    results: List[Dict[str, Any]] = []
    for cid, _ in combined:
        row = meta_by_id.get(cid)
        if row is None:
            continue
        desc = row["c.description"] or ""
        if len(desc) > 200:
            desc = desc[:200] + "..."
        result = {
            "id": cid,
            "title": row["c.title"],
            "type": row["c.type"],
            "description": desc,
            "tags": row["c.tags"],
            "relevance_score": score_by_id[cid],
        }

        # NEW: attach chunks if requested
        if include_chunks:
            chunks = self.search_chunks(
                query=query, limit=3, parent_id=cid
            )
            result["matched_chunks"] = chunks

        results.append(result)
        if len(results) >= limit:
            break

    return results
```

## 7. Graph-Aware Retrieval

### 7.1 Hub score reranking

```python
def _compute_hub_scores(self, concept_ids: List[str]) -> Dict[str, float]:
    """Count incoming LINKS_TO edges for each concept.

    Higher hub score = more concepts point to this one = more authoritative.
    """
    if not concept_ids:
        return {}

    result = self.conn.execute("""
        MATCH (x:Concept)<-[:LINKS_TO]-(c:Concept)
        WHERE c.id IN $ids
        RETURN c.id AS id, count(x) AS cnt
    """, {"ids": concept_ids})
    return {r["id"]: r["cnt"] for r in result.rows_as_dict().get_all()}
```

### 7.2 Multi-hop context expansion

```python
def search_with_context(
    self,
    query: str,
    limit: int = 5,
    context_hops: int = 1,
) -> List[Dict[str, Any]]:
    """Search chunks + expand each result with graph neighborhood context.

    Returns chunks enriched with:
      - incoming_links: concepts that link TO this document
      - outgoing_links: concepts this document links TO
      - ancestry: directory path from root
      - siblings: other concepts in the same parent directory
    """
    chunks = self.search_chunks(query, limit=limit * 2)

    enriched: List[Dict[str, Any]] = []
    for chunk in chunks[:limit]:
        parent_id = chunk["parent_doc_id"]

        incoming = self.traverse(parent_id, "LINKS_TO", "INCOMING",
                                  depth=context_hops)
        outgoing = self.traverse(parent_id, "LINKS_TO", "OUTGOING",
                                  depth=context_hops)
        ancestry = self._get_ancestry(parent_id)
        siblings = self._get_siblings(parent_id)

        enriched.append({
            "chunk": chunk,
            "document": self.get_by_id(parent_id),
            "incoming_links": incoming[:5],
            "outgoing_links": outgoing[:5],
            "ancestry": ancestry,
            "siblings": siblings[:5],
        })

    return enriched
```

### 7.3 Helper methods

```python
def _get_ancestry(self, concept_id: str, max_depth: int = 5) -> List[Dict[str, Any]]:
    """Return directory path from root to this concept."""
    result = self.conn.execute("""
        MATCH path = (d:Directory)-[:CONTAINS*1..$depth]->(c:Concept {id: $id})
        RETURN d.id AS dir_id
        ORDER BY length(path) DESC
        LIMIT 1
    """, {"id": concept_id, "depth": max_depth})
    rows = result.rows_as_dict().get_all()
    if not rows:
        return []
    path_str = rows[0]["dir_id"]
    return [
        {"id": p, "title": p.split("/")[-1]}
        for p in path_str.split("/") if p
    ]


def _get_siblings(self, concept_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Return other concepts in the same parent directory."""
    parent_result = self.conn.execute("""
        MATCH (d:Directory)-[:CONTAINS]->(c:Concept {id: $id})
        RETURN d.id AS parent_id
    """, {"id": concept_id})
    rows = parent_result.rows_as_dict().get_all()
    if not rows:
        return []
    parent_id = rows[0]["parent_id"]

    result = self.conn.execute("""
        MATCH (d:Directory {id: $pid})-[:CONTAINS]->(s:Concept)
        WHERE s.id <> $cid
        RETURN s.id, s.title, s.type
        LIMIT $limit
    """, {"pid": parent_id, "cid": concept_id, "limit": limit})
    return [
        {"id": r["s.id"], "title": r["s.title"], "type": r["s.type"]}
        for r in result.rows_as_dict().get_all()
    ]
```

### 7.4 Graph-aware reranking in search

```python
def search_chunks_with_hub_score(
    self,
    query: str,
    limit: int = 10,
    hub_weight: float = 0.3,
) -> List[Dict[str, Any]]:
    """Search chunks + rerank by graph hub score.

    Final score = (1 - hub_weight) * rrf_score + hub_weight * normalized_hub_score.
    """
    raw_results = self.search_chunks(query, limit=limit * 2)

    # Collect unique parent doc IDs
    parent_ids = list({r["parent_doc_id"] for r in raw_results if r["parent_doc_id"]})
    hub_scores = self._compute_hub_scores(parent_ids)
    max_hub = max(hub_scores.values()) if hub_scores else 1

    for r in raw_results:
        pid = r["parent_doc_id"]
        hub = hub_scores.get(pid, 0)
        normalized_hub = hub / max_hub if max_hub > 0 else 0
        r["hub_score"] = normalized_hub
        r["final_score"] = (
            (1 - hub_weight) * r["rrf_score"] + hub_weight * normalized_hub
        )

    raw_results.sort(key=lambda x: x["final_score"], reverse=True)
    return raw_results[:limit]
```

## 8. Models (models.py)

```python
class ChunkModel(BaseModel):
    """Represents a chunked section of a document."""
    id: str
    parent_doc_id: str
    chunk_index: int
    chunk_text: str
    block_type: str
    start_offset: int = 0
    end_offset: int = 0
    rrf_score: Optional[float] = None
    hub_score: Optional[float] = None
    final_score: Optional[float] = None
    parent_title: Optional[str] = None
    parent_type: Optional[str] = None
    parent_tags: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None  # internal
```

## 9. Tool Definitions (tools.py)

```python
TOOLS = [
    # ... existing tools unchanged ...
    {
        "name": "search_chunks",
        "description": (
            "Search at chunk/paragraph granularity. Returns matched text snippets "
            "with parent document info. Use for precise passage retrieval."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query text.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                    "description": "Maximum number of chunks to return.",
                },
                "max_chunks_per_doc": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 3,
                    "description": "Max chunks to return per document.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_with_context",
        "description": (
            "Search + expand results with graph neighborhood context. "
            "Returns chunks enriched with incoming/outgoing links, "
            "directory ancestry, and sibling concepts. Use when you need "
            "relational context, not just the matched text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query text.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                    "description": "Maximum number of results.",
                },
                "context_hops": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3,
                    "default": 1,
                    "description": "How many hops to expand for context.",
                },
            },
            "required": ["query"],
        },
    },
]
```

## 10. CLI Updates (cli.py)

### 10.1 New arguments

```python
def _add_chunking_args(parser):
    """Add chunking arguments to a subparser."""
    parser.add_argument(
        "--chunk-size", type=int, default=512,
        help="Words per chunk for overlap computation (default: 512)",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=64,
        help="Words of overlap between chunks (default: 64)",
    )
    parser.add_argument(
        "--no-chunking", action="store_true",
        help="Disable chunking (embed whole documents only)",
    )
```

### 10.2 Add to subparsers

```python
# In build_parser(), add to "import" and "shell" subparsers:
p = sub.add_parser("import", ...)
_add_chunking_args(p)

p = sub.add_parser("shell", ...)
_add_chunking_args(p)
```

### 10.3 Shell commands

```python
# In _shell(), add to the banner and command dispatch:
banner += """
  search-chunks <query>              — chunk-level search
  search <query> --chunks            — hybrid search with chunks
  context <query>                    — search with graph context
  hub-search <query>                 — chunk search reranked by hub score
  path <id1> <id2>                   — find path between two concepts
  siblings <id>                      — list sibling concepts
  ancestry <id>                      — show directory path
  reconstruct <id>                   — reconstruct document from chunks
"""

# In command dispatch:
elif cmd == "search-chunks" and rest:
    results = router.search_chunks(rest.strip(), limit=args.limit)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['rrf_score']:.4f}] {r['parent_title']} §{r['chunk_index']}")
        print(f"     {r['chunk_text'][:120]}")

elif cmd == "context" and rest:
    results = router.search_with_context(rest.strip(), limit=args.limit)
    for i, r in enumerate(results, 1):
        chunk = r["chunk"]
        print(f"  {i}. [{chunk['rrf_score']:.4f}] {chunk['parent_title']} §{chunk['chunk_index']}")
        print(f"     {chunk['chunk_text'][:120]}")
        if r["incoming_links"]:
            print(f"     ← linked by: {', '.join(l['title'] for l in r['incoming_links'][:3])}")
        if r["outgoing_links"]:
            print(f"     → links to: {', '.join(l['title'] for l in r['outgoing_links'][:3])}")

elif cmd == "hub-search" and rest:
    results = router.search_chunks_with_hub_score(rest.strip(), limit=args.limit)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['final_score']:.4f}] {r['parent_title']} §{r['chunk_index']}")
        print(f"     hub={r['hub_score']:.2f} rrf={r['rrf_score']:.2f}")
        print(f"     {r['chunk_text'][:120]}")

elif cmd == "reconstruct" and rest:
    doc_text = router.reconstruct_document(rest.strip())
    print(doc_text)

elif cmd == "siblings" and rest:
    siblings = router._get_siblings(rest.strip())
    for s in siblings:
        print(f"  {s['title']} ({s['type']})")

elif cmd == "ancestry" and rest:
    ancestry = router._get_ancestry(rest.strip())
    for a in ancestry:
        print(f"  {a['title']}")
```

## 11. Test Plan

### 11.1 `tests/test_chunking.py`

```python
class TestChunking:
    def setup_method(self):
        self.router = OKFRouter(
            db_path=":memory:", bundle_root="/tmp",
            enable_chunking=True, chunk_overlap=64,
        )

    def test_chunker_get_chunks_yields_extracted_chunks(self):
        """Mordant chunker.get_chunks() yields ExtractedChunk with required attributes."""
        chunker = mordant.MarkdownChunker("# Title\n\nPara.")
        chunks = chunker.get_chunks()
        assert len(chunks) >= 1
        for c in chunks:
            assert hasattr(c, "text")
            assert hasattr(c, "block_type")
            assert hasattr(c, "start_offset")
            assert hasattr(c, "end_offset")
            assert c.block_type in ("Heading", "Paragraph", "List",
                                     "CodeBlock", "Blockquote", "Table", "Other")

    def test_chunker_iterator_yields_str(self):
        """Mordant chunker.__next__ yields str (backward compatible)."""
        chunker = mordant.MarkdownChunker("# Title\n\nPara.")
        for chunk in chunker:
            assert isinstance(chunk, str)

    def test_skip_headings_in_split(self):
        """_split_into_chunks excludes Heading blocks."""
        chunks = self.router._split_into_chunks("# Title\n\nBody.", "doc1")
        block_types = [c["block_type"] for c in chunks]
        assert "Heading" not in block_types

    def test_overlap_preserves_context(self):
        """Overlap payloads contain tail of previous chunk."""
        body = "First para.\n\nSecond para.\n\nThird para."
        chunks = self.router._split_into_chunks(body, "doc1")
        payloads = self.router._compute_overlap_payloads(chunks)
        assert len(payloads) == len(chunks)
        # Second payload should start with tail of first chunk
        assert "First" in payloads[1]["text"]

    def test_mordant_compute_overlap(self):
        """Mordant's built-in compute_overlap_payloads works."""
        chunker = mordant.MarkdownChunker("Para one.\n\nPara two.")
        payloads = chunker.compute_overlap_payloads(2)
        assert len(payloads) == 2
        assert "chunk:0" in payloads[0]
        assert "chunk:1" in payloads[1]

    def test_single_chunk_for_short_doc(self):
        body = "Short doc."
        chunks = self.router._split_into_chunks(body, "doc1")
        assert len(chunks) == 1

    def test_empty_body(self):
        chunks = self.router._split_into_chunks("", "doc1")
        assert chunks == []

    def test_from_file_mmap(self):
        """from_file_mmap works and yields chunks."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Mmap test\n\nPara.")
            f.flush()
            chunker = mordant.MarkdownChunker.from_file_mmap(f.name)
            chunks = chunker.get_chunks()
            assert len(chunks) >= 1

    def test_get_all_chunks_includes_headings(self):
        """get_all_chunks() returns headings as separate chunks."""
        chunker = mordant.MarkdownChunker("# Title\n\nPara.")
        all_chunks = chunker.get_all_chunks()
        block_types = [c.block_type for c in all_chunks]
        assert "Heading" in block_types
        assert "Paragraph" in block_types

    def test_get_chunks_with_context(self):
        """get_chunks_with_context() prepends heading to body chunks."""
        chunker = mordant.MarkdownChunker("# Title\n\nPara.")
        ctx_chunks = chunker.get_chunks_with_context()
        assert len(ctx_chunks) == 1
        assert "# Title" in ctx_chunks[0].text
        assert "Para" in ctx_chunks[0].text

    def test_current_header_tracks_last_heading(self):
        """current_header tracks the last top-level heading."""
        chunker = mordant.MarkdownChunker("# First\n\nPara1\n\n## Second\n\nPara2")
        list(chunker)  # consume iterator
        assert chunker.current_header == "## Second"

    def test_nested_headings_dont_leak(self):
        """Headings inside blockquotes don't update current_header."""
        chunker = mordant.MarkdownChunker("# Outer\n\n> # Nested\n\n> Quote.")
        list(chunker)
        assert chunker.current_header == "# Outer"

    def test_get_delimiter_static(self):
        """get_delimiter is static and deterministic."""
        assert mordant.MarkdownChunker.get_delimiter("List", "List") == "\n"
        assert mordant.MarkdownChunker.get_delimiter("Blockquote", "Blockquote") == "\n> "
        assert mordant.MarkdownChunker.get_delimiter("Paragraph", "CodeBlock") == "\n\n"

    def test_node_count_includes_headings(self):
        """node_count counts all top-level nodes including headings."""
        chunker = mordant.MarkdownChunker("# H1\n\nPara\n\n## H2\n\nPara2")
        assert chunker.node_count == 4  # 2 headings + 2 paragraphs
```

### 11.2 `tests/test_reconstruction.py`

```python
class TestReconstruction:
    def setup_method(self):
        self.router = OKFRouter(
            db_path=":memory:", bundle_root="/tmp",
            enable_chunking=True,
        )

    def test_reconstruct_simple_doc(self):
        """Reconstruction approximates original markdown."""
        body = "# Title\n\nFirst para.\n\nSecond para."
        chunks = self.router._split_into_chunks(body, "doc1")
        # Simulate DB insert
        for i, c in enumerate(chunks):
            c["chunk_index"] = i
        # Reconstruction
        reconstructed = self._reconstruct_from_chunks(chunks)
        assert "Title" in reconstructed
        assert "First para." in reconstructed
        assert "Second para." in reconstructed

    def test_list_delimiter(self):
        """Consecutive lists get correct delimiter."""
        chunks = [
            {"block_type": "List", "chunk_text": "- item 1"},
            {"block_type": "List", "chunk_text": "- item 2"},
        ]
        result = self._reconstruct_from_chunks(chunks)
        assert "- item 1\n- item 2" in result  # single newline between

    def test_blockquote_delimiter(self):
        """Consecutive blockquotes get > prefix."""
        chunks = [
            {"block_type": "Blockquote", "chunk_text": "> quote 1"},
            {"block_type": "Blockquote", "chunk_text": "> quote 2"},
        ]
        result = self._reconstruct_from_chunks(chunks)
        assert "> quote 1\n> quote 2" in result

    def test_empty_doc(self):
        assert self.router.reconstruct_document("nonexistent") == ""

    @staticmethod
    def _reconstruct_from_chunks(chunks):
        """Helper to reconstruct without DB (unit test friendly)."""
        if not chunks:
            return ""
        parts = [chunks[0]["chunk_text"]]
        for i in range(1, len(chunks)):
            # Use mordant's static get_delimiter
            sep = mordant.MarkdownChunker.get_delimiter(
                chunks[i - 1]["block_type"], chunks[i]["block_type"]
            )
            parts.append(sep + chunks[i]["chunk_text"])
        return "".join(parts)
```

### 11.3 `tests/test_chunk_search.py`

```python
class TestChunkSearch:
    def setup_method(self):
        self.router = OKFRouter(
            db_path=":memory:", bundle_root="/tmp",
            enable_chunking=True, chunk_overlap=64,
        )
        # Create test files
        self._create_test_files()

    def _create_test_files(self):
        import tempfile
        import os
        self.tmpdir = tempfile.mkdtemp()
        # Doc 1: machine learning topic
        (Path(self.tmpdir) / "ml.md").write_text(
            "---\ntype: note\ntitle: ML Basics\n---\n"
            "Machine learning is a subset of artificial intelligence.\n\n"
            "Neural networks are a key technique in ML."
        )
        # Doc 2: deep learning topic with link to doc 1
        (Path(self.tmpdir) / "dl.md").write_text(
            "---\ntype: note\ntitle: Deep Learning\n---\n"
            "Deep learning uses neural networks with many layers.\n\n"
            "See [ML Basics](ml.md) for fundamentals."
        )

        with self.router:
            self.router.import_from_okf(Path(self.tmpdir / "ml.md"))
            self.router.import_from_okf(Path(self.tmpdir / "dl.md"))

    def teardown_method(self):
        import shutil
        if hasattr(self, "tmpdir") and os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def test_search_returns_chunks(self):
        results = self.router.search_chunks("machine learning", limit=5)
        assert len(results) > 0
        assert all("chunk_id" in r for r in results)
        assert all("parent_doc_id" in r for r in results)
        assert all("chunk_text" in r for r in results)
        assert all("rrf_score" in r for r in results)

    def test_chunks_have_parent_info(self):
        results = self.router.search_chunks("neural", limit=5)
        for r in results:
            doc = self.router.get_by_id(r["parent_doc_id"])
            assert doc is not None

    def test_rrf_scores_ordered(self):
        results = self.router.search_chunks("learning", limit=10)
        scores = [r["rrf_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_max_chunks_per_doc(self):
        results = self.router.search_chunks("learning", limit=20,
                                             max_chunks_per_doc=1)
        parent_ids = [r["parent_doc_id"] for r in results]
        assert len(parent_ids) == len(set(parent_ids))  # no duplicates

    def test_search_hybrid_with_chunks(self):
        results = self.router.search_hybrid("machine", include_chunks=True)
        assert len(results) > 0
        assert "matched_chunks" in results[0]
        assert len(results[0]["matched_chunks"]) > 0
```

### 11.4 `tests/test_graph_enrichment.py`

```python
class TestGraphEnrichment:
    def setup_method(self):
        self.router = OKFRouter(
            db_path=":memory:", bundle_root="/tmp",
            enable_chunking=True,
        )
        self._create_test_files()

    def _create_test_files(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        (Path(self.tmpdir) / "doc1.md").write_text(
            "---\ntype: note\ntitle: Doc One\n---\nContent of doc one."
        )
        (Path(self.tmpdir) / "doc2.md").write_text(
            "---\ntype: note\ntitle: Doc Two\n---\nContent of doc two. [See Doc One](doc1.md)"
        )
        (Path(self.tmpdir) / "doc3.md").write_text(
            "---\ntype: note\ntitle: Doc Three\n---\nContent of doc three."
        )
        with self.router:
            self.router.import_from_okf(Path(self.tmpdir / "doc1.md"))
            self.router.import_from_okf(Path(self.tmpdir / "doc2.md"))
            self.router.import_from_okf(Path(self.tmpdir / "doc3.md"))

    def teardown_method(self):
        import shutil
        if hasattr(self, "tmpdir") and os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def test_hub_scores(self):
        # doc1 has incoming link from doc2, doc3 has none
        hubs = self.router._compute_hub_scores(["doc1", "doc2", "doc3"])
        assert hubs["doc1"] >= hubs["doc3"]

    def test_context_expansion(self):
        results = self.router.search_with_context("content", limit=3)
        for r in results:
            assert "incoming_links" in r
            assert "outgoing_links" in r
            assert "ancestry" in r
            assert "siblings" in r
            assert r["chunk"]["chunk_text"]
            assert r["document"] is not None

    def test_ancestry(self):
        ancestry = self.router._get_ancestry("doc1")
        # doc1 is in root, so ancestry should be empty or root
        assert isinstance(ancestry, list)

    def test_siblings(self):
        siblings = self.router._get_siblings("doc1")
        # doc2 and doc3 are siblings of doc1
        sibling_ids = [s["id"] for s in siblings]
        assert "doc2" in sibling_ids or "doc3" in sibling_ids

    def test_hub_reranking(self):
        results = self.router.search_chunks_with_hub_score("content", limit=3)
        for r in results:
            assert "hub_score" in r
            assert "final_score" in r
            assert 0 <= r["hub_score"] <= 1
```

### 11.5 `tests/test_integration.py`

```python
class TestEndToEnd:
    def test_import_and_search_chunks(self):
        """Full pipeline: import → chunk → embed → search."""
        import tempfile
        tmpdir = tempfile.mkdtemp()
        try:
            (Path(tmpdir) / "test.md").write_text(
                "---\ntype: note\ntitle: Test\n---\n"
                "This is a test document about machine learning.\n\n"
                "It has multiple paragraphs for chunking."
            )
            with OKFRouter(":memory:", tmpdir, enable_chunking=True) as router:
                router.import_from_okf(Path(tmpdir / "test.md"))
                results = router.search_chunks("machine learning", limit=5)
                assert len(results) > 0
                for r in results:
                    doc = router.get_by_id(r["parent_doc_id"])
                    assert doc is not None
                    assert "machine learning" in doc.body.lower()
        finally:
            shutil.rmtree(tmpdir)

    def test_graph_context_in_search(self):
        """Search with context returns graph enrichment."""
        import tempfile
        tmpdir = tempfile.mkdtemp()
        try:
            (Path(tmpdir) / "a.md").write_text("---\ntype: note\ntitle: A\n---\nA content.")
            (Path(tmpdir) / "b.md").write_text("---\ntype: note\ntitle: B\n---\nB content. [A](a.md)")
            with OKFRouter(":memory:", tmpdir, enable_chunking=True) as router:
                router.import_from_okf(Path(tmpdir / "a.md"))
                router.import_from_okf(Path(tmpdir / "b.md"))
                results = router.search_with_context("content", limit=3)
                for r in results:
                    assert r["chunk"]["chunk_text"]
                    assert r["document"].body
        finally:
            shutil.rmtree(tmpdir)

    def test_reconstruction_preserves_structure(self):
        """Reconstruction produces valid markdown."""
        import tempfile
        tmpdir = tempfile.mkdtemp()
        try:
            body = "# Title\n\nPara 1.\n\n- list item\n\n> quote\n\n`code`."
            (Path(tmpdir) / "test.md").write_text(body)
            with OKFRouter(":memory:", tmpdir, enable_chunking=True) as router:
                router.import_from_okf(Path(tmpdir / "test.md"))
                reconstructed = router.reconstruct_document("test")
                assert "Title" in reconstructed
                assert "Para 1" in reconstructed
        finally:
            shutil.rmtree(tmpdir)
```

### 11.6 Test fixtures

```
tests/fixtures/
├── bundle/
│   ├── index.md          (root index, links to all docs)
│   ├── ml.md             (machine learning, linked by dl.md)
│   ├── dl.md             (deep learning, links to ml.md)
│   ├── nlp.md            (NLP, links to ml.md)
│   └── sub/
│       └── transformers.md (linked by dl.md)
```

## 12. Implementation Phases

| Phase | What | Depends On | Effort | Status |
|-------|------|-----------|--------|--------|
| **0** | Mordant chunker — bare chunks, `get_chunks()`, `get_all_chunks()`, `get_chunks_with_context()`, `get_delimiter()`, `compute_overlap_payloads()` | — | ✅ Done | **Mordant compiled, 1198 tests passing** |
| **1** | Schema + `_split_into_chunks` + `_compute_overlap_payloads` | Phase 0 | 0.5 day | ⬜ |
| **2** | Chunked ingestion in `import_from_okf` + `import_bundle` | Phase 1 | 1 day | ⬜ |
| **3** | `search_chunks` + updated `search_hybrid` | Phase 2 | 1 day | ⬜ |
| **4** | Graph enrichment (`hub_scores`, `search_with_context`, helpers) | Phase 3 | 1 day | ⬜ |
| **5** | CLI updates + shell commands | Phase 4 | 0.5 day | ⬜ |
| **6** | Tests (all files) | All phases | 1 day | ⬜ |

**Total: ~6 days** (Phase 0 complete)

## 13. Open Topics

| Topic | Options | Recommendation |
|-------|---------|---------------|
| Chunk size default | 256, 512, 1024 words | **512** — balances granularity vs context |
| Overlap default | 32, 64, 128 words | **64** — bridges sentence boundaries |
| Hub weight in reranking | 0.1, 0.2, 0.3 | **0.3** — significant but not dominant |
| Context hops default | 1, 2, 3 | **1** — 2 explodes context |
| Chunking toggle | `--no-chunking` flag vs `--chunk-size 0` | **Both** — flag is clearer |
| Migration for existing DBs | Auto-migrate or manual? | **Manual** — `reindex` command rebuilds indexes; users re-import for chunks |
| Chunk deletion on re-import | Delete all old chunks? | **Yes** — `MATCH (c)-[:PART_OF]->(ch:Chunk) DELETE ch` before insert |

## 14. File Change Summary

| File | Changes |
|------|---------|
| `router.py` | Chunk schema, `_split_into_chunks`, `_compute_overlap_payloads`, `reconstruct_document`, `_get_delimiter`, `search_chunks`, updated `search_hybrid`, `_compute_hub_scores`, `search_with_context`, `_get_ancestry`, `_get_siblings`, updated `import_from_okf`, updated `import_bundle`, updated `__init__`, updated `_index_specs` |
| `models.py` | Add `ChunkModel` |
| `tools.py` | Add `search_chunks`, `search_with_context` tool definitions |
| `cli.py` | Add `--chunk-size`, `--chunk-overlap`, `--no-chunking`; new shell commands |
| `tests/test_chunking.py` | NEW |
| `tests/test_reconstruction.py` | NEW |
| `tests/test_chunk_search.py` | NEW |
| `tests/test_graph_enrichment.py` | NEW |
| `tests/test_integration.py` | NEW |
| `tests/fixtures/bundle/` | NEW — sample docs with cross-links |
