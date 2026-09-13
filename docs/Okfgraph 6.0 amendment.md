> **Historical design record — do not follow.** Describes the pre-0.2.x
> optimum/transformers embedding stack and/or the in-tree RapidAI ingest
> engine, neither of which exists anymore. The authoritative surface is
> `architecture.md` v6.0 (as-built for okfgraph 0.2.12: external
> `embroider` crate, bobine converter seam, 5 MCP tools). Kept for
> archaeology, not guidance.

# Response to V6.0 Architecture Review

This is an exceptionally sharp review. Every point lands, and several caught issues that would have surfaced as runtime failures or silent data corruption in Phase 1. Below is a point-by-point response with concrete remediation.

---

## 1. Critical Architectural Feedback

### 1A. The Smart Diff Contradiction — **Accepted in full**

You are correct. The Decision Matrix row for "Typo in body (same meaning)" is logically impossible under the stated implementation. The reasoning chain:

1. Typo changes `new_text`
2. `build_search_text(new_text, new_meta)` produces different output
3. `sha256(different_output) != current_vec_rev.source_text_hash`
4. Gate fails → full re-encode triggers

The only way to achieve "typo immunity" would be a lossy normalization pass in `build_search_text()` (stemming, case-folding, punctuation stripping), which would introduce its own failure modes: "colour" vs "color" would hash identically, but "form" vs "from" would not, making the boundary arbitrary and untestable.

**Remediation — Corrected Decision Matrix:**

| Change Type | New TextRevision? | New VectorRevision? | Re-encode? | New ChunkRevisions? | Stale edges expired? |
|---|---|---|---|---|---|
| File unchanged (hash match) | — | — | — | — | — |
| Frontmatter / tags only | ✅ | ❌ (search-text hash match) | ❌ | ❌ | ❌ |
| **Any body text change (incl. typo)** | ✅ | ✅ | ✅ | ✅ | ✅ |
| Paragraph added / removed | ✅ | ✅ | ✅ | ✅ | ✅ |
| Full rewrite | ✅ | ✅ | ✅ | ✅ | ✅ |

The "typo immunity" row is **removed**. The two-level hash still provides value: frontmatter-only changes (tags, title metadata) skip the encoder because `build_search_text()` excludes non-semantic frontmatter keys. But any change to the body text — including a single character — invalidates both the document vector and all chunk vectors.

**Corrected `smart_upsert` Stage 2 comment:**

```python
# ── STAGE 2: semantic diff (vector gate) ──
# Gates on the ASSEMBLED SEARCH TEXT hash, not the raw file hash.
# Only frontmatter-only changes (tags, title metadata) pass this gate.
# ANY body text change — including a typo fix — invalidates the vector.
new_search_text = self.build_search_text(new_text, new_meta)
new_text_hash   = sha256(new_search_text.encode()).hexdigest()

if current_vec_rev.source_text_hash == new_text_hash:
    # Only non-semantic frontmatter changed → REUSE the vector.
    # Chunks remain valid (their text is unchanged).
    return
```

**On decoupling ChunkRevision text from ChunkRevision embeddings:** Rejected for v6.0. The added schema complexity (a `ChunkEmbedding` table with its own lifecycle, a `CHUNK_HAS_EMBEDDING` rel table, and a second compaction trigger) is disproportionate to the savings. A typo fix re-encoding 5–15 chunks is ~200ms on CPU. The vertical partitioning win is at the *document* level (one 1024-dim vector vs. N chunk vectors), and that's preserved.

---

### 1B. Bitemporal Completeness (`tx_id`) — **Accepted**

The DDL is corrected below. `tx_id` is added to `VectorRevision` and `ChunkRevision`:

```cypher
CREATE NODE TABLE VectorRevision (
    vec_id           STRING PRIMARY KEY,
    concept_id       STRING,
    embedding        FLOAT[dim],
    valid_from       TIMESTAMP,
    valid_to         TIMESTAMP,
    tx_id            STRING,              -- ← ADDED: import transaction correlation
    source_text_hash STRING,
    model_id         STRING,
    dim              INTEGER
);

CREATE NODE TABLE ChunkRevision (
    chunk_rev_id   STRING PRIMARY KEY,
    concept_id     STRING,
    vec_rev_id     STRING,
    chunk_index    INTEGER,
    chunk_text     STRING,
    block_type     STRING,
    start_offset   INTEGER,
    end_offset     INTEGER,
    embedding      FLOAT[dim],
    valid_from     TIMESTAMP,
    valid_to       TIMESTAMP,
    tx_id          STRING                 -- ← ADDED: import transaction correlation
);
```

This enables the transaction-time query pattern across all revision types:

```cypher
-- "What did transaction X write?"
MATCH (tr:TextRevision {tx_id: $tx})
OPTIONAL MATCH (vr:VectorRevision {tx_id: $tx})
OPTIONAL MATCH (cr:ChunkRevision {tx_id: $tx})
RETURN tr.concept_id, tr.title,
       vr.vec_id, vr.model_id,
       collect(cr.chunk_index) AS chunks_written;
```

---

### 1C. Archive Tables & P6 — **Accepted**

A new subsection is added to the schema:

```cypher
-- ════════════════════════════════════════════════════════════
-- 2.7 ARCHIVE SCHEMA: identical structure, no indexes.
-- Populated by `okf compact`. Queried only for forensic/audit.
-- ════════════════════════════════════════════════════════════
CREATE NODE TABLE VectorRevisionArchive (
    vec_id           STRING PRIMARY KEY,
    concept_id       STRING,
    embedding        FLOAT[dim],
    valid_from       TIMESTAMP,
    valid_to         TIMESTAMP,
    tx_id            STRING,
    source_text_hash STRING,
    model_id         STRING,
    dim              INTEGER,
    archived_at      TIMESTAMP            -- when compaction moved this row
);

CREATE NODE TABLE ChunkRevisionArchive (
    chunk_rev_id   STRING PRIMARY KEY,
    concept_id     STRING,
    vec_rev_id     STRING,
    chunk_index    INTEGER,
    chunk_text     STRING,
    block_type     STRING,
    start_offset   INTEGER,
    end_offset     INTEGER,
    embedding      FLOAT[dim],
    valid_from     TIMESTAMP,
    valid_to       TIMESTAMP,
    tx_id          STRING,
    archived_at    TIMESTAMP
);

CREATE NODE TABLE SemanticEdgeArchive (
    id           STRING PRIMARY KEY,
    edge_type    STRING,                  -- LinkEdge|SimilarityEdge|CausalEdge|...
    layer        STRING,
    kind         STRING,
    context      STRING,
    confidence   DOUBLE,
    learned_at   TIMESTAMP,
    expired_at   TIMESTAMP,
    archived_at  TIMESTAMP
);
-- No vector indexes. No FTS indexes. Cold storage only.
```

The `SemanticEdgeArchive` uses a single table with an `edge_type` discriminator rather than five separate archive tables — the edge-specific columns (`similarity`, `property_name`) are sparse and can be stored in a `MAP(STRING, STRING)` payload column if needed. This keeps compaction logic to one `INSERT INTO ... SELECT` pattern.

---

## 2. Schema & Query Corrections

### 2A. Stripped Markdown Operators — **Fixed**

The original source had `<=` and `>` which were consumed by markdown rendering. The canonical query is now fenced and escaped:

```cypher
-- Section 7.1: Point-in-Time Graph State
WITH timestamp('2026-07-01T00:00:00') AS t
MATCH (src:ConceptIdentity)-[:CONNECTS]->(edge)-[:BINDS]->(tgt:ConceptIdentity)
WHERE edge.learned_at <= t
  AND (edge.expired_at IS NULL OR edge.expired_at > t)
  AND src.created_at <= t
  AND (src.tombstone_at IS NULL OR src.tombstone_at > t)
  AND tgt.created_at <= t
  AND (tgt.tombstone_at IS NULL OR tgt.tombstone_at > t)
RETURN src.id, labels(edge)[0] AS rel, edge.kind, tgt.id;
```

All temporal comparison queries in the spec will be audited for stripped operators.

---

### 2B. Inconsistent Property Naming — **Fixed**

This was a markdown rendering artifact: underscores in `learned_at` / `expired_at` were interpreted as italic markers and stripped. All occurrences are corrected:

| Location | Broken | Fixed |
|---|---|---|
| §4 Multi-hop chain | `e1.expiredat IS NULL` | `e1.expired_at IS NULL` |
| §4 Multi-hop chain | `e2.expiredat IS NULL` | `e2.expired_at IS NULL` |
| §7.2 Belief Revision | `e.learnedat, e.expiredat` | `e.learned_at, e.expired_at` |
| §5.1 ingest_thoughts | `learnedat: timestamp($now), expiredat: NULL` | `learned_at: timestamp($now), expired_at: NULL` |
| §7.3 expire_stale_edges | `s.expired_at IS NULL` | *(already correct)* |

The spec source will use fenced code blocks for all Cypher/Python to prevent recurrence.

---

### 2C. Missing `layer` on Directory — **Accepted with caveat**

The reviewer's multi-root scenario (public vs. private workspace) is valid. However, this tensions with P1 ("mechanical, high-volume, zero-metadata"). The compromise:

```cypher
CREATE NODE TABLE Directory (
    id    STRING PRIMARY KEY,
    layer STRING DEFAULT 'core'          -- ← ADDED: prevents cross-tree contamination
);
```

Cost: one nullable STRING column, zero query overhead (columnar scan only fires when `WHERE d.layer = $layer` is present). Benefit: no schema migration when multi-tenancy arrives. The `CONTAINS` traversal gains an optional filter:

```cypher
MATCH (d:Directory {id: $dir_id})-[:CONTAINS]->(child)
WHERE d.layer = $layer OR $layer IS NULL
RETURN child.id, labels(child)[0] AS node_type;
```

---

## 3. Code & API Consistency

### 3A. Python Snake_case — **Fixed (rendering artifact)**

The underscores were stripped by the same markdown issue. The canonical `smart_upsert` signature and body:

```python
def smart_upsert(self, concept_id: str, new_text: str, new_meta: dict) -> None:
    """Vertical-partitioned upsert with vector deduplication."""
    identity = self.get_identity(concept_id)

    if identity is None:
        self.create_identity(concept_id, new_meta)
        text_rev = self.append_text_revision(concept_id, new_text, new_meta)
        vec_rev  = self.append_vector_revision(concept_id, new_text)
        self.append_chunk_revisions(concept_id, vec_rev, new_text)
        self.point_current(concept_id, text_rev, vec_rev)
        return

    current_text_rev = self.get_current_text(concept_id)
    current_vec_rev  = self.get_current_vector(concept_id)

    # ── STAGE 1: text diff ──
    text_changed = (
        current_text_rev.body  != new_text
        or current_text_rev.title != new_meta.get("title")
        or current_text_rev.tags  != new_meta.get("tags", [])
    )
    if not text_changed:
        return

    new_text_rev = self.append_text_revision(concept_id, new_text, new_meta)
    self.supersede(current_text_rev)
    self.repoint_current_text(concept_id, new_text_rev)

    # ── STAGE 2: semantic diff (vector gate) ──
    new_search_text = self.build_search_text(new_text, new_meta)
    new_text_hash   = sha256(new_search_text.encode()).hexdigest()

    if current_vec_rev.source_text_hash == new_text_hash:
        return  # frontmatter-only change; vector and chunks remain valid

    new_vec_rev = self.append_vector_revision(concept_id, new_text)
    self.supersede(current_vec_rev)
    self.repoint_current_vector(concept_id, new_vec_rev)

    self.expire_stale_edges(concept_id)
    self.supersede_chunks(concept_id)
    self.append_chunk_revisions(concept_id, new_vec_rev, new_text)
```

LLM Tool parameter names corrected:

```python
{
    "name": "get_concept_at",
    "parameters": {
        "properties": {
            "concept_id": {"type": "string"},
            "as_of": {"type": "string", "format": "date-time"}
        },
        "required": ["concept_id", "as_of"]
    }
}
```

---

### 3B. Directory Table Definition — **Acknowledged**

The single-property DDL is valid LadybugDB syntax. With the `layer` addition from §2C above, it now has two columns, which future-proofs it without requiring a migration.

---

## 4. Operational Considerations

### 4A. FTS IDF Pollution — **Accepted, documented**

Added to §11 (Index Lifecycle & Compaction):

> **Known behavior — BM25 score drift:** Superseded `TextRevision` and `ChunkRevision` rows remain in the FTS index until compaction. They contribute to IDF calculations, slightly deflating scores for terms that appear in outdated revisions. At the 3:1 compaction threshold, the maximum IDF distortion is bounded: a term appearing in 3 superseded rows + 1 current row has its IDF computed over 4 documents instead of 1, reducing the score by ~`log(4/1)` relative to a clean index. This is acceptable for ranking (relative order is preserved) but not for absolute score thresholds. Applications using hard FTS score cutoffs should trigger `okf compact` before query.

---

### 4B. Similarity Graph Eventual Consistency — **Accepted, documented**

Added to §9.2 (New Methods) under `materialize_similarities`:

> **Eventual consistency window:** When a concept's vector changes, `expire_stale_edges()` immediately severs its similarity connections. The concept appears **isolated** in graph traversal (`traverse()` via `CONNECTS`/`BINDS`) until the next `materialize_similarities()` batch run. This window is by design — real-time pairwise cosine computation at scale is prohibitively expensive.
>
> **Guidance for application developers:**
> - `search_hybrid()` (ANN) is unaffected — it queries the vector index directly, not the similarity graph.
> - `traverse()` with `relationship='SimilarityEdge'` will return empty results for recently-modified concepts.
> - If real-time "similar to X" is required, fall back to an inline ANN query: `QUERY_VECTOR_INDEX('VectorRevision', ...)` filtered by `valid_to IS NULL`.
> - Schedule `materialize_similarities()` as a post-ingestion step in batch pipelines.

---

## 5. Summary of Spec Patches

| # | Section | Change | Severity |
|---|---|---|---|
| 1 | §3.1 Decision Matrix | Remove "typo immunity" row; any body change → re-encode | **Critical** (logical contradiction) |
| 2 | §2.1 DDL | Add `tx_id` to `VectorRevision`, `ChunkRevision` | **High** (bitemporal completeness) |
| 3 | §2.7 (new) | Archive table DDL | **Medium** (P6 integrity) |
| 4 | §7.1, §4, §7.2, §5.1 | Fix stripped `<=`, `>`, `_` characters | **High** (runtime failures) |
| 5 | §2.1 DDL | Add `layer STRING` to `Directory` | **Low** (future-proofing) |
| 6 | All Python/Cypher | Enforce fenced code blocks; audit snake_case | **Medium** (consistency) |
| 7 | §11 | Document BM25 IDF pollution behavior | **Low** (operational awareness) |
| 8 | §9.2 | Document similarity eventual consistency window | **Medium** (developer guidance) |

---

## 6. Implementation Impact

None of these corrections change the Phase 1 scope or effort estimate. The `tx_id` additions are two extra STRING columns in the DDL. The Decision Matrix correction *simplifies* the implementation (no need to reason about "semantic equivalence" of text changes — the hash is the truth). The archive schema is Phase 5 work regardless.

The spec is ready for Phase 1 with these patches applied. Thank you for the review — the Smart Diff contradiction in particular would have produced a very confusing bug report three weeks into implementation.