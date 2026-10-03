> **Historical design record — do not follow.** Describes the pre-0.2.x
> optimum/transformers embedding stack and/or the in-tree RapidAI ingest
> engine, neither of which exists anymore. The authoritative surface is
> `architecture.md` v6.0 (as-built for okfgraph 0.2.12: external
> `embroider` crate, bobine converter seam, 5 MCP tools). Kept for
> archaeology, not guidance.

OKF Knowledge Graph — Architecture Specification

Version: 6.0 (Greenfield — Semantic Spacetime Schema)
Status: Proposed
Date: 2026-07-26
Supersedes: v5.9 flat Concept schema · ADR 001 (merged herein)
Verified against: LadybugDB v0.17.1, Python 3.13.14
Compatibility: None required. v6.0 is a clean break. No migration path from v5.x is provided; existing databases must be re-ingested from their OKF bundles.

Foreword — Why v6.0 Is a Clean Break

Three constraints that shaped v5.x have been removed:

| Constraint (v5.x) | Reality (v6.0) | Consequence |
|---|---|---|
| ~10K-concept scale ceiling | No ceiling | Schema must behave identically at 10K, 100K, 1M concepts |
| Backward compatibility with deployed DBs | None — still in architectural build phase | No migration machinery. Design the ontology once, correctly. |
| Stable, frozen relationship vocabulary | Code is flexible to grow | New relation types must be data, not schema |

v6.0 therefore merges two previously-separate proposals into a single greenfield design:

ADR 001 — Temporal Ledger: vertical partitioning of identity, text, and vectors into independent temporal lifecycles, with smart-diff vector deduplication.
Semantic Spacetime — bipartite edge-node architecture: semantic relationships are first-class nodes with their own type, provenance, confidence, layer, and temporal validity, wired through two polymorphic relationship tables.

The result is a graph where every fact lives in space and time: space (what relates to what, at what abstraction layer, with what confidence) and time (when it was learned, when it expired).

Design Principles

| # | Principle | Statement |
|---|---|---|
| P1 | Bipartite for semantic, direct for structural | Knowledge-bearing relationships (links, similarity, causality, derivation, property) become edge nodes. Mechanical relationships (directory containment, chunk decomposition, image attachment) stay single-hop. |
| P2 | Every node and edge lives in time | learnedat / expiredat on all semantic nodes. validfrom / validto on revisions. Temporal validity is never optional. |
| P3 | Layer is a first-class axis | core → domain → instance → meta on every entity. Search, traversal, and LLM tools filter by layer. This is the primary noise-control mechanism at scale. |
| P4 | Edge nodes are the extension point | A new relation type is a new kind value, or a new edge-node table added to the CONNECTS/BINDS union. No per-feature CREATE REL TABLE dispatch in application code. |
| P5 | Vectors are partitioned from text | Identity, text, and vectors have independent temporal lifecycles. A typo fix never duplicates a vector. |
| P6 | The schema IS the ontology | CREATE NODE TABLE / CREATE REL TABLE statements are the formal specification of what exists. No side-channel conventions. |

Database Schema (LadybugDB)
cypher
-- Extensions
INSTALL vector;
LOAD vector;
INSTALL fts;
LOAD fts;

2.1. Entity Layer
cypher
-- ════════════════════════════════════════════════════════════
-- IDENTITY: immutable anchor for every concept.
-- Created once. Mutated only on tombstone.
-- ════════════════════════════════════════════════════════════
CREATE NODE TABLE ConceptIdentity (
    id           STRING PRIMARY KEY,    -- stable ID (relative path sans .md)
    type         STRING,                -- "chapter"|"section"|"thought"|"definition"|...
    layer        STRING,                -- "core"|"domain"|"instance"|"meta"
    created_at   TIMESTAMP,             -- when the concept first entered the graph
    tombstone_at TIMESTAMP,             -- soft-delete marker (NULL = alive)
    extra        MAP(STRING, STRING)    -- arbitrary OKF frontmatter keys
);

-- ════════════════════════════════════════════════════════════
-- TEXT REVISIONS: append-only, one row per textual edit.
-- Lightweight — no vector payload.
-- ════════════════════════════════════════════════════════════
CREATE NODE TABLE TextRevision (
    rev_id      STRING PRIMARY KEY,     -- UUID7 (time-ordered)
    concept_id  STRING,                 -- → ConceptIdentity.id
    title       STRING,
    description STRING,
    resource    STRING,
    tags        STRING[],
    body        STRING,
    valid_from  TIMESTAMP,              -- when this text became effective
    valid_to    TIMESTAMP,              -- when superseded (NULL = current)
    tx_id       STRING,                 -- import transaction correlation
    source_hash STRING                  -- SHA-256 of the source .md file
);

-- ════════════════════════════════════════════════════════════
-- VECTOR REVISIONS: append-only, one row ONLY when semantic
-- content actually changes (smart diff gates the write).
-- ════════════════════════════════════════════════════════════
CREATE NODE TABLE VectorRevision (
    vec_id           STRING PRIMARY KEY,-- UUID7
    concept_id       STRING,            -- → ConceptIdentity.id
    embedding        FLOAT[dim],        -- Jina v5 (Matryoshka 32–1024)
    valid_from       TIMESTAMP,
    valid_to         TIMESTAMP,
    sourcetexthash STRING,            -- SHA-256 of assembled search text
    model_id         STRING,            -- encoder provenance
    dim              INTEGER            -- actual dimension used
);

-- ════════════════════════════════════════════════════════════
-- CHUNK REVISIONS: append-only, tied to a vector epoch.
-- ════════════════════════════════════════════════════════════
CREATE NODE TABLE ChunkRevision (
    chunkrevid STRING PRIMARY KEY,
    concept_id   STRING,                -- → ConceptIdentity.id
    vecrevid   STRING,                -- → VectorRevision.vec_id
    chunk_index  INTEGER,
    chunk_text   STRING,
    block_type   STRING,                -- paragraph|heading|code|list|blockquote|table|diagram
    start_offset INTEGER,
    end_offset   INTEGER,
    embedding    FLOAT[dim],
    valid_from   TIMESTAMP,
    valid_to     TIMESTAMP
);

-- ════════════════════════════════════════════════════════════
-- IMAGE ASSETS: binary storage + unified vector space.
-- Content-addressed; not text-versioned.
-- ════════════════════════════════════════════════════════════
CREATE NODE TABLE ImageAsset (
    id           STRING PRIMARY KEY,
    file_name    STRING,
    mime_type    STRING,
    alt_text     STRING,
    caption      STRING,                -- provenance: how the image was embedded
    embed_route  STRING,                -- "text" | "omni"
    content_hash STRING,                -- SHA-256 change-detection key
    data         BLOB,
    embedding    FLOAT[dim],            -- shared text/omni vector space
    learned_at   TIMESTAMP,
    expired_at   TIMESTAMP
);

-- ════════════════════════════════════════════════════════════
-- DIRECTORY: structural hierarchy, derived from concept IDs.
-- ════════════════════════════════════════════════════════════
CREATE NODE TABLE Directory (
    id STRING PRIMARY KEY
);

-- ════════════════════════════════════════════════════════════
-- DELTA DETECTION (carried from v5.1 / v5.5)
-- ════════════════════════════════════════════════════════════
CREATE NODE TABLE FileHash (
    path       STRING PRIMARY KEY,
    hash       STRING,                  -- SHA-256 of file contents
    concept_id STRING                   -- maps path → ConceptIdentity.id
);

CREATE NODE TABLE DirHash (
    path STRING PRIMARY KEY,
    hash STRING
);

-- ════════════════════════════════════════════════════════════
-- META: key/value store for epochs, schema version, counters.
-- ════════════════════════════════════════════════════════════
CREATE NODE TABLE Meta (
    key   STRING PRIMARY KEY,
    value STRING
);

2.2. Semantic Edge Nodes (Bipartite)

Five edge-node tables. Each is a first-class citizen with its own properties, temporal validity, and layer. Together they cover the four fundamental operations of a knowledge space (proximity, topology, state, causality) plus multi-source provenance.
cypher
-- ── LINK: explicit or inferred references between concepts ──
CREATE NODE TABLE LinkEdge (
    id         STRING PRIMARY KEY,
    layer      STRING,                  -- core|domain|instance|meta
    kind       STRING,                  -- explicit|inferred|see-also|cited-by
    context    STRING,                  -- "concepts/intro.md:L42"|"auto:export"|"auto:similarity"
    confidence DOUBLE,                  -- 1.0 for explicit markdown links, cosine score for inferred
    learned_at TIMESTAMP,
    expired_at TIMESTAMP
);

-- ── SIMILARITY: pre-computed semantic proximity ──
CREATE NODE TABLE SimilarityEdge (
    id         STRING PRIMARY KEY,
    layer      STRING,
    kind       STRING,                  -- semantic|structural|functional
    context    STRING,                  -- model + dim that produced the score
    similarity DOUBLE,                  -- [0.0 … 1.0] cosine
    model_id   STRING,                  -- encoder provenance
    learned_at TIMESTAMP,
    expired_at TIMESTAMP
);

-- ── CAUSAL: one concept/event leads to another ──
CREATE NODE TABLE CausalEdge (
    id         STRING PRIMARY KEY,
    layer      STRING,
    kind       STRING,                  -- causal|sequential|inferential|temporal
    context    STRING,                  -- "reasoning-chain-"|"project:alpha"
    confidence DOUBLE,
    learned_at TIMESTAMP,
    expired_at TIMESTAMP
);

-- ── PROPERTY: a concept carries a relational attribute ──
CREATE NODE TABLE PropertyEdge (
    id            STRING PRIMARY KEY,
    layer         STRING,
    kind          STRING,               -- intrinsic|relational|derived
    property_name STRING,               -- "initiated-by"|"version"|"status"
    learned_at    TIMESTAMP,
    expired_at    TIMESTAMP
);

-- ── DERIVATION: hyperedge for multi-source provenance ──
CREATE NODE TABLE DerivationEdge (
    id         STRING PRIMARY KEY,
    layer      STRING,
    kind       STRING,                  -- synthesis|summarization|extraction|reasoning
    context    STRING,                  -- "ingest_thoughts"|"pdf:paper.pdf:pages=3-7"
    learned_at TIMESTAMP,
    expired_at TIMESTAMP
);

2.3. Polymorphic Bipartite Rel Tables

Two tables enforce the bipartite constraint for all semantic relationships. Adding a future edge type (e.g. ContradictsEdge) means adding it to the union — one DDL statement, zero application dispatch changes.
cypher
-- Every departure from an entity goes through CONNECTS
CREATE REL TABLE CONNECTS (
    FROM ConceptIdentity
    TO   LinkEdge | SimilarityEdge | CausalEdge | PropertyEdge | DerivationEdge
);

-- Every arrival at an entity goes through BINDS
CREATE REL TABLE BINDS (
    FROM LinkEdge | SimilarityEdge | CausalEdge | PropertyEdge | DerivationEdge
    TO   ConceptIdentity
);

The pattern (ConceptIdentity)-[:CONNECTS]->(Edge)-[:BINDS]->(ConceptIdentity) is the universal traversal idiom. Entity-to-entity semantic hops cannot exist — the schema rejects them at write time.

2.4. Structural Rel Tables (Direct — Not Bipartite)

Mechanical, high-volume, zero-metadata. Bipartite indirection would add a hop for no expressive gain.
cypher
-- Directory hierarchy
CREATE REL TABLE CONTAINS (
    FROM Directory TO Directory,
    FROM Directory TO ConceptIdentity
);

-- Document → chunk decomposition
CREATE REL TABLE PART_OF (
    FROM ConceptIdentity TO ChunkRevision
);

-- Concept → image attachment
CREATE REL TABLE INCLUDES_ASSET (
    FROM ConceptIdentity TO ImageAsset
);

-- Materialized pointers for O(1) current-state reads
CREATE REL TABLE HASCURRENTTEXT   (FROM ConceptIdentity TO TextRevision);
CREATE REL TABLE HASCURRENTVECTOR (FROM ConceptIdentity TO VectorRevision);

-- Audit-trail pointers (all historical revisions)
CREATE REL TABLE HASTEXTHISTORY   (FROM ConceptIdentity TO TextRevision);
CREATE REL TABLE HASVECTORHISTORY (FROM ConceptIdentity TO VectorRevision);

2.5. Indexes
cypher
CALL CREATEVECTORINDEX(
    'VectorRevision', 'vecembeddingidx', 'embedding',
    mu := 30, ml := 60, metric := 'cosine', efc := 200
);

CALL CREATEVECTORINDEX(
    'ChunkRevision', 'chunkembeddingidx', 'embedding',
    mu := 30, ml := 60, metric := 'cosine', efc := 200
);

CALL CREATEVECTORINDEX(
    'ImageAsset', 'imageomniidx', 'embedding',
    mu := 30, ml := 60, metric := 'cosine', efc := 200
);

CALL CREATEFTSINDEX('TextRevision',  'text_fts',  ['title', 'description', 'body']);
CALL CREATEFTSINDEX('ChunkRevision', 'chunkfts', ['chunktext']);

2.6. Matryoshka Dimensions (Carried)

Both jinaai/jina-embeddings-v5-text-small-retrieval and -omni-small-retrieval support truncation at 1024 / 768 / 512 (default) / 256 / 128 / 64 / 32. Last-token pooling + L2 re-normalisation after truncation (see v5.9 §4.3).

Smart Diff — Vector Deduplication

The application layer performs a two-stage diff before writing. Heavy vectors are duplicated only when the semantic content of a document actually changes.
python
def smartupsert(self, conceptid: str, newtext: str, new_meta: dict) -> None:
    """Vertical-partitioned upsert with vector deduplication."""

    identity = self.getidentity(concept_id)

    if identity is None:
        # ── NEW CONCEPT: write all three layers ──
        self.createidentity(conceptid, newmeta)
        textrev = self.appendtextrevision(conceptid, newtext, new_meta)
        vecrev  = self.appendvectorrevision(conceptid, newtext)
        self.appendchunkrevisions(conceptid, vecrev, newtext)
        self.pointcurrent(conceptid, textrev, vec_rev)
        return

    currenttextrev = self.getcurrenttext(conceptid)
    currentvecrev  = self.getcurrentvector(conceptid)

    # ── STAGE 1: text diff ──
    text_changed = (
        currenttextrev.body  != new_text
        or currenttextrev.title != new_meta.get("title")
        or currenttextrev.tags  != new_meta.get("tags", [])
    )
    if not text_changed:
        return  # file hash matched; nothing to do

    newtextrev = self.appendtextrevision(conceptid, newtext, newmeta)
    self.supersede(currenttextrev)              # SET validto = now()
    self.repointcurrenttext(conceptid, newtextrev)

    # ── STAGE 2: semantic diff (vector gate) ──
    newsearchtext = self.buildsearchtext(newtext, new_meta)
    newtexthash   = sha256(newsearchtext.encode()).hexdigest()

    if currentvecrev.sourcetexthash == newtexthash:
        # Only frontmatter/tags changed → REUSE the vector. No re-embed.
        return

    newvecrev = self.appendvectorrevision(conceptid, new_text)
    self.supersede(currentvecrev)               # SET validto = now()
    self.repointcurrentvector(conceptid, newvecrev)

    # Vector changed → expire stale semantic edges, re-chunk
    self.expirestaleedges(conceptid)
    self.supersedechunks(concept_id)
    self.appendchunkrevisions(conceptid, newvecrev, new_text)

3.1. Diff Decision Matrix

| Change Type | New TextRevision? | New VectorRevision? | Re-encode? | Stale edges expired? |
|---|---|---|---|---|
| File unchanged (hash match) | — | — | — | — |
| Typo in body (same meaning) | ✅ | ❌ (hash match) | ❌ | ❌ |
| Frontmatter / tags only | ✅ | ❌ | ❌ | ❌ |
| Paragraph added / removed | ✅ | ✅ | ✅ | ✅ |
| Full rewrite | ✅ | ✅ | ✅ | ✅ |

Two-level hashing: FileHash.hash (raw file bytes) gates the whole pipeline; VectorRevision.sourcetexthash (assembled search text) gates only the expensive ONNX encode.

Universal Traversal Idiom

Every semantic query reduces to one pattern:
cypher
-- What does concept X relate to, and how?
MATCH (src:ConceptIdentity {id: $id})-[:CONNECTS]->(edge)-[:BINDS]->(tgt:ConceptIdentity)
WHERE edge.expired_at IS NULL
RETURN tgt.id, tgt.type, tgt.layer,
       labels(edge)[0] AS relation_type,
       edge.kind       AS relation_kind,
       edge.context    AS provenance,
       edge.learned_at AS established;

Filtered variants:
cypher
-- Only high-confidence explicit links
MATCH (src:ConceptIdentity)-[:CONNECTS]->(e:LinkEdge)-[:BINDS]->(tgt:ConceptIdentity)
WHERE e.kind = 'explicit' AND e.confidence >= 0.9 AND e.expired_at IS NULL
RETURN src.id, tgt.id, e.context;

-- Multi-hop inferential chain (reasoning reconstruction)
MATCH (a:ConceptIdentity)-[:CONNECTS]->(e1:CausalEdge)-[:BINDS]->
      (b:ConceptIdentity)-[:CONNECTS]->(e2:CausalEdge)-[:BINDS]->
      (c:ConceptIdentity)
WHERE e1.kind = 'inferential' AND e2.kind = 'inferential'
  AND e1.expiredat IS NULL AND e2.expiredat IS NULL
RETURN a.id, e1.context, b.id, e2.context, c.id;

Structural queries stay single-hop:
cypher
-- Directory listing (fast, unchanged)
MATCH (d:Directory {id: $dir_id})-[:CONTAINS]->(child)
RETURN child.id, labels(child)[0] AS node_type;

-- Chunks for a concept (fast, unchanged)
MATCH (c:ConceptIdentity {id: $id})-[:PART_OF]->(ch:ChunkRevision)
WHERE ch.valid_to IS NULL
RETURN ch.chunkindex, ch.chunktext, ch.block_type
ORDER BY ch.chunk_index;

Hyperedges

5.1. Multi-Source Thought Synthesis

ingest_thoughts() records what the model was looking at when it reasoned:
python
def ingest_thoughts(self, thoughts: str, topic: str,
                    sourceconceptids: List[str],
                    tags: Optional[List[str]] = None) -> str:
    conceptid = self.createthoughtconcept(thoughts, topic, tags)

    deriv_id = f"deriv:{uuid7()}"
    self._exec("""
        CREATE (d:DerivationEdge {
            id: $did, layer: 'instance', kind: 'reasoning',
            context: 'ingest_thoughts',
            learnedat: timestamp($now), expiredat: NULL
        })
    """, {"did": derivid, "now": now()})

    for srcid in sourceconcept_ids:
        self._exec("""
            MATCH (s:ConceptIdentity {id: $sid}), (d:DerivationEdge {id: $did})
            CREATE (s)-[:CONNECTS]->(d)
        """, {"sid": srcid, "did": derivid})

    self._exec("""
        MATCH (d:DerivationEdge {id: $did}), (t:ConceptIdentity {id: $tid})
        CREATE (d)-[:BINDS]->(t)
    """, {"did": derivid, "tid": conceptid})

    return concept_id

The arity of the DerivationEdge (how many sources it CONNECTS from) directly encodes the cardinality of the synthesis. Query:
cypher
MATCH (src:ConceptIdentity)-[:CONNECTS]->(d:DerivationEdge)-[:BINDS]->(t:ConceptIdentity {id: $id})
WHERE d.expired_at IS NULL
RETURN src.id, d.kind, d.context, d.learned_at;

5.2. PDF Ingestion Provenance
cypher
-- "This concept was extracted from pages 3–7 of paper.pdf"
CREATE (d:DerivationEdge {
    id: 'deriv:pdf-paper-p3-7',
    layer: 'instance', kind: 'extraction',
    context: 'pdf:paper.pdf:pages=3-7:routing=surgical',
    learned_at: timestamp('2026-07-26T10:00:00'),
    expired_at: NULL
});

5.3. N-ary Events (Future)

If okfgraph ingests meeting notes or event descriptions, one edge node binds all participants. No schema change required — arity is expressed by the number of BINDS edges.

Metagraph — The Meta Layer

layer = 'meta' enables reasoning about the graph itself. The meta layer is opt-in — created only when provenance matters — and is excluded from default search.
python
def recordsimilarityprovenance(self, a: str, b: str, score: float,
                                 model_id: str, dim: int) -> None:
    """Materialize a SimilarityEdge AND record its provenance at the meta layer."""
    sim_id = f"sim:{uuid7()}"
    # … create SimilarityEdge, wire A -[:CONNECTS]-> edge -[:BINDS]-> B …

    metaid = f"meta:sim:{simid}"
    self._exec("""
        CREATE (m:ConceptIdentity {
            id: $mid, type: 'provenance', layer: 'meta',
            createdat: timestamp($now), tombstoneat: NULL,
            extra: MAP(['ref_edge', 'model', 'dim'], [$ref, $model, $dim])
        })
    """, {"mid": metaid, "now": now(), "ref": sim_id,
          "model": model_id, "dim": str(dim)})

Query — "how was this similarity computed?"
cypher
MATCH (m:ConceptIdentity {layer: 'meta'})
WHERE m.extra['refedge'] = $edgeid
RETURN m.id, m.extra;

Temporal Queries

7.1. Point-in-Time Graph State
cypher
WITH timestamp('2026-07-01T00:00:00') AS t
MATCH (src:ConceptIdentity)-[:CONNECTS]->(edge)-[:BINDS]->(tgt:ConceptIdentity)
WHERE edge.learned_at  t)
  AND src.created_at     t)
  AND tgt.created_at     t)
RETURN src.id, labels(edge)[0] AS rel, edge.kind, tgt.id;

7.2. Belief Revision History
cypher
-- How has the causal understanding of concept X evolved?
MATCH (src:ConceptIdentity {id: $id})-[:CONNECTS]->(e:CausalEdge)-[:BINDS]->(tgt)
RETURN e.id, e.kind, e.context, e.confidence,
       e.learnedat, e.expiredat, tgt.id AS target
ORDER BY e.learned_at;

Returns the full revision chain — initial belief (expired), refinements (expired), current belief (active).

7.3. Knowledge Decay — Expiring Stale Edges

When a concept's vector changes, its similarity edges become stale. They are expired, not deleted:
python
def expirestaleedges(self, conceptid: str) -> int:
    """Expire all semantic edges touching a concept whose vector just changed."""
    rows = self._exec("""
        MATCH (ci:ConceptIdentity {id: $cid})-[:CONNECTS]->(s:SimilarityEdge)
        WHERE s.expired_at IS NULL
        SET s.expired_at = timestamp($now)
        RETURN count(s) AS n
    """, {"cid": conceptid, "now": now()})
    return rows[0]["n"]

The old similarity remains queryable as history: "these concepts were 0.91 similar under the old text; the text changed on 2026-07-26."

7.4. Bitemporal Axes

| Axis | Columns | Answers |
|---|---|---|
| Valid time | validfrom / validto (revisions) | "What was the concept's effective state at T?" |
| Transaction time | tx_id + UUID7 ordering | "What did we record during transaction X?" |
| Edge lifespan | learnedat / expiredat (edges) | "When did we believe this relation held?" |

Layer Taxonomy

| Layer | Meaning | okfgraph Example |
|---|---|---|
| core | Universal primitives, foundational definitions | "What is a knowledge graph?", "RRF fusion" |
| domain | Domain-specific concepts | "Matryoshka truncation", "hybrid search" |
| instance | Concrete occurrences, logs, thoughts | "Import run 2026-07-26", a specific LLM reasoning trace |
| meta | Relations about relations, provenance | "This similarity was computed by Jina v5 @ dim=512" |

search_hybrid() gains a layer filter. Default excludes meta and instance for definitional queries, so an agent asking "what is X?" is not flooded with 47 thought logs.

OKFRouter API

9.1. Methods That Change

| Method | Change |
|---|---|
| insertconcept() | Creates ConceptIdentity + TextRevision + VectorRevision + LinkEdge nodes |
| batchextract_links() | Emits LinkEdge with kind='explicit', confidence=1.0, context='file:line' |
| searchhybrid() | Adds layer param; reads via HASCURRENTTEXT / HASCURRENT_VECTOR pointers |
| traverse() | Unified: CONNECTS/BINDS for semantic, CONTAINS for structural; relationship → edge_kind filter |
| repairlinks() | Sets expiredat on stale LinkEdge, creates a fresh one. No separate BrokenLink table. |
| ingest_thoughts() | Creates DerivationEdge hyperedge binding source concepts |
| export_bundle() | "See Also" links become LinkEdge with kind='see-also', context='auto:export' |
| findpath() | Traverses CONNECTS/BINDS with expiredat IS NULL |
| Hub-score reranking | SUM(edge.confidence) weighted by edge.kind, replacing uniform COUNT(*) |
| listbrokenlinks() | Queries LinkEdge whose target ConceptIdentity is absent or tombstoned |

9.2. New Methods
python
def getreasoningchain(self, thought_id: str) -> List[Dict]:
    """Follow CausalEdge (kind='inferential') chains from a thought concept."""

def getderivationsources(self, concept_id: str) -> List[Dict]:
    """Source concepts a synthesized concept was derived from (DerivationEdge)."""

def getconceptat(self, conceptid: str, asof: datetime) -> Optional[ConceptModel]:
    """Point-in-time reconstruction: what did this concept say at time T?"""

def getedgehistory(self, concept_id: str) -> List[Dict]:
    """All semantic edges touching a concept, active and expired."""

def materialize_similarities(self, threshold: float = 0.85,
                             limitperconcept: int = 5,
                             layer: str = "domain") -> int:
    """Batch job: pairwise cosine → SimilarityEdge nodes above threshold."""

def expirestaleedges(self, concept_id: str) -> int:
    """Expire semantic edges touching a concept whose vector just changed."""

def compact(self) -> Dict[str, int]:
    """Archive expired edges + superseded revisions to *Archive tables."""

9.3. Removed

| Removed | Replaced By |
|---|---|
| BrokenLink node table | LinkEdge with absent/tombstoned target + expired_at lifecycle |
| DeletedConcept table (v5.8) | ConceptIdentity.tombstone_at |
| Bare LINKS_TO rel table | LinkEdge node + CONNECTS/BINDS |
| PARTOF (Concept→Chunk) | PARTOF (ConceptIdentity→ChunkRevision) |

LLM Tool Definitions (21 tools)

Carried (16): searchhybrid, traverse, getbyid, listdirectory, searchimages, searchchunks, searchwithcontext, searchchunkswithhubscore, expandwithgraphcontext, getchunks, reconstructdocument, findpath, exportbundle, ingestmd, ingestthoughts, ingestpdf.

New (5):
python
{
    "name": "getreasoningchain",
    "description": "Reconstruct the inferential/causal chain that led to a thought concept.",
    "parameters": {"type": "object",
        "properties": {"thought_id": {"type": "string"}},
        "required": ["thought_id"]},
},
{
    "name": "getderivationsources",
    "description": "Find the source concepts a synthesized concept was derived from.",
    "parameters": {"type": "object",
        "properties": {"concept_id": {"type": "string"}},
        "required": ["concept_id"]},
},
{
    "name": "getconceptat",
    "description": "Retrieve a concept's state at a specific point in time (temporal query).",
    "parameters": {"type": "object",
        "properties": {"concept_id": {"type": "string"},
                       "as_of": {"type": "string", "format": "date-time"}},
        "required": ["conceptid", "asof"]},
},
{
    "name": "getedgehistory",
    "description": "Show all semantic relationships (active and expired) for a concept.",
    "parameters": {"type": "object",
        "properties": {"concept_id": {"type": "string"}},
        "required": ["concept_id"]},
},
{
    "name": "materialize_similarities",
    "description": "Batch-compute and store high-confidence similarity edges.",
    "parameters": {"type": "object",
        "properties": {"threshold": {"type": "number", "default": 0.85},
                       "limitperconcept": {"type": "integer", "default": 5},
                       "layer": {"type": "string", "default": "domain"}}},
},

search_hybrid additionally gains a layer parameter.

CLI

New commands (additive to the v5.9 set):

| Command | Description |
|---|---|
| okf temporal  --at  | Point-in-time concept reconstruction |
| okf chain  | Reconstruct an inferential/causal reasoning chain |
| okf sources  | Show derivation sources for a synthesized concept |
| okf edges  [--all] | List semantic edges (--all includes expired) |
| okf materialize-similarity [--threshold 0.85] | Batch similarity-edge materialization |
| okf compact | Archive expired edges + superseded revisions |

Index Lifecycle & Compaction

Append-only tables accumulate superseded rows that pollute ANN indexes. Strategy:

Query-time filter — WHERE validto IS NULL / WHERE expiredat IS NULL on every ANN/FTS read. Negligible cost (NULL check on a TIMESTAMP column).
Ratio-triggered compaction — Meta counters track supersededcount vs currentcount. When the ratio exceeds 3:1, okf compact moves expired/superseded rows to *Archive tables (identical schema, no indexes) and rebuilds the live indexes.

| Table | Compaction trigger |
|---|---|
| VectorRevision | superseded : current > 3:1 |
| ChunkRevision | superseded : current > 3:1 |
| SimilarityEdge / LinkEdge / CausalEdge | expired : active > 3:1 |

Compaction is a manual CLI command — predictable, no daemon, no background thread.

Scale Considerations

| Concern | Mitigation |
|---|---|
| Edge-node table growth | expired_at + compaction to *Archive tables |
| ANN index pollution | Query-time valid_to IS NULL filter; ratio-triggered rebuild |
| Bipartite query cost (2 hops) | Negligible in embedded LadybugDB — typed rel tables resolve as pointer chasing, no network round-trip |
| materialize_similarities() at 100K+ | Batch offline job; scores stored, never recomputed per-query; expired on vector change |
| Layer filtering at scale | layer is a STRING column — columnar segment scan, not full-table scan |
| UUID7 generation | Inline ~20-line implementation; time-ordered prefix enables efficient range scans; no external dependency |
| Write amplification (3 INSERTs + edges per concept) | Batched in single transaction; LadybugDB's columnar engine is optimized for bulk writes |

Explicitly Rejected

| Feature | Why Not |
|---|---|
| Bipartite for structural rels (CONTAINS, PARTOF, INCLUDESASSET) | Mechanical, high-volume, zero-metadata. A directory CONTAINS a concept — no knowledge on that edge worth making first-class. |
| probability on containment | okfgraph containment is structural, not probabilistic. A chunk IS part of a document. |
| Separate EntityNode tables for actors/places/events | okfgraph's domain is documents and concepts. Actors/events are ConceptIdentity with type='actor', layer='instance'. |
| PropertyEdge for frontmatter | Frontmatter lives in ConceptIdentity.extra MAP. PropertyEdge is for relational properties only. |
| v5.x migration machinery | No installed base. Re-ingest from OKF bundles instead. |

Implementation Phases

| Phase | Scope | Effort |
|---|---|---|
| 1 | Schema DDL + ConceptIdentity/TextRevision/VectorRevision + smartupsert() | 2–3 d |
| 2 | Edge-node tables + CONNECTS/BINDS + rewrite batchextract_links() | 1–2 d |
| 3 | Read paths: getbyid, searchhybrid, searchchunks, traverse, listdirectory, findpath | 1–2 d |
| 4 | Temporal API: getconceptat, getedgehistory, getreasoningchain, getderivationsources | 1 d |
| 5 | materializesimilarities + expirestale_edges + compact | 1 d |
| 6 | LLM tools (21) + MCP server + CLI commands | 1–2 d |
| 7 | Tests (~380) + benchmark: verify read latency unchanged, measure write overhead | 1–2 d |

Total: 8–13 days.

Open Questions

| # | Question | Leaning |
|---|---|---|
| 1 | Full body snapshots vs. diffs in TextRevision? | Full snapshots — columnar compression handles repetition; diffs complicate reads |
| 2 | Temporal CONTAINS (directory moves over time)? | Defer — structural rels stay timeless |
| 3 | Automatic vs. manual compaction? | Manual (okf compact) |
| 4 | Should VectorRevision store the full search text? | Hash only — text lives in TextRevision |
| 5 | Confidence semantics for CausalEdge from LLM reasoning? | LLM self-reported, capped at 0.9 — never 1.0 for machine-inferred causality |

This specification is a living artifact. v6.0 is a clean break — update freely while the architecture is still being built.