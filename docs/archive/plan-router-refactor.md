# OKFRouter Refactoring Proposal

**Date**: 2026-07-09  
**Current state**: `router.py` — 4,316 lines, 99 methods, 176 KB, single class  
**Goal**: Split into logical submodules while preserving the public API

---

## Analysis

### Current Structure (by section)

| Section | Lines | Methods | Role |
|---|---|---|---|
| Construction (`__init__`) | 174–303 | 1 | DB connection, model loading, WAL, locking |
| Lifecycle (`close`, `_write_lock_ctx`) | 303–377 | 4 | Context management, lock acquire/release |
| Schema (`_ensure_schema`, migrations) | 378–554 | 6 | Table creation, version tracking, migration functions |
| Search indexes (`reindex`, `_build_search_indexes`) | 555–676 | 3 | Vector/FTS index creation and rebuild |
| Meta / dirty tracking | 677–713 | 4 | Epoch counters, `_indexes_dirty()` |
| Delta detection (file/dir hashes) | 714–953 | 10 | SHA-256 hashing, changed files/dirs |
| Purge (`_purge_concept`) | 954–1064 | 1 | Cascading delete with orphan check |
| Soft-delete (recovery) | 1065–1275 | 6 | DeletedConcept table, recover, purge |
| Embedding (`_encode`, `_encode_batch`, `_encode_image`) | 1276–1420 | 7 | ONNX model inference, Matryoshka truncation |
| Chunking (`_split_into_chunks`, overlap) | 1421–1557 | 2 | Text splitting, overlap payload computation |
| Reconstruction | 1558–1587 | 1 | Reassemble chunks into document |
| Property access (`_get_property`) | 1588–1597 | 1 | DB property lookup |
| Parsing (`_parse_source_file`) | 1598–1638 | 1 | Frontmatter extraction, OKF format |
| Import pipeline (`import_bundle`, phases) | 1639–2402 | 10 | Main import loop: parse, encode, upsert, chunk, link, image |
| Image assets | 2403–2651 | 10 | okf-asset:// URIs, dedup, upsert, search |
| Export (`export_to_okf`, `export_bundle`) | 2652–2899 | 5 | Graph-enriched export, index generation |
| Broken links | 2900–2970 | 2 | Detect and repair broken cross-references |
| Chunk search (`search_chunks`, RRF) | 2971–3330 | 7 | Chunk-level hybrid search, hub score reranking |
| Hybrid search (`search_hybrid`) | 3331–3464 | 1 | Main semantic + keyword search |
| Graph traversal (`find_path`, `traverse`) | 3465–3563 | 2 | Relationship navigation |
| Convenience queries (`get_chunks`, `list_directory`, `get_by_id`) | 3564–3672 | 3 | Single-concept lookups |
| Linting (`_lint_converted_md`, `_lint_converted_md_str`) | 3673–3798 | 2 | Mordant lint/fix for markdown |
| Single-concept import (`_import_single_concept`) | 3799–3915 | 1 | Shared helper for ingest methods |
| PDF ingestion (`ingest_pdf`) | 3916–4103 | 1 | Full PDF pipeline |
| MD ingestion (`ingest_md`) | 4104–4205 | 2 | Markdown file import |
| Thoughts ingestion (`ingest_thoughts`) | 4206–end | 2 | LLM reasoning storage |

### Dependency Map

```
OKFRouter (monolith)

┌─────────────────────────────────────────────────────────────┐
│  PUBLIC API (must stay on OKFRouter or be trivially proxied) │
│  ─────────────────────────────────────────────────────────  │
│  import_bundle, ingest_pdf, ingest_md, ingest_thoughts      │
│  search_hybrid, search_chunks, search_with_context          │
│  traverse, find_path, get_by_id, list_directory, get_chunks │
│  export_to_okf, export_bundle                               │
│  reindex, list_broken_links, repair_links                   │
│  list_images, get_image_data, search_images_with_text       │
│  list_deleted_concepts, purge_deleted_concepts              │
│  reconstruct_document                                       │
├─────────────────────────────────────────────────────────────┤
│  INTERNAL (can be extracted to submodules)                   │
│  ─────────────────────────────────────────────────────────  │
│  Schema/migrations → schema.py                              │
│  Delta detection → delta.py                                 │
│  Embedding/chunking → embed.py                              │
│  Soft-delete → soft_delete.py                               │
│  Image assets → images.py (already exists, can expand)      │
│  Export → export.py                                         │
│  Linting → lint.py                                          │
│  Ingestion entry points → ingest.py (already exists)        │
└─────────────────────────────────────────────────────────────┘
```

---

## Proposed Architecture

### Option A: Mixin-Based Split (Recommended)

Split into mixins that each own a vertical slice, composed into `OKFRouter`. The main class stays thin (just `__init__` + composition).

```
okfgraph/
├── router.py              ← thin facade (≈300 lines)
├── _schema.py             ← schema creation + migrations (≈350 lines)
├── _delta.py              ← delta detection + hashing (≈250 lines)
├── _embed.py              ← embedding + chunking (≈250 lines)
├── _search.py             ← search + traversal (≈700 lines)
├── _import.py             ← import pipeline (≈600 lines)
├── _soft_delete.py        ← soft-delete + recovery (≈250 lines)
├── _purge.py              ← cascade delete (≈110 lines)
├── _export.py             ← export + enrichment (≈300 lines)
├── _image_assets.py       ← okf-asset URIs (≈250 lines)
├── _lint.py               ← mordant linting (≈130 lines)
└── _ingest.py             ← PDF/MD/thoughts entry points (≈450 lines)
```

**Why mixins over submodules?**
- Zero API changes — all methods stay on `OKFRouter`
- No proxying overhead
- Shared state (`self.conn`, `self.embedder`, etc.) is naturally accessible
- Each mixin is a self-contained concern

### Option B: Submodule with Facade

Extract logic into independent submodules, proxy from `OKFRouter`:

```python
# router.py
def search_hybrid(self, query, ...):
    return _search.search_hybrid(self, query, ...)
```

**Disadvantages**: Every method call is a proxy, harder to navigate, more boilerplate.

### Recommendation: Option A (Mixins)

---

## Detailed Mixin Breakdown

### 1. `_SchemaMixin` (`okfgraph/_schema.py`)

**Responsibilities**: Schema creation, version tracking, migrations

| Method | Lines |
|---|---|
| `_ensure_schema` | 382–526 |
| `_adopt_existing_embedding_dim` | 527–554 |
| `_run_schema_migrations` | 559–605 |
| `_migrate_v1_to_v2` | 87–118 |
| `_migrate_v2_to_v3` | 120–133 |
| `_migrate_v3_to_v4` | 135–148 |
| `_migrate_v4_to_v5` | 150–167 |

**Dependencies**: `self.conn`, `SCHEMA_VERSION`, `_MIGRATIONS` dict  
**Size**: ~350 lines

### 2. `_DeltaMixin` (`okfgraph/_delta.py`)

**Responsibilities**: File/directory hashing, changed file detection

| Method | Lines |
|---|---|
| `_file_hash` | 718–721 |
| `_compute_directory_hash` | 722–737 |
| `_compute_directory_hash_with_files` | 738–754 |
| `_load_directory_hashes` | 755–774 |
| `_store_directory_hashes` | 775–789 |
| `_changed_directories` | 790–860 |
| `_store_file_hashes` | 861–880 |
| `_load_file_hashes` | 881–892 |
| `_load_file_hash_concept_ids` | 893–904 |
| `_changed_files` | 905–953 |

**Dependencies**: `self.conn`, `self.bundle_root`  
**Size**: ~250 lines

### 3. `_EmbedMixin` (`okfgraph/_embed.py`)

**Responsibilities**: ONNX inference, Matryoshka truncation, chunking

| Method | Lines |
|---|---|
| `_encode` | 1280–1328 |
| `_truncate_normalize` | 1329–1342 |
| `_encode_batch` | 1343–1377 |
| `_get_omni` | 1378–1395 |
| `_encode_image` | 1396–1413 |
| `_encode_omni_text` | 1414–1425 |
| `default_cache_dir` | 1426–1431 |
| `model_info` | 1432–1471 |
| `_split_into_chunks` | 1472–1504 |
| `_compute_overlap_payloads` | 1505–1558 |
| `reconstruct_document` | 1559–1588 |

**Dependencies**: `self.embedder`, `self.tokenizer`, `self.embedding_dim`, `self._omni`  
**Size**: ~250 lines

### 4. `_SearchMixin` (`okfgraph/_search.py`)

**Responsibilities**: All search, traversal, and graph queries

| Method | Lines |
|---|---|
| `search_hybrid` | 3335–3468 |
| `search_chunks` | 2975–3110 |
| `_compute_hub_scores` | 3111–3125 |
| `_get_ancestry` | 3126–3138 |
| `_get_siblings` | 3139–3171 |
| `search_with_context` | 3172–3209 |
| `search_chunks_with_hub_score` | 3210–3238 |
| `expand_with_graph_context` | 3239–3293 |
| `rerank_with_hub_score` | 3294–3334 |
| `find_path` | 3469–3510 |
| `traverse` | 3511–3567 |
| `get_chunks` | 3568–3595 |
| `list_directory` | 3596–3646 |
| `get_by_id` | 3647–3676 |

**Dependencies**: `self.conn`, `self.tokenizer`, `self.embedding_dim`  
**Size**: ~700 lines

### 5. `_ImportMixin` (`okfgraph/_import.py`)

**Responsibilities**: Main import pipeline, batch operations

| Method | Lines |
|---|---|
| `_parse_source_file` | 1606–1639 |
| `import_from_okf` | 1640–1753 |
| `import_bundle` | 1754–1780 |
| `_import_bundle_inner` | 1781–1952 |
| `_import_chunks_for_concept` | 1953–2046 |
| `_batch_upsert_concepts` | 2047–2130 |
| `_batch_build_directories` | 2131–2177 |
| `_batch_extract_links` | 2178–2232 |
| `_extract_links_for_concept` | 2233–2282 |
| `_insert_concept` | 2283–2406 |

**Dependencies**: `self.conn`, `_DeltaMixin`, `_EmbedMixin`, `_write_lock_ctx`  
**Size**: ~600 lines

### 6. `_SoftDeleteMixin` (`okfgraph/_soft_delete.py`)

**Responsibilities**: Soft-delete, recovery, purge

| Method | Lines |
|---|---|
| `_soft_delete_concept` | 1069–1085 |
| `_soft_delete_concept_inner` | 1086–1137 |
| `_recover_concept` | 1138–1150 |
| `_recover_concept_inner` | 1151–1203 |
| `list_deleted_concepts` | 1204–1232 |
| `purge_deleted_concepts` | 1233–1246 |
| `_purge_deleted_concepts_inner` | 1247–1279 |

**Dependencies**: `self.conn`, `_PurgeMixin`, `_write_lock_ctx`  
**Size**: ~250 lines

### 7. `_PurgeMixin` (`okfgraph/_purge.py`)

**Responsibilities**: Cascading delete with orphan check

| Method | Lines |
|---|---|
| `_purge_concept` | 958–1064 |

**Dependencies**: `self.conn`  
**Size**: ~110 lines

### 8. `_ExportMixin` (`okfgraph/_export.py`)

**Responsibilities**: Graph-enriched export, bundle generation

| Method | Lines |
|---|---|
| `export_to_okf` | 2653–2660 |
| `_enrich_body_with_graph_links` | 2661–2730 |
| `_generate_index_files` | 2731–2767 |
| `_write_okf` | 2768–2791 |
| `export_bundle` | 2792–2899 |
| `_fetch_concepts` | 2848–2899 |
| `_is_under_directory` | 2900–2912 |

**Dependencies**: `self.conn`  
**Size**: ~300 lines

### 9. `_ImageAssetsMixin` (`okfgraph/_image_assets.py`)

**Responsibilities**: okf-asset:// URIs, dedup, upsert, image search

| Method | Lines |
|---|---|
| `_ingest_concept_images` | 2407–2497 |
| `_content_hash` | 2498–2505 |
| `_concept_has_assets` | 2506–2516 |
| `_existing_asset_hashes` | 2517–2529 |
| `_delete_image_asset` | 2530–2555 |
| `_upsert_image_asset` | 2556–2589 |
| `list_images` | 2590–2601 |
| `get_image_data` | 2602–2614 |
| `search_images_with_text` | 2615–2652 |

**Dependencies**: `self.conn`, `_EmbedMixin`  
**Size**: ~250 lines

### 10. `_LintMixin` (`okfgraph/_lint.py`)

**Responsibilities**: Mordant lint/fix for markdown

| Method | Lines |
|---|---|
| `_lint_converted_md` | 3677–3753 |
| `_lint_converted_md_str` | 3754–3798 |

**Dependencies**: `mordant` import  
**Size**: ~130 lines

### 11. `_IngestMixin` (`okfgraph/_ingest.py`)

**Responsibilities**: PDF/MD/thoughts entry points

| Method | Lines |
|---|---|
| `_import_single_concept` | 3799–3919 |
| `ingest_pdf` | 3920–4107 |
| `ingest_md` | 4108–4148 |
| `_ingest_md_inner` | 4149–4209 |
| `ingest_thoughts` | 4210–4244 |
| `_ingest_thoughts_inner` | 4245–end |

**Dependencies**: `self.conn`, `_ImportMixin`, `_LintMixin`, `_write_lock_ctx`  
**Size**: ~450 lines

### 12. `_IndexMixin` (`okfgraph/_index.py`)

**Responsibilities**: Search index lifecycle, dirty tracking, reindex

| Method | Lines |
|---|---|
| `_index_specs` | 606–621 |
| `_build_search_indexes` | 622–665 |
| `reindex` | 666–676 |
| `_get_meta` | 681–689 |
| `_set_meta` | 690–697 |
| `_bump_write_epoch` | 698–710 |
| `_indexes_dirty` | 711–713 |

**Dependencies**: `self.conn`  
**Size**: ~130 lines

---

## Composed OKFRouter (facade)

```python
"""OKFRouter — Ladybug-backed knowledge graph with ONNX + Jina v5 embeddings."""

from okfgraph._schema import _SchemaMixin
from okfgraph._delta import _DeltaMixin
from okfgraph._embed import _EmbedMixin
from okfgraph._search import _SearchMixin
from okfgraph._import import _ImportMixin
from okfgraph._soft_delete import _SoftDeleteMixin
from okfgraph._purge import _PurgeMixin
from okfgraph._export import _ExportMixin
from okfgraph._image_assets import _ImageAssetsMixin
from okfgraph._lint import _LintMixin
from okfgraph._ingest import _IngestMixin
from okfgraph._index import _IndexMixin


class OKFRouter(
    _SchemaMixin,
    _DeltaMixin,
    _EmbedMixin,
    _SearchMixin,
    _ImportMixin,
    _SoftDeleteMixin,
    _PurgeMixin,
    _ExportMixin,
    _ImageAssetsMixin,
    _LintMixin,
    _IngestMixin,
    _IndexMixin,
):
    """Routes OKF concepts through a Ladybug graph + vector + FTS database."""

    ALLOWED_DIMS = (128, 256, 512, 768, 1024)
    SCHEMA_VERSION = 5
    SOFT_DELETE_WINDOW = 24 * 60 * 60

    _MIGRATIONS = {
        2: _SchemaMixin._migrate_v1_to_v2,
        3: _SchemaMixin._migrate_v2_to_v3,
        4: _SchemaMixin._migrate_v3_to_v4,
        5: _SchemaMixin._migrate_v4_to_v5,
    }

    def __init__(self, ...):
        # ~130 lines: db connection, model loading, WAL, locking, schema init
        ...

    def close(self):
        ...

    @contextmanager
    def _write_lock_ctx(self):
        ...

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
```

**Estimated size of composed `router.py`**: ~300 lines (down from 4,316)

---

## Migration Strategy

### Phase 1: Extract low-risk mixins (no API changes)

Start with mixins that have minimal dependencies on other sections:

| Mixin | Risk | Dependencies |
|---|---|---|
| `_LintMixin` | None | Only `mordant` import |
| `_PurgeMixin` | Low | Only `self.conn` |
| `_DeltaMixin` | Low | `self.conn`, `self.bundle_root` |
| `_IndexMixin` | Low | `self.conn` |

### Phase 2: Extract core mixins

| Mixin | Risk | Dependencies |
|---|---|---|
| `_SchemaMixin` | Low | `self.conn`, class constants |
| `_EmbedMixin` | Medium | `self.embedder`, `self.tokenizer`, `self._omni` |
| `_SoftDeleteMixin` | Medium | `_PurgeMixin`, `_write_lock_ctx` |
| `_ImageAssetsMixin` | Medium | `_EmbedMixin` |
| `_ExportMixin` | Low | `self.conn` |

### Phase 3: Extract complex mixins

| Mixin | Risk | Dependencies |
|---|---|---|
| `_SearchMixin` | Medium | `self.conn`, `self.tokenizer`, `_EmbedMixin` |
| `_ImportMixin` | High | `_DeltaMixin`, `_EmbedMixin`, `_write_lock_ctx` |
| `_IngestMixin` | High | `_ImportMixin`, `_LintMixin`, `_write_lock_ctx` |

### Phase 4: Compose and verify

1. Create thin `OKFRouter` facade with mixin composition
2. Run full test suite (335 tests)
3. Verify no behavior changes

---

## Alternative: Submodule Approach (if mixins feel too implicit)

If you prefer explicit delegation over implicit mixin inheritance:

```python
# okfgraph/router.py — explicit delegation
class OKFRouter:
    def __init__(self, ...):
        self._schema = _Schema(self.conn, ...)
        self._delta = _Delta(self.conn, self.bundle_root)
        self._embed = _Embed(self.embedder, self.tokenizer, ...)
        ...

    def search_hybrid(self, query, ...):
        return self._search.search_hybrid(query, ...)

    def import_bundle(self, bundle_path, ...):
        return self._import.import_bundle(bundle_path, ...)
```

**Trade-off**: More explicit, but every public method is a proxy call. Adds ~100 lines of boilerplate to `router.py`.

---

## Recommendation

**Start with Phase 1 (low-risk mixins)** as a proof of concept. If the mixin approach works well with tests, proceed to Phases 2–3. The total target is:

| File | Before | After |
|---|---|---|
| `router.py` | 4,316 lines | ~300 lines |
| `_schema.py` | — | ~350 lines |
| `_delta.py` | — | ~250 lines |
| `_embed.py` | — | ~250 lines |
| `_search.py` | — | ~700 lines |
| `_import.py` | — | ~600 lines |
| `_soft_delete.py` | — | ~250 lines |
| `_purge.py` | — | ~110 lines |
| `_export.py` | — | ~300 lines |
| `_image_assets.py` | — | ~250 lines |
| `_lint.py` | — | ~130 lines |
| `_ingest.py` | — | ~450 lines |
| `_index.py` | — | ~130 lines |

**Net change**: Same total lines, but logically separated into 12 focused modules. Each module can be tested independently, reviewed independently, and evolved independently.

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Mixin method name collisions | Prefix each mixin's private methods with mixin-specific prefix |
| MRO (Method Resolution Order) issues | Explicit base class order, test with `OKFRouter.__mro__` |
| Shared state access patterns | Document which attributes each mixin requires |
| Test breakage | Run full test suite after each phase |
| Import cycles | Use lazy imports or restructure import graph |
