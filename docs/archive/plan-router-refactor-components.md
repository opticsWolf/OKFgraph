# OKFRouter Refactoring — Option 2: Explicit Component Objects

**Date**: 2026-07-09  
**Supersedes**: `docs/plan-router-refactor.md` (mixin approach — pivoted per review)  
**Target**: `router.py` (4,316 lines, 99 methods, 176 KB) → facade + 9 components + lint module

---

## 1. Executive Summary

Per the review, the **mixin approach was rejected** because it hides cross-section coupling behind shared `self` state. We pivot to **explicit component objects** (dependency injection): `OKFRouter` becomes a thin facade that owns resources (DB connection, ONNX model, lock) and delegates to focused, independently-instantiable components.

**Key design rules:**
1. Each component receives its dependencies via `__init__` — no hidden `self` magic.
2. Resources (connection, model, tokenizer, lock) are **owned by the facade** and passed down.
3. Public API stays on the facade as explicit 1-line proxies (typed, IDE-friendly).
4. Components are unit-testable in isolation (pass a mock `conn` + stub embedder).

---

## 2. Target File Structure

```
okfgraph/
├── router.py                  ← facade: owns resources, wires components, proxies public API (~200 lines)
├── components/
│   ├── __init__.py
│   ├── schema.py              ← SchemaManager        (~450 lines)
│   ├── delta.py               ← DeltaDetector        (~250 lines)
│   ├── purge.py               ← PurgeManager         (~320 lines)
│   ├── embedding.py           ← EmbeddingEngine      (~400 lines)
│   ├── image_assets.py        ← ImageAssetManager    (~250 lines)
│   ├── search.py              ← SearchEngine         (~700 lines)
│   ├── import_.py             ← ImportManager        (~600 lines)
│   ├── export.py              ← ExportManager        (~300 lines)
│   ├── ingest.py              ← IngestManager        (~450 lines)
│   └── lint.py                ← module-level functions (~130 lines)
```

> Note: `images.py` (existing) handles *conversion-time* asset staging. `image_assets.py` (new) handles *DB-side* asset storage/query — distinct concerns.

---

## 3. Component Specifications

### 3.1 `SchemaManager(conn, embedding_dim, bundle_root, write_lock_ctx)`
**Owns**: table creation, version migrations, search-index lifecycle, meta/epoch tracking, broken-link detection.

| Method | From router.py | Notes |
|---|---|---|
| `_ensure_schema()` | 382–526 | |
| `_adopt_existing_embedding_dim()` | 527–554 | |
| `_run_schema_migrations()` | 559–605 | |
| `_migrate_v1_to_v2/3/4/5()` | 87–167 | now instance methods (use `self.conn`, `self._index_specs()`) |
| `_index_specs()` | 607–621 | |
| `_build_search_indexes(rebuild, force)` | 622–665 | |
| `reindex(force=True)` | 666–676 | **public → proxied** |
| `_get_meta(key, default)` | 681–689 | heavily tested |
| `_set_meta(key, value)` | 690–697 | heavily tested |
| `_bump_write_epoch()` | 698–710 | |
| `_indexes_dirty()` | 711–713 | heavily tested |
| `list_broken_links()` | 2913–2925 | **public → proxied** |
| `repair_links()` | 2926–2970 | **public → proxied** |

**Internal deps**: none (only `conn`, `embedding_dim`, `bundle_root`).

---

### 3.2 `DeltaDetector(conn, bundle_root)`
**Owns**: file/directory SHA-256 hashing, changed-file detection.

| Method | From router.py |
|---|---|
| `_file_hash(path)` | 718–721 |
| `_compute_directory_hash(dir)` | 722–737 |
| `_compute_directory_hash_with_files(dir)` | 738–754 |
| `_load_directory_hashes()` | 755–774 |
| `_store_directory_hashes(hashes)` | 775–789 |
| `_changed_directories(source_files)` | 790–860 |
| `_store_file_hashes(hashes)` | 861–880 |
| `_load_file_hashes()` | 881–892 |
| `_load_file_hash_concept_ids()` | 893–904 |
| `_changed_files(source_files)` | 905–953 |

**Internal deps**: none. **Heavily tested** (`_file_hash`, `_compute_directory_hash`, `_changed_files`, `_load_directory_hashes`, `_load_file_hash_concept_ids`).

---

### 3.3 `PurgeManager(conn, write_lock_ctx)`
**Owns**: hard purge, soft-delete, recovery window.

| Method | From router.py | Public? |
|---|---|---|
| `_purge_concept(id)` | 958–1064 | tested |
| `_soft_delete_concept(id)` | 1069–1085 | |
| `_soft_delete_concept_inner(id)` | 1086–1137 | |
| `_recover_concept(id)` | 1138–1150 | |
| `_recover_concept_inner(id)` | 1151–1203 | |
| `list_deleted_concepts()` | 1204–1232 | **public → proxied** |
| `purge_deleted_concepts(older_than)` | 1233–1246 | **public → proxied** |
| `_purge_deleted_concepts_inner(older_than)` | 1247–1279 | |

**Internal deps**: `write_lock_ctx` (for all mutating ops). Calls `SchemaManager`? No — soft-delete reads/writes `DeletedConcept` table directly via `conn`.

---

### 3.4 `EmbeddingEngine(embedder, tokenizer, embedding_dim, device, cache_dir, model_id, omni_model_id, chunk_size, chunk_overlap, enable_chunking, conn)`
**Owns**: ONNX inference, Matryoshka truncation, chunk splitting, document reconstruction.

| Method | From router.py | Public? |
|---|---|---|
| `_encode(text, task)` | 1280–1328 | tested via `_encode_batch` |
| `_truncate_normalize(vec)` | 1329–1342 | |
| `_encode_batch(texts, task)` | 1343–1377 | **tested** |
| `_get_omni()` | 1378–1395 | |
| `_encode_image(data)` | 1396–1413 | |
| `_encode_omni_text(text, task)` | 1414–1425 | **tested** |
| `default_cache_dir()` | 1426–1431 | **classmethod → proxied** |
| `model_info(model_id)` | 1432–1471 | **classmethod → proxied** |
| `_split_into_chunks(text)` | 1472–1504 | |
| `_compute_overlap_payloads(chunks)` | 1505–1558 | |
| `reconstruct_document(doc_id)` | 1559–1588 | **public → proxied** |

**Internal deps**: `conn` (for `reconstruct_document` to fetch chunk text). Lazily loads `_omni` model.

---

### 3.5 `ImageAssetManager(conn, embed_engine, allow_remote_images, allowed_image_domains, bundle_root)`
**Owns**: okf-asset:// URI storage, content-hash dedup, image search.

| Method | From router.py | Public? |
|---|---|---|
| `_ingest_concept_images(...)` | 2407–2497 | |
| `_content_hash(route, payload)` | 2498–2505 | |
| `_concept_has_assets(id)` | 2506–2516 | |
| `_existing_asset_hashes(id)` | 2517–2529 | |
| `_delete_image_asset(id, asset_id)` | 2530–2555 | |
| `_upsert_image_asset(id, item)` | 2556–2589 | |
| `list_images(concept_id)` | 2590–2601 | **public → proxied** |
| `get_image_data(asset_id)` | 2602–2614 | **public → proxied** |
| `search_images_with_text(text_query, ...)` | 2615–2652 | **public → proxied** |

**Internal deps**: `embed_engine` (for `_encode_image` during search).

---

### 3.6 `SearchEngine(conn, tokenizer, embedding_dim, embed_engine)`
**Owns**: all read queries — hybrid search, chunk search, graph traversal, hub-score reranking.

| Method | From router.py | Public? |
|---|---|---|
| `search_chunks(...)` | 2975–3110 | **proxied** |
| `_compute_hub_scores(ids)` | 3111–3125 | **tested** |
| `_get_ancestry(id, max_depth)` | 3126–3138 | **tested** |
| `_get_siblings(id, limit)` | 3139–3171 | **tested** |
| `search_with_context(...)` | 3172–3209 | **proxied** |
| `search_chunks_with_hub_score(...)` | 3210–3238 | **proxied** |
| `expand_with_graph_context(...)` | 3239–3293 | **proxied** |
| `rerank_with_hub_score(...)` | 3294–3334 | **proxied** |
| `search_hybrid(...)` | 3335–3468 | **proxied** |
| `find_path(...)` | 3469–3510 | **proxied** |
| `traverse(...)` | 3511–3567 | **proxied** |
| `get_chunks(concept_id)` | 3568–3595 | **proxied** |
| `list_directory(directory_id)` | 3596–3646 | **proxied** |
| `get_by_id(concept_id)` | 3647–3676 | **proxied** |

**Internal deps**: `embed_engine` (for `_encode` in `search_hybrid`, `search_chunks`, `search_images_with_text` lives in ImageAssetManager though — note `search_images_with_text` is in ImageAssetManager, not SearchEngine).

---

### 3.7 `ImportManager(conn, schema_mgr, delta_mgr, embed_engine, image_mgr, write_lock_ctx, bundle_root)`
**Owns**: the full import pipeline — parse, encode, upsert, chunk, link, image, index.

| Method | From router.py | Public? |
|---|---|---|
| `_parse_source_file(path, root)` | 1606–1639 | |
| `import_from_okf(...)` | 1640–1753 | **proxied** |
| `import_bundle(...)` | 1754–1780 | **proxied** |
| `_import_bundle_inner(...)` | 1781–1952 | |
| `_import_single_concept(concept, body, mode)` | 3799–3919 | shared helper |
| `_import_chunks_for_concept(item)` | 1953–2046 | **tested** |
| `_batch_upsert_concepts(cids)` | 2047–2130 | |
| `_batch_build_directories(cids)` | 2131–2177 | |
| `_batch_extract_links(parsed)` | 2178–2232 | |
| `_extract_links_for_concept(id, body)` | 2233–2282 | |
| `_insert_concept(...)` | 2283–2406 | |

**Internal deps**: `schema_mgr` (for `_build_search_indexes` after import, `_get_meta`), `delta_mgr` (for `_changed_directories`, `_store_file_hashes`), `embed_engine` (for `_encode_batch`, `_encode_image`), `image_mgr` (for `_ingest_concept_images`), `write_lock_ctx`.

---

### 3.8 `ExportManager(conn)`
**Owns**: graph-enriched export, index file generation.

| Method | From router.py | Public? |
|---|---|---|
| `export_to_okf(concept_id, output_path)` | 2653–2660 | **proxied** |
| `_enrich_body_with_graph_links(...)` | 2661–2730 | **tested** |
| `_generate_index_files(...)` | 2731–2767 | **tested** |
| `_write_okf(concept, output_path)` | 2768–2791 | |
| `export_bundle(...)` | 2792–2899 | **proxied** |
| `_fetch_concepts(...)` | 2848–2899 | |
| `_is_under_directory(id, dir_id)` | 2900–2912 | |

**Internal deps**: none (only `conn`).

---

### 3.9 `IngestManager(conn, import_mgr, schema_mgr, write_lock_ctx, bundle_root)`
**Owns**: high-level ingest entry points (PDF / MD / thoughts).

| Method | From router.py | Public? |
|---|---|---|
| `ingest_pdf(...)` | 3920–4107 | **proxied** |
| `ingest_md(...)` | 4108–4148 | **proxied** |
| `_ingest_md_inner(...)` | 4149–4209 | |
| `ingest_thoughts(...)` | 4210–4244 | **proxied** |
| `_ingest_thoughts_inner(...)` | 4245–end | |

**Internal deps**: `import_mgr` (for `import_bundle`, `_import_single_concept`), `schema_mgr` (for `_build_search_indexes`), `write_lock_ctx`, `lint` module (for `_lint_converted_md` / `_lint_converted_md_str`).

---

### 3.10 `lint.py` — module-level functions
**Owns**: mordant lint/fix for markdown. No state.

```python
def lint_converted_md(md_path: Path, auto_fix: bool = True) -> dict:
    """Lint a markdown file; return {content, fixed, fixed_count, errors}."""

def lint_converted_md_str(text: str, auto_fix: bool = True) -> dict:
    """Same as above but operates on an in-memory string."""
```

Called by `ImportManager` (after PDF conversion) and `IngestManager` (MD/thoughts).

---

## 4. Facade Specification (`router.py`)

### 4.1 What the facade owns
- `self.db`, `self.conn` — Ladybug connection (passed to all components)
- `self.embedder`, `self.tokenizer`, `self._omni` — ONNX model (passed to `EmbeddingEngine`)
- `self._write_lock`, `self._write_lock_timeout`, `self._write_lock_ctx` — lock (passed to mutating components)
- All config scalars (`embedding_dim`, `bundle_root`, `device`, `cache_dir`, `model_id`, `omni_model_id`, `chunk_size`, `chunk_overlap`, `enable_chunking`, `allow_remote_images`, `allowed_image_domains`)

### 4.2 Wiring order (in `__init__`)
```python
self.db = lb.Database(db_path)
self.conn = lb.Connection(self.db)
# ... WAL, lock, model loading ...

self.schema_mgr   = SchemaManager(self.conn, self.embedding_dim, self.bundle_root, self._write_lock_ctx)
self.delta_mgr    = DeltaDetector(self.conn, self.bundle_root)
self.embed_engine = EmbeddingEngine(self.embedder, self.tokenizer, self.embedding_dim,
                                    self.device, self.cache_dir, self.model_id, self.omni_model_id,
                                    self.chunk_size, self.chunk_overlap, self.enable_chunking, self.conn)
self.image_mgr    = ImageAssetManager(self.conn, self.embed_engine, self.allow_remote_images,
                                      self.allowed_image_domains, self.bundle_root)
self.search_engine = SearchEngine(self.conn, self.tokenizer, self.embedding_dim, self.embed_engine)
self.import_mgr   = ImportManager(self.conn, self.schema_mgr, self.delta_mgr, self.embed_engine,
                                  self.image_mgr, self._write_lock_ctx, self.bundle_root)
self.export_mgr   = ExportManager(self.conn)
self.ingest_mgr   = IngestManager(self.conn, self.import_mgr, self.schema_mgr,
                                  self._write_lock_ctx, self.bundle_root)

self._ensure_schema()   # now: self.schema_mgr._ensure_schema()
```

### 4.3 Public API proxies (explicit, typed)
```python
# Schema / index
def reindex(self, force=True):            return self.schema_mgr.reindex(force)
def list_broken_links(self):             return self.schema_mgr.list_broken_links()
def repair_links(self) -> int:           return self.schema_mgr.repair_links()

# Soft-delete
def list_deleted_concepts(self):         return self.purge_mgr.list_deleted_concepts()
def purge_deleted_concepts(self, older_than=None): return self.purge_mgr.purge_deleted_concepts(older_than)

# Embedding / chunking
def reconstruct_document(self, doc_id):   return self.embed_engine.reconstruct_document(doc_id)
default_cache_dir = EmbeddingEngine.default_cache_dir   # classmethod alias
model_info = EmbeddingEngine.model_info                   # classmethod alias

# Import
def import_from_okf(self, *a, **k):       return self.import_mgr.import_from_okf(*a, **k)
def import_bundle(self, *a, **k):         return self.import_mgr.import_bundle(*a, **k)

# Images
def list_images(self, cid):              return self.image_mgr.list_images(cid)
def get_image_data(self, aid):           return self.image_mgr.get_image_data(aid)
def search_images_with_text(self, *a, **k): return self.image_mgr.search_images_with_text(*a, **k)

# Export
def export_to_okf(self, *a, **k):         return self.export_mgr.export_to_okf(*a, **k)
def export_bundle(self, *a, **k):         return self.export_mgr.export_bundle(*a, **k)

# Search / traversal
def search_hybrid(self, *a, **k):         return self.search_engine.search_hybrid(*a, **k)
def search_chunks(self, *a, **k):         return self.search_engine.search_chunks(*a, **k)
def search_with_context(self, *a, **k):   return self.search_engine.search_with_context(*a, **k)
def search_chunks_with_hub_score(self, *a, **k): return self.search_engine.search_chunks_with_hub_score(*a, **k)
def expand_with_graph_context(self, *a, **k): return self.search_engine.expand_with_graph_context(*a, **k)
def rerank_with_hub_score(self, *a, **k): return self.search_engine.rerank_with_hub_score(*a, **k)
def find_path(self, *a, **k):             return self.search_engine.find_path(*a, **k)
def traverse(self, *a, **k):              return self.search_engine.traverse(*a, **k)
def get_chunks(self, cid):                return self.search_engine.get_chunks(cid)
def list_directory(self, did):            return self.search_engine.list_directory(did)
def get_by_id(self, cid):                 return self.search_engine.get_by_id(cid)

# Ingest
def ingest_pdf(self, *a, **k):            return self.ingest_mgr.ingest_pdf(*a, **k)
def ingest_md(self, *a, **k):             return self.ingest_mgr.ingest_md(*a, **k)
def ingest_thoughts(self, *a, **k):       return self.ingest_mgr.ingest_thoughts(*a, **k)

# Lifecycle (stays on facade)
def close(self): ...
def __enter__(self): return self
def __exit__(self, *exc): self.close()
```

### 4.4 Test-bridge strategy (critical)

**Problem**: 50+ test call sites use private methods directly:
```
router._set_meta, router._get_meta, router._indexes_dirty, router._run_schema_migrations,
router._purge_concept, router._enrich_body_with_graph_links, router._changed_files,
router._load_file_hash_concept_ids, router._import_chunks_for_concept, router._get_siblings,
router._get_ancestry, router._generate_index_files, router._compute_hub_scores,
router._bump_write_epoch, router._adopt_existing_embedding_dim,
r._compute_directory_hash, r._file_hash, r._encode_batch, r._encode_omni_text, ...
```

**Two options:**

**Option A — Update tests (recommended for cleanliness):** Change `router._set_meta(k, v)` → `router.schema_mgr._set_meta(k, v)` etc. One-time ~50 edits across test files. Result: zero facade magic, honest tests.

**Option B — Temporary `__getattr__` bridge:** Add to facade during transition:
```python
def __getattr__(self, name):
    for comp in (self.schema_mgr, self.delta_mgr, self.purge_mgr,
                 self.embed_engine, self.image_mgr, self.search_engine,
                 self.import_mgr, self.export_mgr, self.ingest_mgr):
        if hasattr(comp, name):
            return getattr(comp, name)
    raise AttributeError(name)
```
Keeps all 335 tests green with zero test edits. **Remove after tests migrated to Option A.**

**Recommendation**: Use Option B as a development bridge (keeps CI green while you move code), then migrate tests to Option A and delete `__getattr__`. Final state uses explicit proxies only.

> `router.conn` (30 test refs) stays on the facade — no change needed.

---

## 5. Dependency Wiring Diagram

```
                         OKFRouter (facade)
                    ┌────────────────────────┐
                    │ owns: conn, embedder,   │
                    │        tokenizer, lock  │
                    └────────────────────────┘
                     │ wires ↓ (explicit DI)
   ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
   │SchemaMgr │DeltaDet  │PurgeMgr │EmbedEng  │ImageAsst │SearchEng│ImportMgr│ExportMgr│IngestMgr│
   └──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
        ↑           ↑          ↑          ↑           ↑           ↑          ↑
        └───────────┴──────────┴──────────┴───────────┴───────────┴──────────┘
                  ImportManager depends on: SchemaMgr, DeltaDet, EmbedEng, ImageAsst
                  IngestManager depends on: ImportMgr, SchemaMgr
                  SearchEngine depends on: EmbedEng
                  ImageAssetManager depends on: EmbedEng
```

No circular dependencies: facade wires leaf-to-root (Embedding/Delta/Schema first, then Importers/Search/Images, then Ingest).

---

## 6. Phased Implementation Plan

### Phase 0 — Scaffold (no behavior change) ✅ DONE
- Created `okfgraph/components/` package with `__init__.py`.
- Created component classes + `lint.py`. `lint.py` is **fully implemented** (self-contained, no router state).
- Facade imports them but still defines all methods inline.
- **Gate**: `import okfgraph.router` succeeds. ✅

### Phase 1 — Extract leaf components (no internal deps) ✅ DONE
| Component | Status | Notes |
|---|---|---|
| `lint.py` | ✅ done | module-level, fully implemented in Phase 0 |
| `DeltaDetector` | ✅ done | verbatim bodies; uses Ladybug Cypher graph nodes (NOT sqlite tables) |
| `SchemaManager` | ✅ done | schema + migrations + meta + search-index rebuild; `@staticmethod router` migrations converted to instance `self` |
| `EmbeddingEngine` | ✅ done | encode/chunk/reconstruct; `model_info`/`default_cache_dir` kept as classmethod/staticmethod aliases |
| `ExportManager` | ⏸ **DEFERRED → Phase 3** | `_enrich_body_with_graph_links` depends on `_get_siblings`/`_get_ancestry` (graph-traversal helpers that move in Phase 3). Left as a clean stub; real methods stay on the facade. |

**What moved**: 34 methods extracted verbatim from `router.py` into the three components
(13 schema + 10 delta + 11 embedding). `router.py` went 4,316 → 3,437 lines.

**Bridge**: `__getattr__` on the facade resolves private helpers via `_components`
(`schema_mgr`, `delta_mgr`, `embed_engine`). Reads `self.__dict__` directly to avoid recursion.
Instance-level `hasattr`/`call` sites resolve correctly (e.g. the 6 internal `self._encode(...)`
calls in the facade delegate to `embed_engine._encode`).

**Class-level aliases added** (bridge cannot serve class attribute access):
- `OKFRouter.SCHEMA_VERSION` / `OKFRouter._MIGRATIONS` → `SchemaManager`
- `OKFRouter.model_info` / `OKFRouter.default_cache_dir` → `EmbeddingEngine`

**Test change required** (one, unavoidable): `test_router.py::test_encode_method_exists`
checked `hasattr(OKFRouter, "_encode")` at *class* level. Since `_encode` now lives on
`EmbeddingEngine` and the bridge only handles instances, the test was updated to
`hasattr(EmbeddingEngine, "_encode")`. (Adding a class alias would have shadowed the
bridge and broken the 6 internal `self._encode(...)` calls — so updating the test is correct.)

**Gotcha — `bundle_root` is mutable**: `ingest_pdf` temporarily rebinds `self.bundle_root`
to a temp work dir. `DeltaDetector.bundle_root` is a separate injected copy, so
`ingest_pdf` now also syncs `self.delta_mgr.bundle_root = work_dir` (and restores it).
Without this, PDF ingestion raised `ValueError: ... not in the subpath of ...`.

**Gotcha — Cypher, not SQL**: `SchemaManager` and `DeltaDetector` bodies use Ladybug
Cypher (`MATCH (m:Meta ...)`, `MERGE (d:DirectoryHash ...)`), not sqlite `meta`/`*_hashes`
tables. The components were generated by extracting the *verbatim* original bodies, which
avoids re-deriving the storage layer incorrectly.

**Gate**: full suite green (346 passed, 10 skipped; was 335). ✅

### Phase 2 — Extract mid-tier components ✅ DONE
| Component | Status | Notes |
|---|---|---|
| `PurgeManager` | ✅ done | 8 methods + `SOFT_DELETE_WINDOW` class attr (31 methods extracted across P1+P2) |
| `ImageAssetManager` | ✅ done | 9 methods; `schema_mgr` injected for `_bump_write_epoch`; `embed_engine` injected for `_encode*` |
| `SearchEngine` | ✅ done | 14 methods; `embed_engine` injected for `_encode`; `_search_available` synced from `schema_mgr` |

**What moved**: 31 methods extracted verbatim from `router.py` into the three components
(8 purge + 9 image + 14 search). `router.py` went 3,437 → 2,237 lines.

**Injection changes vs the original plan** (cross-component deps required it):
- `ImageAssetManager.__init__` gained a `schema_mgr` parameter (used by
  `_delete_image_asset` / `_upsert_image_asset` → `self.schema_mgr._bump_write_epoch()`).
- `SearchEngine` methods that encode text now call `self.embed_engine._encode(...)`,
  `self.embed_engine._encode_omni_text(...)`, `self.embed_engine._encode_image(...)`.
- `_search_available` is owned by `SchemaManager` (set in `_ensure_schema`); the facade
  syncs it onto `SearchEngine` (`self.search_engine._search_available = self.schema_mgr._search_available`)
  so `search_hybrid`/`search_chunks` know whether search is available.

**Public API proxies added** (plan §4.3): `get_by_id`, `list_directory`, `search_hybrid`,
`traverse` are now explicit 1-line facade proxies delegating to `search_engine`. This makes
`hasattr(OKFRouter, ...)` true at class level for the `method_exists` tests and keeps the
instance API explicit (the bridge still resolves the rest).

**Scan lesson**: the dependency scan must treat `self.<attr>` injected attributes as
component-local, and must walk the *generated* component files (not just the source) to
catch `self._encode`-style calls that resolve to `self.embed_engine`. Multi-line method
headers (closing `):` at the `def` indent) need a header-aware span extractor or the
closing line is orphaned.

**Gate**: full suite green (346 passed, 10 skipped). ✅

### Phase 3 — Extract orchestrators
| Component | Risk | Tests affected |
|---|---|---|
| `ImportManager` | High | test_integration, test_delta, test_router |
| `IngestManager` | High | test_ingest, test_pdf_e2e, test_router |

**Gate**: full suite green.

### Phase 4 — Cleanup ✅ DONE
- `__getattr__` bridge removed.
- ~218 call sites migrated to component references (171 in tests, 44 in source:
  `cli.py`, `mcp_server.py`), plus `hasattr`/`getattr` instance checks and 7 test edits.
- Full suite green without the bridge (346 passed, 10 skipped).
- `router.py` slimmed 4,316 → 488 lines (facade: resources, `__init__` wiring, 8 public
  proxies, lifecycle). The "~200 lines" target was aspirational; the real slim is ~9× smaller
  and dominated by the `__init__` wiring of nine components.

**Migration gotchas (documented in the status doc)**:
- Source files (`cli.py`, `mcp_server.py`) also used the bridge — a test-only migration left
  them broken until re-run over `okfgraph/`.
- The rewrite regex must emit `recv + instance` (not `recv + "." + instance`) or it produces
  `router..instance.method` (syntax error).
- Exclude class aliases (`model_info`, `default_cache_dir`) and the 8 public proxies from the
  method→instance map.
- Instance `hasattr`/`getattr`/`callable` checks on moved methods must be rewritten to the
  owning component too.


---

## 7. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Circular import (component ↔ router) | Medium | Components never import `router`; use `TYPE_CHECKING` + string annotations for type hints |
| `__getattr__` hides missing methods | Low | Bridge removed in Phase 4; explicit proxies are the real API |
| Test churn (50 private calls) | High | Bridge (Phase 0–3) keeps CI green; migrate in Phase 4 |
| Component constructor signature drift | Medium | Pin signatures in this plan; add a smoke test that instantiates each component with a mock conn |
| `conn` shared mutable state across components | Low | Intended — single connection, serialized by `write_lock_ctx` |
| Performance regression from proxy calls | Low | Proxies are 1-line passthroughs; no overhead beyond one function call |

---

## 8. Effort Estimate

| Phase | Effort | Test impact |
|---|---|---|
| 0 — Scaffold | 0.5 hr | None |
| 1 — Leaf components | 3–4 hrs | Bridge keeps green |
| 2 — Mid-tier | 3–4 hrs | Bridge keeps green |
| 3 — Orchestrators | 4–6 hrs | Bridge keeps green |
| 4 — Cleanup + test migration | 2–3 hrs | ~50 test edits |
| **Total** | **~13–18 hrs** | 335 tests stay green throughout |

### Before / After

| Metric | Before | After |
|---|---|---|
| `router.py` size | 4,316 lines | 488 lines (facade: wiring + 8 proxies + lifecycle) |
| Largest component | — | `search.py` (~700) |
| Public API call sites | inline | 1-line proxies |
| Components unit-testable alone | No | **Yes** (mock `conn` + stub embedder) |
| MRO / hidden coupling | N/A | **None** (explicit DI) |

---

## 9. Status & Recommended Next Step

**Phase 0 ✅, Phase 1 ✅, Phase 2 ✅, Phase 3 ✅, and Phase 4 ✅ are complete** (full suite
 Green: 346 passed, 10 skipped — zero regressions across all phases). The facade now owns
resources and delegates to **nine** components (schema/delta/embedding/purge/image/search/
import/ingest/export) via explicit DI. The `__getattr__` bridge has been removed; every call
site (tests + source) references components directly, and the 8 public proxies are the
explicit public surface.

**The refactor is finished.** Follow-ups (unit-testing components in isolation, the prior
roadmap phases 6–12) are separate work.
