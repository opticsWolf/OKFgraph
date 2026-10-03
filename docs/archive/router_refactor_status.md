# OKFRouter Refactor — Status Report

**Date**: 2026-07-09
**Approach**: Option 2 — Explicit Component Objects (dependency injection) — *supersedes the rejected mixin plan* (`docs/plan-router-refactor.md`)
**Source plan**: `docs/plan-router-refactor-components.md`
**Current `router.py`**: 3,437 lines (down from 4,316)

---

## 1. Executive Summary

The monolithic `OKFRouter` class (4,316 lines, 99 methods) is being split into a thin
**facade** that owns shared resources (DB connection, ONNX embedder, tokenizer, inter-process
lock) and delegates to focused, independently-instantiable **component objects**.

**Phase 0 (scaffold) and Phase 1 (leaf components) are complete.** The full test suite is
green: **346 passed, 10 skipped** (baseline was 335). `router.py` dropped ~880 lines by
extracting 34 methods verbatim into three components; the facade reaches them through an
explicit DI wiring plus a temporary `__getattr__` bridge (removed in Phase 4).

`ExportManager` was deliberately **pulled out of Phase 1** and deferred to Phase 3 because its
`_enrich_body_with_graph_links` depends on graph-traversal helpers (`_get_siblings`,
`_get_ancestry`) that move later.

---

## 2. What Is Implemented

### 2.1 Components (real logic)

| Component | File | Methods | Notes |
|---|---|---|---|
| `lint` (module) | `components/lint.py` | 2 funcs | Fully implemented in Phase 0; self-contained, no router state. Used by `ingest/engine.py`. |
| `SchemaManager` | `components/schema.py` | 15 | Schema migrations, meta KV store, search-index rebuild. Migrations converted from `@staticmethod(router)` → instance `self`; `_MIGRATIONS` registry + `SCHEMA_VERSION` as class attrs. |
| `DeltaDetector` | `components/delta.py` | 11 | Directory/file change detection. |
| `EmbeddingEngine` | `components/embedding.py` | 13 | Vector encode/chunk/reconstruct. `model_info` (classmethod) and `default_cache_dir` (staticmethod) preserved. |
| `PurgeManager` | `components/purge.py` | 9 | Soft-delete / hard-purge / recovery window. `SOFT_DELETE_WINDOW` class attr. |
| `ImageAssetManager` | `components/image_assets.py` | 10 | `okf-asset://` storage, content-hash dedup, image search. `schema_mgr` + `embed_engine` injected. |
| `SearchEngine` | `components/search.py` | 15 | Chunk/graph/hybrid search, hub scores, ancestry/siblings, convenience queries. `embed_engine` injected; `_search_available` synced from `schema_mgr`. |
| `ImportManager` | `components/import_.py` | 14 | Bundle/single-file OKF import. Injected: `conn`, `bundle_root`, `_write_lock_ctx`, `tokenizer`, `enable_chunking`, `schema_mgr`, `delta_mgr`, `embed_engine`, `image_mgr`, `purge_mgr`. `SUPPORTED_SOURCE_EXTS` class attr. |
| `IngestManager` | `components/ingest.py` | 7 | PDF/MD/thoughts ingest; owns the `_lint_converted_md`/`_lint_converted_md_str` wrappers. `import_mgr` + `delta_mgr` injected (cross-P3). |
| `ExportManager` | `components/export.py` | 7 | OKF export. `conn` + `search_engine` injected (graph links / ancestry via `SearchEngine`). |

> **Storage layer is Ladybug Cypher, not sqlite.** `SchemaManager` and `DeltaDetector` bodies
> use `MATCH (m:Meta …)` / `MERGE (d:DirectoryHash …)` graph nodes — *not* `meta` / `*_hashes`
> tables. Bodies were extracted **verbatim** from the original `router.py` to avoid
> re-deriving the storage layer incorrectly.

### 2.2 Components (scaffold stubs — none remaining)

All nine components now carry real logic (Phases 0–3). There are no remaining `...` scaffolds
or `NotImplementedError` stubs in `okfgraph/components/`.

### 2.3 Facade wiring (`OKFRouter.__init__`)

```python
self.schema_mgr  = SchemaManager(self.conn, self.embedding_dim, self._write_lock_ctx)
self.delta_mgr   = DeltaDetector(self.conn, self.bundle_root)
self.purge_mgr   = PurgeManager(self.conn, self._write_lock_ctx)
self._ensure_schema()                 # adopts stored embedding dim
self.embedding_dim = self.schema_mgr.embedding_dim
self.embed_engine  = EmbeddingEngine(self.embedder, self.tokenizer, self.embedding_dim,
                                     self.device, self.cache_dir, self.model_id,
                                     self.omni_model_id, self._omni, self.chunk_size,
                                     self.chunk_overlap, self.enable_chunking, self.conn)
self.image_mgr    = ImageAssetManager(self.conn, self.embed_engine, self.schema_mgr,
                                      self.allow_remote_images, self.allowed_image_domains,
                                      self.bundle_root)
self.search_engine = SearchEngine(self.conn, self.tokenizer, self.embedding_dim,
                                  self.embed_engine)
self.search_engine._search_available = self.schema_mgr._search_available
self.import_mgr  = ImportManager(self.conn, self.bundle_root, self._write_lock_ctx,
                                 self.tokenizer, self.enable_chunking, self.schema_mgr,
                                 self.delta_mgr, self.embed_engine, self.image_mgr,
                                 self.purge_mgr)
self.ingest_mgr  = IngestManager(self._write_lock_ctx, self.bundle_root, self.device,
                                 self.import_mgr, self.delta_mgr)
self.export_mgr  = ExportManager(self.conn, self.search_engine)
```

**Public API proxies** (plan §4.3) on the facade delegate to components, e.g.:
```python
def get_by_id(self, concept_id): return self.search_engine.get_by_id(concept_id)
def search_hybrid(self, *a, **k): return self.search_engine.search_hybrid(*a, **k)
def import_from_okf(self, *a, **k): return self.import_mgr.import_from_okf(*a, **k)
def export_to_okf(self, *a, **k): return self.export_mgr.export_to_okf(*a, **k)
def list_broken_links(self, *a, **k): return self.import_mgr.list_broken_links(*a, **k)
def repair_links(self, *a, **k): return self.import_mgr.repair_links(*a, **k)
```
These make `hasattr(OKFRouter, X)` true at class level (the `method_exists` tests) and keep
the public surface explicit. **As of Phase 4, the temporary `__getattr__` bridge is removed**;
every remaining call site (in the test-suite and in source: `cli.py`, `mcp_server.py`) now
references the owning component directly (e.g. `router.schema_mgr._set_meta(...)`).

**Class-level aliases** (these serve class-attribute references in code/tests):

- `OKFRouter.SCHEMA_VERSION` / `OKFRouter._MIGRATIONS` → `SchemaManager`
- `OKFRouter.model_info` / `OKFRouter.default_cache_dir` → `EmbeddingEngine`

**Bridge removed in Phase 4.** The earlier `__getattr__` mechanism that resolved
`moved methods` via `_components` is gone; all such references are now explicit.

---

## 3. Test Status

| Metric | Value |
|---|---|
| Tests passing | **346** |
| Skipped | 10 |
| Failing | 0 |
| Baseline before refactor | 335 |

**Verified end-to-end**: schema migrations, meta get/set, write-epoch / index-dirty tracking,
directory & file hashing, embedding encode/batch/chunk, document reconstruction, search-index
rebuild, PDF ingestion (end-to-end), full integration + ingest suites.

**Test changes were required** (unavoidable, documented in the plan):
- `tests/test_router.py::test_encode_method_exists` — updated to `hasattr(EmbeddingEngine, "_encode")`
  (class-level `hasattr(OKFRouter, "_encode")` can't use the instance bridge; a class alias would
  have shadowed the bridge and broken the 6 internal `self._encode(...)` calls).
- `tests/test_router.py::test_get_by_id_method_exists`, `test_list_directory_method_exists`,
  `test_search_hybrid_method_exists`, `test_traverse_method_exists` — these now resolve via
  **explicit public proxies** added to the facade (`get_by_id`, `list_directory`, `search_hybrid`,
  `traverse`), which make `hasattr(OKFRouter, X)` true at class level and keep the API explicit.
- Phase 3 added four more public proxies for class-level `hasattr` tests:
  `import_from_okf`, `export_to_okf`, `list_broken_links`, `repair_links`
  (delegating to `import_mgr` / `export_mgr`).
- `tests/test_logging.py::TestImportBundleTiming` — `import_bundle` now logs under
  `okfgraph.components.import_` (was `okfgraph.router`); the test now attaches its capture
  handler to that logger and sets the level explicitly (order-independent).

## 4. Status — All Phases Complete

The router refactor (Phases 0–4) is **finished**. `router.py` is a thin facade (resources,
wiring, 8 public proxies, lifecycle); all domain logic lives in nine explicit, dependency-
injected components. The `__getattr__` bridge was removed in Phase 4 and every call site in
the test-suite **and** the source (`cli.py`, `mcp_server.py`) now references components directly.

- Phase 4 done: bridge removed; ~218 call sites migrated to component references
  (171 in tests, 44 in source) plus `hasattr`/`getattr` instance checks and 7 test edits;
  `router.py` slimmed 4,316 → 488 lines.
- Full suite green: **346 passed, 10 skipped** (zero regressions across all phases).

---

## 5. Gotchas / Lessons Learned

1. **`bundle_root` is mutable and shared.** `ingest_pdf` temporarily rebinds `self.bundle_root`
   to a temp work dir; `DeltaDetector.bundle_root` is a *separate injected copy*, so `ingest_pdf`
   now also syncs `self.delta_mgr.bundle_root = work_dir` (and restores it). Without this, PDF
   import raised `ValueError: … not in the subpath of …`.
2. **Method extractor must keep decorators.** The first extraction pass started at the `def`
   line and silently dropped `@classmethod`/`@staticmethod`, breaking `model_info`/`default_cache_dir`.
   Fixed by backtracking over preceding `@` lines.
3. **Migration functions: `@staticmethod(router)` → instance method.** Converted to
   `def _migrate_vX_to_vY(self)` with `router.` → `self.`; registry rebuilt as a class attr.
4. **Orphan decorators.** After block deletion, any decorator not immediately followed by a
   `def` is an orphan and must be dropped (cleanup pass).
5. **Class-level `hasattr` can't use the instance bridge.** Class-attribute references
   (`OKFRouter._encode`, `OKFRouter.SCHEMA_VERSION`) need explicit aliases or test updates.
6. **Multi-line method signatures need header-aware extraction.** A method whose closing
   `):` sits at the same indent as `def` (e.g. `def f(\n  ...\n) -> X:`) is truncated by a
   naive "stop at first line indented ≤ def" scanner — the `):` line is orphaned and the
   captured body is missing its header. Scan to the first `:`-terminated line, then the body.
7. **Cross-component `self.<method>` calls must be rewritten to the injected handle.**
   `SearchEngine`/`ImageAssetManager` encode via `EmbeddingEngine`; their `self._encode*`
   calls become `self.embed_engine._encode*`. Likewise `ImageAssetManager._delete/_upsert_
   image_asset` route write-epoch bumps to `self.schema_mgr._bump_write_epoch`. The dependency
   scan must walk the *generated* component files (not just `router.py`) to catch these.
8. **Instance attributes owned by one component must be synced onto the consumer.**
   `_search_available` lives on `SchemaManager` (set in `_ensure_schema`); `SearchEngine`
   needs it, so the facade syncs `self.search_engine._search_available = self.schema_mgr._search_available`.
9. **Cross-P3 dependency: `IngestManager` → `ImportManager`.** `ingest_pdf`/`ingest_md` delegate
   storage to `ImportManager` methods (`_import_single_concept`, `import_bundle`). `import_mgr`
   is injected into `IngestManager` and those `self.X(` calls are rewritten to `self.import_mgr.X(`.
   ImportManager does *not* call back into IngestManager, so there is no cycle.
10. **`bundle_root` must be synced onto EVERY injected copy that reads it.** In PDF ingestion,
    `ingest_pdf` rebinds its own `bundle_root` and must also sync `delta_mgr.bundle_root` **and**
    `import_mgr.bundle_root` to the temp work dir (and restore all three). Missing the `import_mgr`
    sync reproduces `ValueError: … not in the subpath of …` inside `import_bundle`.
11. **Class attributes move with the component.** `SUPPORTED_SOURCE_EXTS` (a class attr on the
    facade) was copied as a class attribute on `ImportManager`. The dependency scan must flag
    `self.SUPPORTED_SOURCE_EXTS` (a constant, not a method) so it isn't mistaken for a missing method.
12. **Moved methods change their logger name.** After extraction, `import_bundle` logs under
    `okfgraph.components.import_` instead of `okfgraph.router`. Logging-instrumentation tests
    must follow the code to the new module (and set the logger level explicitly so capture does
    not depend on global logging config / test order).
13. **Migrate source files too, not just tests.** The bridge was used by *any* `router.X`
    call — including `cli.py` and `mcp_server.py`. After removal, those broke (`test_cli`
    export). Re-run the same method→component rewrite over `okfgraph/` (excluding `components/`
    and `router.py`); a test-only migration leaves source call sites broken.
14. **Regex replacement must not double the dot.** Capturing the trailing `.` in the receiver
    group and then emitting `.instance` yields `router..instance.method` (syntax error).
    Emit `recv + instance` (recv already ends with `.`); a post-pass collapsing
    `RECEIVER..INSTANCE.` → `RECEIVER.INSTANCE.` recovers from the mistake.
15. **Exclude class aliases and public proxies from the rewrite.** `model_info` /
    `default_cache_dir` are class attributes on `OKFRouter` (work without the bridge);
    rewriting them to `router.embed_engine.model_info` is wrong/unnecessary. Likewise the 8
    public proxies stay on the facade. Keep the method→instance map limited to truly-moved methods.
16. **`self._ensure_schema()` was bridge-dependent.** `_ensure_schema` lives only on
    `SchemaManager`, so the facade's `self._ensure_schema()` became `self.schema_mgr._ensure_schema()`
    when the bridge was removed.
17. **Instance-level `hasattr`/`getattr` on moved methods break too.** Tests like
    `hasattr(router, "_split_into_chunks")` and `getattr(router, "_search_available", False)`
    resolved via the bridge; rewrite them to the owning component (`router.embed_engine`,
    `router.schema_mgr`). Also catch non-call forms like `callable(router.ingest_pdf)`.

---

## 6. Next Steps

The component refactor is complete. Suggested follow-ups (separate from this refactor):
- Consider unit-testing individual components in isolation (mock `conn` + stub embedder), now
  that each component is self-contained.
- The prior roadmap (phases 6–12: hardening, observability, perf) can proceed against the
  slimmed facade + components.
- `router.py` is 488 lines (mostly `__init__` wiring + 8 proxies + lifecycle) — already ~9×
  smaller than the original 4,316; no further slimming required.

---

## 7. File Inventory

```
okfgraph/
├── router.py                    488 lines    (facade: resources, wiring, 8 public proxies, lifecycle — bridge removed, no _components)
└── components/
    ├── __init__.py              exports all 9 component classes + lint
    ├── lint.py                  IMPLEMENTED (module funcs)
    ├── schema.py                IMPLEMENTED (15 methods + migrations + registry)
    ├── delta.py                 IMPLEMENTED (11 methods)
    ├── embedding.py             IMPLEMENTED (13 methods + classmethod/staticmethod)
    ├── purge.py                 IMPLEMENTED (9 methods)
    ├── image_assets.py          IMPLEMENTED (10 methods)
    ├── search.py                IMPLEMENTED (15 methods)
    ├── import_.py               IMPLEMENTED (14 methods)
    ├── ingest.py                IMPLEMENTED (7 methods)
    └── export.py                IMPLEMENTED (7 methods)
```

**Modified in this work**: `okfgraph/router.py`, `okfgraph/components/{__init__,schema,delta,
embedding,purge,image_assets,search,import_,ingest,export}.py`, `okfgraph/cli.py`,
`okfgraph/mcp_server.py`, `tests/test_router.py`, `tests/test_logging.py`, `tests/test_ingest.py`,
`tests/test_chunking.py`, `tests/test_pdf_e2e.py`, `tests/test_router_misc.py`, and the other
Phase-4 migrated test files, `docs/plan-router-refactor-components.md`.
