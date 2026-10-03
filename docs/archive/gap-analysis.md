> **Historical design record — do not follow.** Describes the pre-0.2.x
> optimum/transformers embedding stack and/or the in-tree RapidAI ingest
> engine, neither of which exists anymore. The authoritative surface is
> `architecture.md` v6.0 (as-built for okfgraph 0.2.12: external
> `embroider` crate, bobine converter seam, 5 MCP tools). Kept for
> archaeology, not guidance.

# OKFGraph — Gap Analysis & Closure Options

**Version**: 3.4  
**Date**: 2026-07-09  
**Architecture baseline**: v5.9 (Core Gaps: Concurrency, Security, Config, PDF Tests)  
**Scope**: Production-readiness gaps between architecture spec and implementation  
**Status**: All 16 gaps CLOSED

---

## Table of Contents

| # | Gap | Severity | Category | Status |
|---|---|---|---|---|
| 1 | Incremental Import / Delta Detection | **High** | Data Integrity | ✅ **#1b Closed** (v5.5) |
| 2 | Index Rebuild / Dirty Tracking | **Medium** | Observability | ✅ **Closed** (v5.1) |
| 3 | Batch Encoding Strategy | **Low** | Documentation | ✅ **Closed** (v5.1) |
| 4 | Adopting Existing Embedding Dimension | **Low** | Documentation | ✅ **Closed** (v5.1) |
| 5 | PDF Ingestion → Import Integration | **High** | Feature Completeness | ✅ **#5b Closed** (v5.4) |
| 6 | Error Handling & Retry Semantics | **High** | Reliability | ✅ **#6d Closed** (v5.1) |
| 7 | Concurrent Access / Locking | **Medium** | Reliability | ✅ **#7b Closed** (v5.9) |
| 8 | Migration / Schema Versioning | **High** | Data Integrity | ✅ **Closed** (v5.3) |
| 9 | Security | **Medium** | Compliance | ✅ **#9b Closed** (v5.9) |
| 10 | Observability | **Medium** | Operations | ✅ **#10a Closed** (v5.4) |
| 11 | Configuration Management | **Low** | Operations | ✅ **#11b Closed** (v5.9) |
| 12 | Testing Strategy Gaps | **High** | Quality | ✅ **#12c Closed** (v5.9) |
| 13 | LLM Tool: `export_bundle` in Architecture | **Low** | Documentation | ✅ **Closed** (v5.1) |
| 14 | Context Window & Truncation Behavior | **Medium** | Correctness | ✅ **#14a Closed** (v5.1) |
| 15 | RapidAI Version Pinning | **Medium** | Maintainability | ✅ **#15a Closed** (v5.4) |
| 16 | MCP Server Integration | **Medium** | Interoperability | ✅ **Closed** (v5.6) |

---

## 1. Incremental Import / Delta Detection

**Status**: ✅ **Closed** — implemented in v5.1  
**Closure option**: **Option D+A** (file-level SHA-256 hash skip + `concept_id` tracking)

### What was implemented

**FileHash node table** with three columns:

```cypher
CREATE NODE TABLE FileHash (
    path STRING PRIMARY KEY,    -- relative path from bundle root
    hash STRING,                -- SHA-256 hex digest of file contents
    concept_id STRING           -- maps file path → Concept.id
);
```

**Delta detection** in `import_bundle()` (Phase 0):

| Method | Role |
|---|---|
| `_file_hash(path)` | Compute SHA-256 hex digest |
| `_store_file_hashes(paths, concept_ids)` | Upsert FileHash entries after import |
| `_load_file_hashes()` | Return `{path: hash}` dict from DB |
| `_load_file_hash_concept_ids()` | Return `{path: concept_id}` dict from DB |
| `_changed_files(source_files)` | Return `(changed_paths, deleted_paths)` tuple |

**Purge** of deleted concepts (`purge_deleted=True`):

| Method | Role |
|---|---|
| `_purge_concept(concept_id)` | Safe cascading delete (concept, chunks, links, orphaned assets) |

Orphan check preserves shared `ImageAsset` nodes when multiple concepts reference the same `okf-asset://<id>` URI.

**CLI integration**: `okf import --all --purge`

**Test coverage**: 21 tests in `tests/test_delta.py` (delta detection, purge, shared assets, edge cases).

### Directory-level hash aggregation (v5.5)

**DirHash node table** for subtree-level delta detection:

```cypher
CREATE NODE TABLE DirHash (
    path STRING PRIMARY KEY,    -- relative directory path from bundle root
    hash STRING,                -- SHA-256 of combined file hashes in directory
    files STRING                -- JSON array of relative file paths for purge tracking
);
```

**Methods** in `OKFRouter`:

| Method | Role |
|---|---|
| `_compute_directory_hash(dir_path)` | Compute combined hash for directory subtree |
| `_compute_directory_hash_with_files(dir_path)` | Compute hash and return file paths |
| `_load_directory_hashes()` | Return `{dir_path: {hash, files}}` dict from DB |
| `_store_directory_hashes(hashes)` | Upsert DirHash entries after import |
| `_changed_directories(source_files)` | Return `(changed_files, deleted_files)` using directory-level hashing |

Files in the same directory share a combined hash. When a single file changes, the directory hash changes and all files in that directory are re-imported. When a directory is deleted, the stored file paths enable purge of all concepts in that subtree.

**Test coverage**: 10 tests in `tests/test_directory_hash.py`.

### Remaining gaps (follow-ups)

| Gap | Severity | Notes |
|---|---|---|
| **Soft-delete with recovery window** | Low | Undo purge within N hours; move to `DeletedConcept` table instead of hard delete |
| **Search text hash (Option C)** | Low | Hash the actual string that gets embedded, not the file bytes — catches frontmatter-only changes more accurately |

### Original analysis (archived)

The original gap analysis proposed four options (A: body hash on Concept, B: full content hash, C: search text hash, D: file-level mtime/hash skip). **Option D+A** was chosen as the Phase 1 implementation — file-level SHA-256 with `concept_id` tracking — because it requires zero schema changes to the Concept table, skips all 7 import phases for unchanged files, and is trivial to implement. Option C (search text hash) is deferred as a follow-up for more accurate embedding dedup.

---

## 2. Index Rebuild / Dirty Tracking

**Status**: ✅ **Closed** — documented in architecture §4.15 (v5.1)

### What was documented

Added §4.15 "Index Lifecycle" covering:
- Two epoch counters (`write_epoch`, `indexed_epoch`) in the `Meta` node table
- Change-driven rebuild strategy: `_indexes_dirty()` compares epochs
- Three rebuild modes (construction, change-driven, force)
- Ladybug index quirks (DROP INDEX leaves stale state, O(N) rebuild cost)
- CLI exposure (`okf reindex [--if-dirty]`)

### Remaining gaps (follow-ups)

| Gap | Severity | Notes |
|---|---|---|
| **`--if-dirty` flag for CLI** | Low | Not yet implemented — deferred to original Option B |
| **`okf index-status` command** | Low | Reports epoch, dirty state, and last rebuild time |

---

## 3. Batch Encoding Strategy

**Status**: ✅ **Closed** — documented in architecture §4.3 (v5.1)

### What was documented

Added "Batch Encoding Algorithm" subsection to §4.3 covering:
- Sequential single-pass encoding vs padded batching
- Attention compute analysis: O(batch × max_len²) vs O(Σ len²)
- Benchmark results: 1.1x faster for variable-length texts
- Real speedup comes from DB-level optimizations, not ONNX batching

No code changes required.

---

## 4. Adopting Existing Embedding Dimension

**Status**: ✅ **Closed** — documented in architecture §4.1 (v5.1)

### What was documented

Added "Auto-Detect Embedding Dimension" subsection to §4.1 covering:
- `_adopt_existing_embedding_dim()` reads `FLOAT[N]` from `CALL TABLE_INFO('Concept')`
- Warns and adopts stored dimension if it differs from requested `embedding_dim`
- `--dim` flag acts as override only for **new** databases; existing databases always use on-disk dimension

No code changes required.

---

## 5. PDF Ingestion → Import Integration

**Status**: ✅ **Closed** — implemented in v5.3, programmatic API in v5.4  
**Closure option**: **Option A** (CLI `okf ingest <pdf>` command) + **Option B** (router method)

### What was implemented

**CLI `okf ingest` subcommand** with two modes:

| Mode | Description |
|---|---|
| **Output-only** | Converts PDF to `.md` + `_assets/` directory. User inspects, then runs `okf import --all`. |
| **`--auto-import`** | Converts to temp dir, imports into graph, cleans up temp files. One-shot workflow. |

**CLI flags**:

| Flag | Description |
|---|---|
| `--auto-import` | Import directly into the graph (temp dir + cleanup) |
| `--output <dir>` | Output directory for converted markdown (default: current dir) |
| `--routing-mode` | ONNX routing: auto/surgical/always/never (default: auto) |
| `--mode` | Image ingestion mode: text/optional/omni (default: text) |
| `--batch-size` | Batch size for encoding during auto-import (default: 32) |
| `--purge` | Purge deleted concepts during auto-import |
| `--no-extract-images` | Skip image extraction (faster for text-only PDFs) |

**Shell support**: `ingest <pdf>` command in the interactive REPL.

### Gap #5b — Router method `ingest_pdf()` (v5.4)

**Status**: ✅ **Closed** — `OKFRouter.ingest_pdf()` implemented (v5.4)

**What was implemented**:

| Component | Description |
|---|---|
| `OKFRouter.ingest_pdf()` | Programmatic API for scripts/notebooks |
| Parameters | `pdf_path`, `auto_import`, `output_dir`, `routing_mode`, `mode`, `batch_size`, `purge_deleted`, `extract_images`, `on_page` |
| Returns | Dict with `md_path`, `concept_ids`, `image_dir`, `page_count` |
| Auto-import | Converts to temp dir, imports via `import_bundle()`, cleans up |
| Output-only | Converts to disk, stages images as `okf-asset://` URIs |
| Test coverage | 4 tests in `tests/test_ingest.py::TestIngestPdfMethod` |

### Remaining gaps (follow-ups)

| Gap | Severity | Notes |
|---|---|---|
| **#5c LLM tool definitions** | Low | ✅ **Closed** — `ingest_md` + `ingest_thoughts` tools added, all 16 tools updated, mordant linting in PDF pipeline (v5.7) |
| **#5d Progress callbacks** | Low | Better progress reporting during conversion (page-level) |

### Original analysis (archived)

The original gap analysis proposed four options (A: CLI command, B: router method, C: standalone script, D: streaming ingest). **Option A** was chosen as the implementation — CLI command with `--auto-import` flag — because it's the least invasive, most user-visible, and composes existing components.

---

## 6. Error Handling & Retry Semantics

**Status**: ✅ **Gap #6d Closed** — per-concept error isolation implemented (v5.1)

### What was implemented

**Per-concept error isolation** in `import_bundle()`:

| Phase | Behavior |
|---|---|
| **Phase 1 (Parse)** | Already had try/except per file — preserved |
| **Phase 3.5 (Chunks)** | Extracted to `_import_chunks_for_concept()` — wrapped in try/except, failures logged and counted |
| **Phase 6 (Images)** | Already had try/except per concept — switched from `print()` to `logger.warning()` |
| **End-of-import report** | Aggregates chunk + image failures, logs summary warning |

**`_import_chunks_for_concept(parsed_item)`**: New method that handles split, encode, upsert, and PART_OF creation for a single concept. Isolated so one bad concept doesn't block the rest.

### Remaining gaps (follow-ups)

| Gap | Severity | Notes |
|---|---|---|
| **#6a Transactional imports** | Medium | Full rollback on Phase 3 failure — already implemented via BEGIN/COMMIT/ROLLBACK |
| **#6b Checkpoint resume** | Low | Resume from last good state after crash |
| **#6c Phase recovery CLI** | Low | `okf recover --phase links` |

---

## 7. Concurrent Access / Locking

**Problem**: Two CLI invocations hitting the same DB simultaneously could corrupt indexes or create duplicate concepts. No locking strategy documented or implemented.

**Impact**: Data corruption under concurrent use. Silent duplicates in the graph.

### Option A: SQLite WAL mode + read/write locks

Enable WAL mode on the Ladybug connection. Reads can proceed during writes. Writes are serialized by SQLite's internal locking.

| Aspect | Detail |
|---|---|
| **Benefits** | SQLite's native solution. No application-level locking needed. Reads don't block writes. |
| **Disadvantages** | WAL mode changes the DB file layout (adds `-wal` and `-shm` files). Ladybug may not expose WAL configuration. |
| **Risks** | Low — WAL is SQLite's recommended mode for concurrent access. |

### Option B: Application-level file lock

Use `filelock` or `portalocker` to acquire an exclusive lock on the DB file before any write operation. Reads don't need the lock.

| Aspect | Detail |
|---|---|
| **Benefits** | Explicit control. Works regardless of SQLite mode. Cross-platform. |
| **Disadvantages** | Adds a dependency. Lock contention on slow filesystems. Deadlock risk if lock isn't released on error. |
| **Risks** | Medium — file locks are platform-dependent (flock vs LockFile). Must handle orphaned locks on crash. |

### Option C: Document single-writer constraint

Explicitly document that the DB supports one writer at a time. Multiple readers are safe. Add a startup check that warns if the DB appears locked.

| Aspect | Detail |
|---|---|
| **Benefits** | Zero code changes. Sets correct expectations. |
| **Disadvantages** | Doesn't prevent the problem — just documents it. Users can still cause corruption. |
| **Risks** | None — pure documentation. |

### Recommendation

**Option A + C** — enable WAL mode if Ladybug supports it, and document the constraint. The CLI is primarily single-user, so aggressive locking is over-engineering. If multi-process access becomes a requirement, revisit Option B.

---

## 8. Migration / Schema Versioning

**Status**: ✅ **Closed** — implemented in v5.3  
**Closure option**: **Option A** (schema version column + migration functions)

### What was implemented

**Schema version tracking** in the Meta table:

| Component | Description |
|---|---|
| **`SCHEMA_VERSION`** | Class constant (= 3). Incremented on each schema change. |
| **`schema_version` Meta entry** | Stores the on-disk version. Read on startup. |
| **`_MIGRATIONS` dict** | Maps version → migration function. Currently: v1→v2, v2→v3. |
| **`_run_schema_migrations()`** | Auto-runs on startup. Idempotent, version-stamped, error-reporting. |

**Migration functions**:

| Version | Migration | Description |
|---|---|---|
| **v1 → v2** | `_migrate_v1_to_v2()` | Chunk table, PART_OF rel, Chunk vector/FTS indexes |
| **v2 → v3** | `_migrate_v2_to_v3()` | FileHash table with concept_id column |

**Fresh database handling**: Version 0 (no migrations needed — `_ensure_schema` creates full schema).

**Error reporting**: Logs version, error, and stops on migration failure — no silent corruption.

**Test coverage**: 10 tests in `test_router_misc.py` (version stamping, idempotency, partial migrations, table creation).

### Remaining gaps (follow-ups)

| Gap | Severity | Notes |
|---|---|---|
| **#8b Rollback migrations** | Low | Downgrade functions for emergency recovery |
| **#8c Schema dump tests** | Low | Test migrations against dumped DBs from each version |
| **#8d CLI schema inspect** | Low | `okf schema --version` command |

### Original analysis (archived)

The original gap analysis proposed three options (A: versioned migrations, B: schema-on-read, C: dump-and-restore). **Option A** was chosen — versioned migrations with the Meta table — because it provides an explicit upgrade path, is idempotent, and scales cleanly to future versions.

---

Don't migrate. Write queries that work against both old and new schemas. Detect missing tables/columns at query time and adapt.

| Aspect | Detail |
|---|---|
| **Benefits** | No migration code. Graceful degradation. |
| **Disadvantages** | Every query must handle multiple schema versions. Code complexity grows with each version. Eventually unmaintainable. |
| **Risks** | High — conditional query logic is a bug magnet. Easy to miss a branch. |

### Option C: Dump-and-restore migration

On version mismatch, export the old DB to OKF markdown, delete the DB, re-create with the new schema, re-import.

| Aspect | Detail |
|---|---|
| **Benefits** | Guaranteed clean state. No partial migrations. Simple to implement. |
| **Disadvantages** | Expensive — full re-embed. Downtime during migration. Loses data not representable in OKF format (e.g., raw image BLOBs if export doesn't handle them). |
| **Risks** | Medium — data loss if export is lossy. Long migration times for large DBs. |

### Recommendation

**Option A** — versioned migrations. The metadata table already exists (`_get_meta`/`_set_meta`). Write migration functions for v4→v5 (add Chunk table, indexes, PART_OF rel). This is the standard approach and scales cleanly to future versions.

---

## 9. Security

**Problem**: No security architecture. Gaps in:
- `--allow-remote-images` SSRF risks
- Untrusted markdown execution
- Database file permissions
- Model cache integrity

**Impact**: SSRF attacks via malicious PDFs or markdown. Supply chain attacks via HuggingFace cache. Data exposure via loose file permissions.

### Option A: URL allowlist for remote images

Restrict `--allow-remote-images` to a configurable allowlist of domains. Block `file://`, `http://0.0.0.0`, and internal IP ranges.

| Aspect | Detail |
|---|---|
| **Benefits** | Prevents SSRF. Simple to implement. Configurable per deployment. |
| **Disadvantages** | Adds configuration complexity. Legitimate internal URLs may be blocked. |
| **Risks** | Low — URL filtering is well-understood. |

### Option B: Sandboxed markdown parsing

Run markdown parsing in a restricted environment. No shell commands, no file system access beyond the bundle root.

| Aspect | Detail |
|---|---|
| **Benefits** | Prevents path traversal attacks. Limits blast radius of malicious markdown. |
| **Disadvantages** | Significant implementation effort. May break legitimate use cases (e.g., symlinks in bundles). |
| **Risks** | Medium — sandboxing adds overhead and complexity. |

### Option C: Security documentation + best practices

Document the threat model, recommend file permissions, and add warnings for `--allow-remote-images`. No code changes.

| Aspect | Detail |
|---|---|
| **Benefits** | Zero code changes. Sets expectations. |
| **Disadvantages** | Doesn't prevent attacks — just warns about them. |
| **Risks** | None — pure documentation. |

### Option D: HuggingFace cache verification

Verify model file hashes on first load. Pin model revisions in `pyproject.toml`.

| Aspect | Detail |
|---|---|
| **Benefits** | Prevents supply chain attacks. Reproducible builds. |
| **Disadvantages** | Model revisions change. Pinning requires manual updates. Hash verification adds startup time. |
| **Risks** | Low — hash verification is standard practice. |

### Recommendation

**Option A + C** as immediate fixes. Add URL allowlist and security documentation. Defer Option B (sandboxing) until there's a real threat model. Option D (cache verification) is worth implementing if the system is deployed in security-sensitive environments.

---

## 10. Observability

**Status**: ✅ **Gap #10a Closed** — structured logging + profiling hooks implemented (v5.4)
**Closure option**: **Option A + C** (stdlib logging + cProfile profiling)

### What was implemented

**`okfgraph/cli.py`** — CLI logging setup:

| Component | Description |
|---|---|
| `_setup_logging()` | Configures stdlib logging with level control (DEBUG/INFO/ERROR) |
| `_teardown_logging()` | Cleans up handlers to avoid leaks |
| `--verbose / -v` | Enable DEBUG logging |
| `--quiet / -q` | Suppress all logging except errors |
| `--log-file` | Write logs to file with 5MB rotation |
| `--profile` | Enable cProfile for the current invocation |

**`okfgraph/router.py`** — Timing instrumentation in `import_bundle()`:

| Phase | Log message |
|---|---|
| Phase 0: Delta detection | `delta: %d changed, %d deleted (%.1fs)` |
| Phase 1: Parse | `parsed %d concept(s)` |
| Phase 2: Encode | `encode: %d texts in %.1fs` |
| Phase 3: Upsert | `upsert: %d concepts in %.1fs` |
| Phase 3.5: Chunk | `chunk: %d concepts in %.1fs` |
| Phase 4: Directories | `directories: %d in %.1fs` |
| Phase 5: Links | `links: %d concepts in %.1fs` |
| Phase 6: Images | `images: %d concepts in %.1fs` |
| Phase 7: Reindex | `reindex: %.1fs` |
| Summary | `import_bundle: %d concept(s) in %.1fs` |

**Test coverage**: 10 tests in `tests/test_logging.py` (logging setup, timing instrumentation, profiling).

### Design decision: stdlib over loguru

Loguru was proposed but rejected in favor of stdlib `logging` because:
- OKFgraph already uses stdlib `logging` (`logger = logging.getLogger(__name__)`)
- Loguru is a third-party dependency for a CLI tool where stdlib is sufficient
- The real issue was inconsistent `print()` usage, not logging configuration complexity
- stdlib `logging` is perfectly capable for structured, debuggable logging

### Remaining gaps (follow-ups)

| Gap | Severity | Notes |
|---|---|---|
| **#10b Prometheus metrics** | Low | Add `/metrics` endpoint or CLI flag for embedding duration histograms |
| **#10c Query latency tracking** | Low | Log search query latency in `search_hybrid()`, `search_chunks()` |
| **#10d Embedding cache hit rates** | Low | Track ONNX model cache hit/miss ratios |

---

---

## 11. Configuration Management

**Problem**: All config is CLI args or Python defaults. No config file for persistent settings. Users must repeat `--db`, `--dim`, `--device`, etc. on every invocation.

**Impact**: Repetitive CLI usage. Hard to standardise settings across a team. No way to version-control deployment config.

### Option A: TOML config file (`okfgraph.toml`)

Read settings from `okfgraph.toml` in the bundle root or `~/.config/okfgraph/`. CLI args override file settings. File settings override defaults.

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

| Aspect | Detail |
|---|---|
| **Benefits** | Persistent settings. Version-controllable. Team-wide standardisation. |
| **Disadvantages** | Adds a config parser. Precedence rules (CLI > file > defaults) must be documented. |
| **Risks** | Low — TOML parsing is straightforward. Precedence is a design decision, not a technical risk. |

### Option B: Environment variables

Support `OKFGRAPH_DB`, `OKFGRAPH_DIM`, `OKFGRAPH_DEVICE`, etc. No config file needed.

| Aspect | Detail |
|---|---|
| **Benefits** | No new dependencies. Works with Docker, CI/CD, and shell profiles. |
| **Disadvantages** | Environment variables are scattered. No single source of truth. Hard to document all supported vars. |
| **Risks** | Low — env vars are additive and backward-compatible. |

### Option C: Status quo — CLI args only

Don't add config file support. Document that users should create shell aliases or wrapper scripts.

| Aspect | Detail |
|---|---|
| **Benefits** | Zero changes. CLI args are explicit and auditable. |
| **Disadvantages** | Repetitive. No standardisation. Hard to manage in CI/CD. |
| **Risks** | None — current behavior is preserved. |

### Recommendation

**Option A + B** — TOML config file with env var overrides. Precedence: CLI > env > file > defaults. This is the standard pattern for CLI tools and covers all deployment scenarios.

---

## 12. Testing Strategy Gaps

**Status**: ✅ **Gap #12a Closed** — missing router unit tests implemented (v5.1)

### What was implemented

Created `tests/test_router_misc.py` (23 tests across 6 test classes):

| Class | Tests | Covers |
|---|---|---|
| `TestMetaAndEpoch` | 7 | `_get_meta()`, `_set_meta()`, `_bump_write_epoch()`, `_indexes_dirty()` |
| `TestReindex` | 4 | `reindex()` with force, clean, and dirty states |
| `TestBrokenLinks` | 4 | `list_broken_links()`, `repair_links()`, broken link lifecycle |
| `TestAdoptExistingDim` | 2 | `_adopt_existing_embedding_dim()` on existing and new DBs |
| `TestPerConceptErrorIsolation` | 4 | `_import_chunks_for_concept()`, per-concept isolation |
| `TestContextWindowWarning` | 2 | Context-window occupancy warning, normal chunks no warning |

Total test suite: **142 passing** across 9 files.

### Remaining gaps (follow-ups)

| Gap | Severity | Notes |
|---|---|---|
| **#12b GPU integration tests** | Low | ✅ **Closed** — 13 GPU integration tests implemented (v5.6) |
| **#12c End-to-end PDF tests** | Low | Requires test PDFs + RapidAI packages pinned |
| **#12d Property-based testing** | Low | `hypothesis` for embedding pipeline numerical correctness |

---

## 13. LLM Tool: `export_bundle` in Architecture

**Status**: ✅ **Closed** — documented in architecture §6 (v5.1)

### What was documented

Added `export_bundle` entry to the §6 LLM Tool Definitions table with:
- Name, description (graph-enriched export with See Also + Cited By)
- Parameter schema: `output_dir` (required), `directory_id`, `concept_type`, `tags` (optional)

No code changes required.

---

## 14. Context Window & Truncation Behavior

**Status**: ✅ **Gap #14a Closed** — context window warning implemented (v5.1)

### What was implemented

Added context-window occupancy warning in `_import_chunks_for_concept()`:

| Check | Behavior |
|---|---|
| **Threshold** | 90% of `tokenizer.model_max_length` |
| **Per-chunk** | Counts tokens via `self.tokenizer.encode(t, add_special_tokens=False)` |
| **Warning** | Logs concept ID, chunk index, token count, % of window, and total window size |
| **Normal chunks** | No warning for chunks well below threshold |

### Remaining gaps (follow-ups)

| Gap | Severity | Notes |
|---|---|---|
| **#14b Hard truncation error** | Low | Raise error if chunk exceeds 100% of context window |
| **#14c Automatic sub-chunking** | Low | Split oversized chunks at paragraph boundaries |

---

## 15. RapidAI Version Pinning

**Status**: ✅ **Gap #15a Closed** — version pinning + runtime warning implemented (v5.4)
**Closure option**: **Option A + C** (pin exact versions + runtime warning)

### What was implemented

**`pyproject.toml` optional-dependencies** (`pdf-ingest` group):

```toml
[project.optional-dependencies]
pdf-ingest = [
    "pdf_oxide>=0.2.1",
    "rapidocr==1.5.2",
    "rapid_latex_ocr==1.0.13",
    "rapid_layout==0.2.0",
    "rapid_table==1.0.3",
]
```

**`okfgraph/ingest/versions.py`** — runtime version checking module:

| Component | Description |
|---|---|
| `_KNOWN_GOOD` dict | Maps package name → (known-good version, display name) |
| `_parse_version()` | Parses version strings into comparable tuples |
| `_is_within_tolerance()` | Checks if installed version is within ±1 minor / ±1 patch of known-good |
| `check_rapid_versions()` | Main check function — returns list of warning messages |
| **Auto-run on import** | Runs automatically when `okfgraph.ingest` is imported |
| **Env var silence** | `OKFGRAPH_INGEST_ALLOW_UNPINNED=1` silences all warnings |

**Exported in `__init__.py`**: `check_rapid_versions` is part of the public API for manual invocation.

**Test coverage**: 8 tests in `tests/test_ingest.py::TestVersionChecking` (version parsing, tolerance logic, env var silence, warn flag).

### Remaining gaps (follow-ups)

| Gap | Severity | Notes |
|---|---|---|
| **#15b Tighten tolerance** | Low | Consider reducing tolerance to exact version only once ecosystem stabilizes |
| **#15c Automated version bump CI** | Low | CI check that fails if `pyproject.toml` pins don't match installed versions |
| **#15d Runtime error (not warning)** | Low | Upgrade from warning to error for major version mismatches |

### Original analysis (archived)

The original gap analysis proposed three options (A: pin exact versions, B: compatibility matrix, C: version detection + runtime warning). **Option A + C** was chosen — pin exact versions in `pyproject.toml` and add a startup check that warns if a different version is installed. This gives both build-time and runtime protection.

---

## Priority Matrix

| Priority | Phase | Items | Rationale |
|---|---|---|---|
| **P0 — Blockers** | Phase 1–2 | ✅ #5, ✅ #8, ✅ #15, ✅ #6d | All resolved (v5.3) |
| **P1 — Important** | Phase 3–5 | ✅ #10, ✅ #5b, ✅ #12b, ✅ #1b, ✅ #5c, ✅ #11a, ✅ #9a, ✅ #7a, ✅ #1d, ✅ #7b, ✅ #9b, ✅ #9d, ✅ #11b, ✅ #12c | All resolved (v5.9) |
| **P2 — Quick Wins** | Phase 6 | #2-if-dirty, #8d, #10c, #14b, #15b | <1 hr total, high visibility |
| **P3 — Reliability** | Phase 7 | #14c, #1, #8b, #15d | Data correctness, production safety |
| **P4 — Observability** | Phase 8 | #10b, #10d, #2-index-status | Operations visibility |
| **P5 — MCP Production** | Phase 9 | #16-auth, #16-rate, #16-http | Only needed if deploying MCP publicly |
| **P6 — Security** | Phase 10 | #9b-sandbox | Only needed for untrusted input |
| **P7 — Testing** | Phase 11 | #12d, #8c | Quality assurance depth |
| **P8 — CI/CD & DX** | Phase 12 | #15c, #6b, #6c, #5d | Developer workflow improvements |

---

## Recommended Implementation Order

```
Phase 1 (Documentation + Low-risk)  ✅ COMPLETE
├── #13  Add export_bundle to §6 tool table          ✅
├── #3   Document batch encoding in §4.3              ✅
├── #4   Document auto-dim in §4.1                    ✅
├── #2   Document index lifecycle in §4.15            ✅
├── #14a Document context window + add warning        ✅
└── #1c  [follow-up] Hash search text for more accurate delta detection

Phase 2 (Core Reliability)  ✅ #6d, #12a, #8a, #5a, #15 COMPLETE
├── #15  Pin RapidAI versions + runtime warning          ✅
├── #6d  Defensive per-concept error isolation            ✅
├── #12a Add missing router unit tests                    ✅
├── #8a  Schema versioning with migration functions       ✅
└── #5a  CLI okf ingest <pdf> command                     ✅

Phase 3 (Feature Completeness)  ✅ COMPLETE
├── #10a Structured logging (stdlib) + profiling hooks   ✅
├── #5b  Router method ingest_pdf() for programmatic use ✅
├── #12b GPU integration tests                            ✅
├── #1b  [follow-up] Directory-level hash aggregation     ✅
├── #5c  LLM tool definition (follow-up)                  ✅
├── #5c  `ingest_thoughts` linting (v5.7)                 ✅
├── #5c  CLI `okf ingest` parity (v5.7)                   ✅
├── #5c  `ingest_pdf` MCP tool (v5.7)                     ✅
└── #5c  Tool parameter parity (v5.7)                     ✅

Phase 4 (Operations)  ✅ COMPLETE
├── #11a TOML config file + env var support                   ✅
├── #9a  URL allowlist for remote images                       ✅
├── #7a  WAL mode + documentation                              ✅
├── #12c End-to-end PDF tests                                  ✅
└── #1d  [follow-up] Soft-delete with recovery window          ✅

Phase 5 (Core Gaps Closure)  ✅ COMPLETE (v5.9)
├── #7b  Filelock write locking (fasteners)                    ✅
├── #9b  Path traversal sandboxing                             ✅
├── #9d  Model cache verification                              ✅
├── #11b TOML schema validation                                ✅
└── #12c End-to-end PDF tests (6 tests)                        ✅

Phase 6 (Quick Wins)  ⏳ DEFERRED — 5 items, all Low effort
├── #2   --if-dirty flag for CLI reindex                        🟢 10 min
├── #8d  okf schema --version command                           🟢 10 min
├── #10c Query latency tracking in search_hybrid/search_chunks  🟢 15 min
├── #14b Hard truncation error (>100% context window)           🟢 15 min
└── #15b Tighten version tolerance to exact match               🟢 5 min

Phase 7 (Reliability & Correctness)  ⏳ DEFERRED — 4 items, Medium effort
├── #14c Automatic sub-chunking at paragraph boundaries         🟡 2–4 hrs
├── #1   Search text hash (Option C) for accurate delta detect  🟡 2–3 hrs
├── #8b  Rollback migrations for emergency recovery             🟡 1–2 hrs
└── #15d Runtime error (not warning) for major version mismatch 🟡 1 hr

Phase 8 (Observability & Monitoring)  ⏳ DEFERRED — 3 items, Medium effort
├── #10b Prometheus metrics endpoint or CLI flag                🟡 2–4 hrs
├── #10d Embedding cache hit/miss ratios                        🟡 1–2 hrs
└── #2   okf index-status command (epoch, dirty, last rebuild)  🟡 1 hr

Phase 9 (MCP Production Readiness)  ⏳ DEFERRED — 3 items, High effort
├── #16-auth  API key / OAuth authentication                    🔴 4–8 hrs
├── #16-rate  Rate limiting for shared server instances          🔴 2–4 hrs
└── #16-http  SSE/HTTP transport for remote clients              🔴 4–8 hrs

Phase 10 (Security Hardening)  ⏳ DEFERRED — 1 item, High effort
└── #9b-sandbox Sandboxed markdown parsing (restricted env)     🔴 4–8 hrs

Phase 11 (Testing Depth)  ⏳ DEFERRED — 2 items, Low effort
├── #12d Property-based testing with hypothesis                 🟢 1–2 hrs
└── #8c  Schema dump tests against each version                 🟢 1 hr

Phase 12 (CI/CD & Developer Experience)  ⏳ DEFERRED — 4 items, Medium effort
├── #15c Automated version bump CI check                        🟡 2–3 hrs
├── #6b  Checkpoint resume from last good state                 🟡 3–4 hrs
├── #6c  Phase recovery CLI (okf recover --phase links)         🟡 2–3 hrs
└── #5d  Better page-level progress callbacks during conversion 🟡 1–2 hrs
```

---

## 16. MCP Server Integration

**Status**: ✅ **Closed** — implemented in v5.6  
**Closure option**: **FastMCP with lifespan context manager**

### What was implemented

**`okfgraph.mcp_server`** — Full MCP server exposing all 16 OKFgraph tools:

| Component | Implementation |
|---|---|
| **Framework** | `mcp.server.fastmcp.FastMCP` |
| **Transport** | stdio (default for MCP clients) |
| **Lifespan** | `@asynccontextmanager` + `GraphContext` dataclass |
| **Tool Registry** | 12 read tools + 4 write tools |
| **Annotations** | `read_only_hint`, `destructive_hint`, `idempotent_hint`, `open_world_hint` |
| **CLI Entry Point** | `okf-mcp` (configured in `pyproject.toml`) |
| **Test Coverage** | 11 tests in `tests/test_mcp_server.py` |

**Lifespan pattern**:

```python
@asynccontextmanager
async def _lifespan(mcp: FastMCP):
    router = OKFRouter(db_path=db_path, ...)
    try:
        yield GraphContext(router=router)
    finally:
        router.close()
```

**Tool access via context injection**:

```python
@mcp.tool()
def search_hybrid(query: str, ctx: Context) -> str:
    router = _get_router(ctx)  # extracts OKFRouter from lifespan context
    results = router.search_hybrid(query)
    return json.dumps(results, default=str, indent=2)
```

### Read Tools (12)

| Tool | Description | `read_only_hint` |
|---|---|---|
| `search_hybrid` | Semantic + keyword search | `True` |
| `traverse` | Navigate relationships | `True` |
| `get_by_id` | Fetch concept by ID | `True` |
| `list_directory` | List directory contents | `True` |
| `search_images` | Find images by text description | `True` |
| `search_chunks` | Chunk-level RRF-fused search | `True` |
| `search_with_context` | Search + graph neighborhood | `True` |
| `search_chunks_with_hub_score` | Rerank by hub score | `True` |
| `expand_with_graph_context` | Expand chunks with neighbours | `True` |
| `get_chunks` | Get chunks for a concept | `True` |
| `reconstruct_document` | Reconstruct from chunks | `True` |
| `find_path` | Shortest path between concepts | `True` |

### Write Tools (4)

| Tool | Description | `read_only_hint` |
|---|---|---|
| `export_bundle` | Export to OKF bundle | `False` |
| `ingest_md` | Import markdown file | `False` |
| `ingest_thoughts` | Store LLM reasoning | `False` |
| `ingest_pdf` | Convert PDF and import | `False` |

### CLI Usage

```bash
# Start MCP server
okf-mcp --db-path ./my_graph.db

# With GPU acceleration
okf-mcp --db-path ./my_graph.db --device cuda

# With custom embedding dimension
okf-mcp --db-path ./my_graph.db --embedding-dim 512
```

### Follow-up Items

- [ ] **MCP-HTTP transport** — Add support for SSE/HTTP transport for remote clients
- [ ] **Authentication** — Add API key or OAuth for production deployments
- [ ] **Rate limiting** — Throttle tool calls for shared server instances

### CLI Parity Fix (v5.7)

The `okf ingest` CLI command had two bugs that made its pipeline diverge from `OKFRouter.ingest_pdf()`:

| Bug | Before | After |
|---|---|---|
| `stage_images_as_okf_assets` | 2 arguments (broken) | 5 arguments: `md_text, img_dir, pdf_path, work_dir, stem` |
| Mordant linting | Missing | Calls `router._lint_converted_md()` with auto-fix |
| `glob` pattern | `glob("*.")` (misses files without dots) | `glob("*")` (matches all files) |

All three PDF ingestion entry points (CLI, router, MCP) now share the same pipeline:
`HybridConverter` → `stage_images_as_okf_assets` (5 args) → `_lint_converted_md` → `import_bundle`.

### `ingest_thoughts` Linting (v5.7)

Added `_lint_converted_md_str()` — an in-memory lint helper that takes a string directly (no temp file) and returns the same dict structure as `_lint_converted_md()`. Used by `ingest_thoughts()` to defensively lint the LLM-provided thoughts text before import.

### Tool Parameter Parity (v5.7)

Three bugs discovered where LLM tool definitions and MCP tool wrappers diverged from router signatures:

| Bug | Before | After |
|---|---|---|
| `search_hybrid` type filter | MCP passed `kwargs["type"]`, LLM defined `"type"` param | MCP passes `kwargs["concept_type"]`, LLM defines `"concept_type"` param — matches router |
| `ingest_pdf` extract_images default | `"default": "true"` (string) | `"default": True` (bool) |
| `traverse` missing defaults | No defaults for `relationship`, `direction`, `depth` | Defaults: `"CONTAINS"`, `"OUTGOING"`, `1` — matches MCP/router |

All 16 tools now have consistent parameter names, types, and defaults across LLM tool definitions, MCP tool wrappers, and router method signatures.

---

## Phase 4 Implementation (v5.8)

### #11a TOML Config + Env Vars

**`okfgraph/config.py`** — Configuration management with three layers:

| Layer | Source | Precedence |
|---|---|---|
| **Defaults** | Hardcoded in dataclasses | Lowest |
| **TOML file** | `okfgraph.toml` in CWD, bundle root, or `~/.config/okfgraph/` | Medium |
| **Env vars** | `OKFGRAPH_*` environment variables | High |
| **CLI args** | Command-line arguments | Highest |

**TOML structure**:

```toml
[database]
path = "okfgraph.db"
dim = 512
wal_mode = true

[embedding]
device = "cuda"
cache_dir = "/mnt/models"
omni_model_id = "jinaai/jina-embeddings-v5-omni-small-retrieval"

[import]
mode = "optional"
batch_size = 64
chunk_size = 512
chunk_overlap = 40
allow_remote_images = false
allowed_image_domains = ["example.com", "cdn.example.com"]
no_chunking = false
```

**Environment variables**: `OKFGRAPH_DB`, `OKFGRAPH_DIM`, `OKFGRAPH_DEVICE`, `OKFGRAPH_CACHE_DIR`, `OKFGRAPH_BUNDLE`, `OKFGRAPH_OMNI_MODEL_ID`, `OKFGRAPH_CHUNK_SIZE`, `OKFGRAPH_CHUNK_OVERLAP`, `OKFGRAPH_BATCH_SIZE`, `OKFGRAPH_MODE`, `OKFGRAPH_WAL_MODE`, `OKFGRAPH_ALLOW_REMOTE_IMAGES`, `OKFGRAPH_ALLOWED_IMAGE_DOMAINS`, `OKFGRAPH_NO_CHUNKING`.

### #7a WAL Mode

**SQLite WAL (Write-Ahead Logging)** enables concurrent reads during writes:

- Enabled via `--wal-mode` CLI flag or `wal_mode = true` in TOML config
- Sets `PRAGMA journal_mode=WAL` on the Ladybug connection
- Reads don't block writes; writes are serialized by SQLite's internal locking
- Adds `-wal` and `-shm` companion files to the database directory

### #9a URL Allowlist

**SSRF prevention for remote images**:

- `allowed_image_domains` parameter on `OKFRouter` and `load_image_bytes()`
- `_domain_allowed()` helper validates domains against the allowlist
- Supports exact matches (`example.com`) and wildcard subdomains (`*.example.com`)
- Blocks private IP ranges (`10.`, `192.168.`, `172.16-31.`) and localhost
- Configurable via `--allowed-image-domains` CLI flag or TOML config

### #1d Soft-Delete with Recovery

**DeletedConcept table** preserves purged concepts for 24-hour recovery:

```cypher
CREATE NODE TABLE DeletedConcept (
    id STRING PRIMARY KEY,          -- unique deleted_<concept_id>_<timestamp>
    original_id STRING,              -- original concept ID
    title STRING,                    -- preserved title
    body STRING,                     -- preserved markdown body
    deleted_at STRING,               -- ISO timestamp
    type STRING,                     -- concept type
    tags STRING                      -- JSON array of tags
);
```

**Methods**:

| Method | Role |
|---|---|
| `_soft_delete_concept(id)` | Move concept to DeletedConcept, then hard purge |
| `_recover_concept(id)` | Restore concept from DeletedConcept (within window) |
| `list_deleted_concepts()` | List all soft-deleted concepts with recovery status |
| `purge_deleted_concepts(older_than)` | Permanently delete expired concepts |

**CLI commands**:

| Command | Description |
|---|---|
| `okf deleted-list` | List soft-deleted concepts with recovery status |
| `okf deleted-recover <id>` | Recover a soft-deleted concept |
| `okf deleted-purge [--older-than N]` | Permanently delete expired concepts |

**Schema migration**: v4 → v5 adds DeletedConcept table (idempotent).

---

## Phase 5 Implementation (v5.9)

### #7b Filelock Write Locking

**`fasteners.InterProcessLock`** for multi-process write safety:

- Lock file created alongside the DB (`okfgraph.db.lock`)
- 5-minute timeout for lock acquisition
- All write operations wrapped: `import_bundle`, `ingest_md`, `ingest_thoughts`, `reindex`, soft-delete methods
- Lock released on router `close()`

**Context manager**:

```python
@contextmanager
def _write_lock_ctx(self):
    """Acquire file-based lock before any write operation."""
    acquired = self._write_lock.acquire(timeout=self._write_lock_timeout)
    if not acquired:
        raise RuntimeError("Write lock acquisition timed out")
    try:
        yield
    finally:
        self._write_lock.release()
```

### #9b Path Traversal Sandboxing

**`okfgraph/security.py`** module with:

| Function | Purpose |
|---|---|
| `is_path_safe(path, root)` | Validates file paths are within bundle root |
| `is_private_ip(host)` | Blocks private/internal IP ranges |
| `validate_image_src(src, root, ...)` | Comprehensive image source validation |
| `ModelCacheVerifier` | SHA-256 hash verification for model cache files |

**Integrated into `load_image_bytes()`**:

- Blocks `file://` URLs (SSRF risk)
- Validates local paths are within bundle root via `_is_path_within()`
- Supports `bundle_root` parameter for path validation

### #9d Model Cache Verification

**`ModelCacheVerifier`** class for supply chain security:

```python
verifier = ModelCacheVerifier()
verifier.register_model("jinaai/jina-embeddings-v5", {
    "model.onnx": "sha256:abc123...",
})
failures = verifier.verify_all(model_files)
```

- Register expected hashes for model files
- Verify files on first load, cache results for speed
- Warn on hash mismatches (first-time loads trusted)

### #11b TOML Schema Validation

**`validate()` methods** on all config dataclasses:

| Section | Validations |
|---|---|
| `DatabaseConfig` | path non-empty, dim in [32, 1024], recommended Matryoshka dims |
| `EmbeddingConfig` | device in [cpu, cuda, mps, auto], cache_dir absolute |
| `ImportConfig` | mode in [text, optional, omni], batch_size in [1, 256], chunk_size in [64, 8192] |
| `OKFConfig` | aggregates all section validations |

Validation warnings logged on config load (non-blocking — CLI args can override).

### #12c End-to-End PDF Tests

**`tests/test_pdf_e2e.py`** with 6 tests:

| Test | Verifies |
|---|---|
| `test_pdf_ingest_creates_concept` | Full pipeline: convert → stage → lint → import |
| `test_pdf_ingest_output_only` | PDF conversion without auto-import |
| `test_pdf_ingest_nonexistent_file` | FileNotFoundError for missing PDF |
| `test_pdf_ingest_with_progress_callback` | Page progress callback invocation |
| `test_router_pipeline_has_linting` | Router has `_lint_converted_md` and `_lint_converted_md_str` |
| `test_router_pipeline_has_image_staging` | `stage_images_as_okf_assets` has 5-arg signature |

---

## Phase 6 Plan (Quick Wins)

**Estimated effort**: <1 hr total  
**Dependencies**: None  
**Risk**: None — all additive changes

### #2 `--if-dirty` flag for CLI reindex

Add `--if-dirty` to `okf reindex` so it only rebuilds when `write_epoch != indexed_epoch`:

```python
@click.command()
@click.option("--if-dirty", is_flag=True, help="Only reindex if indexes are dirty")
def reindex(if_dirty):
    if if_dirty and not router._indexes_dirty():
        click.echo("Indexes are up-to-date — skipping")
        return
    router.reindex()
```

### #8d CLI schema inspect

Add `okf schema --version` command:

```python
@click.command()
@click.option("--version", is_flag=True, help="Show schema version")
def schema(version):
    if version:
        ver = router._get_meta("schema_version", "0")
        click.echo(f"Schema version: {ver}")
```

### #10c Query latency tracking

Add timing instrumentation to `search_hybrid()` and `search_chunks()`:

```python
start = time.time()
results = self._search_hybrid_inner(query, ...)
logger.debug("search_hybrid: %.3fs (%d results)", time.time() - start, len(results))
```

### #14b Hard truncation error

Raise `ValueError` if a chunk exceeds 100% of context window (currently only warns at 90%):

```python
if tokens > self.tokenizer.model_max_length:
    raise ValueError(
        f"Chunk {chunk_idx} for concept {concept_id} exceeds context window: "
        f"{tokens} > {self.tokenizer.model_max_length}"
    )
```

### #15b Tighten version tolerance

Change `_is_within_tolerance()` from ±1 minor / ±1 patch to exact match only:

```python
def _is_within_tolerance(installed: tuple, expected: tuple) -> bool:
    return installed == expected  # exact match
```

---

## Phase 7 Plan (Reliability & Correctness)

**Estimated effort**: 6–11 hrs total  
**Dependencies**: Phase 6 (#14b)  
**Risk**: Medium — changes to chunking/delta logic need careful testing

### #14c Automatic sub-chunking

When a chunk exceeds the context window, split at paragraph boundaries instead of silently truncating:

1. Detect oversized chunk (tokens > `model_max_length`)
2. Split at nearest paragraph break (`\n\n`)
3. Recursively sub-chunk until all pieces fit
4. Create hierarchical PART_OF relationships

### #1 Search text hash (Option C)

Hash the actual string that gets embedded, not the file bytes:

1. In Phase 1 (Parse), compute SHA-256 of the search text (body after frontmatter stripping)
2. Store in new `SearchHash` table alongside `FileHash`
3. Compare `SearchHash` instead of `FileHash` for delta detection
4. Catches frontmatter-only changes that don't affect embeddings

### #8b Rollback migrations

Add downgrade functions for emergency recovery:

```python
_MIGRATION_ROLLBACKS = {
    5: _rollback_v5_to_v4,  # drop DeletedConcept table
    4: _rollback_v4_to_v3,  # drop FileHash.concept_id column
}
```

### #15d Runtime error for major version mismatch

Upgrade from warning to error when major version differs:

```python
if installed[0] != expected[0]:
    raise RuntimeError(
        f"{pkg} major version mismatch: {installed} != {expected}. "
        f"This may cause silent data corruption."
    )
```

---

## Phase 8 Plan (Observability & Monitoring)

**Estimated effort**: 4–7 hrs total  
**Dependencies**: Phase 6 (#10c)  
**Risk**: Low — additive instrumentation

### #10b Prometheus metrics

Add `/metrics` endpoint or CLI flag for embedding duration histograms:

```python
from prometheus_client import Histogram, Counter, generate_latest

EMBED_DURATION = Histogram("okf_embed_duration_seconds", "Embedding duration")
SEARCH_COUNT = Counter("okf_search_total", "Search queries", ["type"])
```

### #10d Embedding cache hit/miss ratios

Track ONNX model cache hit/miss:

```python
logger.debug("model_cache: hits=%d, misses=%d, ratio=%.1f%%",
             hits, misses, hits / (hits + misses) * 100)
```

### #2 `okf index-status` command

Reports epoch, dirty state, and last rebuild time:

```python
@click.command()
def index_status():
    write_epoch = router._get_meta("write_epoch", "0")
    indexed_epoch = router._get_meta("indexed_epoch", "0")
    dirty = write_epoch != indexed_epoch
    click.echo(f"write_epoch: {write_epoch}")
    click.echo(f"indexed_epoch: {indexed_epoch}")
    click.echo(f"dirty: {dirty}")
```

---

## Phase 9 Plan (MCP Production Readiness)

**Estimated effort**: 10–20 hrs total  
**Dependencies**: None (independent of core gaps)  
**Risk**: Medium — auth/rate-limiting adds complexity

### #16-auth API key / OAuth

Add authentication to MCP server:

```python
@mcp.middleware
def auth_middleware(request):
    api_key = request.headers.get("X-API-Key")
    if api_key != EXPECTED_KEY:
        raise HTTPException(401, "Unauthorized")
```

### #16-rate Rate limiting

Throttle tool calls for shared server instances:

```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@mcp.tool()
@limiter.limit("10/minute")
def search_hybrid(query: str):
    ...
```

### #16-http SSE/HTTP transport

Add support for HTTP transport alongside stdio:

```python
from mcp.server.sse import SSETransport
app = FastAPI()
transport = SSETransport()
```

---

## Phase 10 Plan (Security Hardening)

**Estimated effort**: 4–8 hrs  
**Dependencies**: None  
**Risk**: Medium — sandboxing may break legitimate use cases

### #9b-sandbox Sandboxed markdown parsing

Run markdown parsing in restricted environment:

1. No shell command execution
2. File system access limited to bundle root
3. No network access during parsing
4. Symlink resolution disabled or restricted

---

## Phase 11 Plan (Testing Depth)

**Estimated effort**: 2–3 hrs total  
**Dependencies**: None  
**Risk**: None

### #12d Property-based testing

Use `hypothesis` for embedding pipeline numerical correctness:

```python
from hypothesis import given, strategies as st

@given(st.text(min_size=1, max_size=1000))
def test_embedding_determinism(text):
    emb1 = router.encode([text])[0]
    emb2 = router.encode([text])[0]
    assert np.allclose(emb1, emb2, atol=1e-6)
```

### #8c Schema dump tests

Test migrations against dumped DBs from each version:

```python
@pytest.mark.parametrize("schema_version", [1, 2, 3, 4, 5])
def test_migration_from_version(schema_version):
    db_path = f"test_dbs/v{schema_version}.db"
    router = OKFRouter(db_path=db_path)
    assert router._get_meta("schema_version") == str(SCHEMA_VERSION)
```

---

## Phase 12 Plan (CI/CD & Developer Experience)

**Estimated effort**: 7–12 hrs total  
**Dependencies**: None  
**Risk**: Low

### #15c Automated version bump CI

CI check that fails if `pyproject.toml` pins don't match installed versions.

### #6b Checkpoint resume

Resume from last good state after crash:

1. After each phase completes, write checkpoint to `Meta` table
2. On restart, detect incomplete phase and resume from checkpoint
3. Rollback partial phase data before retry

### #6c Phase recovery CLI

`okf recover --phase links` to re-run a specific phase.

### #5d Progress callbacks

Better page-level progress reporting during PDF conversion.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Schema migration fails on existing DB | Medium | High | ✅ **Mitigated** — tested migrations (10 tests). Add rollback function (Gap #8b). |
| Delta detection false-negative (misses change) | Low | High | ✅ **Mitigated** — SHA-256 is collision-resistant. Add manual `--force-reembed` flag. |
| PDF ingest breaks on RapidAI update | High | Medium | ✅ **Mitigated** — versions pinned in `pyproject.toml`, runtime check in `okfgraph.ingest.versions`. |
| Concurrent writes corrupt DB | Low | High | ✅ **Mitigated** — WAL mode enabled via `--wal-mode` or TOML config (v5.8). |
| Long chunks silently truncated | Medium | Medium | Warning at 90% threshold (Gap #14). Auto-subchunk in Phase 3. |
| SSRF via remote images | Low | High | ✅ **Mitigated** — URL allowlist + path traversal sandboxing (v5.9). |
| Partial import leaves DB inconsistent | Medium | Medium | Per-concept error isolation (Gap #6). Failure report at end of import. |
| Purge deletes shared assets | Low | Medium | ✅ **Mitigated** — orphan check preserves assets referenced by multiple concepts. |

---

## Appendix: Current Test Coverage by Module

| Module | Tests | Coverage | Gaps |
|---|---|---|---|
| `router.py` — chunking | `test_chunking.py` (13) | High | GPU path, context window truncation |
| `router.py` — reconstruction | `test_reconstruction.py` (9) | Medium | Large document reconstruction |
| `router.py` — chunk search | `test_chunk_search.py` (9) | Medium | Graph filter combinations |
| `router.py` — graph enrichment | `test_graph_enrichment.py` (11) | Medium | Hub score edge cases |
| `router.py` — integration | `test_integration.py` (16) | Medium | Re-import delta detection |
| `router.py` — export | `test_export_compliance.py` (13) | High | Multi-directory export |
| `router.py` — delta/purge | `test_delta.py` (21) | High | Directory-level hash aggregation, soft-delete recovery |
| `router.py` — misc (Gap #12a) | `test_router_misc.py` (33) | High | GPU path, omni cross-modal, content hash dedup, schema migration |
| `router.py` — general | `test_router.py` (41) | High | Smoke, cache, device selection, tools, ingest_thoughts linting |
| `ingest/` — config/engine/versions | `test_ingest.py` (39) | Medium | End-to-end PDF conversion, version checking, ingest_pdf |
| `test_logging.py` | 10 | High | Logging setup, timing instrumentation, profiling |
| `test_mcp_server.py` | `test_mcp_server.py` (11) | High | Server creation, tool registry, schemas, annotations |
| `test_security.py` | 24 | High | Path traversal, SSRF, private IP, model cache verification |
| `test_config.py` | 25 | High | TOML schema validation, env var precedence |
| `test_pdf_e2e.py` | 6 | Medium | End-to-end PDF ingestion pipeline |
| `images.py` | `test_images.py` | Unknown | Content hash dedup |
| `cli.py` | `test_cli.py` | Unknown | PDF ingest command, --purge flag |
| `converter.py` | `test_converter.py` (2) | Medium | Asset staging, round-trip (PySide6 stubbed) |
| `search_browser.py` | `test_search_browser.py` (9) | Medium | Backend helpers (PySide6 stubbed) |

**Total**: 335 tests across 19 files (304 core + 15 GPU integration + 6 PDF e2e + 24 security + 25 config + 1 skipped symlink).

---

*This document is a living artifact. Update the severity and recommendations as the codebase evolves.*
