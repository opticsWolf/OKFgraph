# Changelog

All notable changes to OKFgraph are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com); entries are grouped from
commit history, newest first.

## [0.2.18] — 2026-09-13

### Fixed
- Image ingest hardening (found live on the 44-doc family-kb import:
  three successive deaths right after the links phase, one with a torn
  WAL):
  - ladybug 0.20.3 segfaults (access violation) on
    `WHERE NOT EXISTS { }` + `DETACH DELETE` against the HNSW-indexed
    ImageAsset table when it runs after an earlier image-ingest
    transaction in the same session (deterministic two-doc recipe;
    isolated statements survive). Orphan cleanup now counts edges then
    plain-deletes — proven safe against the same recipe. Revisit on a
    ladybug upgrade.
  - Phantom refs (`![alt](rel)`, `![x](<id>)` mentioned in prose) no
    longer grow asset rows: unresolvable refs are skipped unless the
    asset already exists with real bytes (DB-only preservation kept,
    remote srcs exempt).
  - Reused assets re-link their concept: concept replacement drops
    INCLUDES_ASSET edges and reuse previously wrote nothing, silently
    unlinking images on every doc edit. Asset lookup is now node-direct
    by planned id (edge joins go blind after replacement).

## [0.2.17] — 2026-09-13

### Fixed
- Length-bucketed batch encoding for bundle imports: concept
  search_texts span three orders of magnitude, and naive batches of 32
  padded every sequence to the batch max — one giant doc inflated 31
  small ones into a 32x8192-token forward (tens of GB, tens of minutes
  on CPU; found live on the 44-doc family-kb import). Texts now encode
  shortest-first in buckets of 8 with per-bucket progress logs.
  Stored vectors are unchanged (padding is masked out; only compute
  order changed).

## [0.2.16] — 2026-09-13

Phase 1 (plan-multi-root-detach): `okf detach` ends the mirror
relationship — the database becomes the artifact. A sources-vs-graph
fidelity check runs by default (mismatches abort; untracked/source-only
files are WILL-NOT-SURVIVE and require --force); the
FileHash/DirHash/DeletedPath baseline is dropped and provenance recorded
(schema v7 `SourceRoot` + `Meta` flags, auto-migrated on open).

### Added
- `okf detach [--bundle DIR] [--no-verify] [--force]` (CLI-only).
- `--force` on `import` / `ingest`: re-attaches when the bundle matches
  the recorded source tree (baseline rebuilt, detached state cleared);
  allows explicitly addressed single-file / thought writes without
  re-attaching. PDF auto-import cannot re-attach (its temp dir never
  matches provenance). The graph's own db files are excluded from the
  source-only walk.
- `doctor` reports detach state (informational; no score/`--strict`
  impact).

### Changed
- Schema v6 → v7: `SourceRoot(alias, path, file_count, last_seen)`
  provenance table; `Meta` gains `detached` / `detached_at` (both INT64).
- Mirror writes on a detached graph refuse with a clear error unless
  forced; reads/search/traverse/export/recover/reindex/`deleted-*` keep
  working.

## [0.2.15] — 2026-09-13

Phase 0 (plan-multi-root-detach): import crash-consistency and true
deletion detection. An interrupted `import --all` previously committed
file/dir hashes *before* parsing — the next run trusted them and skipped
everything, leaving a graph with hash rows and zero concepts (observed
live with 44 hashes / 0 concepts). Hashes are now persisted only after
concepts commit, so a crash degrades to wasted re-import work, never a
silent wedge.

### Fixed
- Hash-before-commit wedge: `DirHash`/`FileHash` are written post-commit
  (failed parses keep no hash state and retry; their directories are
  re-walked next run).
- Per-file deletions were invisible unless a whole directory emptied:
  surviving-but-changed directories now diff stored vs current file
  lists. Deletions persist as `DeletedPath` tombstones, so they survive
  no-purge runs until a later `--purge-deleted` consumes them.
- Purge log reported `len(deleted)` instead of the actual purged count.
- PDF auto-import (the MCP default) imported from a `TemporaryDirectory`
  with `bundle_root` pointed at it, clobbering the real bundle's
  top-level delta row (mass silent re-encode on the next import) and
  leaking temp keys. Work-dir imports now bypass the delta baseline.
- `okf doctor` gains an `orphan_hash` error rule (hash rows with no
  concept — the wedge signature) and `okf doctor --fix` clears them plus
  their parent dir rows, so a wedged pre-0.2.15 DB heals on next import.
- Removed dead `_changed_files` detector (tests-only); one delta
  implementation remains.

## [0.2.14] — 2026-09-13

Unified embedding dimension: **512 is the default everywhere**. `okf-mcp`
previously defaulted to `--embedding-dim 1024` while the router/CLI
defaulted to 512, so a graph created over MCP opened requesting 512 on
CLI (and vice versa) — harmless via dim adoption, but confusing and one
stale-closure bug away from a real mismatch (fixed in 0.2.13's session
factory). The dim stays freely settable at creation on both surfaces
(`--dim` / `--embedding-dim`, full Matryoshka ladder
32/64/128/256/512/768/1024); existing DBs keep their on-disk dim.

### Changed
- `okf-mcp --embedding-dim` default 1024 → 512 (signature, flag, help).
- `okfgraph.toml` dim validation accepts the full ladder (32/64 now
  allowed, matching `OKFRouter.ALLOWED_DIMS`; off-ladder stays a router
  warning, not an error).
- `--dim` / `--embedding-dim` help texts list the ladder.

## [0.2.13] — 2026-09-13

Batch G hardening: the new error-path tests caught a completely broken
soft-delete path (never executed since extraction), plus the first
wire-level MCP coverage and a scheduled full-suite workflow.

### Fixed
- Soft-delete lifecycle was dead code: missing `time`/`json` imports
  (`NameError` on first call), SQLite `INSERT INTO` against the Cypher
  store, and fresh DBs never getting the `DeletedConcept` table at all.
  All fixed; `deleted-list`/`-recover`/`-purge` work end to end.
- Recovered concepts kept only title/body/type/tags — the vector and
  metadata were silently dropped. `DeletedConcept` now carries a
  full-fidelity `snapshot` (schema v6; v5 DBs migrate on open via ALTER
  or CREATE branch), so recover restores embeddings byte-identically.
- `purge_deleted_concepts` counts tombstone removals (the concept body
  is usually already gone — soft-delete purges it).

### Added
- `tests/test_mcp_stdio.py`: wire-level MCP test — the real `okf-mcp`
  entry point as a child process, raw JSON-RPC over stdio (handshake,
  5-tool list, root-listing `traverse`, unknown-tool error). Model-free,
  in the CI fast list.
- `tests/test_error_paths.py` (14 tests): links ambiguity/precedence/
  normalization, schema re-init/dim-adoption/v5→v6 both branches, purge
  lifecycle/idempotency, dylib pass-through, corrupt-ONNX fail-fast.
  Model-free, in the CI fast list.
- `.github/workflows/full.yml`: weekly (Sun 02:00 UTC, default branch)
  + dispatchable full non-slow suite with warmed model cache. Deliberately
  weekly, not nightly — the frozen lockfile gives daily re-runs no drift
  signal.

## [0.2.12] — 2026-09-13

Spin-off Phase 3 (docs/plan-embed-spinoff.md): the embedding engine is no
longer built in-tree. `rust/okf-embed/` is deleted and okfgraph now depends
on the released external crate — `embroider` (github.com/opticsWolf/embroider,
crates.io rlib + PyPI wheels, version floor `>=0.1,<0.2`). Same Python
surface, same vectors, no Rust toolchain needed to build or use okfgraph.

### Changed
- Dependency swap: `okf-embed>=0.1` (path dep) → `embroider>=0.1,<0.2`
  (plain PyPI pin); `[tool.uv.sources]` removed; `uv.lock` re-resolved.
- Module import `okf_embed` → `embroider` in the router wiring, lazy
  encoder, and all tests — class names, signatures, and behavior
  unchanged (Jina contract frozen; golden parity tests pass against the
  0.1.3 wheel byte-identically).
- `rust/okf-embed/` deleted from the tree (in-tree copy frozen at 0.2.0
  in git history; live development is in the embroider repo).
- CI: `cargo test` job removed (nothing Rust left in-tree); Rust
  toolchain steps dropped from the fast job; Requires-Dist sanity check
  now pins `embroider>=`.
- Release workflow: okf-embed wheel build/publish jobs removed —
  okfgraph releases now publish only the okfgraph sdist+wheel; embroider
  wheels live from their own repo/releases.
- Docs: README + `docs/harness-integration.md` repointed (external
  wheel, no path dep, embroider repo owns Rust tests).

## [0.2.11] — 2026-09-13

ONNX-runtime alignment with bobine (`docs/plan-onnx-alignment.md`): robust
cross-platform runtime discovery, graceful accelerator fallback, lazy
session init, and air-gapped model loading. Ships with `okf-embed 0.2.0`
wheels (Rust changes in every phase below).

### Added
- Cross-platform ONNX Runtime discovery: explicit `ORT_DYLIB_PATH` still
  wins, otherwise pip-installed `onnxruntime`/`onnxruntime-gpu` is resolved
  on Windows (`capi/onnxruntime.dll`), macOS
  (`capi/libonnxruntime.dylib`), and Linux (preferred versioned
  `capi/libonnxruntime.so.*`, fallback `capi/libonnxruntime.so`). GPU DLL
  warming and Windows DLL-directory setup are best-effort and never fatal.
  The resolved runtime is exposed as `OKFRouter.ort_dylib` /
  `EmbeddingEngine.ort_dylib` before the native module is imported.
- Explicit local model files: `JinaV5.open_files(onnx_path, tokenizer_path)`
  and `JinaTokenizer.open_files(tokenizer_path)` load with zero network
  access (air-gapped / reproducible installs). `OKFRouter(model_path=...,
  tokenizer_path=...)` threads them through the lazy factories; the pair
  must be given together, missing files fail fast at construction, and an
  external-data sidecar must sit next to the ONNX file. Vectors are
  identical to HF acquisition (pinned by test).
- New `JinaTokenizer` handle in `okf-embed`: exact token counts without the
  ONNX session, so budgeted reads and the context-window guard never warm
  the session. Counts are identical to the session tokenizer path.

### Changed
- Lazy text-encoder initialization: `OKFRouter` construction no longer
  downloads the model or builds the ONNX session. The `okf-embed` wheel
  import is still validated eagerly, and an invalid `device` still fails at
  construction — but `JinaV5.open` waits for the first real encode, so
  model-free commands (PPR search, budgeted reads, diff, doctor) stay cold.
  The CUDA fallback warning now fires on first encode instead of at
  construction. Open failures are cached and re-raised (fail fast once).
- `okf-embed`: clone-and-fallback provider application (bobine pattern) —
  unknown provider names warn and are skipped, registration failure
  degrades to CPU; EP registration features
  (CUDA/ROCm/DirectML/OpenVINO/CoreML/TensorRT/half) enabled. The public
  device surface is unchanged (`auto`/`cpu`/`cuda`) and embedding numerics
  are untouched.
- `okf-embed`: explicit optional `extension-module` Cargo feature — pure
  Rust crate by default, Python extension only for wheel builds (maturin
  enables the feature; `cargo test` stays link-clean).
- Session/threading policy measured and kept (Level3, intra=phys/2,
  inter=1): fastest of the tested configs; results in
  `rust/okf-embed/README.md`.

### Fixed
- CUDA availability probing stays on the EP availability check: a
  session-builder registration probe reported CUDA on CPU-only runtimes
  (ort 2.0.0-rc.13 returns `Ok` from `with_execution_providers` even when
  the loaded library has no CUDA EP). The clone-and-fallback provider
  helper is retained for genuine registration failures.

## [0.2.7] — 2026-09-13

### Fixed
- Declare both package READMEs as published project descriptions: the root
  `README.md` for `okfgraph` and `rust/okf-embed/README.md` for `okf-embed`
  (released as `okf-embed 0.1.1` so PyPI picks up the corrected metadata).

## [0.2.6] — 2026-09-13

### Changed
- `okf ingest` now requires an explicit `--kind md|pdf|thoughts`. This removes
  the unsafe implicit PDF default and aligns the CLI discriminator with the
  required MCP `ingest` `kind` parameter.

### Added
- Rust unit tests for `okf-embed` (`cargo test`): device parsing, task-
  prefix idempotence, the L2 → truncate → re-normalise Matryoshka math,
  contract constants, and `open()` validation firing before any network
  access. Pooling math extracted into `task_prefixed` / `l2_truncate`
  helpers (behavior pinned by the Python parity suite — wheel rebuilt and
  re-verified). CI runs `cargo test --locked` on Ubuntu.

### Fixed
- Replaced three-clause `MERGE` hierarchy writes with equivalent separate
  node and edge statements after isolating a deterministic ladybug 0.20.3
  access violation on repeated sibling imports. `test_integration.py` now
  passes (11/11).
- Updated stale CLI contract tests for the consolidated five-verb CLI and
  made the CUDA fallback test deterministic on machines without a CUDA
  execution provider.

## [0.2.4] — 2026-09-12

### Added
- `okf lint [DIR] [--json]` — pre-import bundle gate: parse errors, missing
  `type:`/`title:` (warnings; import synthesizes them), dangling path links
  and unresolvable/ambiguous wikilinks (errors). No database, no model load.
  Exit codes: 0 clean / 1 errors / 2 bad directory (`--json` machine form).

## [0.2.3] — 2026-09-12

### Added
- `resource:` URI sanitization at parse time — credentials in
  `scheme://user:pass@host` rewritten to `***@` so secrets never persist into
  the graph, exports, or vaults. Covers bundle import, single-file import,
  and all ingest kinds.

## [0.2.2] — 2026-09-12

### Fixed
- Reserved filenames: bulk import, diff, and delta detection now skip
  `index.md` (case-insensitive). Re-importing an exported bundle no longer
  creates `*/index` noise concepts; export → re-import round-trips with a
  clean drift diff. Explicit single-file import of `index.md` still works.

## [0.2.1] — 2026-09-12

### Changed
- Pydantic hardening: `ConfigDict(extra="allow")`, tags coercion
  (`tags: foo` no longer crashes), model-owned serialization surfaces
  (`ConceptModel.public_dict()`, `export_frontmatter()`).
- MCP `read` and CLI `get` no longer emit the 1024-float embedding vector.
- Timestamp validator uses native `datetime.fromisoformat` (garbage input
  raises a proper `ValidationError` instead of guessing).

## [0.2.0] — 2026-09-12

### Added
- Rust embedding engine `okf-embed` (`rust/okf-embed`): Jina v5 via ONNX
  Runtime (ort), last-token pooling, Matryoshka truncation; replaces the
  Python ONNX stack entirely (numerics pinned by parity tests).
- `DocumentConverter` seam (`okfgraph/components/converters.py`) with
  `BobineConverter` as the default PDF converter; bring-your-own converter
  supported, no legacy fallback.
- Model-free retrieval round (`docs/plan-retrieval-roundup.md`):
  - lexical seeds + exact Personalized PageRank — `search --rank ppr`
    (MCP `rank` param), zero embedding-model load;
  - token-budgeted reads — `read --max-tokens` (PPR-ranked neighbours);
  - structural `diff` — snapshot (dir vs dir) and drift (graph vs dir)
    with CI exit codes and `--json`;
  - scored `doctor` (0–100) with safe `--fix` that never touches
    `reviewed: true` concepts;
  - Obsidian-compatible wikilinks (`[[name]]` via uid → aliases → title →
    stem; ambiguous names never resolve) and `export --flavor obsidian`.
- Harness integration: `.mcp.json`, harness-neutral skills
  (`okfgraph-mcp`, `okfgraph-cli`, `okfgraph-ingest`),
  `docs/harness-integration.md`.
- CI (frozen `uv sync`, no-model test subset, wheel metadata check) and
  PyPI release pipeline (maturin wheels, OIDC trusted publishing, tag
  version guard).

### Changed
- Dependency surface slimmed to a fraction: `ladybug==0.20.3` (pinned),
  `okf-embed`, `onnxruntime==1.29.0` (single shared ORT, `ORT_DYLIB_PATH`-
  overridable), `mordant>=0.9`, `mcp>=2.0`, `pydantic`, `pyyaml`, `numpy`,
  `python-frontmatter`, `fasteners`. PDF via optional `bobine` extra;
  images via optional `omni` extra.
- MCP server migrated to `mcp >= 2.0` (`MCPServer`, `ToolAnnotations`).
- Agent surface consolidated: **16 MCP tools → 5**
  (`search`, `read`, `traverse`, `ingest`, `export_bundle`); CLI mirrors
  the same 5 verbs plus maintenance commands.
- CLI token surface slimmed (top-level help 5987 → 2251 tokens; global
  options documented once and hidden per-command).
- Self-routing MCP tool descriptions (skills optional, not required).

### Removed
- Legacy `okfgraph/ingest/` ONNX/Rapid PDF engine (replaced by the bobine
  seam).
- `optimum`, `transformers`, `torch` dependencies.
- Granular CLI commands (`search-chunks`, `hub-search`, `path`, `chunks`,
  `reconstruct`, `search-images`, `context`, `siblings`, `ancestry`,
  `get` — folded into `search`/`read`/`traverse`/`export`).

## [0.1.0] — 2026-06-25 to 2026-07-10

Initial development (per commit history): Ladybug-backed OKF knowledge
graph with hybrid RRF search (vector + FTS), mordant chunking with
graph-aware retrieval (hub rerank, context expansion, reconstruction),
three-mode image ingestion (`text`/`optional`/`omni`) with content-hash
dedup, incremental import with SHA-256 delta detection and `--purge`,
OKF-compliant export with See Also/Cited By enrichment and index files,
ONNX/Rapid PDF OCR engine, OpenAI-compatible LLM tool definitions + MCP
server, CLI, TOML config, WAL mode, SSRF guards, and router refactor into
injected components (`okfgraph/components/`).

[Unreleased]: changes land on `dev` and merge to `main` per feature with a
`0.0.1` version bump each.
