# Changelog

All notable changes to OKFgraph are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com); entries are grouped from
commit history, newest first.

## [Unreleased]

### Fixed
- CUDA availability probing stays on the EP availability check: a
  session-builder registration probe reported CUDA on CPU-only runtimes
  (ort 2.0.0-rc.13 returns `Ok` from `with_execution_providers` even when
  the loaded library has no CUDA EP). The clone-and-fallback provider
  helper is retained for genuine registration failures.

### Added
- Explicit local model files: `JinaV5.open_files(onnx_path, tokenizer_path)`
  and `JinaTokenizer.open_files(tokenizer_path)` load with zero network
  access (air-gapped / reproducible installs). `OKFRouter(model_path=...,
  tokenizer_path=...)` threads them through the lazy factories; the pair
  must be given together, missing files fail fast at construction, and an
  external-data sidecar must sit next to the ONNX file. Vectors are
  identical to HF acquisition (pinned by test).

### Changed
- Lazy text-encoder initialization: `OKFRouter` construction no longer
  downloads the model or builds the ONNX session. The `okf-embed` wheel
  import is still validated eagerly, and an invalid `device` still fails at
  construction — but `JinaV5.open` waits for the first real encode, so
  model-free commands (PPR search, budgeted reads, diff, doctor) stay cold.
  The CUDA fallback warning now fires on first encode instead of at
  construction. Open failures are cached and re-raised (fail fast once).
- New `JinaTokenizer` handle in `okf-embed`: exact token counts without the
  ONNX session, so budgeted reads and the context-window guard never warm
  the session. Counts are identical to the session tokenizer path.

### Changed
- `okf-embed`: accelerator fallback now uses session-builder registration
  probing (bobine pattern) instead of a static CUDA availability flag;
  unknown provider names warn and are skipped, and EP registration features
  (CUDA/ROCm/DirectML/OpenVINO/CoreML/TensorRT/half) are enabled with CPU
  fallback. The public device surface is unchanged (`auto`/`cpu`/`cuda`)
  and embedding numerics are untouched.
- `okf-embed`: explicit optional `extension-module` Cargo feature — pure
  Rust crate by default, Python extension only for wheel builds (maturin
  enables the feature; `cargo test` stays link-clean).

### Added
- Cross-platform ONNX Runtime discovery: explicit `ORT_DYLIB_PATH` still
  wins, otherwise pip-installed `onnxruntime`/`onnxruntime-gpu` is resolved
  on Windows (`capi/onnxruntime.dll`), macOS
  (`capi/libonnxruntime.dylib`), and Linux (preferred versioned
  `capi/libonnxruntime.so.*`, fallback `capi/libonnxruntime.so`). GPU DLL
  warming and Windows DLL-directory setup are best-effort and never fatal.
  The resolved runtime is exposed as `OKFRouter.ort_dylib` /
  `EmbeddingEngine.ort_dylib` before the native module is imported.

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
