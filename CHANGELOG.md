# Changelog

All notable changes to OKFgraph are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com); entries are grouped from
commit history, newest first.

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
