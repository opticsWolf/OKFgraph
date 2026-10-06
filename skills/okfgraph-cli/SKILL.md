---
name: okfgraph-cli
description: >
  OKFgraph knowledge graph via the `okf` shell CLI (search, read,
  traverse, ingest, import, export, images, lint, diff, doctor). Use when
  no okf-mcp server is wired and the task needs persistent project
  knowledge from the graph: search past decisions and docs, navigate
  relationships, read stored documents, or persist new knowledge. Prefers
  graph lookup over re-deriving context the project already recorded.
  When MCP tools are available, use the okfgraph-mcp skill instead.
---

# OKFgraph skill (CLI)

Same graph and the same ops as the MCP skill, through `okf` commands.

## Setup (once per project)

1. Put the connection in `okfgraph.toml` (CWD, or the `--bundle-root`
   dir) so every command stays short:

   ```toml
   bundle_root = "kb"          # top-level key, before any table

   [database]
   db_path = "kb.db"

   [[roots]]                   # optional extra named roots
   alias = "notes"
   path = "../notes"           # relative to this file
   ```

   Flags (`--db-path`, `--bundle-root`, `--root ALIAS=PATH`, ...) and
   `OKFGRAPH_*` env vars override it per call: CLI > env > TOML > defaults.
2. `okf init` creates the schema (any command would; init just reports
   the adopted pins). It does **not** persist flags — keep roots in TOML.
3. `okf import --all` loads every configured root. Verify with
   `okf traverse` (root listing; an empty graph prints a message).
4. Prefer `uv run --project <root> okf ...` so dependencies resolve from
   the project.

## Commands

| Need | Command |
|---|---|
| Open-ended question over stored knowledge | `search QUERY` |
| Cold / deterministic topic search (no model load) | `search --rank ppr QUERY` |
| Exact passage inside long documents | `search --target chunks QUERY` |
| Answer + supporting graph neighborhood | `search --target chunks --expand QUERY` |
| Importance-ranked chunk hits | `search --target chunks --hub-rerank QUERY` |
| Image assets | `search --target images QUERY`; `images ID`; `image ASSET_ID --output-path F` |
| Full text / chunks / rebuilt doc of one concept | `read ID [--include body\|chunks\|document]` |
| Same, capped to a token budget (PPR-ranked neighbours) | `read ID --max-tokens N` |
| Links + ancestry + siblings | `read ID --include context` |
| Follow CONTAINS / LINKS_TO | `traverse ID [--relationship LINKS_TO] [--depth 1-3]` |
| How two concepts connect | `traverse ID1 --target ID2` |
| Browse a directory | `traverse` (root) or `traverse DIR_ID` |
| Store markdown / PDF / reasoning | `ingest --kind md\|pdf\|thoughts ...` (okfgraph-ingest skill) |
| Re-sync a bundle (delta-aware) | `import --all [--bundle-path DIR] [--prune-missing]`; single files: `import F...` |
| Export | `export --all --output-dir D` or `export --concept-id ID --output-dir D` (`--flavor obsidian`) |
| Validate a bundle before import (no DB, no model) | `lint [DIR]` |
| Generate a bundle from SQLite | `produce --from sqlite --source DB [--output-dir D]` |
| Structural drift | `diff OLD NEW` (dirs) or `diff [DIR]` (graph vs dir / roots) |
| Health score + safe repairs | `doctor [--fix] [--strict]` |
| End the mirror: the DB becomes the artifact | `detach [--bundle-path D] [--no-verify] [--force]` |
| Soft-deleted concepts | `deleted-list`, `deleted-recover ID`, `deleted-purge` |
| Remove a wrongly imported concept | `delete ID` — soft-delete, recoverable 24 h; **refuses file-backed concepts while their source file exists** (remedy: remove the file, then `import --all --prune-missing`) |

Search refuses parameters its chosen path ignores (e.g. `--context-hops`
without `--expand`, `--hub-weight` without `--rank hub`/`--hub-rerank`,
filters on `--target images`) instead of silently dropping them.

## Output and errors

- Every command takes `--json`: the envelope `{ok, op, data, warnings,
  error}` on stdout. Without it, a human rendering on stdout; logs go to
  stderr (`-q` silences logs, never results).
- Failures print `[ERROR] CODE: message (remedy)` on stderr. Exit 2 =
  usage (`BAD_VALUE`, `MISSING_PARAM`, `FILE_NOT_FOUND`,
  `CONFIG_INVALID`), 1 = state (`UNKNOWN_CONCEPT`, `DETACHED`, ...) or
  outcome.
- Outcomes still render their report: `lint` with errors
  (`LINT_ERRORS`), `diff` that differs (`DIFF_DIFFERENT`), `doctor
  --strict` with findings (`DOCTOR_FINDINGS`) exit 1 with the report on
  stdout (in `data` under `--json`). Use them as CI gates.

## Cost model

- Commands that embed text (`search` except `--rank ppr`, `import`,
  `ingest`) load the ONNX model: ~30s cold per process. `read`,
  `traverse`, `lint`, `diff`, `doctor`, `export` never load it.
- So batch: one `search --target chunks --expand` beats N `read`s;
  `okf shell` keeps one warm process (`search chunks:<q> hub`,
  `read <id> document`, `traverse <a> <b>`; any CLI flag after the verb).
- One process per database: run `okf` commands sequentially. A second
  process on the same db (a parallel `okf`, or an `okf-mcp` server) is
  refused with `DB_LOCKED`; use that MCP server's tools instead.

## Models and dimensions

- Defaults just work: `--device auto` (CUDA when present, else CPU),
  `--precision auto` (CUDA→FP16, CPU→FP32), `--embedding-dim 512`, text
  model jina-v5-text-small. All are pinned per graph at creation; reopening
  adopts them, and a mismatching flag fails closed (`*_PIN_MISMATCH`).
- `--embedding-dim` takes the Matryoshka ladder 32..1024; `--model-id`
  text-nano is the faster CPU choice. Changing model, precision or
  `--max-length` (default 8192) means a fresh graph and full reimport.
- Chunks are capped at `chunk_size` tokens (512); overlap tails are
  `chunk_overlap` words (40).

## Multi-root graphs

- Extra roots (`--root ALIAS=PATH` or TOML `[[roots]]`) mint `@alias/rel`
  IDs; link across roots with `[[alias/Name]]`.
- `import --all` covers every root; `--bundle-path DIR` pins one tree for
  that call. Absent roots are skipped with a warning (unmounted ≠
  deleted) and block `--prune-missing`.

## Conventions

- `search` prints ids; drill in with `read`, pivot with `traverse`.
  Chunk hits carry `parent_doc_id`.
- Bundle files are the source of truth: edit markdown, re-import.
