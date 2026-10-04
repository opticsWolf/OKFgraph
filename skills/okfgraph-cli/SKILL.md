---
name: okfgraph-cli
description: >
  OKFgraph knowledge graph via the `okf` shell CLI (search, read,
  traverse, ingest, export, images, diff, doctor). Use when no okf-mcp
  server is wired and the
  task needs persistent project knowledge from the graph: search past
  decisions and docs, navigate relationships, read stored documents, or
  persist new knowledge. Prefers graph lookup over re-deriving context
  the project already recorded. When MCP tools are available, use the
  okfgraph-mcp skill instead.
---

# OKFgraph skill (CLI)

Same graph as the MCP skill, through `okf` shell commands. Core verbs:

- `okf search [--target concepts|chunks|images] [--expand] [--hub-rerank] [--rank none|hub|ppr] QUERY`
- `okf read [--include body|chunks|document|context] [--max-tokens N] CONCEPT_ID`
- `okf traverse [START_ID] [--relationship ...] [--target ID]`
- `okf ingest --kind md|pdf|thoughts ...`
- `okf import [--all] FILES [--bundle-path DIR] [--prune-missing]` — delta-aware file import
- `okf export --all|--concept-id ID --output-dir DIR [--flavor okf|obsidian]`
- `okf images CONCEPT_ID` / `okf image ASSET_ID [--output-path F]` — image assets
- `okf diff [OLD_DIR] [NEW_DIR]` — structural drift (exit 1 when different)
- `okf doctor [--strict] [--fix]` — health score + safe repairs
- `okf detach [--bundle-path DIR] [--no-verify] [--force]` — end the mirror: the DB
  becomes the artifact (verify-first; imports refuse without `--force` re-attach)
- Global `--max-length N` (1..=32768, default 8192): token truncation ceiling.
  Raising it changes long-doc vectors — reimport fully after changing.
- Inference defaults just work: `--device auto` (CUDA when present, else CPU)
  with `--precision auto` following it (CUDA→FP16 mirror weights, CPU→FP32,
  pinned per graph — never mixed). Pin `--precision fp32` for bit-stable
  CPU vectors; `--cpu-arena` trades ~8x RSS for ~1.4x encode speed.
  Chunks are capped at 512 tokens (`chunk_size`); overlap tails never exceed
  the receiving chunk (`chunk_overlap`, default 40 words).
- `--model-id ID` selects the text embedding model (registry: text-small
  default, text-nano; `--precision int8` needs a measured artifact).
  The default is frozen — switching models pins fail-closed and forces a
  fresh reimport (different weights = different space). Nano is the
  supported CPU-side choice (5-7x faster), never the default.
- `okf init --root ALIAS=PATH` (repeatable, or TOML `[[roots]]`) joins
  additional named roots; their files mint `@alias/rel` IDs.
  Absent roots are skipped with a warning (unmounted ≠ deleted);
  `--prune-missing` skips while any root is absent. `[[alias/Name]]` links
  resolve cross-root; `okf detach` covers all roots; `--force` re-attach
  needs the full set.
- `okf import --all` imports **every configured root** (delta-aware);
  `--bundle-path DIR` pins one tree for this call only — deliberate, for
  reimporting one root without touching the others. `--bundle-root PATH`
  (single value) names the primary root that mints bare IDs.
- Every command accepts `--json`: the result envelope
  (`{ok, op, data, warnings, error}`) on stdout (D5); failures print
  `[ERROR] code: message` on stderr with exit 2 = usage, 1 = state /
  outcome (diff-different, doctor findings, lint errors).

## Setup (once per project)

1. `okf init --db-path kb.db --bundle-root kb/` creates the database and
   bundle root. `okf init --root ALIAS=PATH` joins extra named roots.
2. Persist them in `okfgraph.toml` so later commands need no flags.
3. Verify: `okf traverse` lists the root (empty graph prints a message,
   which still proves the wiring works).
4. Prefer `uv run --project <root> okf ...` so dependencies resolve from
   the project instead of the ambient environment.

## Embedding dimension

- Default is **512 on every surface** (router, MCP `--embedding-dim`,
  `okfgraph.toml`) — unified, no per-surface surprise.
- Set freely at creation: `--embedding-dim` accepts the Matryoshka ladder (32, 64,
  128, 256, 512, 768, 1024).
- Reopening adopts the stored dim — the flag only matters for new graphs.

## Which command when

| Need | Command |
|---|---|
| Open-ended question over stored knowledge | `search QUERY` |
| Exact passage inside long documents | `search --target chunks QUERY` |
| Answer + supporting graph neighborhood | `search --target chunks --expand QUERY` |
| Importance-ranked chunk hits | `search --target chunks --hub-rerank QUERY` |
| Image assets | `search --target images QUERY`, `images ID` (list on a concept), `image ASSET_ID --output-path F` (fetch bytes) |
| Follow CONTAINS / LINKS_TO relationships | `traverse ID` (depth 1–3 first) |
| How two concepts connect | `traverse ID1 --target ID2` |
| Browse a directory | `traverse` (empty id = root) or `traverse ID` (CONTAINS) |
| Full text / chunks / rebuilt doc of one concept | `read [--include ...] ID` |
| Same, capped to a token budget (PPR-ranked neighbours) | `read --max-tokens N ID` |
| Links + ancestry + siblings of one concept | `read --include context ID` |
| Store a markdown file / PDF / reasoning | `ingest --kind ...` |
| Validate a bundle before importing (frontmatter + links, no model load) | `lint [DIR]` (exit 0 clean / 1 errors) |
| Generate a bundle from a data source (today: SQLite → one concept per table) | `produce --from sqlite --source DB --output-dir DIR` (lint pre-flight included; adds `## Observations` + `log.md` changelog, both import-skipped) |

## Conventions

- Connection flags (`--db-path`, `--bundle-root`, `--embedding-dim`, ...) are
  documented once in `okf --help` and accepted on every command; most come
  from `okfgraph.toml`, so invocations stay short. The REPL (`okf shell`)
  resolves lines through the same parser: `search chunks:<q> hub`,
  `read <id> document`, `traverse <a> <b>`.
- Every CLI call cold-boots the router (model load ~30s). Batch reads:
  prefer one `search --expand` over N `read` calls.
- `search` prints concept/chunk hits with ids; drill in with `read`.
  `--rank ppr` is model-free (no ~30s load): prefer it for topic queries
  and cold sessions; default ranking stays hybrid.
- `traverse` needs a concept id — search first, then traverse from hits.
  Chunk hits carry `parent_doc_id`, so pivots rarely need re-search.
