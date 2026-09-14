---
name: okfgraph-cli
description: >
  OKFgraph knowledge graph via the `okf` shell CLI (search, read,
  traverse, ingest, export). Use when no okf-mcp server is wired and the
  task needs persistent project knowledge from the graph: search past
  decisions and docs, navigate relationships, read stored documents, or
  persist new knowledge. Prefers graph lookup over re-deriving context
  the project already recorded. When MCP tools are available, use the
  okfgraph-mcp skill instead.
---

# OKFgraph skill (CLI)

Same graph as the MCP skill, through `okf` shell commands. Five verbs:

- `okf search [--target concepts|chunks|images] [--expand] [--hub-rerank] [--rank none|hub|ppr] QUERY`
- `okf read [--include body|chunks|document|context] [--max-tokens N] CONCEPT_ID`
- `okf traverse [START_ID] [--relationship ...] [--target ID]`
- `okf ingest --kind md|pdf|thoughts ...`
- `okf export --all|--concept-id ID --output DIR [--flavor okf|obsidian]`
- `okf diff [OLD_DIR] [NEW_DIR]` — structural drift (exit 1 when different)
- `okf doctor [--strict] [--fix]` — health score + safe repairs
- `okf detach [--bundle DIR] [--no-verify] [--force]` — end the mirror: the DB
  becomes the artifact (verify-first; imports refuse without `--force` re-attach)
- Global `--max-length N` (1..=32768, default 8192): token truncation ceiling.
  Raising it changes long-doc vectors — reimport fully after changing.
- `--bundle-root ALIAS=PATH` (repeatable, combines with `--bundle`):
  additional named roots mint `@alias/rel` IDs (TOML `[[roots]]` equivalent).
  Absent roots are skipped with a warning (unmounted ≠ deleted); `--purge`
  refuses while any root is absent. `[[alias/Name]]` links resolve cross-root;
  `okf detach` covers all roots; `--force` re-attach needs the full set.
- `okf import --all` scope: **with `--bundle`, only that tree imports**
  (warning names the untouched roots) — deliberate, for reimporting one
  root without touching the others. **Full multi-root rebuild:**
  `okf import --all --db DB --primary PRIMARY --bundle-root ALIAS=PATH
  ...` — `--primary` sets the bare-ID root without pinning; `--bundle`
  + `--primary` together is a scope clash (refused). Without either
  flag the primary defaults to the CWD (`cd <primary>` first) or to
  TOML `bundle`. Passing `--bundle` *with* `--bundle-root` registers
  the roots but imports just the one tree.

## Setup (once per project)

1. `okf init --db kb.db --bundle kb/` creates the database and bundle root.
2. Persist them in `okfgraph.toml` so later commands need no flags.
3. Verify: `okf traverse` lists the root (empty graph prints a message,
   which still proves the wiring works).
4. Prefer `uv run --project <root> okf ...` so dependencies resolve from
   the project instead of the ambient environment.

## Embedding dimension

- Default is **512 on every surface** (router, MCP `--embedding-dim`,
  `okfgraph.toml`) — unified, no per-surface surprise.
- Set freely at creation: `--dim` accepts the Matryoshka ladder (32, 64,
  128, 256, 512, 768, 1024).
- Reopening adopts the stored dim — the flag only matters for new graphs.

## Which command when

| Need | Command |
|---|---|
| Open-ended question over stored knowledge | `search QUERY` |
| Exact passage inside long documents | `search --target chunks QUERY` |
| Answer + supporting graph neighborhood | `search --target chunks --expand QUERY` |
| Importance-ranked chunk hits | `search --target chunks --hub-rerank QUERY` |
| Image assets | `search --target images QUERY` |
| Follow CONTAINS / LINKS_TO relationships | `traverse ID` (depth 1–3 first) |
| How two concepts connect | `traverse ID1 --target ID2` |
| Browse a directory | `traverse` (empty id = root) or `traverse ID` (CONTAINS) |
| Full text / chunks / rebuilt doc of one concept | `read [--include ...] ID` |
| Same, capped to a token budget (PPR-ranked neighbours) | `read --max-tokens N ID` |
| Links + ancestry + siblings of one concept | `read --include context ID` |
| Store a markdown file / PDF / reasoning | `ingest --kind ...` |
| Validate a bundle before importing (frontmatter + links, no model load) | `lint [DIR]` (exit 0 clean / 1 errors) |

## Conventions

- Connection flags (`--db`, `--bundle`, `--dim`, ...) are documented once
  in `okf --help` and accepted on every command; most come from
  `okfgraph.toml`, so invocations stay short.
- Every CLI call cold-boots the router (model load ~30s). Batch reads:
  prefer one `search --expand` over N `read` calls.
- `search` prints concept/chunk hits with ids; drill in with `read`.
  `--rank ppr` is model-free (no ~30s load): prefer it for topic queries
  and cold sessions; default ranking stays hybrid.
- `traverse` needs a concept id — search first, then traverse from hits.
  Chunk hits carry `parent_doc_id`, so pivots rarely need re-search.
