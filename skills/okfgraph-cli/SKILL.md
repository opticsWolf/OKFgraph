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

## Setup (once per project)

1. `okf init --db kb.db --bundle kb/` creates the database and bundle root.
2. Persist them in `okfgraph.toml` so later commands need no flags.
3. Verify: `okf traverse` lists the root (empty graph prints a message,
   which still proves the wiring works).
4. Prefer `uv run --project <root> okf ...` so dependencies resolve from
   the project instead of the ambient environment.

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
