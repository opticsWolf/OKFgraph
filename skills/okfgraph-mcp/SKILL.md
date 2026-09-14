---
name: okfgraph-mcp
description: >
  Ladybug-backed knowledge graph with Jina v5 semantic search, via MCP
  tools (search, read, traverse, ingest, export_bundle). Use when the
  task needs persistent project knowledge and okf-mcp is wired: search
  past decisions and docs, navigate concept relationships, read stored
  documents, or persist new knowledge. Prefers graph lookup over
  re-deriving context the project already recorded. For shell-only
  environments without MCP, use the okfgraph-cli skill instead.
---

# OKFgraph skill (MCP)

OKFgraph is a persistent knowledge graph: markdown concepts with semantic
(vector) + keyword (FTS) search and graph traversal, exposed as 5 MCP tools.

## Setup (once per project)

1. The harness must wire `okf-mcp` with a stable `--db-path` and `--bundle-root`
   (a temp dir means an empty graph every session). Additional live trees
   join via repeatable `--root ALIAS=PATH` (their files mint `@alias/rel`
   ids; link them as `[[alias/Name]]`).
2. First server boot creates the schema automatically — no init call needed.
3. `ingest` / `export_bundle` are write tools: they need harness approval
   unless pre-approved. For a knowledge workflow, pre-approve them.
   On a detached graph (`okf detach` was run) all five tools serve
   normally, but `ingest` refuses — re-attach from the CLI first.
4. Verify: `search` anything (empty graph returns `[]`, which still proves
   the wiring works).

## Embedding dimension

- Default is **512 on every surface** (router, CLI `--dim`, MCP
  `--embedding-dim`, `okfgraph.toml`) — unified, no per-surface surprise.
- Set freely at creation: `--embedding-dim` accepts the Matryoshka ladder
  (32, 64, 128, 256, 512, 768, 1024).
- Opening an existing DB adopts its on-disk dim — pass the flag only when
  creating a new graph.

## Which tool when

| Need | Tool |
|---|---|
| Open-ended question over stored knowledge | `search` (target=concepts) |
| Exact passage inside long documents | `search` target=chunks |
| Answer + supporting graph neighborhood | `search` target=chunks expand=true |
| Importance-ranked chunk hits | `search` target=chunks hub_rerank=true |
| Cold / deterministic topic search (no model load) | `search` rank=ppr |
| Image assets | `search` target=images |
| Follow CONTAINS / LINKS_TO relationships | `traverse` (depth 1–3 first) |
| How two concepts connect | `traverse` target=<id> |
| Browse a directory | `traverse` (CONTAINS, depth 1; empty id = root) |
| Full text / chunks / rebuilt doc of one concept | `read` include=body\|chunks\|document |
| Same, capped to a token budget | `read` include=... max_tokens=N |
| Links + ancestry + siblings of one concept | `read` include=context |
| Store a markdown file / PDF / reasoning | `ingest` kind=md\|pdf\|thoughts |

## Conventions

- `search` returns concept or chunk hits; drill in with `read`.
  `rank=ppr` answers from the link graph alone (lexical seeds → graph
  walk): use it when the embedder is cold, when results must be
  deterministic, or when the query names topics rather than phrases.
- `traverse` needs a concept id — search first, then traverse from hits.
  Chunk hits carry `parent_doc_id`, so graph pivots rarely need re-search.
- Ingestion lints with mordant and auto-fixes formatting; content is
  chunked, embedded (Jina v5), and linked into the graph.
- Image search (`search` target=images) matches text queries against image assets
  in the shared vector space.
- If a tool errors with "Search is unavailable", the vector/FTS
  extensions failed to load — ingestion and graph reads still work.
- To persist knowledge (not just find it), follow the okfgraph-ingest skill.
