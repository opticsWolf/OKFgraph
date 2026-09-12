---
name: okfgraph
description: >
  Ladybug-backed knowledge graph with Jina v5 semantic search. Use when the
  task needs persistent project knowledge: search past decisions and docs
  (search), navigate concept relationships (traverse), read stored
  documents (read), or persist new knowledge (ingest). Prefers
  graph lookup over re-deriving context the project already recorded.
---

# OKFgraph skill

OKFgraph is a persistent knowledge graph: markdown concepts with semantic
(vector) + keyword (FTS) search and graph traversal. Two access paths:

1. **MCP tools** (preferred when the harness wires `okf-mcp`): 5 tools —
   `search`, `read`, `traverse`, `ingest`, `export_bundle`.
2. **CLI fallback**: `okf <command>` with the same five verbs (`search`,
   `read`, `traverse`, `ingest`, `export`).
   Run `okf --help` for flags.

## Which tool when

| Need | Tool |
|---|---|
| Open-ended question over stored knowledge | `search` (target=concepts) |
| Exact passage inside long documents | `search` target=chunks |
| Answer + supporting graph neighborhood | `search` target=chunks expand=true |
| Importance-ranked chunk hits | `search` target=chunks hub_rerank=true |
| Image assets | `search` target=images |
| Follow CONTAINS / LINKS_TO relationships | `traverse` (depth 1–3 first) |
| How two concepts connect | `traverse` target=<id> |
| Browse a directory | `traverse` (CONTAINS, depth 1; empty id = root) |
| Full text / chunks / rebuilt doc of one concept | `read` include=body\|chunks\|document |
| Links + ancestry + siblings of one concept | `read` include=context |
| Store a markdown file / PDF / reasoning | `ingest` kind=md\|pdf\|thoughts |

## Conventions

- `search` returns concept or chunk hits; drill in with `read`.
- `traverse` needs a concept id — search first, then traverse from hits.
  Chunk hits carry `parent_doc_id`, so graph pivots rarely need re-search.
- Ingestion lints with mordant and auto-fixes formatting; content is
  chunked, embedded (Jina v5), and linked into the graph.
- Image search (`search` target=images) matches text queries against image assets
  in the shared vector space.
- If an MCP tool errors with "Search is unavailable", the vector/FTS
  extensions failed to load — ingestion and graph reads still work.
