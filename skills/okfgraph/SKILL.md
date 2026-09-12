---
name: okfgraph
description: >
  Ladybug-backed knowledge graph with Jina v5 semantic search. Use when the
  task needs persistent project knowledge: search past decisions and docs
  (search_hybrid, search_chunks), navigate concept relationships (traverse,
  find_path), read stored documents (get_by_id, reconstruct_document), or
  persist new knowledge (ingest_md, ingest_pdf, ingest_thoughts). Prefers
  graph lookup over re-deriving context the project already recorded.
---

# OKFgraph skill

OKFgraph is a persistent knowledge graph: markdown concepts with semantic
(vector) + keyword (FTS) search and graph traversal. Two access paths:

1. **MCP tools** (preferred when the harness wires `okf-mcp`): 16 tools —
   `search_hybrid`, `search_chunks`, `search_with_context`,
   `search_chunks_with_hub_score`, `search_images`, `traverse`, `find_path`,
   `get_by_id`, `get_chunks`, `reconstruct_document`, `list_directory`,
   `expand_with_graph_context`, `export_bundle`, `ingest_md`,
   `ingest_thoughts`, `ingest_pdf`.
2. **CLI fallback**: `okf <command>` (`search`, `search-chunks`, `traverse`,
   `get`, `ingest`, `context`, `hub-search`, `path`, `reconstruct`, ...).
   Run `okf --help` for the full list.

## Which tool when

| Need | Tool |
|---|---|
| Open-ended question over stored knowledge | `search_hybrid` |
| Exact passage inside long documents | `search_chunks` (RRF vector+FTS) |
| Answer + supporting graph neighborhood | `search_with_context` |
| Follow CONTAINS / LINKS_TO relationships | `traverse` (depth 1–3 first) |
| How two concepts connect | `find_path` |
| Full text of one concept | `get_by_id` |
| Original doc rebuilt from chunks | `reconstruct_document` |
| Store a markdown file | `ingest_md` |
| Store a PDF | `ingest_pdf` (`routing_mode: "never"` = fast, no ONNX) |
| Store reasoning/decisions being made now | `ingest_thoughts` (topic + thoughts) |

## Conventions

- `search_hybrid` returns concept hits; drill in with `get_by_id`.
- `traverse` needs a concept id — search first, then traverse from hits.
- Ingestion lints with mordant and auto-fixes formatting; content is
  chunked, embedded (Jina v5), and linked into the graph.
- Image search (`search_images`) matches text queries against image assets
  in the shared vector space.
- If an MCP tool errors with "Search is unavailable", the vector/FTS
  extensions failed to load — ingestion and graph reads still work.
