---
name: okfgraph-mcp
description: >
  Ladybug-backed knowledge graph with Jina v5 semantic search, via MCP
  tools (search, read, traverse, ingest, export_bundle, export_concept,
  list_images, get_image). Use when the task needs persistent project
  knowledge and okf-mcp is wired: search past decisions and docs,
  navigate concept relationships, read stored documents, or persist new
  knowledge. Prefers graph lookup over re-deriving context the project
  already recorded. For shell-only environments without MCP, use the
  okfgraph-cli skill instead.
---

# OKFgraph skill (MCP)

OKFgraph is a persistent knowledge graph: markdown concepts with semantic
(vector) + keyword (FTS) search and graph traversal, exposed as 8 MCP tools.

## Setup (once per project)

1. The harness wires `okf-mcp` with a stable database: `--db-path`,
   `OKFGRAPH_DB_PATH`, or `[database] db_path` in an `okfgraph.toml` in
   the server's CWD. The server refuses to boot (`CONFIG_INVALID`, exit 2)
   when none is set — a temp default would mean an empty graph every
   session. `--bundle-root` is optional (default: the DB's parent dir);
   `--root ALIAS=PATH` (repeatable) joins extra named roots.
2. The first boot creates the schema; no init call needed. Bulk
   (re)import of a bundle tree is CLI/Python only (`okf import --all`);
   `ingest` adds one file or thought at a time.
3. `ingest`, `export_bundle` and `export_concept` write: they need
   harness approval unless pre-approved. On a detached graph
   (`okf detach`) everything serves, but `ingest` refuses (`DETACHED`).
4. Verify: `search` anything (an empty graph returns `data: []`).

## Which tool when

| Need | Tool |
|---|---|
| Open-ended question over stored knowledge | `search` |
| Cold / deterministic topic search (no model load) | `search` rank="ppr" |
| Exact passage inside long documents | `search` target="chunks" |
| Answer + supporting graph neighborhood | `search` target="chunks", expand=true |
| Importance-ranked chunk hits | `search` target="chunks", hub_rerank=true |
| Image assets | `search` target="images"; `list_images` concept_id; `get_image` asset_id |
| Full text / chunks / rebuilt doc of one concept | `read` include="body"\|"chunks"\|"document" |
| Same, capped to a token budget (PPR-ranked neighbours) | `read` max_tokens=N |
| Links + ancestry + siblings | `read` include="context" |
| Follow CONTAINS / LINKS_TO | `traverse` start_id, relationship, depth 1–3 |
| How two concepts connect | `traverse` start_id, target |
| Browse a directory | `traverse` (empty start_id = root) |
| Store markdown / PDF / reasoning / bibliography | `ingest` kind="md"\|"pdf"\|"thoughts"\|"bib" (okfgraph-ingest skill) |
| Export | `export_bundle` output_dir (+ filters, flavor); `export_concept` concept_id, output_dir |

## Results and errors

- Every tool returns the envelope `{ok, op, data, warnings, error}` as
  JSON text. Read `data`; `warnings` carries non-fatal notes.
- Failures set `isError: true` and carry the error envelope:
  `error: {code, message, fields, remedy}`. Act on `remedy`:
  `UNKNOWN_CONCEPT` → search first to find IDs; `BAD_VALUE` names the
  offending param in `fields`; `SEARCH_UNAVAILABLE` → the vector/FTS
  extensions failed to load (graph reads still work).
- The server holds the database for its lifetime: a shell `okf` on the
  same db fails with `DB_LOCKED`, so query through these tools instead.
- Params a path ignores are refused, never silently dropped:
  `context_hops` needs `expand`; `hub_weight` needs rank="hub" or
  `hub_rerank`; filters (`concept_type`, `tags`, `parent_id`) apply to
  concepts and plain chunk search only; target="images" takes only
  `query`/`limit`; `rank` is concepts-only. `hub_rerank` wins over
  `expand`.

## Conventions

- `search` returns ids; drill in with `read`, pivot with `traverse`.
  Chunk hits carry `parent_doc_id`, so pivots rarely need re-search.
- rank="ppr" answers from the link graph alone (lexical seeds → graph
  walk): use it when the embedder is cold, results must be
  deterministic, or the query names topics rather than phrases. Other
  searches and `ingest` load the ONNX model (~30s once per server).
- Model, precision and dimension are pinned per graph at creation;
  reopening adopts them (defaults: jina-v5-text-small, 512 dims,
  device/precision auto). Changing them is a CLI-side fresh rebuild.
- To persist knowledge, follow the okfgraph-ingest skill.
