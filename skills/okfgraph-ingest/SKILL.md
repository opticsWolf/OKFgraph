---
name: okfgraph-ingest
description: >
  Feeding the OKFgraph knowledge graph: what to store, which ingest kind
  to use (md, pdf, thoughts), converter modes, and topic discipline. Use
  whenever persisting project knowledge — decisions, docs, papers, or
  reasoning — through MCP (ingest) or CLI (okf ingest). Complements the
  okfgraph-mcp and okfgraph-cli skills, which cover finding knowledge.
---

# OKFgraph skill (ingest)

The graph compounds only if sessions deposit what they learn. Feed it
early, feed it structured.

## What to store, and as what

| Situation | Kind | Notes |
|---|---|---|
| Decision, insight, or reasoning from this session | `thoughts` | Cheapest, highest value. Always set `topic`. |
| Existing markdown (docs, notes, ADRs) | `md` | One file per call; bulk dirs via `okf import`. |
| Papers, reports, scans | `pdf` | See converter modes below. |
| Whole bundle directory changed | `okf import` (CLI) | Delta-aware: only changed files re-embed. |

Rule of thumb: thoughts > md > pdf. Reasoning you already hold beats
re-extracting it from files.

## MCP: `ingest`

- `kind="thoughts"`: `thoughts` + `topic` (both required). Topic scheme per
  project, e.g. `auth-refactor`, `api-design` — consistency makes search work.
- `kind="md"`: `md_path` (+ optional `concept_id`, `title`, `description`,
  `tags`, `mode`).
- `kind="pdf"`: `pdf_path` (+ `routing_mode`, `extract_images`, `mode`).

## CLI: `okf ingest`

- `--kind thoughts --thoughts TEXT --topic TOPIC [--tags a,b]`
- `--kind md --md-file F [--concept-id ID] [--title T] [--tags a,b]`
- `--kind pdf --pdf-file F [--auto-import] [--routing-mode ...]`
  Bulk: `okf import --all --bundle DIR` (`--purge` drops deleted concepts).

## Converter modes (pdf only)

`routing_mode` controls when the converter spends ONNX compute:

- `never` — fast path, text extraction only. Default choice for text PDFs.
- `auto` — ONNX only where the fast path looks insufficient.
- `surgical` / `always` — force heavy passes (scans, complex layouts, tables).

`extract_images=false` skips embedded images (pure-text import). Image
*embedding* depth is separate: `mode=text|optional|omni` (omni needs the
`omni` extra installed). Converted markdown is mordant-linted and temp
dirs are cleaned automatically.

## Conventions

- Bundle files are source of truth: edit markdown, re-import — never try
  to "edit the graph."
- `[[Wikilinks]]` resolve by name (`id` → `aliases` → `title` → filename),
  so vault notes survive file moves. Ambiguous names never resolve — keep
  titles/`aliases` unique. `aliases: [...]` and `id:` frontmatter are
  honoured (exported back losslessly; `export --flavor obsidian` writes
  `[[Title]]` links with no index files).
- `md_path` in pdf results is transient (temp dir); the content lives in
  the graph — verify with `search`, not by reading the path.
- Verify every ingest: search for something only the new content contains.
- Frontmatter (`title`, `type`, `tags`) on md files beats post-hoc overrides.
