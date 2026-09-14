# OKFgraph 0.3.0

[![PyPI](https://img.shields.io/pypi/v/okfgraph)](https://pypi.org/project/okfgraph/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-green)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-%E2%89%A52.0-purple)](https://modelcontextprotocol.io/)
[![Ladybug](https://img.shields.io/badge/ladybug-0.20.3-orange)](https://pypi.org/project/ladybug/)

**Ladybug-backed knowledge graph with Rust-driven Jina v5 embeddings, model-free
graph retrieval, and agent-first MCP + CLI surfaces.**

OKFgraph is a Python library, CLI (`okf`), and MCP server (`okf-mcp`) for
building, querying, and maintaining a knowledge graph from Markdown/OKF
documents. It combines hybrid semantic search (vector + FTS, RRF-fused),
chunk-level retrieval, **model-free PPR retrieval that works with no embedding
model loaded**, structural diffing, scored health checks, and Obsidian-vault
compatible wikilinks — in a single LadybugDB file.

Design stance: **slim dependencies, no legacy fallbacks.** Embeddings run
through the `embroider` Rust wheel (ONNX Runtime, no torch / transformers /
optimum anywhere). PDF conversion runs through the `bobine` Rust engine behind
a swappable `DocumentConverter` seam. What isn't needed isn't installed.

---

## Features

| Category | Features |
|---|---|
| **Embeddings** | Jina v5 (`jina-embeddings-v5-text-small-retrieval`) via the `embroider` Rust wheel; last-token pooling, Matryoshka truncation (32–1024, default 512); token ceiling configurable (`--max-length`, default 8192, up to 32768 — the Qwen3 position limit); omni model (`…-omni-small-retrieval`) lazy-loaded for images only |
| **Search** | Hybrid RRF fusion (vector + FTS) at concept and chunk granularity; `rank=none\|hub\|ppr` — including **PPR**: lexical seeds → exact Personalized PageRank, zero model load, deterministic |
| **Read** | Body / chunks / rebuilt document / graph context, with optional **token budgets** (`max_tokens`): self first, then PPR-ranked neighbours, index-first for context |
| **Storage** | LadybugDB `==0.20.3` (pinned — newer 0.20.x segfaults index builds): graph + vector + FTS in one file |
| **Links** | Path links (`](doc.md)`) + `[[wikilinks]]` resolved by name (`uid` → `aliases` → `title` → filename stem); ambiguous names never resolve; broken links tracked and repairable |
| **Import** | Single files, whole bundles (delta-aware: only changed files re-embed, `--purge` drops deleted concepts), markdown / PDF / raw thoughts; mordant lint on the way in |
| **PDF** | `DocumentConverter` seam with `BobineConverter` default (routing `auto\|surgical\|always\|never`); bring your own converter, no code changes |
| **Diff** | `okf diff`: structural snapshot (dir vs dir, no model load) and drift (graph vs dir) modes; CI exit codes + `--json` |
| **Doctor** | `okf doctor`: 0–100 health score (broken/orphan/stale/duplicate-title/missing-description), safe `--fix` that never touches `reviewed: true`, `--strict` CI gate |
| **Export** | OKF round-trip with See Also / Cited By enrichment + index files, or `--flavor obsidian` (`[[Title]]` wikilinks, no index files, edge-lossless re-import) |
| **Images** | Modes `text` / `optional` / `omni`, shared text+image vector space, content-hash dedup, `okf-asset://` protocol |
| **MCP** | 5 tools (`search`, `read`, `traverse`, `ingest`, `export_bundle`), MCP ≥ 2.0 (`MCPServer` + `ToolAnnotations`), stdio transport |
| **CLI** | Same 5 verbs plus maintenance (`init`, `import`, `diff`, `doctor`, `shell`, `reindex`, `broken-links`, `deleted-*`, …), slim per-command help, `okfgraph.toml` config |
| **Skills** | 3 harness-neutral skills (`okfgraph-mcp`, `okfgraph-cli`, `okfgraph-ingest`), `.mcp.json` wiring included |

---

## Installation

Requires Python ≥ 3.11.

```bash
pip install "okfgraph[pdf,omni]"   # PyPI (embeddings come from the published `embroider` wheels)
```

Or from source with `uv` (no Rust toolchain needed — the embedding
engine is the external `embroider` package):

```bash
git clone <repo> && cd OKFgraph
uv sync                 # core: ladybug, embroider, onnxruntime, mordant, mcp, …
uv sync --extra pdf     # bobine PDF converter
uv sync --extra omni    # sentence-transformers + Pillow (image embeddings)
uv sync --extra dev     # pytest
```

Core dependencies are deliberately few: `ladybug==0.20.3`, `embroider>=0.1,<0.2`
(the shared embedding engine — github.com/opticsWolf/embroider, also used by
bobine), `onnxruntime==1.29.0` (one pinned ORT binary shared by bobine +
embroider; `ORT_DYLIB_PATH`-overridable), `mordant`, `mcp>=2.0`, `pydantic`,
`pyyaml`, `numpy`, `python-frontmatter`, `fasteners`. There is no torch, no
transformers, no optimum, no PDF stack in the core install.

---

## Quick start

```bash
okf init --db kb.db --bundle kb/          # once; persists to okfgraph.toml
okf import --all --bundle kb/             # delta-aware bulk import
okf search "honey badger defense"         # hybrid semantic search
okf search --rank ppr "honey badger"      # same question, no model load
okf read savanna --include context --max-tokens 1500
okf traverse savanna --relationship LINKS_TO --direction BOTH
okf diff                                  # drift: graph vs bundle dir
okf doctor                                # health score + findings
okf lint kb/                               # pre-import gate: frontmatter + links, no model
```

### Programmatic usage

```python
from pathlib import Path
from okfgraph import OKFRouter

router = OKFRouter(db_path="kb.db", bundle_root="kb/", embedding_dim=512)

# Import: whole bundle (delta-aware), one file, or raw reasoning
ids = router.import_mgr.import_bundle(Path("kb/"), purge_deleted=False)
cid = router.import_from_okf(Path("kb/auth.md"), mode="text")
th  = router.ingest_mgr.ingest_thoughts("Session decided X because Y", topic="auth-refactor")

# Search: hybrid (default), hub-blended, or model-free PPR
hits   = router.search_hybrid("session auth decisions", limit=5)
hubbed = router.search_hybrid("session auth decisions", rank="hub")
cold   = router.search_engine.search_with_ppr("auth refactor")  # no ONNX load

# Read: full shapes, or a token-budgeted section list
doc     = router.get_by_id("auth")
reading = router.search_engine.read_with_budget("auth", include="context", max_tokens=1500)

# Maintain: drift, health, links
print(router.diff_db_dir(Path("kb/"))["identical"])   # True when in sync
print(router.diagnose()["score"])                     # 0-100
print(router.doctor_fix())                            # safe repairs only
print(router.repair_links())                          # re-point broken links

# Export: OKF bundle or Obsidian vault
router.export_mgr.export_bundle(Path("out-okf/"))
router.export_mgr.export_bundle(Path("out-vault/"), flavor="obsidian")

router.close()
```

### Ingestion kinds

| Kind | Entry point | Notes |
|---|---|---|
| `md` | `ingest_md(md_path, concept_id?, title?, tags?, mode?)` | mordant-linted; frontmatter wins over overrides |
| `pdf` | `ingest_pdf(pdf_path, auto_import?, routing_mode?, …)` | bobine converter; `md_path` is transient when auto-importing — verify by searching, not by reading the path |
| `thoughts` | `ingest_thoughts(text, topic, tags?)` | cheapest, highest value: persist session reasoning with a stable topic scheme (`auth-refactor`, `api-design`) |

Frontmatter that matters: `title`, `type`, `tags`, `aliases: [...]` (wikilink
names), `id:` (stable identity, preserved as `uid`, written back on export),
`reviewed: true` (immune to `doctor --fix`). Credentials in `resource:`
URIs are stripped to `***@` at parse time (graphs imported before 0.2.3
keep old values until re-import). Rule of thumb: thoughts > md >
pdf — reasoning you already hold beats re-extracting it from files.

---

## CLI reference

Five verbs mirror the MCP tools; maintenance commands cover the rest. Global
flags (`--db`, `--bundle`, `--dim`, `--max-length`, …) are documented once in `okf --help`,
accepted everywhere, and usually live in `okfgraph.toml`.

| Command | Description |
|---|---|
| `okf search QUERY [--target concepts\|chunks\|images] [--rank none\|hub\|ppr] [--expand] [--hub-rerank]` | Concepts (RRF hybrid), chunks (passages), images; `--rank ppr` is model-free |
| `okf read ID [--include body\|chunks\|document\|context] [--max-tokens N]` | Full shapes uncapped; budgeted section list with `--max-tokens` |
| `okf traverse [ID] [--relationship CONTAINS\|LINKS_TO\|PART_OF\|INCLUDES_ASSET] [--direction …] [--target ID]` | Relationships, directory listing (empty ID = root), shortest path via `--target` |
| `okf ingest --kind md\|pdf\|thoughts …` | `--md-file`, `--pdf-file` (`--routing-mode`, `--auto-import`), `--thoughts --topic` |
| `okf export --all\|--concept-id ID --output DIR [--flavor okf\|obsidian]` | Bundle export; obsidian = `[[Title]]` links, no index files |
| `okf diff [OLD] [NEW] [--json]` | Snapshot (two dirs, no model) or drift (graph vs dir); exit 0 identical / 1 different |
| `okf lint [DIR] [--json]` | Pre-import gate (no DB, no model); exit 0 clean / 1 errors / 2 bad dir |
| `okf doctor [--fix] [--strict] [--stale-days N] [--json]` | Score + findings; `--fix` repairs safely, `--strict` exits 1 on any finding |
| `okf import [--all] [--purge] [--mode text\|optional\|omni] [--force]` | Bulk/single import, delta-aware (`--force` re-attaches a detached graph) |
| `okf detach [--bundle DIR] [--no-verify] [--force]` | End the mirror: the DB becomes the artifact (verify-first; imports refuse without `--force`) |
| `okf init`, `okf model-info`, `okf shell`, `okf reindex`, `okf broken-links`, `okf repair-links`, `okf deleted-*` | Setup, cache inspection, REPL, index rebuild, link + soft-delete maintenance |

Every CLI call cold-boots the router (model load ~30s when the embedder is
needed) — batch reads, and reach for `--rank ppr` for topic queries in cold
sessions.

---

## MCP server

5 tools, MCP ≥ 2.0, stdio transport. Wire once per project with a **stable**
`--db-path` + `--bundle` (a temp dir means amnesia every session); first boot
creates the schema; pre-approve the `ingest` / `export_bundle` write tools for
knowledge workflows. `.mcp.json` ships a ready config (`uv run --project .
okf-mcp`).

```bash
okf-mcp --db-path ./kb.db --bundle-root ./kb
```

| Tool | Routing |
|---|---|
| `search(query, target?, rank?, …)` | `concepts` = open questions; `chunks` = exact passages (`expand`, `hub_rerank`); `images` = assets; `rank=ppr` = model-free topic search |
| `read(concept_id, include?, max_tokens?)` | Full shapes, or a budgeted section list |
| `traverse(start_id, relationship?, direction?, target?, …)` | Walk edges, list dirs, connect two concepts |
| `ingest(kind, …)` | `md` / `pdf` / `thoughts` with per-kind params |
| `export_bundle(output_dir, …, flavor?)` | `okf` or `obsidian` |

Verify wiring with any `search` — an empty graph returns `[]`, which still
proves the plumbing works.

### Skills

Harness-neutral skill sources live in `skills/` (Claude Code auto-loads
`.claude/skills/<name>/SKILL.md`; other harnesses have their own dirs — see
`docs/harness-integration.md`):

- **`okfgraph-mcp`** — finding knowledge via MCP tools (pick this *or* CLI).
- **`okfgraph-cli`** — same graph through `okf` shell commands.
- **`okfgraph-ingest`** — feeding discipline: what to store, which kind,
  converter modes, wikilink/alias conventions. Install alongside either
  finder when the agent should persist knowledge.

---

## Architecture

```
                ┌──────────── MCP (5 tools) / CLI (5 verbs + maintenance)
                │                        │
          OKFRouter (facade: owns conn, encoder, lock; thin proxies)
                │
   ┌────────────┼──────────────────────────────────────────┐
   │            │                                          │
Import ──► Embed ──► Search ◄── ranking ──► Export ──► Doctor/Diff
   │         │         │        (seeds+PPR)     │           │
   │         │         │                   links.py        │
Delta ──► Schema ──► ImageAssets ──► Purge ──► Ingest ──► Converters
   │                                                    (bobine seam)
   └──────────────── LadybugDB (graph + vector + FTS, one file) ──┘
                         ▲
              Rust embroider (Jina v5, ORT) — tokenize (ceiling `--max-length`,
              default 8192, max 32768), last-token pool, L2-norm,
              Matryoshka truncate; numerics pinned by tests/test_parity.py
```

Components live in `okfgraph/components/` (one concern each, dependencies
injected): `search`, `ranking` (pure seeds + exact PPR), `links` (pure
extraction + name index), `import_` (batch/single upsert, delta-aware),
`export` (okf/obsidian flavors), `diff` (structural compare), `doctor`
(scored scan + safe fixes), `embedding` (Rust bridge + chunking),
`converters` (`DocumentConverter` protocol + `BobineConverter`),
`image_assets`, `delta`, `purge`, `schema`, `ingest`.

Key design decisions:

- **Rust-only embeddings, fail-fast** — no Python fallback; a mid-run stack
  switch would silently mix vector spaces in one index.
- **Lazy session init** — router construction never downloads the model or
  builds the ONNX session; first encode opens once (thread-safe), token
  counting uses a tokenizer-only handle, so PPR search, budgeted reads,
  diff, and doctor stay cold.
- **Last-token pooling** (not mean) — required by Jina v5; mean pooling
  breaks alignment with omni image embeddings.
- **Single pinned ORT** (`onnxruntime==1.29.0`, `ORT_DYLIB_PATH`-overridable)
  shared by bobine + embroider.
- **Bobine is a plugin, not a dependency** — `DocumentConverter.convert()`
  is the seam; provider owns its options (`routing_mode`,
  `extract_images`); missing bobine → clear `RuntimeError`, never a silent
  fallback path.
- **Deterministic by construction** — sorted traversal order, fixed PPR
  constants and accumulation order, golden fixtures under `tests/fixtures/`
  (PPR scores, diff reports, doctor scores, vault round-trips).
- **Bundle is source of truth, graph is the index** — edit markdown,
  re-import; verify ingests by searching, since empty `[]` proves wiring
  while errors prove broken setup.
- **MCP-first surface** — CLI mirrors the same 5 verbs; maintenance
  (`diff`, `doctor`) is CLI-only to keep the agent tool surface at 5.

---

## API reference (`OKFRouter` proxies)

| Method | Description |
|---|---|
| `import_from_okf(file_path, mode?)` / `import_mgr.import_bundle(dir?, batch_size?, mode?, purge_deleted?)` | Single / bulk import |
| `ingest_mgr.ingest_md / ingest_pdf / ingest_thoughts` | Kind-based ingestion with lint + chunk + embed + link |
| `search_hybrid(query, …, rank="none"\|"hub"\|"ppr")` | RRF hybrid; `hub` blends authority, `ppr` is model-free |
| `search_engine.search_with_ppr / read_with_budget / search_chunks / traverse / find_path / get_chunks / get_by_id / list_directory` | Model-free PPR, budgeted reads, retrieval + navigation |
| `embed_engine.reconstruct_document(id)` / `count_tokens(text)` | ~98% chunk→markdown rebuild; Rust counter with chars/4 fallback |
| `export_mgr.export_bundle / export_to_okf(..., flavor="okf"\|"obsidian")` | Filtered export, both flavors |
| `diff_mgr.diff_dirs(old, new)` / `diff_db_dir(bundle_dir)` | Snapshot / drift structural reports |
| `diagnose(stale_days?)` / `doctor_fix()` | Health report / safe repairs |
| `list_broken_links()` / `repair_links(skip_sources?)` | Unresolved refs; exact-id + unique-name repair |
| `model_info(...)` / `default_cache_dir()` | Cache inspection without model load |

---

## Testing

```bash
uv run --project . pytest tests/ -q        # full suite (model loads; takes a while)
uv run --project . pytest tests/test_ranking.py tests/test_mcp_server.py -q   # fast subset
# Rust unit tests live in the embroider repo:
# https://github.com/opticsWolf/embroider (cargo test --locked)
```

Golden fixtures under `tests/fixtures/` (`ppr_graph`, `diff_a`/`diff_b`,
`doctor_bundle`, `obsidian_vault`, `ppr_bundle`) lock deterministic behavior:
PPR scores byte-stable across runs, diff reports exact, doctor score pinned
(83 on the fixture bundle), obsidian export→import edge-identical. Live suites
reuse one class-scoped router each so the ONNX model loads once per class.

---

## Project structure

```text
okfgraph/
├── okfgraph/
│   ├── __init__.py        # ConceptModel, OKFRouter, cli_main
│   ├── models.py          # ConceptModel / ChunkModel / ImageAssetModel (extra frontmatter allowed)
│   ├── router.py          # OKFRouter facade (owns resources, thin proxies)
│   ├── cli.py             # okf: 5 verbs + maintenance, slim help, okfgraph.toml
│   ├── mcp_server.py      # okf-mcp: 5 tools, MCP ≥ 2.0, lifespan-managed router
│   ├── config.py          # okfgraph.toml + env + CLI merge
│   ├── images.py          # IngestMode (text|optional|omni), planning helpers
│   ├── security.py        # SSRF/domain guards for remote images
│   ├── tools.py           # legacy tool definitions (superseded by mcp_server)
│   └── components/        # ranking, links, search, lint, import_, export, diff, doctor,
│                          # embedding, converters, image_assets, delta, purge, schema, ingest
├── skills/                # okfgraph-mcp, okfgraph-cli, okfgraph-ingest (harness-neutral source)
├── tests/fixtures/        # conformance corpus: ppr, diff, doctor, obsidian, bundles
├── docs/                  # converters, harness-integration, plan-retrieval-roundup, diagnostics…
├── .mcp.json              # ready MCP wiring (uv run --project . okf-mcp)
├── architecture.md        # long-form architecture spec (v6.0, as-built for 0.2.12)
└── pyproject.toml         # slim core deps + pdf/omni/dev extras
```

---

## Requirements

Core (`uv sync`): `ladybug==0.20.3`, `embroider>=0.1,<0.2` (PyPI wheels,
github.com/opticsWolf/embroider), `onnxruntime==1.29.0`, `mordant>=0.9`,
`mcp>=2.0`, `pydantic>=2`, `pyyaml>=6`,
`numpy>=1.26`, `python-frontmatter>=1`, `fasteners>=0.19`. Python ≥ 3.11.

- `--extra pdf`: `bobine>=0.5` (default PDF converter).
- `--extra omni`: `sentence-transformers>=3`, `Pillow>=10` (image embeddings).
- `--extra dev`: `pytest>=8`.

---

## License

See LICENSE for details (Apache-2.0 OR MIT).
