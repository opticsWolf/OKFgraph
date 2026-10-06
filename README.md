# OKFgraph

[![PyPI](https://img.shields.io/pypi/v/okfgraph)](https://pypi.org/project/okfgraph/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-green)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-%E2%89%A52.0-purple)](https://modelcontextprotocol.io/)
[![Ladybug](https://img.shields.io/badge/ladybug-0.21.2-orange)](https://pypi.org/project/ladybug/)
[![Website](https://img.shields.io/badge/website-OKFgraph-blue)](https://opticswolf.github.io/OKFgraph/)

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
| **Embeddings** | Jina v5 (default `…-text-small-retrieval`, `--model-id` selects registry: text-nano) via the `embroider` Rust wheel; last-token pooling, Matryoshka truncation (default 512, per-model ladder); device `auto` (CUDA→FP16 mirror weights, CPU→FP32, `--precision` pins incl. explicit `int8`); CPU arena off by default (`--cpu-arena`); token ceiling configurable (`--max-length`); model + precision pinned per graph, never mixed. No torch anywhere (0.7.0+) |
| **Search** | Hybrid RRF fusion (vector + FTS) at concept and chunk granularity; `rank=none\|hub\|ppr` — including **PPR**: lexical seeds → exact Personalized PageRank, zero model load, deterministic |
| **Read** | Body / chunks / rebuilt document / graph context, with optional **token budgets** (`max_tokens`): self first, then PPR-ranked neighbours, index-first for context |
| **Storage** | LadybugDB `==0.21.2` (pinned — 0.20.x segfaulted 2nd in-process vector-index builds): graph + vector + FTS in one file |
| **Links** | Path links (`](doc.md)`) + `[[wikilinks]]` resolved by name (`uid` → `aliases` → `title` → filename stem); ambiguous names never resolve; broken links tracked and repairable |
| **Import** | Single files, whole bundles (delta-aware: only changed files re-embed, `--prune-missing` drops deleted concepts), markdown / PDF+Office / raw thoughts; mordant lint on the way in; `okf produce` generates bundles from SQLite (`## Observations` + `log.md` changelog) |
| **PDF** | `DocumentConverter` seam with `BobineConverter` default (routing `auto\|surgical\|always\|never`); bring your own converter, no code changes |
| **Diff** | `okf diff`: structural snapshot (dir vs dir, no model load) and drift (graph vs dir) modes; CI exit codes + `--json` |
| **Doctor** | `okf doctor`: 0–100 health score (broken/orphan/stale/duplicate-title/missing-description), safe `--fix` that never touches `reviewed: true`, `--strict` CI gate |
| **Export** | OKF round-trip with See Also / Cited By enrichment + index files, or `--flavor obsidian` (`[[Title]]` wikilinks, no index files, edge-lossless re-import) |
| **Images** | Caption-based (`mode=text`) or image-content (`mode=optional`/`omni` via the ONNX vision model, needs a text-nano graph; 0.8.0+) content-hash dedup, `okf-asset://` protocol |
| **MCP** | 8 tools (`search`, `read`, `traverse`, `ingest`, `export_bundle`, `export_concept`, `list_images`, `get_image`), MCP ≥ 2.0 (`MCPServer` + `ToolAnnotations`), stdio transport; success result = envelope, failures = `isError: true` + error envelope (D7) |
| **CLI** | Same 8 core verbs (`search`, `read`, `traverse`, `ingest`, `import`, `export`, `diff`, `doctor`) plus maintenance (`shell`, `lint`, `produce`, `reindex`, `broken-links`, `deleted-*`, `delete`, `images`, `image`, …), global `--json` (D5), slim per-command help, `okfgraph.toml` config |
| **Skills** | 3 harness-neutral skills (`okfgraph-mcp`, `okfgraph-cli`, `okfgraph-ingest`), `.mcp.json` wiring included |

---

## Installation

Requires Python ≥ 3.11. You must pick **exactly one** ONNX Runtime
extra, `cpu` or `gpu`:

### CPU or GPU?

| | `[gpu]` | `[cpu]` |
|---|---|---|
| **Pick it when** | the machine has an NVIDIA GPU (`nvidia-smi` prints it) | no NVIDIA GPU, macOS, CI, or a small install |
| Runtime | `onnxruntime-gpu[cuda,cudnn]==1.29.0` (bundles the CUDA + cuDNN runtime wheels: a large download) | `onnxruntime==1.29.0` (small) |
| `--device auto` lands on | CUDA, FP16 weights (much faster bulk import) | CPU, FP32 weights |
| Platforms | Windows, Linux x86-64 with a recent NVIDIA driver | all |

Not sure? Run `nvidia-smi`: if it lists a GPU, take `[gpu]`. After
installing, `okf doctor` shows `cuda_usable=True/False`. With a GPU
present but the CPU runtime installed it adds an `ort_gpu_unused` note,
and `--device auto` logs a warning when it falls back to CPU. Switching
later is safe: the extras conflict, so installing one replaces the
other. Vectors made on CPU (FP32) and GPU (FP16) cannot share one graph
(the precision pin refuses it), so pick before the first import or plan
a reimport.

**PyPI**

```bash
pip install "okfgraph[gpu,pdf]"   # NVIDIA GPU
pip install "okfgraph[cpu,pdf]"   # everything else
```

**From source** with `uv` (no Rust toolchain needed: the embedding
engine is the external `embroider` package). Name every extra in **one**
`uv sync`: each sync makes the venv exactly what it names, so a later
`uv sync --extra pdf` would uninstall the runtime again.

```bash
git clone <repo> && cd OKFgraph
uv sync --extra gpu --extra pdf --extra dev   # NVIDIA GPU
uv sync --extra cpu --extra pdf --extra dev   # everything else
```

(`pdf` = bobine PDF converter, `dev` = pytest; drop what you don't need.
Bare `uv sync` installs the core only, with no runtime: the first
encode then fails with an install hint.)

Prefer a system-wide ONNX Runtime (conda, distro package, self-built)?
Skip both extras and point `ORT_DYLIB_PATH` at your onnxruntime 1.29
library instead — but never rely on bare OS-loader discovery: a stale
system DLL (e.g. in `System32`) fails the load instead of being used.

Core dependencies are deliberately few: `ladybug==0.21.2`, `embroider>=0.3.2,<0.4`
(the shared embedding engine — github.com/opticsWolf/embroider, also used by
bobine), `onnxruntime==1.29.0` (one pinned ORT binary shared by bobine +
embroider; `ORT_DYLIB_PATH`-overridable), `mordant`, `mcp>=2.0`, `pydantic`,
`pyyaml`, `numpy`, `python-frontmatter`, `fasteners`. There is no torch, no
transformers, no optimum, no PDF stack in the core install.

---

## Quick start

```bash
okf init --db-path kb.db --bundle-root kb/   # once; persists to okfgraph.toml
okf import --all                             # delta-aware bulk import
okf search "honey badger defense"         # hybrid semantic search
okf search --rank ppr "honey badger"      # same question, no model load
okf read savanna --include context --max-tokens 1500
okf traverse savanna --relationship LINKS_TO --direction BOTH
okf diff                                  # drift: graph vs bundle dir
okf doctor                                # health score + findings
okf lint kb/                               # pre-import gate: frontmatter + links, no model
okf produce --from sqlite --source shop.db --output-dir kb/  # SQLite → bundle (lint pre-flighted)
okf images savanna                        # image assets on one concept
okf image img-1 --output-path asset.png   # fetch one asset
```

### Programmatic usage

```python
from pathlib import Path
from okfgraph import OKFRouter

router = OKFRouter(db_path="kb.db", bundle_root="kb/", embedding_dim=512)

# One vocabulary: every operation lives on the router (24 ops), raises
# OKFError(code) with a stable exit plan, and returns dict/list data.

# Import: whole bundle (delta-aware), one file, or raw reasoning
result = router.import_bundle(None)   # every configured root -> {"concept_ids", "images"}
cid = router.import_file("kb/auth.md", mode="text")["concept_id"]
th = router.ingest("thoughts", thoughts="Session decided X because Y",
                   topic="auth-refactor")["concept_id"]

# Search: hybrid (default), hub-blended, or model-free PPR
hits   = router.search("session auth decisions", limit=5)
hubbed = router.search("session auth decisions", rank="hub")
cold   = router.search("auth refactor", rank="ppr")  # no ONNX load

# Read: full shapes, or a token-budgeted section list
doc     = router.read("auth", include="body")
reading = router.read("auth", include="context", max_tokens=1500)

# Image assets are first-class read ops
imgs = router.list_images(cid)
row = router.get_image(imgs[0]["id"])

# Maintain: drift, health, links
print(router.diff(Path("kb/"))["identical"])   # in sync; differs -> DIFF_DIFFERENT (report on err.data)
print(router.doctor()["report"]["score"])      # 0-100
print(router.doctor(stale_days=365, fix=True))  # repair pass, then re-score
print(router.repair_links())                    # re-point broken links -> {"repaired": n}

# Export: OKF bundle or Obsidian vault
router.export_bundle(Path("out-okf/"))
router.export_bundle(Path("out-vault/"), flavor="obsidian")

router.close()
```

### Ingestion kinds

| Kind | Entry point | Notes |
|---|---|---|
| `md` | `router.ingest("md", md_path=…, concept_id?, title?, description?, tags?, mode?) -> {"concept_id","chunk_count","image_count","lint_issues",…}` | mordant-linted, report-only (`lint_issues.fixable_count` — the source file is never rewritten); `image_count` counts real image refs resolved next to the file; frontmatter wins over overrides |
| `pdf` | `router.ingest("pdf", pdf_path=…, routing_mode?, extract_images?, auto_import?, output_dir?) -> {"concept_ids","page_count","md_path","image_dir"}` | bobine converter; auto-imports by default (`auto_import=False` / `--no-auto-import` converts only, into `output_dir` or next to the source) |
| `thoughts` | `router.ingest("thoughts", thoughts=…, topic=…, concept_id?, tags?) -> {"concept_id","topic","chunk_count",…}` | cheapest, highest value: persist session reasoning with a stable topic scheme (`auth-refactor`, `api-design`) |

Params that belong to another kind are refused (`BAD_VALUE` naming them),
not silently dropped.

Frontmatter that matters: `title`, `type`, `tags`, `aliases: [...]` (wikilink
names), `id:` (stable identity, preserved as `uid`, written back on export),
`reviewed: true` (immune to `doctor --fix`). Credentials in `resource:`
URIs are stripped to `***@` at parse time (graphs imported before 0.2.3
keep old values until re-import). Rule of thumb: thoughts > md >
pdf — reasoning you already hold beats re-extracting it from files.

---

## CLI reference

Eight verbs mirror the MCP tools; maintenance commands cover the rest. Global
flags (`--db-path`, `--bundle-root`, `--embedding-dim`, `--model-id`,
`--max-length`, …) are documented once in `okf --help`, accepted everywhere,
and usually live in `okfgraph.toml`. Every command accepts `--json` (D5):
human renderer by default, the result envelope
(`{ok, op, data, warnings, error}`) on stdout with `--json`. Results always
go to stdout and logs to stderr (`-q` silences logs, never results).
Failures print `[ERROR] CODE: message (remedy)` on stderr (plus the error
envelope on stdout under `--json`) and exit 0 ok / 1 state·outcome / 2
usage. Outcomes (`lint` errors, `diff` differences, `doctor --strict`
findings) still render their report before exiting 1. Typed state codes
include `DB_LOCKED` (the db is open in another process — `okf` commands
run one at a time per database; when `okf-mcp` is serving, query through
it), `NO_ORT_RUNTIME` (no/broken ONNX Runtime), and the pin set.
Concurrency: one process per db file; the 0-byte `<db>.lock` beside it is
ladybug's own lock marker and is harmless when no process holds it.

| Command | Description |
|---|---|
| `okf search QUERY [--target concepts\|chunks\|images] [--rank none\|hub\|ppr] [--expand] [--hub-rerank]` | Concepts (RRF hybrid), chunks (passages), images; `--rank ppr` is model-free |
| `okf read ID [--include body\|chunks\|document\|context] [--max-tokens N]` | Full shapes uncapped; budgeted section list with `--max-tokens` |
| `okf traverse [ID] [--relationship CONTAINS\|LINKS_TO\|PART_OF\|INCLUDES_ASSET] [--direction …] [--target ID]` | Relationships, directory listing (empty ID = root), shortest path via `--target` |
| `okf ingest --kind md\|pdf\|thoughts …` | `--md-path`, `--pdf-path` (`--routing-mode`, auto-imports by default; `--no-auto-import` converts only), `--thoughts --topic` |
| `okf export --all\|--concept-id ID --output-dir DIR [--flavor okf\|obsidian]` | Bundle or single-concept export; obsidian = `[[Title]]` links, no index files |
| `okf images CONCEPT_ID` / `okf image ASSET_ID [--output-path F]` | Image assets on a concept; metadata (+ bytes) per asset |
| `okf diff [OLD] [NEW] [--json]` | Snapshot (two dirs, no model) or drift (graph vs dir); exit 0 identical / 1 different |
| `okf lint [DIR] [--json]` | Pre-import gate (no DB, no model); exit 0 clean / 1 errors / 2 bad dir |
| `okf doctor [--fix] [--strict] [--stale-days N] [--json]` | Score + findings; `--fix` repairs safely, `--strict` exits 1 on any finding |
| `okf import [FILE...] [--all] [--prune-missing] [--mode text] [--force]` | Single files / bulk import, delta-aware (`--force` re-attaches a detached graph); TOML `[[roots]]` (or per-call `--root ALIAS=PATH`) add named roots (`@alias/` IDs, unmounted ≠ deleted, `--prune-missing` skips while any root is absent); `--bundle-path DIR` pins one tree for this call |
| `okf produce --from sqlite --source DB --output-dir DIR` | Generate a bundle from a data source (one concept per table, FK links, `## Observations` notes, `log.md` changelog); lint pre-flighted, CLI-only |
| `okf detach [--bundle-path DIR] [--no-verify] [--force]` | End the mirror: the DB becomes the artifact (verify-first; imports refuse without `--force`) |
| `okf init`, `okf model-info`, `okf shell`, `okf reindex`, `okf broken-links`, `okf repair-links`, `okf deleted-*`, `okf delete ID` | Setup, cache inspection, REPL, index rebuild, link + soft-delete maintenance (`delete` refuses file-backed concepts while their source exists) |

Only commands that embed text (`search` except `--rank ppr`, `import`,
`ingest`) load the model (~30s cold per process); `read`, `traverse`,
`lint`, `diff`, `doctor` and `export` never do — batch searches, use
`okf shell` for a warm session, and reach for `--rank ppr` for topic
queries in cold sessions. Omit `--bundle-root` entirely for file-free mode: thoughts ingest,
search, read, traverse, doctor, and export work from the DB alone (file-side
ops fail fast naming the missing root). Thought IDs are namespaced
(`thoughts/<topic>/<ts>_<id>`), so a fileless graph exports into a tidy tree.

---

## MCP server

8 tools, MCP ≥ 2.0, stdio transport. Wire once per project with a **stable**
database — `--db-path`, `OKFGRAPH_DB_PATH` or `[database] db_path` in
`okfgraph.toml`; the server refuses to boot without one (`CONFIG_INVALID`,
exit 2) since a temp dir means amnesia every session. `--bundle-root`
defaults to the database's parent directory. First
boot creates the schema; pre-approve the `ingest` / `export_bundle` /
`export_concept` write tools for knowledge workflows. `.mcp.json` ships a
ready config (`uv run --project . okf-mcp`). Every tool returns the result
envelope (`{ok, op, data, warnings, error}`); failures raise so the result
carries `isError: true` with the error envelope as text — never a bare
traceback.

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
| `export_concept(concept_id, output_dir, flavor?)` | Single concept → `<output_dir>/<id>.md` |
| `list_images(concept_id)` | Image assets on one concept (metadata) |
| `get_image(asset_id)` | One asset: metadata + base64 `data` |

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
                ┌──────────── MCP (8 tools) / CLI (8 verbs + maintenance)
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
              Rust embroider (Jina v5, ORT) — device `auto` (CUDA→FP16 weights
              from the published mirror, CPU→FP32; `--precision` pins it),
              CPU arena off by default (`--cpu-arena` opts in), tokenize
              (ceiling `--max-length`, default 8192, max 32768), last-token
              pool, L2-norm, Matryoshka truncate; precision pinned per graph
              (Meta, fail-closed); numerics pinned by tests/test_parity.py
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
  breaks alignment with stored vectors.
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
- **MCP-first surface** — CLI mirrors the same verbs; maintenance
  (`diff`, `doctor`) is CLI-only to keep the agent tool surface at 8.

---

## API reference (`OKFRouter` ops)

One vocabulary of 24 operations on the facade (mixins `QueryOps`,
`IngestOps`, `ExportOps`, `AdminOps`); every failure raises
`OKFError(code, message, fields?, remedy?)` — `UsageError` (exit 2) for
caller mistakes, `StateError`/`OutcomeError` (exit 1) for graph state and
domain outcomes (the class always follows the code). Outcome errors
(`DIFF_DIFFERENT`, `DOCTOR_FINDINGS`, `LINT_ERRORS`) carry the full result
on `err.data`. A passed param the chosen path ignores is refused with
`BAD_VALUE` naming it in `fields["ignored"]`.

| Op | Description |
|---|---|
| `search(query, target="concepts", limit?, concept_type?, tags?, parent_id?, include_chunks?, max_chunks_per_doc?, expand?, context_hops?, hub_rerank?, hub_weight?, rank="none"\|"hub"\|"ppr")` | RRF hybrid; `hub` blends authority, `ppr` is model-free; chunk paths: `hub_rerank` > `expand` > plain |
| `read(concept_id, include="body"\|"chunks"\|"document"\|"context", max_tokens?)` | Concept dict / chunk dicts (no embeddings) / `{"concept_id","markdown"}` / `{incoming_links, outgoing_links, ancestry, siblings}`; or a budgeted `{sections, used, budget, truncated}` |
| `traverse(start_id, relationship?, direction?, depth?, node_type?, target?, max_path_length?)` | Walk, list (empty id = root), connect |
| `ingest(kind, …)` | `md` / `pdf` / `thoughts`, per-kind params, lint + chunk + embed + link |
| `import_file(path, mode?) {"concept_id","images"}` / `import_bundle(path?, …, prune_missing?) {"concept_ids","images"}` | Single / whole-bundle delta import |
| `export_bundle(output_dir, directory_id?, concept_type?, tags?, flavor?) {"concept_ids",…}` / `export_concept(concept_id, output_dir, flavor?) {"path"}` | Both flavors |
| `list_images(concept_id)` / `get_image(asset_id)` | Image ops (metadata; bytes + base64) |
| `init() {"db_path","embedding_dim","model_id"}` / `model_info(…)` (static) / `reindex(if_dirty?)` | Schema, cache inspection, index rebuild |
| `doctor(stale_days?, fix?, strict?) {"report","fixed"}` / `lint(dir)` (static) / `produce(source_type, source_path?, output_dir?, prefix?, overwrite?)` (static) / `diff(old?, new?)` / `diff_dirs(old, new)` (static) | Health, pre-import gate, bundles from sources, structural diff; `strict` findings, lint errors and differences raise outcome errors with the result on `err.data` |
| `list_deleted()` / `recover_deleted(id)` / `purge_deleted(older_than?)` / `delete(concept_id)` / `detach(bundle_path?, verify?, force?)` | Soft-delete trio + `delete` (refuses file-backed concepts whose source exists — remedy: prune-missing) + mirror lifecycle |
| `list_broken_links() [{source, target}]` / `repair_links() {"repaired"}` | Unresolved refs; exact-id + unique-name repair |

Components remain reachable for advanced use (`search_engine.search_hybrid`,
`embed_engine.count_tokens`, `export_mgr`, `diff_mgr`, …); the ops are the
stable surface. `embed_engine.reconstruct_document(id)` stays the ~98%
chunk→markdown rebuild.

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
│   ├── settings.py        # okfgraph.toml + env + CLI merge (per-key precedence, config spellings raise)
│   ├── errors.py          # OKFError(code, fields, remedy) + UsageError / StateError / OutcomeError
│   ├── router.py          # OKFRouter facade = OKFRouter(AdminOps, ExportOps, IngestOps, QueryOps)
│   ├── ops/               # the 24-op vocabulary: query.py, ingest.py, export.py, admin.py
│   ├── cli.py             # okf: 8 core verbs + maintenance + images/image, --json (D5/D6)
│   ├── mcp_server.py      # okf-mcp: 8 tools, MCP ≥ 2.0, lifespan-managed router, envelope/_ToolFailure
│   ├── images.py          # IngestMode (text + vision-onnx since 0.8.0), planning helpers
│   ├── security.py        # SSRF/domain guards for remote images
│   └── components/        # ranking, links, search, lint, import_, export, diff, doctor,
│                          # embedding, converters, image_assets, delta, purge, schema, ingest
├── skills/                # okfgraph-mcp, okfgraph-cli, okfgraph-ingest (harness-neutral source)
├── tests/fixtures/        # conformance corpus: ppr, diff, doctor, obsidian, bundles
├── docs/                  # architecture spec (v6.5), converters, chunking, harness guides; archive/ holds completed plans
├── .mcp.json              # ready MCP wiring (uv run --project . okf-mcp)
└── pyproject.toml         # slim core deps + pdf/dev extras
```

---

## Requirements

Core (`uv sync`): `ladybug==0.21.2`, `embroider>=0.3.2,<0.4` (PyPI wheels,
github.com/opticsWolf/embroider), `mordant>=0.9`,
`mcp>=2.0`, `pydantic>=2`, `pyyaml>=6`,
`numpy>=1.26`, `python-frontmatter>=1`, `fasteners>=0.19`. Python ≥ 3.11.
Plus exactly one ORT provider: `okfgraph[cpu]` (`onnxruntime==1.29.0`)
or `okfgraph[gpu]` (`onnxruntime-gpu[cuda,cudnn]==1.29.0`) — shared with
bobine via `ORT_DYLIB_PATH`.

- `--extra pdf`: `bobine>=0.6` (default PDF converter).
- `--extra dev`: `pytest>=8`.

---

## Model licences

okfgraph's **code** is Apache-2.0 OR MIT. The **model weights** it downloads
at runtime are not part of this repository and carry their own licences.
Check them before any commercial use.

| Used for | Model (Hugging Face repo) | Fetched by | Licence |
|---|---|---|---|
| Text embeddings (default, fp32) | `jinaai/jina-embeddings-v5-text-small-retrieval` | embroider | CC BY-NC 4.0 |
| Text embeddings (fp16) | `opticsWolf/jina-embeddings-v5-text-small-retrieval-onnx-fp16` (ONNX conversion of the above) | embroider | CC BY-NC 4.0 |
| Text embeddings (nano) | `jinaai/jina-embeddings-v5-text-nano-retrieval` | embroider | CC BY-NC 4.0 |
| Image embeddings (≤0.6.x `--extra omni`, removed 0.7.0) | `jinaai/jina-embeddings-v5-omni-small-retrieval` | sentence-transformers (torch path, removed) | CC BY-NC 4.0 |
| Image embeddings (vision, 0.8.0+) | `opticsWolf/jina-embeddings-v5-omni-nano-retrieval-onnx` (dynamic-grid ONNX export of `jinaai/jina-embeddings-v5-omni-nano-retrieval`) | embroider (`JinaV5Vision`) | CC BY-NC 4.0 |
| PDF layout (`--extra pdf`) | `wybxc/DocLayout-YOLO-DocStructBench-onnx` | bobine | Apache-2.0 |
| PDF OCR det/rec (`--extra pdf`) | `SWHL/RapidOCR` (PP-OCRv4) | bobine | Apache-2.0 |
| PDF tables (`--extra pdf`) | `opendatalab/PDF-Extract-Kit-1.0` (`models/TabRec/SlanetPlus/slanet-plus.onnx`) | bobine | repo declares AGPL-3.0 |
| PDF formulas (`--extra pdf`) | `OleehyO/TexTeller` | bobine | Apache-2.0 |

- **Every Jina v5 model is non-commercial (CC BY-NC 4.0).** okfgraph cannot
  embed anything without one, so using okfgraph commercially requires a
  commercial licence from Jina AI (see the model cards).
- **PDF tables:** bobine fetches SLANet-plus from a repository whose card
  declares AGPL-3.0. Accepted as a runtime fetch (2026-10-01): weights are
  downloaded, never redistributed. Review before redistributing a service
  built on `--extra pdf`.
- Licences are as declared in each repository's model card when checked
  (2026-09-27). The model cards are authoritative and can change. This
  table is a pointer, not legal advice.

---

## License

The code is licensed Apache-2.0 OR MIT (see LICENSE). Downloaded model
weights are licensed separately; see [Model licences](#model-licences).
