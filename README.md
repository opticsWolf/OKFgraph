# OKFgraph 0.16.0

[![PyPI](https://img.shields.io/badge/pypi-okfgraph-blue)](https://pypi.org/project/okfgraph/)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/core-Rust-orange)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-green)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-%E2%89%A52.0-purple)](https://modelcontextprotocol.io/)
[![Ladybug](https://img.shields.io/badge/ladybug-0.22.1-orange)](https://pypi.org/project/ladybug/)
[![Website](https://img.shields.io/badge/website-OKFgraph-blue)](https://opticswolf.github.io/OKFgraph/)

**Ladybug-backed knowledge graph with Rust retrieval, model-free graph ranking,
and agent-first MCP + CLI + HTTP surfaces.**

OKFgraph builds, queries, and maintains a knowledge graph from Markdown/OKF
documents. It combines hybrid semantic search (vector + FTS, RRF-fused),
chunk-level retrieval, **model-free PPR retrieval that works with no embedding
model loaded**, structural diffing, scored health checks, and Obsidian-vault
compatible wikilinks — in a single LadybugDB file.

Starting with 0.16.0 the implementation is **Rust**. One native core powers the
CLI, MCP server, HTTP owner and Python facade. Python is a thin PyO3 layer with
the same public API as 0.15; there is no Python core and no Python fallback.

Design stance: **one wheel, few dependencies, no legacy fallbacks.** Embeddings
run through the `embroider` Rust crate (ONNX Runtime, no torch / transformers /
optimum anywhere). PDF and Office conversion is the `bobine` Rust engine, bundled
inside the wheel. What isn't needed isn't installed.

---

## Features

| Category | Features |
|---|---|
| **Core** | Rust core in `src/`; 30 router ops shared by the Python facade, CLI, HTTP and MCP; transactional imports; one writer per database |
| **Embeddings** | `google/embeddinggemma-2` (default), Jina v5 text-nano, and the gemma vision pairing through `embroider`; device `cpu`/`cuda`, precision `auto`/`fp32`/`fp16`/`int8`; model, precision and vector space pinned per graph, never mixed |
| **Search** | Hybrid RRF fusion (vector + FTS) at concept and chunk granularity; `rank=none\|hub\|ppr\|cross`; PPR is lexical-seeded and model-free; chunk collapse, expansion, token budgets, filters |
| **Read** | Body / chunks / rebuilt document / graph context, with optional token budgets (`max_tokens`) |
| **Storage** | LadybugDB `==0.22.1` (pinned): graph + vector + FTS in one file; schema v10 |
| **Links** | Path links (`](doc.md)`) and `[[wikilinks]]` resolved by `uid` → `aliases` → `title` → filename stem; ambiguity never resolves; broken links tracked and repairable |
| **Import** | Single files, bundles (delta-aware: only changed files re-embed, `--prune-missing` drops deleted concepts), PDF/Office, raw thoughts, BibTeX; mordant lint on the way in; `okf produce` builds bundles from SQLite |
| **PDF** | Bundled native converter (`bobine`): PDF, DOCX/XLSX/PPTX and legacy Office; routing `auto\|surgical\|always\|never`; one warm converter per batch |
| **Diff** | `okf diff`: structural snapshot (dir vs dir, no model) and drift (graph vs dir); CI exit codes and `--json` |
| **Doctor** | `okf doctor`: 0–100 health score, safe `--fix` that never touches `reviewed: true`, `--strict` CI gate, cache/stale/lock probes |
| **Export** | OKF round-trip with See Also / Cited By enrichment and index files, or `--flavor obsidian` (`[[Title]]` wikilinks, no index files, edge-lossless re-import) |
| **Images** | Caption-based (`mode=text`) or image-content (`mode=optional`/`omni`) with content-hash dedup and the `okf-asset://` protocol; unsafe asset ids are refused |
| **Surfaces** | Python (`OKFRouter`, `OKFClient`), 29-command `okf` CLI, 11 MCP tools over stdio or `/mcp`, and one `okf serve` owner over TCP or POSIX UDS |
| **Jobs** | In-memory HTTP job lane; `reembed` is job-capable and claims the active slot before it is queued |

---

## Installation

Requires Python 3.12, 3.13 or 3.14. The extras decide what you get:

| Extra | Installs | Gives you |
|---|---|---|
| `cpu` **or** `gpu` (pick exactly one) | `onnxruntime==1.31.1` / `onnxruntime-gpu[cuda,cudnn]==1.31.1` | the ONNX Runtime that embedding, search and rerank run on |
| `pdf` | nothing (compatibility alias) | PDF/Office conversion is already in the wheel |
| `dev` | `pytest` | the Python compatibility suite |

**Why the runtime isn't installed by default.** `onnxruntime` and
`onnxruntime-gpu` are two distributions of the same `onnxruntime` module, and
pip can't pick one from the hardware or install both side by side. A bare
`pip install okfgraph` installs, but the first embedding fails with
`NO_ORT_RUNTIME` and the install line to run.

**Why `pdf` is a no-op.** 0.16 ships one wheel containing the facade, the Rust
core and the `bobine` converter. The extra stays so 0.15 install commands
(`okfgraph[cpu,pdf]`) keep working.

### CPU or GPU?

| | `[gpu]` | `[cpu]` |
|---|---|---|
| **Pick it when** | the machine has an NVIDIA GPU (`nvidia-smi` prints it) | no NVIDIA GPU, macOS, CI, or a small install |
| Runtime | `onnxruntime-gpu[cuda,cudnn]==1.31.1` (large download) | `onnxruntime==1.31.1` (small) |
| `--device auto` lands on | CUDA, FP16 weights | CPU, FP32 weights |
| Platforms | Windows and Linux x86-64 with a recent NVIDIA driver | all |

Not sure? Run `nvidia-smi`: if it lists a GPU, take `[gpu]`. After installing,
`okf doctor` shows `cuda_usable=True/False`. With a GPU present but the CPU
runtime installed it adds an `ort_gpu_unused` note. Switching later is safe for
the install, but vectors made on CPU (FP32) and GPU (FP16) cannot share one
graph (the precision pin refuses it), so pick before the first import or plan a
reimport.

**PyPI**

```bash
pip install "okfgraph[gpu,pdf]"   # NVIDIA GPU
pip install "okfgraph[cpu,pdf]"   # everything else
```

**Supported targets:** Windows x64, Linux x86_64 and macOS arm64. Python 3.11,
Linux aarch64 and macOS x86_64 are not part of the first release.

**From source** needs a Rust toolchain (1.88+) and OpenSSL. On Windows use
vcpkg:

```bash
export VCPKG_ROOT='C:\path\to\vcpkg'
export VCPKGRS_DYNAMIC=1
vcpkg install --classic openssl:x64-windows
python bindings/python/build_wheel.py --release
```

On macOS install `openssl@3` and set `OPENSSL_ROOT_DIR=$(brew --prefix
openssl@3)`. Linux needs the usual OpenSSL development package.

Prefer a system-wide ONNX Runtime (conda, distro package, self-built)? Skip
both extras and point `ORT_DYLIB_PATH` at your onnxruntime 1.31 library instead
— but never rely on bare OS-loader discovery: a stale system DLL (e.g. in
`System32`) fails the load instead of being used.

After installing, copy the agent skills into your harness:
`okf setup skills` writes them to `.claude/skills/` (Claude Code); `--to DIR`
for another harness, `--skill NAME` for one of them.

### Models in another process

Each model can run in its own server instead of inside okfgraph:
`[embedding] backend = "external"` (text), `[search] rerank_backend =
"external"` (reranker) and `[embedding] image_backend = "external"` (vision),
each with an endpoint URL (`endpoint_url`, `rerank_endpoint_url`,
`image_endpoint_url`). An embroider host serves any of them:

```bash
python -m embroider.serve --kind vision --model <image model id> --dim <graph dim> --port 8091
```

```toml
[embedding]
image_backend = "external"
image_endpoint_url = "http://127.0.0.1:8091"
```

The image lane has no declared mode: the host must be embroider and report its
model, width, precision, vector space and text partner on `GET /v1/models`;
okfgraph refuses a host whose model, width or text partner doesn't match the
graph. okfgraph sends the original image bytes and the host resizes them, so
vectors match the in-process lane. An unreachable host is
`EXTERNAL_UNAVAILABLE`.

---

## Quick start

```bash
okf init --db-path kb.db --bundle-root kb/   # once; persists to okfgraph.toml
okf import --all                             # delta-aware bulk import
okf search "honey badger defense"            # hybrid semantic search
okf search --rank ppr "honey badger"         # same question, no model load
okf read savanna --include context --max-tokens 1500
okf traverse savanna --relationship LINKS_TO --direction BOTH
okf diff                                     # drift: graph vs bundle dir
okf doctor                                   # health score + findings
okf lint kb/                                 # pre-import gate, no model
okf produce --from sqlite --source shop.db --output-dir kb/
okf images savanna                           # image assets on one concept
okf image img-1 --output-path asset.png      # fetch one asset
```

### Programmatic usage

```python
from pathlib import Path
from okfgraph import OKFRouter

# The constructor creates a missing database and its parent directory.
router = OKFRouter(db_path="kb.db", bundle_root="kb/", embedding_dim=512)

# Import: whole bundle (delta-aware), one file, or raw reasoning
result = router.import_bundle(None)          # every configured root
cid = router.import_file("kb/auth.md", mode="text")["concept_id"]
th = router.ingest("thoughts", thoughts="Session decided X because Y",
                   topic="auth-refactor")["concept_id"]

# Search: hybrid (default), hub-blended, model-free PPR, cross-encoder
hits   = router.search("session auth decisions", limit=5)
hubbed = router.search("session auth decisions", rank="hub")
cold   = router.search("auth refactor", rank="ppr")
ranked = router.search("auth refactor", rank="cross")

# Read: full shapes, or a token-budgeted section list
doc     = router.read("auth", include="body")
reading = router.read("auth", include="context", max_tokens=1500)

# Image assets are first-class read ops
imgs = router.list_images(cid)
row = router.get_image(imgs[0]["id"])

# Maintain: drift, health, links
print(router.diff(Path("kb/"))["identical"])    # differs -> DIFF_DIFFERENT on err.data
print(router.doctor()["report"]["score"])       # 0-100
print(router.doctor(stale_days=365, fix=True))
print(router.repair_links())                    # {"repaired": n}

# Export: OKF bundle or Obsidian vault
router.export_bundle(Path("out-okf/"))
router.export_bundle(Path("out-vault/"), flavor="obsidian")

router.close()
```

The supported Python surface is documented in
[`docs/python-binding.md`](docs/python-binding.md). Internal modules
(`okfgraph.components.*`, `okfgraph.ops.*`) are intentionally gone; use the
router, the CLI or MCP.

### Ingestion kinds

| Kind | Entry point | Notes |
|---|---|---|
| `md` | `router.ingest("md", md_path=…, concept_id?, title?, description?, tags?, mode?) -> {"concept_id","chunk_count","image_count","lint_issues",…}` | mordant-linted, report-only (the source file is never rewritten); frontmatter wins over overrides |
| `pdf` | `router.ingest("pdf", pdf_path=…, routing_mode?, extract_images?, auto_import?, output_dir?) -> {"concept_ids","page_count","md_path","image_dir"}` | bundled converter; auto-imports by default (`auto_import=False` converts only) |
| `thoughts` | `router.ingest("thoughts", thoughts=…, topic=…, concept_id?, tags?) -> {"concept_id","topic","chunk_count",…}` | persist session reasoning under `thoughts/<topic>/<ts>_<id>` |
| `bib` | `router.ingest("bib", bib_path=…, tags?) -> {"concept_ids","entry_count","skipped_entries",…}` | one `reference` concept per `@entry` as `refs/<key>`; malformed entries are reported per entry, never fatal |

Params that belong to another kind are refused (`BAD_VALUE` naming them), not
silently dropped. `okf ingest --kind pdf --pdf-path A.pdf B.docx` (or a folder
or a `*.pdf` pattern) converts a batch with one warm converter and reports each
file.

Frontmatter that matters: `title`, `type`, `tags`, `aliases: [...]`,
`id:` (stable identity, preserved as `uid`, written back on export),
`reviewed: true` (immune to `doctor --fix`). Credentials in `resource:` URIs
are stripped to `***@` at parse time. Rule of thumb: thoughts > md > pdf —
reasoning you already hold beats re-extracting it from files.

---

## CLI reference

The native `okf` binary exposes the full 29-command oracle surface. Global
flags (`--db-path`, `--bundle-root`, `--embedding-dim`, `--model-id`,
`--max-length`, …) are documented once in `okf --help`, accepted before or
after the command, and usually live in `okfgraph.toml`.

| What | Set by | Default |
|---|---|---|
| database | `--db-path`, `OKFGRAPH_DB_PATH`, `[database] db_path` | `.okfgraph/okfgraph.db` in the current directory (`okf-mcp` requires one) |
| bundle root | `--bundle-root`, `OKFGRAPH_BUNDLE_ROOT`, `bundle_root` | `.` for the CLI; the database parent for `okf serve`/MCP |
| config file | — | `okfgraph.toml` in the current directory, else in the bundle root, else the user config dir (first found wins) |

Precedence is flag > environment > `okfgraph.toml` > default. `OKFRouter` and
`okf` create a missing database and parent directories; a read-only Rust store
refuses one. Every command accepts `--json`: human output by default, the
envelope `{ok, op, data, warnings, error}` on stdout with `--json`. Results go
to stdout and logs to stderr (`-q` silences logs, never results). Failures
print `[ERROR] CODE: message (remedy)` on stderr and exit 0 ok / 1
state·outcome / 2 usage. Outcomes (`lint` errors, `diff` differences,
`doctor --strict` findings) still render their report before exiting 1.

Concurrency: one process per database file. Writers take an inter-process
`<db>.lock` compatible with Python `fasteners` (300 s timeout); a live
`okf serve` serializes ops, jobs and MCP through one owner lock, and CLI/MCP
calls borrow that owner unless `OKFGRAPH_NO_SERVER=1`. `DB_LOCKED` means the
database is held elsewhere — query through the owner instead.

Unexpected failures (`INTERNAL`) write a redacted traceback report with argv,
versions and platform to `OKF_CRASH_DIR` or the platform state directory
(newest 20 kept). The final error line names the file, and `--json` carries it
in `error.fields.crash_report`.

| Command | Description |
|---|---|
| `okf search QUERY [--target concepts\|chunks\|images] [--rank none\|hub\|ppr\|cross] [--expand] [--hub-rerank]` | Concepts (RRF hybrid), chunks, images; `--rank ppr` is model-free |
| `okf read ID [--include body\|chunks\|document\|context] [--max-tokens N]` | Full shapes uncapped; budgeted section list with `--max-tokens` |
| `okf traverse [ID] [--relationship CONTAINS\|LINKS_TO\|PART_OF\|INCLUDES_ASSET] [--direction …] [--target ID]` | Relationships, directory listing (empty ID = root), shortest path via `--target` |
| `okf ingest --kind md\|pdf\|thoughts\|bib …` | `--md-path`, `--pdf-path FILE\|DIR\|PATTERN...`, `--thoughts --topic`, `--bib-path` |
| `okf export --all\|--concept-id ID --output-dir DIR [--flavor okf\|obsidian]` | Bundle or single-concept export |
| `okf images CONCEPT_ID` / `okf image ASSET_ID [--output-path F]` | Image metadata; `image` also returns bytes |
| `okf diff [OLD] [NEW] [--json]` | Snapshot (two dirs, no model) or drift (graph vs dir) |
| `okf lint [DIR] [--json]` | Pre-import gate (no DB, no model) |
| `okf doctor [--fix] [--strict] [--stale-days N] [--cache-dir DIR] [--json]` | Score + findings; `--fix` repairs safely, `--strict` exits 1 on any finding |
| `okf import [FILE...] [--all] [--prune-missing] [--mode text] [--force]` | Delta-aware import; TOML `[[roots]]` or `--root ALIAS=PATH` add named roots; `--bundle-path DIR` pins one tree |
| `okf produce --from sqlite --source DB --output-dir DIR` | Generate a bundle from SQLite; lint pre-flighted, CLI-only |
| `okf detach [--bundle-path DIR] [--no-verify] [--force]` | End the mirror: the DB becomes the artifact |
| `okf serve --listen HOST:PORT\|unix:///PATH` | One owner for HTTP ops, jobs and `/mcp`; writes `<db>.serve.json` |
| `okf init`, `okf model-info`, `okf shell`, `okf reindex`, `okf reembed [--dry-run]`, `okf broken-links`, `okf repair-links`, `okf retitle-thoughts`, `okf deleted-*`, `okf delete ID`, `okf update`, `okf supersede`, `okf setup` | Setup, cache inspection, REPL, index rebuild, link/thought/soft-delete maintenance, frontmatter writes, skills and extensions |

Only commands that embed text (`search` except `--rank ppr`, `import`,
`ingest`) load the model; `read`, `traverse`, `lint`, `diff`, `doctor` and
`export` never do. Batch searches, use `okf shell` for a warm session, and
reach for `--rank ppr` for topic queries in cold sessions. With `okf serve`
running, every `okf` call reuses the server's loaded model.

Omit `--bundle-root` entirely for file-free mode: thoughts ingest, search,
read, traverse, doctor and export work from the DB alone (file-side ops fail
fast naming the missing root).

---

## HTTP and MCP

Run one owner process per database:

```bash
okf serve --listen 127.0.0.1:8765                 # TCP
okf serve --listen unix:///tmp/okfgraph.sock     # POSIX UDS
```

It exposes:

- `GET /health`, `GET /ops`, `POST /ops/<op>`
- `POST /jobs`, `GET /jobs`, `GET /jobs/<id>`, cancel
- `/mcp` with all 11 annotated MCP tools

Jobs are in-memory and non-durable, and only `reembed` is job-capable
(`JOB_OPS`). Every HTTP op, job and MCP tool shares one owner lock.

The public `OKFClient` is stdlib-only (`http://host:port` or
`unix:///absolute/path`):

```python
from okfgraph import OKFClient, discover, serve_file_for

url = discover("kb.db") or "http://127.0.0.1:8765"
client = OKFClient(url)
client.health()
client.search("retrieval", limit=5)
print(serve_file_for("kb.db"))
```

`okf-mcp` serves MCP over stdio, or proxies through a live owner:

```bash
okf-mcp                                   # stdio, direct DB
okf-mcp --connect http://127.0.0.1:8765   # proxy through owner
```

The MCP server needs a stable database: `--db-path`, `OKFGRAPH_DB_PATH` or
`[database] db_path`. It refuses to boot without one (`CONFIG_INVALID`)
because a temp dir means amnesia every session. Every tool returns the result
envelope; failures set `isError: true` with the error envelope as text.

| Tool | Routing |
|---|---|
| `search(query, target?, rank?, …)` | `concepts` = open questions; `chunks` = exact passages; `images` = assets; `rank=ppr` = model-free |
| `read(concept_id, include?, max_tokens?)` | Full shapes, or a budgeted section list |
| `traverse(start_id, relationship?, direction?, target?, …)` | Walk edges, list dirs, connect two concepts |
| `ingest(kind, …)` | `md` / `pdf` / `thoughts` / `bib` with per-kind params |
| `export_bundle(output_dir, …, flavor?)` | `okf` or `obsidian` |
| `export_concept(concept_id, output_dir, flavor?)` | Single concept |
| `list_images(concept_id)` / `get_image(asset_id)` | Asset metadata; `get_image` adds base64 `data` |
| `update(concept_id, …)` / `supersede(old_id, new_id)` / `delete(concept_id, …)` | Frontmatter writes and soft delete |

### Skills

Harness-neutral skill sources live in `src/cli/skills/` and ship in the wheel.
`okf setup skills` copies them to `.claude/skills/<name>/SKILL.md`; `--to DIR`
targets another harness and `--skill NAME` (repeatable) picks some.

- **`okfgraph-mcp`** — finding knowledge via MCP tools.
- **`okfgraph-cli`** — the same graph through `okf` commands.
- **`okfgraph-ingest`** — feeding discipline: what to store, which kind,
  converter modes, wikilink/alias conventions.

---

## Architecture

```
src/                     Rust core (all business logic)
  store/                 v10 Ladybug schema, one owned connection, write lock
  embed/                 native and external embed/rerank/vision lanes
  search/                hybrid RRF/PPR, filters, expansion, budget
  ingest/                import, delta, links, export, doctor, images
  ops/                   the public 30-op router
  cli/, server/, mcp/    feature-gated adapters (also native binaries)
  util/                  paths, hashing, SSRF, extension checks, crash reports
bindings/python/         PyO3 adapter, no business logic
python/okfgraph/         pure-Python public facade
addons/pdf/              native converter crate, bundled into the wheel
parity/                  tag-based Python 0.15 oracle and differential gates
```

Key design decisions:

- **One owner per database.** `okf serve`, stdio MCP and direct CLI use share
  one router; the discovery file `<db>.serve.json` is verified against
  `GET /health` before a borrowed owner is used.
- **Rust-only core, fail-fast.** No Python fallback and no mixed core in one
  process; a mid-run stack switch would silently mix vector spaces in one
  index.
- **Lazy sessions.** Router construction never downloads a model; the first
  encode opens the session once. PPR search, budgeted reads, diff and doctor
  stay cold.
- **Transactional ingest.** Vectors are encoded before any destructive delete,
  then one write transaction replaces the concept, chunks, links and ledger
  rows.
- **Pinned runtime.** `onnxruntime==1.31.1` shared by the core and the bundled
  converter; `ORT_DYLIB_PATH` overrides discovery.
- **Deterministic by construction.** Sorted traversal, fixed PPR constants and
  accumulation order, checked-in goldens for chunking, frontmatter, search,
  diff, doctor and export.
- **Bundle is source of truth, graph is the index.** Edit Markdown, re-import;
  verify ingests by searching, since empty `[]` proves wiring while errors
  prove broken setup.

---

## API reference (`OKFRouter` ops)

The facade delegates every op to the native core. Failures raise
`OKFError(code, message, fields?, remedy?)` with the same closed code set as
0.15; outcome errors (`DIFF_DIFFERENT`, `DOCTOR_FINDINGS`, `LINT_ERRORS`)
carry the report on `err.data`. A passed param the chosen path ignores is
refused with `BAD_VALUE` naming it.

Read-oriented (13):

| Op | Description |
|---|---|
| `search(query, target?, limit?, rank?, filters…, rerank?)` | RRF hybrid; `hub` blends authority, `ppr` is model-free, `cross` reorders by the cross-encoder; chunk collapse, expansion, budgets, `where`/tag/time/`near_id` filters |
| `read(concept_id, include?, max_tokens?)` | Body / chunks / document / context, or a budgeted section list |
| `traverse(start_id, relationship?, direction?, depth?, target?, max_path_length?, link_kinds?)` | Walk, list (empty id = root), shortest path |
| `embed(texts, task?, dim?)` | Embed with the graph's pinned encoder |
| `doctor(stale_days?, fix?, strict?, cache_dir?)` | Scored findings; `strict` raises `DOCTOR_FINDINGS` |
| `pin_info()` / `lanes()` | Pins and active embed/rerank/vision lanes |
| `list_broken_links()` / `list_deleted()` / `list_images()` / `list_directory()` | Maintenance listings |
| `get_image(asset_id)` / `get_by_id(concept_id)` | One asset (bytes + base64) or one concept |

Write/ingest (17):

| Op | Description |
|---|---|
| `ingest(kind, …)` / `import_file(path, …)` / `import_files(paths, …)` / `import_bundle(path?, …, prune_missing?, rebuild?)` | Import paths; `rebuild` = `auto\|defer\|force` |
| `sync(producer, source?, output_dir?, prefix?, rebuild?)` | Incremental producer run with a stored cursor |
| `update(concept_id, …)` / `supersede(old_id, new_id)` | Edit frontmatter, then import |
| `delete(concept_id, remove_file?)` / `recover_deleted(id)` / `purge_deleted(older_than?)` | Soft delete, 24-hour recovery, purge |
| `detach(bundle_path?, verify?, force?)` | End or refuse the mirror |
| `repair_links()` | Re-point broken links |
| `export_bundle(output_dir, …, flavor?)` / `export_concept(concept_id, output_dir, flavor?)` | OKF or Obsidian |
| `diff(old?, new?)` / `reindex(if_dirty?)` / `reembed(dry_run?)` | Drift, index rebuild, re-encode under the configured engine |

---

## Compatibility and migration

- v10 databases and bundles remain compatible. Rust opens and writes them
  without conversion.
- Older or schema-less databases are refused with `CONFIG_INVALID`. Migrate
  once with Python 0.15, then reopen with 0.16.0.
- Do not run a Python 0.15 core and the Rust core in one process. The
  `<db>.lock` protocol excludes writers across both.
- The public Python API, settings constants, models, errors, console scripts
  and module entry shims are preserved. `okfgraph.components.*` and
  `okfgraph.ops.*` are removed.
- A source build without the bundled converter returns `CONVERSION_FAILED`
  with the install hint; Python 0.15 returned a raw `INTERNAL`.
- See [`docs/migration-guide.md`](docs/migration-guide.md) before cutover.

---

## Development

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo test --workspace --all-features --lib
cargo test -p okfgraph --no-default-features --lib
cargo test -p okfgraph --no-default-features --test fuzz_robustness
cargo deny check
python bindings/python/build_wheel.py --profile dev
```

The differential matrix uses the frozen `python-oracle-0.15.0` git tag, not a
root Python package:

```bash
python parity/tools/run_matrix.py --pdf-plugin target/debug/okfgraph_pdf.dll
python parity/tools/check_coverage.py
python parity/tools/check_perf_bench.py
```

Hosted CI is a build/smoke gate. The full GPU parity suite and paired
performance benchmark are a documented manual release gate; see
[`docs/release-gate.md`](docs/release-gate.md).

---

## Project structure

```text
src/                     Rust core and native binaries
  bin/okf.rs             CLI entry point
  bin/okf-mcp.rs         MCP entry point
bindings/python/         PyO3 binding (no business logic)
python/okfgraph/         public Python facade, client, models, errors
addons/pdf/              native PDF/Office converter crate
parity/                  oracle worktree, corpus, scenarios, gate tools
docs/                    current docs, historical 0.15 records, release gates
```

---

## Requirements

- Rust 1.88+ to build from source; Python 3.12–3.14 for the wheel.
- LadybugDB `0.22.1` (statically linked; storage and extensions move with the
  minor version).
- Exactly one ORT provider: `okfgraph[cpu]` or `okfgraph[gpu]`; shared by the
  core and the bundled converter via `ORT_DYLIB_PATH`.
- No torch, transformers, optimum or Python PDF stack anywhere.

### Upgrading ladybug (0.22.1)

The first open of a graph under ladybug 0.22 rewrites its storage version
(47 → 49). **This is one-way:** a Python 0.15 install still on ladybug 0.21.2
can no longer open that file afterwards. Before upgrading, back up each graph
file (`kb.db` plus `kb.db.wal` if present) and stop every old process on that
graph.

Ladybug fetches its `vector` and `fts` extensions per version into
`~/.lbdb/extension/<version>/`. An existing graph needs them at open time, so
the first open after the upgrade installs them; offline, it fails with
`SEARCH_UNAVAILABLE` until one online open has cached them. `okf setup
extensions --from DIR` fills an offline directory.

---

## Model licences

okfgraph's **code** is Apache-2.0 OR MIT. The **model weights** it downloads
at runtime are not part of this repository and carry their own licences.
Check them before any commercial use.

| Used for | Model (Hugging Face repo) | Fetched by | Licence |
|---|---|---|---|
| Text embeddings (default) | `google/embeddinggemma-2` (`onnx-community/embeddinggemma-2-ONNX` fp32; fp16 mirror) | embroider | Apache-2.0 |
| Text embeddings (Jina v5) | `jinaai/jina-embeddings-v5-text-small-retrieval` / `-nano-retrieval` | embroider | CC BY-NC 4.0 — check before commercial use |
| Image embeddings (gemma slot) | `onnx-community/embeddinggemma-2-vision` | embroider | Apache-2.0 |
| Image embeddings (Jina slot) | `opticsWolf/jina-embeddings-v5-omni-nano-retrieval-onnx` | embroider | CC BY-NC 4.0 |
| PDF layout | `wybxc/DocLayout-YOLO-DocStructBench-onnx` | bobine | Apache-2.0 |
| PDF OCR det/rec | `SWHL/RapidOCR` (PP-OCRv4) | bobine | Apache-2.0 |
| PDF tables | `opendatalab/PDF-Extract-Kit-1.0` (`SlanetPlus`) | bobine | repo declares AGPL-3.0 |
| PDF formulas | `OleehyO/TexTeller` | bobine | Apache-2.0 |

- **Every Jina v5 model is non-commercial (CC BY-NC 4.0).** okfgraph cannot
  embed anything without a model, so commercial use requires a commercial
  licence from Jina AI (see the model cards).
- **PDF tables:** bobine fetches SLANet-plus from a repository whose card
  declares AGPL-3.0. Weights are downloaded, never redistributed; review
  before redistributing a service built on the PDF converter.
- Licences are as declared in each repository's model card when checked. The
  model cards are authoritative and can change. This table is a pointer, not
  legal advice.

---

## License

The code is licensed Apache-2.0 OR MIT (see [`LICENSE`](LICENSE),
[`LICENSES`](LICENSES), and [`docs/license-review.md`](docs/license-review.md)).
Downloaded model weights are licensed separately; see
[Model licences](#model-licences).
