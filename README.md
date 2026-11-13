# OKFgraph

[![PyPI](https://img.shields.io/pypi/v/okfgraph)](https://pypi.org/project/okfgraph/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-green)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-%E2%89%A52.0-purple)](https://modelcontextprotocol.io/)
[![Ladybug](https://img.shields.io/badge/ladybug-0.22.1-orange)](https://pypi.org/project/ladybug/)
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
| **Embeddings** | Jina v5 (default `…-text-small-retrieval`, `--model-id` selects registry: text-nano) via the `embroider` Rust wheel; last-token pooling, Matryoshka truncation (default 512, per-model ladder); device `auto` (CUDA→FP16 mirror weights, CPU→FP32, `--precision` pins incl. explicit `int8`); CPU arena off by default (`--cpu-arena`); token ceiling configurable (`--max-length`); model, precision and vector space (0.13: embroider's `space` id) pinned per graph, never mixed. No torch anywhere (0.7.0+) |
| **Search** | Hybrid RRF fusion (vector + FTS) at concept and chunk granularity; `rank=none\|hub\|ppr` — including **PPR**: lexical seeds → exact Personalized PageRank, zero model load, deterministic |
| **Read** | Body / chunks / rebuilt document / graph context, with optional **token budgets** (`max_tokens`): self first, then PPR-ranked neighbours, index-first for context |
| **Storage** | LadybugDB `==0.22.1` (pinned — storage format and extensions move with each minor; see [Upgrading ladybug](#upgrading-ladybug-0221)): graph + vector + FTS in one file |
| **Links** | Path links (`](doc.md)`) + `[[wikilinks]]` resolved by name (`uid` → `aliases` → `title` → filename stem); ambiguous names never resolve; broken links tracked and repairable |
| **Import** | Single files, whole bundles (delta-aware: only changed files re-embed, `--prune-missing` drops deleted concepts), markdown / PDF+Office / raw thoughts; mordant lint on the way in; `okf produce` generates bundles from SQLite (`## Observations` + `log.md` changelog) |
| **PDF** | `DocumentConverter` seam with `BobineConverter` default (routing `auto\|surgical\|always\|never`); bring your own converter, no code changes |
| **Diff** | `okf diff`: structural snapshot (dir vs dir, no model load) and drift (graph vs dir) modes; CI exit codes + `--json` |
| **Doctor** | `okf doctor`: 0–100 health score (broken/orphan/stale/duplicate-title/missing-description), safe `--fix` that never touches `reviewed: true`, `--strict` CI gate |
| **Export** | OKF round-trip with See Also / Cited By enrichment + index files, or `--flavor obsidian` (`[[Title]]` wikilinks, no index files, edge-lossless re-import) |
| **Images** | Caption-based (`mode=text`) or image-content (`mode=optional`/`omni` via the ONNX vision model — the registry vision pairing of the graph's text model: the gemma-2 vision slot on gemma-2 graphs, the jina omni slot on text-nano graphs; needs embroider>=0.14; since 0.14 the vision model may also run in an `embroider.serve` host, so no ONNX loads here) content-hash dedup, `okf-asset://` protocol |
| **MCP** | 11 tools (`search`, `read`, `traverse`, `ingest`, `export_bundle`, `export_concept`, `list_images`, `get_image`, and since 0.11 `update`, `supersede`, `delete`), MCP ≥ 2.0 (`MCPServer` + `ToolAnnotations`), stdio transport — or one shared owner via `okf serve` + `okf-mcp --connect URL`; success result = envelope, failures = `isError: true` + error envelope (D7) |
| **CLI** | Same 8 core verbs (`search`, `read`, `traverse`, `ingest`, `import`, `export`, `diff`, `doctor`) plus maintenance (`shell`, `lint`, `produce`, `reindex`, `broken-links`, `deleted-*`, `delete`, `images`, `image`, …), global `--json` (D5), slim per-command help, `okfgraph.toml` config |
| **Skills** | 3 harness-neutral skills (`okfgraph-mcp`, `okfgraph-cli`, `okfgraph-ingest`), `.mcp.json` wiring included |

---

## Installation

Requires Python ≥ 3.11. The extras decide what you get:

| Extra | Installs | Gives you |
|---|---|---|
| `cpu` **or** `gpu` (pick exactly one) | `onnxruntime==1.31.1` / `onnxruntime-gpu[cuda,cudnn]==1.31.1` | the ONNX Runtime that embedding, search and rerank run on |
| `pdf` | `bobine>=0.8.1` | `okf ingest --kind pdf`: PDF and Office files (docx/xlsx/pptx, legacy doc/xls/ppt) converted to markdown, then imported |
| `dev` | `pytest` | the test suite |

**Why the runtime isn't installed by default.** `onnxruntime` and
`onnxruntime-gpu` are two distributions of the same `onnxruntime`
module, and pip can't pick one by the hardware or install both side by
side. So okfgraph names neither in its core dependencies: a bare
`pip install okfgraph` installs, but the first embedding fails with
`NO_ORT_RUNTIME` and the install line to run.

**How bobine comes in.** bobine is a plugin, not a dependency: only the
`pdf` extra installs it, and okfgraph imports it only when it converts a
document. Without it, Markdown, thoughts and BibTeX work as usual and
`ingest --kind pdf` fails asking for `pip install "okfgraph[pdf]"`. bobine uses
the same ONNX Runtime that `cpu`/`gpu` installed (0.8.1 pins the same
1.31.1), so it needs no runtime of its own.

### CPU or GPU?

| | `[gpu]` | `[cpu]` |
|---|---|---|
| **Pick it when** | the machine has an NVIDIA GPU (`nvidia-smi` prints it) | no NVIDIA GPU, macOS, CI, or a small install |
| Runtime | `onnxruntime-gpu[cuda,cudnn]==1.31.1` (bundles the CUDA + cuDNN runtime wheels: a large download) | `onnxruntime==1.31.1` (small) |
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

> **Windows GPU note:** on the `[gpu]` wheel, `--device auto` can stall
> at model load while probing CUDA. Pin the device explicitly —
> `--device cuda` (GPU) or `--device cpu` — which opens a session
> normally (`OKFGRAPH_DEVICE=cuda` works the same). Recursive probes
> also refuse stale `onnxruntime.dll` copies in `C:\Windows\System32`:
> if a direct probe aborts with a version mismatch, okfgraph's own
> resolver already pins the wheel's DLL.

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
Skip both extras and point `ORT_DYLIB_PATH` at your onnxruntime 1.31
library instead — but never rely on bare OS-loader discovery: a stale
system DLL (e.g. in `System32`) fails the load instead of being used.

Core dependencies are deliberately few: `ladybug==0.22.1`, `embroider>=0.15,<0.16`
(the shared embedding engine — github.com/opticsWolf/embroider, also used by
bobine), `mordant`, `mcp>=2.0`, `pydantic`, `pyyaml`, `numpy`,
`python-frontmatter`, `fasteners`, `Pillow`. The ONNX Runtime comes from
the `cpu`/`gpu` extra (one pinned 1.31.1 binary shared by bobine and
embroider; `ORT_DYLIB_PATH`-overridable). There is no torch, no
transformers, no optimum, no PDF stack in the core install.

After installing, copy the agent skills into your harness:
`okf setup skills` writes them to `.claude/skills/` (Claude Code);
`--to DIR` for another harness, `--skill NAME` for one of them.

### Models in another process

Each model can run in its own server instead of inside okfgraph:
`[embedding] backend = "external"` (text), `[search] rerank_backend =
"external"` (reranker) and, since 0.14, `[embedding] image_backend =
"external"` (vision), each with its endpoint URL (`endpoint_url`,
`rerank_endpoint_url`, `image_endpoint_url`). An embroider host serves
any of them:

```bash
python -m embroider.serve --kind vision --model <image model id> --dim <graph dim> --port 8091
```

```toml
[embedding]
image_backend = "external"
image_endpoint_url = "http://127.0.0.1:8091"
```

The image lane has no declared mode: the host must be embroider and
report its model, width, precision, vector space and text partner on
`GET /v1/models`; okfgraph refuses a host whose model, width or text
partner doesn't match the graph. okfgraph sends the original image bytes
and the host resizes them, so vectors match the in-process lane and a
graph can switch lanes without re-embedding. With all three lanes
external, neither ONNX Runtime nor the `embroider` wheel needs to load in
the okfgraph process. An unreachable host is `EXTERNAL_UNAVAILABLE`.

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
| `bib` | `router.ingest("bib", bib_path=…, tags?) -> {"concept_ids","entry_count","skipped_entries",…}` | one `reference` concept per `@entry` as `refs/<key>` (stable IDs — re-ingesting upserts; slug collisions disambiguated, never silently overwritten); malformed entries reported per-entry in `skipped_entries`, never fatal |

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
`--max-length`, …) are documented once in `okf --help`, accepted before or
after the command (`okf --db-path kb.db search x` = `okf search x --db-path
kb.db`), and usually live in `okfgraph.toml`. Where to look:

| What | Set by | Default |
|---|---|---|
| database | `--db-path`, `OKFGRAPH_DB_PATH`, `[database] db_path` | `.okfgraph/okfgraph.db` in the current directory (okf-mcp: must be set) |
| bundle root | `--bundle-root`, `OKFGRAPH_BUNDLE_ROOT`, top-level `bundle_root` | `bundle/` beside a db in a `.okfgraph` folder; else `okf` uses the current directory, `okf-mcp`/`okf serve` the db's folder |
| config file | — | `okfgraph.toml` in the current directory, else in the bundle root, else `~/.config/okfgraph/config.toml` (first found wins) |

Precedence is flag > environment > `okfgraph.toml` > default. Relative
paths count from the current directory, in `okfgraph.toml` too (except
`[[roots]]`, which count from the file). Only `okf init`, `okf import`
and `okf ingest` create a missing database, folders included; every other
command refuses it with `FILE_NOT_FOUND`, so a mistyped path never leaves
an empty graph behind (0.13.1). `okf-mcp` and `okf serve` create the
database on first boot from the path you set. Every command accepts `--json` (D5):
human renderer by default, the result envelope
(`{ok, op, data, warnings, error}`) on stdout with `--json`. Results always
go to stdout and logs to stderr (`-q` silences logs, never results).
Failures print `[ERROR] CODE: message (remedy)` on stderr (plus the error
envelope on stdout under `--json`) and exit 0 ok / 1 state·outcome / 2
usage. Outcomes (`lint` errors, `diff` differences, `doctor --strict`
findings) still render their report before exiting 1. Typed state codes
include `DB_LOCKED` (the db is open in another process — `okf` commands
run one at a time per database; when `okf-mcp` is serving, query through
it), `NO_ORT_RUNTIME` (no/broken ONNX Runtime), `EXTERNAL_UNAVAILABLE`
(an external embedding/rerank/vision host doesn't answer; 0.14),
`CONVERSION_FAILED` (bobine could not convert that file; 0.14), and the
pin set.
Concurrency: one process per db file; the 0-byte `<db>.lock` beside it is
ladybug's own lock marker and is harmless when no process holds it.
A leftover after a crash/kill is indistinguishable from a live hold at first
glance — check processes first, then remove by hand if stale (`okf doctor`
reports it; the CLI never auto-deletes). Without any configuration a
graph lives in `.okfgraph/` under the CWD: the database at
`.okfgraph/okfgraph.db` and its bundle tree at `.okfgraph/bundle`, both
created on first use (0.12.0). A CWD that still holds a pre-0.12.0
`okfgraph.db` (and no `.okfgraph/`) keeps using it, bundle root `.`,
with one warning naming the new default; nothing is ever moved. Any db
inside a `.okfgraph` folder gets that folder's `bundle/` as its default
bundle root, on every surface. Pin the database in `okfgraph.toml`
(`db_path = "kb.db"`, overridable per-invocation with `OKFGRAPH_DB_PATH`)
so a held-open default can never silently fork a second database.
Unexpected failures (`INTERNAL`) also write the full traceback — with argv,
versions and platform — to a crash file (`%LOCALAPPDATA%\okfgraph\crash\`,
else `$XDG_STATE_HOME`/`~/.local/state`; `OKF_CRASH_DIR` overrides, newest
20 kept). The final `[ERROR]` line names the file, so it survives `tail`,
and `--json` carries it in `error.fields.crash_report`. Start-up failures
(a broken install, a native DLL, a half-edited checkout) are covered too:
the `okf` launcher enters through `okfgraph._entry`, which guards the
package import itself.

| Command | Description |
|---|---|
| `okf search QUERY [--target concepts\|chunks\|images] [--rank none\|hub\|ppr] [--expand] [--hub-rerank]` | Concepts (RRF hybrid), chunks (passages), images; `--rank ppr` is model-free |
| `okf read ID [--include body\|chunks\|document\|context] [--max-tokens N]` | Full shapes uncapped; budgeted section list with `--max-tokens` |
| `okf traverse [ID] [--relationship CONTAINS\|LINKS_TO\|PART_OF\|INCLUDES_ASSET] [--direction …] [--target ID]` | Relationships, directory listing (empty ID = root), shortest path via `--target` |
| `okf ingest --kind md\|pdf\|thoughts\|bib …` | `--md-path`, `--pdf-path FILE...` (several files in one process on one loaded model; a failed file is reported and the rest continue; `--routing-mode`, auto-imports by default; `--no-auto-import` converts only), `--thoughts --topic`, `--bib-path` (one `refs/<key>` concept per BibTeX entry) |
| `okf export --all\|--concept-id ID --output-dir DIR [--flavor okf\|obsidian]` | Bundle or single-concept export; obsidian = `[[Title]]` links, no index files |
| `okf images CONCEPT_ID` / `okf image ASSET_ID [--output-path F]` | Image assets on a concept; metadata (+ bytes) per asset |
| `okf diff [OLD] [NEW] [--json]` | Snapshot (two dirs, no model) or drift (graph vs dir); exit 0 identical / 1 different |
| `okf lint [DIR] [--json]` | Pre-import gate (no DB, no model); exit 0 clean / 1 errors / 2 bad dir |
| `okf doctor [--fix] [--strict] [--stale-days N] [--json]` | Score + findings; `--fix` repairs safely, `--strict` exits 1 on any finding |
| `okf import [FILE...] [--all] [--prune-missing] [--mode text] [--force]` | Single files / bulk import, delta-aware (`--force` re-attaches a detached graph); TOML `[[roots]]` (or per-call `--root ALIAS=PATH`) add named roots (`@alias/` IDs, unmounted ≠ deleted, `--prune-missing` skips while any root is absent); `--bundle-path DIR` pins one tree for this call |
| `okf produce --from sqlite --source DB --output-dir DIR` | Generate a bundle from a data source (one concept per table, FK links, `## Observations` notes, `log.md` changelog); lint pre-flighted, CLI-only |
| `okf detach [--bundle-path DIR] [--no-verify] [--force]` | End the mirror: the DB becomes the artifact (verify-first; imports refuse without `--force`) |
| `okf init`, `okf model-info`, `okf shell`, `okf reindex`, `okf reembed [--dry-run]`, `okf broken-links`, `okf repair-links`, `okf retitle-thoughts [--topic T] [--apply]`, `okf deleted-*`, `okf delete ID` | Setup, cache inspection, REPL, index rebuild, 0.12 embedder remodel (`reembed` needs `OKFGRAPH_ALLOW_REPIN=1` across a pin change), link + thought-title + soft-delete maintenance (`delete` refuses file-backed concepts while their source exists) |

Only commands that embed text (`search` except `--rank ppr`, `import`,
`ingest`) load the model (~30s cold per process); `read`, `traverse`,
`lint`, `diff`, `doctor` and `export` never do — batch searches, use
`okf shell` for a warm session, and reach for `--rank ppr` for topic
queries in cold sessions. Bulk input: many Markdown files go through
`okf import --all` (or `okf import FILE...`) — `.md`, `.markdown` and
`.txt`, one model load, batched writes. PDF and Office files go through
`okf ingest --kind pdf --pdf-path A.pdf B.docx …`: one model load for the
batch, but bobine opens its converter per file. With `okf serve` running,
every `okf` call reuses the server's loaded model. Omit `--bundle-root` entirely for file-free mode: thoughts ingest,
search, read, traverse, doctor, and export work from the DB alone (file-side
ops fail fast naming the missing root). Thought IDs are namespaced
(`thoughts/<topic>/<ts>_<id>`), so a fileless graph exports into a tidy tree.

---

## MCP server

11 tools, MCP ≥ 2.0, stdio transport. Wire once per project with a **stable**
database — `--db-path`, `OKFGRAPH_DB_PATH` or `[database] db_path` in
`okfgraph.toml`; the server refuses to boot without one (`CONFIG_INVALID`,
exit 2) since a temp dir means amnesia every session. `--bundle-root`
defaults to the `bundle/` beside a database inside a `.okfgraph` folder
(`<root>/.okfgraph/okfgraph.db` → `<root>/.okfgraph/bundle`), else to the
database's parent directory. First
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
| `ingest(kind, …)` | `md` / `pdf` / `thoughts` / `bib` with per-kind params |
| `export_bundle(output_dir, …, flavor?)` | `okf` or `obsidian` |
| `export_concept(concept_id, output_dir, flavor?)` | Single concept → `<output_dir>/<id>.md` |
| `list_images(concept_id)` | Image assets on one concept (metadata) |
| `get_image(asset_id)` | One asset: metadata + base64 `data` |

Verify wiring with any `search` — an empty graph returns `[]`, which still
proves the plumbing works.

### Skills

Harness-neutral skill sources live in `okfgraph/skills/` and ship in the
wheel (0.13.1). `okf setup skills` copies them to `.claude/skills/<name>/SKILL.md`,
where Claude Code loads them; `--to DIR` targets another harness's folder
(see `docs/harness-integration.md`) and `--skill NAME` (repeatable) picks
some. Rerun it after an upgrade: changed files are updated, the rest left alone.

- **`okfgraph-mcp`** — finding knowledge via MCP tools (pick this *or* CLI).
- **`okfgraph-cli`** — same graph through `okf` shell commands.
- **`okfgraph-ingest`** — feeding discipline: what to store, which kind,
  converter modes, wikilink/alias conventions. Install alongside either
  finder when the agent should persist knowledge. It is separate so the
  rules are written once for both finders, and so a read-only agent can
  get a finder without it.

---

## Architecture

```
                ┌──────────── MCP (11 tools) / CLI (8 verbs + maintenance)
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
- **Single pinned ORT** (`onnxruntime==1.31.1`, `ORT_DYLIB_PATH`-overridable; embroider also self-serves the CUDA loader dirs since its 0.14 wheel)
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

One vocabulary of operations on the facade (mixins `QueryOps`,
`IngestOps`, `ExportOps`, `AdminOps`, and since 0.11 `WriteOps`); every failure raises
`OKFError(code, message, fields?, remedy?)` — `UsageError` (exit 2) for
caller mistakes, `StateError`/`OutcomeError` (exit 1) for graph state and
domain outcomes (the class always follows the code). Outcome errors
(`DIFF_DIFFERENT`, `DOCTOR_FINDINGS`, `LINT_ERRORS`) carry the full result
on `err.data`. A passed param the chosen path ignores is refused with
`BAD_VALUE` naming it in `fields["ignored"]`.

| Op | Description |
|---|---|
| `search(query, target="concepts", limit?, concept_type?, tags?, parent_id?, include_chunks?, max_chunks_per_doc?, expand?, context_hops?, hub_rerank?, hub_weight?, rank="none"\|"hub"\|"ppr"\|"cross", where?, since?, until?, exclude_tags?, link_kinds?, near_id?, hops=2, offset=0, recency?, recency_weight=0.3, rerank=False, efs?, dedupe=False)` | RRF hybrid; `hub` blends authority, `ppr` is model-free, `cross` reorders the top hits by a cross-encoder; chunk paths: `hub_rerank` > `expand` > plain, `rerank` (plain or `hub_rerank`) reorders by the cross-encoder, adding `cross_score` (opt-in; `[search] rerank_*` settings). `efs` (16–1000) widens the vector index search per call (`[search] efs` default). `dedupe` (chunks) drops passages that repeat a higher-ranked hit (cosine >= 0.97), listing them in `collapsed`. 0.11: `where` = equality on extra keys, `since`/`until` on `effective_at` (authored date, else last content change), `exclude_tags` (default from `[search] exclude_tags`), `link_kinds` for `hub`/`ppr`; `near_id` + `hops` (1–3) = only concepts within that many links, scored exactly; `offset` pages (exact within the first 25 results); `recency` = half-life in days, blends recency into the order (`recency_weight` share) |
| `read(concept_id, include="body"\|"chunks"\|"document"\|"context", max_tokens?)` | Concept dict / chunk dicts (no embeddings) / `{"concept_id","markdown"}` / `{incoming_links, outgoing_links, ancestry, siblings}`; or a budgeted `{sections, used, budget, truncated}` |
| `traverse(start_id, relationship?, direction?, depth?, node_type?, target?, max_path_length?, link_kinds?)` | Walk, list (empty id = root), connect; `link_kinds` (0.11) follows only those `LINKS_TO` kinds (`mention`, `supersedes`, …) |
| `ingest(kind, …, type?, extra?, materialize?, rebuild?, related?, folder?)` | `md` / `pdf` / `thoughts` / `bib`, per-kind params, lint + chunk + embed + link; 0.11: an explicit `concept_id` upserts, `materialize` writes `<concept_id>.md` into the bundle and imports it; thoughts: `related` = typed links, `folder` = write `<folder>/<title>.md` and import it |
| `import_file(path, mode?, rebuild?) {"concept_id","images"}` / `import_files(paths, rebuild?) {"concept_ids","images"}` / `import_bundle(path?, …, prune_missing?, rebuild?) {"concept_ids","images","duplicates","moved"}` | Single / list (one index step) / whole-bundle delta import (per-file since 0.11); `rebuild` = `auto`\|`defer`\|`force` |
| `update(concept_id, title?, description?, type?, tags?\|add_tags?\|remove_tags?, extra?, relations?)` / `supersede(old_id, new_id)` | 0.11 write ops: edit the bundle file's frontmatter, then import it |
| `sync(producer, source?, output_dir?, prefix?, rebuild?)` / `setup_extensions(target?, source?, download?)` (static) | 0.11: incremental producer run with a stored cursor; fill an offline extension directory |
| `export_bundle(output_dir, directory_id?, concept_type?, tags?, flavor?) {"concept_ids",…}` / `export_concept(concept_id, output_dir, flavor?) {"path"}` | Both flavors |
| `list_images(concept_id)` / `get_image(asset_id)` | Image ops (metadata; bytes + base64) |
| `init() {"db_path","embedding_dim","model_id"}` / `model_info(…)` (static) / `reindex(if_dirty?)` / `reembed(dry_run?)` | Schema, cache inspection, index rebuild; 0.12: `reembed` re-encodes every Concept/Chunk under the **configured** embed engine and re-pins (`dry_run=True` = the plan; refuses dim migrations and multimodal graphs — a pinned graph opened under a new model needs `OKFGRAPH_ALLOW_REPIN=1` for that boot) |
| `embed(texts, task?, dim?)` (0.12, CLI/HTTP) | Embed arbitrary texts with the graph's **own pinned encoder** (crate or external): `{vectors, dim, model_id, task, renormalised, honours_dimensions, counter}`; per-call `dim` slices to a Matryoshka step and renormalises; caps 256 texts / 1 MiB |
| `doctor(stale_days?, fix?, strict?) {"report","fixed"}` / `lint(dir)` (static) / `produce(source_type, source_path?, output_dir?, prefix?, overwrite?)` (static) / `diff(old?, new?)` / `diff_dirs(old, new)` (static) | Health, pre-import gate, bundles from sources, structural diff; `strict` findings, lint errors and differences raise outcome errors with the result on `err.data` |
| `list_deleted()` / `recover_deleted(id)` / `purge_deleted(older_than?)` / `delete(concept_id, remove_file?)` / `detach(bundle_path?, verify?, force?)` | Soft-delete trio + `delete` (refuses file-backed concepts whose source exists unless `remove_file=True`, which moves the file to `.okf-trash/`; `recover_deleted` moves it back) + mirror lifecycle |
| `list_broken_links() [{source, target, kind, reason, candidates, raw, id}]` / `repair_links() {"repaired"}` | Unresolved refs (`reason` `missing`\|`ambiguous`); exact-id, path and unique-name repair — also run by every import since 0.11 |

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
│   ├── cli.py             # okf: query verbs + maintenance + images/image, --json (D5/D6)
│   ├── mcp_server.py      # okf-mcp: 11 tools, MCP ≥ 2.0, lifespan-managed router, envelope/_ToolFailure
│   ├── images.py          # IngestMode (text + vision-onnx since 0.8.0), planning helpers
│   ├── security.py        # SSRF/domain guards for remote images
│   ├── skills/            # okfgraph-mcp, okfgraph-cli, okfgraph-ingest (shipped; okf setup skills)
│   └── components/        # ranking, links, search, lint, import_, export, diff, doctor,
│                          # embedding, converters, image_assets, delta, purge, schema, ingest, skills
├── tests/fixtures/        # conformance corpus: ppr, diff, doctor, obsidian, bundles
├── docs/                  # architecture spec (v6.5), converters, chunking, harness guides; archive/ holds completed plans
├── .mcp.json              # ready MCP wiring (uv run --project . okf-mcp)
└── pyproject.toml         # slim core deps + pdf/dev extras
```

---

## Requirements

Core (`uv sync`): `ladybug==0.22.1`, `embroider>=0.15,<0.16` (PyPI wheels,
github.com/opticsWolf/embroider), `mordant>=0.9`,
`mcp>=2.0`, `pydantic>=2`, `pyyaml>=6`,
`numpy>=1.26`, `python-frontmatter>=1`, `fasteners>=0.19`. Python ≥ 3.11.
Plus exactly one ORT provider: `okfgraph[cpu]` (`onnxruntime==1.31.1`)
or `okfgraph[gpu]` (`onnxruntime-gpu[cuda,cudnn]==1.31.1`) — shared with
bobine via `ORT_DYLIB_PATH`. embroider finds the pip-installed runtime
itself (`locate_runtime()`, 0.15) and self-serves the CUDA EP's loader
dirs.

- `--extra pdf`: `bobine>=0.8.1` (default PDF and Office converter; 0.8.1 pinned in the lock).
- `--extra dev`: `pytest>=8`.

### Upgrading ladybug (0.22.1)

The first open of a graph under ladybug 0.22 rewrites its storage
version (47 → 49). **This is one-way:** an okfgraph still on ladybug
0.21.2 can no longer open that file afterwards ("Database file version:
49, Current build storage version: 47") — even if the 0.22 process only
read from it. Before upgrading:

1. Back up each graph file (`kb.db`, plus `kb.db.wal` if present).
2. Stop every `okf-mcp` server and `okf` process running from an older
   install on the same graph, and upgrade them together.

ladybug fetches its `vector` and `fts` extensions per version into
`~/.lbdb/extension/<version>/` (~32 MB from `extension.ladybugdb.com`).
An existing graph needs them at open time, so the first open after the
upgrade installs them and retries; offline, it fails with
`SEARCH_UNAVAILABLE` until one online open has cached them.

---

## Model licences

okfgraph's **code** is Apache-2.0 OR MIT. The **model weights** it downloads
at runtime are not part of this repository and carry their own licences.
Check them before any commercial use.

| Used for | Model (Hugging Face repo) | Fetched by | Licence |
|---|---|---|---|
| Text embeddings (default, 0.12.0) | `google/embeddinggemma-2` (`onnx-community/embeddinggemma-2-ONNX` fp32; `opticsWolf/embeddinggemma-2-onnx-fp16` CUDA mirror) | embroider | Apache-2.0 |
| Text embeddings (0.11-era default, fp32) | `jinaai/jina-embeddings-v5-text-small-retrieval` | embroider | CC BY-NC 4.0 — check them before any commercial use |
| Text embeddings (fp16) | `opticsWolf/jina-embeddings-v5-text-small-retrieval-onnx-fp16` (ONNX conversion of the above) | embroider | CC BY-NC 4.0 |
| Text embeddings (nano) | `jinaai/jina-embeddings-v5-text-nano-retrieval` | embroider | CC BY-NC 4.0 |
| Image embeddings (≤0.6.x `--extra omni`, removed 0.7.0) | `jinaai/jina-embeddings-v5-omni-small-retrieval` | sentence-transformers (torch path, removed) | CC BY-NC 4.0 |
| Image embeddings (vision gemma slot, 0.12.0+) | `onnx-community/embeddinggemma-2-vision` (Apache-2.0, fp32 in-repo + fp16 mirror; image vectors share the gemma-2 text space) | embroider (`GemmaVision`) | Apache-2.0 |
| Image embeddings (vision jina slot, 0.8.0+) | `opticsWolf/jina-embeddings-v5-omni-nano-retrieval-onnx` (dynamic-grid ONNX export of `jinaai/jina-embeddings-v5-omni-nano-retrieval`) | embroider (`JinaV5Vision`) | CC BY-NC 4.0 |
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
