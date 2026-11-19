# OKFgraph Rust core + PyO3 binding — analysis and port plan

**Status:** M1 read-only slice in progress — Windows spikes green, `pycompat`
green, 52-setting table/errors/models/store/ops/CLI/PyO3 facade landed;
14-op Python↔Rust read parity is green. Linux/macOS spikes and write/ingest
stages remain.  
**Target:** okfgraph 0.16.0 — Rust core is the only implementation; Python is a thin
binding/facade  
**Oracle:** the current Python tree at tag `python-oracle-0.15.0` (24,242 LOC,
164 modules, 1,241 tests)  
**Baseline schema:** v10 (17 tables: 13 node + 4 relationship; 5 indexes)

**Locked decisions (2026-11-18):**

| # | Decision |
|---|---|
| a | Python compatibility is **public API only**: `OKFRouter`, `OKFClient`, `models`, `errors`, `settings` constants, `okf`/`okf-mcp`, module entry shims. `okfgraph.components.*` and `okfgraph.ops.*` are not preserved. |
| b | **Legacy v10 databases and bundles must keep working**: Rust opens/migrates/writes them, and Python 0.15 can reopen Rust-written databases until cutover. |
| c | **Slim wheel + PDF add-on.** The default wheel excludes `bobine`; PDF/Office conversion is an optional native add-on (or a Cargo feature for Rust users). |
| d | **Targets:** Windows x64, Linux x86_64, macOS arm64; Python 3.12, 3.13, 3.14. No Python 3.11, no Linux aarch64, no macOS x86_64 in the first release. |
| e | **Staged delivery:** (A) read-only Rust core on existing graphs, (B) write/ingest path + beta wheel, (C) full surfaces + stable cutover. |
| f | **One `okfgraph` crate**, Rust core under `src/`; Python facade under `python/`. The PyO3 binding is a separate layer (no logic); the PDF add-on is the only optional Rust build target. |
| g | **Stage A allows schema migrations** (schema writes, `okf init`) but no concept/content/import writes. |
| h | **PDF add-on shares the `cpu`/`gpu` ONNX Runtime.** `okfgraph[pdf]` alone installs the plugin but not ORT; document `okfgraph[cpu,pdf]` / `okfgraph[gpu,pdf]`. |
| i | **Missing/incompatible PDF add-on returns `CONVERSION_FAILED`** with an install hint. This intentionally replaces today's `INTERNAL`/crash-report path and is recorded as a 0.16 behavior change. |

---

## 1. Executive summary

The port is feasible without re-implementing any of the hard native engines. Every
heavy dependency already has a first-class Rust crate:

| Capability | Today (Python) | Rust path | Fit |
|---|---|---|---|
| Graph + vector + FTS | `ladybug==0.22.1` wheel | official `lbug` crate, same C++ core | high |
| Text/vision embeddings, rerank | `embroider>=0.15.1` wheel | `embroider` crate, same Rust core | high |
| PDF/Office → Markdown | `bobine>=0.9` wheel | `bobine` crate, same Rust core | high |
| CommonMark/GFM + chunker | `mordant>=0.10.1` wheel | `mordant` crate, same Rust core | high |
| MCP server/client | Python `mcp>=2.0` | `rmcp` 3.11 | medium |
| HTTP serve | Starlette | `axum`/`tokio` | medium |
| CLI | `argparse` | `clap` (built from the settings table) | medium |
| Python API | Pydantic models | PyO3 0.30 + pure-Python Pydantic shims | medium |

The real work is **behavioral parity**, not algorithms:

- 24k LOC of Python semantics, including many documented edge cases (delta
  detection, link repair, identity conflicts, pin refusals, ingest materialization).
- Python-specific serialization that is part of the on-disk contract:
  `json.dumps(..., ensure_ascii=True)` inside `extra`, `sort_keys=True` inside
  `content_sig`, PyYAML 1.1 scalar resolution, naive-UTC datetime formatting.
- A 17-table schema that must stay readable by the Python 0.15 oracle during the
  transition (and by Rust after cutover).
- 1,241 tests, many of which encode behavior rather than structure.

**Recommended shape**

1. **One core crate:** the repository root is the `okfgraph` package; all core
   Rust lives under `src/`. CLI/MCP/server modules are feature-gated in the same
   crate and exposed as `src/bin/okf.rs` and `src/bin/okf-mcp.rs`.
2. **One binding layer:** a tiny `bindings/python` crate builds the PyO3 `cdylib`
   `okfgraph._native`. It contains **no business logic** — argument marshalling,
   GIL release, error translation, JSON ↔ Python. Keeping it a separate Cargo
   target avoids forcing `cdylib` linking and PyO3 onto every library build.
3. **One optional add-on:** `addons/pdf` builds the native PDF/Office plugin
   (bobine) installed by `pip install okfgraph[pdf]`. Rust users can instead build
   the core with `--features pdf`.
4. The Python package becomes a compatibility facade under `python/okfgraph/`:
   Pydantic models, exception classes, `OKFRouter`/`OKFClient` signatures, and
   console-script shims.
5. Build the Rust tree alongside the untouched Python tree on a long-lived branch;
   compare against the Python oracle continuously; flip the package only after the
   parity gates pass. Never run a mixed Python/Rust core in one process.

**Effort:** ~19–23 engineer-weeks for full parity with one engineer; ~11–13 weeks
with two engineers working the independent tracks (store/ingest vs.
embed/search/surfaces). A useful read-only Rust binary should exist in ~3 weeks;
a fully replaceable package in ~4–5 months.

**Go/no-go before committing:** the Phase 0 spike must prove on Windows x64 and
Linux x86_64 that `lbug` 0.22.1 can open a Python-created v10 database, load the
`vector` and `fts` extensions, read/write, and be reopened by Python. If that
fails, the fallback is a narrow `lbug-sys`/C-API shim behind the same `store`
trait — not a change to the rest of this plan.

---

## 2. What exists today

### 2.1 Size and shape

| Area | Files / LOC | Notes |
|---|---|---|
| CLI / MCP / serve / client / router | 5 / 4,540 | 29 CLI commands, 11 MCP tools, 30 client ops |
| Ops vocabulary | 5 / 2,170 | `search`, `read`, `traverse`, `ingest`, `export_*`, import/admin/write |
| Ingest / import / links / delta | 13 / ~9,400 | largest and subtlest subsystem (`import_.py` alone 2,463) |
| Embed / search / rerank | 7 / ~5,400 | hybrid RRF, chunk search, hub/PPR, cross-encoder, external lanes |
| Core / schema / settings / security | 9 / ~2,850 | 52 settings, schema v10, migrations v1→v10 |
| **Total** | **164 / 24,242** | 107 test files, 1,241 tests |

### 2.2 Public surfaces that must keep working

**Python (documented, used by the README and harnesses):**

- `from okfgraph import OKFRouter`
- `from okfgraph.errors import OKFError, UsageError, StateError, OutcomeError, envelope`
- `from okfgraph.models import ConceptModel, ChunkModel, ImageAssetModel`
- `from okfgraph.client import OKFClient, OPS, discover, serve_file_for`
- `import okfgraph.settings` (constants and the settings table)
- console scripts `okf`, `okf-mcp`
- `python -m okfgraph.cli`, `python -m okfgraph.mcp_server`
- context-manager and `close()` behavior on `OKFRouter`

**CLI:** 29 commands — `init`, `model-info`, `import`, `ingest`, `search`, `read`,
`traverse`, `export`, `images`, `image`, `shell`, `broken-links`, `repair-links`,
`retitle-thoughts`, `diff`, `doctor`, `lint`, `produce`, `reindex`, `reembed`,
`deleted-list`, `deleted-recover`, `delete`, `deleted-purge`, `detach`, `update`,
`supersede`, `serve`, `setup`.

**MCP:** 11 tools — `search`, `read`, `traverse`, `ingest`, `export_bundle`,
`export_concept`, `list_images`, `get_image`, `update`, `supersede`, `delete`.

**Ops (30):** `search`, `read`, `traverse`, `embed`, `ingest`, `export_bundle`,
`export_concept`, `import_bundle`, `import_file`, `import_files`, `sync`, `update`,
`supersede`, `delete`, `doctor`, `diff`, `reindex`, `reembed`, `pin_info`, `lanes`,
`list_broken_links`, `repair_links`, `list_deleted`, `recover_deleted`,
`purge_deleted`, `detach`, `list_images`, `get_image`, `get_by_id`,
`list_directory`.

### 2.3 On-disk contract

- **Schema v10**, `Meta.schema_version = 10`.
- **Node tables:** `Concept`, `Directory`, `ImageAsset`, `Chunk`, `BrokenLink`,
  `DeletedConcept`, `Meta`, `MetaText`, `FileHash`, `DirHash`, `DeletedPath`,
  `SourceRoot`, `ProducerCursor`.
- **Relationship tables:** `CONTAINS`, `LINKS_TO` (`kind`, default `mention`),
  `INCLUDES_ASSET`, `PART_OF`.
- **Indexes:** `concept_embedding`, `concept_fts`, `image_omni_idx`,
  `chunk_embedding`, `chunk_fts`; vector index recipe
  `mu := 30, ml := 60, metric := 'cosine', efc := 200`.
- **Pins:** `Meta` (`embedding_precision`, `embedding_image_precision`, write/index
  epochs) and `MetaText` (`embedding_model`, `embedding_space`,
  `embedding_image_model`, `embedding_image_space`,
  `embedding_image_text_space`). These are refusal contracts, not cache hints.
- **`Concept.extra` is `MAP(STRING, STRING)`**: strings stay raw, dates become ISO
  strings, everything else is `json.dumps(value, default=str)`. Python's JSON
  encoder uses ASCII escaping and default separators. This is part of the file
  format.
- **`content_sig`** = SHA-256 of
  `json.dumps(frontmatter_minus_body_embedding, sort_keys=True, default=str) + "\0" + body`.
- **Chunking** stores mordant blocks with exact byte offsets, a `+` continuation
  suffix on post-split pieces, and no stored overlap; overlap is reconstructed at
  embedding time.

Any port that changes these values silently re-embeds or re-timestamps a graph.
They are compatibility work items, not implementation details.

---

## 3. Feasibility findings

### 3.1 `lbug` — the only real platform risk

`LadybugDB/ladybug-rust` publishes `lbug` 0.22.1/0.22.3 on crates.io; the crate is
a `cxx` wrapper over the same C++ engine the Python wheel uses. It exposes
`Database`, `Connection`, prepared statements, `Value` (including `Map`, `List`,
`Timestamp`, `Blob`), and `QueryResult` iteration.

Constraints to design around:

- The build downloads a precompiled `liblbug` archive or compiles the C++ source,
  and links OpenSSL. CI and developer machines need that toolchain; cache
  `LBUG_PREBUILT_CACHE_DIR`.
- **Windows OpenSSL is a prerequisite, not a crate pin problem.** `lbug 0.22.1`
  already contains the Windows resolver from
  [PR #26](https://github.com/LadybugDB/ladybug-rust/pull/26) (published in the
  0.20.x line; see [issue #30](https://github.com/LadybugDB/ladybug-rust/issues/30)).
  Its `build.rs` searches, in order: `OPENSSL_DIR`/`OPENSSL_ROOT_DIR`, vcpkg,
  `pkg-config --variable=libdir openssl`, Homebrew paths (macOS), then standard
  Linux paths. On Windows it emits `dylib=libssl` and `dylib=libcrypto`, so the
  linker must find **`libssl.lib` and `libcrypto.lib`** — do not rename them to
  `ssl.lib`/`crypto.lib`. Preferred setup is vcpkg `install openssl:x64-windows`
  with `VCPKG_ROOT` (or vcpkg integration) set; a ShiningLight/Chocolatey OpenSSL 3
  dev install plus `OPENSSL_DIR` is also supported. The matching
  `libssl-3-x64.dll`/`libcrypto-3-x64.dll` must be on `PATH` at runtime. Do not
  switch to `lbug` `main` or change the pin to work around this. On this machine,
  **do not install ShiningLight, Chocolatey, or any other system OpenSSL without
  explicit user confirmation**; use an existing install via `OPENSSL_DIR` or ask
  first.
- **Windows setup used for the green spike (local):** clone vcpkg outside the
  repo, run `bootstrap-vcpkg.bat -disableMetrics`, then
  `vcpkg.exe install openssl:x64-windows --disable-metrics`. Because this is the
  **dynamic** triplet, Cargo also needs `VCPKGRS_DYNAMIC=1`; set
  `VCPKG_ROOT='C:\Users\Main\Documents\Rust\vcpkg'` (Windows-style path) and
  prepend `...\installed\x64-windows\bin` to `PATH` for the runtime DLLs.
  A build-script change to `VCPKG_ROOT` requires `cargo clean -p lbug` to force
  OpenSSL re-discovery.
- Extensions (`vector`, `fts`) are dynamically loaded `.lbug_extension` files.
  On non-Windows static builds the final binary needs `-rdynamic` (or equivalent)
  so the extension can resolve host symbols. Each binary and the cdylib must emit
  that link arg.
- Pin `lbug = "=0.22.1"` to match the Python wheel's storage/extension ABI. Treat
  0.22.3 as a deliberate upgrade with a cross-open test, not an incidental bump.
- LLVM/CMake build time is significant; keep `lbug` out of the hot edit loop
  (workspace caching, `sccache`, prebuilt cache).

**Spike gate:** create a DB with Python 0.22.1, open it in Rust, `CALL TABLE_INFO`,
read `Concept`, load `vector`/`fts`, run `QUERY_VECTOR_INDEX` and
`QUERY_FTS_INDEX`, write a row, reopen in Python. Repeat on Windows x64 and Linux
x86_64. Only then commit to the crate path.

### 3.2 `embroider`, `bobine`, `mordant`

All three are Rust cores with optional PyO3 bindings, already used by the Python
wheel. Calling the Rust APIs removes the Python packages from the wheel entirely
and should be numerically identical:

- `embroider::JinaV5` (text and Gemma-2), `JinaV5Vision`/`GemmaVision`,
  `Reranker`, registry/acquisition, device/precision resolution, runtime
  discovery, token counting.
- `bobine::HybridConverter` with `ConverterConfig`; same conversion contract and
  routing.
- `mordant::MarkdownChunker::{get_all_chunks,get_delimiter}`; the Python-side
  `_cap_chunk_budget` and overlap logic must still be ported.

**Exception:** `embroider.prepare_image` is Python + Pillow (`_image.py`), not in
the Rust crate. Image decode and Pillow-compatible bicubic resize must be ported
(see risk R5).

Pin exact native crate versions to the oracle (`embroider = "=0.15.1"`,
`mordant = "=0.10.1"`, `bobine = "=0.9.1"`, `lbug = "=0.22.1"`) so a routine
`cargo update` cannot silently change tokenization, chunking, conversion, or
storage behavior. Upgrades are deliberate parity events.

### 3.3 MCP and HTTP

`rmcp` 3.11 is the official Rust SDK and supports stdio plus streamable HTTP.
`axum` 0.8 covers `/ops/*`, `/jobs/*`, `/health`, `/mcp`, discovery files, and
Unix sockets (via `hyper-util`). The Python `mcp` SDK stays as a **test client**,
not as a runtime dependency.

### 3.4 PyO3

PyO3 0.30.1 + maturin support `abi3-py312` wheels. The binding is a single
`cdylib` (`okfgraph._native`) with PyO3 classes and functions; long calls release
the GIL. In PyO3 0.30 the `allow_threads` API is named `Python::detach`; use that
at the FFI boundary. Wheels target Python 3.12–3.14. Panics must be caught at the FFI
boundary (`catch_unwind` / PyO3's panic conversion); segfaults inside Ladybug/ORT
are not recoverable and need the crash-report discipline retained by the shim.

### 3.5 M0 spike results — Windows x64 **GO**

Recorded on `x86_64-pc-windows-msvc` with `lbug = 0.22.1`, Python oracle 0.15.0,
Python 3.13.14, Rust 1.100.0:

- **OpenSSL/vcpkg:** `vcpkg install openssl:x64-windows` built `3.6.5#2` in
  7.1 min. With `VCPKG_ROOT` (Windows-style path), `VCPKGRS_DYNAMIC=1`, and the
  vcpkg `bin` directory on `PATH`, the final link succeeds. Initial clean
  `cargo check -p lbug`: 54.61 s; warm clean `lbug` rebuild: ~21.7 s.
- **lbug cross-open:** Python-created `parity/fixtures/oracle-v10.db` opens in
  Rust; `CALL TABLE_INFO('Concept')` returns the expected 14 columns including
  `FLOAT[3]` and `MAP(STRING, STRING)`. `INSTALL`/`LOAD vector` and `fts` succeed;
  `QUERY_VECTOR_INDEX` and `QUERY_FTS_INDEX` both return the spike row. A Rust
  `CREATE` succeeds, and `parity/tools/check_rust_write.py` confirms Python 0.15
  reopens the copy and sees `['one', 'rust-write']`.
- **Ladybug note:** a table with vector/FTS indexes needs `LOAD vector`/`LOAD fts`
  **in every process that writes to it**; otherwise Ladybug reports “Trying to
  insert into an index on table … but its extension is not loaded.” The Rust
  spike now loads both before writes. No storage/ABI deviation was observed.
- **PyO3/maturin:** `python bindings/python/build_wheel.py --release` produces
  `okfgraph-0.16.0a1-cp312-abi3-win_amd64.whl` (164 KB compressed, well under the
  60 MB budget). A clean venv imports the facade, reads `version()`,
  `schema_version()`, and 30 op names, and two concurrent 150 ms native calls
  finish in 0.151 s, proving GIL release. PyO3 0.30 names this API
  `Python::detach`, not `allow_threads`. Release wheel build: 2 min 38 s warm.
- **rmcp:** `examples/rmcp_spike.rs` serves an `echo` tool over stdio; the Python
  `mcp` client lists it and receives `echo:ping`.

**Gate status:** Windows x64 is **GO** for all three spikes. Linux x86_64 and
macOS arm64 cross-open/spike runs are still required before Stage A ships. No
`lbug-sys` fallback is needed on Windows.

---

## 4. Target architecture

### 4.1 Repository layout (locked)

```
Cargo.toml                     # [package] okfgraph + workspace members
rust-toolchain.toml
src/                           # ALL core Rust (one `okfgraph` crate)
  lib.rs
  bin/okf.rs                   # required-features = ["cli", "server", "mcp"]
  bin/okf-mcp.rs               # required-features = ["cli", "server", "mcp"]
  settings/                    # 52-setting table, TOML/env/CLI precedence, validation
  error.rs                     # codes, classes, envelope
  model/                       # Concept/Chunk/Image, extra, timestamps, pyjson
  markdown/                    # frontmatter, PyYAML-compatible YAML, chunking, lint
  store/                       # lbug wrapper, schema v10, migrations, indexes, lock
  embed/                       # Embedder trait, embroider, external HTTP, pins, GGUF
  ingest/                      # walk, parse, import, delta, links, images, purge,
                               # export, producers, bibtex, converters
  search/                      # hybrid/RRF, chunk, hub, PPR, rerank, read, traverse
  ops/                         # 30 op requests/results, validation, Router facade
  server/                      # axum + rmcp + Client + discovery + jobs (features)
  cli/                         # clap command builder + renderers (feature)
  util/                        # paths, hashing, clock, net/SSRF, fs, crash
tests/                         # Rust integration tests against temp DBs
bindings/python/               # okfgraph-py: PyO3 cdylib (binding only)
addons/pdf/                    # okfgraph-pdf: native PDF/Office plugin (bobine)
python/okfgraph/               # compatibility facade (pure Python)
python/okfgraph_pdf/           # PDF add-on locator (plugin_path, version)
python/tests/                  # public-API compatibility suite
parity/                        # oracle scenarios, corpus, differential harness
```

`Cargo.toml` at the root is the `okfgraph` package. CLI/MCP/server are
feature-gated modules and two thin `src/bin` targets, **not** separate crates. The
only other Rust build targets are `bindings/python` (the separate PyO3 layer) and
`addons/pdf` (the slim-wheel add-on); neither is part of the core library's
dependency graph unless built explicitly. If the project later prefers a single
Cargo target for the binding as well, `bindings/python` can be folded into
`src/bindings/python` behind a `python` feature; the public Python package and
this plan's contracts do not change.

Why one core crate: the user-facing artifact is `cargo add okfgraph`; module
boundaries are enforced by visibility and tests. The adapters are feature-gated
so a plain library build does not pull clap/axum/rmcp. If compile times become a
problem, `store`, `embed`, and `ingest` can be extracted later without changing
the public facade. Do not start with crate fragmentation as a goal.

### 4.2 Dependency direction

```
util, model, settings, error
        ↓
      store
        ↓
 embed   markdown
    ↓       ↓
     ingest      search
          ↓
         ops
          ↓
server, cli, bindings/python
```

`store` must not know about HTTP, MCP, Python, or CLI. `ingest`/`search` operate on
traits (`Embedder`, `Converter`, `Clock`, `FileSystem`) so tests can substitute
fakes. `ops` is the only place that orchestrates a whole operation.

### 4.3 Feature flags

| Feature | Default | Pulls in | Purpose |
|---|---|---|---|
| `embed` | yes | `embroider` | text/vision embeddings, rerank, tokens |
| `external` | yes | `reqwest` | OpenAI-shaped embed/rerank/vision lanes |
| `vision` | yes | `embroider` vision | ONNX image-content search |
| `cli` | no | `clap` | `okf` binary and CLI renderers |
| `server` | no | `axum`, `tokio` | `okf serve`, jobs, discovery, Client |
| `mcp` | no | `rmcp` | 11 MCP tools, stdio and streamable HTTP |
| `pdf` | no | `bobine` | static PDF/Office conversion (Rust users) |
| `bench` | no | `criterion` | benchmarks |
| `test-fixtures` | no | fixtures | golden helpers |

**Build profiles:**

- Library: `cargo add okfgraph` → `embed + external + vision`, no clap/axum/rmcp.
- CLI: `cargo install okfgraph --features cli,server,mcp,pdf` (or without `pdf`
  and use the plugin). `server` is required for `okf serve`; `mcp` for
  `okf-mcp` and the `/mcp` mount.
- Python wheel (slim): core + binding with `cli,server,mcp,vision,external`;
  **no `pdf`**.
- PDF add-on: `pip install okfgraph[pdf]` installs the separate `okfgraph-pdf`
  native plugin; the Python facade sets `OKFGRAPH_PDF_PLUGIN` from
  `okfgraph_pdf.plugin_path()` before opening a router. Conversion still needs
  ORT from `okfgraph[cpu]` or `okfgraph[gpu]`.

**PDF add-on design.** `addons/pdf` is a `cdylib` with a deliberately tiny C ABI
(`abi_version`, `convert(input, workdir, config_json) -> markdown/error`, `free`),
loaded lazily with `libloading`. It wraps the same `bobine` crate. This keeps the
slim wheel free of bobine while letting both the Rust CLI and the Python binding
use the add-on. Rust users who do not want a plugin can enable the static `pdf`
feature instead. Do **not** reintroduce the Python `bobine` package as a runtime
dependency.

**Missing-plugin behavior is `CONVERSION_FAILED` (locked).** In Python 0.15 a
missing `bobine` raises a plain `RuntimeError`, which the CLI catch-all turns into
`INTERNAL` plus a crash report. The port intentionally fixes this: return
`CONVERSION_FAILED` with a `pip install "okfgraph[cpu,pdf]"` (or `gpu`) remedy,
and record it as a 0.16 behavior change. An ABI-incompatible plugin is also
`CONVERSION_FAILED`; never a bare import error or crash report.

**PDF shares the ONNX Runtime (locked).** `pip install "okfgraph[pdf]"` installs
the plugin, but conversion uses the same ORT lane as embedding (bobine's models
are ONNX). The documented install is `okfgraph[cpu,pdf]` or
`okfgraph[gpu,pdf]`; the plugin resolves ORT through the same
`ORT_DYLIB_PATH`/Python `onnxruntime` discovery. The add-on does not bundle ORT.

**The import surface is extras-independent.** The Python package is always
`okfgraph`; the public names (`OKFRouter`, `OKFClient`, models, errors,
`settings`) never change. The `pdf` extra installs a *separate distribution*
(`okfgraph-pdf`) containing the plugin and a locator module; user code never
imports it. The facade does not use `try: import okfgraph_pdf` in the user's
namespace — it resolves the plugin path before opening a router (or the core
looks for it via `OKFGRAPH_PDF_PLUGIN`/executable-relative paths).

Consequences:

- Reading/searching an existing graph with PDF-derived concepts works without the
  add-on; only new PDF/Office conversion fails, at the operation, with
  `CONVERSION_FAILED` (or `NO_ORT_RUNTIME` when the runtime is also absent)
  naming `pip install "okfgraph[cpu,pdf]"` / `"okfgraph[gpu,pdf]"`.
- A missing or ABI-incompatible add-on never breaks `import okfgraph`, the CLI
  read commands, or non-PDF ingest.
- `okf doctor` reports the PDF lane as available/unavailable, so the capability
  is discoverable without importing an optional module.

### 4.4 Public Rust API (sketch)

```rust
use okfgraph::{Router, Settings, SearchRequest, ReadRequest, ImportRequest, Op};

let router = Router::open(
    Settings::builder()
        .db_path(".okfgraph/okfgraph.db")
        .bundle_root("bundle")
        .build()?,
)?;

let hits = router.search(SearchRequest::new("badger diet"))?;
let concept = router.read(ReadRequest::new("notes/badgers"))?;
let report = router.import_bundle(ImportRequest::default())?;

// Adapter/FFI escape hatch used by CLI, MCP, HTTP and Python:
let envelope = router.call(Op::Search, serde_json::json!({"query": "badger"}))?;
```

Rules:

- `Router` is `Send + Sync` **via interior synchronization** around mutable
  store/engine state (a `Mutex` initially; a read pool can come later). The lock
  scope is documented and never held across a callback or Python re-entry. It is
  cheap to wrap in `Arc`; Python calls release the GIL around blocking work.
- Every op has a typed request struct with `Default`, `Deserialize`, and
  validation; every op is also reachable through `call(Op, Value)` so adapters
  cannot drift.
- Public errors are `thiserror`-based `OkfError`; no `anyhow` in the public API.
  Error codes are the same closed set as `okfgraph/errors.py`.
- Core ops are synchronous and may block (Ladybug, ONNX). The `server` module
  runs them on a blocking pool. No hidden global runtime.
- `Clock` is injected (system clock by default) so timestamps are testable.
- `Settings` is a plain struct plus a public `SETTINGS: &[SettingDef]` table used
  to generate CLI/MCP flags and docs.

### 4.5 Python binding layer (exactly one)

`bindings/python` (Cargo package `okfgraph-py`):

- `#[pymodule] okfgraph._native` with:
  - `NativeRouter` (pyclass): `open`, `call`, `close`, read-only properties;
  - `NativeClient` (pyclass): HTTP/Unix-socket client, jobs;
  - `NativeError` (pyclass): `code`, `message`, `fields`, `remedy`, `data`;
  - `run_cli(argv) -> i32`, `run_mcp(argv) -> i32`, `run_serve(argv) -> i32`;
  - `version()`, `schema_version()`, `settings_table()`, `settings_defaults()`.
- No embedding/search/import logic. No parsing of frontmatter. No SQL.
- Long operations are wrapped in `py.allow_threads`; results are converted with
  `serde_json`/`pythonize`.
- Errors cross the boundary as `NativeError`; `okfgraph/errors.py` translates to
  the historical `OKFError`/`UsageError`/`StateError`/`OutcomeError` classes and
  preserves `.code`, `.fields`, `.remedy`, `.data`, `.envelope()`.
- Python facade files live under `python/okfgraph/`:
  - `python/okfgraph/__init__.py` — lazy `OKFRouter`, `ConceptModel`, `cli_main`
    exports.
  - `python/okfgraph/router.py` — `OKFRouter` with the current constructor kwargs
    and 30 methods; each method builds a params dict, calls `_native`, converts
    results.
  - `python/okfgraph/models.py` — Pydantic models kept for `isinstance` and
    `.model_dump()`; no graph logic.
  - `python/okfgraph/errors.py` — exception classes and `envelope()`; no graph
    logic.
  - `python/okfgraph/client.py` — `OKFClient` proxy over `NativeClient`.
  - `python/okfgraph/settings.py` — constants and a read-only settings table view.
  - `python/okfgraph/cli.py`, `python/okfgraph/mcp_server.py`,
    `python/okfgraph/serve.py`, `python/okfgraph/_entry.py` — thin entry shims
    calling native `run_*`.
  - `python/okfgraph/crash.py` — crash report writer for start-up failures (pure
    Python).
  - `python/okfgraph/py.typed` — PEP 561 marker.
- **Not preserved:** `okfgraph.components.*`, `okfgraph.ops.*`, and private
  helpers in `cli.py`/`router.py`. These are internals; the port is a major
  version for Python consumers who imported them. Tests that depend on them are
  ported to Rust, not emulated.

### 4.6 Mapping of today's Python modules

| Python | Rust module | Notes |
|---|---|---|
| `settings.py` | `settings` | table-driven; same precedence and legacy-name refusals |
| `errors.py` | `error` | same codes/classes/envelope |
| `models.py` | `model` | serde + flatten `extra`; Pydantic shim stays in Python |
| `security.py` | `util::security` | SSRF, path sandbox, cache verification |
| `crash.py` | `util::crash` + shim | native crash reports; Python start-up guard remains |
| `components/timestamps.py` | `model::timestamps` | `pyjson` for signatures |
| `components/schema.py` | `store::schema` | v10 DDL, migrations v1→v10, index builds |
| `components/extensions.py` | `store::extensions` | INSTALL/LOAD, offline `extension_dir` |
| `components/delta.py` | `ingest::delta` | file/dir hashes, tombstones, multi-root |
| `components/roots.py` | `util::paths` + `ingest` | alias/namespace/resolution |
| `components/import_.py` | `ingest::import` | parse, identity, upsert, links, reports |
| `components/ingest.py` | `ingest::ingest` | md/pdf/thoughts/bib, materialize |
| `components/converters.py` | `ingest::convert` | bobine adapter |
| `components/image_assets.py`, `images.py` | `ingest::images` | decode/resize, hashes, routes |
| `components/embedding.py` | `embed` + `markdown::chunk` | engine, pins, chunker, reconstruction |
| `components/external_embedder.py` | `embed::external` | OpenAI-shaped + identity probe |
| `components/external_reranker.py` | `search::rerank::external` | TEI/standard/rerank dialects |
| `components/external_vision.py` | `embed::external::vision` | host identity/text partner |
| `components/gguf_meta.py` | `embed::counter::gguf` | chars/token prior |
| `components/search.py` | `search` | hybrid, chunk, filters, PPR/hub, rerank |
| `components/ranking.py` | `search::ppr` | pure algorithms |
| `components/links.py` | `ingest::links` | wikilinks/path links/broken links |
| `components/diff.py` | `ingest::diff` | structural diff |
| `components/purge.py` | `ingest::purge` | soft-delete/trash/recover |
| `components/export.py` | `ingest::export` | OKF/Obsidian, index files |
| `components/producers.py` | `ingest::producers` | built-ins + registration; Python entry points bridged |
| `components/doctor.py` | `ops::doctor` | session checks, scored report |
| `components/bibtex.py` | `ingest::bibtex` | stdlib-only semantics |
| `components/lint.py` | `markdown::lint` | mordant linter + diagnostics |
| `components/skills.py` | `server`/`cli` assets | `include_str!` skill files |
| `ops/*.py` | `ops` | one request/result per op; validation preserved |
| `router.py` | `ops::Router` | facade, lock, lifecycle |
| `client.py` | `server::client` + Python shim | HTTP/UDS, discovery |
| `serve.py` | `server` | axum, jobs, health, MCP, discovery |
| `mcp_server.py` | `server` + `src/bin/okf-mcp.rs` | rmcp tools, stdio/connect |
| `cli.py` | `cli` + `src/bin/okf.rs` | clap from `SETTINGS`, renderers, exit codes |

---

## 5. Compatibility contract (must be explicit in code and tests)

### 5.1 Database

- Open and migrate v1→v10; write v10. Never bump the schema during the port.
- Same table/column names, primary keys, `MAP` encoding, index names/recipes.
- Same `Meta`/`MetaText` pin keys and refusal semantics
  (`MODEL_PIN_MISMATCH`, `PRECISION_PIN_MISMATCH`, `SPACE_PIN_MISMATCH`,
  `IMAGE_PIN_MISMATCH`, `DIM_MISMATCH`).
- Same write-epoch/index-epoch dirty tracking; `reindex` force drops/recreates.
- Same single-writer discipline: inter-process file lock + `okf serve` owner +
  discovery file; `DB_LOCKED` / `WRITE_LOCK_TIMEOUT` unchanged.

### 5.2 Values and hashing

Implement a `pycompat` module and pin it with fixtures generated from Python:

- Python `json.dumps` behavior: `ensure_ascii=True`, default separators
  `", "`/`": "`, insertion-order maps, `sort_keys` where used, `default=str`
  for objects JSON cannot encode. **Do not use `serde_json` defaults for
  parity-critical strings:** Rust emits UTF-8 where Python emits `\uXXXX`,
  uses compact separators, sorts object keys without `preserve_order`, and
  spells float exponents differently. Write/configure a Python-compatible
  encoder and test non-ASCII, nested maps, floats, control characters, and
  U+2028/U+2029.
- `content_sig` uses `default=str`, so a YAML `datetime` becomes
  `"YYYY-MM-DD HH:MM:SS"` (space, not `T`), while `extra_value` special-cases
  dates/datetimes to `.isoformat()`. Do not collapse these two paths.
- `extra_value`: raw string, ISO date/datetime, else Python-JSON.
- `content_sig`: exact bytes and SHA-256.
- `datetime.isoformat()` formatting (naive UTC; fractional seconds only when
  present) at the API boundary.
- Float formatting for scores and `Value` conversion; never emit `NaN`/`Inf`.
- Python `str()` of booleans/floats where it leaks into stored strings.

### 5.3 Frontmatter and Markdown

- `python-frontmatter` delimiter behavior and file decoding.
- PyYAML **1.1** scalar resolution as used by `yaml.safe_load`: `yes/no/on/off`,
  `null`, ints (including `0x`/`0o`), floats, dates/timestamps, quoted scalars,
  lists/maps, duplicate-key behavior.
- Pydantic model behavior: unknown keys preserved, `tags` coercion, `id`→`uid`,
  `source_path` under frontmatter identity, `resource` credential sanitization.
- Mordant block boundaries/types/offsets; `_cap_chunk_budget` exact tiling;
  continuation suffix; delimiter selection for reconstruction.
- Lint diagnostics used by `ingest` and `lint` commands (mordant linter + the
  Python post-processing).

### 5.4 Search and ranking

- Same Cypher semantics (`QUERY_VECTOR_INDEX`, `QUERY_FTS_INDEX`, `efs`,
  `top := k`), same RRF constant/weights, same tie-breaking by id.
- Same filter semantics: `where` on stored `extra` strings, tags/exclude tags,
  parent subtree, `since`/`until` on `effective_at`, `near_id` neighbourhood,
  `offset`, `recency`, `dedupe`, `link_kinds`.
- Same chunk collapse (cosine ≥ 0.97), hub/PPR math, cross-encoder windowing,
  token-budgeted reads.
- Model-free PPR must work with no embedder loaded.

### 5.5 Output surfaces

- Envelope field order and values: `{ok, op, data, warnings, error}`. If byte
  order matters, serialize a struct/ordered map; `serde_json::Value` maps sort
  keys by default unless `preserve_order` is enabled.
- One approved error change: a missing/incompatible PDF add-on is
  `CONVERSION_FAILED` with an install hint, not the 0.15 `INTERNAL`/crash report
  (see §4.3).
- Error `code`/`message`/`fields`/`remedy`/`data`; exit code 2 for usage, 1 for
  state/outcome.
- MCP tool names, descriptions, annotations, JSON schemas, `isError` behavior,
  content shape.
- CLI command/flag spellings, `--json`, human renderers, shell, `setup`, help
  text (help prose may change; flags/spellings may not).
- `okf serve` routes and job lifecycle (`/ops/<op>`, `mode=job`, `/jobs`,
  `/jobs/<id>`, `/jobs/<id>/cancel`, `/health`, `/mcp`, discovery file).

### 5.6 Explicit breakages

Only internals are allowed to break:

- `okfgraph.components.*` and `okfgraph.ops.*` imports.
- Private helpers inside `cli.py`/`router.py` (e.g. `build_parser`,
  `_SHELL_BANNER`).
- Byte-for-byte CLI help prose and export YAML formatting (semantic parity is
  required; exact whitespace is not unless tests demand it).
- Python package dependencies (`ladybug`, `embroider`, `bobine`, `mordant`,
  `python-frontmatter`, `PyYAML`, `Pillow`, `numpy`, `fasteners`, `mcp`).
- The README's "Components remain reachable for advanced use"
  (`search_engine.search_hybrid`, `embed_engine.count_tokens`, `export_mgr`,
  `diff_mgr`, ...). This is a documented advanced surface, but it is part of the
  internals being removed; the migration guide must call it out explicitly.
- The missing-PDF-add-on path changes from `INTERNAL` + crash report to
  `CONVERSION_FAILED` + install hint. This is an intentional, approved behavior
  change, not a parity failure; the parity report must list it.

Everything in §5.1–§5.5 is a release blocker.

---

## 6. Work breakdown

Estimates are engineer-weeks for one experienced Rust engineer familiar with the
Python tree. Two engineers can run M1+M2 and M3 in parallel after M0.

### Delivery stages (locked)

| Stage | Milestones | Ships | Python package state |
|---|---|---|---|
| **A — Read-only content** | M0, M1, M2 | `okfgraph` crate + `okf` read commands on existing graphs; read ops in the PyO3 binding; schema migration/`okf init` allowed, no content/import writes. | 0.15 stays the released package; Rust is an alpha crate/binary. |
| **B — Write path** | M3 | Import/ingest/export/update/delete and the full write ops; DB dump parity. | `0.16.0bN` wheel (beta) replaces the implementation; 0.15 remains installable. |
| **C — Full surfaces** | M4, M5, M6 | All 29 CLI commands, 11 MCP tools, `okf serve`, slim wheel + PDF add-on, Python public API parity. | `0.16.0` stable; Python core deleted. |

A stage is not released until its gate passes; the Python oracle tag remains
buildable throughout for differential testing and cross-open.

### M0 — Feasibility, baseline, harness (1.5 weeks)

- Tag the Python tree `python-oracle-0.15.0`; freeze its lockfile and build a
  reusable oracle venv/Docker image.
- Scaffold the single `okfgraph` crate (root `Cargo.toml`, `src/`), the
  `bindings/python` PyO3 crate, `python/okfgraph`, and the `addons/pdf` stub;
  toolchain, fmt/clippy/deny, CI stub.
- **lbug spike:** Python-created v10 DB → Rust open/read/write → Python reopen;
  load `vector` + `fts`; run both index queries; Windows + Linux. Windows CI and
  dev machines must first provision OpenSSL 3 (`vcpkg install
  openssl:x64-windows` or `OPENSSL_DIR`), as recorded in §3.1.
- **PyO3 spike:** maturin wheel with `_native.version()` and a blocking call with
  GIL release. Maturin only includes the facade when `python-source` resolves to
  a readable local directory; use `bindings/python/build_wheel.py` to stage
  `python/okfgraph` into `bindings/python/python/` before building.
- **rmcp spike:** stdio server answering one tool; Python `mcp` client connects.
- Build the parity harness skeleton:
  - corpus format (bundle + settings + op script);
  - Python runner emitting normalized JSON;
  - Rust runner (`--json`) and comparator;
  - DB dump (all tables, sorted) and export-file normalization;
  - clock injection/masking.
- Write the compatibility spec (`docs/rust-compat.md`) with §5 as the checklist.

**Gate:** all three spikes pass on Windows x64 and Linux x86_64; the go/no-go
decision is recorded. If Ladybug fails, the fallback is a `lbug-sys` C-API shim;
if extensions cannot load, stop and reassess before any port work.

### M1 — Core, store, read-only slice (3 weeks)

- `pycompat` (JSON/YAML/time), `model`, `error`, `settings` (52 rows, precedence,
  TOML discovery, legacy refusals, validation).
- `store`: schema v10, migrations v1→v10, `Concept`/`Chunk`/`ImageAsset`/link/
  hash/meta mapping, extension install/load, index create/drop, file lock.
- `util`: paths/aliases, SHA-256, clock, security/SSRF, crash reports.
- Ops: `get_by_id`, `list_directory`, `read`, `traverse`, `pin_info`, `lanes`,
  `doctor` (basic), `list_broken_links`, `list_deleted`, `list_images`,
  `get_image`.
- CLI: `okf read`, `okf traverse`, `okf images`, `okf image`, `okf doctor`,
  `okf model-info`, `okf init` (schema creation; no source import).
- PyO3: `NativeRouter.open/call/close`; Python `OKFRouter` constructor and these
  read ops.
- Tests: open legacy fixture DB, row-by-row parity for every table, type
  round-trips (timestamps, maps, blobs), migration tests from v1/v5/v9 fixtures.

**M1 progress:** `src/settings/table.rs` is generated from the oracle (52 rows);
`src/store` opens the v10 fixture, dumps all tables, and implements
`get_by_id`, `list_directory`, `read`, `traverse`, image/list ops, pins, lanes
and a basic `doctor`. `parity/scenarios/read-only.json` runs 14 ops through
both runtimes and compares with **0 differences**. The PyO3 `NativeRouter` and
Python `OKFRouter`/`ConceptModel` facade pass `parity/tools/check_facade.py`;
`okf read`/`init`/`doctor` work from a build with `--features cli,server,mcp`.
Remaining M1 gaps: full migration fixtures (v1/v5/v9), byte-for-byte DB dump
relationships, and `max_tokens`/`document`/`context` read parity.

**Gate:** Rust reads a Python-created DB identically (JSON value parity) and the
Python shim returns the same `ConceptModel` shape. No **content** writes yet;
`okf init` and schema migrations are the only writes allowed in Stage A.

### M2 — Embedding, chunking, search (3 weeks)

- `embed`: `Embedder` trait; embroider text/vision/reranker; device/precision
  resolution; model acquisition/cache; identity/space pins; `model_info`/`lanes`;
  token counters; external embed/rerank/vision lanes; GGUF prior.
- `markdown`: frontmatter parser, PyYAML resolver, mordant chunker,
  `_cap_chunk_budget`, overlap payloads, reconstruction, lint/fix.
- `search`: index query wrappers, RRF, filters, hybrid, chunks (plain/expand/hub),
  hub/PPR, recency, dedupe, cross-encoder rerank, budgeted read, path/traverse,
  image search.
- Ops: `search`, `embed`, `read` (all includes), `traverse` (all modes).
- Tests: Python-vs-Rust vector parity (text and image), chunk/offset parity,
  search result parity on the corpus and a legacy DB, model-free PPR, pin refusals.

**Gate:** identical search result IDs/order/scores on the parity corpus
(floats within 1e-6); vector cosine ≥ 0.99999 for text and ≥ 0.999 for images;
chunk text/offsets byte-identical.

### M3 — Ingest, import, export (4 weeks)

- `ingest::parse`: frontmatter identity, synthesized titles/types, source-path
  provenance, resource sanitization.
- `ingest::links`: wikilinks, source-relative then root-relative path links,
  aliases, typed relations, ambiguous/missing `BrokenLink`, repair.
- `ingest::delta`: file/dir hashes, changed/deleted classification, tombstones,
  isolated pinned trees, per-file skip, move detection, multi-root, detach.
- `ingest::import`: batch upsert, capture/restore incoming links, directory
  hierarchy, duplicate identity refusal, per-file reports, index epochs.
- `ingest::ingest`: md, pdf/docx/xlsx/pptx (bobine), thoughts, bib; materialize;
  converter lifecycle/idle close; work-dir import.
- `ingest::images`: asset IDs, extraction, remote fetch/SSRF, caption vs vision
  routes, content hashes, Pillow-compatible decode/resize.
- `ingest::export`, `purge`, `producers`, `bibtex`, `diff`.
- Ops: `import_bundle`, `import_file`, `import_files`, `sync`, `ingest`,
  `export_bundle`, `export_concept`, `delete`, `recover_deleted`,
  `purge_deleted`, `detach`, `repair_links`, `diff`, `lint`, `produce`,
  `reindex`, `reembed`.
- Tests: fresh-import DB dump parity, re-import idempotence, link preservation
  across re-import, moved/duplicate files, hidden dirs, reserved names, multi-root,
  tombstone/purge, PDF/Office/bib/thoughts, export round-trip.

**Gate:** a fresh Rust import of the parity corpus produces a DB dump and export
tree semantically identical to Python; re-import of unchanged files performs no
embedding; a Python 0.15 process can reopen the Rust-written DB.

### M4 — Ops, CLI, MCP, serve (3 weeks)

- Complete the 30 ops, including validation messages/fields and outcome errors.
- CLI: all 29 commands, generated global flags, rendering, `--json`, exit codes,
  shell, `setup extensions/skills`, `model-info`, project discovery.
- `server` module + `src/bin/okf-mcp.rs`: axum routes, job manager with
  progress/cancel, discovery file + stale-pid handling, TCP and Unix socket,
  health, MCP mount.
- MCP: 11 tools with rmcp, stdio and `--connect` proxy.
- `server::client`: HTTP/UDS client, jobs, discovery; Python `OKFClient` shim.
- Tests: CLI golden JSON, exit codes, MCP schema/annotation snapshots, serve
  lifecycle, concurrent clients, stale discovery, job cancel, UDS, Python client.

**Gate:** CLI/MCP/serve contract suite green; Python `OKFRouter` and `OKFClient`
perform every op with parity; no Python transport/runtime dependency.

### M5 — Python binding and packaging (2 weeks)

- Complete `bindings/python`: all classes/functions, `allow_threads`, panic
  mapping, `NativeError` → `OKFError` translation, JSON conversion.
- Python facade under `python/okfgraph`: `OKFRouter` (52 constructor kwargs + 30
  methods), `models`, `errors`, `client`, `settings`, entry shims, `py.typed`.
- Maturin: `abi3-py312`, `python-source = "python"`, slim wheel features
  (`cli,server,mcp,vision,external`; **no `pdf`**), extras (`cpu`/`gpu` only),
  auditwheel/delocate, stripped release profile.
- PDF add-on wheel: `addons/pdf` plugin + `python/okfgraph_pdf` locator; extra
  `pdf` installs it; missing plugin raises `CONVERSION_FAILED` with the install
  hint. Document that conversion still needs `cpu`/`gpu` ORT.
- CI matrix: Windows x64, Linux x86_64, macOS arm64; Python 3.12, 3.13, 3.14;
  wheel smoke tests.
- Public compatibility suite: the documented Python usage, import time, no
  imports of removed packages, console scripts.

**Gate:** `pip install` from a built wheel on a clean machine; every public Python
API test passes; `pip show okfgraph` lists only `pydantic` (+ ORT extras); `okf`
and `okf-mcp` run from the wheel.

### M6 — Cutover and hardening (3 weeks)

- Delete the Python implementation from the package (keep it on the oracle tag);
  move/port the remaining tests.
- Differential suite becomes release-gating; run weekly and before releases.
- Fuzz: frontmatter/YAML, wikilinks, path resolution, chunker, op params.
- Performance: import throughput, search latency, memory, startup; compare to
  Python and publish benchmarks.
- Security: SSRF/path sandbox, extension integrity, crash reports, dependency
  audit (`cargo deny`), license review.
- Docs: README Rust-first, architecture update, Python binding guide, migration
  guide (DB/bundle compatible; internals broken), release notes.
- Release 0.16.0 stable (Stage C) with the parity report attached. Stage A and B
  are prereleases/beta, not silent cutovers.

**Gate:** parity suite green on all target platforms; all 1,241 behaviors either
ported, covered by an equivalent test, or explicitly waived in the parity report;
no Python core code in the wheel.

### Schedule summary

| Phase | Stage | Weeks (solo) | Parallel track |
|---|---|---|---|
| M0 | A | 1.5 | — |
| M1 | A | 3 | store/core |
| M2 | A | 3 | embed/search |
| M3 | B | 4 | ingest/import |
| M4 | C | 3 | surfaces |
| M5 | C | 2 | binding/packaging |
| M6 | C | 3 | hardening |
| **Total** | | **~19.5** | **~11–13 with 2 engineers** |

The estimates assume the Python tree is frozen (bug fixes cherry-picked, no new
features) and that the parity harness exists from M0.

---

## 7. Parity and test strategy

### 7.1 Oracle and corpus

- **Oracle:** Python 0.15.0 in a frozen venv/Docker image. It owns the reference
  outputs and can still write DBs for cross-open tests.
- **Corpus** (checked in under `parity/corpus/`):
  - `minimal/` — one note, one link.
  - `full/` — frontmatter variety, tags, relations, wikilinks, path links,
    ambiguous names, broken links, code/tables/math, long docs, unicode.
  - `identity/` — frontmatter ids, moves, duplicate ids.
  - `multiroot/` — primary + aliased roots, absent roots, detach/re-attach.
  - `media/` — PNG/JPEG/WebP/EXIF/palette/alpha images, PDF, DOCX/XLSX/PPTX.
  - `bib/`, `thoughts/`, `producers/`.
  - `hidden/` — `.git`, `.obsidian`, reserved `index.md`/`log.md`.
  - `corrupt/` — bad YAML, invalid ids, unreadable files, non-UTF-8.
- **Scenarios:** JSON scripts listing settings, setup, and op calls. Each scenario
  has a Python output and a Rust output.

### 7.2 Comparison rules

| Surface | Rule |
|---|---|
| IDs, counts, types, booleans, envelopes, errors | exact |
| Text, frontmatter, chunk text/offsets, link sets | exact |
| Search order and membership | exact for the corpus; ties by id |
| Scores / RRF / hub / PPR | `abs(a-b) ≤ 1e-6` (or relative 1e-6) |
| Vectors | cosine ≥ 0.99999 text, ≥ 0.999 image (target bit-identical) |
| Timestamps | injected clock; mask wall-clock fields |
| DB tables | sorted JSON dump, all columns; exclude volatile epochs |
| Export files | parse frontmatter + body and compare semantically; optional byte diff |

### 7.3 Test layers

1. **Rust unit tests** for pure logic (ranking, PPR, hashing, pyjson, YAML,
   chunking, path/alias, settings precedence, link resolution, diff).
2. **Rust integration tests** with a temp DB and a fake/real embedder; migration
   fixtures v1/v5/v9/v10; extension-offline tests.
3. **Parity tests** (oracle vs Rust) for ops, DB dumps, and exports.
4. **Surface tests**: `assert_cmd` for CLI, Python `mcp` client for MCP,
   `reqwest`/Python client for serve, `insta` snapshots.
5. **Python compatibility tests** against the built wheel (public API only).
6. **Heavy tests** (`#[ignore]`/env-gated): real models, CUDA, PDF, large corpus;
   run weekly.

Do not port tests 1:1 by filename; port **behavior**. Maintain a mapping from the
old test files to the new tests so the parity report can prove coverage.

---

## 8. Risks and mitigations

| ID | Risk | Impact | Mitigation |
|---|---|---|---|
| R1 | `lbug` build/extension ABI fails on a target | blocker | M0 spike; pin `=0.22.1`; `-rdynamic` per binary; Windows: provision OpenSSL via vcpkg `openssl:x64-windows` or `OPENSSL_DIR` pointing at `libssl.lib`/`libcrypto.lib`, with DLLs on `PATH`; keep `Store` behind a trait so a C-API shim is possible; test Python↔Rust cross-open |
| R2 | Python JSON/`str` semantics change `extra`/`content_sig` | silent re-imports, parity failures | `pycompat` module + Python-generated golden cases; property tests; DB dump gate |
| R3 | PyYAML 1.1 quirks not matched | wrong frontmatter/types | implement PyYAML resolver rules; fixture corpus; differential fuzz against `yaml.safe_load` |
| R4 | Import/delta/link edge cases diverge | data loss, link loss | port as a state machine; scenario corpus; DB dump + link invariants; run both on copies |
| R5 | Pillow decode/bicubic resize differs | image vector drift | port Pillow-compatible resampler or use libjpeg-turbo; cosine gate; if needed, one-time image re-embed with a schema/pin bump (last resort) |
| R6 | MCP/HTTP behavior mismatch | harness breakage | rmcp/axum spikes; Python SDK contract tests; snapshot schemas/annotations |
| R7 | PyO3 panics/GIL/segfaults | crashes in Python | catch_unwind at boundary; `panic = "unwind"`; no unwinding into Python; crash reports; stress tests; never share a Ladybug connection across threads without synchronization |
| R8 | Build time/wheel size | developer and user friction | `lbug` prebuilt cache, `sccache`, slim wheel + PDF add-on, stripped wheels, CI caching; publish a size budget (target ≤ 60 MB slim wheel) |
| R9 | Scope/test migration | schedule slip | feature freeze; behavior-based test triage; parity report; staged milestones with go/no-go |
| R10 | Cross-platform paths/clocks | Windows-only failures | path normalization tests, `\\`/`/` rules, case sensitivity, UNC/long paths, naive-UTC clock tests |
| R11 | Python 0.15 and Rust use different file-lock primitives (`fasteners` vs `fs4`) | two writers can bypass each other during the transition | keep one writer per DB; test lock interoperability on Windows/Linux; use the same OS primitive or a sidecar guard; document that 0.15 and 0.16 must not run concurrently on one DB |
| R12 | ORT dynamic-library discovery differs between the Python `onnxruntime` wheel and Rust `ort` | `NO_ORT_RUNTIME`, stale DLLs, GPU/CPU mismatch | share the resolver; honor `ORT_DYLIB_PATH`; reproduce the Windows stale-DLL refusal; test `[cpu]`/`[gpu]` in clean venvs |

---

## 9. Packaging, CI/CD, and distribution

**Crates:**

- `okfgraph` — one core library at the repository root, `MIT OR Apache-2.0`,
  MSRV pinned in `rust-toolchain.toml` (start at 1.85+ because of
  bobine/embroider; verify). Bins `okf` and `okf-mcp` live in `src/bin` behind
  the `cli`/`mcp` features.
- `okfgraph-py` — `bindings/python`, not published as a library; built by
  maturin.
- `okfgraph-pdf` — `addons/pdf`, the optional native plugin; published as a
  separate PyPI wheel and, if useful, to crates.io as a plugin crate.

**PyPI:**

- `pip install okfgraph` → slim wheel with `okfgraph._native` + pure-Python
  facade; no bobine.
- Extras: `cpu` (`onnxruntime==1.31.1`), `gpu`
  (`onnxruntime-gpu[cuda,cudnn]==1.31.1`), `pdf` (`okfgraph-pdf`), `dev`
  (pytest). `pip install "okfgraph[pdf]"` installs the add-on; the facade wires
  its plugin path before opening a router.
- `abi3-py312` wheels; `manylinux_2_28` x86_64, `macosx_11_0` arm64, `win_amd64`.
- `okf`/`okf-mcp` console scripts call native entry points. `cargo install
  okfgraph --features cli,server,mcp` gives the same binaries without Python
  (add `pdf` for static conversion).
- Wheel smoke test in a clean venv: import, `okf --version`, `okf init`, import a
  two-file bundle, search, MCP handshake; with `[cpu,pdf]`, convert one PDF.

**CI:**

- Fast Rust: fmt, clippy `-D warnings`, unit/integration, `cargo deny`.
- Python: build wheel, install, public compatibility suite, `pip check`.
- Heavy: weekly model/PDF/CUDA; differential oracle weekly and before release.
- Cross-open: Python-created DB → Rust → Python reopen on each OS.
- Benchmarks: criterion on ranking/chunking/settings; import/search on corpus.

---

## 10. Non-goals and deferred items

- WASM/browser build.
- Mobile targets.
- Redesigning the schema or `extra` storage during the port (a later v11 can
  replace Python-JSON strings, but not while the oracle must interoperate).
- Preserving Python internal modules/private helpers.
- New graph features, new model families, or new retrieval algorithms.
- Byte-identical CLI help prose or YAML export whitespace.
- Free-threaded Python builds (`3.13t`/`3.14t`) in the first release; `abi3`
  wheels target standard GIL builds. Add a separate free-threaded target only if
  there is demand.
- A Python fallback implementation. If the native extension fails to import, the
  package fails loudly (as today's missing native deps do).

---

## 11. Locked decisions and remaining implementation choices

**Locked** (see the table at the top): public API only; legacy v10 DBs; slim
wheel + PDF add-on; Windows x64/Linux x86_64/macOS arm64; Python 3.12–3.14;
staged A/B/C; one core crate with `src/` and `python/`; schema migration allowed
in Stage A; shared ORT for PDF; missing plugin is `CONVERSION_FAILED`.

Remaining choices, to settle during M0:

1. **PDF plugin ABI details:** exact function signatures, config JSON shape,
   progress/cancel hooks, and how the add-on version is pinned to the core. The
   recommended minimum is `abi_version`, `convert`, `free`; add progress only if
   the CLI job UI needs it in Stage C.
2. **Slim wheel size budget:** target ≤ 60 MB compressed. If the C++ Ladybug
   static lib pushes beyond that, evaluate `LBUG_SHARED` or a platform-specific
   dynamic link; do not drop vision to save size (embroidery code is small
   compared with bobine).
3. **`serve` job progress:** whether Stage C ports the Python progress callback
   shape exactly or maps it to job records only. The HTTP contract is fixed; the
   internal callback shape is not public.
4. **Python entry-point producers:** `okfgraph.producers` is a Python entry-point
   group. Plan: keep discovery in the Python binding, have the producer write a
   bundle/work dir, and call Rust `sync_from_bundle`; built-in producers stay in
   Rust. Confirm no other Python plugin seam is needed.

Resolved during M0 planning: Stage A may migrate schemas (`okf init` and
v1→v10 migrations are allowed; content writes are not), the PDF add-on shares
the `cpu`/`gpu` ORT runtime, and a missing/incompatible plugin is
`CONVERSION_FAILED` with an install hint.


---

## 12. First 10 working days (concrete)

1. Create branch `rust-core`; tag `python-oracle-0.15.0`; freeze feature work.
2. Scaffold the root `okfgraph` crate (`src/`), `bindings/python`, `python/`, and
   `addons/pdf`; add `rustfmt`, `clippy`, `cargo-deny`, `nextest`, `insta`,
   `proptest`.
3. Run the `lbug` spike on Windows x64 and Linux x86_64 (macOS arm64 before
   Stage A ships):
   - Python creates `fixtures/oracle-v10.db` with concepts, chunks, links, images,
     hashes, pins.
   - Rust opens it, dumps every table, loads both extensions, runs one vector and
     one FTS query, writes a concept, closes.
   - Python reopens the Rust-written copy and verifies the row.
4. Run the PyO3 + maturin spike; produce a wheel with `_native.version()`.
5. Run the rmcp stdio spike with the Python `mcp` client.
6. Write `docs/rust-compat.md` (the §5 checklist) and `parity/README.md`
   (scenario format, comparison rules).
7. Implement `pycompat` + fixtures first (JSON/extra/content_sig/YAML/time); this
   is the highest-leverage parity work and it unblocks every later phase.
   **Done for JSON/extra/content_sig/time:** `src/pycompat.rs` passes 17
   Python-generated golden cases from `parity/golden/pycompat.json`. PyYAML 1.1
   scalar resolution remains for the `markdown` work.
8. Record the go/no-go decision and update this plan with measured build times,
   wheel size, and any Ladybug deviations.

---

## 13. Definition of done

- `cargo add okfgraph` gives a usable core library; `cargo install okfgraph
  --features cli,server,mcp` gives `okf` and `okf-mcp` (add `pdf` for static
  PDF).
- `pip install okfgraph` installs one slim native wheel and a thin Python facade;
  `pip install "okfgraph[pdf]"` adds the native PDF add-on. No Python
  graph/embed/PDF/Markdown implementation is present.
- Public Python compatibility is exactly the documented surface: `OKFRouter`,
  `OKFClient`, `models`, `errors`, `settings` constants, and the console scripts.
  `okfgraph.components.*`/`okfgraph.ops.*` are intentionally gone.
- Existing v10 databases and bundles open, query, import, and export with parity;
  Python 0.15 can reopen a Rust-written DB.
- The parity suite (ops, DB dumps, exports, CLI, MCP, serve, Python public API)
  is green on all target platforms.
- Search/import performance is at least Python-level; startup is faster; memory
  is bounded and documented.
- The Python binding has no business logic: every graph/retrieval/import behavior
  is in Rust, and the binding only marshals, releases the GIL, and translates
  errors.
- The old Python core is deleted from the package; the oracle tag and parity
  report remain for audit.
