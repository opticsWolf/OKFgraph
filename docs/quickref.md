# Quick reference — OKFgraph 0.16.0

One-page reference for the Rust-backed release. For explanation, see
[`docs/rust-architecture.md`](rust-architecture.md) and
[`docs/migration-guide.md`](migration-guide.md).

## Install

```bash
pip install "okfgraph[cpu]"              # CPU ONNX Runtime
pip install "okfgraph[gpu]"              # NVIDIA GPU ONNX Runtime
pip install "okfgraph[cpu,pdf]"          # `pdf` is a no-op alias
pip install "okfgraph[gpu,pdf]"          # PDF/Office is bundled
```

Use one of `cpu` or `gpu`, never both. Supported: Windows x64, Linux x86_64,
macOS arm64; Python 3.12–3.14. One wheel carries the facade, the Rust core and
the native PDF/Office plugin; `pdf` remains an accepted alias.

## First graph

```bash
okf init --db-path graph.db --embedding-dim 512
okf import --all --bundle-root ./notes --db-path graph.db
okf search --db-path graph.db "how does retrieval work?"
okf export --output-dir ./export --flavor okf --db-path graph.db
```

```python
from okfgraph import OKFRouter

router = OKFRouter(db_path="graph.db", bundle_root="./notes")
router.import_bundle()
hits = router.search("how does retrieval work?", limit=5)
router.close()
```

## CLI verbs (29)

`init`, `model-info`, `import`, `ingest`, `search`, `read`, `traverse`,
`export`, `images`, `image`, `shell`, `broken-links`, `repair-links`,
`retitle-thoughts`, `diff`, `doctor`, `lint`, `produce`, `reindex`,
`reembed`, `deleted-list`, `deleted-recover`, `delete`, `deleted-purge`,
`detach`, `update`, `supersede`, `serve`, `setup`.

`okf --help` shows global settings flags; `--json` emits the canonical
`{ok, op, data, warnings, error}` envelope. Exit codes: `0` success, `2`
usage, `1` state/outcome error.

`okf ingest --kind pdf --pdf-path` takes one or more files, folders, or
`*?` patterns. Multiple matches are deduplicated and sorted; the result is a
per-file `data.files` report and the first failure is the envelope error.

## Router ops (30)

Read-oriented: `search`, `read`, `traverse`, `embed`, `doctor`, `pin_info`,
`lanes`, `list_broken_links`, `list_deleted`, `list_images`, `get_image`,
`get_by_id`, `list_directory`.

Write/ingest: `ingest`, `export_bundle`, `export_concept`, `import_bundle`,
`import_file`, `import_files`, `sync`, `update`, `supersede`, `delete`,
`repair_links`, `recover_deleted`, `purge_deleted`, `detach`, `diff`,
`reindex`, `reembed`.

`init` is CLI-only; it is not a router op.

## Serve and clients

```bash
okf serve --listen 127.0.0.1:8765                 # TCP
okf serve --listen unix:///tmp/okfgraph.sock     # POSIX UDS
```

```python
from okfgraph import OKFClient, discover, serve_file_for

url = discover("graph.db") or "http://127.0.0.1:8765"
client = OKFClient(url)
client.health()
client.search("retrieval", limit=5)
print(serve_file_for("graph.db"))
```

HTTP: `GET /health`, `GET /ops`, `POST /ops/<op>`, `/jobs`, `/mcp`.
Jobs are in-memory and non-durable, and only `reembed` is job-capable
(`JOB_OPS`). A reembed job claims the active slot before it is queued; every
HTTP op, job and MCP tool shares one owner lock. Discovery file:
`<db>.serve.json`.

```bash
okf-mcp                                   # stdio, direct DB
okf-mcp --connect http://127.0.0.1:8765   # proxy through owner
```

## Common settings

| Setting | Purpose |
| --- | --- |
| `db_path` | Ladybug DB path; default `.okfgraph/okfgraph.db`. `OKFRouter`/`okf` create the file and parent dirs; `open_read_only` refuses a missing file |
| `bundle_root` | Markdown root(s), `ALIAS=PATH` for multi-root. Defaults to `.` for the CLI and the DB parent for `serve`/MCP |
| `model_id`, `embedding_dim` | Text engine identity and width |
| `device`, `precision` | `cpu`/`cuda`, `auto`/`fp16`/`int8`/`fp32` |
| `enable_chunking`, `chunk_size`, `chunk_overlap` | Chunking contract |
| `mode` | Per-ingest image mode: `text`, `optional`, `omni` |
| `extension_dir` | Directory with `libvector`/`libfts` extensions |
| `allow_remote_images`, `allowed_image_domains` | SSRF opt-in and allowlist |
| `wal_mode` | Accepted for compatibility; no-op in 0.16.0 |

Config file: `okfgraph.toml` in the working directory or user config dir.
CLI flags override environment variables, which override TOML.

## Environment

| Variable | Purpose |
| --- | --- |
| `OKFGRAPH_DB`, `OKFGRAPH_DIM`, `OKFGRAPH_MODEL`, `OKFGRAPH_BUNDLE` | Common settings aliases |
| `OKFGRAPH_NO_SERVER=1` | Force CLI/MCP to open the DB locally |
| `OKFGRAPH_PDF_PLUGIN` | Override native PDF plugin path |
| `OKFGRAPH_NO_CHUNKING` | Disable chunking (inverted `OKFGRAPH_ENABLE_CHUNKING`) |
| `OKFGRAPH_WRITE_LOCK_TIMEOUT` | Test-only lock timeout override |
| `OKFGRAPH_PARITY_READ_ONLY=1` | Force read-only parity runner |
| `OKFGRAPH_JOB_TEST_DELAY_MS` | Smoke-only job delay hook |
| `OKFGRAPH_ORACLE_ROOT`, `OKFGRAPH_ORACLE_TAG` | Tag-oracle overrides |
| `ORT_DYLIB_PATH`, `OKFGRAPH_ORT_DLL_DIRS` | ONNX Runtime library selection |
| `OKF_CRASH_DIR` | Crash report directory |

## Error codes

Closed set shared with Python 0.15:

- Usage: `BAD_VALUE`, `MISSING_PARAM`, `FILE_NOT_FOUND`, `CONFIG_INVALID`
- State: `UNKNOWN_CONCEPT`, `UNKNOWN_ASSET`, `NOT_RECOVERABLE`, `DETACHED`,
  `DETACH_REFUSED`, `PURGE_REFUSED_ABSENT_ROOT`, `SEARCH_UNAVAILABLE`,
  `WRITE_LOCK_TIMEOUT`, `DB_LOCKED`, `DIM_MISMATCH`, `MODEL_PIN_MISMATCH`,
  `PRECISION_PIN_MISMATCH`, `SPACE_PIN_MISMATCH`, `IMAGE_PIN_MISMATCH`,
  `VISION_INCOMPATIBLE`, `EXTERNAL_UNAVAILABLE`, `CONVERSION_FAILED`,
  `NO_ORT_RUNTIME`, `EMBROIDER_TOO_OLD`, `REEMBED_IN_PROGRESS`, `INTERNAL`
- Outcome: `DOCTOR_FINDINGS`, `LINT_ERRORS`, `DIFF_DIFFERENT`

Unknown codes normalize to `INTERNAL`. Empty Python `fields` are `null`.

## Files and paths

| Path | Meaning |
| --- | --- |
| `<db>.lock` | Inter-process write lock; compatible with Python `fasteners` |
| `<db>.serve.json` | Live `okf serve` discovery record |
| `<root>/.okf-trash/<stamp>/` | Trashed source files (recoverable for 24 h) |
| `<root>/_assets/` | Bundled image assets |
| `OKF_CRASH_DIR` or platform state dir | Redacted crash reports (newest 20) |

## Verification shortcuts

```bash
cargo fmt --all -- --check
cargo deny check
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo test --workspace --all-features --lib
python parity/tools/run_matrix.py
python parity/tools/check_coverage.py
python parity/tools/check_perf_bench.py
```
