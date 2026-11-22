# OKFgraph 0.16.0

OKFgraph is a Ladybug-backed knowledge graph with hybrid lexical/vector search,
chunking, image assets, ingestion, export, and a small public Python API.
Starting with 0.16.0 the implementation is **Rust**; Python is a thin PyO3
facade. There is no Python core in the wheel and no Python fallback.

## Status

- Rust core: `src/` — store, ops, embeddings, search, ingest, CLI, HTTP server,
  MCP, and diagnostics.
- Python facade: `python/okfgraph/` — `OKFRouter`, `OKFClient`, models, errors,
  settings constants, and entry shims only.
- PDF/Office add-on: `addons/pdf/` — optional native `okfgraph-pdf` wheel.
- Compatibility: v10 databases and bundles only. Python 0.15 remains the
  one-time migration path for older schemas.

## Installation

```bash
# CPU inference
pip install "okfgraph[cpu]"

# NVIDIA GPU inference
pip install "okfgraph[gpu]"

# Add PDF/Office conversion (optional)
pip install "okfgraph[cpu,pdf]"
# or
pip install "okfgraph[gpu,pdf]"
```

Supported first-release targets:

| Platform | Architecture |
| --- | --- |
| Windows x64 | x86_64 |
| Linux | x86_64 |
| macOS | arm64 |

Python 3.12, 3.13 and 3.14 are supported. Python 3.11, Linux aarch64 and
macOS x86_64 are not part of the first release. The slim wheel does not
contain `bobine`; the PDF add-on is installed only through the `pdf` extra.

If a native import fails, the package fails loudly. It never silently falls
back to a Python core.

## Quick start

```python
from okfgraph import OKFRouter

router = OKFRouter(
    db_path="graph.db",
    bundle_root="./notes",
    model_id="google/embeddinggemma-2",
    embedding_dim=512,
    device="cuda",
    precision="auto",
)

result = router.import_bundle()
print(result["added"], result["updated"])

for hit in router.search("how does retrieval work?", limit=5):
    print(hit["id"], hit["score"])

router.close()
```

The supported Python surface is documented in
[`docs/python-binding.md`](docs/python-binding.md). Internal modules
(`okfgraph.components.*`, `okfgraph.ops.*`) are intentionally gone.

## CLI

The native `okf` binary exposes the full 29-command oracle surface, including:

```bash
okf init --db-path graph.db --embedding-dim 512
okf import --all --bundle-root ./notes
okf search --db-path graph.db "retrieval augmented generation"
okf export --style okf --out-dir ./export
okf serve --listen 127.0.0.1:8765
okf setup extensions
```

`okf --help` lists the exact command tree. Commands accept the same global
settings flags as Python 0.15 and emit either human output or `--json`
envelopes.

## HTTP and MCP

Run one owner process per database:

```bash
okf serve --listen 127.0.0.1:8765
```

It exposes:

- `GET /health`, `GET /ops`, `POST /ops/<op>`
- `POST /jobs`, `GET /jobs`, `GET /jobs/<id>`, cancel
- `/mcp` with all 11 annotated MCP tools

The public `OKFClient` is stdlib-only:

```python
from okfgraph import OKFClient

client = OKFClient("http://127.0.0.1:8765", db="graph.db")
hits = client.search("retrieval", limit=5)
```

`okf-mcp` serves MCP over stdio, or proxies through a live owner with
`--connect http://127.0.0.1:8765`.

## Architecture

```
src/                     Rust core (all business logic)
  store/                 v10 Ladybug schema, shared read/write connection
  embed/                 native and external embed/rerank/vision lanes
  search/                hybrid RRF/PPR, filters, expansion, budget
  ingest/                import, delta, links, export, doctor, images
  ops/                   public 30-op router
  cli/, server/, mcp/    feature-gated adapters
  util/                  paths, hashing, security, crash reports
bindings/python/         PyO3 adapter, no business logic
python/okfgraph/         pure-Python public facade
addons/pdf/              optional native conversion plugin
parity/                  tag-based Python 0.15 oracle and differential gates
```

See [`docs/rust-architecture.md`](docs/rust-architecture.md) for the runtime
model, locking, and compatibility boundaries.

## Migration from 0.15

- v10 databases and bundles stay usable without conversion.
- Open a pre-v10 database once with Python 0.15 to migrate it, then use Rust.
- `okfgraph.components.*` and `okfgraph.ops.*` are not available; use
  `OKFRouter` methods or the CLI.
- Missing PDF add-on returns `CONVERSION_FAILED` with an install hint rather
  than Python 0.15's raw `INTERNAL`.
- Review [`docs/migration-guide.md`](docs/migration-guide.md) before cutover.

## Development

```bash
cargo fmt --all -- --check
cargo deny check
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo test --workspace --all-features --lib
cargo test -p okfgraph --no-default-features --test fuzz_robustness
python bindings/python/build_wheel.py --profile dev
python addons/pdf/build_wheel.py --profile dev
```

The differential matrix uses the frozen `python-oracle-0.15.0` git tag, not
the removed root Python package:

```bash
python parity/tools/run_matrix.py
python parity/tools/check_coverage.py
```

Hosted CI is a build/smoke gate. The full GPU parity suite and paired
performance benchmark are a documented manual release gate; see
[`docs/release-gate.md`](docs/release-gate.md).

## License

Apache-2.0 OR MIT. See [`LICENSE`](LICENSE), [`LICENSES`](LICENSES), and
[`docs/license-review.md`](docs/license-review.md).
