# Python binding guide (0.16.0)

The Python package is a facade over the native Rust core. It deliberately
supports only the documented public surface. For commands, ops and environment
variables, see [`quickref.md`](quickref.md).

## Public imports

```python
from okfgraph import (
    OKFRouter,
    OKFClient,
    Concept,
    Chunk,
    ImageAsset,
    OKFError,
    SETTINGS,
    discover,
    serve_file_for,
)
```

Also available:

- `okfgraph.client.OPS`, `OKFClient`, `discover`, `serve_file_for`
- `okfgraph.errors` exception classes and `envelope()`
- `okfgraph.models` Pydantic models
- `okfgraph.settings` constants and a read-only settings view
- `okfgraph.cli.main`, `okfgraph.mcp_server.main` entry shims
- `okfgraph._pdf.bundled_plugin()` for the plugin shipped in the wheel

`okfgraph.components.*` and `okfgraph.ops.*` do not exist.

## Router

`OKFRouter` keeps the 0.15 constructor and method names. It is a thin
delegator to `okfgraph._native.NativeRouter`; no graph logic is implemented in
Python. Every documented operation is available as a method, including
`search`, `read`, `traverse`, `import_bundle`, `ingest`, `doctor`, `reembed`
and `sync`. The constructor creates a missing database and its parent
directory; a read-only open still refuses a missing file.

```python
router = OKFRouter(db_path="graph.db", device="cuda")
try:
    print(router.get_by_id("alpha").title)
finally:
    router.close()
```

Errors are raised as `OKFError` with stable `code`, `message`, `op`, `fields`,
`remedy` and `data`. Empty error `fields` are `None`, not `{}`.

## HTTP client

`OKFClient` uses only the standard library and speaks `POST /ops/<op>`:

```python
from okfgraph import OKFClient

client = OKFClient("http://127.0.0.1:8765", db="graph.db")
client.health()
client.search("retrieval", limit=5)
```

It also accepts `unix:///absolute/path` on POSIX, discovers an owner with
`discover(db)`, and exposes `jobs()`, `job(id)`, `cancel_job(id)` and
`reembed_job(...)`. Only `reembed` is job-capable, and jobs share the owner
lock with HTTP ops and MCP.

## Failure behavior

- Native extension import failure is fatal; there is no Python fallback.
- A build without the bundled plugin raises `CONVERSION_FAILED` with the
  install hint `pip install "okfgraph[cpu,pdf]" or "okfgraph[gpu,pdf]"`.
- Loaded converter failures carry `fields.path` and `fields.converter`.
- Console scripts resolve native CLI/MCP entry points through thin shims.

## Wheel layout

One wheel contains `okfgraph/`, `okfgraph/_native*.pyd|so`,
`okfgraph/_plugin/` (the native PDF/Office plugin), and the OpenSSL DLLs it
needs. `cpu`/`gpu` install ONNX Runtime; `pdf` is a no-op alias for 0.15
install commands. Importability is extras-independent.

## Verification

```bash
python parity/tools/check_facade.py
python parity/tools/check_wheel.py
python parity/tools/check_python_matrix.py
```

These check signatures, op delegation, typed errors, GIL release, wheel
metadata, console scripts, and Python 3.12/3.13/3.14 installs.
