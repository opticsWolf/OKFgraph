# Migration guide — Python 0.15 to Rust 0.16.0

## Databases and bundles

- v10 databases and bundle trees are compatible. Rust opens and writes them
  without conversion.
- Older or schema-less databases are refused with `CONFIG_INVALID`. Migrate
  them once with Python 0.15, then reopen with 0.16.0.
- Do not run a Python 0.15 core and the Rust core in one process. Use separate
  processes; the `<db>.lock` protocol excludes writers across both.

## Python API

| 0.15 | 0.16.0 |
| --- | --- |
| `from okfgraph import OKFRouter` | unchanged |
| `OKFRouter(...)` constructor kwargs | unchanged |
| `router.<op>(...)` public methods | unchanged |
| `okfgraph.models`, `okfgraph.errors`, `okfgraph.settings` | unchanged public constants/classes |
| `okfgraph.components.*` | removed; use `OKFRouter` or `okf` |
| `okfgraph.ops.*` | removed; use `OKFRouter` or `okf` |
| Python fallback on native import failure | fails loudly instead |
| `doctor()` with no kwargs | `doctor(*, stale_days=365, fix=False, strict=False, cache_dir=None)` |

Errors keep the closed code set and envelope order
`{ok, op, data, warnings, error}`.

## CLI and MCP

- All 29 oracle CLI verbs are registered. Global settings flags are unchanged.
- JSON envelopes are byte-compatible except for documented host-specific
  doctor/model-info fields.
- `okf serve` owns a database and exposes HTTP ops, jobs and `/mcp`.
- Rust CLI and `okf-mcp` borrow a live owner unless `OKFGRAPH_NO_SERVER=1`.

## Packaging

- Install one ORT extra: `okfgraph[cpu]` XOR `okfgraph[gpu]`.
- PDF/Office conversion requires `okfgraph[cpu,pdf]` or
  `okfgraph[gpu,pdf]`; the slim wheel no longer bundles `bobine`.
- A missing plugin returns `CONVERSION_FAILED` with the install hint instead
  of Python 0.15's raw `INTERNAL`.
- Only Python 3.12–3.14 on Windows x64, Linux x86_64 and macOS arm64.

## Behavior changes

- Engine pins are checked against the loaded engine; `max_length` is ignored.
- External lanes pin only declared model ids; embroider-backed hosts also pin
  precision and space.
- Image import mode is per operation (`mode=text|optional|omni`).
- HTTP jobs are in-memory and non-durable.
- `wal_mode` is accepted for compatibility but is a no-op: `lbug 0.22.1`
  always journals through WAL.

## Verification before cutover

```bash
python parity/tools/run_matrix.py
python parity/tools/check_perf_bench.py
python parity/tools/check_coverage.py
python parity/tools/check_facade.py
python parity/tools/check_python_matrix.py
```

See [`docs/release-gate.md`](release-gate.md) for the full sign-off checklist.
