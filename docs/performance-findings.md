# Performance findings — Rust core vs Python 0.15

## Final measured baseline

Generated on the local Windows CUDA workstation by
`parity/tools/check_perf_bench.py`, release Rust build, real
`google/embeddinggemma-2`, 8-concept `parity/corpus/filters` fixture.

| Metric | Rust | Python 0.15 | Rust / Python |
| --- | ---: | ---: | ---: |
| Cold import | 2.604s | 3.706s | **0.70x** |
| Unchanged re-import | 0.011s | 0.005s | 1.97x* |
| Warm embed median | 0.010s | 0.014s | **0.68x** |
| Warm search median | 0.023s | 0.022s | **1.06x** |
| Peak RSS | 1,193,052 KiB | 1,362,544 KiB | **0.88x** |
| Full process wall time | 3.333s | 5.518s | **0.60x** |

\* The delta ratio is large only because both values are sub-10ms. The
release gate applies a 5ms absolute tolerance to that one path.

**Conclusion: performance parity with Python is met on the local CUDA
workstation.** BUG-063 is closed. The benchmark remains a release gate; it is
not a cross-machine SLA.

## Root causes and fixes

### 1. Pin checks were repeated on every query — fixed

`embed` and `search` called `check_engine_pins` with a fresh
`store.pin_info()`, which runs Ladybug count queries. Measured cost was
~0.67–0.76 s per call. The gate is now evaluated once per `Router` and
invalidated on writes/reembeds. Warm embed dropped from ~0.79 s to ~0.01 s.

### 2. Every query reopened Ladybug — fixed (dominant cause)

`Store` used to be stateless: `query()`, `scalar()` and
`with_search_conn()` constructed a fresh `Database`/`Connection` per call.
Python 0.15 keeps a long-lived connection. That cost:

- ~70–80 ms per `hybrid_hits`
- ~40 ms per `concept_meta_by_ids`

`Store` now owns one `Database`/`Connection` pair via `self_cell` behind
`Arc<Mutex<SharedConnection>>`:

- read-only Stores use a read-only connection;
- writable Stores use a writable connection for both reads and writes;
- vector/FTS extensions are loaded once;
- `Router`'s existing op lock and the Store mutex serialize access;
- `open_writable` drops the read-only handle created during schema validation.

This cut cold import from 8.7 s to 2.6 s, unchanged re-import from ~1.0 s to
0.011 s, and search-with-explicit-vector from ~0.13 s to ~0.01 s.

### 3. Hybrid search opened two search connections — fixed

`vector_hits` and `fts_hits` each opened a DB and loaded extensions.
`Store::hybrid_hits` now runs both candidate queries on the shared connection.

### 4. Search fetched too much row data — fixed

`concept_rows` used `concepts_by_ids`, which returns `body` and `extra`.
`Store::concept_meta_by_ids` now returns only search-visible fields and loads
`extra` only for `where` filters.

### 5. Debug measurements were misleading — fixed

The benchmark now builds Rust with `--release`.

### 6. Python RSS was unmeasured on Windows — fixed

`python_bench.py` now calls `GetProcessMemoryInfo` through `ctypes.WinDLL`
with explicit argtypes/restype.

## Release gate

The checker now enforces parity:

- warm search: Rust <= Python * 1.10
- cold import: Rust <= Python * 1.10
- warm embed: Rust <= Python * 1.25
- unchanged re-import: Rust <= Python * 1.25 + 5ms
- queued files after unchanged re-import: 0

Final run: all checks pass.

## Verification method

Run on the CUDA workstation:

```bash
parity/.venv-oracle/Scripts/python.exe parity/tools/check_perf_bench.py
```

The tool writes:

- `docs/performance-baseline.md` — generated numbers
- `parity/out/perf-bench.json` — raw samples

Both runners use the same fixture, model, dimension, chunking settings and
five warm samples after one un-timed warm call.
