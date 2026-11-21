# Performance findings — Rust core vs Python 0.15

## Current measured baseline

Generated on the local Windows CUDA workstation by
`parity/tools/check_perf_bench.py`, release Rust build, real
`google/embeddinggemma-2`, 8-concept `parity/corpus/filters` fixture.

| Metric | Rust | Python 0.15 | Rust / Python |
| --- | ---: | ---: | ---: |
| Cold import | 8.697s | 3.663s | 2.37x |
| Unchanged re-import | 0.994s | 0.007s | 148.52x |
| Warm embed median | 0.042s | 0.017s | 2.41x |
| Warm search median | 0.139s | 0.022s | 6.21x |
| Full process wall time | 12.467s | 5.130s | 2.43x |

**Conclusion: performance parity with Python is not met.** This blocks stable
0.16.0. The benchmark is now a release gate, and the current result is a
documented failure, not a waiver.

## What was already fixed during the investigation

1. **Pin checks were repeated on every query.** `embed` and `search` called
   `check_engine_pins` with a fresh `store.pin_info()`, which runs Ladybug
   count queries. Measured cost: ~0.67–0.76 s per call. The gate is now
   evaluated once per `Router`, and invalidated on writes/reembeds. This took
   warm embed from ~0.79 s to ~0.012–0.042 s.

2. **Hybrid search opened two search connections.** `vector_hits` and
   `fts_hits` each opened a read-only Ladybug DB and loaded the extensions.
   `Store::hybrid_hits` now runs both candidate queries on one connection.

3. **Search fetched too much row data.** `concept_rows` used
   `concepts_by_ids`, which returns `body` and `extra` for every candidate.
   `Store::concept_meta_by_ids` now returns only the fields search actually
   needs, loading `extra` only for `where` filters. This cut the row-assembly
   stage from ~68 ms to ~40 ms.

4. **Debug measurements were misleading.** The first Rust numbers came from
   a dev-profile example. The benchmark now builds Rust with `--release`.

## Remaining suspected bottlenecks

### 1. Store is still stateless; every query reopens Ladybug

`Store` holds only `path`, `settings` and `read_only`. `query()`, `scalar()`,
`with_search_conn()` and related helpers construct a fresh `Database` and
`Connection` per call. Python 0.15 holds a long-lived connection in
`OKFRouter`.

Evidence: after the fixes above, a warm Rust search still spends roughly:

- ~70–80 ms in `hybrid_hits` (DB open + extension load + vector/FTS queries)
- ~40 ms in `concept_meta_by_ids`

with no model work in that path (`search` with an explicit `query_vector` is
~0.13 s). This is the dominant remaining gap.

**Required fix:** keep a reusable read-only search connection with
vector/FTS extensions loaded once. The likely implementation is a dedicated
search-owner struct holding a `'static` Ladybug `Database`/`Connection`, or a
self-referential wrapper, behind `Arc<Mutex<...>>`. Reuse it for
`vector_hits`, `fts_hits` and `hybrid_hits`. Do not change query semantics.

### 2. Normal metadata queries also reopen the DB

`concept_meta_by_ids`, `counts`, `pin_info`, and most read ops use the same
fresh-connection pattern. Once the search connection is cached, extend the
same owner to normal read queries. Writes must keep using their own
transactional connection.

### 3. Delta re-import is still very slow

Unchanged re-import is ~0.99 s in Rust vs ~0.007 s in Python (148x). The
Rust path rescans/re-hashes more of the bundle than the Python delta path and
then opens several DB connections for hash/index comparisons. This needs a
separate profile of `FileHash`/`DirHash` lookup and file hashing before
changing semantics.

### 4. Embed is close but still ~2.4x slower

Warm embed is now ~0.042 s vs ~0.017 s. Most of the earlier 60x gap was the
pin-check bug. The remainder is likely per-op serde conversion plus ORT
session call overhead, but this should be re-measured after connection reuse
because the benchmark process also performs DB work.

## Performance parity gate

Stable 0.16.0 requires:

- warm search median Rust <= Python median * 1.10 (10% noise band)
- cold import Rust <= Python median * 1.10
- unchanged re-import queued = 0 and no worse than 1.25x Python
- warm embed median Rust <= Python median * 1.25

Current status: **FAIL**. No waiver is proposed.

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
