# OKFgraph 0.16.0 release notes

> **Historical release notes.** These describe the 0.16.0 cutover. The current
> patch release is [`0.16.1`](release-notes-0.16.1.md).

## Summary

0.16.0 is the Rust-core cutover. The entire graph, retrieval, ingest, CLI,
HTTP and MCP implementation now lives in `src/`. Python is a thin public
facade over the native extension; there is no Python core and no Python
fallback. The Python 0.15 implementation is frozen at tag
`python-oracle-0.15.0` and is used only as the differential oracle.

Documentation starts at [`docs/README.md`](README.md). The one-page
[`docs/quickref.md`](quickref.md) lists every command, op, setting,
environment variable, error code and important path.

## Added

- Native Rust core implementing all 30 router ops.
- Full 29-command `okf` CLI with generated global settings flags and `--json`
  envelopes.
- 11 MCP tools over stdio, `--connect` proxying, and `okf serve /mcp`.
- `okf serve` HTTP ops, job lane and discovery file, with TCP and POSIX UDS
  transports.
- Public stdlib `OKFClient`, `discover()` and `serve_file_for()`.
- External embed/rerank/vision lanes behind the `external` feature, with
  fail-closed identity and typed outages.
- Bundled native PDF/Office plugin (`okfgraph/_plugin`) with a stable C ABI:
  one wheel, one PyPI project.
- Python 3.12, 3.13 and 3.14 wheels for Windows x64, Linux x86_64 and macOS
  arm64.
- Differential, fuzz, performance and security gates, including `cargo deny`.

## Changed

- Python API classes and method names are preserved, but all work is delegated
  to `okfgraph._native`.
- Engine pins now consult the loaded engine; `max_length` never counts as a
  space difference.
- Image import mode is per operation (`text`, `optional`, `omni`).
- `doctor` accepts `stale_days`, `fix`, `strict` and `cache_dir` and reports
  scored findings plus portable info probes.
- HTTP jobs are in-memory and non-durable. Terminal jobs are not restarted.
- Rust CLI and MCP proxy borrow a live owner through `<db>.serve.json` unless
  `OKFGRAPH_NO_SERVER=1`.
- `wal_mode` is accepted for compatibility but is a no-op: Ladybug 0.22.1
  always journals through WAL.
- Settings resolve as CLI > environment > `okfgraph.toml` > defaults; programmatic
  callers get strict validation, while CLI `--root` and CSV shapes are
  normalized centrally.
- `OKFRouter`/`okf` create a missing database and parent directory; the graph
  width is adopted from the stored schema.
- `okf ingest --kind pdf --pdf-path` takes multiple files, folders and globs
  and reports each file.
- Only `reembed` is job-capable; HTTP ops, jobs and MCP tools share one owner
  lock.

## Removed

- The root Python implementation, including `okfgraph.components.*` and
  `okfgraph.ops.*`.
- The root Python `pyproject.toml` and Python 3.11 support.
- The separate `okfgraph-pdf` distribution. The plugin now ships inside the
  `okfgraph` wheel; `pdf` remains a no-op install alias.
- Linux aarch64 and macOS x86_64 from the first release target matrix.

## Compatibility

- v10 databases and bundles remain compatible.
- Older or schema-less databases are refused with `CONFIG_INVALID`. Migrate
  once with Python 0.15, then reopen with Rust.
- Public Python API, settings constants, models, errors, console scripts and
  module entry shims are preserved.
- PDF/Office conversion is included in the wheel; a source build without the
  plugin returns `CONVERSION_FAILED` with the install hint
  `pip install "okfgraph[cpu,pdf]" or "okfgraph[gpu,pdf]"`.
- Loaded plugin conversion failures carry `fields.path` and
  `fields.converter`.

See [`docs/migration-guide.md`](migration-guide.md) and
[`docs/rust-compat.md`](rust-compat.md).

## Hardening

- Imports are transactional: encode before destructive delete, one write
  transaction per replacement.
- File hashes, frontmatter identity, resource paths, export paths and
  bundle roots are normalized.
- Doctor fix is lock-safe and skips reviewed sources.
- Remote image fetch resolves and screens addresses, pins one safe address,
  disables redirects and caps bodies at 20 MiB.
- Extension setup loads native files into a scratch Ladybug DB before
  reporting success.
- CLI `INTERNAL` errors write redacted crash reports named on the final error
  line and in `fields.crash_report`.
- `deny.toml`, `docs/license-review.md` and CI `cargo deny check` cover the
  dependency audit.

## Post-cutover fixes

The review after the cutover closed BUG-067–BUG-089. The shipped behavior
includes:

- settings precedence and strict validation, create-on-open, `:memory:`
  schemas and schema-width adoption;
- shared named-root resolution, graph-only writes, detach guards, staged
  delete and the 24-hour recovery window;
- path-mode `traverse`, `_as_ts` search bounds, Python-shaped frontmatter
  errors, code-span-aware link extraction and case-insensitive discovery;
- caption fallback for non-outage vision failures and confined
  `okf-asset://` ids;
- sparse BibTeX citations, See Also matching inside link destinations, and
  multi-path/folder/glob PDF ingest;
- reembed-only job claiming and owner-wide MCP serialization.

## Performance

The paired CUDA benchmark meets the release gate:

| Metric | Rust / Python 0.15 |
| --- | ---: |
| Cold import | 0.70x |
| Warm embed | 0.68x |
| Warm search | 1.06x |
| Unchanged re-import | 1.97x* |
| Peak RSS | 0.88x |
| Full process wall time | 0.60x |

\* Sub-10ms path; the gate applies a 5ms absolute tolerance.

See [`docs/performance-baseline.md`](performance-baseline.md) and
[`docs/performance-findings.md`](performance-findings.md).

## Verification

- 72 comparable differential scenarios green (73 reported); one intentional
  missing-plugin divergence.
- 30/30 success ops, 21 typed-error ops, nine documented waivers, 78 output
  files scanned.
- 111 all-feature Rust tests, 86 no-default tests, 9 fuzz properties.
- Wheel, facade, client, jobs, PDF, CLI, CLI PDF batches, external-lane, MCP
  and Python 3.12–3.14 smokes green on the manual workstation gate.
- Hosted CI covers format, clippy, no-default checks/tests, wheel build and
  smokes on Windows, Linux and macOS.

See [`docs/release-gate.md`](release-gate.md) for sign-off conditions and the
evidence record format.

## Upgrade checklist

1. Read [`docs/migration-guide.md`](migration-guide.md).
2. Migrate any pre-v10 database once with Python 0.15.
3. Choose exactly one ORT extra (`cpu` or `gpu`) and add `pdf` if needed.
4. Replace any imports from `okfgraph.components.*` or `okfgraph.ops.*`.
5. Run `okf doctor` and `okf reindex --force` if an older graph has stale
   index markers.
