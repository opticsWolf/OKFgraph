# Rust architecture (0.16.1)

This is the architecture of the shipped implementation. The older
`docs/architecture.md` is retained as historical Python 0.15 design evidence.
See [`docs/README.md`](README.md) for the full documentation map and
[`quickref.md`](quickref.md) for command/ops/environment lookup.

## Ownership

One `Router` owns one database. The HTTP server `okf serve`, the stdio MCP
server, the CLI and the Python facade either create that owner or borrow a
live one through the discovery file `<db>.serve.json`. A stale discovery file
is verified against `GET /health` before use.

All ops run through `Router::call`. The server serializes them with one
per-owner `op_lock`; writes additionally take a 300 s inter-process
`<db>.lock` compatible with Python `fasteners`. Cancellation stays lock-free
and cooperative. HTTP ops, reembed jobs and the MCP tools all pass through
that same `op_lock`; only `reembed` is accepted as a job, and it claims the
active reembed slot before the queued record is published.

## Store and connections

`Store` owns a shared `Database`/`Connection` pair through `self_cell` behind
`Arc<Mutex<SharedConnection>>`:

- read-only stores open the DB read-only;
- writable stores use one connection for reads and writes;
- vector/FTS extensions load once per connection;
- each `with_write_tx` uses the same connection and commits atomically.

This keeps Python 0.15 connection lifetime behavior without per-query DB
opens. Pin checks are cached per `Router` and invalidated on writes.

`Router::open` is the writable constructor: it creates a missing database and
its parent directory, and it initializes a `:memory:` schema. `open_read_only`
still refuses a missing file. The graph width comes from the stored schema
`FLOAT[N]`, never from `Meta.embedding_dim`.

## Data model

Schema v10 is the only supported schema. There are 13 node tables and 4
relationship tables. `Meta.embedding_dim` is never written by `okf init`; the
width is adopted from the schema `FLOAT[N]`. Older/missing schemas are refused
with `CONFIG_INVALID`.

## Search

`search` combines vector and FTS candidates with RRF, optional hub PPR,
chunk expansion, token budget, cross-encoder rerank, filters, and image
routes. Both candidate queries use one shared connection; row projection is
limited to search-visible fields. Text, chunk and image spaces are pinned and
verified against the loaded engine, excluding volatile `max_length`.

## Ingest

Import is delta-aware and transactional: vectors are encoded before any
destructive delete, then one write transaction replaces the concept, chunks,
links and ledger rows. File hashes are keyed root-relative after absolute
normalization. Remote images are fetched only after DNS/IP screening, with
redirects disabled and a 20 MiB body cap. Failed remote fetches keep a
caption-row asset for retry. A vision-route failure that is not a lane outage
falls back to the same caption row instead of aborting the import, and
`okf-asset://` ids are confined to one safe name inside the asset store.

## Roots, writes and lifecycle

`src/roots.rs` owns alias validation, longest-prefix root matching and
namespaced key resolution; import, materialize, update and delete all write
through the owning root. Every write path checks `detach::require_attached`,
and `rebuild=defer|force` is honored on reattach. Delete stages the file move
and graph transaction so a failure rolls both back; trashed files stay
recoverable for 24 hours before purge.

## Bundled PDF plugin

The `okfgraph` wheel contains `okfgraph/_plugin/` with the native converter
(`addons/pdf`, `bobine` linked in). The plugin exports the stable C ABI
`okf_pdf_abi_version`, `okf_pdf_convert`, `okf_pdf_free`; the facade points
`OKFGRAPH_PDF_PLUGIN` at it before the first router call. A wheel or source
build without the plugin returns `CONVERSION_FAILED` with the install hint; a
loaded plugin failure carries `fields.path` and `fields.converter`.

## Security and diagnostics

`util::security` paths and SSRF rules, `util::extensions` scratch-DB load
verification, and `util::crash` redacted crash reports are part of the core.
`doctor` reports scored findings and portable `info` probes; host-specific
probes are normalized only by name in the parity gate.

## Python boundary

`bindings/python` is a PyO3 adapter with no business logic. The facade in
`python/okfgraph/` exposes only the documented public API. Native errors map
to Python exception classes through the same closed code set, and empty
`fields` serialize as `null`.

## Parity

`parity/` resolves the frozen Python 0.15 oracle from the
`python-oracle-0.15.0` git tag into a cached detached worktree. All parity
tools import from that worktree, so the old root package is absent from the
default tree. The full GPU matrix and paired performance benchmark are manual
release gates; hosted CI builds and smokes every target.
