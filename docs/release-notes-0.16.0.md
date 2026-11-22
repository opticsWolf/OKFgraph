# OKFgraph 0.16.0 release notes

## Rust core cutover

0.16.0 replaces the Python implementation with a Rust core. The Python
package is a thin PyO3 facade over `OKFRouter`, `OKFClient`, models, errors,
settings constants and entry shims. `okfgraph.components.*` and
`okfgraph.ops.*` are removed. There is no Python fallback.

## Compatibility

- v10 databases and bundles remain compatible.
- Older or schema-less databases are refused with `CONFIG_INVALID`; Python
  0.15 is the one-time migration path.
- The public Python API, 30 ops, 29 CLI commands and 11 MCP tools are
  preserved.
- HTTP jobs are in-memory and non-durable.

## New packaging

- `okfgraph[cpu]` and `okfgraph[gpu]` select the ORT provider.
- `okfgraph[pdf]` installs the optional `okfgraph-pdf` native add-on.
- Supported targets: Windows x64, Linux x86_64, macOS arm64; Python 3.12–3.14.
- A missing PDF add-on returns `CONVERSION_FAILED` with an install hint.

## Surfaces

- `okf serve` exposes ops, jobs and `/mcp` with one owner per database.
- `OKFClient` is stdlib-only HTTP/1.1 over `POST /ops/<op>`, including UDS.
- Rust CLI and MCP proxy borrow a live owner through `<db>.serve.json`.

## Hardening

- Import is transactional; file hashes, frontmatter identity and resource
  paths are normalized.
- Doctor findings, safe fixes, scored summaries and portable info probes are
  implemented.
- External embed/rerank/vision lanes are fail-closed and covered by mock-host
  differential tests.
- Remote image fetch screens resolved IPs, pins one address, disables
  redirects and caps the body at 20 MiB.
- Extension setup loads both native files into a scratch DB before reporting
  success.
- CLI `INTERNAL` errors write redacted crash reports.
- Dependency policy is checked with `cargo deny`; see
  [`docs/license-review.md`](license-review.md).

## Performance

The paired CUDA benchmark meets the release gate:

| Metric | Rust / Python 0.15 |
| --- | ---: |
| Cold import | 0.70x |
| Warm embed | 0.68x |
| Warm search | 1.06x |
| Peak RSS | 0.88x |

See [`docs/performance-baseline.md`](performance-baseline.md) and
[`docs/performance-findings.md`](performance-findings.md).

## Verification

- 58 comparable differential scenarios green, one intentional missing-plugin
  divergence.
- 30/30 success ops, 21 typed-error ops, nine documented waivers.
- 82 all-feature Rust tests, 60 no-default tests, 9 fuzz properties.
- Wheel, facade, client, jobs, CLI, MCP, PDF and Python 3.12–3.14 smokes green.

See [`docs/release-gate.md`](release-gate.md) for the sign-off procedure and
`parity/out/` for recorded evidence.

## Upgrading

Read [`docs/migration-guide.md`](migration-guide.md) before switching.
