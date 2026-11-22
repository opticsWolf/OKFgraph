# Rust dependency license review

Generated from `cargo metadata --format-version 1` on `rust-core`.

No GPL-only, AGPL, SSPL or non-redistributable license is present. Several
crates are dual-licensed; `deny.toml` records the permissive expression
from their metadata.

| License expression | Crates |
| --- | ---: |
| `(Apache-2.0 OR MIT) AND BSD-3-Clause` | 1 |
| `(MIT OR Apache-2.0) AND NCSA` | 1 |
| `(MIT OR Apache-2.0) AND Unicode-3.0` | 4 |
| `0BSD OR MIT OR Apache-2.0` | 1 |
| `Apache-2.0` | 24 |
| `Apache-2.0 / MIT` | 1 |
| `Apache-2.0 AND ISC` | 1 |
| `Apache-2.0 OR BSL-1.0` | 1 |
| `Apache-2.0 OR BSL-1.0 OR MIT` | 2 |
| `Apache-2.0 OR GPL-2.0-only` | 1 |
| `Apache-2.0 OR ISC OR MIT` | 3 |
| `Apache-2.0 OR MIT` | 38 |
| `Apache-2.0 OR MIT OR Zlib` | 2 |
| `Apache-2.0 WITH LLVM-exception` | 1 |
| `Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT` | 6 |
| `Apache-2.0/MIT` | 2 |
| `BSD-2-Clause` | 6 |
| `BSD-2-Clause OR Apache-2.0 OR MIT` | 2 |
| `BSD-3-Clause` | 10 |
| `BSD-3-Clause AND MIT` | 1 |
| `BSD-3-Clause OR Apache-2.0` | 2 |
| `BSD-3-Clause/MIT` | 1 |
| `CC0-1.0 OR Apache-2.0` | 1 |
| `CC0-1.0 OR Apache-2.0 OR Apache-2.0 WITH LLVM-exception` | 1 |
| `CC0-1.0 OR MIT-0 OR Apache-2.0` | 3 |
| `CDLA-Permissive-2.0` | 2 |
| `ISC` | 5 |
| `ISC AND (Apache-2.0 OR ISC)` | 1 |
| `ISC AND (Apache-2.0 OR ISC) AND Apache-2.0 AND MIT AND BSD-3-Clause AND (Apache-2.0 OR ISC OR MIT) AND (Apache-2.0 OR ISC OR MIT-0)` | 1 |
| `MIT` | 143 |
| `MIT AND BSD-3-Clause` | 1 |
| `MIT OR Apache-2.0` | 324 |
| `MIT OR Apache-2.0 OR LGPL-2.1-or-later` | 2 |
| `MIT OR Apache-2.0 OR Zlib` | 4 |
| `MIT OR Zlib OR Apache-2.0` | 2 |
| `MIT/Apache-2.0` | 32 |
| `MPL-2.0` | 2 |
| `Unicode-3.0` | 18 |
| `Unlicense OR MIT` | 10 |
| `Unlicense OR MIT OR Apache-2.0 OR CC0-1.0` | 1 |
| `Unlicense/MIT` | 4 |
| `Zlib` | 7 |
| `Zlib OR Apache-2.0 OR MIT` | 8 |

Workspace crates: `okfgraph` (Apache-2.0 OR MIT), `okfgraph-py` and
`okfgraph-pdf` inherit the workspace license. The vendored
`esaxx-rs` patch changes only the CRT setting and is excepted explicitly.

Run `cargo deny check` in CI for the authoritative audit.
