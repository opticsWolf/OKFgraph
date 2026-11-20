# Rust core compatibility checklist

This is the executable form of §5 of `docs/plan-rust-core.md`. Every unchecked
item is a release blocker for its stage. Status values: `open`, `wip`,
`covered`, `waived` (waivers require the parity report).

## A. Database (Stage A)

- [x] open and write **v10 only**; never bump the schema during the port.
      Older/missing schemas are refused with `CONFIG_INVALID`; Python 0.15
      remains the one-time migration path (BUG-028 disposition).
- [ ] same table/column names, primary keys, index names and recipes
- [ ] same `Meta`/`MetaText` pin keys and refusal semantics
- [ ] same write/index epoch dirty tracking; `reindex` force drops/recreates
- [ ] same single-writer discipline; `DB_LOCKED`/`WRITE_LOCK_TIMEOUT`
- [ ] Python 0.15 reopens Rust-written DBs
- [x] schema migration is **not** part of the Rust core: v10 compatibility is
      explicit, and stage A allowed no content writes.
- [x] `wal_mode` is accepted for 0.15 compatibility but is a documented no-op:
      `lbug 0.22.1` always journals through its WAL and exposes no separate
      journal mode to switch. The setting's help text remains unchanged so the
      generated settings table matches the oracle.

## B. Values and hashing (Stage A)

- [ ] Python-compatible JSON encoder: ensure_ascii=True, comma-space and colon-space separators, insertion order, default=str
- [ ] `extra_value`: raw string, ISO date/datetime, else Python-JSON
- [ ] `content_sig`: exact bytes and SHA-256; `default=str` datetime has a space separator
- [ ] `datetime.isoformat()`: naive UTC; fractional seconds only when present
- [ ] float formatting; no `NaN`/`Inf` at the API boundary
- [ ] Python `str()` of booleans/floats where stored as strings
- [ ] tests for non-ASCII, nested maps, floats, control chars, U+2028/U+2029

## C. Frontmatter and Markdown (Stage A/M2)

- [ ] `python-frontmatter` delimiter behavior and decoding
- [ ] PyYAML 1.1 scalar resolution, duplicate keys, dates/timestamps
- [ ] Pydantic model behavior: unknown keys, tags, `id`→`uid`, resource sanitization
- [ ] mordant block boundaries/types/byte offsets
- [ ] `_cap_chunk_budget` tiling and `+` continuation suffix
- [ ] lint diagnostics and `fix` behavior

## D. Search and ranking (Stage A/M2)

- [ ] `QUERY_VECTOR_INDEX`/`QUERY_FTS_INDEX` semantics, `efs`, `top := k`
- [ ] RRF constant/weights and id tie-breaking
- [ ] filters: `where`, tags, parent, `since`/`until`, `near_id`, `offset`, `recency`, `dedupe`, `link_kinds`
- [ ] chunk collapse ≥ 0.97, hub/PPR, cross-encoder windowing, token budget
- [ ] model-free PPR

## E. Output surfaces (Stage C)

- [ ] envelope field order `{ok, op, data, warnings, error}`
- [ ] error code/message/fields/remedy/data; exit 2 usage, 1 state/outcome
- [x] 11 MCP tool names, schemas, annotations and envelope content shape
      (`check_mcp.py`); `--connect` proxy (`check_mcp_connect.py`).
- [ ] 29 CLI commands, flags, `--json`, exit codes; help prose may change
- [x] `okf serve` routes `/health`, `/ops`, `/ops/{op}`, `/jobs*`, discovery
      file, and in-memory jobs with submit/poll/list/cancel (`check_jobs.py`).
      The HTTP `/mcp` mount and UDS transport remain open.

## F. Public Python API (Stage C)

- [ ] `from okfgraph import OKFRouter`
- [ ] `okfgraph.errors`: classes, codes, `envelope()`
- [ ] `okfgraph.models`: `ConceptModel`, `ChunkModel`, `ImageAssetModel`
- [x] `okfgraph.client`: `OKFClient`, `OPS`, `discover`, `serve_file_for`,
      including `reembed_job()` through the HTTP job lane (`check_client.py`,
      `check_jobs.py`).
- [ ] `okfgraph.settings` constants and table view
- [x] `okf`, `okf-mcp`, `-m okfgraph.cli`, `-m okfgraph.mcp_server` (all
      delegate to the Rust binaries/functions).
- [ ] extras-independent imports; no optional import in user code

## G. Approved breakages / changes

- [x] `okfgraph.components.*` and `okfgraph.ops.*` removed
- [x] private `cli.py`/`router.py` helpers removed
- [x] byte-identical help prose and YAML whitespace not required
- [x] missing/incompatible PDF plugin = `CONVERSION_FAILED` + install hint (was `INTERNAL`)
- [x] Python README's advanced component access intentionally gone
- [x] free-threaded Python builds out of scope for first release

## H. On-disk / native pins

- [x] `lbug = "=0.22.1"`
- [x] `embroider = "=0.15.1"`
- [x] `bobine = "=0.9.1"`
- [x] `mordant = "=0.10.1"`
