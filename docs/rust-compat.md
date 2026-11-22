# Rust core compatibility checklist

This is the executable form of §5 of `docs/plan-rust-core.md`. Every unchecked
item is a release blocker for its stage. Status values: `open`, `wip`,
`covered`, `waived` (waivers require the parity report).

**Status note (0.16.1):** sections A–D keep their M0–M3 planning granularity;
they are exercised end-to-end by the differential matrix, semantic DB dumps
and fuzz properties rather than by ticking each box. Sections E–J record the
binding contract for the shipped release.

Command and environment lookup: [`docs/quickref.md`](quickref.md).

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
- [x] 29 CLI verbs, generated global flags, `--json`, exit codes; help prose
      may change (`check_cli.py`, `cli::tests::command_tree_has_all_29_verbs`).
- [x] `okf serve` routes `/health`, `/ops`, `/ops/{op}`, `/jobs*`, discovery
      file, and in-memory jobs with submit/poll/list/cancel (`check_jobs.py`),
      plus the 11 MCP tools on `/mcp` (`check_mcp_http.py`). Job records use
      the Python field names/statuses and cooperative reembed cancellation.
- [x] Concurrent callers are serialised by one per-owner `op_lock`
      (`check_client.py`).
- [x] CLI/mcp proxy borrow a live owner via `server::client` discovery;
      `OKFGRAPH_NO_SERVER=1` opts out (`check_client.py`).
- [x] UDS client smoke: `check_uds.py` runs in the Linux/macOS wheel jobs and
      exercises the public `OKFClient` over a real Unix socket.

## E1. External lanes (BUG-013)

- [x] `backend='external'` speaks OpenAI-shaped `/v1/embeddings` with declared
      prefixes, identity probes, Matryoshka `dimensions`, and typed counters.
- [x] `rerank_backend='external'` speaks `/v1/rerank` with TEI fallback and
      session down-latching.
- [x] `image_backend='external'` requires an embroider vision identity and
      posts original bytes to `/v1/images/embeddings`.
- [x] Identity, bearer key, precision/space pins, and outage/refusal mapping
      are compared by `check_external_lanes.py`.

## F. Public Python API (Stage C)

- [x] `from okfgraph import OKFRouter` and `OKFClient` (`check_facade.py`)
- [x] `okfgraph.errors`: classes, codes, `envelope()`
- [x] `okfgraph.models`: `ConceptModel`, `ChunkModel`, `ImageAssetModel`
- [x] `okfgraph.client`: `OKFClient`, `OPS`, `discover`, `serve_file_for`,
      including `reembed_job()` through the HTTP job lane (`check_client.py`,
      `check_jobs.py`).
- [x] `okfgraph.settings` constants and 52-row table view (`check_facade.py`)
- [x] `okf`, `okf-mcp`, `-m okfgraph.cli`, `-m okfgraph.mcp_server` (all
      delegate to the Rust binaries/functions).
- [x] extras-independent imports; no optional import in user code
      (`check_wheel.py`, `check_packaging.py`).

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
- [x] `bobine = "=0.10.0"`
- [x] `mordant = "=0.11.0"` (explicit 0.16 engine set)

## I. M6 security hardening (Stage C)

- [x] remote image URLs are fetched only after resolving and screening every
      address; one safe address is pinned, redirects are disabled, and the
      response body is capped at 20 MiB (BUG-064).
- [x] a failed remote fetch still produces a caption-route `ImageAsset`, so
      Rust matches Python 0.15 and retries on the next import.
- [x] `okf setup extensions` loads both native files into a scratch Ladybug
      DB before reporting `ok` (BUG-065).
- [x] CLI `INTERNAL` errors write a redacted crash report named on the final
      error line and in `fields.crash_report` (BUG-066).
- [x] dependency audit policy is checked in as `deny.toml`; the generated
      license inventory is `docs/license-review.md`.

## J. Post-cutover contract (BUG-067–089)

- [x] `Router` adopts the stored schema width; `open` creates missing DBs and
      parent directories plus `:memory:` schemas, while `open_read_only`
      refuses missing files.
- [x] `Settings::from_map` validates and normalizes programmatic callers;
      `Settings::load` applies CLI > environment > TOML > defaults.
- [x] `roots.rs` validates aliases and resolves longest-prefix owners for
      import, materialize, update, delete and detach.
- [x] every write path refuses while detached; `rebuild=defer|force` is honored
      on reattach, and a per-call import `batch_size` is refused.
- [x] delete stages file and graph changes together and keeps trashed files
      recoverable for 24 hours.
- [x] `traverse(target=...)` validates ignored params and both endpoints, and
      supports all four relationship kinds in both directions.
- [x] search time bounds follow Python `_as_ts`; invalid bounds are
      `BAD_VALUE`.
- [x] frontmatter null/type errors are typed, failed files retry on the next
      import, and YAML anchors/aliases are rejected.
- [x] link extraction ignores code spans, decodes destinations and reports
      ambiguity.
- [x] discovery matches source extensions case-insensitively.
- [x] image titles and `<...>` sources do not change asset ids; non-outage
      vision errors fall back to caption rows; asset ids stay inside the
      store (intentional hardening divergence from Python 0.15).
- [x] BibTeX brace unwrapping/trimming and export See Also matching follow
      Python.
- [x] CLI PDF batches expand files, folders and globs with per-file reports.
- [x] `JOB_OPS = ["reembed"]` claims the active slot at submit; job views keep
      Python phase, elapsed and kwargs.
- [x] MCP HTTP tools acquire the owner `op_lock`, matching HTTP ops and jobs.
