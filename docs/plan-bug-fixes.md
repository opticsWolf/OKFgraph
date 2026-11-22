# OKFgraph Rust Core — Bug Findings and Fix Plan

**Status:** review plan from `6a71fec`, updated after the complete M6 cutover
and the post-cutover review at `a62d1c9`. All code findings through BUG-066 are
closed. The post-cutover review added **BUG-067–BUG-089**, all open in section
8A; section 11 is the authoritative register and records the one intentional
divergence and the ongoing gate obligations. The original review below is kept
as the evidence trail.

**Scope:** the Rust core, the PyO3 binding, the Python facade, the optional
PDF plugin, and the parity harness. Stage C surfaces (`okf serve`, MCP) are
in scope where a finding or gate references them.

**Implementation status (rust-core branch, after the fix series):** the P0
crash/data-loss paths, Phase 1 import integrity, Phase 2 identity/resource/
path safety, most Phase 3 semantic parity, multi-root/prune, ingest
materialize/folder, BibTeX/macros/lint, pins/lanes, schema refusal, read-only
enforcement, op vocabulary, YAML/time, export filtering, remote-image opt-in,
and the write lock are implemented. The M4 serve/MCP/client slice is also in:
11 rmcp tools, `--connect`, the public `OKFClient`, the HTTP job lane, the
full 29-command CLI, and typed conversion-error envelopes. The broad differential matrix is green for **58 comparable scenarios**
after normalizing wall-clock graph times/internal IDs and the documented
intentional differences; the 59th, `ingest-pdf-no-plugin`, records the
intentional missing-plugin error divergence instead of a green comparison.
Current verification is listed under “Verification status” below and the
authoritative register in section 11. The post-cutover review at `a62d1c9`
found additional API, settings, named-root, detach, parser and surface
regressions; they are listed in section 8A and are not yet exercised by the
58-scenario matrix.

**Important:** the original review below describes the pre-fix state. It is
kept as the evidence trail; the implementation status section records what is
now closed and what remains. The existing parity matrix was green before the
fixes but only covered a narrow fixture set. “M3 complete” should not be
claimed until the remaining gate items close.

### Verification status

Final verification on `rust-core` through the M6 security hardening round:

- `parity/tools/run_matrix.py --pdf-plugin target/debug/okfgraph_pdf.dll`:
  `TOTAL:56 FAILS:0` in **14m37s** on the CUDA dev box. The new
  `frontmatter-cr-only` scenario followed the fuzz fix with
  `TOTAL:57 FAILS:0` in **15m55s**. After M6 security hardening the expanded
  59-scenario suite was rerun scenario-by-scenario: **58/58 green**, with
  `ingest-pdf-no-plugin` the intentional divergence. The single full-suite
  run was interrupted by workstation standby; the chunked rerun is the
  recorded evidence. The same 58/58 result was repeated after the root Python
  package was deleted for the stable cutover.
- `parity/tools/check_coverage.py`: 30/30 success ops, 21 typed-error ops,
  nine documented waivers, 63 output files scanned.
- `parity/tools/check_external_lanes.py`: green against four local mock hosts,
  including TEI rerank fallback and a generated valid GGUF counter fixture.
- Paired performance gate is green: import 0.70x Python, embed 0.68x,
  search 1.06x, RSS 0.88x (`docs/performance-baseline.md`).
- M6 security hardening is green: SSRF unit tests plus
  `images-ssrf-private`, scratch-DB extension verification, redacted crash
  reports, `deny.toml` and `docs/license-review.md`.
- Fuzz properties cover frontmatter/YAML, chunk offsets, safe ids/paths,
  wikilink scanning/resolution, op params and closed error codes; the
  generated regression seed is checked in.
- The same full matrix, `check_cli.py` and `check_external_lanes.py` also pass
  when the root `okfgraph/` package was hidden and, after cutover, with that
  package deleted: all oracle imports come from a cached worktree of
  `python-oracle-0.15.0`.
- Rust suites: 82 all-feature tests, 60 no-default tests and 9 fuzz
  properties pass; Clippy is clean with all features and with no default
  features.
- The full six-job CI matrix was green on `44899657898` for code commit
  `333f8df`. The final commit `66df990` changes only docs and the parity
  harness, and its CI run was stopped by the operator before completion.

Closed with code + tests/scenarios (details in the individual findings):

- BUG-001/002 crash safety; BUG-003/004/005 import integrity; BUG-006/007
  identity/resource safety; BUG-008 dimension adoption; BUG-009 delete/trash/
  restore; BUG-010/011 image pruning/counts; BUG-012 path safety.
- BUG-013 is closed: the external embed, rerank and vision lanes speak the
  oracle's HTTP wires behind `external`; `check_external_lanes.py` compares
  both runners against three mock hosts.
- BUG-014/015 pins/lanes; BUG-016/017/018/019 ingest parameters/materialize/
  lint/thoughts; BUG-022/023/024/025 time, YAML, export paths.
- BUG-020 multi-root; BUG-021 prune/tombstones/absent-root refusal; BUG-033
  BibTeX macros/latin-1/slug collisions; BUG-035/036/037/038/039/041/042
  (including root-relative `FileHash` keys when the bundle root is configured
  as a relative path).
- BUG-026/027 facade signatures, op methods, and native error `op`/`data`;
  BUG-031 index errors; BUG-032 walk/symlink/hidden-file handling.
- BUG-028 is **closed by explicit disposition**: compatibility is v10-only.
  Old/missing schemas are refused with `CONFIG_INVALID`, `okf init` no longer
  stamps an existing DB, and `docs/rust-compat.md` §A states that Python 0.15
  remains the one-time migration path. The v1→v10 registry is not ported.
- BUG-029 in-process mutex + `fd-lock`, read-only enforcement, and the
  cross-process Python↔Rust lock check (`parity/tools/check_lock_interop.py`)
  plus read-only check (`check_read_only.py`).
- BUG-034 real one-page PDF and DOCX conversion/import parity
  (`ingest-pdf-real`, `ingest-office-real`), warm converter reuse under
  `pdf_idle_seconds`, plugin config/lint/temp cleanup/discovery/cache.
- BUG-016 missing-param/`rebuild` ingest envelopes (`ingest-params`);
  BUG-007 resource sanitization (`import-resource-sanitize`); BUG-014
  model/dim/precision/space pin unit tests plus empty-graph re-pin.
- BUG-040 is closed: whole-concept transactions/atomicity plus
  `examples/import_bench.rs` and `parity/tools/check_import_bench.py`. On the
  Windows x64/CUDA dev box the filters corpus imports 8 concepts in ~10.3 s;
  an unchanged re-import takes ~1.1 s and queues zero files.
- BUG-043 documentation alignment; BUG-044 is **partial**:
  `parity/tools/check_coverage.py` proves every public op appears in a
  scenario, but the required per-op success/error matrix is open.
- Python facade/wheel smoke checks (`check_facade.py`, `check_wheel.py`) run
  against a freshly built wheel.

Deliberately partial / remaining before stable:

- BUG-030: `extension_dir` and the warm PDF converter/idle close are used.
  `wal_mode` is accepted but lbug always journals through its WAL (upstream
  has no separate mode); the compatibility-only behavior is now explicitly
  documented in `docs/rust-compat.md` §A.
- BUG-034 is covered end-to-end for PDF, DOCX, XLSX and PPTX through the same
  bobine converter path. `ingest-office-formats` converts and imports a valid
  minimal PPTX in both runners. The earlier PPTX refusal came from a malformed
  fixture whose root relationship named `xl/workbook.xml`, not from missing
  PPTX support in bobine 0.9.1; `check_fixtures.py` now guards that invariant.
- BUG-045 is closed: `doctor` now accepts `stale_days`/`fix`/`strict`/
  `cache_dir`, returns `DOCTOR_FINDINGS`, and the safe fix repairs links and
  orphan delta rows; `doctor-params` covers it.
- BUG-046 is closed: portable `doctor` info probes (index inventory,
  missing-index findings, stale/omni image routes, stale write lock, lane
  identity) are ported and compared with only host-specific rules ignored.
- `doctor` now ports the scored findings (broken/ambiguous links, orphan
  hashes, duplicate titles, orphan/missing-description/stale concepts),
  `summary`/`score`, `detached`, per-root status, and the portable `info`
  probes (index inventory, image routes, stale lock, lane identity). Only
  host-specific `ort_*`, `converter_cache`, `cache_hygiene`, `pdf_converter`,
  and `db_created` entries are ignored by `compare.py --ignore-info-rule`.
- Known intentional parity divergence: Python 0.15 returns a raw `INTERNAL`
  “bobine required” error when the PDF add-on is absent; Rust returns
  `CONVERSION_FAILED` with the documented install hint.
- Stable comparison normalizes Python set-ordered `tags`; materialized
  thought files therefore compare by semantic fields/DB rows rather than raw
  `FileHash.hash` when wall-clock `created` differs.
- PDF auto-import paths are not byte-stable: both runners use a fresh temp
  directory and the result exposes it, so `md_path`/`image_dir` are ignored;
  the primary output-dir tree and DB dump are compared byte/semantic-for-byte.
- BUG-050 is closed: a loaded plugin's real conversion failure now carries
  Python 0.15's `fields`/`remedy` instead of the missing-add-on install hint;
  `ingest-office-formats` covers valid DOCX/XLSX/PPTX conversion plus a
  deliberately malformed PPTX in `parity/corpus/invalid/`.
- M4 surface checks outside the scenario counter: `check_client.py`,
  `check_mcp.py`, `check_mcp_connect.py`, `check_mcp_http.py`,
  `check_uds.py`, `check_jobs.py`, `check_cli.py`, `check_packaging.py`,
  `check_pdf_wheel.py`, `check_python_matrix.py`,
  `check_lock_interop.py`,
  `check_read_only.py`, `check_facade.py`, and `check_wheel.py`.

P0 and P1 items below are closed and covered by differential tests.

---

## 1. Severity and completion rules

| Level | Meaning | Release rule |
| --- | --- | --- |
| **P0** | Process crash, data loss, secret leak, path traversal, or silent graph corruption | Block all releases and the beta cutover |
| **P1** | Documented public behavior is wrong or a documented M3 feature is missing | Block the stable cutover; beta may ship only with an explicit exception list |
| **P2** | API, storage, packaging, or performance defect with a workaround | Fix before stable; track in the beta notes |
| **P3** | Documentation, diagnostics, or low-risk edge case | Fix opportunistically |

A finding is “done” only when:

1. the code path is fixed,
2. a regression test exists (unit or differential),
3. the relevant parity scenario is green, and
4. the plan/README no longer claims the old behavior.

---

## 2. Executive summary

### Confirmed P0 issues

- Unknown error codes (`IO_ERROR`, `STATE_INVALID`) hit an `assert!` and
  abort the process instead of returning an error.
- A non-ASCII percent escape in an image reference panics during import.
- Re-import deletes the old concept before encoding the new vectors and is
  not transactional; a failure leaves a partially written or missing graph.
- Thoughts/BibTeX ingest fails with `INTERNAL` whenever `bundle_root` is set.
- `identity="frontmatter"` is ignored on import, and `resource:` credentials
  are stored verbatim.
- Non-default embedding dimensions are not adopted; a 256-dim graph opens as
  512.
- `delete` does not implement the file-backed refusal/trash/restore contract.
- Image assets are never deleted because `count(*)` is always 1 after an
  `OPTIONAL MATCH`; per-concept markdown `image_count` is graph-wide.
- Export/update join raw concept IDs into filesystem paths without validation.

### Confirmed P1 issues

- External embed/rerank/vision backends are accepted by settings but never
  implemented.
- Precision/space pins are not enforced; `lanes` reports fabricated values.
- `ingest` rejects parameters Python accepts and ignores `materialize`,
  `folder`, `extra`, and lint reports.
- Thoughts title/slug/related semantics diverge.
- Multi-root, `prune_missing`, `force`, and delta tombstones are missing.
- Microsecond formatting, timezone conversion, and local-vs-UTC clocks differ.
- The YAML parser/dumper is incomplete and can silently drop data.
- Export filtering, `.md` path construction, and filtered index generation
  are wrong.
- The Python facade is not API-compatible (no positional `db_path`, no write
  methods, error `data`/`op` lost).
- Schema migrations and single-writer locking are absent.
- Symlink/hidden-file walking, index error handling, and PDF conversion
  lifecycle need work.

### Post-cutover additions (open)

The `a62d1c9` review found 23 additional open findings, detailed in section
8A. The highest-impact clusters are: named-root path/import resolution
(BUG-072), detached-graph write enforcement (BUG-074), delete atomicity
(BUG-075), settings/env/default-root wiring (BUG-069), schema-width adoption
(BUG-067), fresh-database creation (BUG-068), and the `okf-asset://` local
file disclosure (BUG-084). The existing matrix does not exercise any of them.

### Why the green matrix misses this

The scenarios currently use:

- default path identity,
- no `bundle_root` for `ingest`,
- no file-backed `delete`,
- no non-default dimension,
- no external backend,
- no block-scalar YAML,
- no long or duplicate thoughts,
- no image removal,
- no filtered export.

Section 7 adds the missing scenarios.

---

## 3. Implementation order at a glance

Do not start with broad refactors. Land the crash fixes first, then data
integrity, then security, then behavior parity, then the missing features,
then API/storage hardening.

| Phase | Work package | Findings | Size | Depends on |
| --- | --- | --- | --- | --- |
| 0 | Crash/process safety | BUG-001, BUG-002 | S | — |
| 1 | Import data integrity | BUG-003, BUG-004, BUG-005, BUG-008, BUG-009, BUG-010, BUG-011 | L | Phase 0 |
| 2 | Security/privacy | BUG-006, BUG-007, BUG-012, BUG-041 | M | Phase 1 |
| 3 | Core semantic parity | BUG-014, BUG-015, BUG-016, BUG-017, BUG-018, BUG-019, BUG-022, BUG-023, BUG-024, BUG-025 | XL | Phase 1–2 |
| 4 | Missing documented features | BUG-013, BUG-020, BUG-021, BUG-033, BUG-034, BUG-035, BUG-036, BUG-037 | XL | Phase 3 |
| 5 | API, storage, packaging | BUG-026, BUG-027, BUG-028, BUG-029, BUG-030, BUG-031, BUG-032, BUG-038, BUG-039, BUG-040, BUG-042, BUG-043, BUG-045 | L | Phases 1–4 |
| 6 | Verification and release gates | BUG-044, BUG-046, BUG-047, BUG-048, BUG-049 | M | All |
| 7 | Post-cutover review | BUG-067–BUG-089 | XL | All prior phases |

“Size” is S (< half day), M (1–3 days), L (3–5 days), XL (1–2 weeks) for one
experienced Rust developer; split each work package into reviewable commits.

---

## 4. P0 — Crash, data loss, and security

### BUG-001 — Unknown error codes panic the process

**Severity:** P0, S

**Where:**

- `src/error.rs:85` (`assert!` in `OkfError::new`)
- `src/error.rs:136` (`expect` in `OkfError::kind`)
- `src/export/mod.rs:469`, `src/export/mod.rs:481`
- `src/ops/mod.rs:372`, `src/ops/mod.rs:417`
- `src/converters.rs:420`
- `src/ingest/mod.rs:493`
- `src/purge.rs:67`

**Evidence:** `OkfError::new("IO_ERROR", …)` and
`OkfError::new("STATE_INVALID", …)` are used, but neither code is in
`USAGE_CODES`, `STATE_CODES`, or `OUTCOME_CODES`. The constructor asserts and
panics. A non-UTF-8 BibTeX file, an unwritable export directory, an
unreadable source file, a partially initialized database, or a corrupt
recovery snapshot can trigger it.

**Impact:** process abort (CLI) or `PanicException` (PyO3), not a typed
`OKFError`. It also bypasses the documented `CONVERSION_FAILED`/`INTERNAL`
behavior.

**Fix:**

1. Do not assert on runtime error construction. Add a fallible constructor or
   normalize unknown codes to `INTERNAL` with a `type` field.
2. Map filesystem failures to `INTERNAL` (or `FILE_NOT_FOUND` where
   appropriate) with a clear remedy.
3. Map “no v10 schema” to `INTERNAL` with `remedy="run okf init"` (or
   `CONFIG_INVALID`), not `STATE_INVALID`.
4. Map corrupt snapshots to `INTERNAL`/`NOT_RECOVERABLE`.
5. Keep a debug assertion/test that all literal codes are known; add a test
   that every code emitted by the crate is in the closed set.

**Tests:** force an export write failure, an unreadable update source, a
non-UTF-8 `.bib`, a missing schema row, and a corrupt snapshot; assert a typed
error envelope and no panic.

---

### BUG-002 — Non-ASCII percent escapes panic image ingestion

**Severity:** P0, S

**Where:** `src/image_ingest.rs:45-63` (`percent_decode`), used by
`filename_from_src` and `load_bytes`.

**Evidence:** `&text[i + 1..i + 3]` can slice inside a multi-byte character.
Confirmed with `![x](%€)`:

```text
thread 'main' panicked at src\image_ingest.rs:51:28:
end byte index 3 is not a char boundary; it is inside '€' (bytes 1..4 of string)
```

The same path is reachable from `data:` URIs that are not base64.

**Impact:** a malformed bundle can crash import; no graph write should happen
before validation.

**Fix:** decode on `&[u8]`; accept exactly two ASCII hex digits; append raw
bytes and convert with `String::from_utf8_lossy` only at the end. Do not slice
`str` at computed byte offsets unless `is_char_boundary` is checked.

**Tests:** `%€`, `%é`, truncated `%4`, `%GG`, valid `%20`, UTF-8 percent
payloads, and a data URI with the same cases.

---

### BUG-003 — Re-import is destructive and non-atomic

**Severity:** P0, L

**Where:** `src/ingest/mod.rs:732-797` (`import_one`), `:296-340`
(`commit_parsed`), `src/store/mod.rs:117-137` (`with_write_conn`/`write`).

**Evidence:** `import_one` runs:

```rust
MATCH (ch:Chunk …) DETACH DELETE ch
MATCH (c:Concept …) DETACH DELETE c
```

before `engine.encode_many(...)` and `engine.encode(...)`. Every `Store::write`
opens a new connection and commits separately. If embedding fails, the old
concept/chunks are gone; if a later write fails, the graph is partially
written. Incoming links are captured before the loop but restored only after
all concepts succeed.

**Impact:** GPU OOM, model-load failure, dimension mismatch, cancellation, or
a disk error can erase a concept and leave orphan chunks. This is the most
serious data-loss path in the port.

**Fix:**

1. Read the previous row (`created_at`, `updated_at`, `content_sig`) before
   deletion.
2. Encode all chunk and concept vectors before deleting anything, or use a
   single `with_write_conn` transaction for delete + chunk + concept writes.
3. If `lbug` transactions are available, wrap the per-concept replace and the
   batch upsert; otherwise stage all writes and perform the destructive step
   last.
4. Restore incoming links in the same transaction as the concept creation.
5. Add a rollback/partial-state test using a fake embedder that fails on the
   second concept.

**Tests:** inject an encode failure after the first concept; assert the old
concept, chunks, links, and timestamps are unchanged. Re-run the same import
twice and compare `content_sig`/`updated_at`.

---

### BUG-004 — `created_at`/`updated_at` are reset on every re-import

**Severity:** P0, M

**Where:** `src/ingest/mod.rs:829-870` (`write_concept`).

**Evidence:** `write_concept` always sets:

```rust
("created_at", timestamp_value(&now)),
("updated_at", timestamp_value(&now)),
```

Python’s `stamp()` keeps `created_at` from the previous row and only advances
`updated_at` when `content_sig` changed. `effective_at` falls back to
`updated_at` when there is no authored date.

**Impact:** re-importing unchanged content changes graph provenance and
recency/`since`/`until` results. `created_at` is lost permanently. The parity
comparator ignores these fields, so the matrix cannot see it.

**Fix:** port `stamp()`:

- `created_at = prev.created_at or now`
- `updated_at = prev.updated_at` unless `prev.content_sig != new_sig`
- `effective_at = authored_time or updated_at`
- write all four fields together with the concept.

Add a pure unit test for the stamp matrix (new, unchanged, changed, authored
date, timezone-aware authored date).

---

### BUG-005 — Thoughts/BibTeX ingest fails with `bundle_root`

**Severity:** P0, M

**Where:** `src/converters.rs:172-230` (`commit`, `temp_markdown`,
`parse_markdown`), `src/ingest/mod.rs:891-910` (`record_file_hash`).

**Evidence:** `parse_markdown` writes a temp file, parses it, and deletes it.
`commit` then calls `commit_parsed` with `root=None`; `import_one` calls
`record_file_hash`, which sees `settings.bundle_root`, tries to hash the
deleted temp path, and returns `INTERNAL`. Confirmed with a thoughts ingest
and `bundle_root` set:

```json
{
  "ok": false,
  "error": {
    "code": "INTERNAL",
    "message": "Das System kann die angegebene Datei nicht finden. (os error 2)",
    "fields": { "type": "std::io::Error" }
  }
}
```

The concept/chunks are written before the hash step, so this also leaves a
partial graph. The temp file is leaked if parsing fails.

**Impact:** a standard `okf` configuration cannot use `ingest(kind='thoughts')`
or `ingest(kind='bib')`.

**Fix:**

1. Add `Parsed::synthetic: bool` (or `source_path: Option<PathBuf>`) and skip
   `record_file_hash` for synthesized concepts.
2. Alternatively parse markdown from an in-memory `Frontmatter` value rather
   than a temp file.
3. Ensure temp files are removed on both success and error (`scopeguard` or a
   small RAII wrapper).
4. Make `commit` atomic with BUG-003.

**Tests:** thoughts and BibTeX ingest with `bundle_root` set; assert success,
no `FileHash` row for the synthetic concept, and no temp file left behind.

---

### BUG-006 — `identity="frontmatter"` is ignored on import

**Severity:** P0, M

**Where:** `src/ingest/mod.rs:540-551` (`relative_id`/`parse_file`);
`settings.identity` is only read in `src/diagnostics.rs:71,482` and
`src/ops/mod.rs:401`.

**Evidence:** `parse_file` always derives the ID from the path. Python
`parse_source_file` (`okfgraph/components/import_.py:182-245`) implements:

- frontmatter `id:` becomes the concept ID under `identity="frontmatter"`,
- unsafe IDs (`..`, backslashes, leading `@`/`/`) are a parse error,
- `uid` keeps the written id,
- `source_path` records the bundle-relative path.

Rust imports `id: custom` as the path ID. `diagnostics::parse_dir` manually
remaps `uid` to `item.id`, so `diff`/`lint` and import disagree.

**Impact:** frontmatter-identity graphs silently use wrong IDs and break
moved-file link preservation.

**Fix:** pass `settings.identity` into `parse_file`; port the safe-ID regex
and the `uid`/`source_path` handling; make `diagnostics` use the same parser
instead of remapping after the fact.

**Tests:** a file at `dir/a.md` with `id: custom` imports as `custom` in
frontmatter mode; a move preserves links; unsafe IDs return `BAD_VALUE` with
the Python fields.

---

### BUG-007 — `resource:` credentials are stored verbatim

**Severity:** P0, S

**Where:** `src/ingest/mod.rs:579` (`resource = string_field(...)`).

**Evidence:** Python `sanitize_resource` (`components/import_.py:146-160`)
rewrites `scheme://user:pass@host` to `scheme://***@host` before storing.
Rust has no sanitization anywhere in `ingest`.

**Impact:** database credentials, tokens, and connection strings are persisted
into the graph, exports, and vaults.

**Fix:** port `sanitize_resource` and apply it in `parse_file` before
`content_sig`, concept write, and export. Preserve non-string values and
non-userinfo URIs.

**Tests:** `postgres://user:pass@host/db`, `https://token@host`,
`mailto:a@b`, bare paths, `okf-asset://id`, and non-string YAML values.

---

### BUG-008 — Non-default embedding dimensions are not adopted

**Severity:** P0, M

**Where:** `src/store/mod.rs:62` (`meta_int("embedding_dim")`), `:453`
(`create_schema`), `:1431` (`pin_info`), `src/ops/mod.rs:260` (`reembed`).

**Evidence:** `Store::init(path, 256)` creates `FLOAT[256]` but does not write
the dimension to `Meta`. `Store::open` only reads a Meta key that Python-created
databases normally do not have. Confirmed:

```text
Store::init(path, 256)
Store::open(...).settings().embedding_dim == 512
```

`pin_info` then reports `pinned.embedding_dim = 512`, so the dim check in
`check_pins` is dead for normal DBs.

**Impact:** a 256-dim graph opens with 512-dim query/import vectors; imports
fail or search returns nothing. `reembed` cannot detect the mismatch.

**Fix:**

1. On open, run `CALL TABLE_INFO('Concept')`, parse `FLOAT[N]`, and adopt it
   with a warning when it differs from settings.
2. Write `Meta.embedding_dim` on `init` for new DBs (and as a migration).
3. Make `pin_info`/`reembed` use the schema width, not the configured value.

**Tests:** init at 128/256/512/768/1024, reopen without an explicit dim, and
run import/search; assert the adopted width. Add a v9 fixture with no Meta
dimension.

---

### BUG-009 — `delete` does not implement file-backed safety/trash/restore

**Severity:** P0, L

**Where:** `src/ops/mod.rs:167-205` (`delete`, `recover_deleted`).

**Evidence:** Rust accepts only `concept_id`, ignores `remove_file`, never
checks `FileHash`, always returns `trashed_files: []`, and always returns
`restored_files: []`. Python `ops/admin.py:1080`:

- refuses `BAD_VALUE` while a backing file exists and `remove_file=False`,
- with `remove_file=True`, moves files to `<root>/.okf-trash/<ts>/`,
- remembers trash paths in the deleted snapshot,
- restores them on `recover_deleted`.

**Impact:** deleting a file-backed concept leaves the file in place; the next
`import --all` resurrects it. Recovery cannot restore files. This violates the
documented single-writer/bundle contract.

**Fix:** implement the Python path:

1. Resolve `FileHash` paths through roots.
2. If files exist and `remove_file=False`, raise `BAD_VALUE` with `sources`.
3. If `remove_file=True`, move to `.okf-trash/<timestamp>/`, retire delta
   bookkeeping, and store the trash list in the snapshot.
4. On recover, move files back, re-import them, and return `restored_files`.

If the full behavior cannot land before beta, reject `remove_file=True` and
refuse deletion when a source exists; do not silently ignore the parameter.

**Tests:** file-backed delete refusal, forced trash, recover with file
restoration, and re-import after delete.

---

### BUG-010 — Image assets are never pruned

**Severity:** P0, M

**Where:** `src/store/mod.rs:794-805` (`delete_image_asset`).

**Evidence:** the query uses:

```cypher
OPTIONAL MATCH (i)<-[:INCLUDES_ASSET]-(:Concept)
WITH i, count(*) AS n WHERE n = 0 DELETE i
```

`OPTIONAL MATCH` yields one row with a null concept, so `count(*)` is 1 and
the delete never runs. Confirmed on the images fixture:

```text
links: 3
count before 3
count after 3
```

**Impact:** re-imports leave orphan `ImageAsset` nodes. Search can return
unowned assets, `image_count` is inflated, and `reembed` refuses a graph with
orphans as `IMAGE_PIN_MISMATCH` forever.

**Fix:** use `count(c)` (or `collect(c)`/`size`), delete only when the count is
zero, and add a test that removes an image reference and re-imports. Also
check the same pattern in any other orphan-cleanup query.

---

### BUG-011 — Markdown `image_count` is graph-wide, not per concept

**Severity:** P0, S

**Where:** `src/converters.rs:161-169` (`counts`), used at `:390`.

**Evidence:** `counts` runs `MATCH (i:ImageAsset) RETURN count(i)` and returns
the total for every concept. Python `_import_single_concept` returns the
per-concept `stats["total"] - stats["skipped"]`.

**Impact:** `ingest(kind='md')` reports wrong counts; tests with no images hide
it.

**Fix:** count through `MATCH (c:Concept {id: $id})-[:INCLUDES_ASSET]->(i)` or
reuse `store.list_images(id).len()`.

**Tests:** a graph with images on concept A and a markdown ingest of concept B;
assert B reports zero and A reports its own count.

---

### BUG-012 — Concept IDs can escape the bundle/output directory

**Severity:** P0, M

**Where:**

- `src/export/mod.rs:333-340` (`output_path`)
- `src/export/mod.rs:112`, `:144` (`export_concept` path join)
- `src/ops/mod.rs:396-430` (`write_and_import` materialization)
- no equivalent of Python `_check_concept_id` (`components/ingest.py:95`)

**Evidence:** `output_path` pushes each `id.split('/')` component and
`set_extension("md")`; `write_and_import` does
`Path::new(root).join(format!("{relative}.md"))`. A concept id containing
`..`, an absolute component, or a leading `/` can write outside the root.
Python validates IDs before materialization; Rust has no check on export or
update.

**Impact:** arbitrary `.md` write outside the bundle/output directory from a
crafted concept id or a manipulated database.

**Fix:**

1. Add `validate_bundle_id` equivalent to Python’s `_SAFE_ID_RE`: letters,
   digits, `_`, `.`, `-`, `/`; reject `..` components, leading `/`, leading
   `@` unless an alias is supported, backslashes, empty components.
2. Use a safe join helper that rejects `Component::ParentDir` and absolute
   paths.
3. Append `.md` with `format!("{id}.md")` rather than `set_extension`, so
   dotted IDs keep their name.
4. Apply it in export, update/materialize, and any path built from a concept
   id.

**Tests:** `../evil`, `/abs`, `a/../../b`, `a\\b`, `@alias/x` (until multi-root
lands), and a normal `notes/v1.0`.

---

## 5. P1 — Documented behavior and missing M3 features

### BUG-013 — External embedding/rerank/vision lanes were not implemented

**Severity:** P1, XL — **Status: closed** (BUG-013 external-lane slice).

**Where:** `src/embed/external.rs`, `src/embed/mod.rs`, `src/image_ingest.rs`,
`src/ops/mod.rs`, `parity/tools/check_external_lanes.py`.

**Resolution:** all three lanes are implemented behind the existing
`external` feature. They share endpoint normalisation, bearer-key headers,
`GET /v1/models` identity probes, and the typed outage/refusal split from
Python's `_http` module. The text lane speaks OpenAI-shaped
`/v1/embeddings`, applies declared prefixes, requests Matryoshka
`dimensions` with a loud client-side fallback, and supports the four
declared token counters. The rerank lane probes the standard `/v1/rerank`
wire and falls back to TEI `/rerank`, latches its dialect, and disables
itself for the session after a failure. The vision lane requires an
embroider identity with a text partner and posts original image bytes to
`/v1/images/embeddings`.

`pin_info` now reports the external family and loaded image precision/space,
and a declared engine pins only the model id while an embroider host pins
precision and space exactly like the native lane. Transport failures are
`EXTERNAL_UNAVAILABLE`; per-request refusals stay typed `INTERNAL` with the
host message.

**Tests:** `parity/tools/check_external_lanes.py` runs both runners against
four Python mock hosts and compares envelopes and DB dumps while asserting
request shape, bearer keys, identity probes, tokenize, rerank payloads and
image bytes. It also drives the TEI `/rerank` fallback and a generated but
valid GGUF counter fixture. Rust unit tests cover declared
prefixes/dimensions, outage mapping and GGUF header parsing; all-feature and
no-default suites stay green.

---

### BUG-014 — Precision/space pins are not enforced

**Severity:** P1, M

**Where:** `src/embed/mod.rs:535-573` (`check_pins`), `src/ingest/mod.rs:524`
(`store_pin_json`), `src/store/mod.rs:1382` (`pin_info`).

**Evidence:** `check_pins` checks only `model_id`; its dim branch receives no
dim from `store_pin_json`. It never checks precision or space, never rejects
unknown precision codes, and never re-pins an empty graph on model/precision
change (Python does). `pin_info.pinned.embedding_dim` is the configured value,
not the schema width.

**Impact:** importing with a different precision/space silently mixes vector
spaces; `PRECISION_PIN_MISMATCH`/`SPACE_PIN_MISMATCH` are dead codes.

**Fix:** build one `Pins` value containing model, precision, space, and schema
dim; check all four; port the Python empty-graph re-pin; reject unknown codes;
use it from import, search, embed, reembed, and pin_info.

**Tests:** mismatch for each field; empty graph re-pin; unknown code; a graph
with no pins.

---

### BUG-015 — `lanes` reports fabricated/null values

**Severity:** P1, M

**Where:** `src/store/mod.rs:1465-1515`.

**Evidence:** `space`, `pinned_space`, image precision/space/text partner, and
rerank identity/precision are hardcoded null or derived from settings, not
from the loaded session or stored pins. Python `lanes` reads the open session
and stored `pinned_space`.

**Impact:** `okf lanes`/`/health` lie about the active model and pins; clients
cannot detect a mixed graph.

**Fix:** return the actual loaded embedder/reranker/vision metadata when
available, stored pins otherwise, and `null` only when genuinely unknown.
Reuse the pin source from BUG-014.

**Tests:** cold and warm lanes for native and external; a graph with pins;
image and rerank lanes.

---

### BUG-016 — `ingest` parameter compatibility is wrong

**Severity:** P1, L

**Where:** `src/converters.rs:18-105` (`honours`, `validate`), `:812`
(`ingest_pdf`).

**Evidence:** Python’s ignored-param check only considers the 14 declared
`passed` params; `mode`, `batch_size`, `prune_missing`, `force`, `rebuild`,
etc. are accepted for every kind. Rust treats every supplied param as a
candidate and rejects common calls:

- `ingest(kind="md", md_path=..., mode="omni")`
- `ingest(kind="bib", bib_path=..., mode="omni")`
- `ingest(kind="pdf", pdf_path=..., mode="omni", batch_size=..., prune_missing=True)`

For PDF, `output_dir` is accepted even when `auto_import=True`; Python refuses
it with “output_dir needs auto_import=False”. `routing_mode` and
`extract_images` are allowed but ignored (`src/pdf.rs:153-154` hardcodes
`auto`/`true`). `pdf_device`/`pdf_precision` are accepted but not applied.

**Impact:** valid Python calls fail with `BAD_VALUE`; other calls succeed but
silently ignore user intent.

**Fix:**

1. Mirror Python’s `passed` dict: only the declared per-kind fields participate
   in the ignored check.
2. Accept and forward `mode`, `batch_size`, `prune_missing`, `force`, and
   `rebuild`; implement them or reject with a clear “not supported in this
   build” error, never silently.
3. Implement PDF `routing_mode`/`extract_images` in the plugin config; reject
   `output_dir` when `auto_import=True`; pass batch/prune/force to the import.
4. Make the unknown-kind error match Python’s message/fields.

**Tests:** one scenario per kind covering every accepted parameter and every
ignored-param envelope; PDF config round-trip with a fake plugin.

---

### BUG-017 — `materialize` and `folder` are accepted but ignored

**Severity:** P1, M

**Where:** `src/converters.rs:26`, `:38-40` (allowlists); no implementation in
`ingest_md`/`ingest_thoughts`.

**Evidence:** Python `materialize=True` writes `<concept_id>.md` into the
bundle and imports that file; `folder` derives the thought ID from
`<folder>/<title-slug>.md` and implies `materialize=True`. Rust only validates
the params and then ignores them.

**Impact:** callers believe a file was written and re-imported; it was not.
`folder` also silently drops the requested location.

**Fix:** implement the `write_and_import` path already used by `update`, with
the safe-ID rules from BUG-012; for `folder`, port `_folder_thought_id`
(conflict with `concept_id`, `materialize=False` refusal, title slug).

**Tests:** md materialize, thoughts materialize, folder with/without title,
folder + concept_id conflict, folder + materialize=False.

---

### BUG-018 — Markdown ingest ignores `extra` and hardcodes lint results

**Severity:** P1, M

**Where:** `src/converters.rs:351-406` (`ingest_md`).

**Evidence:**

- `let _extra = check_extra(...)` validates but discards `extra`; Python merges
  it into the concept.
- `lint_issues` is always `fixed_count: 0, fixable_count: 0,
  unfixable_count: 0, error_count: 0`; Python runs the structural linter and
  reports real counts.
- `title=""`/`description=""` are applied as empty strings; Python treats an
  empty value as not supplied and keeps the frontmatter value.
- `type=""` is applied; Python keeps the frontmatter/default type.

**Impact:** md ingest output and stored graph differ for any file with extra
fields, lint issues, or empty overrides.

**Fix:** apply `extra` after the frontmatter fields; run the shared lint/fix
logic (or at least report counts); use truthy checks for overrides; add the
same behavior for Bib/thoughts where applicable.

**Tests:** `extra` map, trailing-space file, missing title/type, empty
override, and a file with a dangling link.

---

### BUG-019 — Thoughts semantics diverge

**Severity:** P1, L

**Where:** `src/converters.rs:542-800`.

**Evidence:**

- **Title uniqueness:** Python refuses a duplicate explicit title in a topic
  and suffixes auto-titles with `(2)`, `(3)`, …; Rust always creates a
  duplicate.
- **Slug:** Python collapses runs of disallowed chars, lowercases, truncates
  to 30 chars, and falls back to `"untitled"`; Rust maps each char to `_`,
  does not truncate, and can emit `thoughts//<timestamp>`.
- **Auto-title:** Python cuts at 72 chars; Rust stops at a space at/after
  byte 60 and can produce an empty title for a long unbroken sentence.
- **`related`:** Python merges into `extra["relations"]["related"]`, preserves
  other relation kinds, and rejects non-list/non-string/blank entries; Rust
  overwrites `relations` and silently drops invalid entries.
- **`created`:** Rust always adds `created` to `Concept.extra`; Python only
  stores explicitly supplied extra.
- **Tags:** Python uses `list(set([...]))`; Rust preserves insertion order and
  special-cases only `topic == "thought"`.

**Impact:** thought IDs/titles, DB `extra`, and relation maps differ for normal
topics, duplicates, and related links. The current scenario uses a fixed
title/topic and hides all of it.

**Fix:** port `_slugify_topic`, `_auto_thought_title`,
`_thought_titles_in_topic`, and `_with_related`; decide the tag-order contract
(Python set order is hash-randomized, so either normalize both to a set in
comparison or choose a stable order and document the divergence); stop
injecting `created` into extra.

**Tests:** duplicate explicit title, duplicate auto title, long title, empty
slug, `related` with an existing `relations` map, invalid related entries, and
tag order.

---

### BUG-020 — Multi-root (`roots`/`@alias`) is not implemented

**Severity:** P1, XL

**Where:** `src/settings/mod.rs:155` (`roots`), `src/ingest/mod.rs:540`
(`relative_id`), link index, `src/ops/mod.rs:396` (update path), `detach`.

**Evidence:** `roots` is parsed/validated but never used by import, link
resolution, update, or detach. `relative_id` never prefixes `@alias/`;
`parse_file` has no alias parameter; link qualification has no alias
knowledge; `SourceRoot` is not written by import. Python supports named roots,
namespaced IDs, alias link qualification, and absent-root purge refusal.

**Impact:** multi-root graphs silently collapse IDs, links cannot resolve
across roots, and `@alias/...` concept IDs are treated as literal paths.

**Fix:** thread an `alias`/root through `walk`/`parse_file`/`LinkIndex`;
prefix IDs with `@alias/`; record `SourceRoot`; resolve `@alias` in update,
delete, export, and detach; implement absent-root `PURGE_REFUSED_ABSENT_ROOT`.

**Tests:** two roots with the same relative path, cross-root links, alias
links, `@alias` materialization, and a missing root during prune.

---

### BUG-021 — `prune_missing`, `force`, and delta tombstones are missing

**Severity:** P1, L

**Where:** `src/ingest/mod.rs:91-340` (`import_bundle`), `src/ops/mod.rs:816`
(`import_bundle`), `:657` (`sync`).

**Evidence:** `import_bundle` only records `DeletedPath` rows; it never
consumes them, never removes the concept, and does not accept `prune_missing`,
`force`, `batch_size`, or `rebuild`. Stale `DirHash` rows for deleted
directories are not removed. Python warns without `--prune-missing`, purges
with it, and protects multi-root graphs with `PURGE_REFUSED_ABSENT_ROOT`.

**Impact:** deleted source files leave concepts in the graph forever; there is
no way to prune them. `force` and rebuild policy are ignored.

**Fix:** implement the delta lifecycle:

1. Accept `prune_missing`/`force`/`batch_size`/`rebuild` and validate them.
2. On prune, map pending `DeletedPath` → `FileHash.concept_id`, skip concepts
   re-imported in the same run, purge the rest, and clear tombstones.
3. Delete `DirHash` rows for vanished directories.
4. Implement the absent-root refusal for multi-root graphs.
5. Keep `sync`’s cursor semantics.

**Tests:** delete a file and re-import without/with prune; move a file; delete
a directory; missing root; force duplicate resolution; deferred rebuild.

---

### BUG-022 — Timestamp formatting, timezones, and clocks differ

**Severity:** P1, M

**Where:**

- `src/model/mod.rs:102-124` (`isoformat` trims zeros)
- `src/model/mod.rs:126-143` (`format_py_datetime` trims zeros)
- `src/model/mod.rs:53-63` (`parse` chops `+HH:MM`)
- `src/ingest/mod.rs:686-695` (`value_to_extra` uses space)
- `src/ops/mod.rs:182` (`recoverable_until` uses UTC)
- `src/purge.rs:18-30`, `:155-190` (UTC delete/purge)
- `src/converters.rs:606` (thoughts `created` UTC)

**Evidence:** Python emits six fractional digits for nonzero microseconds
(`2024-01-02T03:04:05.100000`), keeps `T` in `extra_value`, converts offsets
to UTC, and uses local naive `datetime.now()` for delete/purge/thoughts
`created`. This machine is UTC+1: Python wrote `2026-11-20T00:20…`, Rust
`2026-11-19T23:20…`. Parity ignores these fields.

**Impact:** `recoverable_until`, `deleted_at`, `age_seconds`, content hashes,
exported timestamps, and thought metadata are wrong or shifted.

**Fix:** centralize Python datetime formatting:

- always 6 fractional digits for `isoformat()`/`str(datetime)` when nonzero,
- `extra_value` uses `.isoformat()` (`T`),
- parse offsets and convert to UTC,
- use the same clock source as Python for each field (UTC for concept times,
  local naive for delete/purge/thoughts `created`), or explicitly change the
  Python contract and both implementations together.

**Tests:** microsecond values ending in zero, offsets `+05:30`/`-08:00`/`Z`,
extra datetime round-trip, delete/recover times under a non-UTC `TZ`.

---

### BUG-023 — YAML parser is incomplete and can drop/corrupt data

**Severity:** P1, L

**Where:** `src/markdown/mod.rs:260-463` (`parse_yaml`/`parse_block`/
`resolve_scalar`).

**Evidence (confirmed):**

```text
"description: |\n  line one\n  line two\n" -> {"description":"|"}
"a:\n    b: 1\n"                          -> {"a":{}}
"title: Foo # comment\n"                 -> {"title":"Foo # comment"}
```

Missing: block scalars (`|`, `>`), 4-space nested mappings, inline comments,
YAML 1.1 octal (`010`), lists of maps, and proper `.inf`/`.nan` floats
(currently tagged objects). These feed `content_sig`, `extra`, and the
concept columns.

**Impact:** valid frontmatter is silently truncated or mis-typed; the graph
and re-import signature differ from Python.

**Fix:** extend the parser for block scalars, indentation, comments, octal,
and sequence-of-map values; or adopt a YAML parser and layer the PyYAML 1.1
resolver on top. Add golden cases before changing behavior.

**Tests:** the confirmed cases above plus anchors? (document non-support),
multi-line descriptions, quoted keys, nested maps, lists of maps, `.inf`,
`.nan`, and duplicate keys.

---

### BUG-024 — YAML dumper loses nested structures and can emit invalid YAML

**Severity:** P1, M

**Where:** `src/export/mod.rs:365-465` (`dump_yaml`, `yaml_scalar`,
`yaml_string`).

**Evidence:** `yaml_scalar(Json::Object)` returns `"{}"` and array items that
are objects/arrays are passed through `yaml_scalar`, so `a: [{"x": 1}]`
exports as `a:\n- {}`. Top-level arrays render as `[]`. Object keys with `:`,
`#`, or newlines are not quoted. `yaml_string` only quotes a limited set.

**Impact:** `extra` fields with structured values are silently lost on export,
update, and materialization; some keys produce invalid YAML.

**Fix:** implement recursive block YAML for maps/sequences, quote keys and
scalars using PyYAML-compatible rules, and add round-trip tests. Do not use a
serializer that changes PyYAML 1.1 scalar resolution without a compatibility
layer.

**Tests:** array of maps, array of arrays, nested map, keys with `:`/`#`,
multiline strings, booleans/numbers/strings that look typed.

---

### BUG-025 — Export filtering and `.md` paths are wrong

**Severity:** P1, M

**Where:** `src/export/mod.rs:76` (`selected`), `:98` (`write_indexes`),
`:333-340` (`output_path`).

**Evidence:**

- `selected` is built before the `directory_id` subtree filter; it includes
  concepts that were not exported.
- `write_indexes` receives all type/tag-filtered concepts, including those
  outside the subtree; Python indexes only the exported set.
- `path.set_extension("md")` replaces the last extension, so `notes/v1.0`
  writes `notes/v1.md` instead of `notes/v1.0.md`.

**Impact:** filtered exports return wrong IDs and index files; dotted concept
IDs overwrite/rename files.

**Fix:** apply all filters before building `selected`/indexes; append `.md`
with `format!("{id}.md")`; use the safe join from BUG-012.

**Tests:** directory/type/tag filtered exports, dotted IDs, obsidian flavor,
and index contents.

---

## 6. P2/P3 — API, storage, packaging, performance

### BUG-026 — Python facade is not API-compatible

**Severity:** P2, L

**Where:** `python/okfgraph/router.py:28` (`__init__`), `:12`
(`_READ_METHODS`), `:60` (`read`), `python/okfgraph/client.py`.

**Evidence:**

- `OKFRouter.__init__` rejects positional arguments; Python 0.15’s first
  parameter is positional `db_path`, so `OKFRouter("graph.db")` raises.
- Only read methods are exposed; `search`, `embed`, `import_*`, `ingest`,
  `update`, `delete`, `export_*`, `sync`, `produce`, `lint`, `diff` are absent.
- `read` has no `max_tokens`/other Python parameters.
- `OKFClient` is a Stage C stub.

**Impact:** the beta wheel cannot be a drop-in for the documented public API.

**Fix:** restore the constructor signature (positional `db_path`), add every
public op method with the Python keyword signature, and keep `.call` as the
escape hatch. Generate the method list from a single op table to prevent drift.

**Tests:** import the facade, instantiate with positional and keyword
`db_path`, call every public method, and compare signatures with the oracle.

---

### BUG-027 — Facade errors lose `op` and `data`

**Severity:** P2, S

**Where:** `bindings/python/src/lib.rs:40` (`to_py_err`),
`python/okfgraph/errors.py:59` (`from_native`).

**Evidence:** `to_py_err` serializes only `error.envelope_body()`, which
contains code/message/fields/remedy but not `op`/`data`. `from_native` sets
`op=None`, `data=None`. Outcome errors (`LINT_ERRORS`, `DIFF_DIFFERENT`,
`DOCTOR_FINDINGS`) lose their report.

**Fix:** include `op` and `data` in the native error payload; populate
`OKFError.op/.data` in `from_native`; keep the public envelope shape unchanged.

**Tests:** catch `lint`/`diff` outcome errors from Python and assert `.data`,
`.op`, `.envelope()`, and `str()`.

---

### BUG-028 — Schema migrations are absent; `okf init` can stamp an old DB

**Severity:** P2, L

**Where:** `src/store/mod.rs:64-72` (only rejects newer), `:453-465`
(`create_schema`), `:467` (`schema_version` returns 0 when missing).

**Evidence:** no v1→v10 migration code exists, despite the module docs and the
M1 plan. `Store::init` runs `CREATE ... IF NOT EXISTS` and stamps version 10,
so running it on an older DB can partially add v10 tables and skip migrations.
A missing `Meta.schema_version` is treated as 0 and open proceeds.

**Fix:** port the Python `_MIGRATIONS` registry; add v1/v5/v9 fixtures; make
`open` migrate (or refuse with a clear `CONFIG_INVALID`) and `init` refuse an
existing non-v10 DB rather than stamping it. Add a migration test that dumps
all tables before/after.

**Tests:** open v1/v5/v9 fixtures, migrate, reopen with Python 0.15; run
`init` on an existing DB; missing version row.

---

### BUG-029 — No write lock; read-only mode is not enforced

**Severity:** P2, L

**Where:** `src/store/mod.rs:43` (`read_only`), `:93-95`
(`open_writable`), `:117-137` (`with_write_conn`/`write`), `src/ops/mod.rs`
`Router`.

**Evidence:** `WRITE_LOCK_TIMEOUT`/`DB_LOCKED` exist but are unused. There is
no inter-process file lock. `Store`/`Router` are `Sync`; two Python threads
can enter `NativeRouter.call` concurrently with the GIL released. `write()`
ignores `read_only` and always opens a writable connection.

**Impact:** concurrent writers can corrupt the graph; a “read-only” handle can
delete/import/reembed through `.call`.

**Fix:** port the Python fasteners lock semantics (`fs4`/`fd-lock`), acquire it
for every write transaction, expose `DB_LOCKED`/`WRITE_LOCK_TIMEOUT`, and make
`write()` reject when `read_only` is set. Add a cross-process Python↔Rust lock
test.

**Tests:** two processes writing; two Python threads; read-only handle calling
a write op; lock timeout.

---

### BUG-030 — `extension_dir`/`wal_mode`/`pdf_idle_seconds` are accepted but unused

**Status:** closed by documented disposition (`cee784a`): `extension_dir` and
`pdf_idle_seconds` are implemented; `wal_mode` remains a compatibility-only
no-op and is now explicitly recorded in `docs/rust-compat.md` §A. The 0.15
help text is preserved so the generated settings table still matches; no
upstream-supported journal switch exists in `lbug 0.22.1`.

**Severity:** P2, S

**Where:** `src/settings/mod.rs:105-106`, `:139`; `src/store/mod.rs:117-128`.

**Evidence:** `extension_dir` is validated but never used for
`LOAD vector`/`LOAD fts`; `wal_mode` is never passed to lbug;
`pdf_idle_seconds` is never used because the plugin has no warm converter.

**Impact:** offline machines cannot point at a local extension directory;
settings that appear supported do nothing.

**Fix:** use `extension_dir` in the extension load path, pass `wal_mode` to the
lbug config if supported, and implement the warm PDF converter/idle close or
reject the setting. Remove settings that cannot be honored.

**Tests:** extension load from a custom directory; setting round-trip; PDF
converter reuse.

---

### BUG-031 — Index creation/drop swallows all errors

**Severity:** P2, S

**Where:** `src/store/mod.rs:167-196`.

**Evidence:** `let _ = conn.query(sql)` ignores every error, not just
“already exists”. A failed rebuild (dimension mismatch, missing extension,
corrupt index) is reported as success.

**Impact:** `reindex`/`reembed` can claim success while search is unavailable.

**Fix:** inspect the error; ignore only the known “already exists”/“not found”
cases; propagate the rest as `SEARCH_UNAVAILABLE`/`INTERNAL` with a remedy.

**Tests:** create an index with a wrong width; drop a missing index; assert
the expected error/success.

---

### BUG-032 — Directory walk follows symlinks, swallows errors, and imports hidden files

**Severity:** P2, M

**Where:** `src/ingest/mod.rs:398-440` (`walk_dir`), `src/detach.rs:55-90`.

**Evidence:** `path.is_dir()` follows directory symlinks; Python’s default
`rglob` does not recurse into symlinked directories. `read_dir` errors are
silently ignored, so an unreadable directory looks like deleted files.
Hidden files (`.secret.md`) are imported because only directory names are
checked; Python’s `in_skipped_dir` checks every path component.

**Impact:** symlink cycles/hangs, reads outside the bundle, false tombstones,
and accidental import of dotfiles.

**Fix:** use `symlink_metadata` and skip symlinked directories (or explicitly
document/follow with cycle detection); skip any component starting with `.`
including the file name; propagate or report `read_dir` errors instead of
silently returning.

**Tests:** symlink loop, symlink to outside root, hidden file, unreadable
directory.

---

### BUG-033 — BibTeX parser/encoding/skip/lint differences

**Severity:** P2, M

**Where:** `src/converters.rs:234-535`.

**Evidence:** Rust uses UTF-8 only (Python falls back to latin-1), always
returns `skipped_entries: []` (Python reports slug collisions/parser skips),
does not lint-fix generated markdown, and its slugify does not collapse runs or
truncate to 64 chars. The parser does not handle `@string`/`@comment` like
Python’s parser.

**Impact:** non-UTF-8 bibliographies fail/panic; collision reports and IDs
differ; generated markdown can be structurally invalid.

**Fix:** add latin-1 fallback; port `slugify_key`; report collisions/skips;
reuse the shared lint/fix; extend the parser for `@string`/`@comment` and
nested values; add tests with duplicate keys and accented fields.

**Tests:** latin-1 file, duplicate keys, `@string`, malformed entry, long key,
trailing-space body.

---

### BUG-034 — PDF plugin lifecycle/config gaps

**Severity:** P2, L

**Where:** `src/converters.rs:812-880`, `src/pdf.rs:146-159`,
`addons/pdf/src/lib.rs`.

**Evidence:**

- Auto-import creates a temp dir and never deletes it (`:837`).
- Converted markdown is not lint-fixed (Python runs `_lint_converted_md`).
- `routing_mode`/`extract_images` are hardcoded in `pdf::config`.
- `pdf_idle_seconds` is ignored; the plugin is loaded per call.
- Cache fallback ignores `HF_HOME`/`XDG_CACHE_HOME`.
- Plugin discovery only checks `OKFGRAPH_PDF_PLUGIN` and the executable
  directory; native CLI use outside the Python facade may not find a
  site-packages plugin.

**Impact:** disk leaks, different converted output, ignored settings, and
packaging surprises. There is no end-to-end PDF/Office parity fixture.

**Fix:** use an RAII temp dir, run the shared lint/fix, pass all knobs through,
implement warm converter/idle close, use `hf_hub`-compatible cache resolution,
and add discovery from the package/plugin path. Add a real PDF fixture with
cached bobine models and compare markdown/DB/export.

**Tests:** missing plugin envelope, real PDF conversion, `auto_import=False`,
`output_dir` refusal, routing/extract images, temp cleanup, cache env.

---

### BUG-035 — GIF/WebP vision ingest cannot decode

**Severity:** P2, S

**Where:** `Cargo.toml` (`image` features `png`,`jpeg`), `src/image_ingest.rs`
(`sniff_mime` accepts GIF/WebP), `src/embed/mod.rs` (`prepare_image`).

**Evidence:** the image crate is built without GIF/WebP decoders, but MIME
sniffing and the text route accept those formats. Vision mode on a GIF/WebP
fails to decode.

**Fix:** enable `gif`/`webp` features, or reject those formats in vision mode
with a clear `CONVERSION_FAILED`/`BAD_VALUE`; test both routes.

---

### BUG-036 — `import_file` always reports zero images

**Severity:** P2, S

**Where:** `src/ops/mod.rs:837-858`.

**Evidence:** the response hardcodes `"images": 0`, discarding the report from
`import_files`.

**Fix:** return the per-concept image count from the report/`list_images`.

**Tests:** import a file with an image and assert the count.

---

### BUG-037 — Unknown ingest kind error shape differs

**Severity:** P2, S

**Where:** `src/converters.rs:67-73`, `:883-905`.

**Evidence:** Rust returns `BAD_VALUE: unknown ingest kind '…'` without
`fields.kind`; Python returns
`kind must be 'md', 'pdf', 'thoughts' or 'bib', got '…'` with `fields.kind`.

**Fix:** match Python’s message/fields/remedy exactly.

**Tests:** each unknown kind, missing `kind`, and `kind=None`.

---

### BUG-038 — `OP_NAMES` exposes 32 ops vs the oracle’s 30

**Severity:** P2, S

**Where:** `src/ops/mod.rs:19` (`OP_NAMES`), `python/okfgraph/client.py:9`.

**Evidence:** Rust includes `lint` and `produce` in the public op list; the
Python 0.15 oracle exposes 30 names. `check_wheel.py` only prints the count, so
the drift is not asserted.

**Impact:** public `OPS` changes, loops/tests that expect 30 break, and the
“30 ops” docs are wrong.

**Fix:** decide whether `lint`/`produce` are public ops or CLI-only. If
CLI-only, remove them from `OP_NAMES` (keep a separate CLI list); if public,
update the Python oracle contract/docs and all counts. Add an exact-list
assertion.

**Tests:** `check_wheel.py` compares the full sorted list, not just length.

---

### BUG-039 — `.inf`/`.nan` are not Python floats

**Severity:** P3, S

**Where:** `src/markdown/mod.rs:438-463`, `src/pycompat.rs` (`from_golden`).

**Evidence:** `parse_float` returns
`{"__type__":"float","value":"inf"}`; `PyValue::from_golden` maps unknown
tagged types to `Other("inf")`, which serializes as `"inf"`, while Python uses
`Infinity`/`NaN` in `json.dumps`.

**Fix:** represent non-finite floats explicitly in `PyValue` (or a dedicated
tag) and serialize like Python; keep YAML parsing consistent.

**Tests:** `.inf`, `-.inf`, `.nan` in frontmatter and extra; `content_sig`.

---

### BUG-040 — Per-row write connections are slow and prevent batching

**Severity:** P3, M (performance) — **Status: closed** (fix series after
`cee784a`).

**Where:** `src/store/mod.rs:117-137`, `src/ingest/mod.rs` (`import_one`,
`write_chunk`), `src/image_ingest.rs`.

**Evidence:** every `Store::write` opens a new `Database`/`Connection`, loads
`vector`/`fts`, executes one statement, and drops it. A 1,000-chunk document
opens the DB 1,000+ times. This also causes the atomicity problem in BUG-003.

**Impact:** imports can be orders of magnitude slower than Python and are
non-transactional.

**Fix:** add a scoped write transaction API (`Store::with_write_tx`) and use it
for a whole concept/batch; keep the fresh-connection fallback only for
single-statement callers. Benchmark import on the filters corpus.

**Resolution:** `Store::with_write_tx` batches a whole concept replacement,
and `examples/import_bench.rs` / `parity/tools/check_import_bench.py` provide
the real-model timing smoke. Measured on the Windows x64/CUDA dev box: 8
concepts imported in ~10.3 s; unchanged re-import ~1.1 s with zero queued
files. Thresholds default to 60 s / 5 s and are overridable per machine.

**Tests:** `check_import_bench.py` and the unit transaction/rollback tests.

---

### BUG-041 — Remote image fetch/SSRF settings are ignored

**Severity:** P1/P2, M

**Where:** `src/image_ingest.rs:246-254` (`load_bytes` returns `None` for
remote), `src/settings/mod.rs:132-133`.

**Evidence:** `allow_remote_images` and `allowed_image_domains` are validated
but never used; `load_bytes` always skips `http(s)` with the comment “stays
opt-in and out of Stage B parity”. Python supports opt-in fetch with an
allowlist and SSRF guard.

**Impact:** documented opt-in behavior is missing; users cannot ingest remote
images.

**Fix:** implement the HTTP fetch behind `allow_remote_images`, enforce the
domain allowlist and SSRF rules (no loopback/private/link-local unless
explicitly allowed), set timeouts/size caps, and return a typed error for
blocked URLs. Add a local mock-server test.

---

### BUG-042 — `sync` can record hashes under the wrong root

**Severity:** P2, S

**Where:** `src/ops/mod.rs:657-750`, `src/ingest/mod.rs:891-910`.

**Evidence:** `sync` passes `self.settings` to `import_files`; if
`output_dir != settings.bundle_root`, `record_file_hash` computes paths
relative to the bundle root and may store absolute paths. The parity scenario
sets them equal.

**Fix:** clone settings with `bundle_root = output_dir` for the sync import, or
pass an explicit root to `import_files`.

**Tests:** sync with `output_dir` different from `bundle_root`; assert relative
`FileHash.path` values.

---

### BUG-043 — Documentation still claims unsupported behavior

**Severity:** P3, S

**Where:** `docs/plan-rust-core.md`, `parity/README.md`, `README.md`.

**Evidence:** the plan says M3 has no gaps; the parity README says all Stage B
scenarios are green without listing the untested paths; the public docs claim
30 ops while Rust exposes 32; some settings are documented as active but are
unused.

**Fix:** update the status to “parity matrix green for covered fixtures; P0/P1
items in `docs/plan-bug-fixes.md` remain”; list the missing scenarios; align
the op count and settings.

---

### BUG-044 — Parity coverage does not exercise the broken paths

**Severity:** P1 (process), M

**Where:** `parity/scenarios/*.json`, `parity/tools/*`.

**Evidence:** all scenarios use default identity, no `bundle_root` on ingest,
no file-backed delete, no non-default dim, no external backend, no block YAML,
no duplicate thoughts, no image removal, and no filtered export.

**Fix:** add the scenarios in section 7 and make the broad loop assert
non-zero differences as failures. Add a coverage check that every public op
appears in at least one success and one error scenario.

---

### BUG-061 — CLI opened a database already owned by `okf serve`

**Severity:** P2, S — **Status: closed** (M4 owner-routing slice).

Python's CLI asks `discover(db_path)` for a live owner and sends every op
through `OKFClient`; the Rust CLI always called `Router::open`, so a second
`okf` process next to a server failed with `DB_LOCKED`. `src/server/client.rs`
now provides the Rust discovery/op client (HTTP and POSIX UDS), `okf` borrows
a live owner unless `OKFGRAPH_NO_SERVER=1`, and `okf-mcp --connect` reuses the
same transport. `parity/tools/check_client.py` runs a CLI op while the server
owns the fixture and checks stale discovery rejection.

### BUG-060 — HTTP job records did not match Python 0.15

**Severity:** P2, S — **Status: closed** (M4 owner-routing slice).

The Rust job lane used `job-N` ids, `succeeded`, `progress: f64` and
`created_at`; Python 0.15 exposes 12-hex ids, `done`/`cancelled`/`failed`,
`phase`/`done`/`total`, `kwargs`, `submitted`/`started`/`finished`/`elapsed`,
retires terminal records after an hour, and refuses a second reembed with
`REEMBED_IN_PROGRESS`. It also validates `mode` as `sync`/`job`, returns
`{job,status,op}` from submit, and checks cancellation between reembed batches.
All of these now match: `Router::reembed_with_control()` carries the
cooperative hooks, `JobManager` emits the Python view, duplicate jobs are
refused with 409, and unknown op/job use 404. `check_jobs.py` verifies mode
validation, dedupe, the Python record shape, and terminal behavior.

### BUG-059 — Concurrent HTTP calls contended for the Ladybug connection

**Severity:** P1, S — **Status: closed** (M4 owner-routing slice).

`okf serve` ran each blocking op on `spawn_blocking` but did not serialise
them. Four parallel `OKFClient.get_by_id` calls produced
`IO exception: Could not set lock on file ... (Error: 33)`. The owner now has
one `op_lock` shared by normal ops and jobs, mirroring Python `LockedRouter`;
cancel stays lock-free. `check_client.py` includes a four-client concurrency
assertion.

### BUG-049 — Public error envelopes drifted for several ops

**Severity:** P2, S — **Status: closed** (fix series after `cee784a`).

**Where:** `src/ops/mod.rs` (`embed`, `traverse`, `import_file`, `sync`),
`src/ingest/mod.rs` (`import_files`), `src/util/mod.rs` (`python_path_str`).

**Evidence:** the new `op-error-matrix` scenario exposed five envelope diffs:
missing `fields.n` on empty `embed`, missing `fields.depth` on invalid
`traverse`, `import_file` emitting the batch `file(s) not found`/`files`
shape instead of Python's `file not found`/`file_path`, unnormalized `/` in
`import_files` error paths on Windows, and `sync` validating `output_dir`
before the producer name (Python checks the producer first and includes
`fields.available`).

**Resolution:** each envelope now matches the oracle, including Windows path
normalization via `python_path_str`. The scenario covers all five ops plus
`export_bundle`, `export_concept`, `recover_deleted`, `list_images`, and
`import_files`; `check_coverage.py` verifies the success/error matrix.

**Tests:** `op-error-matrix` op/DB parity and full matrix.

---

### BUG-058 — Python version matrix was unverified

**Severity:** P2, S — **Status: closed** (M5 packaging slice).

Only Python 3.13 had been exercised against the abi3 wheel. The build metadata
claims 3.12–3.14, so `parity/tools/check_python_matrix.py` now creates a clean
venv for each supported interpreter, installs both wheels, checks import
surface/console scripts/facade, and optionally converts a real PDF. All three
Windows versions pass locally; CI runs the same matrix on Linux and macOS.

### BUG-055 — no installable PDF add-on wheel

**Severity:** P2, S — **Status: closed** (M5 packaging slice).

`okfgraph[pdf]` declared a dependency on `okfgraph-pdf`, but no wheel existed:
the native plugin was only available as a `target/debug` file through an env
override. `addons/pdf` now ships a PEP 517 backend (`build_wheel.py`) that
builds the cdylib and packages it with the `okfgraph_pdf` locator. A clean venv
with both wheels discovers the plugin without `OKFGRAPH_PDF_PLUGIN` and
converts `sample.pdf`. `check_packaging.py` and `check_pdf_wheel.py` pin this.

### BUG-056 — packaging metadata and import surface drifted

**Severity:** P2, S — **Status: closed** (M5 packaging slice).

The wheel allowed Python 3.15+, shipped stale `__pycache__` files from the
staging copy, and did not expose the public `OKFClient` / settings constants at
the package root. Metadata now pins 3.12–3.14 with platform classifiers, the
stage ignores bytecode, and the facade exports the documented client/settings
names. `check_packaging.py` and `check_facade.py` cover the metadata and root
imports; `check_wheel.py` rejects imports of removed packages.

### BUG-057 — wheel script could silently skip repair

**Severity:** P2, S — **Status: closed** (M5 packaging slice).

Passing `--profile dev` to `bindings/python/build_wheel.py` replaced the default
argument list and dropped `--auditwheel=repair`. The script now always passes
repair unless the caller overrides it explicitly, clears maturin's scratch
directory so a cached patched artifact cannot skip DLL staging, and the
packaging check requires the Windows DLLs in `okfgraph.libs/`. A no-op repair is
valid on platforms with no external shared-library dependencies.

### BUG-053 — HTTP `/mcp` mount was missing

**Severity:** P2, S — **Status: closed** (M4 HTTP mount slice).

`okf serve` exposed the op plane and job lane, but MCP was only reachable from
the separate stdio binary. The axum router now mounts the same 11 rmcp tools at
`/mcp` through a `StreamableHttpService` backed by the owning router. The
Python MCP SDK connects with `streamable_http_client`, lists all annotated
tools, reads a concept, and receives a typed envelope for a missing concept.

### BUG-054 — UDS transport was missing

**Severity:** P2, S — **Status: open (code landed, POSIX verification pending)**.

`--listen` accepted only `host:port` even though the public `OKFClient` already
understood `unix:///path`. The server now parses `unix://` targets, binds a
`tokio::net::UnixListener`, replaces stale socket files, refuses to displace a
live listener, and removes the socket on graceful shutdown. POSIX-only unit tests cover stale-socket replacement and the health envelope;
`parity/tools/check_uds.py` checks health/op/error through the public client. This environment is Windows and
cannot start a Rust UDS listener, so the smoke currently SKIPs; close this bug
after the Linux x86_64 or macOS arm64 CI run is green.

### BUG-052 — CLI command tree was incomplete

**Severity:** P2, S — **Status: closed** (M4 CLI slice).

The Rust `okf` binary registered only 10 of the oracle's 29 verbs, omitted the
presentation globals (`-v`, `-q`, `--log-file`, `--profile`), and reported a
generic `cli` op on errors. It now generates the full command tree, shares
settings flags with `okf-mcp`, and maps each verb to its op or local helper
(`model-info`, `lint`, `produce`, `diff`, `setup`, `retitle-thoughts`, `shell`).
Errors use the invoked command as the fallback op. A unit test pins the 29
verbs and `parity/tools/check_cli.py` compares 13 `--json` cases with the
oracle. Host-specific model cache details and doctor root paths are normalized
the same way as the doctor `info` rules.

### BUG-051 — HTTP job endpoints were placeholders

**Severity:** P2, S — **Status: closed** (M4 job lane).

`okf serve` advertised `/jobs`, `/jobs/{id}`, and `/jobs/{id}/cancel`, but
always returned `[]` or `unknown job`. `OKFClient.reembed_job()` sent
`mode="job"` and got a synchronous reembed result instead of a job id. The
owner now keeps in-memory job records, runs each op on the same
`spawn_blocking` pool, and exposes submit/poll/list/cancel. A cancel before
or between reembed batches produces the checked `cancelled: true` report with
partial counts; a cancel after a job is terminal does not rewrite its result,
and terminal jobs are not restarted (BUG-060 refined this behavior).
`src/server/mod.rs` unit tests cover completion and late cancel, and
`parity/tools/check_jobs.py` checks the HTTP contract end-to-end. The
`OKFGRAPH_JOB_TEST_DELAY_MS` hook is test-only.

### BUG-050 — Loaded-plugin conversion errors used the install hint

**Severity:** P2, S — **Status: closed** (M4 serve/MCP slice).

`Plugin::convert` mapped every plugin `{"error": ...}` to
`CONVERSION_FAILED` plus the `pip install "okfgraph[cpu,pdf]"` remedy. That
hint is correct when the add-on is missing or ABI-incompatible, but wrong once
the plugin is loaded and bobine rejects one file. Python 0.15 emits
`fields: {path, converter: "bobine"}` and the “check the file opens in another
reader” remedy. `addons/pdf` now returns those fields for `ingest_document`
failures, and the core copies them into the typed error while retaining the
install hint for bare ABI/setup errors. `ingest-office-formats` verifies
XLSX/PPTX success and a typed failure for a deliberately malformed PPTX with
zero differences. The original `sample.pptx` was also malformed (it carried
the XLSX root relationship); that was a fixture bug, not a converter refusal.

### BUG-045 — `doctor` parameters were ignored

**Severity:** P2, M — **Status: closed** (fix series after `391b46a`).

**Where:** `src/ops/mod.rs` (`doctor`/`doctor_fix`),
`python/okfgraph/router.py:177` (`def doctor(...)`).

**Historical evidence:** Python exposes
`doctor(stale_days=365, fix=False, strict=False, cache_dir=None)`.
The Rust op originally took no parameters: the stale threshold was hardcoded
to 365 days, `fix=True` returned `fixed: null`, `strict=True` never raised
`DOCTOR_FINDINGS`, and the facade exposed no keyword arguments.

**Resolution:** `doctor` now parses/validates the four parameters; threads
`stale_days` into stale findings; returns the `DOCTOR_FINDINGS` outcome with
the full report on `err.data` and `findings`/`score` fields for `strict`;
implements safe `fix` (reviewed-concept skipping, link repair, orphan
FileHash/DirHash clearing) and reports `skipped_reviewed`; and the facade
forwards the keyword arguments. `parity/scenarios/doctor-params.json` covers
custom thresholds, fix counts, strict outcome, and `cache_dir` acceptance.
Timestamp normalization is a no-op for canonical v10 TIMESTAMP columns (the
doctor fixture reports 0, matching Python).

**Tests:** `doctor-params` op/DB parity and the full matrix; `cargo test`
all/no-default, clippy green.

---

### BUG-046 — `doctor` environment/info probes were not ported

**Severity:** P3, M — **Status: closed** (fix series after `cee784a`).

**Where:** `src/ops/mod.rs:1315-1518` (`doctor` report construction).

**Evidence:** Python `DoctorManager.diagnose()` emits `info` entries for
`search_index`, image routes, ORT runtime/wheel hints, converter cache, PDF
converter state, cache hygiene, embed/rerank/image lanes, stale write locks,
and the `db_created` state. The Rust implementation emits only
`hub_concentration`; the parity harness ignores the entire `info` array
(`--ignore-key info`), so the gap is invisible.

**Impact:** `okf doctor` gives operators a materially poorer report than the
Python release and can hide missing indexes, unavailable search, an
unexpected ORT wheel, a stale lock, or a warm/released PDF converter.

**Resolution:** `Store::index_inventory()` queries `SHOW_INDEXES()` on a
connection with vector/FTS loaded, producing the same `search_index` lines and
`missing_index` findings as the oracle. `doctor` now emits `stale_omni_images`
and `image_routes` (with pinned model/precision), probes `<db>.lock`
non-blockingly (`stale_lock`/`db_lock_held`), and appends the same three
`*_lane` lines as the Python admin layer. `compare.py --ignore-info-rule`
hides only `ort_runtime`, `ort_gpu_unused`, `converter_cache`,
`cache_hygiene`, `pdf_converter`, and `db_created`; the rest of `info` is
compared exactly. PDF converter state needs a status ABI extension and
ORT/cache/hygiene probes are host-specific, so they remain documented omissions.

**Tests:** `doctor-empty`, `doctor-params`, and `read-only` compare all
portable `info` entries (0 differences); unit tests cover HNSW metric parsing
and lane-message shape.

---

### BUG-048 — `doctor --fix` bypassed the write lock; lock sidecars leaked

**Severity:** P2, S — **Status: closed** (fix series after `cee784a`).

**Where:** `src/ops/mod.rs` (`Router::call`, `Router::doctor`, new
`with_write_lock`/`cleanup_write_lock_file`), `bindings/python/src/lib.rs`
(`NativeRouter.close`), `src/bin/parity-run.rs`.

**Evidence:** `doctor` is a read op, so `call()` never took the inter-process
write lock. With `fix=True` it nevertheless repaired links and deleted delta
rows, violating the single-writer contract. Separately, Rust left `<db>.lock`
on disk after release while Python 0.15 unlinks an idle lock file, which made
the new `doctor` stale-lock probe report a phantom leftover in later ops of
the same process.

**Resolution:** the lock acquisition loop was extracted into
`with_write_lock`, used by all write ops and by `doctor(fix=True)`. A best-effort
`cleanup_write_lock_file()` mirrors Python's idle-lock unlink and is called
after `doctor --fix`, by `NativeRouter.close`, and at parity-run shutdown.

**Tests:** `doctor-params` op/DB parity (including stable stale-lock info),
`check_lock_interop.py`, and `check_read_only.py`.

---

### BUG-047 — Unit tests shared the mutable oracle fixture

**Severity:** P3, S — **Status: closed** (fix series after `391b46a`).

**Where:** `src/store/mod.rs` (`read_only_store_refuses_writes`,
`opens_oracle_fixture`), `src/ops/mod.rs` (`opens_fixture_and_reads`).

**Historical evidence:** three tests opened
`parity/fixtures/oracle-v10.db` directly. Because Ladybug can leave checkpoint
recovery state around an open/close, running the suites in sequence (or in
parallel) occasionally produced `checkpoint recovery pending` and a spurious
failure, while the same test passed in isolation.

**Resolution:** each test copies the fixture into a fresh `TempDir` before
opening it. The fixture is no longer mutated by the unit suite, and both
`--no-default-features` and `--all-features` runs are stable.

**Tests:** both library suites pass consecutively without restoring the fixture
between them.

---

## 7. Required regression/parity scenarios

Add these before calling the beta usable:

| Scenario | Covers |
| --- | --- |
| `ingest-thoughts-bundle-root` | BUG-005, BUG-019 |
| `ingest-bib-bundle-root` | BUG-005, BUG-033 |
| `ingest-materialize` | BUG-017 |
| `ingest-params` | BUG-016, BUG-018 |
| `import-dim-256` / `import-dim-768` | BUG-008 |
| `import-identity-frontmatter` | BUG-006 |
| `import-resource-sanitize` | BUG-007 |
| `import-reimport-stamps` | BUG-003, BUG-004 |
| `import-prune-missing` | BUG-021 |
| `import-multi-root` | BUG-020 |
| `delete-file-backed` | BUG-009 |
| `images-prune` | BUG-010, BUG-011 |
| `export-filtered` | BUG-025 |
| `export-dotted-id` | BUG-025 |
| `yaml-block-and-nested` | BUG-023, BUG-024 |
| `time-microseconds-offset` | BUG-022 |
| `percent-escape-nonascii` | BUG-002 |
| `pdf-real-fixture` | BUG-034 |
| `ingest-office-formats` | BUG-034, BUG-050 (DOCX/XLSX/PPTX success, malformed PPTX typed failure) |
| `external-lanes-mock` | BUG-013 |
| `pins-mismatch` | BUG-014, BUG-015 |
| `facade-api` | BUG-026, BUG-027 |
| `lock-and-read-only` | BUG-029 |
| `doctor-params` | BUG-045, BUG-046, BUG-048 |
| `doctor-empty` | BUG-046 |
| `op-names` | BUG-038 |
| `op-error-matrix` | BUG-044 |
| `settings-open-and-env` | BUG-067, BUG-068, BUG-069, BUG-071 |
| `cli-settings-parsing` | BUG-070 |
| `import-named-roots` | BUG-072 |
| `file-free-writes` | BUG-073 |
| `detached-write-refusal` | BUG-074 |
| `delete-atomic-read-only` | BUG-075 |
| `recover-window` | BUG-076 |
| `traverse-path` | BUG-077 |
| `search-time-and-params` | BUG-078 |
| `frontmatter-null-id` | BUG-079 |
| `frontmatter-type-errors` | BUG-080 |
| `links-code-encoding-ambiguity` | BUG-081 |
| `import-uppercase-extensions` | BUG-082 |
| `images-corrupt-and-title` | BUG-083 |
| `images-asset-traversal` | BUG-084 |
| `bibtex-sparse-entry` | BUG-085 |
| `export-existing-links` | BUG-086 |
| `ingest-pdf-multiple` | BUG-087 |
| `jobs-reembed-lock` | BUG-088 |
| `mcp-http-serialization` | BUG-089 |

For each scenario, include both success and typed-error envelopes, and compare
the semantic DB dump plus the export tree where applicable.

---

## 8. Release gates

### Gate A — after Phase 0

- No panic from `OkfError::new`/`kind` or `percent_decode`.
- All error codes emitted by the crate are in the closed set.
- `cargo test -p okfgraph --no-default-features` and clippy green.

### Gate B — after Phase 1

- Re-import preserves `created_at`, `updated_at` when `content_sig` is
  unchanged, and all old links on encode failure.
- Thoughts/BibTeX ingest works with `bundle_root`.
- 256/768-dim init → reopen → import/search works.
- File-backed delete and image pruning are covered.
- No orphan `ImageAsset` rows after a reference removal.

### Gate C — after Phase 2–3

- `identity="frontmatter"` and resource sanitization match Python.
- No path traversal from concept IDs.
- `ingest` parameter matrix matches Python.
- Thoughts/BibTeX/markdown output and YAML/time bytes match the oracle.
- Pins (model/precision/space/dim) and `lanes` are truthful.

### Gate D — before stable cutover

- Multi-root and prune/force are implemented or explicitly removed from the
  documented API.
- External lanes are implemented and covered by mock-host differential checks.
- Python facade signature/method/error parity is verified against 0.15.
- Schema migration fixtures pass and Python 0.15 can reopen Rust-written DBs.
- File lock interoperability is tested across Python and Rust processes.
- PDF/Office end-to-end parity has a real fixture.

### Gate E — after the post-cutover review

- A fresh `OKFRouter`/CLI open creates a v10 database; every graph opens with
  its schema-adopted width, not the caller’s default.
- TOML, environment variables, default bundle roots and CLI `--root`/CSV
  flags reach `Settings`; unknown settings and invalid values fail typed.
- Named-root update/delete/import/detach resolve the owning tree and leave no
  stray files; file-free update/supersede works.
- Detached graphs refuse writes unless a successful `--force` reattach;
  `delete(remove_file=True)` is atomic under failure and read-only mode.
- `recover_deleted` obeys the 24-hour window.
- Traverse path mode, search time bounds, frontmatter/link parsing and
  uppercase extensions match the oracle scenarios in section 7.
- Corrupt images fall back to captions; `okf-asset://` cannot read outside the
  bundle.
- Reembed jobs take the process write lock; MCP tools share the owner lock;
  job records match Python’s phase/elapsed/`JOB_OPS` contract.
- New scenarios and typed-error coverage are in `run_matrix.py` and
  `check_coverage.py`, with no waivers for these findings.

---

### BUG-062 — Lone CR frontmatter diverged from Python universal newlines

**Severity:** P2, S — **Status: closed** (M6 fuzz hardening).

**Where:** `src/markdown/mod.rs`, `parity/scenarios/frontmatter-cr-only.json`,
`tests/fuzz_robustness.rs`.

**Cause:** `parse_frontmatter` normalized CRLF to LF but left lone `\r` bytes.
Python's `Path.read_text` uses universal-newline translation, so CR-only
frontmatter produced different titles, tags, bodies and content signatures.

**Resolution:** both CRLF and lone CR are normalized to LF before delimiters
are split or YAML is parsed. The proptest seed is checked in, a unit regression
covers CR-only delimiters and bodies, and the differential harness can write a
byte-exact CR-only fixture via `setup.write_file`.

**Tests:** `tests/fuzz_robustness.rs`,
`markdown::tests::universal_newlines_normalize_lone_carriage_returns`, and
`parity/scenarios/frontmatter-cr-only.json` (ops and DB dumps green).

### BUG-063 — Rust performance was below the Python 0.15 oracle

**Severity:** P1, L — **Status: closed** (M6 performance hardening).

**Where:** `src/store/mod.rs`, `src/ops/mod.rs`, `examples/perf_bench.rs`,
`parity/tools/check_perf_bench.py`, `docs/performance-findings.md`.

**Cause:** `Store` reopened a Ladybug DB and loaded extensions for every
query, while Python 0.15 held a long-lived connection. Per-query pin checks,
two separate search connections and oversized row fetches added further cost.

**Resolution:** the shared `self_cell` connection is reused for reads and
writes, extensions load once, vector/FTS candidates share one connection,
search rows fetch only search-visible fields, and pin checks are cached per
`Router` with write/reembed invalidation. The paired benchmark now enforces a
parity gate.

**Evidence:** `docs/performance-baseline.md`: import 0.70x Python, embed
0.68x, search 1.06x, peak RSS 0.88x. Delta re-import is 0.011s vs 0.005s and
passes the explicit 5ms sub-10ms tolerance.

**Tests:** `parity/tools/check_perf_bench.py`, full 57-scenario matrix,
Rust unit/fuzz suites and wheel/MCP/client/CLI smokes are green.

### BUG-064 — Remote-image SSRF checks were host-string only

**Severity:** P1, S — **Status: closed** (M6 security hardening).

**Where:** `src/image_ingest.rs`.

**Cause:** the domain allowlist checked the URL host text, but reqwest could
follow a redirect and DNS could rebind to a private address. The whole
response was buffered before the size cap was applied.

**Resolution:** resolved addresses are screened against IPv4/IPv6 private,
loopback, link-local, CGNAT, documentation and reserved ranges; one safe
address is pinned for the request; redirects are disabled; and response bytes
are stream-capped at 20 MiB. Failed remote fetches now keep the Python
caption-row asset so the next import retries.

**Evidence:** Rust unit tests cover blocked and allowed IP ranges;
`parity/scenarios/images-ssrf-private.json` compares both runners for a
loopback image URL (ops and DB dumps green).

### BUG-065 — Extension setup checked presence, not loadability

**Severity:** P2, S — **Status: closed** (M6 security hardening).

**Where:** `src/util/extensions.rs`, `src/cli/local.rs`.

**Cause:** `setup extensions` reported `ok` when the two native files merely
existed, so a corrupt or wrong-ABI file could be installed into
`extension_dir` undetected.

**Resolution:** the CLI now loads both files into a scratch Ladybug database
and reports `{ext: "ok" | error}` exactly like Python 0.15's verifier.

**Evidence:** `util::extensions::tests::missing_extensions_are_reported` and
the 13-case CLI differential smoke.

### BUG-066 — Rust INTERNAL errors had no redacted crash report

**Severity:** P2, S — **Status: closed** (M6 security hardening).

**Where:** `src/util/crash.rs`, `src/cli/mod.rs`.

**Cause:** Python 0.15 writes a full traceback named on the final `[ERROR]`
line; the Rust CLI had no equivalent, so a truncated terminal capture lost
the diagnostic context.

**Resolution:** `INTERNAL` CLI errors write a redacted report to
`OKF_CRASH_DIR` (or the platform state dir), keep the newest 20, and add
`fields.crash_report` plus the Python remedy. Home paths, bearer tokens and
key/token/password values are redacted before writing.

**Evidence:** `util::crash` unit tests cover redaction, report contents and
pruning; the CLI differential smoke remains 13/13.

---

## 8A. Post-cutover review findings (BUG-067–BUG-089, open)

**Review baseline:** `a62d1c9` (`docs: refresh Rust-first 0.16 documentation set`).
This review exercised the public Python facade, `okf`, `okf serve`, and the
Rust API directly against the frozen `python-oracle-0.15.0` behavior. The
existing 58/58 differential matrix is green, but none of the scenarios below
is covered by it.

The findings are grouped by root cause. Each needs the normal close rule:
code fix + regression test + a differential scenario (or a documented
exclusion) + a status update in section 11.

### BUG-067 — Router discards the schema-adopted embedding width

**Severity:** P1, M — **Status: open**.

**Where:** `src/ops/mod.rs:168-205` (`Router::open`,
`Router::open_read_only`, `Router::open_writable`), `src/store/mod.rs:92-100`
(`Store::open`).

**Cause:** `Store::open` correctly reads `Concept.embedding` from
`TABLE_INFO` and changes its own `settings.embedding_dim` to the stored width.
Every `Router` constructor then stores `settings.clone()` from the caller
instead of `store.settings().clone()`. `Rust` therefore keeps 512 for a
256- or 3-dim graph while the Store, pins, and schema say otherwise.

**Evidence:** opening the 3-dim `parity/fixtures/oracle-v10.db` through the
public facade with default settings fails at the first model-backed op with
`DIM_MISMATCH: graph stores 3-dim vectors, configured 512`. A 512-dim graph
opened with `--embedding-dim 256` behaves the same way.

**Impact:** the “on-disk width is authoritative” rule from BUG-008 only reaches
`Store`. `embed`, `reembed`, model pin checks, and any path that reads
`Router.settings.embedding_dim` use a stale value. This blocks opening any
existing graph whose width differs from the current default or flag.

**Fix:** build all three `Router` values from the `Store` after adoption, e.g.
`let store = Store::open_writable(settings)?; let settings = store.settings().clone();`.
Audit every `self.settings.embedding_dim` use after that; it should be the
adopted value. Keep `db_created` derived from the original path (path equality
is unaffected).

**Tests:** open 3/256/768-dim fixtures with mismatched `--embedding-dim`;
assert `router.settings().embedding_dim` equals the schema width,
`pin_info().pinned.embedding_dim` equals it, `embed` works, and reembed refuses
only for a genuinely different engine width.

### BUG-068 — Public router no longer creates a fresh database

**Severity:** P1, M — **Status: open**.

**Where:** `src/store/mod.rs:77-85` (`Store::open`), `src/ops/mod.rs:168`
(`Router::open`), `python/okfgraph/router.py`.

**Cause:** `Store::open` requires `Path::exists()` and returns
`FILE_NOT_FOUND` otherwise. Python 0.15's `OKFRouter(db_path=...)` creates the
parent directory and initializes the v10 schema on first use. `Router::init`
exists, but the public facade never calls it. `Router::db_created_for` even
special-cases `:memory:`, while `Store::open` rejects it before that branch is
reachable.

**Evidence:** `OKFRouter(db_path=<fresh temp path>)` raises
`FILE_NOT_FOUND: database not found`. The README and `docs/quickref.md` both
show `OKFRouter(db_path="graph.db", bundle_root="./notes")` without a prior
`okf init`. `okf search`/`okf import` on a missing default database behave the
same way.

**Impact:** the documented Python quick start is broken, first-run CLI flows
that rely on Python's create-on-open behavior fail, and `:memory:` is
unreachable despite being advertised in the code.

**Fix:** add an explicit open-or-init policy used by the public facade and CLI:
create parent directories, initialize the v10 schema when the file is absent,
and keep `okf init` as the explicit operation. For an existing file, still
refuse missing/old schema versions. Wire `:memory:` through the same path or
remove the unsupported claim.

**Tests:** fresh `OKFRouter` creates a v10 DB with the requested width;
reopening it works; an empty zero-byte file is handled intentionally; old and
future schema versions still fail with `CONFIG_INVALID`; `:memory:` either
works or returns a clear `CONFIG_INVALID`.

### BUG-069 — `Settings::load` is dead code: env, TOML, validation, and default root never run

**Severity:** P1, L — **Status: open**.

**Where:** `src/settings/mod.rs:222-310` (`Settings::load`, legacy env
rejection, TOML merge, `checks`), `src/settings/mod.rs:512-530`
(`default_bundle_root`), `src/cli/mod.rs:1297-1333` (`settings_from`).

**Cause:** `Settings::load` has no production call site. `settings_from` only
collects command-line flags and calls `Settings::from_map`; it never reads
`okfgraph.toml`, `OKFGRAPH_*` environment variables, legacy names, or
`default_bundle_root`. `Settings::checks()` is therefore not run for CLI,
MCP, `serve`, parity-run, or facade settings either.

**Evidence:**

```text
OKFGRAPH_DB_PATH=<fixture> okf doctor --json
  -> FILE_NOT_FOUND: database not found: .okfgraph/okfgraph.db
```

After `okf init` in a `.okfgraph` home, `okf import --all` fails with
`MISSING_PARAM: import_bundle needs a bundle path or configured bundle_root`
even though Python would use `.okfgraph/bundle`. `OKFGRAPH_DB` and other
renamed variables are also never rejected, so their migration errors never
surface.

**Impact:** environment and TOML configuration are silently ignored on every
Rust surface; the documented precedence “TOML < env < CLI” is false. Default
bundle roots differ from Python (CLI `.`/`.okfgraph/bundle`, serve/MCP the db
folder), so default imports and serve/MCP roots are wrong. Invalid settings
pass through because `checks` is dead.

**Fix:** make `Settings::load` the single production loader (or call it from
`settings_from` with the real environment and CLI map), then apply
`default_bundle_root` in the CLI/serve/MCP open paths. Ensure `checks` runs
for every surface, including the Python facade. Keep `Settings::from_map` as a
typed internal constructor only.

**Tests:** env-only `okf doctor`; `okfgraph.toml` precedence over env and env
over flags; legacy env rejection; default root for `.okfgraph` homes and
non-`.okfgraph` paths; invalid settings report `CONFIG_INVALID`; MCP required
`db_path` behaves like Python.

### BUG-070 — `--root` and CSV global flags fail before any command runs

**Severity:** P1, S — **Status: open**.

**Where:** `src/cli/mod.rs:1319-1331`, `src/settings/mod.rs:188-194`.

**Cause:** `settings_from` inserts every `--root` occurrence as a JSON array
of `ALIAS=PATH` strings, while `Settings.roots` deserializes as
`BTreeMap<String, String>`. It inserts every Csv setting (for example
`--allowed-image-domains`) as one string, while the struct expects
`Vec<String>`. `Settings::from_map` does not normalize either type.

**Evidence:**

```text
okf doctor --root other=parity/corpus/multi-root/other
  -> CONFIG_INVALID: invalid settings: invalid type: sequence, expected a map

okf doctor --allowed-image-domains a.com,b.com
  -> CONFIG_INVALID: invalid settings: invalid type: string, expected a sequence
```

**Impact:** named roots are unusable from the CLI even though they are a
documented multi-root feature, and at least one exposed CSV flag cannot be
used at all. This also blocks fixing BUG-072 through the CLI.

**Fix:** parse `--root` entries into an alias→path object (reject missing
`=`, empty alias/path, and duplicate aliases consistently with Python) and
parse Csv flags with the same split/trim/drop-empty rule as `Settings::load`.
Alternatively route all CLI values through `parse_value` before
`from_map`, but keep `from_map` strict for programmatic callers.

**Tests:** CLI `--root` with one and several aliases; duplicate/malformed
aliases; `--allowed-image-domains a.com,b.com`; CSV empty entries; the same
values via env/TOML; all produce the expected settings and no `CONFIG_INVALID`.

### BUG-071 — Unknown settings are silently ignored; validation and explicit model files are missing

**Severity:** P1, M — **Status: open**.

**Where:** `src/settings/mod.rs:171-195` (`Settings::from_map`),
`src/settings/table.rs` (no `model_path`/`tokenizer_path` rows),
`python/okfgraph/router.py`.

**Cause:** `from_map` copies only names present in `SETTINGS` (plus `roots`)
into `full`; any other key is dropped. It also never calls `checks`, so
out-of-range and choice-invalid values survive. The Python facade forwards
arbitrary kwargs, so a misspelled setting becomes a silent default.

**Evidence:** all of these are accepted without an error:

```python
OKFRouter(db_path=db, bogus_setting=1)
OKFRouter(db_path=db, embedding_dim=8)
OKFRouter(db_path=db, model_path="x.onnx", tokenizer_path="y.json")
```

Python 0.15 rejects the first two (`TypeError`/`ValueError`) and uses the last
two for air-gapped explicit ONNX files. Rust has no table rows or loader path
for them, so it silently opens the registry model instead.

**Impact:** typo’d configuration appears to work but changes nothing; invalid
dimensions/devices/backends can reach schema creation or the embedder; users
who depend on explicit local model files get a network/model-cache path
instead of a clear refusal.

**Fix:** reject unknown keys in `from_map` (the facade should map Python’s
`embed_backend` alias first), run `checks` before returning a settings object,
and either port `model_path`/`tokenizer_path` end-to-end or return a typed
`CONFIG_INVALID` saying explicit local files are unsupported. Do not accept a
setting and ignore it.

**Tests:** unknown kwarg → `CONFIG_INVALID`; `embedding_dim=8`,
`chunk_size=1`, invalid `device`/choices → `CONFIG_INVALID`; documented
explicit-file pair either loads or fails explicitly when one is missing;
facade and CLI share the same validation.

### BUG-072 — Named-root paths and import identity are not threaded through writes

**Severity:** P0, L — **Status: open**.

**Where:**

- `src/store/mod.rs:1089-1109` (`resolve_file_paths`),
  `1112-1123` (`source_file_for_concept`).
- `src/ingest/mod.rs:81-84` (`import_files` parses with `root=None`),
  `531-550` (`alias_for_path` is exact-match only), `106-405`
  (`import_bundle_one`).
- `src/ops/mod.rs:687-709` (`write_and_import` strips `@alias/` but joins the
  primary root), `304-385` (`delete`/`trash_files`).
- `src/cli/mod.rs:1040-1077` (positional `--bundle-path` is passed but never
  consumed).
- `src/detach.rs:264-380` (single-root detach; clears all ledger rows).

**Cause:** FileHash keys are namespaced (`@alias/relative`), but path
resolution only ever consults `settings.bundle_root`. For an `@alias/...` key
it tries `primary/relative` first and `primary/@alias/relative` second; it
never tries `settings.roots[alias]/relative`. `import_files` has no
`identity_root`/alias lookup, and `write_and_import` deliberately strips the
alias before joining the primary root. `detach` only verifies one tree but
clears `FileHash`/`DirHash`/`DeletedPath` for every root.

**Impact:** with a named root configured, a file-backed concept can resolve
to a same-named primary-root file (wrong file edited), materialize a stray
file into the primary tree, fail “imports as X, not @alias/Y”, or delete the
graph concept while leaving the real file in the named root. Single-file
imports from a secondary root mint bare or stem-only IDs, so they collide with
primary concepts. `detach` can discard the provenance ledger of roots it did
not verify. `okf import --bundle-path DIR file.md` ignores the explicit
identity root, unlike Python’s `identity_root`.

**Fix:** centralize a root resolver that maps an alias/key to
`settings.roots[alias]` (with the documented primary fallback only for unknown
aliases), use it in `resolve_file_paths`, `delete`, `trash`, recovery,
detach verification, and materialization. Add `identity_root`/alias handling
to `import_files` and the CLI, and implement Python’s longest-prefix
`resolve_alias_for_path` for subdirectories. Port `_detach_multi` or explicitly
refuse detach when named roots are configured.

**Tests:** multi-root bundle with identical relative names in primary and
alias roots; update/delete/supersede each resolve the correct file; import a
single file and a subdirectory in a named root; `--bundle-path` with positional
files mints IDs relative to that tree; detach verifies every root or refuses.

### BUG-073 — File-free `update` and `supersede` are refused

**Severity:** P1, M — **Status: open**.

**Where:** `src/ops/mod.rs:653-717` (`write_and_import`), `759-928`
(`update`/`supersede`).

**Cause:** when there is no source file and `bundle_root` is `None`,
`write_and_import` returns `MISSING_PARAM: materialize needs a bundle root`.
Python’s `_write_and_import` explicitly supports this mode: it edits the
stored concept and re-runs the single-concept pipeline, re-embedding and
preserving incoming links.

**Impact:** the documented file-free router can search/read/traverse/doctor
but cannot update or supersede concepts. This is a public API regression, not
just a missing convenience.

**Fix:** mirror Python’s third branch: apply the edit to the graph concept,
run the same single-concept import/encode path with a synthetic `Parsed`, and
preserve incoming links/timestamps. Keep materialization only when a bundle
root exists.

**Tests:** file-free update changes title/tags/extra and re-embeds; incoming
links survive; supersede writes relations and tags in both directions;
materialized and file-backed modes still produce the same result as before.

### BUG-074 — Detached graphs accept writes; `force`/`rebuild`/`batch_size` are ignored

**Severity:** P0, M — **Status: open**.

**Where:** `src/detach.rs:246-377` (only refuses a second detach),
`src/ops/mod.rs:1141-1167` (`import_bundle`), `1170-1206`
(`import_file`/`import_files`), `src/ingest/mod.rs:101-106`
(`ImportOptions.force` is never read), `src/converters.rs` (ingest has no
detached check).

**Cause:** Python calls `_require_attached(force)` from all import/ingest
paths; a detached graph refuses writes with `DETACHED` unless `force` performs
an explicit reattach. Rust has no equivalent guard. `force` is accepted in
`ImportOptions` but never consulted, and `rebuild`/`batch_size` are accepted by
the CLI/op signatures but have no implementation in the import pipeline.

**Impact:** after `okf detach`, imports and ingests still mutate the graph even
though the source tree may be gone, violating the detach contract and creating
a graph that can no longer be audited against a bundle. `--force` cannot
re-attach; `--rebuild defer/force` silently behaves like `auto`; `batch_size`
does not bound import batches.

**Fix:** add a shared `require_attached(force, root)` check to import, ingest,
single-file writes, and materialization. Implement the Python reattach rules
(root must match, every configured root present) or document and reject
`force` explicitly. Implement the index policy for `rebuild` and either honor
`batch_size` in import encoding or reject it as unsupported; never accept a
control that does nothing.

**Tests:** detached `import`/`ingest`/`update` → `DETACHED`; `force` with the
matching root re-attaches; `force` with a different root refuses; `rebuild
defer` skips index creation and `force` rebuilds; `batch_size` changes encode
batch boundaries (or a typed refusal is asserted).

### BUG-075 — `delete` moves files before the database mutation is safe

**Severity:** P0, M — **Status: open**.

**Where:** `src/ops/mod.rs:266-325` (`delete`), `333-385` (`trash_files`).

**Cause:** `delete` resolves live files, moves them to `.okf-trash`, forgets
their FileHash rows, and only then calls `purge::soft_delete`. If that write
fails (read-only Store, disk/lock error, missing table) the concept remains
live but its files are gone from the bundle and no tombstone records the
trash. `trash_files` also commits each move/forget independently, so a
mid-list rename failure leaves a partial move.

**Impact:** data loss or an unrecoverable half-delete, especially through
`Router::open_read_only` where the write refusal happens after the file move.
The user sees an error but the bundle has already changed.

**Fix:** preflight that the Store is writable before touching the filesystem;
perform the tombstone/snapshot write first or stage all moves and roll them
back if the DB transaction fails; record the trash list on the tombstone before
moving when possible, or use a recoverable journal. Make the whole operation
idempotent so a retry can finish a partial move.

**Tests:** read-only `delete(remove_file=True)` changes no file; injected
soft-delete failure restores every moved file and ledger row; injected rename
failure leaves the concept and files intact; retry after a simulated crash
finishes cleanly.

### BUG-076 — Recovery ignores the 24-hour window

**Severity:** P1, S — **Status: open**.

**Where:** `src/purge.rs:128-176` (`recover`), `src/ops/mod.rs:386-430`
(`recover_deleted`), `src/store/mod.rs:1244-1285` (`list_deleted`).

**Cause:** Python’s `_recover_concept_inner` parses `deleted_at` and returns
false when `(now - deleted_at) > SOFT_DELETE_WINDOW` (24h). Rust’s `recover`
never reads or compares `deleted_at`; it restores any snapshot that exists.

**Impact:** `list_deleted` can report `recoverable: false` while
`recover_deleted` still succeeds past the promised window. Tombstones are
supposed to expire into `purge_deleted`; recovery after expiry violates the
lifecycle and can resurrect data that should have been collected.

**Fix:** use the latest tombstone’s `deleted_at`, parse it with the same
clock/timezone rule as Python, and return false (or the typed
`NOT_RECOVERABLE` at the op layer) past 24h. Keep invalid/missing timestamps
conservative and consistent with `list_deleted`.

**Tests:** now-1h recovers; now-25h refuses with `NOT_RECOVERABLE`;
`list_deleted.recoverable` and `recover_deleted` agree; malformed
`deleted_at` follows the documented behavior; timezone-offset timestamps use
the same wall-clock rule as the oracle.

### BUG-077 — `traverse(target=...)` returns the wrong path and ignores path-mode rules

**Severity:** P1, M — **Status: open**.

**Where:** `src/ops/mod.rs:2599-2708` (`traverse`, `find_path`).

**Cause:** `find_path` is a custom BFS that only walks `LINKS_TO` and
`CONTAINS`, never `PART_OF`/`INCLUDES_ASSET` or directory intermediates, and
returns full node objects. It initializes the path empty, so the start node is
missing; when `start == target` it returns `[]`. `max_len` is clamped to
1..=6 while Python allows 1..=10. Empty `start_id` is handled before the
`target` check, so a target without a start silently lists the root instead
of raising. Path mode never rejects `relationship`, `depth`, `node_type`, or
`link_kinds`, which Python reports as ignored.

**Evidence:**

```text
okf traverse alpha --target beta
  -> [{ "id": "beta", ... full node ... }]
```

Python returns alpha **and** beta, projected to `{id, title, type}`.

**Impact:** path queries through directories/chunks/assets fail; the returned
shape is not the documented one; long paths are cut off at 6; invalid
argument combinations silently do the wrong thing.

**Fix:** use the oracle query shape (`SHORTEST 1..N` over the four whitelisted
relationship types, both directions) or an equivalent BFS that includes the
start and projects only `id/title/type`; allow 1..=10; move the empty-start
check before target handling; port the ignored-parameter matrix.

**Tests:** direct link path includes both endpoints; path through a directory,
a chunk, and an image asset; no path returns `[]`; >6-hop path; start==target;
invalid empty-start+target; path-mode ignored params produce the same typed
error as Python.

### BUG-078 — Search time bounds are string comparisons; parameter validation is incomplete

**Severity:** P1, M — **Status: open**.

**Where:** `src/ops/mod.rs:3143-3152` (`since`/`until`), `3933-3952`
(`recency_weight`), `4099-4107` (`query_vector`), `SearchParams::parse`.

**Cause:** Python’s `_as_ts` parses `since`/`until` into datetimes and binds
them to `TIMESTAMP` columns. Rust compares the stored `effective_at` string
lexicographically against the raw caller string. Date-only bounds therefore
behave differently (`until=2024-01-02` still includes that day), offsets and
alternate ISO spellings sort incorrectly, and invalid values are accepted.
`recency_weight` falls back to `0.3` for any non-number instead of raising;
`query_vector` silently drops non-numeric entries and accepts an empty or
wrong-width vector.

**Impact:** filtered searches can include/exclude the wrong day or range,
while invalid parameters that Python rejects produce silently different
results. An empty query vector can reach the vector index and fail later with
an unrelated error.

**Fix:** parse both bounds with the shared `PyDateTime` parser, validate them
as Python does, and compare typed timestamps (or normalized UTC strings).
Reject non-numeric/bool `recency_weight`; validate `query_vector` as a
non-empty numeric array of the engine/graph width. Keep error fields/remedies
in the same shape as Python.

**Tests:** date-only `since`/`until` boundaries; Z/offset/fractional
spellings; invalid bounds → `BAD_VALUE`; string/bool `recency_weight`;
`query_vector` with a string entry, empty list, wrong width, and valid
vector; compare result IDs and envelopes with the oracle.

### BUG-079 — Null frontmatter `id:` becomes the concept id/uid `"None"`

**Severity:** P1, S — **Status: open**.

**Where:** `src/ingest/mod.rs:885-914`, `1074-1084` (`scalar_text`).

**Cause:** Python guards with `fm_id is not None` and only uses a non-empty
string. Rust converts `Json::Null` with `scalar_text` to the literal `"None"`,
which then passes `safe_frontmatter_id` (letters only) and is used as the
concept id under `identity="frontmatter"` or as `uid` under path identity.

**Impact:** a file containing an empty `id:` imports as concept `None`, or
gains `uid: None`, instead of keeping its path identity. That can create a
wrong node and broken/incorrect wikilink resolution.

**Fix:** treat `Json::Null` as absent before stringifying; only string values
(and the documented scalar cases Python actually stringifies) should feed the
id/uid path. Add a dedicated null-empty branch rather than relying on
`scalar_text`.

**Tests:** `id:` (null), `id: ""`, `id: 0`, `id: false`, and no `id:` under
both identity modes; compare concept id/uid and DB dump with Python.

### BUG-080 — Frontmatter type/YAML errors are silently defaulted or dropped

**Severity:** P1, M — **Status: open**.

**Where:** `src/ingest/mod.rs:929-980` (`string_field`,
`parse_tagged_datetime`, title/type/description/resource defaults),
`src/markdown/mod.rs:191-650` (`parse_yaml`, `parse_float`, `tagged_date`).

**Cause:** Pydantic validates `title`/`description`/`resource` as optional
strings, `type` as a required string, and `timestamp` through
`datetime.fromisoformat`; PyYAML raises on malformed YAML. Rust’s
`string_field` returns `None` for a numeric/bool value (then silently uses the
stem/default), invalid timestamps become `None`, and unterminated quotes or
flow collections fall through as raw strings. Several double-quoted escapes
and anchors/aliases are also treated literally.

**Impact:** authored metadata can change or disappear without any error,
content signatures and exports drift from Python, and a typo in frontmatter
is much harder to detect. This is a data-quality bug even when no crash occurs.

**Fix:** validate core frontmatter fields with the same rules as
`ConceptModel` and return a typed `BAD_VALUE`/parse error naming the file and
field. Make the YAML subset fail closed on malformed input rather than
guessing; either implement the relevant PyYAML escapes/anchors or explicitly
reject them with a clear message. Keep supported scalar coercion documented.

**Tests:** numeric/bool `title`, `type`, `description`, `resource`;
invalid/date-only/offset `timestamp`; unterminated single/double quotes and
flow arrays/maps; `\n`/`\u` escapes; anchor/alias input; compare error
envelopes or parsed metadata with the oracle.

### BUG-081 — Link resolution diverges on code spans, encoding, escapes, and >2 ambiguous names

**Severity:** P1, M — **Status: open**.

**Where:** `src/ingest/mod.rs:1495-1524` (`LinkIndex::add`), `1936-2068`
(`markdown_links`, `strip_code`, `link_target`, `wikilinks`).

**Cause/four symptoms:**

1. `LinkIndex::add` removes a colliding key from the map but re-inserts the
   third claimant: the `else if` only checks candidate membership for the new
   id. A name claimed by three concepts can resolve instead of remaining
   ambiguous. Python pops every claimed key and never restores it.
2. `wikilinks()` reads the raw body; Python runs `strip_code` first. Links in
   inline code or fences become real edges. `markdown_links` strips fences but
   not inline code spans.
3. `link_target` never percent-decodes the path; Python calls `unquote`, so
   `my%20note.md` resolves there but not here.
4. `..` segments are popped without tracking depth. An escaping path such as
   `a/../../b` or `/../a` collapses to `b`/`a` and can resolve to a real
   concept; Python’s `normpath` plus the `..`/`../` check marks it broken.

**Impact:** phantom links from code blocks, missed links with encoded spaces,
wrong link targets when names collide three ways, and links that should be
broken resolving to unrelated concepts.

**Fix:** implement Python’s ambiguity bookkeeping (remove on first collision
and never re-add), use `strip_code` for both markdown and wiki extraction,
percent-decode before normalization, and track the normalized path depth so
escaping links return `None` with the same recorded target. Keep case-folding
consistent with Python’s `str.lower()` for non-ASCII if parity requires it.

**Tests:** 3-way duplicate title/stem/uid; links inside inline code, fenced
code, and code with a different fence length; `%20`/UTF-8 escapes; `..` at
root, inside a subdirectory, and with a leading slash; compare edges,
BrokenLink rows, and `diff` output with the oracle.

### BUG-082 — Bulk source discovery is case-sensitive on the extension

**Severity:** P1, S — **Status: open**.

**Where:** `src/ingest/mod.rs:709-712` (`walk_dir`), reused by
`import_bundle`, `detach`, `diagnostics::lint`, and `diff`.

**Cause:** Rust matches `Some("md" | "markdown" | "txt")` exactly. Python uses
`fp.suffix.lower() in SOURCE_EXTS`, so `.MD`, `.Markdown`, and `.TXT` are
concept files.

**Impact:** uppercase-extension notes are silently skipped by import, never
verified by detach (and may be reported as non-source files, refusing detach),
and ignored by lint/diff. A tree can look empty or unchanged for reasons the
user cannot see.

**Fix:** lowercase the extension before comparison, and use one shared
predicate for import, walk, detach, diff, and lint (as Python’s
`is_concept_file` does).

**Tests:** `.md`/`.MD`/`.Markdown`/`.TXT` in import, detach verification,
lint, and diff; reserved `INDEX.MD`/`LOG.MD` behavior; mixed-case directory
names; compare file sets and DB dumps.

### BUG-083 — Image ingestion aborts on vision failures and ignores optional image titles

**Severity:** P1, M — **Status: open**.

**Where:** `src/image_ingest.rs:673-688` (`ingest_concept_images`),
`82-112` (`extract_refs`), `476-481` (`route_and_caption`).

**Cause:** Python wraps per-image vision encoding in `try/except` and falls
back to the caption route for undecodable bytes or a missing/incompatible
vision runtime; only a lane outage aborts. Rust propagates every
`pools.vision.get(...)`/`encode_image(...)` error and aborts the concept.
`extract_refs` also treats everything between `](` and `)` as the source
unless it starts/ends with `<...>`, so `![alt](pic.png "title")` produces the
wrong `src`/asset id and usually no bytes.

**Impact:** one corrupt image or unavailable optional vision lane prevents an
entire document import, where Python keeps the concept with a caption row.
Image titles change asset identity and prevent reuse.

**Fix:** capture per-image vision errors, reclassify non-outage failures to
the caption route (and record the existing failure counters/notes if the
report exposes them), and let only `EXTERNAL_UNAVAILABLE`/lane-outage errors
escape. Parse image references with optional single/double-quoted titles and
angle-bracket sources.

**Tests:** corrupt image bytes in `omni`/`optional` mode produce a caption row;
lane outage still aborts; valid image with `"title"`, `'title'`, and
`<path with spaces>`; compare asset ids, route, caption, and DB dump.

### BUG-084 — `okf-asset://` paths can escape the bundle and read arbitrary local files

**Severity:** P0, S — **Status: open (security; inherited from Python 0.15)**.

**Where:** `src/image_ingest.rs:116-128` (`okf_asset_id`), `229-250`
(`asset_store_bytes`), `382-393` (`load_bytes`).

**Cause:** `okf_asset_id` only strips the scheme and leading/trailing slashes.
`asset_store_bytes` then does `dir.join(asset_id)` (and a prefix glob) without
rejecting `..`, absolute paths, or Windows drive/colon paths, and
`load_bytes` bypasses `within_root` for this branch.

**Impact:** a markdown file can read any file the process can access, store
its bytes in an `ImageAsset`, and expose them through `get_image` as base64.
Examples: `okf-asset://../../etc/passwd`, `okf-asset://C:/Users/...` on
Windows, or a path that resolves outside the bundle. This is a local file
disclosure and also poisons the graph with attacker-controlled bytes.

**Fix:** validate the asset id as a single safe filename component (no `/`,
`\`, `:`, `..`, empty/`.`), reject absolute paths, canonicalize the candidate
and require it under the relevant asset-store/root, and keep the
`within_root` check on every local branch. Prefer an explicit allowlist of
`img_<uuid>`/stored asset ids. Decide whether to mirror the fix into the 0.15
oracle; if not, document the intentional security divergence.

**Tests:** `..`, nested `../`, absolute POSIX, Windows drive, colon, empty,
and encoded traversal ids; valid `img_<uuid>` in `_assets`; shared asset
store; assert no read outside root and no asset row for rejected ids.

### BUG-085 — BibTeX sparse entries lose the citation fallback and keep untrimmed metadata

**Severity:** P2, S — **Status: open**.

**Where:** `src/converters.rs:922-1000` (`ingest_bib`), `547-620`
(`parse_bibtex`).

**Cause:** Python builds `citation = f"{head}. {venue}".strip(" .")` and falls
back to `entry.key` when both are empty; Rust filters the empty parts and
joins, producing `""` for an entry with only a title. Python also
`.strip()`s author/year/venue/DOI/URL and applies `unwrap_braces()` to the
abstract; Rust keeps whitespace/braces in several of those fields.

**Impact:** sparse entries get an empty `description`/Citation section, and
padded or braced fields produce different body/frontmatter bytes and content
signatures than the oracle.

**Fix:** reproduce the Python citation expression and key fallback, trim every
text field before use, and unwrap braces for the abstract (and any other field
Python unwraps). Keep tags/entrytype/year ordering unchanged.

**Tests:** entry with only a title; empty author/year/venue; whitespace-padded
fields; braced abstract/title; DOI/URL padding; compare body, description,
frontmatter, tags, and DB dump.

### BUG-086 — Export “See Also” duplicates links whose target appears mid-URL

**Severity:** P2, S — **Status: open**.

**Where:** `src/export/mod.rs:234-265` (`enrich_body`).

**Cause:** Rust checks `body.contains("]({target}")`, which only matches a
link URL that begins with the target id. Python scans any `](...)` whose URL
contains the target id, so a link to `dir/target.md` suppresses the synthetic
`target.md` See Also entry there but not here.

**Impact:** exported bodies gain duplicate See Also bullets for links that
already exist under a different relative path; round-trip bodies and export
trees differ from Python.

**Fix:** parse link destinations (or use the existing link extractor) and
compare the resolved target id/path suffix, not a prefix-only substring.
Keep the raw wikilink checks and the existing title-case comparison.

**Tests:** body with `[x](target.md)`, `[x](dir/target.md)`,
`[x](../target.md)`, `[[Title]]`, and code-span-only links; assert no
duplicate See Also and compare export trees.

### BUG-087 — CLI PDF ingestion cannot take multiple paths or folders

**Severity:** P1, M — **Status: open**.

**Where:** `src/cli/mod.rs:1103-1107`, `src/converters.rs:1671-1676`.

**Cause:** the CLI always stores `pdf_path` as an array (`num_args(1..)`), but
`ingest_pdf` uses `json_string(params.get("pdf_path"))`, which returns `None`
for an array, so it reports `MISSING_PARAM`. Python 0.15 expands folders and
shell patterns and loops `_ingest_many_pdfs` over one router call per file.

**Impact:** `okf ingest --kind pdf --pdf-path a.pdf b.pdf` and
`--pdf-path docs/` fail, even though they are documented CLI forms; the same
array shape from an API caller is also rejected instead of typed.

**Fix:** either make the op accept a list and loop internally with the Python
report shape, or expand/validate in the CLI and call once per path; support
folder/glob expansion exactly as Python does. Keep single-path behavior and
error envelopes unchanged.

**Tests:** one PDF, two PDFs, one folder, one unmatched glob, a mixed
valid/missing list; assert per-file report/error shape and no partial import
on a preflight failure.

### BUG-088 — Job-lane reembed bypasses the write lock and diverges from Python’s job contract

**Severity:** P1, M — **Status: open**.

**Where:** `src/server/mod.rs:155-226` (`JobManager::submit`), `278-284`
(`view_with_elapsed`), `347-353` (`finish`), `478-520`
(`submit_job_op`/`submit_job_body`).

**Cause:**

- The reembed branch takes only `op_lock` and calls
  `router.reembed_with_control` directly; it never enters
  `Router::with_write_lock`, so a CLI process can write during the job.
- Submission checks `has_running("reembed")` but not
  `Router::_reembed_active`, so a sync reembed and a job can race.
- Any `OP_NAMES` op is accepted as a job; Python’s `JOB_OPS` is only
  `("reembed",)`.
- `view_with_elapsed` computes elapsed from `submitted` for queued jobs
  (Python returns `0.0` until `started`), and `finish` clears `phase`
  (Python retains the last progress phase).
- `/jobs` body submission does not strip `mode` from `kwargs` like Python.

**Impact:** concurrent writers can corrupt the reembed/pin state; a second
reembed can queue instead of getting `REEMBED_IN_PROGRESS`; clients observe
different job records; unsupported jobs run where Python returns a typed
`BAD_VALUE` with `job_ops`.

**Fix:** route both job and sync reembed through the same lock-aware op
wrapper, check both the registry and `_reembed_active` atomically enough for
the contract, restrict jobs to `JOB_OPS`, keep `phase` and elapsed semantics
matching Python, and filter `mode` from stored kwargs.

**Tests:** sync reembed blocks job submission with `REEMBED_IN_PROGRESS`;
cancel/queued/terminal elapsed values; phase after done/cancelled; non-reembed
job refusal; CLI writer during reembed sees `WRITE_LOCK_TIMEOUT`; compare
`check_jobs.py` records with Python.

### BUG-089 — MCP tool calls bypass the owner’s `op_lock`

**Severity:** P1, S — **Status: open**.

**Where:** `src/mcp/mod.rs:82-90` (`invoke`), `src/server/mod.rs:28-50`
(`AppState.op_lock`).

**Cause:** `okf serve /mcp` constructs `McpHttpService` with the `Arc<Router>`
as `OpClient`; `invoke` calls `client.call` directly inside
`spawn_blocking`. Python wraps the same router in `LockedRouter`, so MCP tools
and `/ops`/jobs share one lock. Rust’s HTTP ops and jobs take `op_lock` only.

**Impact:** MCP search/read/delete can run concurrently with a blocking HTTP
op or job. The shared Ladybug connection mutex prevents raw DB races, but file
operations, pin checks, reembed state, and multi-step writes are not protected
as one owner operation, so the “one serialized owner” contract is broken.

**Fix:** pass the owner lock (or a single `LockedOpClient` wrapper) into
`OkfMcp`/`McpHttpService` and acquire it around every `client.call`, matching
the HTTP path. Ensure cancel remains lock-free and the job lane does not
double-lock.

**Tests:** concurrent MCP + HTTP write/search with an injected delay; assert
serialization (no overlapping op marker); MCP delete/import does not interleave
with reembed; `/health` and cancel stay responsive; compare with Python’s
`LockedRouter` behavior.

---

## 9. Deferred / explicitly out of scope for this plan

- UDS client verification is closed by BUG-054; `/mcp` is closed by BUG-053.
- Local Linux/macOS hosts (CI now builds and smokes both platforms).
- Linux aarch64, macOS x86_64, Python 3.11.
- Full model-download CI; tests should use cached artifacts or mocks.
- A complete PyYAML 1.1 replacement beyond the documented frontmatter subset;
  the parser/dumper must at least not lose data and must cover the scenarios in
  section 7.

---

## 10. Suggested commit sequence

1. `fix: return typed errors instead of panicking on unknown codes`
2. `fix: make percent decoding byte-safe`
3. `fix: preserve concept stamps and make imports transactional`
4. `fix: skip file hashes for synthesized ingest documents`
5. `fix: adopt the stored embedding dimension on open`
6. `fix: sanitize resources and honor frontmatter identity`
7. `fix: delete orphan image assets and count images per concept`
8. `fix: implement file-backed delete/trash/restore`
9. `fix: enforce precision/space pins and report real lanes`
10. `feat: align ingest parameters and materialize/folder`
11. `feat: port thought/bibtex semantics and YAML support`
12. `feat: implement multi-root and prune/force delta behavior`
13. `feat: wire external lanes and remote images`
14. `feat: complete PDF/Office lifecycle and image formats`
15. `feat: complete the Python facade and error payloads`
16. `feat: add schema migrations, write lock, and read-only enforcement`
17. `test: add the regression/parity matrix`
18. `docs: update M3 status and release notes`
19. `fix: adopt schema settings and restore database creation`
20. `fix: wire TOML/env settings, validation and default bundle roots`
21. `fix: resolve named roots across import, writes and detach`
22. `fix: guard detached and file-free writes; make delete recoverable`
23. `fix: match traverse, search bounds, links and frontmatter validation`
24. `fix: harden asset resolution and image caption fallbacks`
25. `fix: complete CLI PDF, job-lock and MCP serialization parity`

Each commit should include its tests; do not batch P0 fixes with refactors.

---

## 11. Open follow-ups (status register)

This is the authoritative list of work that is still open after the
`a7da87a`–`391b46a` fix series and the `a62d1c9` post-cutover review. “Closed”
above means the code path is safe or the documented behavior is implemented;
it does **not** mean every release gate is green.

| Finding | State | Why it is open | Required close action |
| --- | --- | --- | --- |
| BUG-013 | Closed | External embed/rerank/vision HTTP clients are implemented behind `external`; `check_external_lanes.py` covers identity, auth, wire shapes, TEI fallback, GGUF counting, pins and typed outages. | Keep the mock differential and all-feature unit tests green. |
| BUG-028 | Closed | Compatibility is explicitly v10-only; older schemas are refused with `CONFIG_INVALID` and Python 0.15 remains the one-time migration path (`docs/rust-compat.md` §A). | Revisit only if a legacy DB must be opened directly by Rust. |
| BUG-030 | Closed | `extension_dir` and warm PDF idle reuse work; `wal_mode` is a documented compatibility no-op (`docs/rust-compat.md` §A). | Revisit only if upstream adds a journal knob. |
| BUG-040 | Closed | Real-model filters import benchmark and delta no-op smoke landed; baseline ~10.3 s / ~1.1 s on the CUDA dev box. | Re-run before release; update the recorded baseline if hardware changes. |
| BUG-044 | Closed | `run_matrix.py` fails on diffs and `check_coverage.py` enforces success+typed-error coverage with nine documented no-error waivers. Final run: 57/57 green, 30/30 success ops. | Keep new ops covered and re-run the matrix before release. |
| BUG-045 | Closed | Parameters, strict outcome, safe fix, facade kwargs, and `doctor-params` parity landed. | Keep covered by `doctor-params`; revisit only if further Python doctor semantics are ported. |
| BUG-046 | Closed | Portable info probes are emitted and compared; only host-specific rules are ignored by name. | Extend only if a portable probe is still missing. |
| BUG-047 | Closed | Unit tests copy the oracle fixture into a temp dir instead of sharing checkpoint state. | Keep the copy-fixture pattern for any new DB tests. |
| BUG-048 | Closed | `doctor --fix` takes the same write lock as other writes; idle lock sidecars are unlinked like Python. | Keep lock interop/read-only checks in the release gate. |
| BUG-049 | Closed | Typed-error envelopes for embed/traverse/import/sync match the oracle. | Keep `op-error-matrix` in the gate. |
| BUG-034 | Closed | PDF, DOCX, XLSX and a valid PPTX convert in both runners; `check_fixtures.py` rejects OOXML fixtures whose root relationship disagrees with their extension. The malformed `parity/corpus/invalid/broken.pptx` still verifies typed conversion failures. | Re-test PPTX if bobine/office_oxide is upgraded. |
| BUG-050 | Closed | Loaded-plugin conversion failures now carry `fields` (`path`, `converter`) and the converter remedy; missing-plugin failures keep the install hint. | Keep covered by `ingest-office-formats`. |
| BUG-051 | Closed | HTTP jobs submit/poll/list/cancel with terminal-state protection; `reembed_job` returns a job id. | Keep covered by `check_jobs.py`; revisit when jobs become durable across restarts. |
| BUG-052 | Closed | All 29 CLI verbs, global flags, op mapping, and typed error ops are wired; `check_cli.py` is green. | Keep new verbs covered; revisit if the oracle CLI gains commands. |
| BUG-053 | Closed | All 11 MCP tools are mounted at `okf serve /mcp` over Streamable HTTP; `check_mcp_http.py` is green. | Keep covered with stdio and `--connect` smokes. |
| BUG-054 | Closed | `--listen unix:///path` handles stale sockets; `check_uds.py` runs in CI on Linux and macOS (run `44861850111`) with the public `OKFClient`. | Keep both POSIX wheel jobs green. |
| BUG-055 | Closed | `okfgraph-pdf` builds as a PEP 517 wheel with the native plugin; clean-venv auto-discovery converts a real PDF. | Keep both wheels in the packaging gate. |
| BUG-056 | Closed | Wheel metadata pins 3.12–3.14, no bytecode in the wheel, public client/settings exports present. | Extend `check_packaging.py` with new metadata rules. |
| BUG-057 | Closed | `build_wheel.py` always passes auditwheel repair, clears the maturin scratch dir, and the Windows wheel contains OpenSSL DLLs. | Keep static gate green. |
| BUG-058 | Closed | Windows 3.12/3.13/3.14 clean-venv installs, facade, console scripts and PDF conversion are green. | Keep the matrix in CI for Linux/macOS. |
| BUG-059 | Closed | One `op_lock` serialises HTTP ops and jobs; four concurrent clients pass `check_client.py`. | Keep the concurrency assertion in the client gate. |
| BUG-060 | Closed | Job ids, statuses, record fields, mode validation, dedupe, 404s and cooperative reembed cancel match Python 0.15 (`check_jobs.py`). | Revisit when jobs become durable or the oracle changes its record shape. |
| BUG-061 | Closed | Rust CLI and MCP proxy borrow a live `okf serve` owner through `server::client`, with `OKFGRAPH_NO_SERVER=1` opt-out. | Keep CLI-to-owner and stale-discovery checks in `check_client.py`. |
| BUG-062 | Closed | Universal-newline parsing now normalizes lone CR; fuzz seed, unit regression and `frontmatter-cr-only` parity scenario are green. | Keep fuzz properties in CI and expand only for new parser surfaces. |
| BUG-063 | Closed | Shared Ladybug connection and cached pin checks bring Rust to 0.70x import, 0.68x embed and 1.06x search vs Python; parity gate enforced by `check_perf_bench.py`. | Keep the paired benchmark green before each release. |
| BUG-064 | Closed | Remote images resolve-and-screen addresses, pin one safe address, disable redirects and stream-cap at 20 MiB; `images-ssrf-private` is green. | Re-run if image fetching changes. |
| BUG-065 | Closed | `setup extensions` loads both files into a scratch DB before reporting `ok`; unit and CLI smoke green. | Re-run if the extension ABI changes. |
| BUG-066 | Closed | Rust CLI writes redacted crash reports with `fields.crash_report` and keeps 20 files; unit tests green. | Keep in the release gate if CLI error handling changes. |
| BUG-067 | Open | `Router` stores the caller’s `embedding_dim` instead of the width adopted by `Store::open`; 3/256/768-dim graphs fail with `DIM_MISMATCH` under default settings. | Build all `Router` constructors from `store.settings().clone()`; add mismatched-dim open/embed/reembed tests. |
| BUG-068 | Open | `OKFRouter`/`Router::open` require an existing DB; Python creates one on first use and supports `:memory:`. README/quickref examples fail fresh. | Implement open-or-init for the public surfaces; define or remove `:memory:`; test fresh, existing, old and future schemas. |
| BUG-069 | Open | `Settings::load` (TOML/env/legacy/checks) and `default_bundle_root` have no production call site. Env/TOML are ignored; default `okf import --all` lacks a root. | Route CLI/MCP/serve/facade through `Settings::load`; apply `default_bundle_root`; test precedence, legacy errors and invalid settings. |
| BUG-070 | Open | CLI `--root` inserts an array where `Settings.roots` expects a map; CSV flags insert a string where `Vec<String>` is expected. Both fail `CONFIG_INVALID`. | Parse roots and Csv values in `settings_from`/`from_map`; add malformed/duplicate/empty tests. |
| BUG-071 | Open | Unknown kwargs are silently dropped; `checks` never runs; `model_path`/`tokenizer_path` are accepted and ignored. | Reject unknown keys, run `checks`, and either implement explicit model files or return typed `CONFIG_INVALID`. |
| BUG-072 | Open | Named-root FileHash keys are resolved only against the primary root; `import_files` has no identity root; write/delete/trash/detach use the wrong tree and can stray/erase files. | Thread `settings.roots` through resolution/import/write/detach, add longest-prefix aliases and multi-root detach, test with colliding paths. |
| BUG-073 | Open | File-free `update`/`supersede` return `MISSING_PARAM`; Python edits the stored concept and re-embeds. | Port the graph-only `_write_and_import` branch; test title/tags/extra, links and supersede. |
| BUG-074 | Open | Detached graphs accept imports/ingest/writes; `force` is never read; `rebuild`/`batch_size` are accepted but unimplemented. | Add `require_attached(force)`/reattach rules, implement or reject index/batch controls, test every write path. |
| BUG-075 | Open | `delete(remove_file=True)` moves files and forgets hashes before `soft_delete`; a failed/read-only DB write leaves the concept live with files in trash and no tombstone. | Preflight writability, make trash/tombstone recoverable and idempotent, test rollback and retry. |
| BUG-076 | Open | `purge::recover` never checks `deleted_at`; expired tombstones still recover while `list_deleted` says non-recoverable. | Enforce the 24-hour window with the oracle clock rules; test now-1h/now-25h and malformed timestamps. |
| BUG-077 | Open | `find_path` misses PART_OF/INCLUDES_ASSET/directory hops, omits the start node, returns full nodes, caps at 6 (Python 10), and ignores path-mode argument rules. | Use the oracle shortest-path semantics/projection, allow 1–10, validate mode and test each edge type. |
| BUG-078 | Open | `since`/`until` compare raw strings, not datetimes; `recency_weight` silently defaults non-numbers; invalid `query_vector` is silently truncated. | Parse/validate time bounds and parameters; test date-only, offset, malformed and vector-width cases. |
| BUG-079 | Open | Null frontmatter `id:` stringifies to `"None"` and can become the concept id/uid. | Treat null as absent before scalar conversion; test null/empty/number/bool under both identities. |
| BUG-080 | Open | Numeric/bool core frontmatter is defaulted/dropped; invalid timestamps become null; malformed YAML/anchors/escapes are guessed instead of rejected. | Match Pydantic/PyYAML validation and fail closed with typed field errors; add the scenario matrix. |
| BUG-081 | Open | Third name collision resolves instead of staying ambiguous; wikilinks/inline code links are not stripped; link paths are not percent-decoded and `..` escape depth is lost. | Port Python ambiguity bookkeeping, `strip_code`, `unquote` and normpath/escape rules; compare edges/BrokenLinks. |
| BUG-082 | Open | Bulk discovery is case-sensitive (`.MD`/`.Markdown`/`.TXT` skipped), unlike Python’s `suffix.lower()`. | Lowercase the extension and share one predicate across import/detach/lint/diff; test mixed-case trees. |
| BUG-083 | Open | Vision-route errors abort the concept; Python falls back to captions except lane outages. Image refs ignore optional titles. | Catch per-image non-outage failures, preserve outage behavior, parse titles/angle sources; compare rows. |
| BUG-084 | Open (security) | `okf-asset://` ids are joined without rejecting `..`/absolute/drive paths, allowing arbitrary local file reads into `ImageAsset`. | Validate to a safe asset id, canonicalize/within-root check every branch, test Windows/POSIX traversal. |
| BUG-085 | Open | BibTeX citations lose the `entry.key` fallback and keep untrimmed/braced author/venue/abstract fields. | Reproduce Python citation fallback, trimming and `unwrap_braces`; test sparse/padded entries. |
| BUG-086 | Open | Export See Also only detects `](target` at the start of a URL; links like `dir/target.md` get duplicated. | Compare resolved link destinations, not a prefix substring; test relative/up-level/alias links. |
| BUG-087 | Open | CLI sends `pdf_path` as an array but `ingest_pdf` expects one string; multi-path/folder/glob PDF ingest fails. | Accept a list or loop in the CLI, add folder/glob expansion, preserve per-file reports. |
| BUG-088 | Open | Job reembed bypasses `with_write_lock`; `_reembed_active` is not checked at submission; any op is accepted; queued elapsed/phase differ. | Route jobs through the same lock/op wrapper, limit `JOB_OPS`, match phase/elapsed/kwargs, test races. |
| BUG-089 | Open | MCP `invoke` calls the router without `op_lock`; Python serializes MCP and HTTP through `LockedRouter`. | Pass the owner lock into `OkfMcp`/`McpHttpService`; test MCP+HTTP write/search serialization. |
| PDF missing-plugin | Intentional divergence | Python 0.15 raises raw `INTERNAL`; Rust returns `CONVERSION_FAILED` with the install hint. | Keep documented; do not “fix” Rust to match the raw Python failure. |

### Immediate order

Completed in the earlier fix series:

1. ~~BUG-028 migrations or an explicit v10-only compatibility statement.~~ Done.
2. ~~Broader Office fixtures (XLSX/PPTX).~~ Done: valid PPTX converts in both runners; an intentionally malformed corpus fixture covers typed failures.
3. ~~BUG-013 external clients.~~ Done: the documented stable surface keeps
   the lanes and the clients are covered by mock-host parity.

Next, post-cutover review:

1. BUG-067, BUG-068, BUG-069, BUG-070, BUG-071 — restore the settings/open
   contract first; these block most other scenarios.
2. BUG-072, BUG-073, BUG-074, BUG-075, BUG-076 — named-root and write
   integrity, detach enforcement, delete atomicity and recovery lifetime.
3. BUG-077, BUG-078, BUG-079, BUG-080, BUG-081, BUG-082 — semantic parity
   for traversal, search bounds, frontmatter, links and file discovery.
4. BUG-083, BUG-084, BUG-085, BUG-086, BUG-087 — images, BibTeX, export and
   CLI PDF surfaces.
5. BUG-088, BUG-089 — job and MCP serialization.

Each close needs code + regression test + parity/scenario (or an explicit
documented exclusion) + updated status in this file.
