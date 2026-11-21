# OKFgraph Rust Core — Bug Findings and Fix Plan

**Status:** review plan from `6a71fec`, updated after the fix series through
the M4/M5 slices and the BUG-013 external-lane implementation. The remaining
open work is tracked in section 11; findings marked “partial” are not closed.

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
full 29-command CLI, and typed conversion-error envelopes. The broad differential matrix is green for **56 scenarios** after
normalizing wall-clock graph times/internal IDs and the documented intentional
differences; the 57th, `ingest-pdf-no-plugin`, records the intentional
missing-plugin error divergence instead of a green comparison. Remaining items are listed under
“Verification status” below and the authoritative register in section 11.

**Important:** the original review below describes the pre-fix state. It is
kept as the evidence trail; the implementation status section records what is
now closed and what remains. The existing parity matrix was green before the
fixes but only covered a narrow fixture set. “M3 complete” should not be
claimed until the remaining gate items close.

### Verification status

Final verification on `rust-core` at `66df990`:

- `parity/tools/run_matrix.py --pdf-plugin target/debug/okfgraph_pdf.dll`:
  `TOTAL:56 FAILS:0` in **14m37s** on the CUDA dev box. The 57th scenario,
  `ingest-pdf-no-plugin`, remains the documented intentional divergence.
- `parity/tools/check_coverage.py`: 30/30 success ops, 21 typed-error ops,
  nine documented waivers, 61 output files scanned.
- `parity/tools/check_external_lanes.py`: green against four local mock hosts,
  including TEI rerank fallback and a generated valid GGUF counter fixture.
- Rust suites: 75 all-feature tests and 54 no-default tests pass; Clippy is
  clean with all features and with no default features.
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
- BUG-034 is covered end-to-end for PDF, DOCX and XLSX through the same bobine
  converter path. A minimal PPTX fixture is also covered: pinned bobine 0.9.1
  refuses it in both runners with a typed `CONVERSION_FAILED`, verified by
  `ingest-office-formats`.
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
  `ingest-office-formats` covers the XLSX success and PPTX refusal paths.
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
install hint for bare ABI/setup errors. `ingest-office-formats` verifies the
XLSX success and the PPTX refusal with zero differences.

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
| `ingest-office-formats` | BUG-034, BUG-050 (XLSX success, PPTX typed failure) |
| `external-lanes-mock` | BUG-013 |
| `pins-mismatch` | BUG-014, BUG-015 |
| `facade-api` | BUG-026, BUG-027 |
| `lock-and-read-only` | BUG-029 |
| `doctor-params` | BUG-045, BUG-046, BUG-048 |
| `doctor-empty` | BUG-046 |
| `op-names` | BUG-038 |
| `op-error-matrix` | BUG-044 |

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

Each commit should include its tests; do not batch P0 fixes with refactors.

---

## 11. Open follow-ups (status register)

This is the authoritative list of work that is still open after the
`a7da87a`–`391b46a` fix series. “Closed” above means the code path is safe or
the documented behavior is implemented; it does **not** mean every release
gate is green.

| Finding | State | Why it is open | Required close action |
| --- | --- | --- | --- |
| BUG-013 | Closed | External embed/rerank/vision HTTP clients are implemented behind `external`; `check_external_lanes.py` covers identity, auth, wire shapes, TEI fallback, GGUF counting, pins and typed outages. | Keep the mock differential and all-feature unit tests green. |
| BUG-028 | Closed | Compatibility is explicitly v10-only; older schemas are refused with `CONFIG_INVALID` and Python 0.15 remains the one-time migration path (`docs/rust-compat.md` §A). | Revisit only if a legacy DB must be opened directly by Rust. |
| BUG-030 | Closed | `extension_dir` and warm PDF idle reuse work; `wal_mode` is a documented compatibility no-op (`docs/rust-compat.md` §A). | Revisit only if upstream adds a journal knob. |
| BUG-040 | Closed | Real-model filters import benchmark and delta no-op smoke landed; baseline ~10.3 s / ~1.1 s on the CUDA dev box. | Re-run before release; update the recorded baseline if hardware changes. |
| BUG-044 | Closed | `run_matrix.py` fails on diffs and `check_coverage.py` enforces success+typed-error coverage with nine documented no-error waivers. Final run: 56/56 green, 30/30 success ops. | Keep new ops covered and re-run the matrix before release. |
| BUG-045 | Closed | Parameters, strict outcome, safe fix, facade kwargs, and `doctor-params` parity landed. | Keep covered by `doctor-params`; revisit only if further Python doctor semantics are ported. |
| BUG-046 | Closed | Portable info probes are emitted and compared; only host-specific rules are ignored by name. | Extend only if a portable probe is still missing. |
| BUG-047 | Closed | Unit tests copy the oracle fixture into a temp dir instead of sharing checkpoint state. | Keep the copy-fixture pattern for any new DB tests. |
| BUG-048 | Closed | `doctor --fix` takes the same write lock as other writes; idle lock sidecars are unlinked like Python. | Keep lock interop/read-only checks in the release gate. |
| BUG-049 | Closed | Typed-error envelopes for embed/traverse/import/sync match the oracle. | Keep `op-error-matrix` in the gate. |
| BUG-034 | Covered for PDF/DOCX/XLSX/PPTX | XLSX converts in both runners; pinned bobine refuses PPTX in both with the same typed envelope (`ingest-office-formats`). | Re-test PPTX if bobine/office_oxide is upgraded. |
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
| PDF missing-plugin | Intentional divergence | Python 0.15 raises raw `INTERNAL`; Rust returns `CONVERSION_FAILED` with the install hint. | Keep documented; do not “fix” Rust to match the raw Python failure. |

### Immediate order

1. ~~BUG-028 migrations or an explicit v10-only compatibility statement.~~ Done.
2. ~~Broader Office fixtures (XLSX/PPTX).~~ Done for the pinned converter.
3. ~~BUG-013 external clients.~~ Done: the documented stable surface keeps
   the lanes and the clients are covered by mock-host parity.

Each close needs code + regression test + parity/scenario (or an explicit
documented exclusion) + updated status in this file.
