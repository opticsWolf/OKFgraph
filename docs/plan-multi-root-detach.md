# Plan: multi-root bundles + detach (rev 3 — post-review, 2nd pass)

**Status:** draft rev 3 — code-reviewed twice. First pass fixed the design
(colon→`@`, Meta/v7, Phase 0 scope); second pass verified every claim and line
reference and found four more code facts (14–17), folded in below.
**Scope:** **(B) multi-root bundles** (N live source trees → one graph) and **`detach`**
(explicitly end the mirror relationship: keep the graph, drop the filebase).
**Prerequisite:** Phase 0 — expanded by review (import crash-consistency **and**
deletion detection), see §0.

Baseline: schema v6, suite 522/10/18, dim default 512 (Matryoshka ladder
32/64/128/256/512/768/1024 per the official Jina model card), Rust-only fail-fast.

## Verified constraints (code facts this plan must respect)

| # | Fact | Code |
|---|------|------|
| 1 | Concept IDs = root-relative path, suffix stripped, `\`→`/`; frontmatter `id:` is preserved as `uid` and never overrides the derived ID; bare-stem fallback for files outside the root | `import_.py:94-119` |
| 2 | `FileHash(path, hash, concept_id)` — `concept_id` re-derived from path; `DirHash(path, hash, files=JSON)` — `files` are dir-relative names | `delta.py:175-186`, `:48-64`, `:86-102` |
| 3 | Delta detection has **side effects**: DirHash persisted inside `_changed_directories`, FileHash persisted by `import_bundle_inner` **before** parse/encode/upsert | `delta.py:152`, `import_.py:458-470` |
| 4 | Deleted-file detection is **directory-granular**: only dirs present in DirHash and now missing yield deleted paths; a file deleted inside a surviving dir is never reported deleted | `delta.py:137-151` |
| 5 | Parse failures are per-file `logger.warning` and do not block the batch; but their hash is already stored → they never retry | `import_.py:477-486` |
| 6 | `purge` is not a command; purge runs inside import (`--purge-deleted`, uses old FileHash `concept_id` map) and via `deleted-purge`/`deleted-recover`/`deleted-list` (DB-local, no baseline needed); the purge log reports `len(deleted)`, not the actual purged count | `import_.py:447-455`, `cli.py:1284-1296` |
| 7 | `Meta(key STRING PK, value INT64)` — integer values only; already used for schema version, write_epoch, index dirty flags | `schema.py:266-267`, `:459-472` |
| 8 | Wiki targets starting with `scheme:` are **silently skipped** (`SCHEME_RE = ^[a-zA-Z][a-zA-Z0-9+.-]*:`, guards `okf-asset:` etc.) — never resolved, not even a BrokenLink; `resolve_wiki` order = exact id → id-minus-`.md` → uid → alias → title → stem, with the **exact-id probe case-sensitive** and all name maps lowercased | `links.py:16,32-34,107-131`, `import_.py:282-291` |
| 9 | Export reconstructs file paths **directly from concept IDs** (`cid.replace("/", os.sep) + ".md"`) — IDs must therefore be valid path segments | `export.py:327-334` |
| 10 | `diff` has two modes: snapshot (dir vs dir, no DB) and drift (graph vs **one** dir, via `router.diff_db_dir`) | `cli.py:619,1232`, `router.py:448` |
| 11 | `--db/--bundle/--dim/...` are added to every subcommand by one helper (`_add_global`); resolved+merged with TOML in `_router` | `cli.py:119`, `:164-182` |
| 12 | MCP `ingest` accepts arbitrary paths; `kind='pdf'` runs with **`auto_import=True` by default** → the temp-dir path (fact 14); construction takes a single `bundle_root` | `mcp_server.py:298-355`, `:44-63` |
| 13 | `doctor` never scans the bundle — all checks (broken_link, duplicate_title, orphan, missing_description, stale) are graph-internal | `doctor.py:110-149` |
| 14 | Work-dir imports **override `bundle_root` in place** (router + delta/import managers) to a `TemporaryDirectory`, then restore — delta rows are written into the SAME tables keyed relative to the temp root, including the shared top-level key `"."` | `ingest.py:437-455` |
| 15 | Ingest mints **bare-stem IDs**: PDF auto-import parses with `root=work_dir` → id = filename stem; `ingest_md` → `cid = concept_id or md_path.stem.replace(" ", "_").lower()` — location never enters the ID, so same-stem sources silently overwrite each other | `ingest.py:64`, `import_.py:100-104` |
| 16 | DirHash/FileHash rows are **never deleted** (MERGE-only): vanished dirs' rows linger (dir-deletion info is durable by accident, and re-reported every run), but a surviving dir's MERGE rewrites `files` — per-file deletion info would be one-shot | `delta.py:86-102,152` |
| 17 | Concept inserts run as **per-concept transactions** (BEGIN/COMMIT in `_insert_concept`, embedding in the same transaction) — post-commit hash writes are natural; migrations auto-run on open (`_get_meta("schema_version")` → ordered `_MIGRATIONS`) | `import_.py:855-884`, `schema.py:342-347` |

## Invariants

- 5 MCP tools unchanged (`search`, `read`, `traverse`, `ingest`, `export`); `detach`
  is CLI-only (like `deleted-*`); multi-root changes construction flags/params only.
- CLI remains a strict superset of MCP. Skills stay 3 files; bodies updated.
- Single-root usage stays behaviorally identical: existing DBs open, existing IDs
  stable, no forced migration (empty-alias rule, §2.1).
- Frozen contracts untouched (Jina v5 space/1024-native/last-token pooling,
  deterministic numerics, Rust-only fail-fast, dim default 512).
- No new required dependencies. dev → CI → ff main; version per change.

## Phase 0 — import crash-consistency + true deletion detection (0.2.15)

Observed live 2026-09-13: an interrupted `import --all` left 44 FileHash + 44 DirHash
rows and **0 concepts**. Every later run trusted the hashes ("0 changed") while the
graph stayed empty — escape required deleting the DB. Facts 3-5 make this a family
of bugs, not one — and the second review pass added fact 14 (the baseline is also
*polluted* by work-dir imports) and fact 16 (deletion info is durable by accident
for dirs, one-shot for files):

**0.1 Post-commit hash writes.** Detection becomes side-effect-free: both hash
writers (fact 3) are replaced by returning pending DirHash/FileHash updates to the
caller, which persists them **only after** the concept transactions commit. Concept
inserts already run as per-concept transactions with the embedding inside
(fact 17), so the natural write point is per file, immediately after its own
commit (tightest window; batch-after-loop is acceptable). Crash window shrinks to
"concepts committed, hashes not" → next run redos those files (idempotent upserts,
wasted work only). Never the reverse. Applies to both `_changed_directories` and
`import_bundle_inner`.

**0.2 Failed parses must retry.** Files whose parse raised (fact 5) are excluded
from the committed hash set → they retry next run, visibly (warning count in the
summary). Today a single parse failure is silently permanent.

**0.3 Per-file deletion detection (fact 4) with durable tombstones (fact 16).**
When a dir's hash changed but the dir still exists, diff stored `DirHash.files` vs
current files → append missing ones to `deleted_paths`. Without this,
`import --purge-deleted` only tombstones whole directories, and §2.3's "purge =
mirror truth" invariant is already broken single-root. Persistence matters: today
dir-deletion info is durable only by accident (DirHash rows are never deleted,
fact 16), while a surviving dir's MERGE rewrites `files` — so per-file deletions
must be persisted explicitly as `DeletedPath(path, detected_at)` rows at detection
time, consumed (and deleted) by purge; otherwise the deletion is visible only in
the one run where the dir changed. Also fix the purge log, which reports
`len(deleted)` even when cid lookups matched nothing (`import_.py:453-455`) —
log the actual count. (The correct comparison logic exists in dead code
`_changed_files` — tests only, `tests/test_delta.py:101`. Wire it in or delete it
in 0.4.)

**0.4** Remove or align `_changed_files` so there is exactly one delta
implementation (fact 3 note: it writes hashes as a side effect, same disease).

**0.5 Work-dir delta isolation (fact 14).** `ingest_pdf(auto_import=True)` — the
MCP default — imports from a `TemporaryDirectory` with `bundle_root` (and the
delta/import managers' copies) pointed at it (`ingest.py:437-455`). Its delta rows
land in the same tables keyed relative to the temp root — including the shared
top-level key `"."`. Consequences today: every PDF auto-import clobbers the real
bundle's `.` DirHash row, so the next regular import sees the whole top level as
changed and silently re-encodes it; deeper temp rows linger forever, and any temp
md in a subdirectory would surface as a "deleted directory" (mass-tombstone hazard
on `--purge-deleted`). Fix: work-dir imports bypass the delta entirely (they are
always full imports) or snapshot/restore the delta state around them; no temp
paths may survive in DirHash/FileHash. Independent of Phase 2 — single-root 0.2.x
users hit this today.

**Tests:** injected-failure resume (upsert raises after K files → re-run completes,
concept count == file count, vectors queryable); parse-failure retry; single-file
delete → `--purge-deleted` tombstones it (and tombstone survives a no-purge run in
between); purge log reports the true count; work-dir isolation (PDF auto-import,
then regular import re-encodes only actually-changed files; no temp keys remain);
`doctor` consistency check for hash-rows-without-concepts (report + repair hint).
Model-free (stub encoder), fast.

**Size:** small-medium (detection refactor, not a two-line move). No schema change.

## Phase 1 — `detach` (0.2.16)

Mirror state → detached state: DB is the artifact, sources may be deleted; reads,
searches, traverses, exports, recover keep working. The mirror relationship ends by
declaration instead of silently rotting.

### 1.1 State (schema v7 — review correction)

Detach provenance needs **strings**; `Meta.value` is INT64 (fact 7). So:

- `Meta` keys: `detached` = 1, `detached_at` = epoch seconds (both fit INT64).
- New node table `SourceRoot(alias STRING PK, path STRING, file_count INT64,
  last_seen INT64)` written at detach (provenance + re-attach safety, §1.3).
- `SCHEMA_VERSION` 6 → **7**, migration `_migrate_v6_to_v7` (CREATE TABLE
  IF NOT EXISTS pattern, like v6's ALTER/CREATE), migration test v6→v7.

### 1.2 `okf detach [--verify|--no-verify] [--db ...]` (bundle not required)

1. **Verify (default on):** compare **sources vs graph**, not export vs DB (the
   latter is tautological — export reads the DB). For each bundle concept file:
   parse via the module-level `_parse_source_file` (no DB) and compare normalized
   frontmatter + body against the stored concept; mismatches are listed. Files with
   no stored counterpart (PDFs, images, other formats) → `WILL-NOT-SURVIVE` list
   (originals are not in the DB; only bobine conversions and referenced assets are).
   Graph-content mismatch aborts; source-only artifacts require `--force`.
   `--no-verify` for huge graphs. The source list is the bundle walk
   (authoritative) — `FileHash` may contain ephemeral temp-dir rows (fact 14);
   nonexistent paths are ignored, and concepts whose source md lives only in the
   DB (PDF conversions, fact 15) are simply expected to survive detach.
2. Clear all `FileHash`/`DirHash` rows (no baseline → nothing is "deleted").
3. Write state rows (§1.1). Print what still works / what changed.

### 1.3 Post-detach semantics (review-corrected)

| Command | Behavior |
|---|---|
| `search` / `read` / `traverse` / `export` / `lint` | unchanged |
| `import` (all modes, incl. `--purge-deleted`) | **refuse** unless `--force`. With `--force`: re-attach allowed only if the root set matches `SourceRoot` provenance; otherwise refuse with "different source tree — create a new DB" (path-derived IDs would silently overwrite archived concepts) |
| `deleted-list` / `deleted-recover` / `deleted-purge` | **allowed** (DB-local, no baseline needed — fact 6; review removed the blanket `deleted-*` refusal) |
| `doctor` | gains an informational detached-state line (fact 13: nothing to "fix", it never scanned the bundle) |

CLI-only; one line in the MCP and CLI skills ("detached graphs serve normally;
lifecycle ops are CLI-only"). Whole-graph only; per-root detach parked (§2.5).

**Tests:** fidelity report lists exactly source-only artifacts; delete sources →
search/read/export/traverse identical before/after; import refuses, mismatched-root
`--force` refuses, matching-root `--force` re-baselines; deleted-* still work;
v6→v7 migration test.

## Phase 2 — multi-root bundles (0.3.0)

One graph, N live roots, no copies, no ID collisions.

### 2.1 Identity — `@alias/rel` (review correction: colon is not usable)

Original draft proposed IDs `alias:rel`. Two code facts kill it:
- `:` is **illegal in Windows filenames**; export writes `cid + ".md"` as a path
  (fact 9) → every export of a multi-root graph would fail or create an NTFS ADS.
- On the link side, any `alias:Name` wikilink matches `SCHEME_RE` and is **silently
  dropped** before resolution (fact 8).

Adopted scheme: **`@alias/rel`** (concept IDs), e.g. `@okfgraph/README`.

- `@` is filename-safe, never a URI scheme start, and cannot appear naturally at
  the start of a relative repo path (reserved-prefix rule: warn/refuse on
  legacy bundles containing a top-level `@*` entry).
- Derivation lives in ONE function: `_parse_source_file` gains the alias
  (roots config); bare IDs remain for `alias=""` (legacy single root) — **no
  data migration**; fact 1 confirms frontmatter `id:` cannot collide (it stays
  `uid`).
- Export/import round-trip works **without any config**: `@okfgraph/README` →
  file `@okfgraph/README.md` → single-root re-import reproduces the exact ID.
- Per-alias root `Directory` nodes come for free: the hierarchy builder splits IDs
  on `/` (`import_.py:886`).
- Validation: aliases non-empty, unique, not starting with `@`, and matching a
  filename-safe charset — suggested `^[A-Za-z0-9][A-Za-z0-9_-]*$` (excludes `:`, so
  an alias can never resemble a URI scheme inside a link target); roots
  non-overlapping (a file in two roots = two identities = reject, fail-fast).
- Ingest namespace (fact 15): PDF auto-import and single-md ingest mint **bare-stem
  IDs** (`report`, `readme`) regardless of location — two same-stem sources
  silently overwrite each other's concepts via upsert. This is an existing hazard,
  not a compatibility guarantee to preserve: in multi-root, PDF work-dir imports
  get a stable ingest namespace (e.g. alias `pdf` + `<pdf-content-hash[:12]>`), and
  path-based md ingest resolves to a root via §2.6 (only truly outside-everything
  files keep the bare-stem fallback).

### 2.2 Delta per root

- FileHash path and DirHash path get the `@alias/` prefix; FileHash `concept_id`
  derivation (`delta.py:180`) becomes alias-aware — otherwise purge's path→id map
  misses every multi-root file.
- Hash writes only per Phase 0 discipline (post-commit); detection side-effect-free.
- Per-root walks merged into one changed/deleted set; deletion detection per
  Phase 0.3 now works cross-root.

### 2.3 Liveness — unmounted ≠ deleted (unchanged core)

Per-root present/absent state resolved at command start (root path exists/readable).

- **Import:** absent roots SKIPPED with loud warning + per-root summary line;
  never treated as deletions.
- **Purge** (`--purge-deleted` inside import, and any future standalone purge):
  fail-closed unless **all** roots present. "Unknown" must never read as "deleted".
  This is THE invariant of the phase; tested hardest.
- **Reads/exports:** unaffected.
- **Doctor:** per-root status section (present/absent, file count, last seen).

### 2.4 Links across roots

- Qualified targets are simply **the namespaced ID**: `[[@okfgraph/README]]` works
  via the existing exact-id-first rule in `resolve_wiki` (`links.py:107-131`).
  Case note: the exact-id probe is case-sensitive while uid/alias/title/stem maps
  are lowercased — decide whether `[[@alias/...]]` gets a lowercased fallback probe
  and test whichever is chosen.
- Human-friendly `[[okfgraph/README]]` (no `@`) resolves via an alias-aware
  pre-step in `_resolve_body_links`: first segment matches a root alias → rewrite to
  `@alias/rest` and resolve. **No tokenizer change, no scheme conflict** because the
  check happens before `is_external` only for known aliases (fact 8).
- Unqualified `[[Name]]` unchanged (uid → alias → title → stem, ambiguity never
  resolves); cross-root collisions become BrokenLinks, repairable.

### 2.5 Directory model

One root `Directory` node per alias (free, §2.1). Whole-graph detach works on
multi-root graphs unchanged (clears all roots' hashes; provenance lists all roots).
Per-root detach/freeze remains parked.

### 2.6 Surfaces

- CLI: `--bundle` stays; new repeatable `--bundle-root alias=path` added once in
  `_add_global` (fact 11) so every command gets it; mutual exclusion with `--bundle`
  validated in `_router`. TOML gains `[[roots]]` alongside `bundle`.
- MCP tools unchanged, **but** path-based `ingest` needs root resolution: longest
  prefix match of the given path against the roots; outside all roots → existing
  bare-stem fallback (fact 15 — tolerated only as the documented outside-everything
  case); PDF work-dir imports get the §2.1 ingest namespace plus 0.5 isolation.
  Server construction gains the roots parameter.
- Skills: CLI (flags, liveness semantics, detach), ingest (namespaced links),
  MCP (construction-time roots; tools unchanged).

### 2.7 Export layout (former open question — resolved by design)

IDs are paths (§2.1), so `export --all` on a multi-root graph writes
`<out>/@alias/rel.md`; a single-root re-import of that tree reproduces IDs exactly.
No per-alias export flag. Round-trip test required.

### 2.8 Diff (review gap — added)

Snapshot mode (dir vs dir) accepts per-alias pairing. Drift mode (graph vs dir,
fact 10) — currently one bundle — must become per-root (or explicitly single-root
only), because a multi-root graph cannot drift-check against one tree.

### 2.9 Tests

Collision (three `README.md` → three namespaced concepts, both searchable);
liveness (root absent → import warns + no tombstones, purge refuses, remount
resumes); validation (duplicate/missing/unsafe aliases, nested roots); qualified
links both forms; export→re-import ID-trip; drift per-root; full suite green
unmodified single-root (backward-compat proof).

### 2.10 Rollout

Order: 2.1 → 2.2 → 2.3 → 2.4 → 2.6/2.7/2.8 → 2.9. Docs: `architecture.md` bundle
model rewrite, skills, CHANGELOG. Release **0.3.0**. Acceptance: OKFgraph + bobine +
embroider live docs in one graph, copies retired; kill one root → serve + refuse
purge + resume on remount.

## Non-goals (parked)

Batch D (ladybug upstream, await macrame 0.18) · Phase 5 bobine text embedding ·
per-root detach · network/remote roots · concurrent multi-writer.

## Open questions (post-review)

1. Exact `--bundle-root` flag spelling and `[[roots]]` TOML shape.
2. Constructor naming: `bundle_roots=[(alias, path)]` vs `roots={alias: path}`.
3. Re-attach `--force` UX: root-set match requirement (§1.3) vs always-new-DB —
   confirm no "resume mirroring after a trip" workflow needs loosening it.

## Review log (rev 3)

- **Temp-dir delta pollution found** (fact 14): `_import_work_dir` overrides
  `bundle_root` in place, so PDF auto-import (the MCP default path) writes delta
  rows keyed relative to a deleted TemporaryDirectory — clobbering the real
  bundle's `"."` DirHash row (mass silent re-encode on next import) and leaving
  lingering temp rows. → Phase 0.5.
- **Deletion-persistence asymmetry found** (fact 16): DirHash rows are never
  deleted, so dir-deletion info is durable by accident; a surviving dir's MERGE
  rewrites `files`, which would make per-file deletions one-shot. → 0.3 now writes
  durable `DeletedPath` tombstones; purge log honesty (reports `len(deleted)`
  blindly, `import_.py:453-455`) folded into 0.3.
- **Ingest IDs are bare stems** (fact 15): PDF auto-import (`root=work_dir`) and
  `ingest_md` (`cid = md_path.stem…`) mint location-independent IDs — same-stem
  collisions overwrite silently. → §2.1 ingest-namespace bullet; §2.6 wording.
- **Line references corrected** from the first-pass citations: DirHash store is
  `delta.py:86-102` (not lumped with FileHash), persist call at `:152`, deletion
  block `:137-151`, purge block `import_.py:447-455`, export path write
  `export.py:327-334`, `resolve_wiki` at `links.py:107` (with the case-sensitivity
  nuance), `_add_global` at `cli.py:119`, `diff_db_dir` at `router.py:448`.
- **Verified this pass:** per-concept BEGIN/COMMIT with embedding in-transaction
  (fact 17 — post-commit hash writes are a natural fit); migrations auto-run on
  open via `_get_meta("schema_version")` (v7 feasible, v5→v6's
  probe-never-assume discipline is the pattern); MCP pdf default
  `auto_import=True`; no `@` handling anywhere (prefix safe); `_changed_files`
  dead in production (tests only); doctor fully graph-internal.

## Review log (rev 2)

- **Colon IDs killed** (fact 8): `alias:` is Windows-illegal on export (fact 9)
  and is swallowed by `SCHEME_RE` in wikilinks. → `@alias/rel`; qualified links
  resolved via alias pre-step; exact-ID form works for free.
- **Meta cannot hold provenance** (fact 7: INT64) → schema v7 + `SourceRoot`,
  migration + test; plan previously claimed "no schema bump".
- **Wedge is a family, not one bug** (facts 3-5): two side-effecting writers before
  commit, parse failures hashed permanently, dir-granular deletion only. Phase 0
  expanded with 0.1-0.4 and re-sized.
- **`deleted-*` refusal removed** (fact 6): those ops need no baseline; only
  `import` refuses post-detach. Also: there is no standalone `purge` command.
- **Detach verify semantics fixed**: compare sources vs graph (export-vs-DB was
  tautological); `WILL-NOT-SURVIVE` list is the user-facing product.
- **`doctor` wording fixed** (fact 13): no bundle scanning exists; detached report
  is new information, not a fix.
- **Gaps added**: root resolution for path-based MCP ingest / PDF auto-import
  (fact 12); drift diff per root (fact 10); export layout resolved by §2.1 rather
  than left open.
- Verified-correct claims kept: empty-alias no-migration rule (fact 1),
  per-root Directory nodes free (import hierarchy split), `_add_global` plumbing
  (fact 11), liveness/purge fail-closed core, frontmatter-uid non-collision.
