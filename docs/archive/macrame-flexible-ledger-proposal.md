# Macrame flexible-ledger proposal (target: 0.18)

**Status:** draft · **Date:** 2026-09-11 · **Branch:** `dev_rework` (OKFgraph side)
**Goal:** per-usecase extensibility **without** user DDL — open the attribute
layer and add guarded sidecars, while the ledger core (time, assertions,
materialization, archive, branching) stays fixed and guarantee-bearing.

**Non-goals:** arbitrary `CREATE TABLE` / user-defined rel types as schema
(that is Ladybug's job; bending the ledger there unmakes Macrame). No changes
to Doctrines II–VII, no primary-key diffs beyond what D-036 already permits
pre-1.0 (additive columns and new indexes only).

**Consumers:** OKFgraph rework is the driving use case (frontmatter envelope,
`FileHash`/`Meta`, image bytes), but every primitive below is app-agnostic.

---

## P1. `concepts.extra` JSON column (highest value, smallest blast radius)

**DDL (new rung, both `concepts` and `cold.concepts` — archive must carry it):**
```sql
ALTER TABLE concepts ADD COLUMN extra TEXT NOT NULL DEFAULT '{}';
-- CHECK (json_valid(extra))              -- loud failure on corrupt writes
-- CHECK (typeof(extra) = 'text')         -- belt: GLOB guard not needed for JSON
```

**Semantics:**
- App-defined attributes (OKFgraph: `type`, `tags[]`, `layer`, `description`,
  `resource`, frontmatter leftovers). Namespaced by convention: `okf:type`,
  `myapp:status`. Flat keys allowed; nesting allowed but indexed paths (below)
  must be stable.
- Versions with the concept row: title/content corrections already mint log
  entries via existing triggers; `extra` rides the same row, so history,
  `as_of_*` attribute modes, snapshots and rehydrate cover it with **zero
  trigger-logic changes** — audit every trigger that lists columns explicitly
  (`SELECT *` would silently include it; explicit lists must add it).
- Size cap: e.g. 64 KiB, enforced at the API boundary (`ValidationError`).
  Bulk bytes are blobs (P4), not attributes. Cap value is a named constant
  crossing to Python (cf. `CHUNK_ROWS_*` precedent).

**Indexing (this is what makes V6.0 P3 layer-filtering fast):**
```sql
CREATE INDEX idx_concepts_extra_layer
    ON concepts (json_extract(extra, '$.layer'));
```
Expression indexes are opt-in per deployment: core ships the `layer`
convention index (NULL-tolerant: rows without the key simply don't match),
apps can register their own via a guarded `register_extra_index(path)`
that validates the JSON path and refuses duplicates. SQLite computes these
from the stored text — no backfill needed beyond index build.

**API:**
- Rust: `ConceptUpsert::extra(impl Into<String>)` builder (validates
  JSON + cap, returns typed error); getter on reads/`NodeAttributes`.
- Python: `ConceptUpsert(..., extra={...})` accepting `dict`/`str`;
  returned as `dict`. `register_extra_index(path)` classmethod-ish on `Database`.
- Errors: malformed JSON / over-cap → `ValidationError` (existing group, no
  new taxonomy). Round-trip property tests (Rust + Python parity).

**Effort:** ~1 day (rung + trigger audit + bindings + tests + `public-api.txt` bless).

---

## P2. Edge-kind charset relaxation

**Change:** `edge_type` validation `[A-Z0-9]+` → `[A-Za-z0-9_:.\-]+`
(max length unchanged). No DDL, no migration — purely the API-boundary check
(Rust + Python same regex, one shared test vector).

**Convention (docs, not code):** `prefix:Kind` namespacing —
`okf:links-to`, `myapp:cites` — so independent apps sharing a file (or a
reviewer reading one) don't collide. Core kinds stay bare (`LINKSTO` reads
better than `core:linksto` and existing DBs keep working).

**Effort:** ~2 hours + release note.

---

## P3. KV sidecar (replaces external `meta.json` / `FileHash` tables)

**Rationale:** small operational state (hashes, epochs, counters, cursors) is
neither belief (ledger) nor content (concepts) — it needs durability +
snapshot story without history. Doctrine VII precedent (embeddings excluded
from the ledger) applies verbatim.

```sql
CREATE TABLE kv_store (
    key        TEXT PRIMARY KEY,   -- 'okf:filehash:<path>' | 'myapp:epoch' | ...
    value      TEXT NOT NULL,      -- JSON preferred, opaque to the engine
    updated_at TEXT NOT NULL       -- crate-stamped, canonical ts
) WITHOUT ROWID;
-- Deliberately: NO log trigger, NO archive membership, NO branch_id.
-- Snapshots include the file, so KV rides backups; transaction-time
-- reconstruction ignores it (documented, tested).
```

**API:** `kv_put(key, value)`, `kv_get(key) -> Option<String>`,
`kv_delete(key)`, `kv_scan(prefix) -> Vec<(String, String)>` (bounded:
`LIMIT` required arg, no unbounded scans). Key rules: non-empty, ≤256 chars,
`[A-Za-z0-9_:./-]+`; app-prefix convention enforced by docs, optionally by
`kv_scan` prefix (no hard partition — one app's prefix is another's query).
Plain overwrite/delete semantics (no versions — that would re-create the
ledger through the back door).

**Effort:** ~2 days (module + bindings + parity tests + docs).

---

## P4. Content-addressed blob store (replaces external `.assets/` + staging)

```sql
CREATE TABLE blobs (
    sha256     TEXT PRIMARY KEY,   -- hex, content address = dedup key
    mime       TEXT NOT NULL,
    bytes      BLOB NOT NULL,
    size       INTEGER NOT NULL,   -- CHECK (size = length(bytes))
    added_at   TEXT NOT NULL,
    refcount   INTEGER NOT NULL DEFAULT 0
) WITHOUT ROWID;
-- Same exclusions as KV: no log trigger, no archive, no branches.
-- Correlate from the ledger side: concept.extra {"blob": "<sha256>"}
-- or edge properties — links stay small, bytes stay put.
```

**API:**
- `blob_put(bytes, mime) -> sha256` (hash-then-insert; duplicate = refcount+1,
  no re-store — the dedup OKFgraph hand-rolls today).
- `blob_get(sha) -> (mime, bytes)`; `blob_stat(sha)` for size/mime without the copy.
- `blob_link(sha)` / `blob_unlink(sha)` refcount ops; `blob_gc() -> count`
  deletes `refcount = 0` rows (only reclamation path — Doctrine V respected:
  unreferenced bytes leave, nothing referenced ever does).
- Chunk ceilings: new `CHUNK_ROWS_BLOBS` byte-budgeted constant (rows × bytes,
  not row count — the lesson from the dim-512 probe); Python crosses it as a
  flat int per `CHUNK_ROWS_*` precedent. Streaming read (`blob_read_chunk`)
  deferred unless a >100MB use case appears.
- Cap: single-blob cap (e.g. 64 MiB default, tunable at open?) — unbounded
  BLOBs in a WAL file punish every checkpoint; say the number out loud.

**Effort:** ~3 days (module + refcount discipline + GC + bindings + tests).

---

## Cross-cutting

- **Migrations:** one rung per P-item (P1 touches hot + cold tables; P3/P4 are
  greenfield tables, no backfill). All idempotent, all covered by the existing
  migration harness. Archive/rehydrate paths must explicitly **ignore** KV/blobs
  (tests asserting a bulk archive leaves sidecars byte-identical).
- **Branching:** sidecars are lineage-blind by design (one KV, one blob pool
  across branches — like embeddings). A branch asserting `extra.blob = X`
  references the shared pool. Document; test `diff` ignores sidecars.
- **Error taxonomy:** no new groups — `ValidationError` (bad JSON, bad key,
  bad charset, over-cap), existing `NotFoundError` for missing blob/key.
  The `kind()` match stays total without new arms.
- **Public API:** `cargo-public-api` bless + Python stub regen + `test_stubs.py`
  in the same commits (D-205 discipline).
- **Docs:** §4 schema (DDL + the three exclusion rules stated once, referenced
  by P3/P4), decision-register entries (one D-number per P-item, rationale +
  rejected alternatives: user DDL, EAV tables, sidecar files), release note
  with the OKFgraph consumption story.
- **Perf gates:** P1 expression index gets an EXPLAIN assertion; P4 gets a
  byte-scale ladder (1/8/64 MiB blobs) in the non-gating perf script; re-run
  the §5 ladders after P1 (extra column rides every concept write — prove it
  costs ~nothing).

---

## Ordering & OKFgraph consumption

| Step | Macrame work | OKFgraph unblocks |
|---|---|---|
| P2 charset | 2 h | `LINKS_TO`-style kinds as-is; no rename step |
| P1 `extra` | ~1 d | frontmatter/tags/layer native; envelope deleted; layer index |
| P3 KV | ~2 d | `FileHash`/`DirHash`/epochs in-file; `meta.json` deleted |
| P4 blobs | ~3 d | `ImageAsset.data` in-file; `okf-asset://` staging deleted |

P1+P2 are independently shippable (0.18); P3/P4 follow (0.18/0.19). Total
~1 week. None of it blocks starting the OKFgraph rework — P1–P4 each delete
exactly one workaround the rework would otherwise encode (envelope, sidecar
file, external assets), so build the workarounds behind narrow interfaces and
retire them as the primitives land.
