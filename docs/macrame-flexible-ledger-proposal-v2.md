# Macrame flexible-ledger proposal, rev. 2 (target: 0.18 / 0.19)

**Status:** revised after review against Macrame 0.17.0 · **Date:** 2026-09-20
**Branch:** `dev_rework` (OKFgraph side) · `dev/0.18.0` (Macrame side)
**Supersedes:** `macrame-flexible-ledger-proposal.md` (draft, 2026-09-11)

**Goal:** unchanged — per-usecase extensibility **without** user DDL. Open the
attribute layer and add guarded sidecars, while the ledger core (time,
assertions, materialization, archive, branching) stays fixed and
guarantee-bearing.

**Non-goals:** unchanged. No arbitrary `CREATE TABLE`, no user-defined rel types
as schema, no changes to Doctrines II–VII.

---

## What this revision changes, and why

The draft was checked line by line against the 0.17.0 source. Four of its claims
did not survive, and one of those changes a design rather than a cost estimate.

1. **P1 is not trigger-free.** The concept log triggers emit an explicit
   `json_object('v', 2, …)`, not `SELECT *`. An `extra` column absent from that
   list is invisible to `reconstruct` and to every `as_of` attribute mode. This
   is the 0.5.6 Wave 1 defect exactly — the payload omitted `embedding_model`
   and every temporal read silently lost it — so `extra` goes into the payload
   and the version becomes **v3**.

2. **The snapshot container needs a bump too.** `MaterializedState` holds
   `NodeAttributes`, serialized with `bincode`, which is not self-describing.
   Per the container's own doc: adding a field does not make an old file fail to
   parse, *it makes it parse into the wrong values* — and a snapshot is the
   first thing a restart reaches for. `SNAP_FORMAT_VERSION` goes 4 → 5, and the
   new field carries `#[serde(default)]`, on D-221's precedent.

3. **The archive round-trip drops the column unless six sites are updated.**
   `concepts` crosses the boundary through explicit column lists — two hot→cold
   inserts, one cold read, two rehydrate inserts, plus the `cold.concepts` DDL
   and the `ColdConcept` struct. D-130 is explicit that a concept moves *column
   for column, `content` included, because a move that drops a column is a
   rewrite and Doctrine V does not permit an absence the ledger cannot explain.*
   Missing `extra` here is a doctrine violation, not an oversight.

4. **P4's reclamation model does not work.** A refcount only knows the present.
   Because `extra` is now fully versioned (1), every historical concept version
   that names a blob references it *permanently*. A `refcount = 0` sweep can
   therefore delete bytes a past ledger state still names — silently, as routine
   housekeeping. **P4 is redesigned below: reclamation is the archive.**

Three things were found to be *cheaper* than the draft assumed, and are noted in
place: D-036 permits additive ledger columns post-1.0 (not only pre-1.0); edge
validation lives in one function that the Python binding reaches rather than
duplicates; and the concept write path is already one shared upsert statement.

---

## P1. `concepts.extra` JSON column

**DDL (new rung, both `concepts` and `cold.concepts`):**
```sql
ALTER TABLE concepts ADD COLUMN extra TEXT NOT NULL DEFAULT '{}';
ALTER TABLE cold.concepts ADD COLUMN extra TEXT NOT NULL DEFAULT '{}';
-- CHECK (json_valid(extra))   -- loud failure on corrupt writes
```
Permitted by D-036 as `ALTER TABLE ADD COLUMN` on a ledger table, which is
additive and does **not** require a major version even after 1.0.

**Semantics:**
- App-defined attributes (OKFgraph: `type`, `tags[]`, `layer`, `description`,
  `resource`, frontmatter leftovers). Namespaced by convention: `okf:type`,
  `myapp:status`.
- **Fully versioned.** `extra` enters the log payload, so `as_of` and
  `reconstruct` answer for it. This is the decision the draft assumed it was
  getting for free; it is the decision, and it is paid for below.
- Size cap 64 KiB at the API boundary (`ValidationError`). Bulk bytes are blobs,
  not attributes. **Open:** sanity-check 64 KiB against OKFgraph's real
  frontmatter sizes before it becomes a constant.

**Concepts payload version 3 — and the ceiling stops being global.** Both
concept log triggers move to `'v', 3` with `extra` in the `json_object`. The
**links** marker stays at 1: the two payload shapes version independently, which
the read side has not honoured since 0.5.6 took concepts to v2 and left links at
1. One constant, `PAYLOAD_VERSION`, is checked against every row before the code
branches on which shape it is, so a links row stamped 2 has passed that gate and
decoded under v1 field names ever since — harmless only because nothing but the
crate writes links versions. 0.18 does not open that gap, it widens it, so the
check moves behind the shape dispatch and reads per-shape ceilings. Cheap: the
constant is crate-private, the error already carries the `max` it would report,
and the Python binding destructures the variant field-for-field, so nothing on
the surface moves. **What this means for OKFgraph:** nothing in the API, and one
fewer way for a hand-written or migrated log row to be folded as something it
is not.

The rung **drops and recreates** the concept triggers — `CREATE TRIGGER IF
NOT EXISTS` on an existing name keeps the *old body*, which is precisely the
D-126 / D-129 failure, and `verify` body-checks only the three delete guards, so
a re-issued baseline would leave a v2 trigger in place and pass verification in
silence. Two decode sites take the new field (`temporal::replay` and
`temporal::as_of`); both already tolerate absent fields, so v1 and v2 payloads
keep folding unchanged.

**Verification learns the version.** `verify` gains a body probe for the version
marker in the two concept log triggers, mirroring the archive-session marker
probe it already runs on the delete guards. A stale trigger then refuses the
open instead of quietly writing incomplete history.

**Snapshot container v5.** `NodeAttributes` gains `extra`, `SNAP_FORMAT_VERSION`
goes to 5, the field carries `#[serde(default)]`. A v4 snapshot meeting a v5
build is refused and the fold runs from the log — the documented behaviour, and
it costs time rather than information. To be precise about which half protects
what: **the version number does all of it.** A serde default pads no bytes into
a short `bincode` buffer, so it cannot rescue a v4 file and is invisible to that
decode. The default buys only that the *field* stays additive if the container
is ever versioned some other way — D-221's hedge, restated here because it reads
as something stronger than it is.

**Archive round-trip.** All six sites carry `extra`, plus a `cold_has_extra`
probe mirroring `cold_has_branch`: a pre-0.18 cold file has no such column, and
the read path tolerates its absence by projecting `'{}' AS extra` rather than
upgrading a file it was asked to read. The archive *writer* upgrades with
`ALTER TABLE cold.… ADD COLUMN`, which is the shape already in use for
`branch_id`. **Acceptance test:** archive a concept carrying `extra`, rehydrate
it, assert byte equality.

**Upsert semantics — omission preserves.** `ConceptUpsert::extra` is
`Option<String>`. `None` means *leave the column alone*; `Some(v)` replaces it
wholesale, matching title/content. Implemented inside the single shared
`UPSERT_CONCEPT` statement by referencing the bind parameter directly in the
`DO UPDATE SET` clause, so the single and chunked write paths still cannot drift
(D-056). The alternative — assigning `excluded.extra` like every other column —
means any caller re-upserting a concept without calling `.extra()` wipes the
app's attributes to `{}`, and because `extra` is versioned that erasure is
recorded as a deliberate belief change. The log trigger reads the post-update
row, so preserve-on-omission puts no gap in the history.

**Clearing is `Some("{}")`, and there is no state below it.** The column is
`NOT NULL DEFAULT '{}'`, so SQL NULL is unreachable and the builder's `Option`
means *unstated*, never *null*. OKFgraph never has to distinguish "no `extra`"
from "`extra` is null" — only one of those can exist.

**Indexing:**
```sql
CREATE INDEX idx_concepts_extra_layer
    ON concepts (json_extract(extra, '$.layer'));
```
`json_extract` is already used throughout the crate, so JSON1 is present in this
libSQL build. **This would be the schema's first expression index** — none exist
today — so the `EXPLAIN` assertion is a real gate, not a formality, and should
be written before the index is relied on.

Apps register their own via `register_extra_index(path)`, which validates the
path and issues a create-if-absent. **No registry table**: the app calls it
unconditionally at startup, so a restored backup rebuilds its indexes on the
next open, and `verify` already tolerates indexes it did not create (it checks
for the absence of what it requires, not for the presence of only what it knows).

**API:**
- Rust: `ConceptUpsert::extra(impl Into<String>)`, getter on reads and on
  `NodeAttributes`, `Database::register_extra_index(path)`.
- Python: `ConceptUpsert(..., extra={...})` accepting `dict`/`str`, returned as
  `dict`.
- Errors: `ValidationError` for malformed JSON and over-cap. No new taxonomy.

**Effort: 4 days.** Rung + drop/recreate + payload v3 + two decode sites +
`verify` probe + snapshot v5 + six archive sites + `cold_has_extra` + upsert
semantics + two bindings + `public-api` bless + stub regen.

---

## P2. Edge-kind charset relaxation

**Change:** `edge_type` validation `[A-Z0-9]+` → `[A-Za-z0-9_:.\-]+`, **plus two
new rules**:

- **One case per kind.** A kind may be all-upper or all-lower; mixing is
  refused. `edge_type` sits inside the primary keys of `links` and
  `links_current`, nothing in the schema uses case-insensitive collation, and
  uppercase-only made case collisions impossible by construction. Without this
  rule `okf:cites` and `okf:Cites` are two different edge kinds that look
  identical in every log line, error message and diff. Three lines in the
  validator; every existing kind still passes. **The rule narrows that hazard
  and does not eliminate it:** `okf:cites` and `OKF:CITES` both remain legal and
  distinct. What it buys is the *near-miss* — two spellings a reader cannot tell
  apart — while collision-impossibility is a property only the uppercase-only
  rule had, and the relaxation spends it. Worth knowing when choosing kind names
  on the OKFgraph side: pick one case and hold to it.
- **Maximum 64 characters.** There is no length limit today — the draft's "max
  length unchanged" describes a rule that does not exist. The kind goes into a
  primary key and into the composed `source|target|type|valid_from` history
  identifier, so an unbounded one bloats an index the traversal path depends on.
  **The cap cannot refuse data already on disk, but not because 64 is generous** —
  with no length bound today, a database may legally hold a 100-character kind.
  It is because validation is *write-only*: the validator's own doc records that
  it runs from `EdgeAssertion::normalized` on the write path and that the
  traversal builder never called it. Nothing re-checks a stored kind on fold,
  verify, archive or rehydrate, so an over-length kind keeps reading and
  traversing and only a new write of one is refused. That property is
  load-bearing for the cap and gets an assertion of its own.

**The validator's test flips and is rewritten as part of the item.** Today
`edge_types_are_uppercase_alphanumeric` asserts seven cases are *rejected*;
`"knows"`, `"KNOWS_WELL"` and `"KNOWS-WELL"` all become legal, while `"A|B"`,
`"O'BRIEN"`, `"ÉTAT"` and the empty string stay refused. New cases for mixed
case, over-length and each newly admitted character go in beside the survivors.

**Convention (docs, not code):** `prefix:kind` namespacing — `okf:links-to`,
`myapp:cites`. **Note the change from the draft's `prefix:Kind`**, which the
single-case rule refuses. Core kinds stay bare (`LINKSTO`), and existing
databases keep working.

The `|`-delimited history identifier is safe: none of `_ : . -` is `|`, so no
component can contain the separator. The doc comment asserting `[A-Z0-9]+` needs
rewriting; the invariant it protects survives.

**Cheaper than the draft assumed.** Validation lives in exactly one function and
the Python binding reaches it through the error type rather than duplicating a
regex, so there is no second implementation to keep in sync and no shared test
vector to maintain.

**Forward compatibility:** a 0.18 database holding lowercase kinds handed to a
0.17 binary fails validation. Release-note line.

**Effort: ~2 hours** + release note.

---

## P3. KV sidecar

**Rationale:** unchanged and still correct. Small operational state (hashes,
epochs, counters, cursors) is neither belief nor content — it needs durability
plus a backup story without history. Doctrine VII's reasoning about embeddings
applies verbatim.

```sql
CREATE TABLE kv_store (
    key        TEXT PRIMARY KEY,   -- 'okf:filehash:<path>' | 'myapp:epoch'
    value      TEXT NOT NULL,      -- JSON preferred, opaque to the engine
    updated_at TEXT NOT NULL,
    CHECK (updated_at GLOB '<canonical-ts-glob>')
) WITHOUT ROWID;
-- NO log trigger, NO archive membership, NO branch_id.
```

**The timestamp is constrained on disk**, using the same `canonical_ts_check!`
macro every table in the schema that stamps a timestamp uses — six of them, not
the four the architecture doc's wording names. D.1 item 3 freezes the canonical
form as "a fact about the disk as well as the API, so it is frozen twice over" —
a column the crate calls canonical should be one the file enforces. The draft
annotated it as canonical and left it unchecked.

**`kv_store` is branch-global, and that is a semantic rather than an omission.**
No `branch_id` means one cursor, one epoch, one counter across every lineage,
unaffected by working on a branch — right for operational state, which describes
the process rather than anything the ledger believes. Nothing in Macrame trips
over it: a branch is a row plus a label carried on writes, so there is no
physical copy and no path that enumerates tables. **The consequence for
OKFgraph:** a `FileHash` or epoch written while on a branch is visible from
every other branch and survives switching away. If per-branch operational state
is ever wanted, the branch name goes in the key.

**`WITHOUT ROWID` is right here** — small rows, text primary key, no large
payload.

**Correction to the draft's backup claim.** "Snapshots include the file, so KV
rides backups" is wrong: a Macrame snapshot is a zstd-compressed `bincode`
`MaterializedState` written to a snapshots *directory*, containing no KV and no
blobs, and it never could. The true statement is that KV lives in the database
file, so **file-level backups** carry it, while transaction-time reconstruction
ignores it entirely. The distinction is the whole of P3's exclusion argument, so
it is worth getting right.

**API:** `kv_put(key, value)`, `kv_get(key) -> Option<String>`,
`kv_delete(key)`, `kv_scan(prefix, limit) -> Vec<(String, String)>` — `limit` a
required argument, no unbounded scans. Key rules: non-empty, ≤256 chars,
`[A-Za-z0-9_:./-]+`. Plain overwrite/delete semantics; no versions, which would
re-create the ledger through the back door.

**Effort: ~2 days** (module + bindings + parity tests + docs).

---

## P4. Content-addressed blob store — **redesigned**

**The draft's reclamation model is withdrawn.** `refcount`, `blob_link`,
`blob_unlink` and `blob_gc` are all removed. Because `extra` is versioned, a
concept version from March still names blob `X` forever; a counter that tracks
only present references will drop to zero and delete bytes that a past state
depends on. The failure is silent, arrives months later, and looks like a
missing image in an old revision with nothing in the logs.

**Reclamation is the archive.** This system already has exactly one way to
reclaim space, and Doctrine V permits no physical delete outside it. A blob
becomes eligible when **every log entry naming it has gone cold** — reachability,
not expiry, which is the predicate shape D-128 established for concepts — and
then it **moves to `cold.blobs`** rather than being deleted. Old states stay
readable from the archive, and there is one reclamation path rather than two
competing ones.

**Eligibility is cross-lineage, which falls out rather than being chosen.** All
lineages write into one log, so "every entry naming it has gone cold" is already
a whole-log question. A blob named only by a side branch stays hot until that
branch's entries go, which they do through `archive_branch` — the named
operation that moves one lineage's rows wholesale. So a blob outlives the branch
that introduced it exactly as long as that branch's log entries do, and
abandoning a branch reclaims its bytes only by archiving it. Merging changes
nothing: a merge writes new entries naming the same content address, which is
one more reference into a pool that was already shared.

```sql
CREATE TABLE blobs (
    sha256     TEXT NOT NULL UNIQUE,   -- hex, content address = dedup key
    mime       TEXT NOT NULL,
    bytes      BLOB NOT NULL,
    size       INTEGER NOT NULL,       -- CHECK (size = length(bytes))
    added_at   TEXT NOT NULL,
    CHECK (added_at GLOB '<canonical-ts-glob>')
);
-- A rowid table, NOT `WITHOUT ROWID`: that form stores the whole row in the
-- index b-tree, so a multi-megabyte payload becomes a long overflow chain
-- hanging off the key structure.
-- `cold.blobs` mirrors this, trigger-free and FK-free like `cold.concepts`.
```

**Size limit: 8 MiB by default, overridable through `Tuning`.** libsql 0.9.30
exposes blobs only as `Value::Blob(Vec<u8>)` — there is **no** incremental blob
API, so the draft's deferred `blob_read_chunk` is not deferred, it is
unimplementable at this driver version. Every write allocates the file whole,
every read allocates it whole again, and the driver's own copy puts peak memory
at roughly two to three times the file size per operation. 8 MiB covers
OKFgraph's images and documents at ~24 MiB peak; the draft's 64 MiB is ~190 MiB
peak, in a crate that boxed an error variant over 168 bytes. A deployment that
genuinely needs larger files says so explicitly via `Tuning`, alongside the
checkpoint policy and cache sizes it already carries.

**Blob writes are exempt from the chunk budget, by contract.** The `CHUNK_ROWS_*`
constants exist to bound *duration* against a 3 ms lock hold; a blob is one row
and cannot be split, so no row count helps and the draft's `CHUNK_ROWS_BLOBS` is
the wrong shape. The precedent is `write_bulk_atomic`: a documented exemption
with its own warn-above-hold threshold, as `BULK_ATOMIC_WARN_HOLD` is for that
path. WAL pressure is managed through the existing `Tuning::wal_autocheckpoint`
policy (D-157) rather than anything new.

**API:**
- `blob_put(bytes, mime) -> sha256` — hash-then-insert; a duplicate is a no-op
  returning the same address, which is the dedup OKFgraph hand-rolls today.
- `blob_get(sha) -> (mime, bytes)`; `blob_stat(sha)` for size and mime without
  the copy.
- No `blob_link` / `blob_unlink` / `blob_gc`.
- Correlate from the ledger side: `concept.extra {"blob": "<sha256>"}` or edge
  properties — links stay small, bytes stay put.

**P4 now depends on P1** and cannot start before it: eligibility is computed
from blob references inside log payloads, which do not exist until `extra` is
versioned.

**Effort: 4 days+**, and it is a 0.19 item.

---

## Cross-cutting

- **The three exclusion rules are no longer shared.** The draft stated "no log
  trigger, no archive membership, no branch_id" once and referenced it from P3
  and P4. After the redesign only **`kv_store`** keeps all three. `blobs` keeps
  two — no log trigger, no lineage — and **is** an archive participant. State
  them per table.
- **Migrations:** one rung per P-item. P1 touches hot and cold `concepts` and
  drops/recreates two triggers; P3 and P4 are greenfield. All idempotent, all
  under the existing harness.
- **Branching:** KV is lineage-blind by design (one store across branches, like
  embeddings). Blobs are lineage-blind in *storage* — one pool — while their
  eligibility is computed across all lineages' logs, so a branch asserting
  `extra.blob = X` keeps the shared bytes alive. Document; test that `diff`
  ignores both.
- **Error taxonomy:** no new groups. `ValidationError` for bad JSON, bad key,
  bad charset, mixed case, over-length, over-cap; `NotFoundError` for a missing
  blob or key. The `kind()` match stays total.
- **Forward compatibility, two hard edges and one corollary:** once 0.18 writes
  any concept, a 0.17 build cannot fold the log at all — it refuses newer
  payload versions by design. And v4 snapshots are refused by a v5 build rather
  than misread. Both belong in the release note; the second costs a rebuild and
  no information. The corollary is worth printing beside them, because it is
  narrower than people will assume: **a 0.18 binary writing only edges stays
  readable by 0.17**, since the marker is per entry and the links shape is
  untouched at 1. Only concept writes close that door.
- **Public API:** `cargo-public-api` bless + Python stub regen + `test_stubs.py`
  in the same commits (D-205).
- **Docs:** §4 schema, decision-register entries D-278 … D-282 (drafted in the
  Macrame repo as `docs/Macrame Update Plan v0.18.0.md`), release note with the
  OKFgraph consumption story.
- **Perf gates:** the P1 expression index gets its `EXPLAIN` assertion — first of
  its kind in this schema, so treat a failure as likely rather than surprising —
  **and a negative control**, confirming the index is *not* chosen for the
  `extra ->> '$.layer'` spelling, which is a different expression tree and
  cannot match it. Without that half the gate passes while a caller quietly
  emits the other form and gets the scan this item exists to delete. **OKFgraph
  must use the pinned `json_extract(extra, '$.layer')` form**, which the binding
  will expose rather than leaving to hand-written SQL. Re-run the §9 budgets
  after P1, since `extra` rides every concept write. P4 gets a byte-scale ladder
  (1 / 4 / 8 MiB) in the non-gating perf script.

---

## Ordering & OKFgraph consumption

The chain is now a **requirement**, not a preference: P4 cannot start before P1.

| Step | Macrame work | Release | OKFgraph unblocks |
|---|---|---|---|
| P2 charset | ~2 h | 0.18 | `okf:links-to`-style kinds as-is; no rename step |
| P3 KV | ~2 d | 0.18 | `FileHash`/`DirHash`/epochs in-file; `meta.json` deleted |
| P1 `extra` | 4 d | 0.18 | frontmatter/tags/layer native; envelope deleted; layer index |
| P4 blobs | 4 d+ | 0.19 | `ImageAsset.data` in-file; `okf-asset://` staging deleted |

0.18 is roughly a week and deletes three of OKFgraph's four workarounds. Blobs
get 0.19 to themselves, which they now need, having acquired a cold-storage
table and a dependency on P1.

None of this blocks starting the OKFgraph rework. Each item still retires
exactly one workaround, so build the workarounds behind narrow interfaces and
delete them as the primitives land — with the ordering above telling you which
interface gets retired first.
