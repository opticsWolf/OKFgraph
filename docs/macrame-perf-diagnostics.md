# Macrame bottleneck diagnostics — test & fix plan

**Status:** proposed · **Date:** 2026-09-10 · **Branch:** `dev_rework`
**Trigger:** spikeladders (same box, seeded RNG, `.venv` with `macrame-db==0.16.0`):

| Observation | Numbers |
|---|---|
| Edge `bulk_import` superlinear | 2k→0.5s, 4k→2.3s, 8k→7.9s, 16k→29s (~4× per 2×; 40k never finished in budget) |
| Concept writes fast | 20k in ~1.0–1.2s |
| Vector build superlinear in dim (N=5k) | 64→11.5s, 256→82s, 512→did not finish |
| Reads fine at all scales | traverse 0.1ms, vector top-10 ≤9ms, keyword ≤21ms @20k |
| Footprint | 260MB + 11MB snapshots vs 79MB ladybug @20k docs/40k edges/64-dim |
| Reference (ladybug) | writes linear ~6k rows/s; vector build ~linear in bytes (3.5/7.5/12.3s for 64/256/512 @5k) |

Goal: attribute each bottleneck to **crate** (triggers, materialization, chunking,
binding) vs **engine** (libSQL FTS5/DiskANN/WAL), then fix at the right layer.

---

## 0. Measurement discipline (read first)

- **Medians of ≥3 sessions.** Session-to-session spread is ~29% on this project's
  own hardware (D-070). Single runs prove nothing.
- **Control query per session.** Time a `SELECT 1` round-trip (or equivalent)
  alongside every measurement; if the control moved, the machine moved, not the code.
- **Fresh DB per run, one open per DB.** R15 (cumulative-`connect()` access
  violation, libSQL 0.9.30): never reuse a path across runs in one process, and
  never open the same file twice concurrently from the harness.
- **Record storage state with every timing:** `PRAGMA page_count`,
  `freelist_count`, WAL bytes, snapshot-dir bytes. Growth explains superlinearity.
- **Per-chunk holds are the primary signal.** Pass `progress=` to
  `write_concepts` / `bulk_import` / `upsert_embeddings` and log
  `(chunk_index, rows, held_ms)`:
  - holds **grow** over the run → cost scales with n (index depth, trigger fan-out,
    transient-index rebuilds, degrading query plans);
  - holds **flat** → fixed per-chunk overhead × chunk count (Write Actor round-trips,
    transaction open/commit, binding/GIL).

---

## 1. Instrument — no code changes

1. **Per-chunk hold curves** for: 16k edges in one `bulk_import`; same 16k as
   8×2k calls; 5k vectors at dim 64/256. Plot hold vs chunk index.
2. **`metrics()` before/after** each run: kind counters + `budget_violations()`.
   (`metrics` ships on in the wheel; no feature flag needed from Python.)
3. **EXPLAIN QUERY PLAN on the hot path.** Find the statements the Write Actor
   executes per chunk (see `graph.rs` bulk paths) and run them under
   `EXPLAIN QUERY PLAN` via `diagnostic_query`/`diagnostic_conn` on a populated
   file. Red flags, in order:
   - `AUTOMATIC COVERING INDEX` — SQLite building a throwaway index per statement,
     O(n) each, O(n²) over a bulk. This is the prime suspect for the edge ladder:
     it is the exact pathology D-151 fixed on the archive path (`SCAN links` →
     per-statement auto-index → added `idx_links_target`).
   - `SCAN links` / `SCAN links_current` anywhere in a per-row/per-chunk statement.
   - `USING INDEX` with the *wrong* index after 10× table growth mid-bulk
     (planner staleness; cf. §3d).
4. **Snapshot accounting.** Snapshot-dir bytes before/after bulk with default
   cadence vs `snapshot_every_entries=None` (one prior run showed
   12.8s vs 24.7s — repeat 3× before concluding; that delta smells like noise).

---

## 2. Layer isolation — crate vs engine

Run these against **raw libSQL** (plain client, no macrame) on identical data:

- (a) Plain `INSERT` 40k rows into a `links`-shaped table, no triggers/indexes → engine row-write floor.
- (b) Same + FTS5 external-content table + triggers → FTS insert tax.
- (c) `F32_BLOB` table + `USING diskann`, insert 5k vectors at dim 64/256/512,
  time inserts and a top-10 query → DiskANN build/query cost in isolation.

Then compare with the macrame path. The deltas attribute cleanly:

| Delta (macrame − raw) | Attributed to |
|---|---|
| (bulk edges) − (a) | triggers + `transaction_log` + `links_current` + actor/chunking |
| (bulk edges) − (b)-ish | materialization + guard checks specifically |
| (upsert_embeddings) − (c) | chunking/binding overhead vs pure index build |

- (d) **Binding cost:** `bulk_import(2000×1 call)` vs 2000× single `assert_edge`
  vs 1× `write_bulk_atomic` → per-call (GIL detach, `block_on`, channel) vs per-row.

---

## 3. Caller-visible knob sweeps (no fork changes)

For a fixed total (16k edges / 5k vectors), sweep one variable at a time:

- (a) **Call size:** 1×16k, 2×8k, 4×4k, 8×2k, 16×1k. The ladder already hints the
  optimum is ~2k/call; confirm and document it as the supported bulk recipe.
- (b) **WAL:** default autocheckpoint vs `"disabled"` + explicit `checkpoint()`
  after bulk. Read back `CheckpointReport` (`busy`, frames moved). Bulk inserts
  through log triggers generate far more WAL pages than the row count suggests;
  repeated autocheckpoints during a bulk are O(WAL) each.
- (c) **Snapshot cadence:** default vs `None`, 3 sessions each.
- (d) **`analyze()` before bulk** (planner stats on empty tables go stale as the
  ledger grows 10–100× mid-run; a mid-bulk plan flip looks exactly like
  superlinearity).
- (e) **Edge kinds:** chain edges (disjoint keys, low degree) vs random edges
  (hub formation) — separates overlap-guard cost (O(versions per key)) from
  degree effects.

---

## 4. Candidate fixes, mapped to findings

| If you find | Fix | Where | Precedent/notes |
|---|---|---|---|
| F1. Per-row `links_current` maintenance dominates (§2b) | Bulk mode that writes ledger rows **without** maintaining the materialization, then `rebuild_current_chunked()` once at the end | crate (new bulk flag or documented recipe) | Sanctioned by Doctrine VI (derivative state is disposable); rebuild + audit paths already exist |
| F2. Transient auto-indexes / SCANs in EXPLAIN (§1.3) | Add the permanent covering index the planner wants | schema rung + migration | Pre-1.0 window per D-036/D-032; precedents D-151 (`idx_links_target`), D-273 (lineage index); add EXPLAIN assertions per project convention |
| F3. WAL checkpoint churn (§3b) | Documented bulk recipe: disable autocheckpoint, explicit `checkpoint()` after | docs + caller code | No crate change; verify via `CheckpointReport` |
| F4. Snapshot anchoring mid-bulk (§1.4) | Defer cadence across bulk (pause/resume or open flag) | crate if knob insufficient | Measure first — prior single run says cadence is *not* the cost |
| F5. Embedding chunk ceiling row-based (`CHUNK_ROWS_EMBEDDINGS`=30) oblivious to width | Byte-budgeted chunking (`rows × dim × 4 B`) instead of row-counted | crate (`connection::chunk_rows` + adaptive loop) | If §2c shows the *index build itself* dominates, this only helps partially — then see F6 |
| F6. DiskANN build itself is superlinear in dim (engine-side) | Accept + route around: default dim 256, wide models as offline backfill; per-model tables make this additive, not a migration | app-side (OKFgraph) + upstream libSQL tuning | Staging alternative unsupported today (index mandatory per D-037 — dropping it disarms dim enforcement) |
| F7. Per-call binding overhead (§2d) | Keep batches at the §3a optimum; document call-size guidance | docs | No code change |

Suggested order: §1 (a day, mostly scripts) → §2c/§3a (same day, biggest lever:
if 8×2k ≈ 10s for 16k edges, the OKFgraph importer just uses that and the urgency
drops) → §2a–c → §4 as findings dictate.

---

## 5. Lock it in — regression gates

- Commit the 2k/20k bench + edge ladder + dim sweep as a **non-gating** perf
  script (absolute budgets are not CI gates per D-055 — shared runners lie;
  use criterion baselines, machine-against-itself, per project convention).
- Gate instead on structure: EXPLAIN assertions for every index-sensitive
  statement (existing convention), and a budget-violation test on the bulk path
  (`metrics().budget_violations()` naming bulk chunks → fail).
- Re-run the ladder after each fix; publish the new table in the release note
  (the README performance table is a per-release measured series — extend it,
  don't overwrite history).

---

## 7. Re-test: macrame 0.16.1 (2026-09-11)

Wheel `whl/macrame_db-0.16.1-cp310-abi3-win_amd64.whl`, same battery, same box.
Correctness 11/11, cross-engine ANN scores agree exactly (0.02004 both).

| Test | 0.16.0 | 0.16.1 | ladybug (ref) |
|---|---|---|---|
| Bulk 2k concepts + 4k edges | 1335ms | 927ms (concepts 132ms, edges 796ms) | 1010ms |
| Write 20k + 40k | 86s | **15.7s** (concepts 1.5s, edges 14.2s) | 10.5s |
| Edge ladder 2k / 4k / 8k / 16k | 0.52 / 2.3 / 7.9 / 29.2s | 0.41 / 1.3 / 3.5 / 8.0s | linear |
| Ladder scaling per 2× batch | ~4× | ~2.3–2.7× | ~2× |
| Vector build 2k×256 | (82s @5k) | 29.8s | 3.5s |
| Vector build 2k×512 | stalled @5k | **58.0s, completes** | 5.6s |
| Vector top-10 @512 | — | 16.4ms | 4.5ms |

Edge-write bottleneck largely addressed (2–3.6× faster, near-linear).
Vector build at wide dims still ~10× ladybug — F5/F6 (byte-budgeted chunks,
backfill strategy) remain the open items; serving latency (16ms) is acceptable.

---

## 6. What the outcomes mean

- **F1/F2 land** (bulk ≈ linear, near concept-write rate) → macrame carries the
  full rework including vectors at 256 now, 512 as additive model later.
- **Only F3/F7 land** (bulk workable via recipe: ~2k calls, checkpoint discipline,
  dim ≤256) → proceed on macrame with the importer written to the recipe;
  512-dim as offline backfill (F6).
- **Nothing moves, engine-bound (F6)** → split backend honestly: macrame for
  graph + temporal + FTS, vectors stay on ladybug (or deferred). More ops
  complexity, but each engine does what it's good at.
