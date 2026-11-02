# Ladybug index-build crash — watch log

Companion to the watch decision (`thoughts/ladybug-index-crash/watch-decision`
in the live graph): occurrences of native halts/aborts during import and
index work get a row here. The repo copy is the durable record; the live
graph gets the graph-side thought on next session.

Environment for every 2026-11-02 entry: editable install of this repo's
working tree (dev_0.11.0), `ladybug==0.22.1`, Windows 11, scratch databases
only (`OKFGRAPH_DB_PATH` to a temp dir), `okf import --all --bundle-root
D:/User/Documents/Python/Macrame_docs` (195 concepts). Live `kb.db` was
never opened from here.

## Occurrences

### 2026-11-02 — post-upsert silent slow-crawl (3 reproductions, watch-class)

Consistent shape across three fresh scratch-DB imports, all on the same
bundle. Each time encode (`~170 s`) and `upsert: 195 concepts` complete and
log, then **zero log output and an enormous, continuous DB-write phase**:

| Run | Start | Observation at stop | f.db |
|---|---|---|---|
| 1 | 06:31 UTC | killed by operator at +25 min (script timeout) | — |
| 2 | 07:00 UTC | silent ≥9 min, killed by operator | 90 MB, WAL 3.9 MB |
| 3 | 08:01 UTC | **watched live until +28 min**: process alive (`okf.exe`, ~5 MB RSS), silent, f.db grew 121→174 MB over 3 min, WAL ~7.5 MB; killed by operator | 174 MB, WAL 7.5 MB |

No crash artefacts in any of the three: no traceback, no crash-file in
`%LOCALAPPDATA%/okfgraph/crash/` (latest remains 2026-10-31), no Windows
fault event. Interpretation: **not a native death — a pathological,
unlogged post-upsert phase** (links/index/FTS build region) that runs
≥28 min for 195 concepts. For scale, the archived pre-upgrade copy of the
same graph is ~90 MB; a completed 0.10.x-era import of this bundle is a
minutes-scale operation, so the silent phase is a large regression in
wall-clock and observability. Classification within the watch: **confirmed
recurring slow-path/hang-suspect, no evidence yet of a hard crash.**
Reproducing cleanly needs a long foreground run; next opportunity is the
Macrame acceptance baseline (implementation plan §11).

### 2026-11-02 10:38–10:43 UTC — CPU-wheel A/B probe ( inconclusive, new datum )

Same fresh import under the **hermetic gate venv** (`uv sync --frozen
--extra dev --extra pdf --extra cpu`, i.e. the shipped CPU wheel, Ladybedbug
same 0.22.1): the run never reached the post-upsert phase — encode itself
slowed geometrically (bucket deltas 13 s → 25 s → 43 s → 93 s, roughly
doubling per bucket; bucket 5 = 8,300 chars in 93 s ≈ 90 chars/s) and was
halted by operator before encode would finish. So the post-upsert
slow-crawl question stays undecided under the CPU wheel, but the run adds
a separate datum: **CPU-wheel encode on this box is drastically slower
than the GPU-wheel venv's encode** (195 texts ≈ 170 s total there vs
>>30 min pace here). Next probe should bisect encoder open (threads/
`cpu_arena`) vs the long-text bucketing path — off the watch list, it is a
performance-fit issue, not a halting issue.

### Verification runs kept honest

The 2026-10-31 22:41/22:44 crash-dir files in
`%LOCALAPPDATA%/okfgraph/crash/` are **Python-level `OKFError`-family
reports** (the 22:44 one is a `PermissionError` from `ingest --kind md`
against a directory path) — **a different bug family**, unrelated to the
watched native import crash; do not confuse them with occurrence rows.

## Separately observed same day (CI, Linux) — for awareness, not this watch

A scratch bisect workflow (`full-diag.yml`, deleted after diagnosis) on
`dev_0.11.0`/ubuntu-24.04 reproduced, on the release-track suite:

- **A dev extra alone / B `'not slow'`:** the whole suite *completed*
  (multi-hundreds of expected no-ort failures/errors), then the process
  **aborted during teardown with a Rust `thread … caused non-unwinding
  panic. aborting.`** (exit 134), i.e. a native component (embroider/Rust
  side) panics while dying — after all tests have printed. Linux-only
  observation; Windows full suite runs did not show this. Real code-level
  cleanup candidate (teardown/drop order of Rust-spawned threads), fix
  wanted before 0.11.0's release gate on CI.
- **D/E jobs with `--extra pdf --extra cpu`:** both were terminated by
  GitHub (`The operation was canceled.`) mid-suite at *different, unrelated*
  test positions, and the historical Sunday runs show the same external
  cancel signature — a GitHub-hosted-runner kill, not our code. Everything
  about the weekly `full.yml` failure pattern is explained by (a) draft
  test-rot (no-ort model tests on the dev extra) plus (b) this external
  cancel. Suite runtime measured at **<4 min incl. test_detach.py** when
  not killed externally.

## Ingest payloads for the live graph (paste for `kind=thoughts`)

Topic `ladybug-index-crash`, tags `watch`, `occurrence`:

- *Confirmed persistent slow-path 2026-11-02 (3 reproductions):* on
  `dev_0.11.0` + ladybug 0.22.1, a fresh `--all` import of Macrame_docs
  (195 concepts) completes encode and upsert with normal logs, then enters
  an unlogged silent phase ≥28 min (run 3 watched alive the whole way,
  f.db 121→174 MB) before being halted by operator. No crash artefacts;
  classify as hang/silent-slow-path rather than a native death until a
  longer run or stack capture says otherwise. Repo log:
  `docs/ladybug-index-crash-log.md`.
