# Ladybug index-build crash — watch log

Companion to the watch decision (`thoughts/ladybug-index-crash/watch-decision`
in the live graph): any new occurrence of the silent halting/segfault during
import (links/index phase) gets a row here — the repo copy is the durable,
reviewable record; the live graph gets the graph-side thought on next session.

Environment for every entry below: editable install of this repo's working
tree (branch as noted), ladybug as pinned per branch, Windows, scratch
databases only (`OKFGRAPH_DB_PATH` to a temp dir), `import --all` against an
absolute `--bundle-root`. Live `kb.db` was never opened from here.

## Occurrences

### 2026-11-02 07:00 UTC — suspected, unconfirmed

- **Branch/code:** `dev_0.11.0` at `f94e506` + local commits (label edits),
  `ladybug==0.22.1`.
- **Workload:** fresh scratch DB, `okf import --all --bundle-root
  D:/User/Documents/Python/Macrame_docs` (195 concepts).
- **Progress:** per-bucket encode logs completed (195 texts in 178.3 s /
  166.9 s in the two runs), `upsert: 195 concepts` logged.
- **Death/hang:** ≥30 min with zero further output past upsert (normally the
  links/images/reindex phase logs or completes in seconds–minutes) and the
  process kept running, unkillable-by-stdout — operator stopped it.
  No traceback, no `okfgraph/crash` report file, no Windows fault event found
  post-hoc. This run **was** operator-killed, so the occurrence can only be
  recorded as *suspected* (a silent hang, not a confirmed segfault).
- **Caveat:** an earlier same-day run on a DB copy was killed by a 25-min
  script timeout mid-reindex — operator-caused, **not** an occurrence.

### 2026-11-02 07:25 UTC — verification attempt, inconclusive

Second fresh import, same shape as above: the run left **no evidence in
either direction** (no output log written, no process after ~5 min, no DB
files created, no crash dir entry, no fault event). The harness itself
misbehaved (empty env capture, vanished redirect file), so instrumentation
failure cannot be excluded. Not counted as an occurrence.

### On file from before the watch (context)

`%LOCALAPPDATA%/okfgraph/crash/` holds two entries: 2026-10-31 22:41
(1857 B) and 22:44 (2523 B, the one carrying the recorded traceback) —
pre-dating `f94e506`.

## Verification protocol (agreed way to get a clean data point)

Run the import **in the foreground with live output** (no `nohup`, no
copied-DB shortcuts during the observation window), poll aliveness from a
second shell, and let either the process exit or the user halt it. Both
listed occurrences failed this — that is why neither is confirmed. The
convenience data point: the next Macrame acceptance baseline run
(implementation plan §11) is exactly this shape, never on a copied DB, and
doubles as the verification run.

## Ingest payloads for the live graph (paste for `kind=thoughts`)

Topic `ladybug-index-crash`, tags `watch`, `occurrence`:

- *Suspected occurrence 2026-11-02 07:00 UTC (unconfirmed):* fresh
  `--all` import of Macrame_docs (195 concepts) on `dev_0.11.0`/ladybug
  0.22.1 logged encode + upsert, then ≥30 min silent with the process still
  alive (links/index phase); halted by operator. No crash-file traceback, no
  WER event. Repo log:
  `docs/ladybug-index-crash-log.md`. Same-day 07:25 verification run is
  inconclusive (instrumentation failed). Foreground protocol agreed; count
  = 1 suspected, 0 confirmed.
