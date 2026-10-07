# OKFgraph 0.10.1 — improvement plan (graph-reunification follow-ups)

**Status:** draft on `dev_0.10.1` (forked from `main@0e803f4`, v0.10.0). No code changed yet.
**Source:** field report `D:\User\Documents\Python\okf-graph-issues-20261007.md` — issues observed 2026-10-07 while reuniting the split `kb.db` / `okfgraph.db` pair in `D:\User\Documents\Python\OKFgraph` (OKFgraph 0.10.0, `.venv` install).
**Scope:** patch release. Forward behavior + diagnostics only. No schema change, no migration, no embedding/doctor-score work, no MCP changes.

Pushing a `v*` tag runs `.github/workflows/release.yml`, which publishes to PyPI. That step cannot be undone (PyPI never accepts the same version twice), so the tag is the last step and needs an explicit go-ahead — same rule as 0.10.0.

## 0. Background: how the split happened

- `kb.db` (~79 MB) is the long-lived graph: decision register, architecture review, improvement plans, thoughts through 2026-10-06.
- On 2026-10-07 two `okf-mcp` servers held `kb.db` open (`DB_LOCKED` on every CLI write), so the day's session thoughts (CodeRadar v0.12 landings) were ingested into a second database, `okfgraph.db` (~17 MB, 8 thoughts, topic `coderadar-v0.12-impl`) — the CLI's default database.
- On revisit no `python.exe` processes remained, so `kb.db` was free and the merge proceeded `okfgraph.db` → `kb.db` (one direction; `kb.db` was the superset except for the 8 new thoughts, ID sets disjoint Oct-6 vs Oct-7, no collisions possible).

Final state per report (verified working tree):

1. `kb.db` holds all 8 Oct-7 thoughts under correct IDs + amended decision register (DR-9/10/11/16/25/27/28/30/32 closed or noted, DR-33 re-baselined, DR-34 added, baseline `ac123e0`) + reunification thought (`thoughts/coderadar-v0.12-impl/20261007140609_c7b9c8`).
2. Live `okfgraph.db` deleted; byte-identical backup at `D:\User\Documents\Python\old_backups\okfgraph-split-20261007.db`.
3. New `okfgraph.toml` pins `db_path = "kb.db"` — the CLI now defaults to the united graph. Currently **untracked** (`?? okfgraph.toml`); `uv.lock` shows deleted (`D uv.lock`). Disposition is a Phase 0 decision (see §5).
4. Temp artifacts removed (`merge-tmp/`, `/tmp/okf-merge*`, amend script); project dir holds `kb.db`, `kb.db.wal`, `okfgraph.toml`, `.venv/`, sources.
5. The 8 flat-ID dupes auto-purge ~2026-10-08 14:00 UTC; nothing to do.

## 1. Issue A — positional `import` mints flat IDs (data landed in the wrong place)

**Severity: High. Silent misplacement + success message.**

Symptom: `okf import /tmp/okf-merge/thoughts/coderadar-v0.12-impl/*.md` reported `[OK] imported: <name>` for all 8 files and grew `kb.db` by ~2 MB, but topic traverse still showed only the original 7 thoughts.

Root cause (traced, not guessed):

- `cli.py:_import()` — positional `FILES` calls `router.import_file(f)` per file (`ops/admin.py:114` → `import_mgr.import_from_okf`).
- ID derivation is `components/import_.py:parse_source_file:119-170`. When `file_path.is_relative_to(root)` fails, `rel_path=None` → fallback `concept_id = file_path.stem` (lines 146-148; comment: "so the import doesn't crash with a relative_to ValueError").
- With `bundle_root="."` (repo root), files under `/tmp/...` can never satisfy `is_relative_to` — all 8 fell back to bare stems (`20261007084024_0004c1`). Content including frontmatter `topic:` intact (verified with `okf read`); only IDs/placement wrong. Invisible to directory traversal and topic-scoped hierarchy queries.

This fallback is **correct for a single stray file** (GUI writes one `.md` next to its source — no meaningful relative path) but **wrong as the silent default for an explicit multi-file list with a shared directory**.

Report's recovery was correct: soft-delete the 8 flats (recoverable 24 h) + re-import through a bundle directory so relative paths were preserved. Verified 15/15 + 1 follow-up (16 total), temp bundle deleted.

### Fix direction (0.10.1)

Do both, in this order:

- **A1 — shared-parent-relative IDs:** when all `FILES` share a non-trivial common parent, derive IDs relative to it (or relative to CWD) instead of bare stems. Single outsider keeps the bare-stem fallback.
- **A2 — warn whenever the fallback fires:** `WARNING hierarchy-dropped: <stem> from <abs path> is outside bundle_root <root>` and print both file path and minted ID. A1 without A2 still leaves the single-file case silent; A2 without A1 still forces the temp-bundle workaround for the common case.

Open lookup for Phase 0: what `root` does `import_from_okf` pass? If it unconditionally passes `self.bundle_root`, even bundle-adjacent flows can't help positional imports — confirm before designing A1's relative base.

No data migration in 0.10.1 — forward behavior + warning only.

## 2. Issue B — relative `--bundle-path` crashes + dot-dir invisibility

Two separate bugs, one section. Split them.

### B1 — relative `--bundle-path` crashes with `ValueError` (okf bug). Severity: High (crash).

Symptom: `okf import --all --bundle-path merge-tmp` (and the `/tmp` variant) crashed:

```
ValueError: 'merge-tmp\\thoughts\\coderadar-v0.12-impl' is not in the subpath of 'D:\\User\\Documents\\Python\\OKFgraph'
[ERROR] INTERNAL: ... (crash log: C:\Users\Main\AppData\Local\okfgraph\crash\okf-20261007-140036-880560-31116.log)
```

Passing the bundle path absolute with forward slashes (`--bundle-path D:/.../merge-tmp`) worked and minted correct hierarchical IDs (`thoughts/coderadar-v0.12-impl/...`).

Analysis:

- The string `is not in the subpath of` does not appear in `okfgraph/` — it is Python 3.13 `pathlib.relative_to` wording. Some call does `Path(user_arg).relative_to(resolved_root)` without normalizing first, or mixes CWD-relative vs resolved paths.
- Candidates: `_alias_for_root` / `resolve_alias_for_path` (`import_.py:580`), `_import_bundle_inner:659,794,801`, `_require_dir` (`ops/admin.py:100-104`), router bundle-root setup. Absolute-with-forward-slashes working while bare-relative crashes points to a missing `.resolve()` / separator mismatch on Windows (`\\` vs `/`) — same guess as the reporter; Phase 0 confirms the exact line from the crash log.
- `cli.py:483` validates `--bundle-path` only with `--all`, so the crash path is `import --all --bundle-path <relative>`. Crash reporting itself worked — keep that log for repro.

Fix: normalize **both** sides (`Path(bundle_path).resolve()`, `root.resolve()`) before any `relative_to` / `is_relative_to`; never let a raw user path reach `relative_to` unguarded — convert residual mismatch to `OKFError(BAD_VALUE)` with a remedy ("use an absolute path or check CWD").

### B2 — dot-dir silently skipped. Severity: Medium (silent, not crash).

Observation: hidden directory `.merge-tmp` invisible to the bundle scanner — `okf lint .merge-tmp` reported `0 file(s)`; renaming to `merge-tmp` showed the real 8 files.

Analysis: by design — `in_skipped_dir` (`import_.py:53-57`) skips any component starting with `.` plus `SKIP_DIR_NAMES`, shared by import/detach/diff/lint/delta so all agree on the file set. The bug is **diagnostic silence**: `0 file(s)` with no "skipped dot-dir" line.

Fix is docs + one log line, not a behavior change: `lint` (and `--all` with 0 files) reports `skipped <n> file(s) in hidden/tool dirs` when the walk was non-empty but fully filtered. Keep skipping `.git/.venv/.obsidian` — just say so.

## 3. Minor observations (evaluated, mostly docs/small hardening)

| # | Claim | Verdict + 0.10.1 action |
|---|---|---|
| stale `.lock` files | `kb.db.lock` / `okfgraph.db.lock` hand-removed; CLI doesn't always clean up; leftover indistinguishable from live hold | Real papercut. `InterProcessLock` file (`router.py:220`) + `close()` release (`543-548`) — a crash/kill leaves the file; next run can't tell live-hold from leftover. **Do not auto-delete** (breaks mutual exclusion). Action: `doctor` reports "stale `.lock`, no live holder" as info (PID check where possible) + one docs line ("check processes first"). |
| `OKFGRAPH_DB_PATH` works | traverse/search/read/delete/import/ingest OK; earlier `DB_LOCKED` was genuinely the MCP servers | Confirms diagnosis. No action; keep as regression note. |
| `deleted-purge` time-gated (0 purged) | 24 h window, "no force flag" | By design (`components/purge.py`, `cli.py:1270-1271` — `--older-than` override exists, so `--older-than 0` already forces). Report is slightly inaccurate. Action: document it; consider an explicit `--force` alias only if it's 2 lines + a test. |
| `doctor` 10/100, all pre-existing broken wikilinks | README → `docs/performance-roadmap`, `dogfood-review-2026-09` → `road_to_v0.9.0`; merge added none (thought bodies carry no out-links; 2 ambiguous-wikilink import warnings resolved to nothing, same as before) | Credible. Deliberately left unrepaired — correct, `doctor --fix` out of scope. Follow-up is editorial (import missing targets vs prune dead links, one decision per link) — **separate task, not 0.10.1**. |
| no `okfgraph.toml` existed → implicit `okfgraph.db` default allowed the silent fork | Fixed by the new pin | Root-cause fix for the whole incident class. Action (Phase 0 decision): commit a tracked `okfgraph.toml.example` + gitignore the local `okfgraph.toml`, or track it if repo-local `kb.db` is canonical. Don't leave it untracked forever. |

## 4. Cross-cutting notes

- `kb.db` (91 MB + WAL) is live working data and untracked. All repro on **scratch DBs**; never run Phase 0-2 scenarios against `kb.db`.
- Full serial suite stays the release gate (Windows `onnx_data` locking forbids sharded runs — prior history).
- Python return shapes for `import_file` / `import_bundle` stay identical; only ID *values* change for shared-parent lists (A1). Flag as behavior fix in CHANGELOG.
- Keep the crash-log pipeline as-is; it worked.

## 5. Work plan

### Phase 0 — repro + pinning (0.5 day)

1. Decide `okfgraph.toml` + `uv.lock` disposition. Proposal: restore `uv.lock` (`git checkout -- uv.lock`); track `okfgraph.toml.example`, gitignore local `okfgraph.toml` — unless repo-local `kb.db` is canonical, in which case track it. Needs explicit call (environment-specific either way).
2. Repro scripts on temp dirs only:
   - A: `okf import <tmpdir>/thoughts/<topic>/*.md` → assert current flat-ID behavior, capture warning absence.
   - B1: `okf import --all --bundle-path <relative>` (forms: `merge-tmp`, `merge-tmp/` forward slash, backslash variant, absolute) → capture `ValueError` + crash-log ref.
   - B2: `okf lint <dot-dir>` → capture `0 file(s)` silence.
3. Read crash log `okf-20261007-140036-*.log`; pin the exact `relative_to` line for B1.
4. Confirm what `root` `import_from_okf` passes (feeds A1 design).

Acceptance: three repros captured on scratch DBs; B1 line identified; toml/lock decision recorded.

### Phase 1 — Issue A: positional import hierarchy (1 day)

- Implement A1 (shared-parent-relative IDs for multi-file lists) + A2 (fallback warning with file → minted ID).
- Update `import FILES --help` (hierarchy rule, 2 lines).
- Tests: new `tests/test_import_positional.py` — (a) shared-parent list mints `thoughts/<topic>/<stem>`; (b) lone outsider warns + mints stem (no crash); (c) `--bundle-path` bulk path unchanged.
- Acceptance: report's 8-file scenario mints correct IDs with no temp-bundle workaround; single-file outsider still imports.

### Phase 2 — Issue B: bundle-path + lint diagnostics (1 day)

- B1: resolve/normalize both sides before `relative_to`; residual mismatch → `OKFError BAD_VALUE` with remedy. Regression tests: relative / absolute / `/` / `\\` on Windows + POSIX.
- B2: shared "skipped in hidden/tool dirs" count in `lint` + `import --all` zero-file path (one helper; walk sites already share `in_skipped_dir`).
- Acceptance: relative `merge-tmp` form imports 8/8 with hierarchical IDs; `.merge-tmp` lint explains why it's empty.

### Phase 3 — small hardening (0.5 day)

- Lock: `doctor` stale-lock info + docs line. No auto-delete.
- Purge: document `--older-than 0` as the force path; `--force` alias only if trivial.
- Default-DB: document `okfgraph.toml` pin + `OKFGRAPH_DB_PATH` precedence; consider a startup/`doctor` hint when default `okfgraph.db` would be implicitly created alongside an existing `kb.db` (at minimum the docs paragraph).

### Phase 4 — verify + release (0.5 day)

- Full serial suite, `git diff --check`, `test_pages`/`test_packaging` only if pages touched (not planned).
- CHANGELOG 0.10.1 entry (import hierarchy warning, relative bundle-path fix, lint skip diagnostics, lock/purge/default-DB docs), version bump `pyproject.toml` (`Cargo.toml`/`Cargo.lock` only if touched — not expected), tag `v0.10.1`, merge to `main`. Tag is last, needs go-ahead.

## 6. Test matrix

| Area | New test | Must still pass |
|---|---|---|
| positional import | `tests/test_import_positional.py` (3 cases above) | existing import/router/lint suites |
| bundle-path | relative/absolute/sep regression (Windows + POSIX) | `test_produce`, `test_router`, multi-root/detach |
| lint skips | dot-dir diagnostic assertion | lint/diff/delta agreement tests |
| purge/lock | `--older-than 0` doc-test or alias test; stale-lock doctor info test | purge recovery-window tests unchanged |

## 7. Risks / unknowns

- B1's exact line unconfirmed until the crash log is read — could sit in router setup rather than import; Phase 0 covers it.
- A1 changes minted IDs for shared-parent lists — must not break MCP/Python consumers relying on old flat stems; CHANGELOG behavior-fix note required.
- Link-rot editorial (doctor 10/100) explicitly excluded — needs per-link decisions, separate task.
