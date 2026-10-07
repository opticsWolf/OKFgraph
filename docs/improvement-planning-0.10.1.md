# OKFgraph 0.10.1 — improvement plan (graph-reunification follow-ups)

**Status:** implemented on `dev_0.10.1` (commits `4e0ae60` + `15ed2bd`, pushed). Full serial suite green (876 passed, 6 skipped, 0 failed). This document stands as the historical record: why the work exists (field report + incident), what was decided, and what was built.
**Rev 3:** Amendment B appended 2026-10-07 — resolves the positional-import root ambiguity, corrects Phase 2's operative B1 contract for outside-root trees, and restores upstream issue filing as an explicit follow-up (see §9).
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

- **A1 — explicit identity root for positional `FILES`:** permit `--bundle-path DIR` with positional files as an identity base (not a recursive scan); resolve the root and file paths, and derive IDs relative to it only when every supplied file is beneath it. Without an explicit `--bundle-path`, keep configured-root resolution; do not infer a base from the files' common parent (for a globbed topic directory that is usually the topic directory itself and would still drop `thoughts/<topic>`). Files outside every configured root retain the bare-stem fallback, but are no longer silent under A2. A lone stray file remains supported.
- **A2 — warn whenever the fallback fires:** `WARNING hierarchy-dropped: <stem> from <abs path> is outside bundle_root <root>` and print both file path and minted ID. A1 without A2 leaves unrooted files silent; A2 without A1 keeps the behavior visible but still requires the caller to supply an identity root to preserve hierarchy.

Resolved lookup: `import_from_okf` consults configured roots, then falls back to `self.bundle_root` (Amendment A §8.5). The remaining A1 decision is CLI contract: positional imports need an explicit identity root for external paths; `--bundle-path` is already parsed but currently rejected unless `--all` (see Amendment B §9).

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
- Candidates: `_alias_for_root` / `resolve_alias_for_path` (`import_.py:580`), `_import_bundle_inner:659,794,801`, `_require_dir` (`ops/admin.py:100-104`), router bundle-root setup. Absolute-with-forward-slashes working while bare-relative crashes points to a missing `.resolve()` / separator mismatch on Windows (`\\` vs `/`) — same guess as the reporter. **Confirmed — see Amendment A §8.2:** crash site is `delta.py:158`; the mechanism is mixed relative/absolute anchors, not separators.
- `cli.py:483` validates `--bundle-path` only with `--all`, so the crash path is `import --all --bundle-path <relative>`. Crash reporting itself worked — keep that log for repro.

Fix (operative — see Amendment B §9.2): resolve `bundle_path` once, then use the same resolved walked tree for enumeration, delta relative-path calculation, and hash lookup. Do not compare valid outside-root paths against the configured root; outside-root trees are supported. Convert only malformed/unresolvable input to `OKFError(BAD_VALUE)` with a remedy. The exact DirHash key/namespace rule is a Phase 0 decision; no schema migration.

### B2 — dot-dir silently skipped. Severity: Medium (silent, not crash).

Observation: hidden directory `.merge-tmp` invisible to the bundle scanner — `okf lint .merge-tmp` reported `0 file(s)`; renaming to `merge-tmp` showed the real 8 files.

Analysis: by design — `in_skipped_dir` (`import_.py:53-57`) skips any component starting with `.` plus `SKIP_DIR_NAMES`, shared by import/detach/diff/lint/delta so all agree on the file set. The bug is **diagnostic silence**: `0 file(s)` with no "skipped dot-dir" line.

Fix is diagnostics only, not a walk behavior change: `lint` (and `--all` with 0 files) reports `skipped <n> file(s) in hidden/tool dirs` when the walk was non-empty but fully filtered. Keep skipping `.git/.venv/.obsidian`; for lint, do not print "lint-clean (safe to import)" when every discovered file was filtered out (Amendment A §8.4).

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
- Python return shapes for `import_file` / `import_bundle` stay identical. Positional imports only mint hierarchy relative to an explicit identity root; adding an optional internal/keyword root parameter is acceptable if needed, but keep the public return shapes unchanged. Flag changed ID values and fallback warnings in CHANGELOG.
- Keep the crash-log pipeline as-is; it worked.

## 5. Work plan

### Phase 0 — repro + pinning (0.5 day)

1. Decide `okfgraph.toml` + `uv.lock` disposition. Proposal: restore `uv.lock` (`git checkout -- uv.lock`); track `okfgraph.toml.example`, gitignore local `okfgraph.toml` — unless repo-local `kb.db` is canonical, in which case track it. Needs explicit call (environment-specific either way).
2. Repro scripts on temp dirs only:
   - A: positional files outside configured roots with and without explicit `--bundle-path`; capture flat IDs and warning behavior, then assert the rooted form can express the intended hierarchy.
   - B1: `okf import --all --bundle-path <relative>` (forward/backslash forms), absolute-inside, and absolute-outside; capture result + crash-log ref.
   - B2: `okf lint <dot-dir>`; assert the fully skipped walk does not claim "lint-clean (safe to import)".
3. ~~Read crash log `okf-20261007-140036-*.log`; pin the exact `relative_to` line for B1.~~ **Done — Amendment A §8.2.**
4. ~~Confirm what `root` `import_from_okf` passes (feeds A1 design).~~ **Done — Amendment A §8.5.**
5. Agree A1 CLI/API contract: `--bundle-path` as identity root for positional `FILES`, validation when any file is outside it, and whether a narrowly-scoped optional root parameter is needed internally.
6. Follow report §6: check for existing upstream Issues A/B; open or link them with a sanitized crash trace (remove local usernames/paths).

Acceptance: scratch-DB repros captured (Amendment A §8.6); A1 identity-root contract and B1 DirHash ledger namespace decision recorded; upstream issue numbers/links or a documented duplicate decision recorded; toml/lock decision recorded.

### Phase 1 — Issue A: positional import hierarchy (1 day)

- Implement A1/A2: allow `--bundle-path DIR` with positional `FILES` as their identity root (not a directory walk); derive IDs relative to it only when all files are contained. Without an explicit root, preserve configured-root behavior and warn for each hierarchy-dropped fallback. Do not infer identity root from the common parent.
- Update `import FILES --help` to distinguish positional identity-root use from `--all` tree scanning.
- Tests: new `tests/test_import_positional.py` — (a) rooted multi-file list mints `thoughts/<topic>/<stem>`; (b) outside-root file with no explicit base keeps stem and warns; (c) file outside a supplied identity root is rejected before importing any file; (d) lone outsider still imports; (e) `--all --bundle-path` bulk behavior unchanged.
- Keep Python return shapes unchanged; if an optional root is threaded internally, test default single-file behavior.
- Acceptance: report's 8-file scenario mints correct IDs when its source root is explicit; no silent hierarchy loss; single-file outsider still imports.

### Phase 2 — Issue B: bundle-path + lint diagnostics (1 day)

- B1: resolve `bundle_path` once at the boundary, then carry the resolved **walk root** through source enumeration, delta directory grouping, and directory hashing. `delta.py` must calculate relative directory keys and hash paths against that same walked tree, not blindly against the configured `self.bundle_root`; otherwise valid absolute trees outside the configured root still fail. Preserve a distinct/qualified DirHash namespace for pinned trees so tree-relative keys cannot collide with the default-root ledger. Decide the namespace in Phase 0; no schema migration. Convert only genuinely invalid/unresolvable inputs to `OKFError(BAD_VALUE)` with a remedy — an existing external tree is valid and must not be rejected merely for being outside the configured root. Regression tests: relative/absolute paths, inside/outside configured root, `/` and `\\` separators on Windows + POSIX.
- B2: shared "skipped in hidden/tool dirs" count in `lint` + `import --all` zero-file path (one helper; walk sites already share `in_skipped_dir`). A fully-filtered non-empty lint walk must not claim "lint-clean (safe to import)".
- Acceptance: relative and absolute-outside forms import the same hierarchical IDs with correct delta bookkeeping; repeated import is stable and default-root DirHash state is not overwritten/collided. `.merge-tmp` lint explains why it is empty without an unsafe-to-import success claim.

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
| positional import | `tests/test_import_positional.py`: explicit identity root, missing root fallback warning, outside-root rejection before partial import | existing import/router/lint suites; Python return shapes |
| bundle-path | relative/absolute × inside/outside root × separator variants; repeated-import and DirHash-namespace checks (Windows + POSIX) | `test_produce`, `test_router`, multi-root/detach |
| lint skips | dot-dir skipped-count + no "lint-clean (safe to import)" on fully-filtered walk | lint/diff/delta agreement tests |
| purge/lock | `--older-than 0` doc-test or alias test; stale-lock doctor info test | purge recovery-window tests unchanged |

## 7. Risks / unknowns

- B1's root cause is confirmed, but pinned-tree DirHash namespace/key semantics remain a design risk; Phase 0 must settle collision behavior before implementation.
- A1 must not guess an identity root from a file list. Explicit `--bundle-path` use for positional `FILES` changes CLI validation/help and may require an optional internal root parameter; preserve Python return shapes and single-file default behavior.
- Link-rot editorial (doctor 10/100) explicitly excluded — needs per-link decisions, separate task.

## 8. Amendment A (2026-10-07, rev 2) — full report read + crash log + verified repro

Written after re-reading `okf-graph-issues-20261007.md` in full (the drafting read was truncated) and running scratch-DB repros. Supersedes the listed base-plan passages where they conflict; base text above is kept for history.

### 8.1 Basis

Full re-read found **no contradictions** with the base plan — every symptom, quote, and observation in §1-§3 above matches the complete report. One base-plan gap: the report's §6 "File Issues A/B upstream with the crash log above" had no matching action. Resolution: with the root cause now confirmed (8.2), no external upstream filing is needed — the issues are tracked here and land as 0.10.1 fixes + CHANGELOG entries. If either is ever filed publicly, redact local usernames/paths from the crash log first.

### 8.2 B1 root cause — CONFIRMED (replaces §2 B1 analysis)

Crash log `C:\Users\Main\AppData\Local\okfgraph\crash\okf-20261007-140036-880560-31116.log` (verified on disk, 38 lines) pins the exact site:

```
File "...\okfgraph\components\delta.py", line 158, in _changed_directories
    parent = str(fp.parent.relative_to(self.bundle_root))
ValueError: 'merge-tmp\\thoughts\\coderadar-v0.12-impl' is not in the subpath of 'D:\\User\\Documents\\Python\\OKFgraph'
```

Call chain: `cli.py:_import` → `ops/admin.py:import_bundle` → `import_.py:import_bundle:1263` → `_import_bundle_inner:679` → `delta.py:_changed_directories:158`.

Mechanism — **mixed anchors, not separators**:

- `fp` comes from `root.rglob("*")` where `root = Path(bundle_path)` is the **raw user argument** (`_import_bundle_inner:602` does not resolve it). With a relative `--bundle-path`, every `fp` is CWD-relative.
- `self.bundle_root` is the **resolved configured root** (`D:\...\OKFgraph`, from `bundle_root = "."`).
- `relative_to` between a relative operand and an absolute anchor raises `ValueError` **always** — regardless of separator style or whether the tree physically lives under the configured root. So relative `--bundle-path` can never work on 0.10.0; it is not a Windows `PurePath` bug.
- The absolute-with-forward-slashes form worked only because `pathlib` normalizes `D:/...` to the same anchor, and the tree happened to live inside the configured root, so the subpath check passed.

### 8.3 NEW crash class — outside-root trees (extends §2 B1 scope)

Scratch-DB probe P3: `--bundle-path <absolute path outside the configured root>` crashes with the **same** `ValueError`, even though the user path was fully absolute:

```
ValueError: 'C:\\Users\\Main\\AppData\\Local\\Temp\\okf-amend-probe-20822\\bundle\\thoughts\\topic'
  is not in the subpath of 'D:\\User\\Documents\\Python\\OKFgraph'
```

This matters because outside-the-root temp dirs are the natural place for scratch bundles — exactly what the reporter's `/tmp/okf-merge` was. **Consequence for the fix:** the base plan's "normalize both sides before `relative_to`" is necessary but **not sufficient** — resolving the user path would still crash on outside-root trees (`fp` under `C:\\...`, anchor `D:\\...\\OKFgraph`). The real fix has two parts:

1. Resolve the user path at import entry (`root = Path(bundle_path).resolve()`), and
2. Align the delta detector with the tree actually walked: `delta.py:158` must key directories against the walked tree (a per-import walk root), not unconditionally against `self.bundle_root`. Related: `_compute_directory_hash_with_files(self.bundle_root / dir_rel)` (~line 166) would likewise probe the wrong path for outside trees.

**Design decision needed (Phase 0):** DirHash ledger keying for pinned trees. Today's absolute-inside success already writes root-relative keys (`scratch_probe_bundle\\thoughts`) into the `""`-alias ledger — precedent exists, but an explicit choice between (a) tree-relative keys, (b) root-relative keys (status quo), or (c) a pinned-tree marker must be made before coding, with collision analysis against the default tree's ledger (`thoughts` at repo root vs a pinned tree's internal `thoughts/`).

### 8.4 B2 sharpened — lint is worse than silent

Scratch-DB probe: `okf lint .scratch_lint_probe` (one real file inside a dot-dir) prints:

```
0 file(s): 0 error(s), 0 warning(s)
Bundle is lint-clean (safe to import).
```

Not just silence — the conclusion line **actively endorses a tree the scanner never looked into**. The base-plan fix (a `skipped <n> file(s) in hidden/tool dirs` count) stands, but the acceptance test must also cover the conclusion line: a fully-filtered non-empty walk must not print "lint-clean (safe to import)".

### 8.5 Phase 0 item 4 resolved — `import_from_okf` root lookup

`import_.py:1717-1718`:

```python
_alias = resolve_alias_for_path(file_path, self.roots) or ""
_root = self.roots[_alias] if _alias else self.bundle_root
```

So positional single-file imports **do** consult configured roots, but files outside every root fall back to `self.bundle_root` → `parse_source_file` gets a root the file is not under → bare-stem fallback. Consequence for A1: hierarchical minting for positional `FILES` lists needs an **explicit base** — either a new optional parameter threaded through `import_file`/`parse_source_file` (Python API surface addition — default single-file behavior unchanged) or computed CLI-side before calling `import_file`. The CLI-layer option keeps the Python API untouched; prefer it unless it forks logic.

### 8.6 Repro results (scratch DBs, `OKFGRAPH_DB_PATH` pointed at temp — `kb.db` untouched)

| Probe | `--bundle-path` form | Result on 0.10.0 |
|---|---|---|
| P1 | absolute, inside root, `/` separators | imports 1 concept, ID `thoughts/topic/probe_a` (matches report) |
| P2 | relative via symlink from repo cwd | `ValueError` (incident repro) |
| P4 | plain relative, inside root, cwd=repo | `ValueError` — relative **never** works |
| P3 | absolute, **outside** root | `ValueError` — new crash class (§8.3) |
| B2 | `okf lint <dot-dir>` | `0 file(s)` + "lint-clean (safe to import)" (§8.4) |

Repro script: `scratchpad/amend_probe.sh` (untracked). Fresh crash logs generated: `okf-20261007-191452-471299-25308.log`, `okf-20261007-191453-331700-36568.log`, `okf-20261007-191459-827784-2516.log` (same crash dir as the incident log). All probe dirs and scratch DBs removed; working tree back to `D uv.lock` + `?? okfgraph.toml` + untracked `scratchpad/`.

### 8.7 Plan deltas caused by this amendment

- **§5 Phase 2 (B1 fix)**: replace "normalize both sides" with the two-part fix of §8.3 (resolve at entry + walk-root alignment) plus the ledger-keying decision.
- **§5 Phase 2 (B2)**: acceptance adds the conclusion-line guard (§8.4).
- **§5 Phase 0**: items 3-4 done; item 1 (toml/lock) still open.
- **§6 test matrix**: add outside-root bundle-path row and a "no 'safe to import' on fully-filtered walk" assertion.
- **§7 risks**: first risk resolved; replaced by the ledger-keying risk (§8.3).
- **§1 A1**: base must be explicit (§8.5); CLI-layer preferred.

## 9. Amendment B (2026-10-07) — make the plan executable

This amendment follows a second full read of the field report and the current plan. It **supersedes conflicting A1 recommendations in §1 and §8.5, the upstream-filing resolution in §8.1, and the prospective-only Phase 2 / test-matrix notes in §8.7**. The revised work-plan sections above are operative; this section records why they changed.

### 9.1 A1 needs an explicit identity root, not an inferred common parent

The base plan's phrase "shared-parent-relative IDs" was underspecified and can produce the wrong hierarchy: for `.../thoughts/<topic>/*.md`, the files' common parent is `<topic>`, so IDs relative to that parent are only `<stem>` — exactly the flattening the fix is meant to prevent. `import_from_okf` already uses configured roots and falls back to `self.bundle_root`, but positional files outside those roots have no meaningful hierarchy base.

The operative design is therefore: allow `--bundle-path DIR` with positional `FILES` as an **explicit identity root**, validating that every file is under it before importing any file; then mint IDs relative to that root. It does not recursively scan the directory in this mode. Without this explicit root, retain existing configured-root behavior and emit A2's hierarchy-dropped warning when a file falls back to its stem. This avoids guessing from an ambiguous file list, provides a direct path for the report's `/tmp/okf-merge` case, and keeps lone GUI-written files supported. The current CLI explicitly rejects `--bundle-path` unless `--all` (`cli.py:_import`), so Phase 1 must intentionally relax that validation and clarify help text.

### 9.2 B1 must treat an outside-root tree as valid input

Amendment A §8.3 proved that a valid absolute tree outside the configured root crashes. The earlier phrase "convert residual mismatch to `BAD_VALUE`" must not be interpreted as rejecting such trees: **outside the configured root is not itself an error**. Normalize once, then use the same resolved walked tree for enumeration, relative directory keys, and hash lookup. Keep pinned-tree delta state distinct or qualified so it cannot collide with the default-root ledger; Phase 0 must choose and document the key/namespace rule. Only malformed or unresolvable paths should produce a user-facing `BAD_VALUE`.

### 9.3 Restore the report's upstream issue follow-up

Amendment A §8.1 prematurely concluded that no upstream filing was needed merely because fixes were planned in this repository. The field report explicitly lists "File Issues A/B upstream with the crash log above" as an open follow-up. Phase 0 now requires checking for duplicates and either opening/linking Issues A and B or recording why an existing issue covers each one. If attaching a crash trace, sanitize local usernames and paths first. This is issue tracking, not a release gate for the fixes themselves.

### 9.4 Coverage and boundaries

The revised Phase 0/1/2 steps and §6 test matrix now explicitly cover: rooted positional imports, no-root fallback warnings, preflight rejection of files outside an explicitly supplied identity root, relative and absolute-outside bundle paths, repeated import/delta stability, and the lint conclusion for a fully skipped dot-dir. These remain scratch-DB tests; there is no live-graph repro or schema migration in scope.

### 9.5 Decision log (implementation session, 2026-10-07)

- **Upstream issue filing (§9.3): REJECTED by maintainer.** No Issues A/B will be filed; the private field report plus this plan and the CHANGELOG entry stand as the record. Phase 0 item 6 is closed as declined, not done.
- **bobine:** installed `bobine==0.6.0` (built from `D:/User/Documents/Rust/bobine`) into `.venv`; the 13 PDF-ingest failures were a missing optional dependency. Full serial suite green: 876 passed, 6 skipped, 0 failed.
- **`uv.lock`:** restored (was deleted in the working tree, breaking CI `--frozen`) and re-locked for 0.10.1 — one-line diff (`okfgraph 0.10.0 → 0.10.1`), `uv lock --check` passes.
- **`okfgraph.toml`:** live repo-local pin (`db_path = "kb.db"`) stays untracked by design; tracked as `okfgraph.toml.example` + `.gitignore` entry. Known quirk retained: CWD-first lookup can shadow test fixtures when the suite runs from the repo root (workaround: move the live TOML aside for the run, restore after).
