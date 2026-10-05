# okfgraph 0.10.0 — release plan

**Status (2026-10-05):** surface unification is implemented and pushed
(`23e111e` on `dev_roadtov010`, 14 commits ahead of `main`). The field
report from the Macrame_docs graph (okfgraph 0.10.0, ladybug 0.21.2,
onnxruntime-gpu 1.29.0, 73 concepts / 28 101 chunks) found eight issues.
Five are fixed in the working tree, not yet committed. This plan lists
everything between here and a published `v0.10.0`.

Pushing a `v*` tag runs `.github/workflows/release.yml`, which publishes
to PyPI. That step cannot be undone (PyPI never accepts the same version
twice), so the tag is the last step and needs an explicit go-ahead.

---

## 0. Where things stand

| # | Field-report issue | State |
|---|---|---|
| 1 | Chunk search segfault (exit 139) | **Fixed, uncommitted.** ladybug 0.21.2 crashes deterministically on an unbounded `QUERY_FTS_INDEX` whose terms match many rows. Both FTS stages now pass `top := limit*3`. Verified on a copy of `macrame_docs.db`. |
| 2 | Parallel CLI processes hit the DB lock | **Fixed (error only), uncommitted.** New state code `DB_LOCKED` with a remedy, replacing the `INTERNAL` crash. Concurrency itself is a ladybug limit (one process per db). |
| 3 | `okf model-info` → `ModuleNotFoundError: huggingface_hub` | **Open — blocker.** See B1. |
| 4 | Misleading "no ONNX Runtime installed" | **Fixed, uncommitted.** Installed-but-unimportable is reported separately, with the import error. Typed `NO_ORT_RUNTIME`. |
| 5 | Phantom `image_count` | **Fixed, uncommitted.** It was `len()` of a 7-key stats dict; no phantom assets were ever stored. Relative image paths now resolve from the file's folder. |
| 6 | No way to delete a wrongly imported concept | **Open — decision D2.** |
| 7 | Log line on stdout breaks `--json` | **Not a bug** (all handlers write to stderr; the harness merged `2>&1`). Optional hardening: D3. |
| 8 | md ingest rewrites the user's source | **Fixed, uncommitted.** md ingest is report-only (`fixable_count`). |

Test state: the full suite on `23e111e` gave 798 passed, 4 failed (all
`model-info`, issue 3), 12 skipped. Tests for every file touched since
then pass; the full suite has **not** been rerun on the working tree.

---

## Phase 1 — Land the field fixes

1. Run the full suite on the working tree (~19 min, CPU):
   `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider -rfE`.
   Expect only the 4 `model-info` failures.
2. Commit on `dev_roadtov010`: "0.10.0: field-report fixes (FTS segfault,
   DB_LOCKED, ORT hint, image_count, md ingest read-only)". Push.

Exit: a commit containing fixes 1, 2, 4, 5 and 8, with the suite green
apart from issue 3.

---

## Phase 2 — Release blockers

### B1. `model-info` without `huggingface_hub` (issue 3)

`EmbeddingEngine.model_info()` (`okfgraph/components/embedding.py`)
imports an undeclared package. Even with that package installed, it would
inspect the wrong repo: embroider maps the default id at FP16 to a mirror
repo and fetches `onnx/model.onnx` plus a sidecar and `tokenizer.json`.

**Recommended: B1-a, the embroider route.** It is fully specified in
`embroider/docs/cache-info-plan.md`.
1. embroider: add `embroider.cache_info(model_id, revision, cache_dir,
   precision)`. It resolves files the same way `open()` does and looks
   them up offline in the hf-hub cache. Add Rust and Python tests. Release
   embroider **0.3.2** (crates.io + PyPI wheels via embroider's own release
   workflow).
2. okfgraph: `model_info()` calls `embroider.cache_info`. The output keeps
   `model_id, cache_dir, cached, snapshot_path, disk_usage_bytes` and adds
   `repo, files`. Raise the floor in `pyproject.toml` to
   `embroider>=0.3.2,<0.4`, then `uv lock`.
3. On an embroider without `cache_info`, raise typed `NO_ORT_RUNTIME`-style
   guidance rather than crash. That can't happen with the raised floor,
   but it guards editable or dev installs.
4. Update the 4 failing tests (`tests/test_router.py::TestCacheManagement`,
   `tests/test_shell.py::TestHousekeeping::test_model_info`) to the new
   keys, using a temp-dir fake cache so they run offline.

**Fallback: B1-b, if the embroider release slips.** Ship 0.10.0 with
`model-info` refusing as a typed error ("needs embroider ≥ 0.3.2"). Skip
the 4 tests behind `importorskip`-style guards. Note it in the CHANGELOG
as a known issue. Do **not** add `huggingface_hub` as a dependency; it
answers for the wrong repo.

Order constraint: embroider 0.3.2 must be live on PyPI **before** the
okfgraph tag, because the release workflow doesn't build embroider.

### B2. Undeclared `huggingface_hub` in tests

`tests/test_parity.py` imports it inside the `baseline` fixture. Add
`pytest.importorskip("huggingface_hub")` (with `transformers`) at the top
of the module, so the parity test is opt-in rather than an error.

### B3. CI configuration

- `full.yml` syncs `--extra omni`, an extra that no longer exists
  (removed in 0.7.0). Check the last scheduled run: uv either rejects the
  unknown extra or silently skips it, and in both cases the run has no
  ORT runtime. Change it to `--extra dev --extra pdf --extra cpu`.
- `ci.yml` triggers only on `dev_rework`, `dev` and `main`, so **no CI has
  run on `dev_roadtov010`**. CI first runs on the merge PR to `main`.
  Optionally add the branch for this cycle.
- `ci.yml`'s model-free test list predates 0.10.0. Add the new model-free
  files `tests/test_envelope.py`, plus `tests/test_surface_contract.py`
  and `tests/test_settings*.py` if they load no model; check each with
  `ORT_DYLIB_PATH` unset and no network.
- `ci.yml` syncs `--extra dev` only (no ORT), which is what the
  "no ORT installed" hints cover. Keep it that way.

### B4. Docs consistency

- `docs/harness-integration.md:22` still says "the `omni` extra only for
  image embeddings". The vision route uses embroider (`JinaV5Vision`) plus
  ORT; remove the omni wording.
- Add `DB_LOCKED` wherever error codes are listed: README CLI errors
  section and `docs/architecture.md` (the error table, §5 dispatch).
  `surface-unification-plan.md` and the skills are already done.
- Concurrency note (README "CLI" and `harness-integration.md`): one
  process per database. When `okf-mcp` is running, query through it.
- README ingest table: md `lint_issues` now has `fixable_count`, and md
  ingest never writes the source.
- `okfgraph/components/search.py` `_index_rows` docstring: the "0.20.3"
  reference is historical. Keep it, but the 0.21.2 `top` rule is the
  operative one (already added).
- CHANGELOG: set the release date on `## [0.10.0]` when tagging.
  `model-info` goes under Fixed (B1-a) or Known issues (B1-b).

### B5. Audit other ladybug index calls

The segfault class is unbounded result sets in native multi-threaded
scoring. Check every index call site against a large graph:

| Call site | Bounded? |
|---|---|
| `search.py` chunk + concept `QUERY_FTS_INDEX` | yes (fixed) |
| `search.py` chunk + concept `QUERY_VECTOR_INDEX` | yes (`k = limit*3`) |
| `image_assets.py:427/439` `QUERY_VECTOR_INDEX` on `ImageAsset` | `k` given — confirm the caller caps it |
| PPR / `read_with_budget` / `traverse` | plain Cypher, no index call |

Add a small stress harness (`scripts/stress_search.py`, not in pytest). It
runs the CLI search matrix (targets × ranks × ~10 queries including
common words such as "index", "the system", "data") against a given db
and reports any non-zero exit. Run it against the Macrame copy.

---

## Phase 3 — Decisions on the remaining field items

### D2. Deleting a concept (issue 6)

Facts: `PurgeManager._soft_delete_concept` already exists. It is
recoverable through `deleted-recover` and purgeable through
`deleted-purge`, but no op exposes it. A file-backed concept that is
soft-deleted comes back on the next `import --all` while its source
exists.

Options:
- **D2-a (recommended for 0.10.0):** new op `delete(concept_id)` →
  `{concept_id, deleted: true, recoverable_until}`. CLI `okf delete ID`,
  Python `router.delete()`, **not** on MCP (destructive; agents get
  `ingest` only). It refuses with `BAD_VALUE` when the concept is
  file-backed and its source exists, and the remedy says to delete or
  move the file and run `import --all --prune-missing`. That keeps graph
  and bundle consistent with no new state. The surface goes from 23 to 24
  ops, so update the op table in the surface plan, the architecture doc,
  README and the CLI skill.
- **D2-b:** D2-a plus `--force` and an exclude list (`DeletedPath`
  already exists as a table) so import skips the path. More state, more
  docs. Defer to 0.11.
- **D2-c:** defer entirely. Document the workaround (delete the file, then
  `import --all --prune-missing`) in the ingest skill.

### D3. `--json` and log noise (issue 7)

Optional: under `--json`, default the console log level to WARNING unless
`-v` is given. That helps harnesses that merge streams. It costs nothing
and changes no data. Recommend yes, with one test.

### D4. Branch and tag placement

The release workflow publishes whatever commit the tag points at.
Recommend: merge `dev_roadtov010` → `main` through a PR (CI runs there,
B3), then tag `v0.10.0` on the merge commit on `main`. That matches the
earlier tags (`v0.9.0` is contained in `main`).

---

## Phase 4 — Validation

1. **Full suite**, local CPU venv (`--extra cpu --extra pdf --extra dev`):
   0 failures.
2. **GPU run** in the Macrame_docs venv (onnxruntime-gpu), with okfgraph
   reinstalled from this repo: `uv pip install -e
   D:/User/Documents/Python/OKFgraph` or a built wheel.
3. **Field acceptance on the Macrame graph** (a copy, never the live db):
   - stress harness (B5): every search returns, exit 0;
   - `okf ingest --kind md --md-path refs.bib`: `image_count` 0, file
     hash unchanged, `fixable_count`/`unfixable_count` reported;
   - two parallel `okf search` processes: the second exits 1 with
     `[ERROR] DB_LOCKED`;
   - `okf model-info`: reports `cached: true` with the FP16 mirror repo
     and its files (B1-a);
   - `okf doctor`: `ort_runtime` names onnxruntime-gpu, `cuda_usable=True`;
   - `okf search ... --json 2>/dev/null | python -m json.tool`: parses.
4. **Clean-install smoke test** (catches undeclared deps like issue 3):
   `uv build`, a fresh venv, `pip install dist/okfgraph-0.10.0-*.whl[cpu]`,
   then `okf --help`, `okf model-info`, `okf init` on a temp dir, `okf-mcp`
   boot with `--db-path` (expect the server to start and stay up).
5. **MCP smoke test:** restart the project's `okf-mcp` (`.mcp.json`) on the
   new code; run `search` (chunks + concepts), `read`, `traverse`, and
   `ingest` thoughts on `kb.db`.

---

## Phase 5 — Release

1. embroider 0.3.2 published (B1-a only). Confirm the wheels are on PyPI.
2. Open a PR `dev_roadtov010` → `main` with the CHANGELOG 0.10.0 section
   as the body. CI green.
3. Merge. Set the CHANGELOG date; bump nothing else (version is 0.10.0).
4. **With explicit approval:** `git tag v0.10.0` on `main`, then
   `git push origin v0.10.0`. `release.yml` checks tag == pyproject and
   publishes to PyPI.
5. Check that `pip install okfgraph[cpu]==0.10.0` works in a fresh venv;
   the Pages site (`pages.yml`) shows 0.10.0.
6. Fast-forward `dev` to `main`.

---

## Housekeeping (not release-gating)

- Delete `scratchpad/ort_providers_shared.locked.dll` and the Macrame db
  copy in the scratchpad.
- `build/lib/okfgraph/` is a stale, git-ignored build tree. Delete it so
  greps stop matching old code.
- Reinstall the Macrame_docs venv from the released wheel.
- The 0-byte `<db>.lock` beside a database is ladybug's own lock file. It
  is harmless when no process holds it; note this in the README FAQ.

## Order of work

Phase 1 → B2, B3, B4, B5 (okfgraph only, in parallel with B1's embroider
work) → B1 → D2/D3 per decision → Phase 4 → Phase 5.

Decisions needed from you: **B1-a vs B1-b**, **D2-a/b/c**, **D3 yes/no**,
**D4 confirmed**.
