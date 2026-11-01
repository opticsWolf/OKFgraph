# OKFgraph issue-solving implementation plan

**Date:** 2026-11-01 · **Baseline:** `main` at `4d0f352` (okfgraph 0.10.2 +
embroider `>=0.12,<0.13`) · **Status:** 0.10.3 code complete on
`dev_0.10.3` — WS1, WS2 (+ `retitle-thoughts`, Q3), WS3 step 1, WS4,
WS5.1 (dangling-seed fix), WS5.3, WS5.4. Still open: Macrame acceptance
re-runs (§2.3, §4.4), the release gate (§8), the raw-link column (0.11.0
schema), and the 0.11.0 rebase.

**Also folded in** (§10): `docs/reranker-integration-plan.md`,
`docs/search-options-analysis.md`, and the open-topic rows of
`docs/chunking-and-graph-retrieval.md` §13.

**Sources (field reports, all from real use on the Macrame and OKFgraph
graphs):**

| Short | File | Version observed | Situation |
|---|---|---|---|
| R07 | `okf-graph-issues-20261007.md` | 0.10.0 | reuniting a split `kb.db` / `okfgraph.db` |
| R29 | `okf-graph-issues-20261029.md` | 0.10.2 | clean first import of `Macrame/docs` (49 concepts, doctor 25/100 on 71 broken links) |
| R31 | `okf-graph-issues-20261031.md` | 0.10.x | a working day of ingest / search / delete on the Macrame graph |

Issue IDs below are prefixed with their report (`R29-A` = issue A of the
10-29 report) so every item traces back to its evidence.

---

## 1. Status of every reported item

Checked against `main` and against the unmerged `dev_0.11.0` draft
(`5d04e03`, one commit, 2026-10-16).

| ID | Issue | Status on `main` | In `dev_0.11.0` draft? | Plan |
|---|---|---|---|---|
| R07-A | positional `import` flattens IDs to bare stems | **fixed in 0.10.2** (A1/A2: `--bundle-path` is an identity root, `hierarchy-dropped` warning) | — | none |
| R07-B | relative `--bundle-path` crashes (`ValueError`) | **fixed in 0.10.2** (B1) | — | none |
| R07-B' | dot-dirs silently skipped by lint | **fixed in 0.10.2** (B2 skip diagnostics) | — | none |
| R07 §4 | `deleted-purge` has no "now" | **fixed in 0.10.2** (`--older-than 0` documented) | — | none |
| R07 §4 | no `okfgraph.toml`, default DB forks silently | **fixed in 0.10.2** (pin documented) | — | none |
| R07 §4 / R29-F | stale `<db>.lock` after clean runs (reported twice) | **open** — doctor detects it (`stale_lock`), nothing removes it | no | **WS4** |
| R29-A | path links not resolved relative to the source file (49 of 71) | **open** | **partly** — O1 reads source-relative first, root-relative fallback | **WS1** |
| R29-B | `%20` never decoded (17 of 71) | **open** | no | **WS1** |
| R29-C | `[[bench]]` etc. read as wikilinks inside code spans | **open** | no | **WS1** |
| R29-D | `MD_LINK_RE` over-captures past a non-`.md` link; `.txt`/`.markdown` targets invisible; unanchored `.replace(".md")` | **open** | no | **WS1** |
| R29-E | `../docs/…` round-trip links in Macrame's own docs | not okfgraph's | — | out of scope (fix in Macrame) |
| R29-G | delta is directory-granular: one edit re-embeds 22–38 of 49 files; log line misreports it | **open** | **yes** — O2 per-file change detection | **WS3** |
| R29-G' | rename without `--prune-missing` leaves a stale concept, silently | **open** | partly — O3 frontmatter identity (opt-in) keeps the concept across moves | **WS3** |
| R31-A | every thought of a topic gets the title `Thought: <topic>`; `--title` refused for thoughts | **open** | **no** — draft still writes `Thought: {topic}` | **WS2** |
| R31-B | minted thought id shown only under `--json` | **open** | no | **WS2** |
| R31-C1 | `--rank ppr` never surfaces thoughts | **open, cause unverified** | no | **WS5** |
| R31-C2 | every CLI search pays a ~30 s cold model load | open (by design per process) | **yes** — O4 `okf serve`, CLI auto-uses a running owner | **WS5** (ride 0.11.0) |
| R31-D | `--json` result keys differ between search modes (`concept_id` vs `id`) | **open** | no | **WS5** |
| R31-E | CLI + `okf-mcp` on one DB only observed idle | observation | O4 makes one owner the norm | none (note) |
| R31 G1–G6 | SQLite / libSQL findings in Macrame | not okfgraph's | — | out of scope |

**Net open work for okfgraph:** 12 items in 5 workstreams. Four of the
items (R29-A, R29-G, R29-G', R31-C2) overlap the 0.11.0 draft, which is
what sets the release split in §8.

---

## 2. WS1 — link extraction and resolution (R29-A, B, C, D)

**Impact:** 69 of the 71 broken links on a clean Macrame import. Doctor
score 25/100 on a corpus with almost no real link rot.

**Root cause in one sentence** (R29 §10): the code that *mints* a concept
id (`parse_source_file`: path relative to the bundle root, any
`SOURCE_EXTS` suffix stripped) and the code that *resolves* a link to it
(`normalize_path_link`: a string transform that knows only `.md`) have no
shared definition of what an id is. The fix is to give them one.

### 2.1 Changes

All in `okfgraph/components/links.py` plus the call site in
`okfgraph/components/import_.py` (`_resolve_body_links`, lines 447–461)
and the structural `diff`, which resolves links through the same helpers.

1. **One extension list.** Move `SOURCE_EXTS` (now duplicated in
   `import_.py:40`, `import_.py:287` and `delta.py:24`) to a single
   constant, e.g. `okfgraph/components/ids.py`, and import it everywhere.
2. **Anchored link pattern (R29-D).** Replace
   `MD_LINK_RE = r"\[.*?\]\((.*?\.md)\)"` with a pattern whose target
   cannot contain `)` or whitespace, so a match can never span two links:
   `\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+"[^"]*")?\s*\)`. Then keep only
   targets whose path (before `#`/`?`) ends in a `SOURCE_EXTS` member.
   Angle-bracket targets (`](<My File.md>)`) and link titles
   (`](a.md "t")`) are accepted, as CommonMark does.
3. **Code-aware extraction (R29-C).** Before running either pattern,
   blank out fenced code blocks (```` ``` ```` and `~~~`, any length ≥ 3)
   and inline code spans (backtick runs matched by length, per
   CommonMark). Blank with spaces of equal length rather than deleting,
   so offsets stay valid for any future diagnostics.
4. **Decode, then resolve against the source (R29-A, R29-B).** New
   `resolve_link_target(raw, source_id, known_ids)`:
   1. drop `#fragment` and `?query`;
   2. `urllib.parse.unquote` (fixes `%20`; same decoding `images.py`
      already applies, so documents and images finally agree);
   3. backslashes → `/`;
   4. a leading `/` means bundle-root-relative; otherwise join with
      `posixpath.dirname(source_id)`;
   5. `posixpath.normpath`; a result that starts with `..` has left the
      bundle → unresolved (recorded as broken, never guessed);
   6. strip one `SOURCE_EXTS` suffix (anchored; fixes `v1.2.mdx →
      v1.2x`);
   7. if the result is not a known id, **fall back to the pre-fix
      reading** (the old `normalize_path_link` output) and accept it only
      if *that* is a known id. Existing bundles that relied on the
      accidental root-relative reading keep resolving. This is the same
      fallback rule the 0.11.0 draft chose for O1, so the two agree.
   8. Return the resolved id or `None`, plus the target as written, for
      the broken-link row.
5. **Broken-link rows keep the raw link.** `okf broken-links` currently
   reports the already-mangled target (R29 §9 caveat). Record the link as
   written next to the normalized target. On `main` that is an extra
   field in the existing row (no schema change if it rides in an existing
   string/JSON column; otherwise defer to the 0.11.0 schema, which adds
   `BrokenLink.raw`).
6. `normalize_path_link` stays as a thin, deprecated wrapper (it is
   imported by tests and possibly by users) that calls the new resolver
   with `source_id=""`.

### 2.2 Tests (`tests/test_links_resolution.py`, new)

One case per measured line in R29, so the report is the regression suite:

- sibling link inside `architecture/` → `architecture/s13-decision-register`;
- self-link resolves to itself;
- `../Macrame%20Road%20to%201.0.md` from `architecture/x` → `Macrame Road to 1.0`;
- `../architecture/s4-schema.md` from a root file → unresolved (left the
  bundle), but from `releases/x` → `architecture/s4-schema`;
- `` `[[bench]]` `` and a fenced block containing `[[example]]` → no links;
- the `appendices.md:770` line → exactly two links, `public-api` and
  `s13-decision-register`, no garbage target;
- `[api](public-api.txt)` and `[r](README.markdown)` extracted and resolved;
- `v1.2.mdx` not mangled; `](<My File.md>)` and `](a.md "title")` parsed;
- fallback: a flat bundle written with root-relative links from a
  subfolder still resolves (back-compat).

Plus an end-to-end check: a fixture bundle that reproduces the Macrame
shape (root + two subfolders, spaces in names, a `.txt` concept, TOML in
backticks) must import with **0 broken links** and doctor 100.

### 2.3 Acceptance

Re-import `Macrame/docs` into a fresh database: broken links drop from 71
to the 2 that are Macrame's own (R29-E), and `architecture/public-api`
is no longer an orphan. Run R29's `okf_classify.py` against the raw
`](...)` targets as an independent cross-check.

### 2.4 Effect on existing graphs

Links are re-resolved on import, so the change applies to each concept the
next time it is imported. To fix a whole existing graph at once, run
`okf repair-links` (it re-resolves stored broken links) and, for full
coverage, `okf import --all --force`. Document this in the CHANGELOG entry.

---

## 3. WS2 — thoughts: unique titles and visible ids (R31-A, R31-B)

**Requirement (user, 2026-10-31):** *each ingested thought has a unique
title*, so `[[wikilinks]]` can resolve to thoughts.

### 3.1 Changes (`okfgraph/components/ingest.py`, `ops/ingest.py`, CLI, MCP)

1. **Accept `title` and `description` for `kind="thoughts"`.** Add both
   to the thoughts parameter set (`ingest.py:22`), CLI `--title` /
   `--description`, MCP `ingest` parameters. Remove the `BAD_VALUE`
   refusal for these two.
2. **Auto-title when none is given.** First sentence of the text,
   whitespace-collapsed, cut at a word boundary to ≤ 72 characters;
   if the text has no usable sentence, `<topic> — <YYYY-MM-DD HH:MM>`.
   Replaces the fixed `Thought: {topic}` at `ingest.py:238` and `:273`.
3. **Uniqueness, scoped to the topic.**
   - Auto-title collides with an existing thought title in the same
     topic → append ` (2)`, ` (3)`, … (never refuse an auto-title).
   - Explicit `--title` collides → refuse `BAD_VALUE` with the
     conflicting concept id in `fields` and a remedy ("choose another
     title, or pass --concept-id <existing> to update that thought").
     Not silently renamed: an explicit title is a name someone will link to.
   - Scope = topic, matching the report. Open question Q2 asks whether it
     should be graph-wide, because wikilink resolution is graph-wide and
     "ambiguous names never resolve".
4. **Show the minted id in the human output (R31-B).** The non-`--json`
   ingest render prints `[OK] ingested thought <concept_id> — "<title>"`
   as its last line, so `tail -1` keeps it.
5. **Skill/doc update.** Delete the line in the `okfgraph-ingest` skill
   that documents "Thoughts share one title per topic, so wikilinks to
   them resolve to nothing" (R31 remedy 3), and describe `--title`.

### 3.2 Existing thoughts

Existing graphs already hold many `Thought: <topic>` titles (six on the
Macrame topic alone). Offer a one-time maintenance op, e.g.
`okf retitle-thoughts [--topic T] [--dry-run]`, that applies rule 2/3 to
thoughts whose title is still the old `Thought: <topic>` form. It is a
write over many concepts, so it defaults to `--dry-run` output and is
CLI-only (not MCP). See open question Q3.

### 3.3 Tests

- thoughts ingest with `--title` stores it; duplicate explicit title in a
  topic → `BAD_VALUE` naming the existing id; same title in another topic
  is allowed (if Q2 = topic scope);
- auto-title from first sentence, truncation at a word boundary,
  collision suffixes `(2)`, `(3)`;
- a `[[<thought title>]]` wikilink from an md concept resolves to the
  thought;
- human ingest output ends with the concept id;
- MCP `ingest` accepts `title`;
- `retitle-thoughts --dry-run` changes nothing and lists the plan.

---

## 4. WS3 — incremental import cost and honesty (R29-G, R29-G')

**Impact:** editing one file re-embeds 22 files (best case, measured
249 s) to 38 of 49 (nested edit, projected 7–8 min on GPU).

### 4.1 Two-step plan

**Step 1, in the patch release (cheap, no behaviour risk):**

1. **Stop ancestor amplification.** `_compute_directory_hash` and
   `_compute_directory_hash_with_files` (`delta.py:56–90`) hash with
   `rglob`; the work set they trigger is direct children only. Hash
   **direct children only** (`iterdir`, files only, `SOURCE_EXTS`
   filter). A nested edit then invalidates only its own directory. This
   removes R29's amplification 2 (38 → 16 for an `architecture/` edit).
   The deletion scan already re-walks recursively (the comment at
   `delta.py:225`), so it is unaffected; verify with the existing nested
   tombstone regression test (0.2.19, 24 false tombstones).
   One-time effect: every stored directory hash changes meaning, so the
   first import after upgrading re-embeds once. Say so in the CHANGELOG.
2. **Honest log line (R29-G remedy 3).** Inside a changed directory,
   compare each file against its `FileHash` row (already written by
   `_store_file_hashes`, `delta.py:282`) only to *count*:
   `directory-delta: 1 changed dir(s) — 22 file(s) queued, 1 actually
   changed (directory-granular)`.
   The same numbers go into the `import` result so `--json` callers see them.

**Step 2, in 0.11.0 (the real fix):** per-file change detection, already
implemented in the draft as O2 (only new/changed files, or files whose
concept is missing, re-import). Take it from the draft rather than
re-implementing it on `main`. Note the draft's behaviour change:
`import_bundle` returns only the re-imported ids.

### 4.2 Renames (R29-G')

- **Patch release:** at the end of an import without `--prune-missing`,
  if files recorded in `FileHash` no longer exist, warn once:
  `N concept(s) have no source file anymore (renamed or deleted?) — run
  'okf import --all --prune-missing' to drop them` and list up to 5.
  Detection only; nothing is deleted.
- **0.11.0:** O3 frontmatter identity (opt-in) keeps a concept across a
  move. Default path-derived identity stays a delete + add.

### 4.3 Tests

- edit a nested file → only that directory's direct children queued
  (step 1), only that file (step 2 / 0.11.0);
- edit a root file → `architecture/` not queued (already true; keep it);
- log line and `--json` result report queued vs actually changed;
- rename without `--prune-missing` → warning naming the old file;
  with it → old concept purged (existing behaviour).

### 4.4 Acceptance

Repeat R29's three timed runs on `Macrame/docs`. Patch release: an
`architecture/` edit queues 16, not 38, and the log line says 1 changed.
0.11.0: queues 1.

---

## 5. WS4 — stale lock file (R07 §4, R29-F; reported twice)

**Cause.** `router.py:220–221` creates `fasteners.InterProcessLock` on
`<db>.lock`. fasteners releases the OS lock on exit but deliberately
leaves the file. Doctor's `stale_lock` correctly tells a leftover from a
live hold; nothing ever removes it.

### 5.1 Change

On `OKFRouter.close()` (and at CLI exit via `_close_routers`), after
releasing the write lock: try to take the lock **non-blocking** once more;
if that succeeds, no other process holds it, so unlink the file, then
release. Any error (another process grabbed it, file busy on Windows) →
leave the file and move on. Never unlink a file whose lock this process
could not acquire.

Keep doctor's `stale_lock` info as the safety net for crashed processes.

### 5.2 Tests

- open + close a router → no `<db>.lock` left;
- a second process holding the lock (subprocess) → first process's close
  leaves the file;
- CLI command run end to end → no lock file afterwards;
- doctor `stale_lock` still reports a planted leftover.

---

## 6. WS5 — search behaviour (R31-C1, R31-C2, R31-D)

### 6.1 `--rank ppr` and thoughts (R31-C1) — investigate first

**Unverified.** Likely mechanisms, to be checked in this order:

1. Thoughts carry no body links (R07 §4: "thought bodies carry no
   out-links"), so they are isolated nodes; PPR only spreads mass along
   `LINKS_TO`. An isolated thought can rank only if it is a seed.
2. Seeds come from lexical matches over the concept corpus
   (`search.py`, `_ppr_corpus` ~line 236 skips ids ending in `index` /
   `log`). Check whether thoughts are in that corpus and how they score
   against long md concepts (BM25 length normalization).
3. Score fusion may drop seeds with no graph neighbours.

**Deliverable:** a probe test on a small fixture (md concepts linked to
each other, one thought matching the query exactly) that shows where the
thought drops out. Then fix at that point: most likely "an isolated seed
keeps its seed mass" or "seed set includes thoughts". If the answer is
that PPR is the wrong mode for thoughts, document that instead and point
to `--target chunks`.

### 6.2 Cold model load per CLI call (R31-C2)

Not fixable inside one process lifetime. Already addressed by:

- now: `okf shell` (one warm process; documented);
- 0.11.0: `okf serve` (O4); the CLI uses a running owner automatically.

**Action:** none beyond shipping O4; mention `okf shell` in the search
section of the README and the skill.

### 6.3 Stable result keys (R31-D)

Every search result row, in every target and rank, carries the concept id
under **`id`**. Rows that are about a sub-part (a chunk, an image) keep
their own id under a specific key (`chunk_id`, `asset_id`) and **also**
carry `concept_id` for the owning concept. Audit all result builders in
`search.py` (lines 242, 307, 486, 659, 756–814) and `ranking.py`.

This is additive (no key removed), so it can go into the patch release.
Document the row shape per target in the README's search section and the
skill's data table. Test: one parametrized test over every
target × rank asserting `id` (and `concept_id` where applicable).

### 6.4 Filtered searches return fewer than `limit` (post-filter starvation)

Source: decision row "Post-filter starvation" in
`docs/chunking-and-graph-retrieval.md` §13. `search_hybrid` and
`search_chunks` over-fetch `limit * 3`, then apply
`concept_type` / `tags` / `parent_id`. A selective filter whose matches
rank outside that window returns fewer than `limit` results, or none,
although matching concepts exist.

The 0.11.0 draft already widens the window when a filter is set
(`fetch_k = max(limit * 10, 50)`, concepts path), because its new filters
(`where`, `since`, `until`, `exclude_tags`) make this worse. For 0.10.3:

1. Same widening on `main`, both concepts and chunks paths.
2. **One deepening retry:** if the filtered result is still short of
   `limit` and the fetch was full (`k` hits came back), refetch once at
   `4 × k`, capped at 1000. Every query keeps an explicit `top := k`
   (the ladybug segfault rule, `search.py` `_index_rows`).
3. Test: 200 concepts, the 3 with `type: decision` ranked lowest for the
   query; `search(limit=3, concept_type="decision")` returns all 3.

---

## 7. Out of scope (recorded so nothing is lost)

- **R29-E** — two `../docs/…` round-trip links in Macrame's own docs:
  fix in Macrame. After WS1 they are reported as "left the bundle", which
  is the honest answer.
- **R31 G1–G6** — SQLite/libSQL planner and connection findings in Macrame.
- **R31-E** — CLI + MCP coexistence only observed idle. The single-owner
  model of `okf serve` (0.11.0) is the structural answer.
- **R07 §4 doctor 10/100 on `kb.db`** — pre-existing link rot from docs
  that were never imported. Re-check after WS1: some of it may be WS1
  defects, the rest needs an editorial decision per link.

---

## 8. Releases and order of work

### 0.10.3 — "links that resolve" (patch, no schema change)

1. **WS1** link extraction/resolution — the largest win (69 of 71).
2. **WS4** stale lock cleanup — small, reported twice.
3. **WS2** thought titles + visible ids (additive parameters; the
   `retitle-thoughts` op if Q3 = yes).
4. **WS3 step 1** — direct-children directory hash + honest log line +
   rename warning.
5. **WS5.3** stable `id` keys (additive).
6. **WS5.4** filtered-search starvation (wider over-fetch + one retry).
7. Carry the already-committed embroider `>=0.12,<0.13` floor and the CI
   fix; add the missing **0.10.1** note to the CHANGELOG (0.10.2's entry
   says it supersedes 0.10.1, so a one-line pointer is enough).

Each workstream lands as its own commit with its tests; full suite and
CI green before the version bump; tag `v0.10.3` publishes.

### 0.11.0 — rebase the draft onto 0.10.3

The draft (`dev_0.11.0`) touches the same code as WS1/WS3
(`links.py`, `import_.py`, `delta.py`), so it must be **rebased onto
0.10.3**, not merged ahead of it:

- WS1's resolver replaces the draft's `relative_path_link` /
  `resolve_path_link` pair; keep the draft's O1 parts that 0.10.3 does
  not cover (incoming links kept on re-import, purge → `BrokenLink`,
  automatic repair).
- WS3 step 2 = the draft's O2; drop 0.10.3's direct-children hash if O2
  makes directory hashes advisory only.
- WS5.1 fix (if any) and WS5.2 (= O4) ride here.
- WS5.4: keep one over-fetch rule; the draft's `max(limit * 10, 50)`
  and 0.10.3's retry merge into one helper used by every filtered path.
- **The search track in §10** (cross-encoder rerank, near-duplicate
  collapse, `efs`, `find_path`, `SHOW_INDEXES`) lands after the rebase,
  on top of the draft's O7 filters, so every new stage sees the final
  filter set.
- Re-run the draft's suite on a machine with the models cached: its
  commit notes 42 failures "same as 0.10.2 in an offline sandbox", which
  needs confirming, not assuming.

### Release gate for both

- Full suite green locally (GPU venv) and the weekly full-suite workflow
  green on GitHub. It has been **cancelled ~3 minutes in every Sunday
  since at least 2026-09-27**, so today nothing on GitHub runs the full
  suite. Diagnose that before 0.10.3 ships.
- Acceptance re-runs on `Macrame/docs` (WS1 §2.3, WS3 §4.4).

---

## 9. Open questions (need a decision before implementation)

| # | Question | Default if no answer |
|---|---|---|
| Q1 | Ship WS1's source-relative resolution in a patch (0.10.3), or hold it for 0.11.0 with the draft's O1? It changes which concept some links point to (only where the old reading was wrong, thanks to the fallback). | patch — it is a correctness fix with a back-compat fallback |
| Q2 | Thought-title uniqueness: per topic (as the report asks) or graph-wide (as wikilink resolution needs)? | per topic, plus a lint warning when a thought title is ambiguous graph-wide |
| Q3 | Add `okf retitle-thoughts` to rename existing `Thought: <topic>` titles? | yes, CLI-only, `--dry-run` by default |
| Q4 | WS3 step 1 re-embeds every bundle once after upgrade (directory hashes change meaning). Acceptable in a patch, or skip step 1 and wait for O2? | acceptable, announced in the CHANGELOG |
| Q5 | Default rerank model: MiniLM-L-6 (English, 52 MB q8) or mMARCO (multilingual, 379 MB q8)? The Macrame/OKFgraph graphs are English, but notes carry German. | **Decided 2026-11-01: mMARCO** (Phase 3: equal on English, MiniLM lowered German chunk nDCG 0.766 → 0.593) |
| Q6 | Ladybug 0.22.1 is out (okfgraph pins 0.21.2). Re-probe the §10.3 knobs on 0.22.1 and move the pin within 0.11.0, or keep 0.21.2? | separate spike before 0.11.0 freezes; keep 0.21.2 unless the segfault and FTS probes pass on 0.22.1. **Spiked 2026-11-01** on branch `dev_ladybug-0.22`: unbounded FTS fixed (0.21.2 control crashes, 0.22.1 5/5 clean on a 31k-chunk copy); storage 47 → 49 is one-way; extensions must be installed before an existing graph opens (router now retries). Suite crash-rate comparison decides the move. |

---

## 10. 0.11.0 search track

Three documents feed this section:

- `docs/reranker-integration-plan.md`: neural rerank, a plan in its own
  right. Updated alongside this one; its dependency is already met.
- `docs/chunking-and-graph-retrieval.md` §13: four new decision rows
  (rerank, diversity + recency, post-filter starvation, graph-constrained
  search).
- `docs/search-options-analysis.md`: ladybug knobs that okfgraph doesn't
  expose yet, with a priority verdict (its §C).

### 10.0 Where each item lands

| Item | Source | Already covered? | Lands in |
|---|---|---|---|
| Cross-encoder rerank (`rerank=True` on chunks, `rank='cross'` on concepts) | reranker plan; §13 row 1 | no | **0.11.0**, §10.1 |
| Near-duplicate collapse | §13 row 2 ("near-dupe collapse first") | no | **0.11.0**, §10.2 |
| MMR diversity flag | §13 row 2 | no | later (0.12), after rerank is measured |
| Recency boost | §13 row 2 | no; draft O7 adds `since`/`until` *filters*, not a boost | later, once `timestamp` coverage is measured (§10.2) |
| Post-filter starvation | §13 row 3 | partly: draft widens over-fetch on the concepts path | **0.10.3** (WS5.4), unified in 0.11.0 |
| `exclude_*` filters | §13 row 4 | **yes**: draft O7 `exclude_tags` | 0.11.0 (draft) |
| `near_id` + hops proximity filter | §13 row 4 | no | deferred until asked (row's own decision) |
| `offset` pagination | §13 row 4 | no | deferred until asked |
| `efs` per-query recall knob (B3) | search options, priority 1 | no | **0.11.0**, §10.3 |
| `find_path` as one `MATCH path … ORDER BY length` (B8) | priority 2 | no | **0.11.0**, §10.3 |
| Brute-force `ARRAY_*` fallback without extensions (B6) | priority 3 | the draft takes the other route: offline extensions (O5) + doctor `search_unavailable` | **dropped**; revisit only if O5 still leaves installs without search |
| FTS `conjunctive` (B5) | priority 4 | no | **blocked**: ladybug's `k` vs `top` scoring split is unexplained, and `top` is the crash-safe form; ask upstream first |
| `SHOW_INDEXES` in doctor (B7) | priority 5 | no | **0.11.0**, §10.3 |
| Vector metric, HNSW build params, FTS stemmer/tokenizer (B1/B2/B4) | "skip until" | — | skipped, as the analysis recommends (`jieba` crashes the process: never offer it) |
| Wire `expand_with_graph_context` / `rerank_with_hub_score` into `QueryOps` (A15) | "only if asked" | — | not wired. `rerank_with_hub_score` stays as the result-contract template the cross-encoder copies (§10.1) |

**Answer to "is everything in `chunking-and-graph-retrieval.md` already
planned":** §1–12 and the older §13 rows describe chunking as it shipped
long ago. The doc's header says so, and the features exist (`search_chunks`,
`expand`, `hub_rerank`, `include_chunks`, `read(include=…)`). Only the four
§13 rows added on 2026-10-08 were open: the table above now places them.
Rerank, near-dupe collapse and starvation are planned; MMR, recency,
`near_id` and `offset` are explicitly deferred.

### 10.1 Cross-encoder rerank (from `reranker-integration-plan.md`)

The plan stands as written (decisions §1, engine method §2, text-supply
gotchas §3, surfaces §4, phases §5, test matrix §6). Changes from
rebasing it onto today's `main` and the 0.11.0 draft:

1. **Dependency already met.** The plan asked for `embroider>=0.4,<0.5`.
   `main` floors `embroider>=0.12,<0.13`, and 0.12.0 ships
   `embroider.Reranker` (`open`, `open_files`, `score`, `score_batch`,
   `score_long`, `count_tokens`, `used_cuda`, `precision`). Both
   cross-encoders are builtin registry entries, revision-pinned.
   **Phase 0 drops the pin bump.**
2. **A `q8` tier exists now** (embroider 0.11.0): MiniLM 52 MB "ranks
   kept", mMARCO 379 MB "nDCG within 0.016" on embroider's benches.
   `rerank.precision` accepts `fp32 | q8`; `auto` → fp32 (embroider never
   resolves `auto` to q8). Phase 3 measures q8 on okfgraph labels before
   it becomes a recommended setting.
3. **Result keys follow WS5.3:** `cross_score` is added next to `id` /
   `concept_id`; no key is renamed.
4. **Filters first, rerank last.** The draft's O7 filters and WS5.4's
   over-fetch shrink the candidate list *before* the cross-encoder runs,
   so `rerank.top_n` (default 30) is spent on rows that can be returned.
5. **Surfaces after the draft.** The draft grows MCP to 11 tools. `rerank`
   becomes a parameter of the existing `search` tool, and the `_HONOURS`
   matrix covers the draft's new paths.
6. **Warm process.** One cross-encoder load per CLI process adds to the
   cold-start cost (R31-C2). With `okf serve` (O4) the session stays warm;
   document `okf shell` / `okf serve` as the recommended way to use rerank.
7. **GPU venv:** `Reranker` reports `used_cuda`; log it next to the
   existing `text embeddings: … cuda=` line.

Effort as in the plan: about 3.5–5.5 days, with Phase 3 (eval harness on
okfgraph's own labels) gating any default or threshold.

### 10.2 Near-duplicate collapse (§13 row 2, first step)

Chunk results often contain the same passage twice: the same paragraph
in two documents, or a section copied into a plan and its successor.
`max_chunks_per_doc` caps repeats per document, not across documents.

*Done 2026-11-01:* `collapse_near_duplicates`, one pairwise
`array_cosine_similarity` query over the result pool (no numpy, ~40 ms
for 120 chunks); also on `hub_rerank` chunks; the pool doubles so
collapsed rows still fill the page. Recency shipped separately (concept
times, schema v10), so the coverage line below is moot.

- After fusion (and after rerank, when on), drop a chunk whose embedding
  has cosine ≥ 0.97 to a higher-ranked kept chunk. Vectors are already
  stored, so this costs no model call, at most `limit × limit` dot
  products. Report dropped rows as `collapsed: [ids]` on the kept row.
- Opt-in: `dedupe=True` (chunks path; `BAD_VALUE` elsewhere via
  `_HONOURS`). Default stays off until measured on the Macrame graph.
- **Recency:** before any recency boost, measure how many concepts carry
  a `timestamp` (doctor info line: `timestamp coverage: N%`). The boost
  is planned only if coverage is high.

### 10.3 Ladybug knobs (from `search-options-analysis.md`)

The analysis was probed on ladybug 0.21.2. Re-run its probes first if Q6
moves the pin.

1. **`efs` passthrough (B3).** *Done 2026-11-01 (re-probed on ladybug
   0.22.1: `efs := $efs` binds; also on `hub_rerank` chunks; refused with
   `near_id`, which uses no index).* `search(…, efs: int | None)`, passed to
   `QUERY_VECTOR_INDEX` as `efs := $efs` on the concepts and chunks
   paths; bounds 16–1000; `BAD_VALUE` on `rank='ppr'` (no vector query).
   Settings default `[search] efs` unset. Test: higher `efs` never lowers
   recall@10 against a brute-force oracle on a fixture.
2. **`find_path` in one query (B8).** *Done 2026-11-01 (`SHORTEST`, unknown ends raise).* Replace the length loop with
   `MATCH path = (a)-[:REL*1..N]-(b) … RETURN nodes(path) ORDER BY
   length(path) LIMIT 1`, keeping the 4-relationship whitelist and the
   `node_exists` check that tells "unknown id" from "no path". Delete
   the stale "not supported" comment in `search.py`. Existing traverse
   tests are the regression net; add one timing check on a 1000-node
   chain.
3. **`SHOW_INDEXES` in doctor (B7).** *Done 2026-11-01.* Info line listing each index (name,
   table, type, and for vector: metric and dimension), plus a
   `missing_index` warning when an expected index is absent. Read-only,
   no score impact. This fits the draft's index policy (O8), which makes
   `reindex` really drop and recreate.

### 10.4 Order inside 0.11.0

1. Rebase the draft onto 0.10.3 (§8); suite green with cached models.
2. §10.3 items 2 and 3 (small, no API change). *Done.*
3. §10.1 Phases 0–2 (rerank, opt-in), then §10.3 item 1 (`efs`, an API
   addition next to `rerank`). *Phases 0–2 done 2026-11-01:* `LazyReranker`,
   `[search] rerank_*` settings, `rerank_with_cross_encoder`, `rerank=True`
   (chunks plain / hub) and `rank='cross'` (concepts) on all surfaces;
   `rerank` and `recency` are refused together. Stub tests plus one live
   test gated on `OKFGRAPH_LIVE_RERANK=1`.
4. §10.2 near-dupe collapse (opt-in).
5. §10.1 Phase 3 eval harness: decides the default rerank model (Q5)
   and whether `dedupe` can default on. *Harness built 2026-11-01*
   (`scripts/eval_search.py`, starter labels for this repo); first numbers
   in `reranker-integration-plan.md` Phase 3. Q5 and the `dedupe`
   default stay open until reviewed labels and a Macrame run exist.
6. Release.
