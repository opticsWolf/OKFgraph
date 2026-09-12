# Plan: Model-free retrieval, structural diff, doctor, Obsidian compat

**Source:** coderadar review of `okf-ingest` (seeds + exact PPR, context blobs,
structural diff, doctor, wikilinks). **Goal:** adopt the model-free ideas that
upgrade okfgraph retrieval and maintenance without touching the embedding or
converter seams. Order: PPR → context budgets → diff → doctor → Obsidian.

**Invariants across all phases:**

- No new MCP tools (stay at 5). Retrieval upgrades are optional params on
  `search`/`read`; maintenance (`diff`, `doctor`) is CLI-only, like
  `broken-links` today.
- No new required dependencies. Everything here is stdlib + ladybug reads.
- Deterministic outputs: sorted traversal order, fixed constants, no wall-clock
  in represented data (timestamps only in generated IDs, as today).
- Each phase lands tested + documented (skills updated where agent-visible).

---

## Phase 1 — Lexical seeds + exact PPR (`search --rank ppr`)

**Problem.** Every `search` pays the ~30s ONNX cold boot, and `hub_rerank` scores
global hubness (query-independent). okf-ingest demonstrates query-seeded PPR
matching local embeddings on hub-heavy wikis with zero model infrastructure.

**Design.**

1. `seeds(query, k)` — pure function over concepts (no embeddings, no FTS):
   lowercase alphanumeric tokens, length ≥ 3, fixed stopword list (~37 words);
   score +3 per distinct token in title, +2 in description/tags, +1 in body;
   sort by score desc, path asc; truncate at `k` (default 20).
2. `ppr(starts, weights)` — exact power iteration over the *undirected*
   resolved-`LINKS_TO` graph: damping 0.85, tol 1e-12, max_iter 200, dangling
   mass returns to seed distribution, multi-seed weights from seed scores.
   Determinism rules: edges iterated in sorted `(src, dst)` id order, L1
   convergence accumulated in ascending node order, scores rounded half-even
   to 10 decimals (via fixed-precision formatting).
3. Adjacency source: read edges from Ladybug once per call
   (`MATCH (a)-[:LINKS_TO]->(b) RETURN ...`), map concept ids to indices in
   sorted order. Dangling concepts (degree 0) keep seed mass only.
4. Surface: `okf search --rank ppr|hub|none` (default `hub`, today's behavior);
   MCP `search(..., rank="ppr"|"hub")` — one optional param, descriptions carry
   the routing ("use ppr when cold, when deterministic, or when the query names
   topics rather than phrases").

**Files:** new `okfgraph/components/ranking.py` (`seeds`, `ppr`, stopwords);
`SearchEngine` gains `search_with_ppr()`; `cli.py` + `mcp_server.py` gain the
param. No schema change.

**Acceptance:**

- `search --rank ppr` returns sensible order on the existing test bundle
  without loading the embedder (assert model never touched — e.g. embedder
  stub that raises).
- Bit-stability: two consecutive runs byte-identical scores; unit test on a
  fixed 6-node graph with hand-computed/locked scores (first conformance-style
  golden fixture: `tests/fixtures/ppr_graph.{json,expected.json}`).
- `hub` default unchanged: existing search tests green unmodified.

**Risks.** PPR over a sparse/orphan-heavy graph degrades to seed order —
acceptable (equals lexical baseline). Large graphs: power iteration is
O(iter·edges); cap resolved-edge read at a sane limit and document it.

## Phase 2 — Token-budgeted `read` with PPR fill

**Problem.** `read(include=document|context)` fills breadth-first: on hub
neighbors the budget fills with whatever the hub links, not what matters.

**Design.**

1. `okf read ID --include document|context --max-tokens N` (default: current
   behavior, uncapped). MCP `read(..., max_tokens=N)`.
2. Assembly order: the concept itself, then neighbors ordered by single-seed
   PPR from the concept (reuse Phase 1 with `starts=[id]`), then depth-2 by
   the same ranking, truncating chunks at the token cap. Token counting via
   the existing tokenizer path (already a dependency), with a chars/4 fallback
   when the tokenizer is unavailable.
3. `include=context` additionally prepends the bundle index concept when one
   exists (index-first, mirroring `okf context`).

**Acceptance:** capped read never exceeds budget (±1 chunk granularity);
uncapped output byte-identical to today; skill `read` rows document the budget
param.

## Phase 3 — Structural `diff` (CLI)

**Problem.** The delta detector is internal plumbing: no user-facing answer to
"what will change if I import?" or "what drifted since last ingest?".

**Design.** `okf diff <a> <b>` where each side is a bundle dir or `--db` graph:

- Concepts added / removed / changed (by stored vs computed content hash),
  retitled / retyped (frontmatter `title`/`type` deltas).
- Edges added / removed, links newly broken / fixed.
- Pure hash/set comparison, output sorted by path; exit 0 identical, 1
  different (CI-gateable). `--json` for harness consumption.
- Drift mode: `okf diff --db kb.db bundle/` (graph side reads stored hashes +
  edges; dir side parses without importing — reuse `_parse_source_file`).
- Snapshot mode: `okf diff old/ new/` (both sides parsed, no db needed).

**Acceptance:** fixture pair (`tests/fixtures/diff_a/`, `diff_b/`) locks output
text; drift mode on a scratch graph detects exactly the edited file; exit codes
covered.

**Not in scope:** word-level body diffs (git already does text hunks).

## Phase 4 — `doctor`: scored health + safe `--fix`

**Problem.** `broken-links`/`repair-links` cover one finding class; orphans,
stale timestamps, duplicate titles, and hub concentration are invisible.

**Design.** `okf doctor [--strict] [--fix] [--stale-days N]` generalizing the
existing link commands (keep old names as aliases this branch, remove next):

- Findings: `broken_link`, `orphan` (no in/out edges, non-index), `stale`
  (timestamp older than N days), `duplicate_title`, `missing_description`;
  info-level `hub_concentration` (never affects score).
- Score 0–100, weighted deductions per severity; `--strict` exits 1 on any
  warning (pre-commit/CI gate); human-readable + `--json`.
- `--fix` applies only unambiguous repairs and reports each: parseable
  non-ISO timestamp normalization, broken-link re-point on exactly one
  basename match (existing `repair_links` logic, moved under doctor).
- `reviewed: true` frontmatter exempts a concept from `--fix` (human-validated
  stability) while still reporting findings.

**Acceptance:** score locked on a fixture bundle with known findings; `--fix`
never touches `reviewed: true` (test); strict exit codes tested.

## Phase 5 — Obsidian-vault compatibility (import + export)

**Problem.** Obsidian/Logseq/Foam vaults use `[[wikilinks]]` resolved by *name*,
not path. Importing a vault today either misses those edges or misresolves
them; exporting produces path-links that work but feel foreign.

**Design — import:**

1. Extract `[[target]]` / `[[target|display]]` alongside `](path)` links.
2. Resolve by name with precedence `id → aliases[] → title → filename-stem`
   (case-insensitive); build the name index per import; names claimed by >1
   concept are ambiguous → link recorded unresolved (never guessed).
3. Support `aliases: [..]` and `id:` frontmatter (additive; existing bundles
   byte-identical in behavior).
4. Unresolved wikilinks join the broken-link set → surfaced by `doctor`,
   repairable by `repair-links` once the target appears.

**Design — export:**

1. `okf export --flavor obsidian` (default flavor stays `okf`): emit
   `[[Title]]` links for edges whose target title is unique in the export set,
   `[[path|Title]]`-style disambiguation otherwise; carry `aliases`/`id`
   frontmatter through so re-import is lossless.
2. Round-trip test: export(obsidian) → import → identical edge set.

**Acceptance:** fixture vault with `[[Name]]`, `[[name|display]]`, duplicate
titles (ambiguity → unresolved), and renames (path changes, links survive);
round-trip edge equality.

**Risks.** Name resolution adds an index build per import — O(concepts),
negligible vs embedding cost. Alias/title collisions across a huge vault could
flood findings; cap reported ambiguous names per run and summarize the rest.

---

## Cross-phase work

- **Skills:** Phase 1–2 update the `read`/`search` rows in both skills
  (rank + budget params, "ppr when cold" routing); Phase 3–4 add `diff`/`doctor`
  to CLI conventions; Phase 5 documents wikilink + alias behavior in the
  ingest skill.
- **Fixtures:** `tests/fixtures/` becomes the conformance-corpus habit
  (ppr_graph, diff_a/b, doctor_bundle, obsidian_vault) — golden files, not
  throwaway tmp dirs.
- **Docs:** `docs/harness-integration.md` gains the "cold search" note;
  `architecture.md` refresh (still pending) absorbs ranking/diff/doctor as
  first-class components.
