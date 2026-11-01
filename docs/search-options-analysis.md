# OKFgraph Search Options — Implemented vs. Ladybug-Native (Gap Analysis)

> Date: 2026-10-08 · ladybug `0.21.2` (probed live on `:memory:`) · OKFgraph `okfgraph/components/search.py`, `ops/query.py`, `components/ranking.py`, `components/image_assets.py`, `components/schema.py`
> Convention per entry: **What** → **Good for** → **Gain** → **Relevant when** → **Skip when**.

---

## A. Implemented in OKFgraph

### A1. Unified entry point — `QueryOps.search(query, target, limit, …)`

Single op behind CLI (`okf search`), MCP (`search`), Python (`router.search`). Validates once, refuses ignored params with `BAD_VALUE` instead of silently dropping them.
**Good for:** one mental model for agents. **Gain:** no per-surface drift. **Relevant:** always — it is the contract. **Skip:** never; extend it rather than bypassing it.

### A2. `target='concepts'` + `rank='none'` → `SearchEngine.search_hybrid` (default path)

RRF fusion (`1/(60+rank)`) of `QUERY_VECTOR_INDEX(Concept, k=limit*3)` + `QUERY_FTS_INDEX(Concept, top:=limit*3)`, over-fetch then filter (`concept_type/tags/parent_id`, `exclude_reserved`), truncate descriptions to 200ch.
**Good for:** open-ended questions ("what do we know about X?"). **Gain:** semantic recall + keyword precision in one ranked list. **Relevant:** default starting point. **Skip:** when you need passages (use `chunks`), images, or the embedder is cold (use `ppr`).

### A3. `target='concepts'` + `rank='hub'` (`hub_weight=0.3`)

Same as A2, then blends `(1-w)*rrf + w*norm_hub`, where hub = incoming `LINKS_TO` count.
**Good for:** "most authoritative / most referenced" queries. **Gain:** surfaces canonical docs over orphans with similar text. **Relevant:** dense, well-linked graphs; index/overview docs. **Skip:** sparse graphs (all hubs ≈ 0, only adds noise) or when textual fit matters more than authority.

### A4. `target='concepts'` + `rank='ppr'` → `search_with_ppr`

Lexical seeds (`ranking.seeds`: +3 title / +2 description·tags / +1 body, tokens ≥3ch minus ~37 stopwords) weighted into exact multi-seed PPR over the undirected `LINKS_TO` graph. No ONNX, no FTS/vector extensions, deterministic.
**Good for:** cold router, offline/air-gapped, "query names topics" rather than phrases. **Gain:** answers from link structure alone; zero model load. **Relevant:** first query in a session, embedder unavailable. **Skip:** unlinked graphs (degrades to lexical seed order — just use FTS then).

### A5. `target='chunks'` plain → `search_chunks(limit, max_chunks_per_doc=3, +filters)`

Chunk-level RRF (same formula, `Chunk` indexes), bulk parent-metadata fetch, `500ch` text cut, per-document cap.
**Good for:** exact passages, quotes, code snippets. **Gain:** pinpoints *where* in a doc the answer lives. **Relevant:** long docs, RAG citation. **Skip:** when a doc-level yes/no ("do we have X?") suffices — cheaper.

### A6. `target='chunks'` + `expand=true` → `search_with_context(limit≤20, context_hops=1..3)`

Each chunk enriched with `incoming_links[:5] + outgoing_links[:5] + ancestry + siblings[:5] + full document`.
**Good for:** "why does this passage matter?" — relational reading. **Gain:** one call replaces search+traverse+read round-trips. **Relevant:** exploration, impact analysis. **Skip:** bulk RAG (payload is 5–10× larger); use plain chunks + selective `read`.

### A7. `target='chunks'` + `hub_rerank=true` → `search_chunks_with_hub_score(hub_weight)`

Re-ranks `limit*2` raw chunks by blended hub score. Wins over `expand` (precedence documented).
**Good for:** "best passage from the most important doc". **Gain:** combines textual fit with authority. **Relevant:** same as A3 but at passage granularity. **Skip:** same caveat — meaningless on sparse graphs; also ignores `concept_type/tags/parent_id` by design (refused, not dropped).

### A8. `include_chunks=true` (concepts `none|hub` only)

Attaches `matched_chunks` (top-3 chunks of that doc) to each concept hit.
**Good for:** doc list with evidence snippets. **Gain:** avoids a second chunk query per hit. **Relevant:** result lists shown to users. **Skip:** with `rank=ppr` (refused — PPR path never loads the embedder) or when you will `read` the winners anyway.

### A9. Filters — `concept_type / tags (ALL match) / parent_id (CONTAINS*1..3)`

Post-fusion SQL `WHERE` (over-fetched candidates fill `limit` after filtering).
**Good for:** scoping ("only decisions under `dir:Y`"). **Gain:** precision without a second query. **Relevant:** typed/tagged corpora, multi-project DBs. **Skip:** small graphs — extra clauses cost more than they save.

### A10. `QueryOps.read(include=body|chunks|document|context, max_tokens?)`

* `body`: full concept dict + markdown. Default deep read.
* `chunks`: stored chunks ordered by index. Passage inventory.
* `document`: markdown rebuilt from chunks (delimiter-aware). Round-trip/export source.
* `context`: incoming/outgoing links + ancestry + siblings (cap 10). One-hop neighbourhood.
* `max_tokens`: `read_with_budget` — self first, then bundle-index + single-seed-PPR neighbours, last section truncated to fit (`{sections, used, budget, truncated}`).
**Good for:** each answers a different question (full text vs. structure vs. neighbourhood vs. token-capped prompt fill). **Gain:** no over-fetch into the context window. **Relevant:** `max_tokens` for LLM prompt assembly; `context` for navigation. **Skip:** `max_tokens` when you need byte-exact fidelity (budgeted reads truncate).

### A11. `QueryOps.traverse` — three modes in one op

* root listing (`start_id=""`), *walk* (`relationship, direction, depth 1..5, node_type`), *path* (`target, max_path_length 1..10` → `find_path` iterative `*1..N` loop + `UNWIND nodes(p)`).
**Good for:** directory browsing, reference following (`LINKS_TO`), chunk/asset fan-out (`PART_OF/INCLUDES_ASSET`), "how are A and B connected?". **Gain:** graph semantics FTS/vector can't express. **Relevant:** navigation, dependency/impact walks. **Skip:** depth >2 on dense graphs (100-row cap, fan-out explodes); path mode on disconnected pairs (returns `[]` — check `node_exists` first to distinguish "unknown id" from "no path").

### A12. Helpers — `get_context / get_chunks / get_by_id / list_directory / node_exists / _compute_hub_scores / _links_adjacency / _seed_corpus`

Plumbing behind A6/A10/A11. `_compute_hub_scores` (incoming-link count) is the single authority signal reused by A3/A7. `_links_adjacency/_seed_corpus` feed the model-free path.
**Good for:** code reuse, testability. **Gain:** one definition of "hub", "neighbourhood", "seed set". **Skip:** not public API — call `search/read/traverse` instead.

### A13. `ranking.seeds / ppr / seed_ranked_ppr` (pure functions, no DB/model)

Deterministic (id-asc ties, sorted edges, ascending L1 sum, round-half-even 10dp).
**Good for:** golden-fixture testing, offline ranking, embedding-free fallback. **Gain:** reproducible order across runs/machines. **Relevant:** cold start, CI without models. **Skip:** when embeddings are loaded — hybrid outranks it on paraphrase queries.

### A14. `search_images_with_text` (`target='images'`, caption-vector only)

Encodes the text query, `QUERY_VECTOR_INDEX(ImageAsset, k=limit)`.
**Good for:** "find the diagram of X". **Gain:** images become searchable without vision inference at query time. **Relevant:** caption-rich bundles. **Skip:** vision-content queries ("red curve going up") need `mode=optional/omni` at ingest; text search can't see pixels.

### A15. Dead-but-tested — `expand_with_graph_context(chunk_ids)` / `rerank_with_hub_score(chunk_results)`

Neighbour discovery from explicit chunk ids; multiplicative re-rank `rrf*(1+0.1*hub)`. Reachable only via `router.search_engine.*`, **not** via `QueryOps.search` (no test or adapter calls them except unit tests).
**Good for:** post-hoc re-rank of an already-fetched list; id-pinned expansion. **Gain:** composability for scripts. **Relevant:** custom pipelines. **Skip:** for normal use — A6/A7 cover it through the validated op; wiring these into `QueryOps` would be the (small) completion work.

### A16. Index plumbing — `_index_specs / _build_search_indexes / reindex / _index_rows`

Fixed specs (cosine HNSW `mu=30,ml=60,efc=200`; FTS defaults), change-driven rebuilds (`write_epoch>indexed_epoch`), short-lived connections + mandatory `top:=k` as segfault workarounds (ladybug `0.20.3/0.21.2`).
**Good for:** correctness on large graphs; cheap no-op imports. **Gain:** `reindex` recovers stale/crashed indexes. **Relevant:** always (invisible). **Skip:** never touch directly — the workarounds encode real crash knowledge.

---

## B. Ladybug-native but not implemented/exposed

### B1. Vector metric — `L2 / L2SQ / DOTPRODUCT / IP` (`CREATE_VECTOR_INDEX metric :=`)

Only `cosine` is created. Probes confirm all four alternatives are accepted.
**Good for:** matching the embedding model's native geometry (dot-product for un-normalised heads, L2 for Euclidean-trained models). **Gain:** small recall win when the model isn't cosine-normalised (Jina-v5 is — so ~zero today). **Relevant:** if the embedder ever changes. **Skip now:** adds an index-per-metric dimension with no measurable gain on current vectors.

### B2. HNSW build tuning — `mu / ml / efc` + `alpha : DOUBLE`

Hardcoded `30/60/200`, `alpha` never set (default). `alpha` is recognised (wrong-type error proves it), `ef/M/threads` are rejected.
**Good for:** build-time recall/latency/memory trade-off on huge graphs. **Gain:** faster indexing (smaller) or higher recall (larger) at scale. **Relevant:** 100k+ vectors or slow-import complaints. **Skip now:** premature optimisation — defaults are sane for the current 28k-chunk scale; expose only as advanced settings, never per-query.

### B3. Search-time recall knob — `efs := INT` (`QUERY_VECTOR_INDEX`)

Valid, never passed (OKF passes only positional `k`).
**Good for:** per-query speed/recall slider (large `efs` = slower but more exact). **Gain:** cheap quality lever for hard queries without rebuilding. **Relevant:** the single most worthwhile vector gap — a `search(..., efs?)` passthrough is ~5 lines. **Skip:** default (`efs` unset) is fine for easy queries; only add when recall complaints arrive.

### B4. FTS index linguistics — `stemmer / tokenizer / stopwords` (`CREATE_FTS_INDEX`)

`stemmer := 'porter'|'none'` valid; `tokenizer := 'simple'` default, `'jieba'` crashes the process (`DictTrie FATAL`, exit 127 — do not offer); `stopwords` wants a table name or file path, not a list. OKF uses all defaults.
**Good for:** stemming lifts recall ("index" ≈ "indexing"); custom stopwords clean noisy corpora; jieba for Chinese. **Gain:** language-dependent recall. **Relevant:** non-English corpora or stemming-sensitive evaluations. **Skip now:** English default is correct; jieba is a crash vector; per-language indexes multiply maintenance — gate behind opt-in.

### B5. FTS query semantics — `conjunctive := BOOL` / `k := INT` (`QUERY_FTS_INDEX`)

Both valid, neither exposed (OKF always `top:=k`). `conjunctive` showed no score delta on the probe corpus (likely AND-vs-OR over terms — corpus-dependent); `k` vs `top` take **different scoring paths** (`0.1088` vs `0.1451` on the same doc — needs upstream clarification).
**Good for:** `conjunctive=true` = strict AND (precision), `false` = OR (recall). **Gain:** a meaningful precision/recall switch for keyword-heavy users. **Relevant:** exact-term searches ("must contain both words"). **Skip now:** semantics are under-documented and the `k/top` divergence is unexplained — confirm with ladybug upstream before surfacing; `top` is the safe, crash-avoiding choice today.

### B6. Brute-force vector SQL — `ARRAY_COSINE_SIMILARITY / ARRAY_DISTANCE / ARRAY_INNER_PRODUCT / ARRAY_SQUARED_DISTANCE`

Scalar functions over stored `FLOAT[N]`; verified working. Never used (search raises `SEARCH_UNAVAILABLE` without extensions).
**Good for:** degraded-but-correct search when the vector extension can't load (offline, no repo); exact ground truth for index-parity tests. **Gain:** availability + test oracle. **Relevant:** air-gapped installs; correctness harness. **Skip:** O(N) scan — never the default path; add as fallback (`try index, else scan + warn`), not as primary.

### B7. Index lifecycle — `SHOW_INDEXES / DROP_VECTOR_INDEX / DROP_FTS_INDEX`

OKF only `CREATE`s idempotently; never inspects or drops.
**Good for:** metric/tokenizer migration (drop + recreate), stale-index diagnosis, admin visibility. **Gain:** recoverability without DB surgery. **Relevant:** the day a metric or stemmer change ships. **Skip now:** read-only `SHOW_INDEXES` in `doctor` output would be harmless, but `DROP` needs no UI until a migration requires it.

### B8. Cypher path expressiveness — `MATCH path = … / nodes(p) / length(p)`

`MATCH path = (a)-[:E*1..2]-(b)` **works** on `0.21.2` despite the `search.py` "not supported" comment (comment predates the upgrade); `find_path` already relies on it second-pass. Genuinely absent: `SHORTEST_PATH/shortestPath` (parser errors), k-shortest, weighted, or label-free `[*..]` (OKF whitelists 4 rels).
**Good for:** single-query shortest path (replacing today's length-loop), path-length ordering, `UNWIND nodes(p)` projection. **Gain:** fewer round-trips, cleaner `find_path`. **Relevant:** next `find_path` cleanup — collapse the loop into one `MATCH path … ORDER BY length(path) LIMIT 1`. **Skip:** weighted/k-shortest/pure-variable patterns — no engine support, and unbounded traversals are a footgun on dense graphs anyway.

---

## C. Verdict — what to do, what to deliberately skip

| Priority | Item | Effort | Payoff |
|---|---|---|---|
| 1 | Expose `efs` as optional `search()` param (B3) | ~5 lines + docs | per-query recall knob, zero migration |
| 2 | Collapse `find_path` to single `MATCH path … ORDER BY length` (B8) | small, test-covered | fewer queries, fixes stale comment |
| 3 | Index-scan fallback via `ARRAY_*` when extensions missing (B6) | medium | search works offline/degraded |
| 4 | `conjunctive` passthrough for FTS (B5) | small | keyword precision switch — **after** upstream clarifies `k` vs `top` |
| 5 | `SHOW_INDEXES` in `doctor` (B7) | tiny | observability, no risk |
| — | Metric switch, `mu/ml/efc/alpha`, stemmer/tokenizer/stopwords (B1/B2/B4) | medium + migration cost | **skip until** a non-cosine model, non-English corpus, or 100k-scale pain arrives |
| — | Wire `expand_with_graph_context/rerank_with_hub_score` into `QueryOps` (A15) | small | only if script users ask — A6/A7 already cover the behaviour |

*Rule of thumb:* OKFgraph's strength is the **composition** (RRF + filters + hub/PPR + budget + traverse) over fixed, crash-safe index settings. Ladybug's remaining knobs are real but mostly trade one scarce resource (build time, memory, precision, maintenance) for another — expose the cheap per-query ones (`efs`, `conjunctive`), keep the build-time ones pinned until a corpus demands otherwise.
